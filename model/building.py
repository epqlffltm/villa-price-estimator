# model/building.py
"""건축물대장 조회: 입력에 없는 전용면적과 준공년도를 지번·호로 찾는다.

국토교통부 건축HUB 건축물대장정보 서비스의 두 기능을 쓴다.
  getBrTitleInfo            표제부. 건물 한 동의 사용승인일(useAprDay), 주용도
  getBrExposPubuseAreaInfo  전유공용면적. 호(hoNm)·층(flrNo)별 전유/공용 면적

API를 부르는 부분과, 받아 온 목록에서 값을 고르는 부분(find_unit_area, built_year)을 나눠
고르는 규칙은 네트워크 없이 테스트한다.
"""
import re
import statistics
import time
from urllib.parse import unquote

import requests

BASE_URL = "https://apis.data.go.kr/1613000/BldRgstHubService"
PAGE_SIZE = 100
MAX_PAGES = 30


class BuildingRegisterError(RuntimeError):
    """건축물대장을 조회하지 못했다. 호출한 쪽에서 다른 방법으로 넘어간다."""


def fetch_items(operation: str, key: str, b_code: str, bonbun: int, bubun: int, is_san: bool = False,
                session=None) -> list[dict]:
    """한 지번의 건축물대장 항목을 모든 쪽에 걸쳐 받아 온다. b_code는 법정동코드 10자리."""
    http = session or requests
    params = {
        "serviceKey": unquote(key.strip()),
        "sigunguCd": b_code[:5],
        "bjdongCd": b_code[5:10],
        "platGbCd": "1" if is_san else "0",
        "bun": f"{int(bonbun):04d}",
        "ji": f"{int(bubun):04d}",
        "numOfRows": PAGE_SIZE,
        "_type": "json",
    }
    items: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        body = _get_body(http, f"{BASE_URL}/{operation}", {**params, "pageNo": page})
        page_items = _as_list(body.get("items"))
        items.extend(page_items)
        if not page_items or len(items) >= int(body.get("totalCount") or 0):
            break
    return items


def _get_body(http, url: str, params: dict) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            res = http.get(url, params=params, timeout=20)
            if res.status_code != 200:
                raise BuildingRegisterError(f"HTTP {res.status_code}")
            response = res.json()["response"]
            code = str(response["header"]["resultCode"])
            if code not in ("00", "000"):
                raise BuildingRegisterError(f"resultCode {code}: {response['header'].get('resultMsg')}")
            return response["body"]
        except BuildingRegisterError:
            raise
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            last_error = e  # 연결 실패나 깨진 응답은 잠시 뒤 다시 시도한다
            time.sleep(1 + attempt)
    raise BuildingRegisterError(f"응답을 읽지 못함: {last_error}")


def _as_list(items) -> list[dict]:
    """결과가 없으면 빈 문자열, 한 건이면 dict, 여러 건이면 list로 오는 것을 list로 맞춘다."""
    if not items:
        return []
    item = items.get("item", []) if isinstance(items, dict) else items
    if isinstance(item, dict):
        return [item]
    return list(item)


def normalize_ho(text) -> str:
    """"제301호", "301 호", "301" -> "301". "B01호" -> "B01"."""
    return re.sub(r"[\s제호]", "", str(text or "")).upper()


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text).lstrip("0")


def find_unit_area(items: list[dict], floor: int, ho: str = "") -> tuple[float, str] | None:
    """전유공용면적 목록에서 대상 호의 전용면적을 찾아 (면적, 찾은 방법)을 돌려준다.

    1. 같은 층에서 호 이름이 같은 것 ("301" = "301호" = "제301호")
    2. 같은 층에서 호의 숫자가 같은 것 ("301" = "0301호")
    3. 층은 다르지만 호 이름이 같은 것 (입력의 층이 대장과 다르게 적힌 경우)
    4. 호가 없거나 못 찾으면 같은 층 호들의 전유 면적 중앙값
    찾은 방법에 "중앙값"이 들어 있으면 호를 하나로 특정하지 못했다는 뜻이다.
    """
    owned = [item for item in items if str(item.get("exposPubuseGbCd")) == "1" and _to_float(item.get("area"))]
    on_floor = [item for item in owned if _same_floor(item, floor)]
    wanted = normalize_ho(ho)
    if wanted:
        candidates = [
            [item for item in on_floor if normalize_ho(item.get("hoNm")) == wanted],
            [item for item in on_floor if _digits(wanted) and _digits(normalize_ho(item.get("hoNm"))) == _digits(wanted)],
            [item for item in owned if normalize_ho(item.get("hoNm")) == wanted],
        ]
        for matched in candidates:
            if matched:
                # 한 호에 전유 항목이 여럿이면 합한다. 한 지번에 동이 여럿이라 같은 호 이름이 겹치면 동별로 따로 본다.
                per_building = _sum_by(matched, lambda item: str(item.get("dongNm") or "").strip())
                name = matched[0].get("hoNm")
                if len(per_building) == 1:
                    return round(next(iter(per_building.values())), 2), f"건축물대장 {name}"
                return (round(statistics.median(per_building.values()), 2),
                        f"건축물대장 {name}, 동 {len(per_building)}개 중앙값")

    per_unit = _sum_by(on_floor, lambda item: (str(item.get("dongNm") or "").strip(), normalize_ho(item.get("hoNm"))))
    if per_unit:
        return round(statistics.median(per_unit.values()), 2), f"건축물대장 {floor}층 {len(per_unit)}개 호 중앙값"
    return None


def _sum_by(items: list[dict], key) -> dict:
    totals: dict = {}
    for item in items:
        totals[key(item)] = totals.get(key(item), 0.0) + _to_float(item["area"])
    return totals


def _same_floor(item: dict, floor: int) -> bool:
    number = _to_float(item.get("flrNo"))
    if number is None:
        return False
    is_basement = str(item.get("flrGbCd")) == "10"  # 10 지하, 20 지상
    return int(number) == abs(floor) and is_basement == (floor < 0)


def _to_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def built_year(title_items: list[dict]) -> int | None:
    """표제부의 사용승인일(YYYYMMDD)에서 준공년도를 꺼낸다. 여러 동이면 주건축물 중 가장 이른 해."""
    years = []
    for item in title_items:
        text = str(item.get("useAprDay") or "").strip()
        if len(text) >= 4 and text[:4].isdigit() and 1900 <= int(text[:4]) <= 2100:
            if str(item.get("mainAtchGbCd", "0")) == "0":
                years.append(int(text[:4]))
    return min(years) if years else None


def house_type_of(title_items: list[dict]) -> str | None:
    """표제부의 용도에서 연립/다세대를 가린다."""
    text = " ".join(f"{item.get('mainPurpsCdNm', '')} {item.get('etcPurps', '')}" for item in title_items)
    if "연립" in text:
        return "연립"
    if "다세대" in text:
        return "다세대"
    return None
