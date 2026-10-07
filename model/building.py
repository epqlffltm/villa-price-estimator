# model/building.py
"""건축물대장 조회: 입력에 없는 전용면적과 준공년도를 지번·호로 찾는다.

국토교통부 건축HUB 건축물대장정보 서비스의 두 기능을 쓴다.
  getBrTitleInfo            표제부. 건물 한 동의 사용승인일(useAprDay), 주용도
  getBrExposPubuseAreaInfo  전유공용면적. 호(hoNm)·층(flrNo)별 전유/공용 면적

API를 부르는 부분과, 받아 온 목록에서 값을 고르는 부분(find_unit_area, built_year)을 나눠
고르는 규칙은 네트워크 없이 테스트한다.

한 지번에 동이 여럿(A동·B동, 101동·102동)이면 같은 호 이름이 동마다 있다. 실제 조회에서 50곳 중 6곳이 그랬다.
입력의 호에 동이 함께 적혀 있으면("B동 201", "102-402") 그 동에서 찾고, 없으면 그 층에서 실제로 거래된 면적과
같은 쪽을 고른다. 그래도 가릴 수 없으면 동별 면적의 중앙값을 쓰고 어림값으로 표시한다.
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


def normalize_dong(text) -> str:
    """"A동", "제101동", " b 동" -> "A", "101", "B"."""
    return re.sub(r"[\s제동]", "", str(text or "")).upper()


def split_dong_ho(text) -> tuple[str, str]:
    """호 입력에 동이 함께 적혀 있으면 (동, 호)로 나눈다. 아니면 ("", 원래 글자).

    "B동 201호" -> ("B", "201호"),  "102동402" -> ("102", "402"),  "B-201" -> ("B", "201")
    "201-1"처럼 동이 아닌 경우도 나눠지므로, 나온 동 이름이 대장에 실제로 있는지는 부르는 쪽에서 확인한다.
    """
    raw = str(text or "").strip()
    for pattern in (r"^(.+?)동\s*-?\s*(\S+)$", r"^([^\s-]+)\s*-\s*(\S+)$", r"^(\S+)\s+(\S+)$"):
        match = re.match(pattern, raw)
        if match:
            return normalize_dong(match.group(1)), match.group(2)
    return "", raw


def is_guess(method: str) -> bool:
    """find_unit_area가 돌려준 '찾은 방법'이 호를 하나로 특정하지 못한 경우인지."""
    return "중앙값" in method or "추정" in method


def find_unit_area(items: list[dict], floor: int, ho: str = "", hint_areas=()) -> tuple[float, str] | None:
    """전유공용면적 목록에서 대상 호의 전용면적을 찾아 (면적, 찾은 방법)을 돌려준다.

    호를 찾는 순서
      1. 같은 층에서 호 이름이 같은 것 ("301" = "301호" = "제301호")
      2. 같은 층에서 호의 숫자가 같은 것 ("301" = "0301호")
      3. 층은 다르지만 호 이름이 같은 것 (입력의 층이 대장과 다르게 적힌 경우)
      4. 호가 없거나 못 찾으면 같은 층 호들의 전유 면적 중앙값

    입력에 동이 함께 적힌 경우 ("B동 201", "102-402")
      - 그 동이 대장에 있으면 그 동 안에서만 찾는다.
      - 대장에 동 이름이 적혀 있는데 그 동이 없으면, 호 이름만으로 찾되 "추정"으로 표시한다.
        (A동만 있는 대장에서 "B동 201"로 A동 201호를 찾은 것은 확정이 아니다.)
      - 대장에 동 이름이 아예 없으면(건물이 하나) 동은 확인할 수 없으므로 호 이름만으로 찾는다.
      "201-2"처럼 호 이름 자체에 줄표가 있는 경우는, 그 이름 그대로인 호가 있으면 그것을 먼저 고른다.

    같은 호 이름이 여러 동에 있으면
      - hint_areas(이 지번·이 층에서 실제로 거래된 면적들)와 면적이 같은 동이 하나뿐이면 그 동의 호 ("추정")
      - 가릴 수 없으면 동별 면적의 중앙값

    is_guess(찾은 방법)이 True면 호를 하나로 특정하지 못했다는 뜻이다.
    """
    owned = [item for item in items if str(item.get("exposPubuseGbCd")) == "1" and _to_float(item.get("area"))]
    wanted_dong, rest = split_dong_ho(ho)
    in_dong = [item for item in owned if wanted_dong and normalize_dong(item.get("dongNm")) == wanted_dong]
    has_dong_names = any(normalize_dong(item.get("dongNm")) for item in owned)

    # (찾을 범위, 호 이름, 이름이 정확히 같은 것만 볼지, 동을 확인하지 못했는지)
    if in_dong:
        attempts = [(in_dong, rest, False, False)]
        owned = in_dong
    elif wanted_dong:
        attempts = [(owned, ho, True, False),                    # "201-2"가 호 이름 그대로인 경우
                    (owned, rest, False, has_dong_names)]        # 동을 뗀 호 이름으로. 대장에 동 이름이 있었다면 동 불일치
    else:
        attempts = [(owned, ho, False, False)]

    for pool, name, exact_only, dong_missing in attempts:
        matched = _match_ho(pool, floor, name, exact_only)
        if matched:
            area, method = _area_of(matched, hint_areas)
            if dong_missing and not is_guess(method):
                method += f", 입력한 동({wanted_dong})이 대장에 없어 호 이름만으로 추정"
            return area, method

    on_floor = [item for item in owned if _same_floor(item, floor)]
    per_unit = _sum_by(on_floor, lambda item: (str(item.get("dongNm") or "").strip(), normalize_ho(item.get("hoNm"))))
    if per_unit:
        return round(statistics.median(per_unit.values()), 2), f"건축물대장 {floor}층 {len(per_unit)}개 호 중앙값"
    return None


def _match_ho(owned: list[dict], floor: int, ho: str, exact_only: bool = False) -> list[dict]:
    """호 이름이 맞는 전유 항목들. exact_only면 같은 층에서 이름이 정확히 같은 것만 본다."""
    wanted = normalize_ho(ho)
    if not wanted:
        return []
    on_floor = [item for item in owned if _same_floor(item, floor)]
    same_name = [item for item in on_floor if normalize_ho(item.get("hoNm")) == wanted]
    if same_name or exact_only:
        return same_name
    same_digits = [item for item in on_floor if _digits(wanted) and _digits(normalize_ho(item.get("hoNm"))) == _digits(wanted)]
    return same_digits or [item for item in owned if normalize_ho(item.get("hoNm")) == wanted]


def _area_of(matched: list[dict], hint_areas=()) -> tuple[float, str]:
    """이름이 맞는 전유 항목들에서 (면적, 찾은 방법)을 정한다. 같은 호 이름이 여러 동에 있으면 가려 본다."""
    # 한 호에 전유 항목이 여럿이면 합한다. 같은 호 이름이 여러 동에 있으면 동별로 따로 본다.
    per_building = _sum_by(matched, lambda item: str(item.get("dongNm") or "").strip())
    name = str(matched[0].get("hoNm")).strip()
    areas = sorted(per_building.values())
    if len(per_building) == 1 or areas[-1] - areas[0] < 0.01:
        dong = next(iter(per_building)) if len(per_building) == 1 else ""
        return round(areas[0], 2), f"건축물대장 {dong + ' ' if dong else ''}{name}"
    traded = {round(area, 2) for area in areas if any(abs(area - hint) < 0.5 for hint in hint_areas)}
    if len(traded) == 1:
        return traded.pop(), f"건축물대장 {name}, 동 {len(per_building)}개 중 이 층에서 거래된 면적으로 추정"
    return round(statistics.median(areas), 2), f"건축물대장 {name}, 동 {len(per_building)}개 중앙값"


def floor_units(items: list[dict], floor: int) -> dict[tuple[str, str], float]:
    """한 층의 호별 전유면적. {(동 이름, 호 이름): 면적}. 한 호에 전유 항목이 여럿이면 합한다."""
    owned = [item for item in items if str(item.get("exposPubuseGbCd")) == "1" and _to_float(item.get("area"))]
    on_floor = [item for item in owned if _same_floor(item, floor) and normalize_ho(item.get("hoNm"))]
    return _sum_by(on_floor, lambda item: (str(item.get("dongNm") or "").strip(), str(item.get("hoNm")).strip()))


def find_unit_by_area(items: list[dict], floor: int, area_m2: float, tolerance: float = 0.5) -> tuple[str, str] | None:
    """같은 층에서 전유면적이 area_m2와 같은 호의 (동 이름, 호 이름)을 돌려준다. 없으면 None.

    검증용이다. 실거래가에는 호가 없어서, 면적이 같은 호를 대장에서 찾아 "이 거래의 호"로 삼는다.
    면적이 가장 가까운 호를 고르고, 같은 면적의 호가 여럿이면 이름순으로 첫 번째를 고른다.
    """
    close = sorted((abs(area - area_m2), ho, dong) for (dong, ho), area in floor_units(items, floor).items()
                   if abs(area - area_m2) < tolerance)
    return (close[0][2], close[0][1]) if close else None


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
