# model/geocode.py
"""지번 주소 한 건을 Kakao 주소 검색 API로 좌표(위도·경도)로 바꾼다.

수집 단계(collect/geocode_trades.py)와 predict.py가 같은 함수를 쓴다.
검색 결과를 그대로 믿지 않고, 돌아온 주소의 동·본번·부번이 요청한 것과 같은지 확인해 상태를 남긴다.
  ok        요청한 지번과 일치
  mismatch  결과는 있지만 다른 지번 (좌표는 저장하되 구분해 둔다)
  not_found 결과 없음
"""
import time

import requests

URL = "https://dapi.kakao.com/v2/local/search/address.json"


class GeocodeError(RuntimeError):
    """키 오류처럼 다시 시도해도 해결되지 않는 문제."""


def pick_match(documents: list[dict], dong: str, bonbun: int, bubun: int) -> dict:
    """검색 결과 목록에서 요청한 지번과 일치하는 주소를 골라 저장할 값으로 만든다."""
    candidates = [doc["address"] for doc in documents if doc.get("address")]
    for address in candidates:
        same_lot = (
            address.get("region_3depth_name") == dong
            and address.get("main_address_no") == str(bonbun)
            and (address.get("sub_address_no") or "0") == str(bubun)
        )
        if same_lot:
            return _to_row("ok", address)
    if candidates:
        return _to_row("mismatch", candidates[0])
    return {"status": "not_found", "lat": None, "lon": None, "b_code": "", "matched": ""}


def _to_row(status: str, address: dict) -> dict:
    try:
        lat, lon = float(address["y"]), float(address["x"])  # Kakao는 x가 경도, y가 위도다.
    except (KeyError, ValueError):
        return {"status": "not_found", "lat": None, "lon": None, "b_code": "", "matched": ""}
    return {
        "status": status,
        "lat": lat,
        "lon": lon,
        "b_code": address.get("b_code", ""),
        "matched": address.get("address_name", ""),
    }


def geocode_lot(key: str, sigungu: str, dong: str, jibun: str, bonbun: int, bubun: int, session=None) -> dict:
    """"서울특별시 관악구 신림동 598-178" 형태로 검색해 pick_match 결과를 돌려준다."""
    http = session or requests
    params = {"query": f"{sigungu} {dong} {jibun}", "analyze_type": "exact"}
    headers = {"Authorization": f"KakaoAK {key}"}
    last_error = None
    for attempt in range(3):
        try:
            res = http.get(URL, params=params, headers=headers, timeout=10)
        except requests.RequestException as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
            continue
        if res.status_code in (401, 403):
            raise GeocodeError(f"Kakao 인증 실패({res.status_code}): {res.text[:300]}")
        if res.status_code == 429 or res.status_code >= 500:  # 호출 한도·서버 오류는 잠시 뒤 다시 시도
            last_error = f"HTTP {res.status_code}"
            time.sleep(2 * (attempt + 1))
            continue
        if res.status_code != 200:
            raise GeocodeError(f"Kakao 응답 오류({res.status_code}): {res.text[:300]}")
        return pick_match(res.json().get("documents", []), dong, bonbun, bubun)
    raise GeocodeError(f"Kakao 호출 실패: {last_error}")
