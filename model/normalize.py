# model/normalize.py
"""실거래가 원본 값(문자열)을 계산에 쓸 수 있는 값으로 바꾸는 함수 모음.

모든 함수는 바꿀 수 없는 값을 만나면 예외를 던지지 않고 None을 돌려준다.
None이 나온 행을 지울지 말지는 clean.py가 결정한다.
"""
import re
from datetime import date

# "602-251", "1740", "산 12-3" 형태만 지번으로 인정한다.
_JIBUN = re.compile(r"^(산)?\s*(\d+)(?:-(\d+))?$")


def parse_price(text: str) -> int | None:
    """거래금액 "44,000"(만원 단위) -> 440000000(원)."""
    digits = text.replace(",", "").strip()
    if not digits.isdigit():
        return None
    return int(digits) * 10_000


def parse_float(text: str) -> float | None:
    """면적 "52.25" -> 52.25. 빈 값이나 숫자가 아니면 None."""
    try:
        return float(text.strip())
    except ValueError:
        return None


def parse_int(text: str) -> int | None:
    """층 "-1" -> -1, 건축년도 "2004" -> 2004. 빈 값이나 숫자가 아니면 None."""
    try:
        return int(text.strip())
    except ValueError:
        return None


def split_jibun(text: str) -> tuple[bool, int, int] | None:
    """지번을 (산 여부, 본번, 부번)으로 나눈다. 부번이 없으면 0.

    "602-251" -> (False, 602, 251)
    "1740"    -> (False, 1740, 0)
    "산 12-3" -> (True, 12, 3)
    """
    match = _JIBUN.match(text.strip())
    if match is None:
        return None
    san, bonbun, bubun = match.groups()
    return (san is not None, int(bonbun), int(bubun or 0))


def make_date(year: str, month: str, day: str) -> str | None:
    """("2026", "9", "12") -> "2026-09-12". 없는 날짜면 None."""
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None
