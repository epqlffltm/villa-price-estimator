# tests/test_normalize.py
"""원본 문자열을 숫자·날짜로 바꾸는 함수가 정상 값과 깨진 값을 각각 어떻게 처리하는지 확인한다."""
import pytest

from model.normalize import make_date, parse_float, parse_int, parse_price, split_jibun


@pytest.mark.parametrize("text, expected", [
    ("44,000", 440_000_000),
    ("7,350", 73_500_000),
    (" 90,000 ", 900_000_000),
    ("120,000", 1_200_000_000),
    ("", None),
    ("-", None),
    ("4.4억", None),
])
def test_parse_price(text, expected):
    assert parse_price(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("52.25", 52.25),
    ("23", 23.0),
    ("", None),
    ("없음", None),
])
def test_parse_float(text, expected):
    assert parse_float(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("4", 4),
    ("-1", -1),
    ("2004", 2004),
    ("", None),
    ("3.5", None),
])
def test_parse_int(text, expected):
    assert parse_int(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("602-251", (False, 602, 251)),
    ("1740", (False, 1740, 0)),
    (" 24-121 ", (False, 24, 121)),
    ("산 12-3", (True, 12, 3)),
    ("산12", (True, 12, 0)),
    ("", None),
    ("12-", None),
    ("1**", None),
])
def test_split_jibun(text, expected):
    assert split_jibun(text) == expected


@pytest.mark.parametrize("year, month, day, expected", [
    ("2026", "9", "12", "2026-09-12"),
    ("2022", "04", "01", "2022-04-01"),
    ("2023", "2", "30", None),
    ("2023", "", "1", None),
])
def test_make_date(year, month, day, expected):
    assert make_date(year, month, day) == expected
