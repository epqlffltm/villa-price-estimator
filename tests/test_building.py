# tests/test_building.py
"""건축물대장 응답에서 전용면적·준공년도를 고르는 규칙과, 여러 쪽에 걸친 응답을 모으는 부분을 확인한다. (실제 API는 호출하지 않는다)"""
import pytest

from model.building import (BuildingRegisterError, built_year, fetch_items, find_unit_area, house_type_of,
                            normalize_ho)


def unit(ho, floor, area, kind="1", basement=False):
    """전유공용면적 한 줄. kind "1" 전유, "2" 공용."""
    return {"hoNm": ho, "flrNo": floor, "flrGbCd": "10" if basement else "20", "exposPubuseGbCd": kind, "area": area}


ITEMS = [
    unit("301호", 3, 29.09), unit("301호", 3, 6.5, kind="2"),
    unit("302호", 3, 31.20), unit("302호", 3, 6.9, kind="2"),
    unit("303호", 3, 45.00),
    unit("401호", 4, 29.31),
    unit("B01호", 1, 38.00, basement=True),
    unit("101호", 1, 52.00),
]


@pytest.mark.parametrize("text, expected", [("301호", "301"), ("제301호", "301"), (" 301 호", "301"), ("b01호", "B01"), ("", ""), (None, "")])
def test_normalize_ho(text, expected):
    assert normalize_ho(text) == expected


def test_finds_owned_area_of_the_unit_and_ignores_shared_area():
    assert find_unit_area(ITEMS, floor=3, ho="301") == (29.09, "건축물대장 301호")


def test_ho_written_with_leading_zero_matches_on_the_same_floor():
    assert find_unit_area([unit("0301호", 3, 29.09), unit("301호", 13, 99.0)], floor=3, ho="301")[0] == 29.09


def test_same_ho_name_in_two_buildings_on_one_lot_is_not_added_together():
    items = [{**unit("301호", 3, 30.0), "dongNm": "A동"}, {**unit("301호", 3, 50.0), "dongNm": "B동"}]
    area, how = find_unit_area(items, floor=3, ho="301")
    assert area == 40.0
    assert "중앙값" in how


def test_ho_on_another_floor_is_used_when_the_floor_has_no_match():
    assert find_unit_area([unit("301호", 4, 33.0)], floor=3, ho="301") == (33.0, "건축물대장 301호")


def test_without_ho_uses_median_of_units_on_the_floor():
    area, how = find_unit_area(ITEMS, floor=3, ho="")
    assert area == 31.20
    assert how == "건축물대장 3층 3개 호 중앙값"


def test_unknown_ho_falls_back_to_the_floor():
    assert find_unit_area(ITEMS, floor=4, ho="999")[0] == 29.31


def test_basement_and_ground_floor_are_not_mixed():
    assert find_unit_area(ITEMS, floor=-1, ho="")[0] == 38.00
    assert find_unit_area(ITEMS, floor=1, ho="")[0] == 52.00


def test_owned_parts_of_one_unit_are_added():
    assert find_unit_area([unit("201호", 2, 30.0), unit("201호", 2, 5.5)], floor=2, ho="201")[0] == 35.5


def test_returns_none_when_nothing_matches():
    assert find_unit_area(ITEMS, floor=7, ho="") is None
    assert find_unit_area([], floor=3, ho="301") is None


def test_built_year_takes_main_building_approval_year():
    items = [{"useAprDay": "20230716", "mainAtchGbCd": "0"}, {"useAprDay": "19990101", "mainAtchGbCd": "1"}]
    assert built_year(items) == 2023
    assert built_year([{"useAprDay": " "}]) is None
    assert built_year([]) is None


def test_house_type_from_purpose():
    assert house_type_of([{"mainPurpsCdNm": "공동주택", "etcPurps": "연립주택"}]) == "연립"
    assert house_type_of([{"mainPurpsCdNm": "공동주택", "etcPurps": "다세대주택"}]) == "다세대"
    assert house_type_of([{"mainPurpsCdNm": "제2종근린생활시설"}]) is None


class FakeResponse:
    def __init__(self, body=None, status_code=200, code="00"):
        self.status_code, self._body, self._code = status_code, body, code

    def json(self):
        return {"response": {"header": {"resultCode": self._code, "resultMsg": "msg"}, "body": self._body}}


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params, timeout):
        self.calls.append((url, params))
        return self.responses.pop(0)


def test_fetch_items_reads_all_pages_and_builds_the_lot_code():
    first = FakeResponse({"items": {"item": [unit(f"{n}호", 1, 10.0) for n in range(100)]}, "totalCount": 101})
    second = FakeResponse({"items": {"item": unit("999호", 9, 10.0)}, "totalCount": 101})  # 한 건이면 dict로 온다
    session = FakeSession(first, second)
    items = fetch_items("getBrExposPubuseAreaInfo", "KEY%2F", "1162010200", 598, 178, session=session)
    assert len(items) == 101
    url, params = session.calls[0]
    assert url.endswith("/getBrExposPubuseAreaInfo")
    assert (params["sigunguCd"], params["bjdongCd"], params["bun"], params["ji"]) == ("11620", "10200", "0598", "0178")
    assert params["serviceKey"] == "KEY/"  # Encoding 키는 풀어서 보낸다
    assert session.calls[1][1]["pageNo"] == 2


def test_fetch_items_returns_empty_list_when_there_is_no_building():
    assert fetch_items("getBrTitleInfo", "KEY", "1162010200", 1, 0, session=FakeSession(FakeResponse({"items": "", "totalCount": 0}))) == []


def test_fetch_items_raises_on_api_error():
    with pytest.raises(BuildingRegisterError):
        fetch_items("getBrTitleInfo", "KEY", "1162010200", 1, 0, session=FakeSession(FakeResponse({}, code="30")))
