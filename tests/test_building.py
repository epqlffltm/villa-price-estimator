# tests/test_building.py
"""건축물대장 응답에서 전용면적·준공년도를 고르는 규칙과, 여러 쪽에 걸친 응답을 모으는 부분을 확인한다. (실제 API는 호출하지 않는다)"""
import pytest

from model.building import (BuildingRegisterError, built_year, fetch_items, find_unit_area, find_unit_by_area, house_type_of,
                            is_guess, normalize_ho, split_dong_ho)


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


def test_find_unit_by_area_picks_the_unit_with_that_owned_area_on_that_floor():
    assert find_unit_by_area(ITEMS, floor=3, area_m2=31.2) == ("", "302호")
    assert find_unit_by_area(ITEMS, floor=4, area_m2=45.0) is None           # 그 면적은 3층에만 있다
    assert find_unit_by_area(ITEMS, floor=3, area_m2=31.0) == ("", "302호")  # 허용 오차 안에서 가장 가까운 호
    assert find_unit_by_area(ITEMS, floor=-1, area_m2=38.0) == ("", "B01호")
    assert find_unit_by_area(ITEMS, floor=3, area_m2=6.5) is None            # 공용면적은 보지 않는다
    assert find_unit_by_area(TWO_DONGS, floor=2, area_m2=43.19) == ("B동", "201")


def in_dong(dong, ho, floor, area):
    return {**unit(ho, floor, area), "dongNm": dong}


# 실제 조회에서 나온 구성: 한 지번에 A동·B동이 있고 같은 호 이름의 면적이 서로 다르다.
TWO_DONGS = [
    in_dong("A동", "201", 2, 45.58), in_dong("A동", "202", 2, 29.99), in_dong("A동", "203", 2, 45.05),
    in_dong("B동", "201", 2, 43.19), in_dong("B동", "202", 2, 29.97), in_dong("B동", "203", 2, 43.19),
]
THREE_DONGS = [
    in_dong("101동", "401", 4, 46.30), in_dong("101동", "402", 4, 45.83),
    in_dong("102동", "401", 4, 49.36), in_dong("102동", "402", 4, 43.28),
    in_dong("103동", "401", 4, 42.52), in_dong("103동", "402", 4, 48.46),
]


@pytest.mark.parametrize("text, expected", [
    ("B동 201호", ("B", "201호")), ("102동402", ("102", "402")), ("B-201", ("B", "201")), ("102 - 402호", ("102", "402호")),
    ("가동 301", ("가", "301")), ("B 201", ("B", "201")), ("201", ("", "201")), ("201호", ("", "201호")), ("", ("", "")),
])
def test_split_dong_ho(text, expected):
    assert split_dong_ho(text) == expected


def test_ho_alone_in_a_lot_with_two_buildings_is_a_guess():
    area, how = find_unit_area(TWO_DONGS, floor=2, ho="201")
    assert area == 44.38 and is_guess(how)  # 45.58과 43.19의 중앙값


@pytest.mark.parametrize("ho", ["B동 201", "B동201호", "B-201", "b 201"])
def test_dong_written_with_the_ho_picks_that_building(ho):
    area, how = find_unit_area(TWO_DONGS, floor=2, ho=ho)
    assert area == 43.19 and not is_guess(how)
    assert "B동" in how


@pytest.mark.parametrize("ho, expected", [("102동 402호", 43.28), ("102-402", 43.28), ("103동 401", 42.52)])
def test_numbered_buildings(ho, expected):
    area, how = find_unit_area(THREE_DONGS, floor=4, ho=ho)
    assert area == expected and not is_guess(how)


def test_traded_area_on_the_floor_tells_the_buildings_apart():
    area, how = find_unit_area(TWO_DONGS, floor=2, ho="201", hint_areas=[43.19])
    assert area == 43.19 and is_guess(how)  # 맞을 가능성이 높지만 확정은 아니므로 어림으로 남긴다
    # 두 동의 면적이 모두 거래된 적이 있으면 가릴 수 없다.
    assert find_unit_area(TWO_DONGS, floor=2, ho="201", hint_areas=[43.19, 45.58])[0] == 44.38


def test_ho_with_a_dash_that_is_not_a_building_still_matches_by_name():
    items = [unit("201-1호", 2, 33.0), unit("201-2호", 2, 35.0)]
    assert find_unit_area(items, floor=2, ho="201-2") == (35.0, "건축물대장 201-2호")


def test_same_area_in_every_building_is_not_a_guess():
    items = [in_dong("A동", "301", 3, 30.0), in_dong("B동", "301", 3, 30.0)]
    area, how = find_unit_area(items, floor=3, ho="301")
    assert area == 30.0 and not is_guess(how)


ONLY_A = [in_dong("A동", "201", 2, 45.58), in_dong("A동", "202", 2, 29.99)]


@pytest.mark.parametrize("ho", ["B동 201", "B동201호", "B-201", "B 201"])
def test_dong_that_is_not_in_the_register_is_not_an_exact_match(ho):
    """대장에 A동만 있는데 B동 201을 넣으면, A동 201호의 면적을 쓰더라도 확정으로 치지 않는다."""
    area, how = find_unit_area(ONLY_A, floor=2, ho=ho)
    assert area == 45.58 and is_guess(how)
    assert "입력한 동(B)이 대장에 없어" in how


def test_ho_without_a_dong_in_a_one_building_register_is_exact():
    assert find_unit_area(ONLY_A, floor=2, ho="201") == (45.58, "건축물대장 A동 201")
    assert find_unit_area(ONLY_A, floor=2, ho="A동 201") == (45.58, "건축물대장 A동 201")


def test_dong_cannot_be_checked_when_the_register_has_no_dong_names():
    """건물이 하나라 대장에 동 이름이 없으면, 입력에 적힌 동은 확인할 길이 없다. 호 이름으로 찾은 것을 그대로 쓴다."""
    area, how = find_unit_area(ITEMS, floor=3, ho="가동 301")
    assert area == 29.09 and not is_guess(how)
    assert find_unit_area(ITEMS, floor=3, ho="102-302")[0] == 31.20


def test_ho_name_with_a_dash_wins_over_reading_it_as_a_dong():
    """동 이름이 있는 대장에서도, "201-2"라는 호가 그대로 있으면 201동 2호로 읽지 않는다."""
    items = [in_dong("A동", "201-1호", 2, 33.0), in_dong("A동", "201-2호", 2, 35.0), in_dong("A동", "2호", 2, 99.0)]
    area, how = find_unit_area(items, floor=2, ho="201-2")
    assert area == 35.0 and not is_guess(how)
