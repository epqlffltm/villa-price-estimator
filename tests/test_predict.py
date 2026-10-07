# tests/test_predict.py
"""입력 한 줄이 출력 한 줄이 되기까지를 가상의 동네로 확인한다: 값 채우기, 실패 사유, 어림값의 신뢰도 감점."""
import pandas as pd
import pytest

from model.confidence import ConfidenceModel
from model.resolve import InputError, Resolver, canonical_jibun, parse_floor
from predict import OUTPUT_COLUMNS, predict
from test_estimate import BASE_PER_M2, LAT, LON, make_neighborhood

CONFIDENCE = ConfidenceModel(
    intercept=1.6,
    coefficients={"근거량": 0.2, "흩어짐": -5.0, "지하층": -1.2, "1층": -0.2, "신축": -0.7, "노후": -0.4},
    half_widths=[0.40, 0.30, 0.22, 0.17],
)
B_CODE = "1162010200"


def make_geocodes(trades: pd.DataFrame) -> pd.DataFrame:
    lots = trades.drop_duplicates("jibun")
    return pd.DataFrame({"sigungu": lots["sigungu"], "dong": lots["dong"], "jibun": lots["jibun"], "status": "ok",
                         "lat": lots["lat"], "lon": lots["lon"], "b_code": B_CODE})


class FakeRegister:
    """건축물대장과 Kakao 응답을 흉내 낸다. 어떤 주소를 호출했는지 기록한다."""

    def __init__(self, area_items=None, title_items=None, kakao_documents=None):
        self.area_items, self.title_items = area_items or [], title_items or []
        self.kakao_documents, self.urls = kakao_documents or [], []

    def get(self, url, params=None, headers=None, timeout=None):
        self.urls.append(url)
        session = self

        class Response:
            status_code = 200

            def json(self):
                if "kakao" in url:
                    return {"documents": session.kakao_documents}
                items = session.area_items if url.endswith("getBrExposPubuseAreaInfo") else session.title_items
                return {"response": {"header": {"resultCode": "00"}, "body": {"items": {"item": items}, "totalCount": len(items)}}}

        return Response()


def make_resolver(session=None, data_key="", kakao_key="") -> Resolver:
    trades = make_neighborhood()
    return Resolver(trades, make_geocodes(trades), data_key, kakao_key, session=session)


def row(**changes) -> dict:
    return {"id": "1", "sigungu": "서울특별시 관악구", "dong": "신림동", "jibun": "100-1", "floor": "2",
            "ho": "", "area_m2": "40", **changes}


@pytest.mark.parametrize("text, expected", [("3", 3), ("3층", 3), ("-1", -1), ("B1", -1), ("지하1층", -1), (2, 2)])
def test_parse_floor(text, expected):
    assert parse_floor(text) == expected


@pytest.mark.parametrize("text", ["", "옥탑", "0"])
def test_parse_floor_rejects_unreadable_values(text):
    with pytest.raises(InputError):
        parse_floor(text)


def test_canonical_jibun():
    assert canonical_jibun("598-178") == ("598-178", False, 598, 178)
    assert canonical_jibun("1740-0")[0] == "1740"
    with pytest.raises(InputError):
        canonical_jibun("abc")


def test_known_property_is_estimated_from_its_building():
    output = predict([row()], make_resolver(), CONFIDENCE)[0]
    assert list(output) == OUTPUT_COLUMNS
    assert output["status"] == "ok"
    assert output["price_est"] == pytest.approx(BASE_PER_M2 * 40 * 1.3, rel=0.05)  # 100-1은 30% 비싼 건물
    assert output["price_low"] < output["price_est"] < output["price_high"]
    assert output["price_est"] % 100_000 == 0
    assert "같은 건물 3건" in output["basis"]  # 4건 중 같은 호(2층)는 빠진다


def test_outputs_keep_input_order_and_bad_rows_fail_with_a_reason():
    outputs = predict([
        row(id="a"),
        row(id="b", sigungu="부산광역시 해운대구"),
        row(id="c", jibun="abc"),
        row(id="d", floor=""),
        row(id="e", sigungu="서울특별시 강서구", dong="등촌동"),
        row(id="f", jibun="9999-9"),
    ], make_resolver(), CONFIDENCE)
    assert [output["id"] for output in outputs] == ["a", "b", "c", "d", "e", "f"]
    assert outputs[0]["status"] == "ok"
    for output, reason in zip(outputs[1:], ["권역 밖", "지번", "층", "화곡동만", "지번을 확인할 수 없음"]):
        assert output["status"].startswith("fail: ")
        assert reason in output["status"]
        assert output["price_est"] == ""


def test_missing_area_is_read_from_the_building_register_by_ho():
    session = FakeRegister(area_items=[
        {"hoNm": "201호", "flrNo": 2, "flrGbCd": "20", "exposPubuseGbCd": "1", "area": 40.0},
        {"hoNm": "202호", "flrNo": 2, "flrGbCd": "20", "exposPubuseGbCd": "1", "area": 80.0},
    ])
    output = predict([row(area_m2="", ho="202")], make_resolver(session, data_key="KEY"), CONFIDENCE)[0]
    assert output["status"] == "ok"
    assert "면적 80.0㎡(건축물대장 202호)" in output["basis"]
    assert any(url.endswith("getBrExposPubuseAreaInfo") for url in session.urls)


def test_dong_missing_from_the_register_lowers_confidence_and_says_so():
    """대장에 A동만 있는데 B동 202를 넣으면, 결과는 내되 근거에 동 불일치를 남기고 신뢰도를 낮춘다."""
    items = [{"hoNm": "201호", "dongNm": "A동", "flrNo": 2, "flrGbCd": "20", "exposPubuseGbCd": "1", "area": 40.0},
             {"hoNm": "202호", "dongNm": "A동", "flrNo": 2, "flrGbCd": "20", "exposPubuseGbCd": "1", "area": 80.0}]
    right = predict([row(area_m2="", ho="A동 202")], make_resolver(FakeRegister(area_items=items), data_key="KEY"), CONFIDENCE)[0]
    wrong = predict([row(area_m2="", ho="B동 202")], make_resolver(FakeRegister(area_items=items), data_key="KEY"), CONFIDENCE)[0]
    assert right["status"] == wrong["status"] == "ok"
    assert "입력한 동(B)이 대장에 없어" in wrong["basis"]
    assert "대장에 없어" not in right["basis"]
    assert wrong["confidence"] < right["confidence"]


def test_missing_area_without_register_uses_same_floor_trades():
    with_area = predict([row()], make_resolver(), CONFIDENCE)[0]
    without_area = predict([row(area_m2="")], make_resolver(), CONFIDENCE)[0]
    assert without_area["status"] == "ok"
    assert "같은 건물 2층 실거래 기준" in without_area["basis"]
    assert without_area["price_est"] == with_area["price_est"]  # 이 건물 2층은 40㎡ 한 종류뿐이다


def test_new_lot_is_located_and_dated_by_lookup():
    kakao = [{"address": {"address_name": "서울 관악구 신림동 777-7", "region_3depth_name": "신림동", "main_address_no": "777",
                          "sub_address_no": "7", "x": str(LON), "y": str(LAT), "b_code": B_CODE}}]
    session = FakeRegister(title_items=[{"useAprDay": "20140301", "mainAtchGbCd": "0", "etcPurps": "다세대주택"}],
                           kakao_documents=kakao)
    output = predict([row(jibun="777-7")], make_resolver(session, data_key="KEY", kakao_key="KEY"), CONFIDENCE)[0]
    assert output["status"] == "ok"
    assert "준공 2014년(건축물대장)" in output["basis"]
    assert "같은 건물 0건" in output["basis"]


def test_guessing_lowers_confidence():
    # 좌표는 찾았지만 건축물대장이 비어 있어 준공년도를 주변 중앙값으로 어림하는 경우
    kakao = [{"address": {"address_name": "서울 관악구 신림동 777-7", "region_3depth_name": "신림동", "main_address_no": "777",
                          "sub_address_no": "7", "x": str(LON), "y": str(LAT), "b_code": B_CODE}}]
    known = FakeRegister(title_items=[{"useAprDay": "20140301", "mainAtchGbCd": "0"}], kakao_documents=kakao)
    unknown = FakeRegister(kakao_documents=kakao)
    sure = predict([row(jibun="777-7")], make_resolver(known, data_key="KEY", kakao_key="KEY"), CONFIDENCE)[0]
    guessed = predict([row(jibun="777-7")], make_resolver(unknown, data_key="KEY", kakao_key="KEY"), CONFIDENCE)[0]
    assert "준공년도를 찾지 못해" in guessed["basis"]
    assert guessed["confidence"] < sure["confidence"]


def test_one_broken_row_does_not_stop_the_rest(monkeypatch):
    resolver = make_resolver()
    original = resolver.resolve

    def flaky(input_row):
        if input_row["id"] == "x":
            raise RuntimeError("boom")
        return original(input_row)

    monkeypatch.setattr(resolver, "resolve", flaky)
    outputs = predict([row(id="x"), row(id="y")], resolver, CONFIDENCE)
    assert outputs[0]["status"].startswith("fail: 처리 중 오류")
    assert outputs[1]["status"] == "ok"


def test_estimate_does_not_depend_on_other_rows():
    """같은 물건은 입력 CSV에 다른 줄이 몇 개 있든 같은 값이 나와야 한다."""
    rows = [row(id="1"), row(id="2", jibun="101-1"), row(id="3", jibun="108-1", floor="3")]
    together = predict(rows, make_resolver(), CONFIDENCE)
    alone = [predict([one], make_resolver(), CONFIDENCE)[0] for one in rows]
    assert together == alone


def test_validation_can_hide_the_targets_own_trade():
    """검증에서 대상 거래를 DB에서 지우면, 그 거래의 면적을 되찾지 못하고 건물의 다른 거래로 어림한다."""
    from validate_predict import predict_without_own_trade

    resolver = make_resolver()
    trades = resolver.trades["관악구"]
    own = trades[(trades["jibun"] == "100-1") & (trades["floor"] == 2)].iloc[0]
    blank = row(id=str(own["trade_id"]), area_m2="")

    seen = predict([blank], resolver, CONFIDENCE)[0]
    unseen = predict_without_own_trade(blank, "관악구", resolver, CONFIDENCE)
    assert "2층 실거래 기준" in seen["basis"]
    assert "같은 건물 실거래 중앙값" in unseen["basis"]
    assert unseen["confidence"] < seen["confidence"]
    assert len(resolver.trades["관악구"]) == len(trades)  # 끝나면 DB가 원래대로 돌아온다
