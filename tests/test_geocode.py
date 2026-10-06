# tests/test_geocode.py
"""Kakao 검색 결과에서 요청한 지번과 일치하는 주소만 ok로 받아들이는지 확인한다. (실제 API는 호출하지 않는다)"""
import pytest

from model.geocode import GeocodeError, geocode_lot, pick_match


def doc(dong="신림동", main="598", sub="178", x="126.93", y="37.48", b_code="1162010200"):
    return {"address": {
        "address_name": f"서울 관악구 {dong} {main}-{sub}", "region_3depth_name": dong,
        "main_address_no": main, "sub_address_no": sub, "x": x, "y": y, "b_code": b_code,
    }}


class FakeResponse:
    def __init__(self, status_code, documents=None, text=""):
        self.status_code, self._documents, self.text = status_code, documents or [], text

    def json(self):
        return {"documents": self._documents}


class FakeSession:
    """정해 둔 응답을 순서대로 돌려준다."""
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params, headers, timeout):
        self.calls.append((params, headers))
        return self.responses.pop(0)


def test_exact_lot_is_ok():
    row = pick_match([doc()], "신림동", 598, 178)
    assert row["status"] == "ok"
    assert (row["lat"], row["lon"]) == (37.48, 126.93)  # y가 위도, x가 경도
    assert row["b_code"] == "1162010200"


def test_lot_without_bubun_matches_empty_sub_number():
    assert pick_match([doc(main="1740", sub="")], "신림동", 1740, 0)["status"] == "ok"


def test_picks_the_matching_document_among_several():
    row = pick_match([doc(sub="1", y="1.0"), doc(y="2.0")], "신림동", 598, 178)
    assert (row["status"], row["lat"]) == ("ok", 2.0)


@pytest.mark.parametrize("documents", [[doc(sub="179")], [doc(dong="봉천동")]])
def test_different_lot_is_mismatch(documents):
    assert pick_match(documents, "신림동", 598, 178)["status"] == "mismatch"


@pytest.mark.parametrize("documents", [[], [{"address": None}], [doc(x="", y="")]])
def test_no_usable_result_is_not_found(documents):
    row = pick_match(documents, "신림동", 598, 178)
    assert (row["status"], row["lat"]) == ("not_found", None)


def test_geocode_lot_builds_query_and_header():
    session = FakeSession(FakeResponse(200, [doc()]))
    row = geocode_lot("KEY", "서울특별시 관악구", "신림동", "598-178", 598, 178, session=session)
    params, headers = session.calls[0]
    assert row["status"] == "ok"
    assert params["query"] == "서울특별시 관악구 신림동 598-178"
    assert headers["Authorization"] == "KakaoAK KEY"


def test_auth_failure_stops_without_retry():
    session = FakeSession(FakeResponse(401, text="invalid key"))
    with pytest.raises(GeocodeError):
        geocode_lot("BAD", "서울특별시 관악구", "신림동", "598-178", 598, 178, session=session)
    assert len(session.calls) == 1


def test_server_error_is_retried(monkeypatch):
    monkeypatch.setattr("model.geocode.time.sleep", lambda seconds: None)
    session = FakeSession(FakeResponse(503), FakeResponse(200, [doc()]))
    row = geocode_lot("KEY", "서울특별시 관악구", "신림동", "598-178", 598, 178, session=session)
    assert row["status"] == "ok"
    assert len(session.calls) == 2
