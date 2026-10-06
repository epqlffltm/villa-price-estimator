# tests/test_clean.py
"""정제 단계가 지워야 할 행만 지우고, 사유별 건수를 맞게 세는지 작은 표본으로 확인한다."""
import pandas as pd

from model.clean import RAW_COLUMNS, clean


def make_raw(rows: list[dict]) -> pd.DataFrame:
    """정상 거래 한 건을 기본값으로 두고, 행마다 바꿀 값만 덮어쓴다."""
    base = {
        "sggCd": "11620", "umdNm": "신림동", "jibun": "598-178", "mhouseNm": "다온캐슬",
        "houseType": "다세대", "floor": "2", "excluUseAr": "29.09", "landAr": "16.83",
        "buildYear": "2023", "dealAmount": "32,300", "dealYear": "2026", "dealMonth": "9",
        "dealDay": "14", "dealingGbn": "중개거래", "buyerGbn": "개인", "slerGbn": "개인",
        "cdealType": "", "cdealDay": "",
    }
    return pd.DataFrame([{**base, **row} for row in rows], columns=RAW_COLUMNS)


def removed(summary: pd.DataFrame, step: str) -> int:
    return int(summary.loc[summary["step"] == step, "removed"].iloc[0])


def test_normal_trade_is_converted():
    trades, summary = clean(make_raw([{}]))
    row = trades.iloc[0]
    assert len(trades) == 1
    assert row["price"] == 323_000_000
    assert row["deal_date"] == "2026-09-14"
    assert (row["bonbun"], row["bubun"]) == (598, 178)
    assert row["sigungu"] == "서울특별시 관악구"
    assert row["price_per_m2"] == round(323_000_000 / 29.09)
    assert row["is_direct"] == 0


def test_cancelled_trades_are_removed():
    trades, summary = clean(make_raw([
        {},
        {"cdealType": "O", "cdealDay": "22.07.26"},
        {"cdealDay": "22.07.26"},  # 해제여부가 비어 있어도 해제일이 있으면 해제로 본다.
    ]))
    assert len(trades) == 1
    assert removed(summary, "해제 거래") == 2


def test_gangseo_keeps_only_hwagok():
    trades, summary = clean(make_raw([
        {"sggCd": "11500", "umdNm": "화곡동"},
        {"sggCd": "11500", "umdNm": "등촌동"},
        {"sggCd": "11680", "umdNm": "역삼동"},  # 강남구는 동을 가리지 않는다.
    ]))
    assert sorted(trades["dong"]) == ["역삼동", "화곡동"]
    assert removed(summary, "화곡동 외(강서구)") == 1


def test_unreadable_rows_are_removed():
    trades, summary = clean(make_raw([
        {},
        {"dealAmount": ""},
        {"excluUseAr": "0"},
        {"jibun": "1**"},
        {"floor": ""},
        {"dealDay": "31", "dealMonth": "2"},
    ]))
    assert len(trades) == 1
    assert removed(summary, "필수값 오류") == 5


def test_missing_build_year_is_kept():
    trades, _ = clean(make_raw([{"buildYear": "", "landAr": ""}]))
    assert len(trades) == 1
    assert pd.isna(trades.iloc[0]["build_year"])
    assert pd.isna(trades.iloc[0]["land_m2"])


def test_basement_and_direct_deal_are_kept_and_flagged():
    trades, _ = clean(make_raw([{"floor": "-1", "dealingGbn": "직거래"}]))
    assert trades.iloc[0]["floor"] == -1
    assert trades.iloc[0]["is_direct"] == 1


def test_summary_counts_add_up():
    _, summary = clean(make_raw([{}, {"cdealType": "O"}, {"sggCd": "11500", "umdNm": "등촌동"}, {"dealAmount": "x"}]))
    assert summary["remaining"].iloc[0] - summary["removed"].sum() == summary["remaining"].iloc[-1]
