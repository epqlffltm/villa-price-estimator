# tests/test_holdout.py
"""검증 규칙(홀드아웃 범위, 같은 호 제외, dev/test 나누기, 오차 계산)과 기준선 모델을 작은 표본으로 확인한다."""
import pandas as pd
import pytest

from model.baseline import estimate_baseline
from model.holdout import N_FOLDS, add_holdout_columns, error_summary, fold_of, same_unit, split_of


def make_trades(rows: list[dict]) -> pd.DataFrame:
    base = {
        "sgg_cd": "11620", "sigungu": "서울특별시 관악구", "dong": "신림동", "jibun": "598-178",
        "floor": 2, "area_m2": 30.0, "price": 300_000_000, "deal_date": "2026-09-06",
    }
    trades = pd.DataFrame([{**base, **row} for row in rows])
    trades.insert(0, "trade_id", range(1, len(trades) + 1))
    return add_holdout_columns(trades, ref_date="2026-10-06")


def test_region_is_dong_for_gangseo_and_gu_for_others():
    trades = make_trades([{}, {"sgg_cd": "11500", "sigungu": "서울특별시 강서구", "dong": "화곡동"}])
    assert list(trades["region"]) == ["관악구", "화곡동"]


def test_holdout_is_last_12_months():
    trades = make_trades([{"deal_date": "2026-09-06"}, {"deal_date": "2025-10-10"}, {"deal_date": "2025-09-01"}])
    assert list(trades["is_holdout"]) == [True, True, False]
    assert trades["months_ago"].iloc[0] == pytest.approx(1.0, abs=0.05)


def test_split_is_stable_and_same_for_one_lot():
    assert split_of("11620 신림동 598-178") == split_of("11620 신림동 598-178")
    trades = make_trades([{"floor": 2}, {"floor": 4}, {"jibun": "1-1"}, {"jibun": "1-2"}, {"jibun": "1-3"}, {"jibun": "1-4"}])
    assert trades["split"].iloc[0] == trades["split"].iloc[1]
    assert set(trades["split"]) == {"dev", "test"}


def test_fold_is_stable_and_within_range():
    folds = {fold_of(f"11620 신림동 {n}-1") for n in range(200)}
    assert folds == set(range(N_FOLDS))
    assert fold_of("11620 신림동 598-178") == fold_of("11620 신림동 598-178")


def test_same_unit_needs_same_lot_floor_and_area():
    trades = make_trades([
        {},                       # 대상
        {"deal_date": "2024-01-01"},  # 같은 호의 예전 거래
        {"area_m2": 30.3},        # 면적 차이 0.5㎡ 미만 -> 같은 호
        {"floor": 3},             # 같은 건물 다른 층
        {"area_m2": 45.0},        # 같은 층 다른 면적
        {"jibun": "598-179"},     # 다른 건물
    ])
    assert list(same_unit(trades, 0)) == [True, True, True, False, False, False]


def test_error_summary():
    # 오차율: +10%, -10%, +40%, 0%
    summary = error_summary([100, 100, 100, 100], [110, 90, 140, 100])
    assert summary["건수"] == 4
    assert summary["중앙값오차율_%"] == 10.0
    assert summary["MAPE_%"] == 15.0
    assert summary["10%이내_%"] == 75.0
    assert summary["20%이내_%"] == 75.0
    assert summary["치우침_%"] == 5.0


def test_baseline_uses_dong_median_without_the_target_unit():
    trades = make_trades([
        {"price": 900_000_000},                       # 대상: 자기 가격은 쓰이면 안 된다
        {"floor": 3, "price": 300_000_000},           # ㎡당 1,000만
        {"floor": 4, "price": 360_000_000},           # ㎡당 1,200만
        {"jibun": "1-1", "area_m2": 50.0, "price": 400_000_000},  # ㎡당 800만
        {"dong": "봉천동", "jibun": "2-2", "price": 30_000_000},     # 다른 동
        {"jibun": "3-3", "price": 30_000_000, "deal_date": "2024-01-01"},  # 12개월 밖
    ])
    assert estimate_baseline(trades, 0) == pytest.approx(10_000_000 * 30.0)


def test_baseline_falls_back_to_region_when_dong_is_empty():
    trades = make_trades([
        {"dong": "남현동", "jibun": "9-9"},
        {"dong": "봉천동", "jibun": "2-2", "price": 600_000_000},
    ])
    assert estimate_baseline(trades, 0) == pytest.approx(600_000_000)
