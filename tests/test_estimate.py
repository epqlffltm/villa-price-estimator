# tests/test_estimate.py
"""실거래 보정이 의도대로 움직이는지 가상의 동네로 확인한다: 같은 호 제외, 같은 건물 우선, 거리·좌표 없음 처리."""
import numpy as np
import pandas as pd
import pytest

from model.estimate import distance_m, estimate, estimate_target, fit_region, make_target, prepare_region, same_unit_as_target
from model.holdout import add_holdout_columns

BASE_PER_M2 = 10_000_000
LAT, LON = 37.48, 126.93


def make_neighborhood(premium_lot: str = "100-1", premium: float = 1.3) -> pd.DataFrame:
    """60개 건물이 약 220m 간격으로 흩어져 있고 건물마다 4건씩 거래된 동네. premium_lot만 30% 비싸다."""
    rows = []
    for building in range(60):
        lot = f"{100 + building}-1"
        for unit in range(4):
            rows.append({
                "sgg_cd": "11620", "sigungu": "서울특별시 관악구", "dong": "신림동", "jibun": lot,
                "floor": unit + 1, "area_m2": 40.0, "build_year": 2014, "house_type": "다세대", "is_direct": 0,
                "deal_date": "2026-08-01", "lat": LAT + 0.002 * (building // 8), "lon": LON + 0.0025 * (building % 8),
                "price": BASE_PER_M2 * 40.0 * (premium if lot == premium_lot else 1.0),
            })
    trades = pd.DataFrame(rows)
    trades.insert(0, "trade_id", range(1, len(trades) + 1))
    return add_holdout_columns(trades, ref_date="2026-10-06")


def target_in(lot: str, floor: int = 2, lat=LAT, lon=LON) -> dict:
    return make_target("신림동", f"11620 신림동 {lot}", floor, 40.0, 2014, "다세대", lat, lon)


def test_distance_between_known_points():
    # 위도 0.001도는 약 111m
    assert distance_m(37.0, 127.0, np.array([37.001]), np.array([127.0]))[0] == pytest.approx(111.2, abs=0.5)


def test_same_building_trades_pull_the_estimate():
    model = fit_region(make_neighborhood())
    result = estimate(model, target_in("100-1"))
    assert result["n_same_building"] == 3  # 4건 중 같은 호(2층)는 뺀다
    assert result["adjustment_pct"] > 20
    assert result["price"] == pytest.approx(BASE_PER_M2 * 40.0 * 1.3, rel=0.05)


def test_ordinary_building_gets_the_market_price():
    model = fit_region(make_neighborhood())
    result = estimate(model, target_in("130-1", lat=LAT + 0.002 * 3, lon=LON + 0.0025 * 6))
    assert result["price"] == pytest.approx(BASE_PER_M2 * 40.0, rel=0.03)


def test_own_trade_price_is_never_used():
    normal = make_neighborhood()
    spoiled = normal.copy()
    own = (spoiled["jibun"] == "100-1") & (spoiled["floor"] == 2)
    spoiled.loc[own, "price"] *= 5  # 대상 호의 거래가만 5배로 바꾼다
    exclude = own.to_numpy()
    a = estimate(fit_region(normal, exclude), target_in("100-1"))
    b = estimate(fit_region(spoiled, exclude), target_in("100-1"))
    assert b["price"] == pytest.approx(a["price"])


def test_far_trades_are_ignored():
    model = fit_region(make_neighborhood())
    # 가장 가까운 건물에서도 1km 넘게 떨어진, 거래 기록이 없는 새 지번
    result = estimate(model, target_in("999-9", lat=LAT - 0.01, lon=LON - 0.01))
    assert result["n_same_building"] == 0
    assert result["n_nearby"] == 0
    assert result["weight_sum"] < 0.001
    assert result["price"] == pytest.approx(result["formula_price"], rel=0.001)


def test_target_without_coordinates_uses_only_its_building():
    model = fit_region(make_neighborhood())
    result = estimate(model, target_in("100-1", lat=None, lon=None))
    assert result["n_same_building"] == 3
    assert result["n_nearby"] == 0
    assert result["adjustment_pct"] > 20


def test_unknown_lot_without_coordinates_falls_back_to_formula():
    model = fit_region(make_neighborhood())
    result = estimate(model, target_in("999-9", lat=None, lon=None))
    assert result["weight_sum"] == 0
    assert np.isnan(result["spread"])
    assert result["price"] == pytest.approx(result["formula_price"])



def test_fast_refit_gives_the_same_price_as_a_full_refit():
    """합계에서 덜어내는 빠른 계산(estimate_target)이, 같은 호를 빼고 처음부터 다시 만든 가격식과 같은 값을 낸다."""
    trades = make_neighborhood()
    region = prepare_region(trades)
    for lot, floor in [("100-1", 2), ("101-1", 1), ("130-1", 4)]:
        target = target_in(lot, floor)
        slow = estimate(fit_region(trades, exclude=same_unit_as_target(trades, target)), target)
        fast = estimate_target(region, target)
        assert fast["price"] == pytest.approx(slow["price"], rel=1e-6)
        assert fast["n_same_building"] == slow["n_same_building"]
