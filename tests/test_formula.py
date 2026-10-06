# tests/test_formula.py
"""가격식이 요인을 숫자로 바꾸는 방식과, 알고 있는 효과를 데이터에서 되찾는지 확인한다."""
import numpy as np
import pandas as pd
import pytest

from model.formula import build_features, fit_formula


def make_frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "dong": "신림동", "months_ago": 1.0, "deal_date": "2026-09-06", "build_year": 2014,
        "area_m2": 40.0, "floor": 2, "house_type": "다세대", "is_direct": 0, "price": 400_000_000,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_reference_property_has_only_the_base_value():
    features = build_features(make_frame([{}]), ["신림동", "봉천동"])
    row = features.iloc[0]
    assert row["기준값"] == 1.0
    assert row.drop("기준값").abs().sum() == 0.0  # 기준 물건은 모든 요인이 0


def test_each_factor_turns_on_its_own_column():
    features = build_features(make_frame([
        {"dong": "봉천동"},
        {"months_ago": 7.5},
        {"build_year": 2026},
        {"build_year": 1990},
        {"floor": -1},
        {"floor": 6},
        {"house_type": "연립"},
        {"is_direct": 1},
        {"months_ago": 200.0},
    ]), ["신림동", "봉천동"])
    assert features.at[0, "동 봉천동"] == 1.0
    assert features.at[1, "시점 6~9개월 전"] == 1.0
    assert features.at[2, "준공 0~1년"] == 1.0
    assert features.at[3, "준공 30년 이상"] == 1.0
    assert features.at[4, "지하층"] == 1.0
    assert features.at[5, "5층 이상"] == 1.0
    assert features.at[6, "연립"] == 1.0
    assert features.at[7, "직거래"] == 1.0
    assert features.at[8, "시점 57~60개월 전"] == 1.0  # 60개월보다 오래된 거래는 마지막 묶음


def test_area_feature_is_log_ratio_to_40():
    features = build_features(make_frame([{"area_m2": 40.0}, {"area_m2": 80.0}]), ["신림동"])
    assert features["면적(로그)"].tolist() == pytest.approx([0.0, np.log(2)])


def make_market(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """지하층은 30% 싸고 신축은 50% 비싼 가상의 시장."""
    rng = np.random.default_rng(seed)
    frame = make_frame([{} for _ in range(n)])
    frame["floor"] = rng.choice([-1, 2, 3], size=n)
    frame["build_year"] = rng.choice([2026, 2014], size=n)
    frame["months_ago"] = rng.uniform(0, 3, size=n)
    per_m2 = 10_000_000 * np.where(frame["floor"] < 0, 0.7, 1.0) * np.where(frame["build_year"] == 2026, 1.5, 1.0)
    frame["price"] = per_m2 * frame["area_m2"] * np.exp(rng.normal(0, 0.02, size=n))
    return frame


def test_fit_recovers_known_effects():
    effects = fit_formula(make_market()).effects()
    assert effects["지하층"] == pytest.approx(-30.0, abs=1.5)
    assert effects["준공 0~1년"] == pytest.approx(50.0, abs=2.0)


def test_excluded_rows_do_not_affect_the_fit():
    market = make_market()
    spoiled = market.copy()
    spoiled.loc[:49, "price"] *= 10
    exclude = np.arange(len(market)) < 50
    clean_fit = fit_formula(market, exclude).coefficients
    spoiled_fit = fit_formula(spoiled, exclude).coefficients
    assert spoiled_fit.to_numpy() == pytest.approx(clean_fit.to_numpy())
