# tests/test_confidence.py
"""신뢰도 식이 조건에 따라 의도한 방향으로 움직이고, 가격 구간이 신뢰도에 맞게 넓어지는지 확인한다."""
import numpy as np
import pandas as pd
import pytest

from model.confidence import (MAX_CONFIDENCE, MIN_CONFIDENCE, PRICE_COLUMN, ConfidenceModel, calibrate,
                              confidence_features, reliability_table)

MODEL = ConfidenceModel(
    intercept=1.6,
    coefficients={"근거량": 0.2, "흩어짐": -5.0, "지하층": -1.2, "1층": -0.2, "신축": -0.7, "노후": -0.4},
    half_widths=[0.40, 0.30, 0.22, 0.17],
)


def test_features_for_one_property():
    row = confidence_features(floor=-1, build_year=1990, weight_sum=0.0, spread=float("nan")).iloc[0]
    assert row["근거량"] == 0.0
    assert row["흩어짐"] == 0.35  # 근거가 없으면 가장 불확실한 값으로 채운다
    assert (row["지하층"], row["1층"], row["신축"], row["노후"]) == (1.0, 0.0, 0.0, 1.0)


def test_features_for_many_properties():
    features = confidence_features([2, 1], [2026, 2010], [10.0, 1000.0], [0.1, 0.9])
    assert features["신축"].tolist() == [1.0, 0.0]
    assert features["1층"].tolist() == [0.0, 1.0]
    assert features["근거량"].iloc[1] == pytest.approx(np.log1p(200))  # 200에서 자른다
    assert features["흩어짐"].iloc[1] == 0.6


def confidence_of(**changes) -> float:
    base = {"floor": 3, "build_year": 2012, "weight_sum": 20.0, "spread": 0.12}
    return MODEL.assess(300_000_000, **{**base, **changes})["confidence"]


def test_confidence_drops_in_harder_conditions():
    normal = confidence_of()
    assert confidence_of(floor=-1) < normal            # 지하층
    assert confidence_of(build_year=1985) < normal      # 노후
    assert confidence_of(build_year=2026) < normal      # 신축
    assert confidence_of(spread=0.4) < normal           # 근거 거래가 서로 많이 다르다
    assert confidence_of(weight_sum=0.5) < normal       # 근거 거래가 거의 없다


def test_confidence_stays_within_limits():
    low = confidence_of(floor=-1, build_year=1980, weight_sum=0.0, spread=float("nan"))
    high = MODEL.confidence(pd.DataFrame({"근거량": [100.0], "흩어짐": [0.0], "지하층": [0.0], "1층": [0.0], "신축": [0.0], "노후": [0.0]}))[0]
    assert MIN_CONFIDENCE <= low < 0.5
    assert high == MAX_CONFIDENCE


def test_interval_contains_the_estimate_and_widens_when_unsure():
    sure = MODEL.assess(300_000_000, floor=3, build_year=2012, weight_sum=50.0, spread=0.05)
    unsure = MODEL.assess(300_000_000, floor=-1, build_year=1985, weight_sum=0.2, spread=0.5)
    for result in (sure, unsure):
        assert result["price_low"] < 300_000_000 < result["price_high"]
    assert sure["confidence"] > unsure["confidence"]
    assert (unsure["price_high"] - unsure["price_low"]) > (sure["price_high"] - sure["price_low"])
    assert sure["price_high"] == pytest.approx(300_000_000 * np.exp(0.17))
    assert unsure["price_low"] == pytest.approx(300_000_000 * np.exp(-0.40))


def test_half_width_follows_band_edges():
    assert MODEL.half_width([0.59, 0.6, 0.69, 0.7, 0.8, 0.95]).tolist() == [0.40, 0.30, 0.30, 0.22, 0.17, 0.17]


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "confidence.json"
    MODEL.save(path)
    assert ConfidenceModel.load(path) == MODEL


def make_holdout(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """지하층은 오차가 크고 지상층은 작은 가상의 검증 결과."""
    rng = np.random.default_rng(seed)
    floor = rng.choice([-1, 2, 3, 4], size=n)
    noise = rng.normal(0, np.where(floor < 0, 0.35, 0.10))
    return pd.DataFrame({
        "floor": floor, "build_year": 2012, "weight_sum": 20.0, "spread": 0.12,
        "price": 300_000_000.0, PRICE_COLUMN: 300_000_000.0 * np.exp(noise),
    })


def test_calibrate_learns_that_basement_is_less_reliable():
    model = calibrate(make_holdout())
    assert model.coefficients["지하층"] < -1.0
    table = reliability_table(make_holdout(seed=1), model)
    # 새 표본에서도 말한 신뢰도와 실제로 맞힌 비율이 비슷해야 한다
    assert (table["평균 신뢰도"] - table["실제 20%이내"]).abs().max() < 0.06
    assert len(model.half_widths) == 4
