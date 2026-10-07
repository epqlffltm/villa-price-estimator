# model/estimate.py
"""시세 추정: 가격식이 낸 값을 같은 건물·주변의 실거래로 보정한다.

    1. 가격식(formula.py)으로 "이 동, 이 연식, 이 층, 이 면적이면 보통 ㎡당 얼마"를 구한다.
    2. 실거래마다 "가격식보다 몇 % 비싸게/싸게 팔렸는지"(잔차)를 계산해 둔다.
    3. 대상 주변 실거래의 잔차를 가중평균한다. 가까울수록, 최근일수록, 같은 건물일수록 크게 반영한다.
    4. 가격식 값 × (1 + 보정) × 전용면적 = 추정 시세

대상과 같은 호로 보이는 거래(같은 지번·층·면적)는 쓰지 않는다.
제출 프로그램(predict.py)은 estimate_target()을 쓴다. 물건마다 그 물건과 같은 호의 거래만 빼고 가격식을 만들기 때문에,
입력 CSV에 다른 물건이 무엇이 있든 같은 물건은 항상 같은 값이 나온다.
아래 네 개의 설정값은 홀드아웃 dev 쪽 오차를 재서 골랐다.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from model.formula import FormulaData, PriceFormula, fit_formula, fit_without, prepare_formula
from model.holdout import REF_DATE, SAME_UNIT_AREA_TOLERANCE

DISTANCE_SCALE_M = 50.0       # 50m 멀어질 때마다 반영 비중이 약 1/2.7로 준다
TIME_SCALE_MONTHS = 24.0      # 24개월 오래될 때마다 반영 비중이 약 1/2.7로 준다
SAME_BUILDING_BOOST = 10.0    # 같은 건물(같은 지번)의 거래는 10배로 반영한다
PRIOR_WEIGHT = 0.3            # 주변 거래가 거의 없으면 보정을 0 쪽으로 당긴다
NEARBY_RADIUS_M = 100.0       # 근거 문장에 "반경 100m 내 N건"으로 셀 범위
EARTH_RADIUS_M = 6_371_000.0


@dataclass
class RegionModel:
    """한 권역의 가격식과, 실거래별 잔차."""
    trades: pd.DataFrame
    formula: PriceFormula
    residual: np.ndarray   # log(실제 ㎡당 가격) - log(가격식 값)
    usable: np.ndarray     # 보정에 쓸 수 있는 거래인지 (학습에서 뺀 행도 보정에는 쓴다)


def fit_region(region_trades: pd.DataFrame, exclude: np.ndarray | None = None) -> RegionModel:
    """가격식을 만들고 모든 거래의 잔차를 계산한다. exclude 행은 가격식 학습에서만 뺀다."""
    region_trades = region_trades.reset_index(drop=True)
    formula = fit_formula(region_trades, exclude)
    actual = np.log((region_trades["price"] / region_trades["area_m2"]).to_numpy().astype(float))
    residual = actual - formula.log_price_per_m2(region_trades)
    usable = region_trades["months_ago"].to_numpy() >= 0
    return RegionModel(region_trades, formula, residual, usable)


def make_target(dong, lot, floor, area_m2, build_year, house_type="다세대", lat=None, lon=None,
                area_is_guess=False) -> dict:
    """추정 대상 한 건. 기준일에 중개거래로 팔린다고 가정한다.

    area_is_guess: 전용면적을 찾지 못해 어림값을 넣었으면 True. 이때는 면적으로 같은 호를 가릴 수 없으므로
    같은 건물·같은 층의 거래를 모두 같은 호로 보고 뺀다.
    """
    return {
        "dong": dong, "lot": lot, "floor": int(floor), "area_m2": float(area_m2),
        "build_year": int(build_year), "house_type": house_type,
        "lat": None if lat is None or pd.isna(lat) else float(lat),
        "lon": None if lon is None or pd.isna(lon) else float(lon),
        "area_is_guess": bool(area_is_guess),
    }


def same_unit_as_target(trades: pd.DataFrame, target: dict) -> np.ndarray:
    """대상과 같은 호로 보이는 거래를 True로 표시한다."""
    same_floor = (trades["lot"].to_numpy() == target["lot"]) & (trades["floor"].to_numpy() == target["floor"])
    if target.get("area_is_guess"):
        return same_floor
    return same_floor & (np.abs(trades["area_m2"].to_numpy() - target["area_m2"]) < SAME_UNIT_AREA_TOLERANCE)


def target_from_trade(region_trades: pd.DataFrame, position: int) -> dict:
    """검증용: 실제 거래 한 건의 주소·층·면적·준공년도로 대상을 만든다. 가격과 거래일은 넘기지 않는다."""
    row = region_trades.iloc[position]
    return make_target(row["dong"], row["lot"], row["floor"], row["area_m2"], row["build_year"],
                       row["house_type"], row["lat"], row["lon"])


def distance_m(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """한 점에서 여러 점까지의 직선 거리(m). 지구를 구로 보는 하버사인 공식."""
    to_rad = np.pi / 180.0
    d_lat = (lats - lat) * to_rad
    d_lon = (lons - lon) * to_rad
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat * to_rad) * np.cos(lats * to_rad) * np.sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def estimate(model: RegionModel, target: dict) -> dict:
    """대상 한 건의 시세와, 그 근거가 된 숫자를 돌려준다."""
    trades = model.trades
    same_building = trades["lot"].to_numpy() == target["lot"]
    same_unit = same_unit_as_target(trades, target)

    # 거리: 같은 건물은 0m. 좌표가 없으면 아주 먼 것으로 보아 반영하지 않는다.
    if target["lat"] is None:
        distance = np.full(len(trades), np.inf)
    else:
        distance = distance_m(target["lat"], target["lon"], trades["lat"].to_numpy(), trades["lon"].to_numpy())
        distance = np.where(np.isnan(distance), np.inf, distance)
    distance = np.where(same_building, 0.0, distance)

    months_ago = trades["months_ago"].to_numpy()
    weight = np.exp(-distance / DISTANCE_SCALE_M) * np.exp(-np.clip(months_ago, 0, None) / TIME_SCALE_MONTHS)
    weight = np.where(same_building, weight * SAME_BUILDING_BOOST, weight)
    weight = np.where(model.usable & ~same_unit, weight, 0.0)

    weight_sum = float(weight.sum())
    adjustment = float((weight * model.residual).sum() / (weight_sum + PRIOR_WEIGHT))
    if weight_sum > 0:
        center = (weight * model.residual).sum() / weight_sum
        spread = float(np.sqrt((weight * (model.residual - center) ** 2).sum() / weight_sum))
    else:
        spread = float("nan")

    # 가격식에는 "기준일에, 중개거래로" 팔리는 것으로 넣는다.
    target_frame = pd.DataFrame([{**target, "months_ago": 0.0, "deal_date": REF_DATE, "is_direct": 0}])
    formula_log = float(model.formula.log_price_per_m2(target_frame)[0])
    formula_price = float(np.exp(formula_log) * target["area_m2"])
    price = float(np.exp(formula_log + adjustment) * target["area_m2"])

    counted = model.usable & ~same_unit
    return {
        "price": price,
        "formula_price": formula_price,                      # 보정 전, 가격식만으로 낸 값
        "adjustment_pct": (np.exp(adjustment) - 1) * 100,     # 주변 실거래로 보정한 비율
        "n_same_building": int((counted & same_building).sum()),
        "n_nearby": int((counted & ~same_building & (distance <= NEARBY_RADIUS_M)).sum()),
        "weight_sum": weight_sum,                             # 근거가 된 거래의 양 (가중치 합)
        "spread": spread,                                     # 근거 거래들끼리 얼마나 흩어져 있는지 (로그 단위)
    }


@dataclass
class PreparedRegion:
    """한 권역의 거래와, 가격식을 빨리 다시 만들기 위해 미리 계산해 둔 값."""
    trades: pd.DataFrame
    formula_data: FormulaData
    usable: np.ndarray


def prepare_region(region_trades: pd.DataFrame) -> PreparedRegion:
    region_trades = region_trades.reset_index(drop=True)
    return PreparedRegion(region_trades, prepare_formula(region_trades), region_trades["months_ago"].to_numpy() >= 0)


def estimate_target(region: PreparedRegion, target: dict) -> dict:
    """대상 한 건을 실행 규칙 그대로 추정한다.

    1. 대상과 같은 호로 보이는 거래를 빼고 가격식을 만든다 (그 물건의 거래가를 외우지 않도록).
    2. 그 가격식을 주변 실거래로 보정한다 (여기서도 같은 호는 빠진다).
    다른 물건의 정보는 전혀 들어가지 않으므로, 한 건만 넣든 20건을 함께 넣든 결과가 같다.
    """
    data = region.formula_data
    formula = fit_without(data, same_unit_as_target(region.trades, target))
    residual = data.log_price - data.features @ formula.coefficients.to_numpy()
    return estimate(RegionModel(region.trades, formula, residual, region.usable), target)
