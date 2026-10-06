# model/baseline.py
"""기준선 모델: "같은 동의 최근 12개월 ㎡당 가격 중앙값 × 전용면적".

층·연식·위치를 전혀 보지 않는 가장 단순한 추정이다. 잘 맞히려는 모델이 아니라,
이후 모델이 요인을 넣을 때마다 오차가 얼마나 줄었는지 비교하는 출발점으로 쓴다.
"""
import numpy as np
import pandas as pd

from model.holdout import HOLDOUT_MONTHS, same_unit


def estimate_baseline(region_trades: pd.DataFrame, position: int) -> float:
    """한 권역의 거래(region_trades)에서 position번째 거래의 가격을 추정한다."""
    dong = region_trades["dong"].to_numpy()
    months_ago = region_trades["months_ago"].to_numpy()
    price_per_m2 = (region_trades["price"] / region_trades["area_m2"]).to_numpy()
    area = region_trades["area_m2"].to_numpy()

    recent = (months_ago >= 0) & (months_ago <= HOLDOUT_MONTHS)
    usable = recent & ~same_unit(region_trades, position)
    same_dong = usable & (dong == dong[position])
    # 같은 동에 쓸 거래가 없으면 권역 전체로 넓힌다.
    pool = same_dong if same_dong.any() else usable
    return float(np.median(price_per_m2[pool]) * area[position])
