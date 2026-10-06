# model/formula.py
"""가격식: ㎡당 가격을 요인별 비율로 설명하는 회귀식 (권역마다 따로 만든다).

    log(㎡당 가격) = 기준값 + 동 + 거래 시점 + 연식 + 면적 + 층 + 주택 유형 + 거래 방식

로그를 쓰기 때문에 각 요인의 계수는 "기준 대비 몇 % 비싼가"로 읽힌다 (예: 지하층 -30%).
기준이 되는 물건은 "가장 거래가 많은 동, 최근 3개월, 준공 10~14년, 전용 40㎡, 2~3층, 다세대, 중개거래"다.

이 식만으로는 같은 동 안의 위치 차이를 모른다. 그 차이는 estimate.py에서 주변 실거래로 보정한다.

실행하면 권역별 요인 효과를 출력한다:
    uv run python -m model.formula
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TIME_BIN_MONTHS = 3   # 거래 시점을 3개월 단위로 묶는다
N_TIME_BINS = 20      # 20묶음 = 60개월. 그보다 오래된 거래는 마지막 묶음에 넣는다
BASE_AREA = 40.0      # ㎡

# (이름, 준공 후 최소 연수, 최대 연수). 10~14년이 기준이라 목록에 없다.
AGE_GROUPS = [
    ("준공 0~1년", 0, 1),
    ("준공 2~4년", 2, 4),
    ("준공 5~9년", 5, 9),
    ("준공 15~19년", 15, 19),
    ("준공 20~29년", 20, 29),
    ("준공 30년 이상", 30, 999),
]


@dataclass
class PriceFormula:
    dongs: list[str]          # 첫 번째가 기준 동
    coefficients: pd.Series   # 요인 이름 -> 계수(로그 단위)

    def log_price_per_m2(self, frame: pd.DataFrame) -> np.ndarray:
        """각 행의 log(㎡당 가격)을 계산한다."""
        features = build_features(frame, self.dongs)
        return features.to_numpy() @ self.coefficients.to_numpy()

    def effects(self) -> pd.Series:
        """요인별 효과를 %로 바꾼다. 기준값과 거래 시점은 뺀다."""
        names = [n for n in self.coefficients.index if n != "기준값" and not n.startswith("시점 ")]
        return ((np.exp(self.coefficients[names]) - 1) * 100).round(1)


def build_features(frame: pd.DataFrame, dongs: list[str]) -> pd.DataFrame:
    """거래(또는 추정 대상)를 회귀식에 넣을 숫자 표로 바꾼다. 해당하면 1, 아니면 0."""
    columns = {"기준값": np.ones(len(frame))}

    dong = frame["dong"].to_numpy()
    for name in dongs[1:]:
        columns[f"동 {name}"] = (dong == name).astype(float)

    time_bin = np.clip((frame["months_ago"].to_numpy() // TIME_BIN_MONTHS).astype(int), 0, N_TIME_BINS - 1)
    for k in range(1, N_TIME_BINS):
        columns[f"시점 {k * TIME_BIN_MONTHS}~{(k + 1) * TIME_BIN_MONTHS}개월 전"] = (time_bin == k).astype(float)

    # 연식 = 거래한 해 - 준공년도
    deal_year = frame["deal_date"].str[:4].astype(int).to_numpy()
    age = np.clip(deal_year - frame["build_year"].to_numpy().astype(int), 0, None)
    for name, low, high in AGE_GROUPS:
        columns[name] = ((age >= low) & (age <= high)).astype(float)

    columns["면적(로그)"] = np.log(frame["area_m2"].to_numpy().astype(float) / BASE_AREA)

    floor = frame["floor"].to_numpy()
    columns["지하층"] = (floor < 0).astype(float)
    columns["1층"] = (floor == 1).astype(float)
    columns["4층"] = (floor == 4).astype(float)
    columns["5층 이상"] = (floor >= 5).astype(float)

    columns["연립"] = (frame["house_type"].to_numpy() == "연립").astype(float)
    columns["직거래"] = frame["is_direct"].to_numpy().astype(float)
    return pd.DataFrame(columns, index=frame.index)


def fit_formula(region_trades: pd.DataFrame, exclude: np.ndarray | None = None) -> PriceFormula:
    """한 권역의 거래로 가격식을 만든다. exclude가 True인 행은 학습에서 뺀다."""
    dongs = list(region_trades["dong"].value_counts().index)  # 거래가 많은 순. 첫 번째가 기준 동
    features = build_features(region_trades, dongs)
    target = np.log((region_trades["price"] / region_trades["area_m2"]).to_numpy().astype(float))
    use = np.ones(len(region_trades), dtype=bool) if exclude is None else ~np.asarray(exclude)
    # 최소제곱법: 실제 log 가격과 식이 낸 값의 차이를 제곱해 더한 값이 가장 작아지는 계수를 찾는다.
    coefficients, *_ = np.linalg.lstsq(features.to_numpy()[use], target[use], rcond=None)
    return PriceFormula(dongs=dongs, coefficients=pd.Series(coefficients, index=features.columns))


def main() -> None:
    from model.holdout import load_trades

    trades = load_trades()
    table = {}
    for region, region_trades in trades.groupby("region"):
        formula = fit_formula(region_trades.reset_index(drop=True))
        effects = formula.effects()
        table[region] = effects[[name for name in effects.index if not name.startswith("동 ")]]
    print("== 요인별 효과 (%), 기준: 준공 10~14년 · 2~3층 · 다세대 · 중개거래 ==")
    print(pd.DataFrame(table).to_string())
    print("\n면적(로그)는 면적이 약 2.7배가 될 때 ㎡당 가격이 변하는 비율이다. 음수면 넓을수록 ㎡당 가격이 싸다.")


if __name__ == "__main__":
    main()
