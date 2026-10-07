# chart_data.py
"""그래프에 쓸 숫자를 CSV로 뽑아 reports/charts/data/ 에 저장한다. (1단계: 데이터 -> CSV)

그래프를 그리는 charts.py 는 여기서 만든 CSV만 읽는다.
CSV는 엑셀에서 바로 열리므로, 그래프의 숫자가 어디서 왔는지 눈으로 확인할 수 있다.

  01_monthly_price.csv        월별 ㎡당 거래가 중앙값 (만원, 3개월 이동 중앙값)
  02_monthly_count.csv        월별 거래 건수
  03_error_bands.csv          오차 구간별 물건 수와 비율 (제출 모델)
  04_model_comparison.csv     권역별 중앙값 오차율: 기준선, 가격식, 가격식+실거래
  05_confidence.csv           신뢰도 구간별: 모델이 말한 신뢰도와 실제로 맞힌 비율
  06_estimate_vs_actual.csv   물건별 실제 거래가와 추정 시세

01~02는 정제된 실거래 전체(data/trades.db), 03~06은 학습에 쓰지 않은 test 표본(reports/holdout.csv)에서 뽑는다.

실행 (검증 결과가 먼저 있어야 한다):
    uv run python validate.py --save reports/holdout.csv
    uv run python -m model.confidence
    uv run python chart_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from model.confidence import BAND_EDGES, BAND_NAMES, HOLDOUT_PATH, MODEL_PATH, PRICE_COLUMN, ConfidenceModel, features_of
from model.holdout import DB_PATH, error_summary, load_trades

DATA_DIR = Path("reports/charts/data")
REGIONS = ["강남구", "관악구", "화곡동"]
MODELS = ["기준선", "가격식", PRICE_COLUMN]
ERROR_EDGES = [10, 20, 30, 40]  # 오차율(%) 구간의 경계
ERROR_NAMES = ["10% 이내", "10~20%", "20~30%", "30~40%", "40% 초과"]
HIT_LIMIT = 0.20  # 오차가 이 안이면 '맞혔다'고 본다


def monthly_tables(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(월별 ㎡당 가격, 월별 거래 건수). 행은 월, 열은 권역이다."""
    frame = trades.assign(월=trades["deal_date"].str[:7], per_m2=trades["price_per_m2"] / 10_000)
    grouped = frame.groupby(["월", "region"])
    price = grouped["per_m2"].median().unstack("region")[REGIONS]
    count = grouped.size().unstack("region")[REGIONS].fillna(0).astype(int)
    # 한 달 거래가 적으면 중앙값이 튀므로, 앞뒤 달을 합쳐 3개월 중앙값으로 고르게 한다.
    price = price.rolling(3, min_periods=2, center=True).median().round(0)
    # 마지막 달은 신고 기한(30일) 때문에 아직 덜 찼으므로 뺀다.
    return price.iloc[:-1].reset_index(), count.iloc[:-1].reset_index()


def abs_error_pct(test: pd.DataFrame, column: str = PRICE_COLUMN) -> pd.Series:
    """추정가가 실제 거래가에서 몇 % 벗어났는지 (부호 없이)."""
    return (test[column] - test["price"]).abs() / test["price"] * 100


def error_bands(test: pd.DataFrame) -> pd.DataFrame:
    band = np.searchsorted(ERROR_EDGES, abs_error_pct(test), side="left")
    counts = [int((band == k).sum()) for k in range(len(ERROR_NAMES))]
    return pd.DataFrame({"오차구간": ERROR_NAMES, "건수": counts, "비율_%": [round(c / len(test) * 100, 1) for c in counts]})


def model_comparison(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in REGIONS + ["전체"]:
        part = test if region == "전체" else test[test["region"] == region]
        row = {"권역": region, "건수": len(part)}
        for name in MODELS:
            row[name] = error_summary(part["price"], part[name])["중앙값오차율_%"]
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_table(test: pd.DataFrame, model: ConfidenceModel) -> pd.DataFrame:
    confidence = model.confidence(features_of(test))
    band = np.searchsorted(BAND_EDGES, confidence, side="right")
    hit = (abs_error_pct(test) <= HIT_LIMIT * 100).to_numpy()
    rows = []
    for k, name in enumerate(BAND_NAMES):
        mask = band == k
        rows.append({"신뢰도구간": name, "건수": int(mask.sum()),
                     "말한신뢰도": round(float(confidence[mask].mean()), 2), "실제적중률": round(float(hit[mask].mean()), 2)})
    return pd.DataFrame(rows)


def estimate_vs_actual(test: pd.DataFrame) -> pd.DataFrame:
    error = (test[PRICE_COLUMN] - test["price"]) / test["price"] * 100
    return pd.DataFrame({
        "권역": test["region"],
        "거래유형": np.where(test["is_direct"] == 1, "직거래", "중개거래"),
        "실제_억": (test["price"] / 1e8).round(3),
        "추정_억": (test[PRICE_COLUMN] / 1e8).round(3),
        "오차율_%": error.round(1),
    })


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    trades = load_trades(DB_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH, encoding="utf-8-sig")
    test = holdout[holdout["split"] == "test"].reset_index(drop=True)
    price, count = monthly_tables(trades)
    tables = {
        "01_monthly_price.csv": price,
        "02_monthly_count.csv": count,
        "03_error_bands.csv": error_bands(test),
        "04_model_comparison.csv": model_comparison(test),
        "05_confidence.csv": confidence_table(test, ConfidenceModel.load(MODEL_PATH)),
        "06_estimate_vs_actual.csv": estimate_vs_actual(test),
    }
    for name, table in tables.items():
        # utf-8-sig 로 저장해야 엑셀에서 한글이 깨지지 않는다.
        table.to_csv(DATA_DIR / name, index=False, encoding="utf-8-sig")
        print(f"저장: {DATA_DIR / name} ({len(table):,}행)")


if __name__ == "__main__":
    main()
