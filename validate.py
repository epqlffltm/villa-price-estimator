# validate.py
"""홀드아웃(최근 12개월 거래)을 모델별로 맞혀 보고 오차표를 출력한다.

모델을 추가할 때는 MODELS에 한 줄만 더하면 같은 대상·같은 규칙으로 비교된다.

실행:
    uv run python validate.py
"""
import argparse
from pathlib import Path

import pandas as pd

from model.baseline import estimate_baseline
from model.holdout import DB_PATH, error_summary, load_trades

# 이름 -> (한 권역의 거래, 대상 위치)를 받아 추정가(원)를 돌려주는 함수
MODELS = {
    "기준선": estimate_baseline,
}


def predict_holdout(trades: pd.DataFrame, estimate) -> pd.DataFrame:
    """권역별로 홀드아웃 거래를 하나씩 추정해 (trade_id, 추정가) 표를 만든다."""
    rows = []
    for _, region_trades in trades.groupby("region"):
        region_trades = region_trades.reset_index(drop=True)
        for position in region_trades.index[region_trades["is_holdout"]]:
            rows.append((region_trades.at[position, "trade_id"], estimate(region_trades, position)))
    return pd.DataFrame(rows, columns=["trade_id", "predicted"])


def error_table(holdout: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in holdout.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(by, keys)), **error_summary(group["price"], group["predicted"])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    trades = load_trades(args.db)
    holdout = trades[trades["is_holdout"]]
    print(f"전체 거래 {len(trades):,}건 / 홀드아웃(최근 12개월) {len(holdout):,}건 "
          f"(dev {int((holdout['split'] == 'dev').sum()):,} · test {int((holdout['split'] == 'test').sum()):,})")

    for name, estimate in MODELS.items():
        predicted = predict_holdout(trades, estimate)
        result = holdout.merge(predicted, on="trade_id")
        print(f"\n== {name}: 권역별 ==")
        print(error_table(result, ["region"]).to_string(index=False))
        print(f"\n== {name}: 권역 × dev/test ==")
        print(error_table(result, ["region", "split"]).to_string(index=False))
        print(f"\n== {name}: 거래 방식별 (0 중개, 1 직거래) ==")
        print(error_table(result, ["is_direct"]).to_string(index=False))


if __name__ == "__main__":
    main()
