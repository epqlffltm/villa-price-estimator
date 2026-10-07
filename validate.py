# validate.py
"""홀드아웃(최근 12개월 거래)을 모델별로 맞혀 보고 오차표를 출력한다.

세 모델을 같은 대상·같은 규칙으로 비교한다.
  기준선        같은 동 ㎡당 중앙값 × 면적
  가격식        연식·층·면적 등 요인만 반영 (주변 실거래 보정 없음)
  가격식+실거래  가격식을 같은 건물·주변 실거래로 보정 (제출 모델)

가격식은 학습이 필요하므로, 지번을 5묶음으로 나눠 맞혀 볼 묶음의 홀드아웃 거래를 빼고 학습한다.
제출 프로그램은 대상과 같은 호의 거래만 빼므로, 여기서는 그보다 많이(묶음의 최근 12개월 거래 전부) 빼고 재는 셈이다.
설정값과 신뢰도 식은 이 결과로 정했다. 제출 프로그램을 그대로 돌려 잰 오차는 validate_predict.py가 낸다.

실행:
    uv run python validate.py
    uv run python validate.py --save reports/holdout.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from model.baseline import estimate_baseline
from model.estimate import estimate, fit_region, target_from_trade
from model.holdout import DB_PATH, N_FOLDS, error_summary, load_trades

FINAL_MODEL = "가격식+실거래"


def predict_region(region_trades: pd.DataFrame) -> pd.DataFrame:
    """한 권역의 홀드아웃 거래를 세 모델로 추정한다. 한 행 = 거래 한 건."""
    region_trades = region_trades.reset_index(drop=True)
    is_holdout = region_trades["is_holdout"].to_numpy()
    fold = region_trades["fold"].to_numpy()
    rows = []
    for k in range(N_FOLDS):
        in_fold = is_holdout & (fold == k)
        model = fit_region(region_trades, exclude=in_fold)  # 이 묶음의 정답을 보지 않고 가격식을 만든다
        for position in in_fold.nonzero()[0]:
            result = estimate(model, target_from_trade(region_trades, position))
            rows.append({
                "trade_id": region_trades.at[position, "trade_id"],
                "기준선": estimate_baseline(region_trades, position),
                "가격식": result["formula_price"],
                FINAL_MODEL: result["price"],
                **{key: result[key] for key in ["adjustment_pct", "n_same_building", "n_nearby", "weight_sum", "spread"]},
            })
    return pd.DataFrame(rows)


def comparison_table(holdout: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """권역 × dev/test별로 모델의 중앙값 오차율과 20% 이내 비율을 나란히 놓는다."""
    rows = []
    for (region, split), group in holdout.groupby(["region", "split"]):
        row = {"region": region, "split": split, "건수": len(group)}
        for name in models:
            summary = error_summary(group["price"], group[name])
            row[f"{name}_중앙값%"] = summary["중앙값오차율_%"]
            row[f"{name}_20%이내"] = summary["20%이내_%"]
        rows.append(row)
    return pd.DataFrame(rows)


def breakdown(holdout: pd.DataFrame, label: pd.Series, model: str) -> pd.DataFrame:
    rows = []
    for key, group in holdout.groupby(label, observed=True):
        rows.append({label.name: key, **error_summary(group["price"], group[model])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--save", type=Path, help="홀드아웃별 추정 결과를 CSV로 저장한다")
    args = parser.parse_args()

    trades = load_trades(args.db)
    predicted = pd.concat([predict_region(group) for _, group in trades.groupby("region")])
    holdout = trades[trades["is_holdout"]].merge(predicted, on="trade_id")
    print(f"전체 거래 {len(trades):,}건 / 홀드아웃(최근 12개월) {len(holdout):,}건")

    models = ["기준선", "가격식", FINAL_MODEL]
    print("\n== 모델 비교: 중앙값 오차율(%)과 20% 이내 비율(%) ==")
    print(comparison_table(holdout, models).to_string(index=False))

    test = holdout[holdout["split"] == "test"]
    print(f"\n== {FINAL_MODEL}: test 권역별 ==")
    print(breakdown(test, test["region"], FINAL_MODEL).to_string(index=False))

    print(f"\n== {FINAL_MODEL}: test 조건별 ==")
    labels = [
        test["is_direct"].map({0: "중개거래", 1: "직거래"}).rename("거래 방식"),
        test["house_type"].rename("주택 유형"),
        pd.cut(test["floor"], [-9, 0, 1, 4, 99], labels=["지하", "1층", "2~4층", "5층 이상"]).rename("층"),
        pd.cut(test["n_same_building"], [-1, 0, 2, 5, 9999], labels=["0건", "1~2건", "3~5건", "6건 이상"]).rename("같은 건물 사례"),
    ]
    for label in labels:
        print(breakdown(test, label, FINAL_MODEL).to_string(index=False))
        print()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        columns = ["trade_id", "region", "split", "dong", "jibun", "building_name", "floor", "area_m2", "build_year",
                   "house_type", "is_direct", "deal_date", "price", *models,
                   "adjustment_pct", "n_same_building", "n_nearby", "weight_sum", "spread"]
        holdout[columns].to_csv(args.save, index=False, encoding="utf-8-sig")
        print(f"저장: {args.save}")


if __name__ == "__main__":
    main()
