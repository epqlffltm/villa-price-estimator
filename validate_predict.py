# validate_predict.py
"""제출 프로그램(predict.py)을 그대로 돌려 오차를 잰다. 면적을 넣은 입력과 비운 입력을 따로 잰다.

validate.py 와의 차이
  validate.py          실거래의 면적·준공년도를 그대로 쓰고, 가격식은 묶음 단위로 정답을 빼고 만든다 (설정값을 고를 때 쓴 방식)
  validate_predict.py  입력 한 줄(시군구·동·지번·층)을 만들어 predict()에 넣는다. 값 채우기, 물건별 가격식, 신뢰도 감점까지
                       채점 때와 같은 경로를 탄다.

대상은 test 홀드아웃(최근 12개월 거래 중 설정에 쓰지 않은 절반)이다. 실거래가에는 호가 없어서 ho는 비워 둔다.
기본은 키 없이 수집해 둔 DB만 쓴다. 이때 비운 면적은 같은 건물 실거래 기록에서 찾는다.
--online 을 주면 환경변수의 키로 건축물대장·Kakao를 조회한다. 호출 수가 많으므로 --sample 과 함께 쓴다.

실행 (1~2분. 100건마다 진행 상황이 출력된다):
    uv run python validate_predict.py
    uv run --env-file .env python validate_predict.py --online --sample 100
"""
import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd

from model.confidence import HIT_TOLERANCE, MODEL_PATH, ConfidenceModel
from model.holdout import DB_PATH, error_summary, load_trades
from model.resolve import Resolver
from predict import predict

SAVE_PATH = Path("reports/predict_check.csv")
SAMPLE_SEED = 2026
CASES = {"면적 입력": True, "면적 없음": False}
CHUNK = 100  # 이만큼씩 넣고 진행 상황을 출력한다


def make_rows(test: pd.DataFrame, with_area: bool) -> list[dict]:
    """실거래 한 건을 입력 CSV의 한 줄로 바꾼다. 가격과 거래일은 넘기지 않는다."""
    return [{
        "id": str(trade["trade_id"]), "sigungu": trade["sigungu"], "dong": trade["dong"], "jibun": trade["jibun"],
        "floor": str(trade["floor"]), "ho": "", "area_m2": f"{trade['area_m2']}" if with_area else "",
    } for trade in test.to_dict("records")]


def run_case(name: str, test: pd.DataFrame, with_area: bool, resolver: Resolver, confidence_model: ConfidenceModel) -> pd.DataFrame:
    rows = make_rows(test, with_area)
    outputs = []
    # 물건은 서로 영향을 주지 않으므로 나눠서 넣어도 결과가 같다. 오래 걸리니 진행 상황을 보여 주려고 나눈다.
    for start in range(0, len(rows), CHUNK):
        outputs += predict(rows[start:start + CHUNK], resolver, confidence_model)
        print(f"\r{name}: {len(outputs):,} / {len(rows):,}건", end="", flush=True)
    print()
    outputs = pd.DataFrame(outputs)
    result = test[["trade_id", "region", "price"]].reset_index(drop=True)
    for column in ["price_est", "price_low", "price_high", "confidence", "status"]:
        result[column] = outputs[column]
    for column in ["price_est", "price_low", "price_high"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")  # 실패한 줄은 빈칸 -> NaN
    return result


def summarize(result: pd.DataFrame) -> dict:
    ok = result[result["status"] == "ok"]
    hit = (ok["price_est"] - ok["price"]).abs() / ok["price"] <= HIT_TOLERANCE
    inside = (ok["price"] >= ok["price_low"]) & (ok["price"] <= ok["price_high"])
    errors = error_summary(ok["price"], ok["price_est"])
    return {
        "건수": len(result), "실패": len(result) - len(ok),
        "중앙값오차율_%": errors["중앙값오차율_%"], "MAPE_%": errors["MAPE_%"], "20%이내_%": errors["20%이내_%"],
        "말한신뢰도": round(float(ok["confidence"].mean()), 2), "실제적중률": round(float(hit.mean()), 2),
        "가격구간포함": round(float(inside.mean()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--sample", type=int, help="test에서 이만큼만 뽑아 잰다 (고정된 난수)")
    parser.add_argument("--online", action="store_true", help="환경변수의 키로 건축물대장·Kakao를 조회한다")
    parser.add_argument("--save", type=Path, default=SAVE_PATH)
    args = parser.parse_args()

    trades = load_trades(args.db)
    con = sqlite3.connect(args.db)
    try:
        geocodes = pd.read_sql("SELECT * FROM geocode", con)
    finally:
        con.close()
    test = trades[trades["is_holdout"] & (trades["split"] == "test")]
    if args.sample:
        test = test.sample(min(args.sample, len(test)), random_state=SAMPLE_SEED)
    data_key = os.environ.get("DATA_GO_KR_KEY", "") if args.online else ""
    kakao_key = os.environ.get("KAKAO_REST_KEY", "") if args.online else ""
    print(f"test {len(test):,}건 / 조회: {'건축물대장·Kakao 사용' if args.online else '키 없이 DB만'}")

    confidence_model = ConfidenceModel.load(MODEL_PATH)
    results, rows = [], []
    for name, with_area in CASES.items():
        result = run_case(name, test, with_area, Resolver(trades, geocodes, data_key, kakao_key), confidence_model)
        results.append(result.assign(조건=name))
        rows.append({"조건": name, **summarize(result)})
        for region, group in result.groupby("region"):
            rows.append({"조건": f"  {region}", **summarize(group)})
    print(pd.DataFrame(rows).to_string(index=False))

    args.save.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(results)[["조건", "trade_id", "region", "price", "price_est", "price_low", "price_high", "confidence", "status"]] \
        .to_csv(args.save, index=False, encoding="utf-8-sig")
    print(f"저장: {args.save}")


if __name__ == "__main__":
    main()
