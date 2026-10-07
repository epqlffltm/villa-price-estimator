# validate_predict.py
"""제출 프로그램(predict.py)을 그대로 돌려 오차를 잰다. 입력 조건을 바꿔 가며 따로 잰다.

validate.py 와의 차이
  validate.py          실거래의 면적·준공년도를 그대로 쓰고, 가격식은 묶음 단위로 정답을 빼고 만든다 (설정값을 고를 때 쓴 방식)
  validate_predict.py  입력 한 줄(시군구·동·지번·층)을 만들어 predict()에 넣는다. 값 채우기, 물건별 가격식, 신뢰도 감점까지
                       채점 때와 같은 경로를 탄다.

대상은 test 홀드아웃(최근 12개월 거래 중 설정에 쓰지 않은 절반)이다.

기본 (키 없이 수집해 둔 DB만 쓴다)
  면적 입력               입력에 면적이 있는 경우
  면적 없음               면적을 비운 경우. 같은 건물 실거래 기록에서 면적을 찾는다.
                          이때 대상 물건 자신의 거래 기록도 DB에 있으므로, 그 기록에서 면적을 되찾는 경우가 많다.
  면적 없음·기록 없는 물건  대상 거래를 DB에서 지우고 잰다. 수집일 뒤에 신고된 거래처럼, DB에 기록이 없는 물건의 성능이다.

--online (환경변수의 키로 건축물대장·Kakao를 조회한다. 호출 수가 많으므로 일부만 뽑아 잰다)
  면적 없음·호 없음        건축물대장의 같은 층 호들에서 면적을 고른다.
  면적 없음·호 입력        실거래가에는 호가 없어서, 대장에서 같은 층·같은 면적의 호를 찾아 그 이름을 입력으로 넣는다.
                          호 이름으로 조회했을 때 그 면적을 되찾는지를 함께 확인한다.
                          한 지번에 동이 여럿이면 같은 호 이름이 겹치므로, 호만 넣은 경우와 동을 함께 적은 경우를 따로 센다.

실행 (2~3분. 100건마다 진행 상황이 출력된다):
    uv run python validate_predict.py
    uv run --env-file .env python validate_predict.py --online --sample 50
"""
import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd

from model import building
from model.confidence import HIT_TOLERANCE, MODEL_PATH, ConfidenceModel
from model.holdout import DB_PATH, SAME_UNIT_AREA_TOLERANCE, error_summary, load_trades
from model.resolve import Resolver
from predict import predict

SAVE_PATH = Path("reports/predict_check.csv")
ONLINE_SAVE_PATH = Path("reports/predict_check_online.csv")  # 전체 검증 결과를 덮어쓰지 않도록 따로 둔다
ONLINE_SAMPLE = 50
SAMPLE_SEED = 2026
PROGRESS_EVERY = 100


def make_row(trade: dict, with_area: bool, ho: str = "") -> dict:
    """실거래 한 건을 입력 CSV의 한 줄로 바꾼다. 가격과 거래일은 넘기지 않는다."""
    return {"id": str(trade["trade_id"]), "sigungu": trade["sigungu"], "dong": trade["dong"], "jibun": trade["jibun"],
            "floor": str(trade["floor"]), "ho": ho, "area_m2": f"{trade['area_m2']}" if with_area else ""}


def predict_without_own_trade(row: dict, region: str, resolver: Resolver, confidence_model: ConfidenceModel) -> dict:
    """대상 거래를 DB에서 지운 상태로 한 건을 추정한다. 끝나면 DB를 되돌린다."""
    full = resolver.trades[region]
    resolver.trades[region] = full[full["trade_id"] != int(row["id"])].reset_index(drop=True)
    try:
        return predict([row], resolver, confidence_model)[0]
    finally:
        resolver.trades[region] = full


def run_case(name: str, test: pd.DataFrame, rows: list[dict], resolver: Resolver, confidence_model: ConfidenceModel,
             without_own_trade: bool = False) -> pd.DataFrame:
    """rows를 predict()에 넣고, 실제 거래가와 나란히 놓은 표를 돌려준다. test와 rows는 같은 순서다."""
    regions = list(test["region"])
    outputs = []
    # 물건은 서로 영향을 주지 않으므로 나눠서 넣어도 결과가 같다. 진행 상황을 보여 주려고 나눈다.
    for start in range(0, len(rows), PROGRESS_EVERY):
        chunk = rows[start:start + PROGRESS_EVERY]
        if without_own_trade:
            outputs += [predict_without_own_trade(row, regions[start + k], resolver, confidence_model)
                        for k, row in enumerate(chunk)]
        else:
            outputs += predict(chunk, resolver, confidence_model)
        print(f"\r{name}: {len(outputs):,} / {len(rows):,}건", end="", flush=True)
    print()
    outputs = pd.DataFrame(outputs)
    result = test[["trade_id", "region", "price"]].reset_index(drop=True)
    for column in ["price_est", "price_low", "price_high"]:
        result[column] = pd.to_numeric(outputs[column], errors="coerce")  # 실패한 줄은 빈칸 -> NaN
    result["confidence"], result["status"] = outputs["confidence"], outputs["status"]
    return result.assign(조건=name)


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


def register_units(resolver: Resolver, trade: dict) -> list[dict]:
    """이 거래가 있는 지번의 건축물대장 전유공용면적 목록. 좌표 표에 법정동코드가 없으면 빈 목록."""
    codes = resolver.geocodes
    known = codes[(codes["sigungu"] == trade["sigungu"]) & (codes["dong"] == trade["dong"]) & (codes["jibun"] == trade["jibun"])]
    if known.empty:
        return []
    return resolver.fetch_register("getBrExposPubuseAreaInfo", str(known.iloc[0]["b_code"]),
                                   int(trade["bonbun"]), int(trade["bubun"]), bool(trade["is_san"]))


def floor_units_text(items: list[dict], floor: int) -> str:
    units = building.floor_units(items, floor)
    return ", ".join(f"{dong + ' ' if dong else ''}{ho} {area:g}㎡" for (dong, ho), area in sorted(units.items()))


def offline_cases(test: pd.DataFrame, resolver: Resolver, confidence_model: ConfidenceModel) -> list[pd.DataFrame]:
    trades = test.to_dict("records")
    with_area = [make_row(trade, with_area=True) for trade in trades]
    no_area = [make_row(trade, with_area=False) for trade in trades]
    return [
        run_case("면적 입력", test, with_area, resolver, confidence_model),
        run_case("면적 없음", test, no_area, resolver, confidence_model),
        run_case("면적 없음·기록 없는 물건", test, no_area, resolver, confidence_model, without_own_trade=True),
    ]


def online_cases(test: pd.DataFrame, resolver: Resolver, confidence_model: ConfidenceModel) -> list[pd.DataFrame]:
    trades = test.to_dict("records")
    results = [run_case("면적 없음·호 없음", test, [make_row(trade, with_area=False) for trade in trades],
                        resolver, confidence_model)]

    # 대장에서 같은 층·같은 면적의 호를 찾아 "이 거래의 호"로 삼는다. (위 조회 결과가 캐시에 있어 다시 호출하지 않는다)
    registered = [bool(register_units(resolver, trade)) for trade in trades]
    units = [building.find_unit_by_area(register_units(resolver, trade), int(trade["floor"]), float(trade["area_m2"]),
                                        SAME_UNIT_AREA_TOLERANCE) for trade in trades]
    found = [k for k, unit in enumerate(units) if unit]
    print(f"건축물대장을 조회한 거래: {sum(registered)} / {len(trades)}건")
    print(f"그중 같은 층에 실거래와 면적이 같은 호가 있는 거래: {len(found)}건")
    if not found:
        return results

    # 호 이름만 넣은 입력과, 동을 함께 적은 입력("B동 201")을 따로 확인한다.
    rows = [make_row(trades[k], with_area=False, ho=units[k][1]) for k in found]
    with_dong = [make_row(trades[k], with_area=False, ho=" ".join(part for part in units[k] if part)) for k in found]
    for label, candidates in [("호 이름만 넣어", rows), ("동과 호를 함께 넣어", with_dong)]:
        mismatches = []
        for k, row in zip(found, candidates):
            trade = trades[k]
            try:
                resolved = resolver.resolve(row)
                area, how = resolved.target["area_m2"], " / ".join(resolved.notes)
            except Exception as e:
                area, how = float("nan"), f"오류: {e}"
            if not abs(area - float(trade["area_m2"])) < SAME_UNIT_AREA_TOLERANCE:
                mismatches.append((trade, row["ho"], area, how))
        print(f"{label} 같은 면적을 되찾은 거래: {len(found) - len(mismatches)} / {len(found)}건")
        for trade, ho, area, how in mismatches:
            # 왜 다른 면적이 나왔는지 볼 수 있게, 대장에 적힌 그 층의 호를 모두 보여 준다.
            floor_text = floor_units_text(register_units(resolver, trade), int(trade["floor"]))
            print(f"  되찾지 못함: {trade['dong']} {trade['jibun']} {trade['floor']}층, 실거래 {trade['area_m2']}㎡, 입력한 호 '{ho}' -> {area}㎡ ({how})")
            print(f"    대장의 {trade['floor']}층: {floor_text}")
    results.append(run_case("면적 없음·호 입력", test.iloc[found], rows, resolver, confidence_model))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--sample", type=int, help="test에서 이만큼만 뽑아 잰다 (고정된 난수)")
    parser.add_argument("--online", action="store_true", help="환경변수의 키로 건축물대장·Kakao를 조회한다")
    parser.add_argument("--save", type=Path, help=f"결과 CSV 경로. 기본은 {SAVE_PATH}, --online이면 {ONLINE_SAVE_PATH}")
    args = parser.parse_args()

    trades = load_trades(args.db)
    con = sqlite3.connect(args.db)
    try:
        geocodes = pd.read_sql("SELECT * FROM geocode", con)
    finally:
        con.close()
    test = trades[trades["is_holdout"] & (trades["split"] == "test")]
    sample = args.sample or (ONLINE_SAMPLE if args.online else None)
    if sample:
        test = test.sample(min(sample, len(test)), random_state=SAMPLE_SEED)
    confidence_model = ConfidenceModel.load(MODEL_PATH)

    if args.online:
        data_key, kakao_key = os.environ.get("DATA_GO_KR_KEY", ""), os.environ.get("KAKAO_REST_KEY", "")
        if not data_key.strip():
            raise SystemExit("--online 에는 환경변수 DATA_GO_KR_KEY가 필요합니다.")
        print(f"test {len(test):,}건 / 건축물대장·Kakao 조회 사용")
        results = online_cases(test, Resolver(trades, geocodes, data_key, kakao_key), confidence_model)
    else:
        print(f"test {len(test):,}건 / 키 없이 DB만")
        results = offline_cases(test, Resolver(trades, geocodes), confidence_model)

    rows = []
    for result in results:
        rows.append({"조건": result["조건"].iloc[0], **summarize(result)})
        if not args.online:
            rows += [{"조건": f"  {region}", **summarize(group)} for region, group in result.groupby("region")]
    print(pd.DataFrame(rows).to_string(index=False))

    save = args.save or (ONLINE_SAVE_PATH if args.online else SAVE_PATH)
    save.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(results)[["조건", "trade_id", "region", "price", "price_est", "price_low", "price_high", "confidence", "status"]] \
        .to_csv(save, index=False, encoding="utf-8-sig")
    print(f"저장: {save}")


if __name__ == "__main__":
    main()
