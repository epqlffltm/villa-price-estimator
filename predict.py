# predict.py
"""입력 CSV의 물건마다 매매 시세를 추정해 출력 CSV로 쓴다.

    python predict.py --input input.csv --output output.csv

입력 컬럼: id, sigungu, dong, jibun, floor, ho(없을 수 있음), area_m2(비어 있을 수 있음)
출력 컬럼: id, price_est, price_low, price_high, confidence, basis, status

수집해 둔 data/trades.db를 쓰고, DB에 없는 값만 실행 중에 조회한다. 조회용 키는 환경변수로 받는다.
  DATA_GO_KR_KEY  건축물대장(전용면적·준공년도). 없으면 실거래 기록으로 어림한다.
  KAKAO_REST_KEY  수집되지 않은 지번의 좌표. 없으면 같은 건물 거래만 반영한다.
어떤 줄에서 문제가 생겨도 멈추지 않고, 그 줄의 status에 fail과 사유를 적는다.
"""
import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model.confidence import MODEL_PATH, ConfidenceModel
from model.estimate import NEARBY_RADIUS_M, estimate, fit_region, same_unit_as_target
from model.holdout import DB_PATH, load_trades
from model.resolve import InputError, Resolved, Resolver

OUTPUT_COLUMNS = ["id", "price_est", "price_low", "price_high", "confidence", "basis", "status"]
PRICE_UNIT = 100_000  # 10만 원 단위로 반올림


def read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"입력 파일이 없습니다: {path}")
    for encoding in ("utf-8-sig", "cp949"):
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        sys.exit(f"{path} 를 읽을 수 없습니다. UTF-8 또는 CP949로 저장해 주세요.")
    frame.columns = [column.strip() for column in frame.columns]
    missing = [column for column in ["id", "sigungu", "dong", "jibun", "floor"] if column not in frame.columns]
    if missing:
        sys.exit(f"입력에 필요한 컬럼이 없습니다: {missing}")
    return frame


def round_price(value: float) -> int:
    return int(round(value / PRICE_UNIT) * PRICE_UNIT)


def basis_text(target: dict, result: dict, notes: list[str]) -> str:
    """산출 근거 한 줄."""
    if result["n_same_building"] or result["n_nearby"]:
        evidence = (f"같은 건물 {result['n_same_building']}건·반경 {NEARBY_RADIUS_M:.0f}m {result['n_nearby']}건 실거래 반영"
                    f"(가격식 대비 {result['adjustment_pct']:+.0f}%)")
    else:
        evidence = "주변 실거래가 없어 가격식만 적용"
    floor = f"지하 {abs(target['floor'])}층" if target["floor"] < 0 else f"{target['floor']}층"
    subject = f"{target['build_year']}년 준공 {floor} {target['area_m2']:g}㎡ {target['house_type']}"
    return " / ".join([evidence, subject, *notes])


def predict(rows: list[dict], resolver: Resolver, confidence_model: ConfidenceModel) -> list[dict]:
    """입력 행 목록을 받아 같은 순서의 출력 행 목록을 돌려준다."""
    outputs = [{"id": row.get("id", ""), "price_est": "", "price_low": "", "price_high": "",
                "confidence": 0.0, "basis": "", "status": ""} for row in rows]
    resolved: dict[int, Resolved] = {}
    for index, row in enumerate(rows):
        try:
            resolved[index] = resolver.resolve(row)
        except InputError as e:
            outputs[index]["status"] = f"fail: {e}"
        except Exception as e:  # 예상하지 못한 문제도 그 줄만 실패로 남긴다
            outputs[index]["status"] = f"fail: 처리 중 오류({type(e).__name__}: {e})"

    for region, region_trades in resolver.trades.items():
        indexes = [index for index, item in resolved.items() if item.region == region]
        if not indexes:
            continue
        # 추정 대상과 같은 호로 보이는 거래는 가격식 학습에서도 뺀다.
        exclude = np.zeros(len(region_trades), dtype=bool)
        for index in indexes:
            exclude |= same_unit_as_target(region_trades, resolved[index].target)
        model = fit_region(region_trades, exclude)
        for index in indexes:
            item = resolved[index]
            try:
                result = estimate(model, item.target)
                rated = confidence_model.assess(result["price"], item.target["floor"], item.target["build_year"],
                                                result["weight_sum"], result["spread"], item.penalty)
                outputs[index].update({
                    "price_est": round_price(result["price"]),
                    "price_low": round_price(rated["price_low"]),
                    "price_high": round_price(rated["price_high"]),
                    "confidence": rated["confidence"],
                    "basis": basis_text(item.target, result, item.notes),
                    "status": "ok",
                })
            except Exception as e:
                outputs[index]["status"] = f"fail: 추정 중 오류({type(e).__name__}: {e})"
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    started = time.time()

    rows = read_input(args.input).to_dict("records")
    trades = load_trades(args.db)
    con = sqlite3.connect(args.db)
    try:
        geocodes = pd.read_sql("SELECT * FROM geocode", con)
    finally:
        con.close()

    data_key = os.environ.get("DATA_GO_KR_KEY", "")
    kakao_key = os.environ.get("KAKAO_REST_KEY", "")
    print(f"건축물대장 조회: {'사용' if data_key.strip() else '키 없음(실거래 기록으로 대체)'}"
          f" / 좌표 조회: {'사용' if kakao_key.strip() else '키 없음(수집된 좌표만 사용)'}")

    outputs = predict(rows, Resolver(trades, geocodes, data_key, kakao_key), ConfidenceModel.load(MODEL_PATH))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs, columns=OUTPUT_COLUMNS).to_csv(args.output, index=False, encoding="utf-8-sig")

    ok = sum(1 for output in outputs if output["status"] == "ok")
    print(f"{len(outputs)}건 중 {ok}건 산출, {len(outputs) - ok}건 실패 / {time.time() - started:.1f}초 / 저장: {args.output}")


if __name__ == "__main__":
    main()
