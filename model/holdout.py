# model/holdout.py
"""검증의 공통 규칙: 어떤 거래를 맞혀 볼지, 무엇을 비교 대상에서 빼는지, 오차를 어떻게 재는지.

회사의 블라인드 검증은 "최근 12개월 실거래가 있는 물건"을 뽑아 실제 거래가와 비교한다.
같은 조건을 흉내 내기 위해 다음 규칙을 모든 모델에 똑같이 적용한다.
  - 맞혀 볼 대상(홀드아웃): 산출 기준일로부터 12개월 안의 거래
  - 대상과 같은 호로 보이는 거래(같은 지번·층·면적)는 계산에서 뺀다. 정답을 보고 답하는 것을 막는다.
  - 지번을 둘로 나눠 dev는 설정값을 고르는 데, test는 마지막 성적을 내는 데만 쓴다.
"""
import sqlite3
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("data/trades.db")
REF_DATE = "2026-10-06"  # 산출 기준일 = 과제 안내일
HOLDOUT_MONTHS = 12
SAME_UNIT_AREA_TOLERANCE = 0.5  # ㎡. 실거래가에는 호가 없어서 지번·층·면적이 같으면 같은 호로 본다.

QUERY = """
SELECT t.*, g.lat, g.lon
FROM trades t
LEFT JOIN geocode g
  ON g.sigungu = t.sigungu AND g.dong = t.dong AND g.jibun = t.jibun AND g.status = 'ok'
ORDER BY t.trade_id
"""


def load_trades(db_path: Path = DB_PATH, ref_date: str = REF_DATE) -> pd.DataFrame:
    """정제된 거래에 좌표를 붙이고, 검증에 쓰는 컬럼을 더해 돌려준다."""
    con = sqlite3.connect(db_path)
    try:
        trades = pd.read_sql(QUERY, con)
    finally:
        con.close()
    return add_holdout_columns(trades, ref_date)


def add_holdout_columns(trades: pd.DataFrame, ref_date: str = REF_DATE) -> pd.DataFrame:
    trades = trades.copy()
    # 권역: 강서구는 화곡동만 남겼으므로 "화곡동", 나머지는 구 이름
    trades["region"] = np.where(trades["sgg_cd"] == "11500", trades["dong"], trades["sigungu"].str.split().str[-1])
    days = (pd.Timestamp(ref_date) - pd.to_datetime(trades["deal_date"])).dt.days
    trades["months_ago"] = days / 30.4375  # 기준일로부터 몇 개월 전 거래인지
    trades["lot"] = trades["sgg_cd"] + " " + trades["dong"] + " " + trades["jibun"]
    trades["split"] = trades["lot"].map(split_of)
    trades["is_holdout"] = (trades["months_ago"] >= 0) & (trades["months_ago"] <= HOLDOUT_MONTHS)
    return trades


def split_of(lot: str) -> str:
    """지번 이름으로 dev/test를 정한다. 같은 건물의 거래는 항상 같은 쪽에 들어가고, 실행할 때마다 결과가 같다."""
    return "dev" if zlib.crc32(lot.encode("utf-8")) % 2 == 0 else "test"


def same_unit(region_trades: pd.DataFrame, position: int) -> np.ndarray:
    """position번째 거래와 같은 호로 보이는 거래(자기 자신 포함)를 True로 표시한다."""
    lot = region_trades["lot"].to_numpy()
    floor = region_trades["floor"].to_numpy()
    area = region_trades["area_m2"].to_numpy()
    return (
        (lot == lot[position])
        & (floor == floor[position])
        & (np.abs(area - area[position]) < SAME_UNIT_AREA_TOLERANCE)
    )


def error_summary(actual, predicted) -> dict:
    """실제 거래가 대비 오차를 요약한다. 오차율 = |추정 - 실제| / 실제."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    signed = (predicted - actual) / actual
    absolute = np.abs(signed)
    return {
        "건수": len(actual),
        "중앙값오차율_%": round(float(np.median(absolute)) * 100, 1),
        "MAPE_%": round(float(absolute.mean()) * 100, 1),
        "10%이내_%": round(float((absolute <= 0.10).mean()) * 100, 1),
        "20%이내_%": round(float((absolute <= 0.20).mean()) * 100, 1),
        "치우침_%": round(float(np.median(signed)) * 100, 1),  # +면 비싸게, -면 싸게 추정하는 경향
    }
