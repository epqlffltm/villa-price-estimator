# model/clean.py
"""수집한 원본 CSV(data/raw/trades/)를 정제해 data/trades.db의 trades 테이블로 만든다.

지우는 것은 세 가지뿐이다.
  1. 해제된 거래 (계약이 취소되어 실제 거래가 아님)
  2. 강서구 중 화곡동이 아닌 거래 (과제 권역 밖)
  3. 금액·면적·층·지번·계약일을 읽을 수 없는 행

비싸거나 싼 "실제 거래"는 여기서 지우지 않는다. 재개발 기대가 붙은 낡은 빌라나
가족 간 직거래처럼 가격이 튀는 거래도 시장에서 일어난 일이라, 지울지 말지는
모델 단계에서 오차를 재 보고 정한다. 대신 판단에 필요한 표시(직거래 여부, ㎡당 가격)를 붙여 둔다.

실행:
    uv run python -m model.clean
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from model.normalize import make_date, parse_float, parse_int, parse_price, split_jibun

RAW_DIR = Path("data/raw/trades")
DB_PATH = Path("data/trades.db")

SIGUNGU = {
    "11500": "서울특별시 강서구",
    "11680": "서울특별시 강남구",
    "11620": "서울특별시 관악구",
}
GANGSEO = "11500"
HWAGOK = "화곡동"

# 원본에서 읽어 쓰는 컬럼. 오래된 달에는 일부 컬럼의 값이 비어 있다(예: buyerGbn, slerGbn).
RAW_COLUMNS = [
    "sggCd", "umdNm", "jibun", "mhouseNm", "houseType", "floor", "excluUseAr", "landAr",
    "buildYear", "dealAmount", "dealYear", "dealMonth", "dealDay", "dealingGbn",
    "buyerGbn", "slerGbn", "cdealType", "cdealDay",
]


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """구·월별 CSV를 전부 읽어 하나로 합친다. 값은 모두 문자열 그대로 둔다."""
    frames = []
    for path in sorted(raw_dir.glob("*.csv")):
        if path.name.startswith("_"):  # _manifest.csv
            continue
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:  # 거래가 0건인 달은 헤더도 없는 빈 파일이다.
            continue
        frames.append(frame)
    if not frames:
        raise SystemExit(f"{raw_dir} 에 원본 CSV가 없습니다. 먼저 collect/fetch_trades.py를 실행하세요.")
    raw = pd.concat(frames, ignore_index=True)
    return raw.reindex(columns=RAW_COLUMNS, fill_value="").fillna("")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본을 정제해 (trades, 단계별 건수 요약)을 돌려준다."""
    steps = [("원본", 0, len(raw))]
    df = raw

    # 1. 해제된 거래: 해제여부(cdealType)가 "O"이거나 해제사유발생일(cdealDay)이 적혀 있다.
    cancelled = (df["cdealType"].str.strip() != "") | (df["cdealDay"].str.strip() != "")
    df = df[~cancelled]
    steps.append(("해제 거래", int(cancelled.sum()), len(df)))

    # 2. 권역 밖: 강서구는 구 전체를 받았으므로 화곡동만 남긴다.
    outside = (df["sggCd"] == GANGSEO) & (df["umdNm"] != HWAGOK)
    df = df[~outside]
    steps.append(("화곡동 외(강서구)", int(outside.sum()), len(df)))

    # 3. 문자열을 숫자·날짜로 바꾼다. 바꿀 수 없으면 None(빈 값)이 된다.
    jibun = df["jibun"].map(split_jibun)
    trades = pd.DataFrame({
        "sgg_cd": df["sggCd"],
        "sigungu": df["sggCd"].map(SIGUNGU),
        "dong": df["umdNm"],
        "jibun": df["jibun"].str.strip(),
        "is_san": jibun.map(lambda j: int(j[0]) if j else None),
        "bonbun": jibun.map(lambda j: j[1] if j else None),
        "bubun": jibun.map(lambda j: j[2] if j else None),
        "building_name": df["mhouseNm"],
        "house_type": df["houseType"],
        "floor": df["floor"].map(parse_int),
        "area_m2": df["excluUseAr"].map(parse_float),
        "land_m2": df["landAr"].map(parse_float),
        "build_year": df["buildYear"].map(parse_int),
        "price": df["dealAmount"].map(parse_price),
        "deal_date": [make_date(y, m, d) for y, m, d in zip(df["dealYear"], df["dealMonth"], df["dealDay"])],
        "is_direct": (df["dealingGbn"] == "직거래").astype(int),
        "seller_type": df["slerGbn"],
        "buyer_type": df["buyerGbn"],
    })

    # 시세 계산에 꼭 필요한 값이 없거나 말이 안 되는 행을 지운다. (0층은 없다. 지하는 -1)
    valid = (
        trades["bonbun"].notna()
        & trades["deal_date"].notna()
        & (trades["price"] > 0)
        & (trades["area_m2"] > 0)
        & trades["floor"].notna()
        & (trades["floor"] != 0)
    )
    trades = trades[valid].copy()
    steps.append(("필수값 오류", int((~valid).sum()), len(trades)))

    # 건축년도·대지권면적은 없어도 남긴다(빈 값으로 저장).
    for column in ["is_san", "bonbun", "bubun", "floor", "price", "build_year"]:
        trades[column] = trades[column].astype("Int64")
    trades["price_per_m2"] = (trades["price"] / trades["area_m2"]).round().astype("Int64")

    # 실행할 때마다 같은 순서·같은 trade_id가 나오도록 정렬한다.
    order = ["sgg_cd", "deal_date", "dong", "bonbun", "bubun", "floor", "area_m2", "price"]
    trades = trades.sort_values(order, kind="stable").reset_index(drop=True)
    trades.insert(0, "trade_id", trades.index + 1)

    summary = pd.DataFrame(steps, columns=["step", "removed", "remaining"])
    return trades, summary


def save(trades: pd.DataFrame, summary: pd.DataFrame, db_path: Path) -> None:
    """DB에 저장한다. 테이블을 통째로 바꾸므로 여러 번 실행해도 결과가 같다."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        trades.to_sql("trades", con, if_exists="replace", index=False)
        summary.to_sql("clean_summary", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_trades_lot ON trades (sgg_cd, dong, bonbun, bubun)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades (deal_date)")
        con.commit()
    finally:
        con.close()


def report(trades: pd.DataFrame, summary: pd.DataFrame) -> None:
    """PR·PPT에 옮겨 적을 숫자를 출력한다."""
    print("== 단계별 건수 ==")
    for step, removed, remaining in summary.itertuples(index=False):
        print(f"{step:<12} 제거 {removed:>7,}  남음 {remaining:>7,}")

    region = trades["dong"].where(trades["sgg_cd"] == GANGSEO, trades["sigungu"].str.split().str[-1])
    recent_from = (pd.Timestamp(trades["deal_date"].max()) - pd.DateOffset(months=12)).date().isoformat()
    recent = trades["deal_date"] >= recent_from

    print(f"\n== 권역별 (기간 {trades['deal_date'].min()} ~ {trades['deal_date'].max()}) ==")
    table = pd.DataFrame({
        "전체": region.value_counts(),
        "최근12개월": region[recent].value_counts(),
        "중앙값_만원/㎡": (trades["price_per_m2"].astype(float) / 10_000).groupby(region).median().round(0),
        "직거래_%": (trades["is_direct"].groupby(region).mean() * 100).round(1),
        "지하_%": ((trades["floor"] < 0).groupby(region).mean() * 100).round(1),
    })
    table["최근12개월"] = table["최근12개월"].fillna(0).astype(int)
    table.index.name = None
    print(table.to_string())

    print("\n== 주택 유형 ==")
    print(trades["house_type"].value_counts().to_string())

    same = trades.duplicated(["sgg_cd", "dong", "jibun", "floor", "area_m2", "price", "deal_date"], keep=False)
    print(f"\n같은 날·같은 지번·층·면적·금액인 행: {int(same.sum()):,}건 (같은 건물의 다른 호일 수 있어 지우지 않음)")
    print(f"건축년도 없음: {int(trades['build_year'].isna().sum()):,}건 / 대지권면적 없음: {int(trades['land_m2'].isna().sum()):,}건")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    trades, summary = clean(load_raw(args.raw_dir))
    save(trades, summary, args.db)
    report(trades, summary)
    print(f"\n저장: {args.db} (trades {len(trades):,}건)")


if __name__ == "__main__":
    main()
