# report.py
"""자체 검증 리포트(reports/validation.md)를 만든다. PPT의 데이터·검증·예시 장에 옮길 표를 한 파일에 모은다.

검증 결과(reports/holdout.csv)와 신뢰도 식(model/confidence.json)을 읽어 표로 정리하고,
권역마다 한 건씩 고른 예시 물건을 predict.py와 같은 경로로 추정해 입력·출력·근거를 싣는다.
문장은 계산된 숫자만 옮긴다. 해석은 싣지 않는다.

먼저 검증과 신뢰도 학습을 돌린 뒤 실행한다:
    uv run python validate.py --save reports/holdout.csv
    uv run python -m model.confidence
    uv run python report.py
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from model.confidence import (BAND_NAMES, HOLDOUT_PATH, INTERVAL_COVERAGE, MODEL_PATH, PRICE_COLUMN, ConfidenceModel,
                              reliability_table)
from model.estimate import (DISTANCE_SCALE_M, PRIOR_WEIGHT, SAME_BUILDING_BOOST, TIME_SCALE_MONTHS)
from model.formula import fit_formula
from model.holdout import DB_PATH, HOLDOUT_MONTHS, REF_DATE, error_summary, load_trades
from model.resolve import Resolver
from predict import predict

OUT_PATH = Path("reports/validation.md")
MODELS = ["기준선", "가격식", PRICE_COLUMN]
LARGE_ERROR = 0.40      # 오차율이 이보다 크면 "크게 틀린 사례"로 본다
EXAMPLE_SEED = 2026     # 예시 물건을 고르는 난수 씨앗. 결과를 보고 고르지 않기 위해 고정한다


def markdown_table(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _cell(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else f"{value:g}"
    return str(value)


def won(value: float) -> str:
    """285000000 -> "2억 8,500만"."""
    man = int(round(value / 10_000))
    eok, rest = divmod(man, 10_000)
    if eok and rest:
        return f"{eok}억 {rest:,}만"
    return f"{eok}억" if eok else f"{rest:,}만"


def summary_rows(frame: pd.DataFrame, label: pd.Series, model: str = PRICE_COLUMN) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(label, observed=True):  # 구간으로 나눈 값은 구간 순서대로 나온다
        rows.append({label.name: key, **error_summary(group["price"], group[model])})
    return pd.DataFrame(rows)


def section_data(trades: pd.DataFrame, con: sqlite3.Connection) -> str:
    steps = pd.read_sql("SELECT step AS 단계, removed AS 제거, remaining AS 남음 FROM clean_summary", con)
    geocode = pd.read_sql("SELECT status AS 상태, COUNT(*) AS 지번수 FROM geocode GROUP BY status", con)
    by_region = trades.groupby("region").agg(
        거래건수=("price", "size"), 지번수=("lot", "nunique"),
        최근12개월=("is_holdout", "sum"),
        직거래_pct=("is_direct", lambda s: round(s.mean() * 100, 1)),
        per_m2=("price_per_m2", lambda s: round(s.median() / 10_000)),
    ).reset_index().rename(columns={"region": "권역", "직거래_pct": "직거래_%", "per_m2": "㎡당 중앙값_만원"})
    types = trades["house_type"].value_counts().rename_axis("주택 유형").reset_index(name="건수")
    return "\n\n".join([
        "## 1. 데이터",
        f"국토교통부 연립다세대 매매 실거래가. 계약일 {trades['deal_date'].min()} ~ {trades['deal_date'].max()}, "
        f"산출 기준일 {REF_DATE}.",
        "### 정제 단계별 건수", markdown_table(steps),
        "### 권역별", markdown_table(by_region),
        "### 주택 유형 (연립 포함)", markdown_table(types),
        "### 지번 좌표 변환 (Kakao 주소 검색)", markdown_table(geocode),
    ])


def section_models(holdout: pd.DataFrame) -> str:
    test = holdout[holdout["split"] == "test"]
    rows = []
    for split_name, part in [("dev", holdout[holdout["split"] == "dev"]), ("test", test)]:
        for region, group in part.groupby("region"):
            row = {"구분": split_name, "권역": region, "건수": len(group)}
            for model in MODELS:
                summary = error_summary(group["price"], group[model])
                row[f"{model} 중앙값오차율_%"] = summary["중앙값오차율_%"]
                row[f"{model} 20%이내_%"] = summary["20%이내_%"]
            rows.append(row)
    overall = pd.DataFrame([{"모델": model, **error_summary(test["price"], test[model])} for model in MODELS])
    return "\n\n".join([
        "## 2. 모델 비교",
        f"홀드아웃: 기준일로부터 {HOLDOUT_MONTHS}개월 안의 거래 {len(holdout):,}건. 지번 기준으로 dev {int((holdout['split'] == 'dev').sum()):,}건과 "
        f"test {len(test):,}건으로 나눴다. 설정값은 dev에서만 골랐고 test는 성적 확인에만 썼다. "
        "대상과 같은 지번·층·면적의 거래는 계산에서 뺐다.",
        f"설정값: 거리 {DISTANCE_SCALE_M:g}m, 기간 {TIME_SCALE_MONTHS:g}개월, 같은 건물 {SAME_BUILDING_BOOST:g}배, 사전 가중 {PRIOR_WEIGHT:g}.",
        "### test 전체", markdown_table(overall),
        "### 권역 × dev/test", markdown_table(pd.DataFrame(rows)),
    ])


def section_breakdown(test: pd.DataFrame) -> str:
    months_ago = (pd.Timestamp(REF_DATE) - pd.to_datetime(test["deal_date"])).dt.days / 30.4375
    labels = [
        test["region"].rename("권역"),
        test["is_direct"].map({0: "중개거래", 1: "직거래"}).rename("거래 방식"),
        test["house_type"].rename("주택 유형"),
        pd.cut(test["floor"], [-9, 0, 1, 4, 99], labels=["지하", "1층", "2~4층", "5층 이상"]).rename("층"),
        pd.cut(int(REF_DATE[:4]) - test["build_year"], [-1, 1, 9, 19, 29, 999],
               labels=["0~1년", "2~9년", "10~19년", "20~29년", "30년 이상"]).rename("연식"),
        pd.cut(test["n_same_building"], [-1, 0, 2, 5, 9999], labels=["0건", "1~2건", "3~5건", "6건 이상"]).rename("같은 건물 사례"),
        pd.cut(months_ago, [-1, 3, 6, 9, 99], labels=["0~3개월 전", "3~6개월 전", "6~9개월 전", "9~12개월 전"]).rename("거래 시점"),
    ]
    parts = ["## 3. 조건별 오차 (test)",
             "치우침은 (추정 - 실제) / 실제의 중앙값이다. 추정은 기준일 시세이고 실제는 과거 거래가라서, "
             "거래 시점별 치우침은 그사이 시세 변화를 포함한다."]
    for label in labels:
        parts += [f"### {label.name}", markdown_table(summary_rows(test, label))]
    return "\n\n".join(parts)


def section_confidence(holdout: pd.DataFrame, model: ConfidenceModel) -> str:
    test = holdout[holdout["split"] == "test"]
    coefficients = pd.DataFrame([{"조건": "기준값", "계수": model.intercept},
                                 *[{"조건": name, "계수": value} for name, value in model.coefficients.items()]])
    widths = pd.DataFrame({"신뢰도 구간": BAND_NAMES,
                           "가격 구간 폭_±%": [round((np.exp(w) - 1) * 100) for w in model.half_widths]})
    return "\n\n".join([
        "## 4. 신뢰도와 가격 구간",
        "신뢰도 = 추정가가 실제 거래가의 ±20% 안에 들 확률. dev 결과로 로지스틱 회귀를 학습했다. "
        f"가격 구간은 신뢰도 구간별로 dev에서 실제 거래가의 {INTERVAL_COVERAGE:.0%}가 들어온 폭이다.",
        "### 신뢰도 식의 계수 (음수면 신뢰도를 낮춘다)", markdown_table(coefficients),
        "### 신뢰도 구간별 가격 구간 폭", markdown_table(widths),
        "### 말한 신뢰도와 실제 (test)", markdown_table(reliability_table(test, model)),
    ])


def section_large_errors(test: pd.DataFrame) -> str:
    test = test.copy()
    test["오차율"] = (test[PRICE_COLUMN] - test["price"]) / test["price"]
    large = test[test["오차율"].abs() > LARGE_ERROR]
    age = int(REF_DATE[:4]) - test["build_year"]
    flags = {
        "직거래": test["is_direct"] == 1,
        "지하층": test["floor"] < 0,
        "준공 30년 이상": age >= 30,
        "같은 건물 사례 0건": test["n_same_building"] == 0,
        "거래가 1.5억 미만": test["price"] < 150_000_000,
        "실제보다 높게 추정": test["오차율"] > 0,
    }
    any_flag = flags["직거래"] | flags["지하층"] | flags["준공 30년 이상"] | flags["같은 건물 사례 0건"]
    shares = pd.DataFrame([
        {"조건": name, "크게 틀린 사례 중_%": round(flag[large.index].mean() * 100, 1), "test 전체 중_%": round(flag.mean() * 100, 1)}
        for name, flag in {**flags, "직거래·지하·노후·사례 0건 중 하나 이상": any_flag}.items()
    ])
    top = large.reindex(large["오차율"].abs().sort_values(ascending=False).index).head(10)
    top_table = pd.DataFrame({
        "권역": top["region"], "주소": top["dong"] + " " + top["jibun"], "층": top["floor"], "면적_㎡": top["area_m2"],
        "준공": top["build_year"], "거래": top["is_direct"].map({0: "중개", 1: "직거래"}),
        "실제": top["price"].map(won), "추정": top[PRICE_COLUMN].map(won),
        "오차율_%": (top["오차율"] * 100).round(0).astype(int), "같은 건물 사례": top["n_same_building"],
    })
    return "\n\n".join([
        "## 5. 크게 틀린 사례",
        f"test {len(test):,}건 중 오차율이 {LARGE_ERROR:.0%}를 넘은 것은 {len(large):,}건({len(large) / len(test):.1%})이다.",
        "### 어떤 조건에 몰려 있는가", markdown_table(shares),
        "### 오차가 가장 큰 10건", markdown_table(top_table),
    ])


def section_examples(trades: pd.DataFrame, holdout: pd.DataFrame, con: sqlite3.Connection, model: ConfidenceModel) -> str:
    """권역마다 test 홀드아웃에서 한 건씩 고정된 난수로 골라, 제출 프로그램과 같은 경로로 추정한다."""
    test = holdout[holdout["split"] == "test"]
    picked = pd.concat([group.sample(1, random_state=EXAMPLE_SEED) for _, group in test.groupby("region")])
    picked = trades[trades["trade_id"].isin(picked["trade_id"])].sort_values("region")
    rows = [{"id": str(n), "sigungu": r["sigungu"], "dong": r["dong"], "jibun": r["jibun"], "floor": str(r["floor"]),
             "ho": "", "area_m2": str(r["area_m2"])} for n, r in enumerate(picked.to_dict("records"), start=1)]
    geocodes = pd.read_sql("SELECT * FROM geocode", con)
    outputs = predict(rows, Resolver(trades, geocodes), model)  # 키 없이: 수집해 둔 데이터만 쓴다

    parts = ["## 6. 예시 산출 3건",
             f"권역마다 test 홀드아웃에서 한 건씩 난수(씨앗 {EXAMPLE_SEED})로 골랐다. 결과를 보고 고르지 않았다. "
             "predict.py와 같은 함수로 추정했고, 실제 거래가는 추정에 쓰지 않았다."]
    for row, output, actual in zip(rows, outputs, picked.to_dict("records")):
        error = (output["price_est"] - actual["price"]) / actual["price"] * 100
        table = pd.DataFrame([
            {"항목": "입력", "값": f"{row['sigungu']} {row['dong']} {row['jibun']}, {row['floor']}층, {row['area_m2']}㎡"},
            {"항목": "추정 시세", "값": f"{won(output['price_est'])} ({output['price_est']:,}원)"},
            {"항목": "가격 구간", "값": f"{won(output['price_low'])} ~ {won(output['price_high'])}"},
            {"항목": "신뢰도", "값": output["confidence"]},
            {"항목": "근거", "값": output["basis"]},
            {"항목": "실제 거래", "값": f"{won(actual['price'])} ({actual['deal_date']}, "
                                    f"{'직거래' if actual['is_direct'] else '중개거래'}) → 오차율 {error:+.1f}%"},
        ])
        parts += [f"### 예시 {row['id']}: {actual['region']}", markdown_table(table)]
    return "\n\n".join(parts)


def section_factors(trades: pd.DataFrame) -> str:
    table = {}
    for region, region_trades in trades.groupby("region"):
        effects = fit_formula(region_trades.reset_index(drop=True)).effects()
        table[region] = effects[[name for name in effects.index if not name.startswith("동 ")]]
    frame = pd.DataFrame(table).rename_axis("요인").reset_index()
    return "\n\n".join([
        "## 7. 요인별 효과 (%)",
        "가격식의 계수를 %로 바꾼 값. 기준은 준공 10~14년, 2~3층, 다세대, 중개거래다. "
        "면적(로그)는 면적이 약 2.7배가 될 때 ㎡당 가격이 변하는 비율이다.",
        markdown_table(frame),
    ])


def main() -> None:
    trades = load_trades(DB_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH, encoding="utf-8-sig")
    confidence_model = ConfidenceModel.load(MODEL_PATH)
    test = holdout[holdout["split"] == "test"]
    con = sqlite3.connect(DB_PATH)
    try:
        sections = [
            "# 자체 검증 리포트\n\n이 파일은 report.py가 만든다. 숫자를 고치려면 파일을 고치지 말고 스크립트를 다시 실행한다.",
            section_data(trades, con),
            section_models(holdout),
            section_breakdown(test),
            section_confidence(holdout, confidence_model),
            section_large_errors(test),
            section_examples(trades, holdout, con, confidence_model),
            section_factors(trades),
        ]
    finally:
        con.close()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"저장: {OUT_PATH} ({len(OUT_PATH.read_text(encoding='utf-8').splitlines())}줄)")


if __name__ == "__main__":
    main()
