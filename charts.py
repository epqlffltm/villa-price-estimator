# charts.py
"""reports/charts/data/ 의 CSV를 읽어 그래프를 그리고 reports/charts/ 에 PNG로 저장한다. (2단계: CSV -> 그래프)

이 파일은 CSV만 읽는다. DB나 모델을 건드리지 않으므로, CSV 숫자를 고치면 그래프도 그대로 바뀐다.

  01_monthly_price.png        월별 ㎡당 거래가
  02_monthly_count.png        월별 거래 건수
  03_error_bands.png          추정이 실제 거래가에서 얼마나 벗어났나
  04_model_comparison.png     모델을 단계별로 고칠 때마다 줄어든 오차
  05_confidence.png           모델이 말한 신뢰도와 실제로 맞힌 비율
  06_estimate_vs_actual.png   물건별 실제 거래가와 추정 시세

실행 (CSV가 먼저 있어야 한다):
    uv run python chart_data.py
    uv run python charts.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 그린다
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

OUT_DIR = Path("reports/charts")
DATA_DIR = OUT_DIR / "data"
REGIONS = ["강남구", "관악구", "화곡동"]
# 색은 색각 이상에서도 구분되는 조합으로 골랐다. 권역마다 항상 같은 색을 쓴다.
REGION_COLOR = {"강남구": "#2A6FBF", "관악구": "#2E9E8F", "화곡동": "#C4553F"}
BLUE, RED, GRAY = "#2A6FBF", "#C4553F", "#8A949C"
INK, MUTED, GRID = "#1F2933", "#4A5560", "#E3E6E8"
KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
HIT_LIMIT_PCT = 20  # 오차가 이 안이면 '맞혔다'고 본다
PRICE_AXIS_MAX = 12  # 산점도에 그리는 가격 범위 (억 원)


def setup_style() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlesize": 16, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 14, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.8, "legend.frameon": False, "font.size": 13, "legend.fontsize": 13,
    })


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, encoding="utf-8-sig")


def save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / name)
    plt.close(fig)
    print(f"저장: {OUT_DIR / name}")


def draw_monthly(table: pd.DataFrame, title: str, unit: str, name: str) -> None:
    """월별 표(행=월, 열=권역)를 권역마다 선 하나로 그린다. 선 끝에 권역 이름과 마지막 값을 적는다."""
    months = list(table["월"])
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for region in REGIONS:
        ax.plot(x, table[region], color=REGION_COLOR[region], linewidth=2.5)
    ax.set_ylim(0, None)
    # 선 끝에 이름표를 단다. 값이 비슷하면 글자가 겹치므로, 아래에서부터 최소 간격만큼 띄운다.
    last = {region: table[region].dropna().iloc[-1] for region in REGIONS}
    min_gap = ax.get_ylim()[1] * 0.07
    label_y = 0.0
    for region in sorted(REGIONS, key=last.get):
        label_y = max(last[region], label_y + min_gap)
        ax.annotate(f"{region} {last[region]:,.0f}{unit}", (x[-1], label_y), xytext=(8, 0), textcoords="offset points",
                    va="center", color=REGION_COLOR[region], fontweight="bold")
    ticks = [i for i, month in enumerate(months) if month.endswith("-01")]
    ax.set_xticks(ticks, [months[i][:4] + "년" for i in ticks])
    ax.set_xlim(0, len(months) + 9)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.set_title(title)
    save(fig, name)


def chart_error_bands() -> None:
    table = read("03_error_bands.csv")
    total = table["건수"].sum()
    within = table["오차구간"].isin(["10% 이내", "10~20%"])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(table["오차구간"], table["비율_%"], color=np.where(within, BLUE, GRAY), width=0.66)
    ax.bar_label(bars, labels=[f"{pct:.1f}%\n({count:,}건)" for pct, count in zip(table["비율_%"], table["건수"])], padding=4, color=INK)
    ax.set_ylim(0, table["비율_%"].max() * 1.28)
    ax.set_ylabel("물건 비율 (%)")
    ax.set_xlabel("추정가가 실제 거래가에서 벗어난 정도")
    ax.set_title(f"{total:,}건 중 {table.loc[within, '비율_%'].sum():.1f}%가 실제 거래가의 ±20% 안에 들었다")
    save(fig, "03_error_bands.png")


def chart_model_comparison() -> None:
    table = read("04_model_comparison.csv")
    models = [column for column in table.columns if column not in ("권역", "건수")]
    colors = [GRAY, BLUE, RED]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = 0.26
    x = np.arange(len(table))
    for k, (name, color) in enumerate(zip(models, colors)):
        bars = ax.bar(x + (k - 1) * (width + 0.02), table[name], width, color=color, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=3, color=INK, fontsize=12)
    ax.set_xticks(x, [f"{region}\n({count:,}건)" for region, count in zip(table["권역"], table["건수"])])
    ax.set_ylabel("중앙값 오차율 (%) · 낮을수록 정확")
    ax.set_ylim(0, table[models].to_numpy().max() * 1.3)
    ax.legend(loc="upper center", ncol=3)
    ax.set_title("모델을 고칠 때마다 오차가 줄었다")
    save(fig, "04_model_comparison.png")


def chart_confidence() -> None:
    table = read("05_confidence.csv")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(table))
    for offset, column, label, color in [(-0.2, "말한신뢰도", "모델이 말한 신뢰도", BLUE), (0.2, "실제적중률", "실제로 ±20% 안에 든 비율", RED)]:
        bars = ax.bar(x + offset, table[column], 0.38, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=3, color=INK, fontsize=12)
    ax.set_xticks(x, [f"{name}\n({count:,}건)" for name, count in zip(table["신뢰도구간"], table["건수"])])
    ax.set_xlabel("모델이 말한 신뢰도 구간")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper center", ncol=2)
    ax.set_title("신뢰도가 높다고 한 물건일수록 실제로 더 잘 맞았다")
    save(fig, "05_confidence.png")


def chart_estimate_vs_actual() -> None:
    table = read("06_estimate_vs_actual.csv")
    hit = table["오차율_%"].abs() <= HIT_LIMIT_PCT
    shown = (table["실제_억"] <= PRICE_AXIS_MAX) & (table["추정_억"] <= PRICE_AXIS_MAX)
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    line = np.array([0, PRICE_AXIS_MAX])
    ax.plot(line, line, color=INK, linewidth=1.2, label="대각선 = 추정과 실제가 같은 자리")
    for mask, label, color in [(hit, f"±{HIT_LIMIT_PCT}% 이내", BLUE), (~hit, f"±{HIT_LIMIT_PCT}% 밖", RED)]:
        part = table[mask & shown]
        ax.scatter(part["실제_억"], part["추정_억"], s=16, color=color, alpha=0.5, linewidths=0, label=f"{label} {int(mask.sum()):,}건")
    ax.text(PRICE_AXIS_MAX * 0.04, PRICE_AXIS_MAX * 0.76, "선보다 위\n= 실제보다 비싸게 추정", color=MUTED, va="top")
    ax.text(PRICE_AXIS_MAX * 0.96, PRICE_AXIS_MAX * 0.05, "선보다 아래\n= 실제보다 싸게 추정", color=MUTED, ha="right", va="bottom")
    ax.set_xlim(0, PRICE_AXIS_MAX)
    ax.set_ylim(0, PRICE_AXIS_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel(f"실제 거래가 (억 원) · {PRICE_AXIS_MAX}억 초과 {int((~shown).sum())}건은 그림에서 생략")
    ax.set_ylabel("추정 시세 (억 원)")
    ax.grid(True, axis="both")
    ax.legend(loc="upper left", markerscale=2)
    ax.set_title(f"점 하나가 물건 하나: 추정 시세와 실제 거래가 {len(table):,}건")
    save(fig, "06_estimate_vs_actual.png")


def main() -> None:
    setup_style()
    draw_monthly(read("01_monthly_price.csv"), "㎡당 거래가: 강남구는 관악구·화곡동의 2~3배", "만원", "01_monthly_price.png")
    draw_monthly(read("02_monthly_count.csv"), "월별 거래 건수: 2022년에 크게 줄어든 뒤 월 100~200건 안팎", "건", "02_monthly_count.png")
    chart_error_bands()
    chart_model_comparison()
    chart_confidence()
    chart_estimate_vs_actual()


if __name__ == "__main__":
    main()
