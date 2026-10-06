# tests/test_report.py
"""리포트에 쓰는 표기 함수(금액, 표)를 확인한다."""
import numpy as np
import pandas as pd
import pytest

from report import markdown_table, summary_rows, won


@pytest.mark.parametrize("value, expected", [
    (285_000_000, "2억 8,500만"),
    (300_000_000, "3억"),
    (80_000_000, "8,000만"),
    (1_691_227_000, "16억 9,123만"),
])
def test_won(value, expected):
    assert won(value) == expected


def test_markdown_table():
    frame = pd.DataFrame({"권역": ["강남구"], "건수": [580], "오차율_%": [11.0], "빈값": [np.nan]})
    assert markdown_table(frame).splitlines() == [
        "| 권역 | 건수 | 오차율_% | 빈값 |",
        "|---|---|---|---|",
        "| 강남구 | 580 | 11 |  |",
    ]


def test_summary_rows_keep_interval_order():
    frame = pd.DataFrame({"price": [100.0] * 4, "가격식+실거래": [110.0, 100.0, 150.0, 90.0], "floor": [-1, 1, 3, 6]})
    label = pd.cut(frame["floor"], [-9, 0, 1, 4, 99], labels=["지하", "1층", "2~4층", "5층 이상"]).rename("층")
    table = summary_rows(frame, label)
    assert table["층"].tolist() == ["지하", "1층", "2~4층", "5층 이상"]
    assert table["중앙값오차율_%"].tolist() == [10.0, 0.0, 50.0, 10.0]
