# model/confidence.py
"""신뢰도와 가격 구간: 추정이 얼마나 믿을 만한지를 홀드아웃 결과로 정한다.

신뢰도 = "추정가가 실제 거래가의 ±20% 안에 들어올 확률".
  홀드아웃 dev 쪽에서 맞힌 물건과 틀린 물건을 보고, 어떤 조건에서 잘 틀리는지를 식으로 만든다(로지스틱 회귀).
  조건은 네 가지다: 근거가 된 거래의 양, 근거 거래끼리 흩어진 정도, 층(지하·1층), 연식(신축·노후).

가격 구간 = 신뢰도 구간별로, dev에서 실제 거래가의 80%가 들어온 폭.
  신뢰도가 낮을수록 구간이 넓어진다.

실행하면 dev로 식을 만들어 model/confidence.json에 저장하고, test에서 신뢰도가 실제와 맞는지 출력한다:
    uv run python -m model.confidence
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model.holdout import REF_DATE

MODEL_PATH = Path("model/confidence.json")
HOLDOUT_PATH = Path("reports/holdout.csv")
PRICE_COLUMN = "가격식+실거래"

HIT_TOLERANCE = 0.20       # ±20% 안이면 맞힌 것으로 본다
INTERVAL_COVERAGE = 0.80   # 가격 구간이 실제 거래가를 담아야 하는 비율
BAND_EDGES = [0.6, 0.7, 0.8]  # 신뢰도 구간: ~0.6 / 0.6~0.7 / 0.7~0.8 / 0.8~
BAND_NAMES = ["0.6 미만", "0.6~0.7", "0.7~0.8", "0.8 이상"]
MIN_CONFIDENCE, MAX_CONFIDENCE = 0.05, 0.95
NO_EVIDENCE_SPREAD = 0.35  # 근거 거래가 없어 흩어짐을 잴 수 없을 때 넣는 값 (가장 불확실한 쪽)


def confidence_features(floor, build_year, weight_sum, spread) -> pd.DataFrame:
    """신뢰도 식에 넣을 조건. 값 하나를 넣어도, 여러 건을 한꺼번에 넣어도 된다."""
    floor = np.atleast_1d(np.asarray(floor, dtype=float))
    age = int(REF_DATE[:4]) - np.atleast_1d(np.asarray(build_year, dtype=float))
    weight_sum = np.atleast_1d(np.asarray(weight_sum, dtype=float))
    spread = np.atleast_1d(np.asarray(spread, dtype=float))
    return pd.DataFrame({
        "근거량": np.log1p(np.clip(weight_sum, 0, 200)),  # 가까운·최근·같은 건물 거래가 많을수록 크다
        "흩어짐": np.clip(np.where(np.isnan(spread), NO_EVIDENCE_SPREAD, spread), 0, 0.6),
        "지하층": (floor < 0).astype(float),
        "1층": (floor == 1).astype(float),
        "신축": (age <= 1).astype(float),
        "노후": (age >= 30).astype(float),
    })


@dataclass
class ConfidenceModel:
    intercept: float
    coefficients: dict[str, float]  # 조건 이름 -> 계수. 음수면 그 조건에서 신뢰도가 내려간다
    half_widths: list[float]        # 신뢰도 구간별 가격 구간 반폭(로그 단위). BAND_NAMES 순서

    def confidence(self, features: pd.DataFrame) -> np.ndarray:
        score = self.intercept + sum(self.coefficients[name] * features[name].to_numpy() for name in self.coefficients)
        probability = 1 / (1 + np.exp(-score))
        return np.clip(probability, MIN_CONFIDENCE, MAX_CONFIDENCE)

    def half_width(self, confidence) -> np.ndarray:
        band = np.searchsorted(BAND_EDGES, np.atleast_1d(confidence), side="right")
        return np.asarray(self.half_widths)[band]

    def assess(self, price: float, floor, build_year, weight_sum, spread, penalty: float = 1.0) -> dict:
        """추정가 한 건의 신뢰도와 가격 구간을 돌려준다.

        penalty: 면적·준공년도를 찾지 못해 어림값을 쓴 경우 신뢰도에 곱하는 값(1보다 작다).
        이런 경우는 홀드아웃에 없어서 실제 비율로 맞추지 못했고, 낮추는 방향으로만 정한 규칙이다.
        """
        confidence = float(self.confidence(confidence_features(floor, build_year, weight_sum, spread))[0])
        confidence = max(MIN_CONFIDENCE, confidence * penalty)
        width = float(self.half_width(confidence)[0])
        return {
            "confidence": round(confidence, 2),
            "price_low": price * np.exp(-width),
            "price_high": price * np.exp(width),
        }

    def save(self, path: Path = MODEL_PATH) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ConfidenceModel":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def with_errors(holdout: pd.DataFrame) -> pd.DataFrame:
    """홀드아웃 결과에 오차율, 맞힘 여부, 로그 오차 크기를 붙인다."""
    holdout = holdout.copy()
    holdout["오차율"] = (holdout[PRICE_COLUMN] - holdout["price"]).abs() / holdout["price"]
    holdout["맞힘"] = (holdout["오차율"] <= HIT_TOLERANCE).astype(int)
    holdout["로그오차"] = np.log(holdout["price"] / holdout[PRICE_COLUMN]).abs()
    return holdout


def features_of(holdout: pd.DataFrame) -> pd.DataFrame:
    return confidence_features(holdout["floor"], holdout["build_year"], holdout["weight_sum"], holdout["spread"])


def calibrate(dev: pd.DataFrame) -> ConfidenceModel:
    """dev 결과로 신뢰도 식과 구간 폭을 정한다."""
    from sklearn.linear_model import LogisticRegression

    dev = with_errors(dev)
    features = features_of(dev)
    fitted = LogisticRegression(C=10, max_iter=1000).fit(features.to_numpy(), dev["맞힘"].to_numpy())
    model = ConfidenceModel(
        intercept=round(float(fitted.intercept_[0]), 4),
        coefficients={name: round(float(value), 4) for name, value in zip(features.columns, fitted.coef_[0])},
        half_widths=[],
    )
    # 신뢰도 구간별로, 로그 오차의 80%가 들어오는 폭
    band = np.searchsorted(BAND_EDGES, model.confidence(features), side="right")
    log_error = dev["로그오차"].to_numpy()
    overall = float(np.quantile(log_error, INTERVAL_COVERAGE))
    model.half_widths = [
        round(float(np.quantile(log_error[band == k], INTERVAL_COVERAGE)) if (band == k).sum() >= 30 else overall, 4)
        for k in range(len(BAND_NAMES))
    ]
    return model


def interval_text(half_width: float) -> str:
    """로그 단위 반폭을 "추정가 대비 -34% ~ +51%"처럼 읽을 수 있게 바꾼다.

    구간은 추정가에 exp(-폭)과 exp(+폭)을 곱해 만든다. 곱하는 방식이라 아래쪽 폭과 위쪽 폭이 다르다.
    (가격이 반이 되는 것과 두 배가 되는 것을 같은 크기의 변화로 본다.)
    """
    return f"{(np.exp(-half_width) - 1) * 100:+.0f}% ~ {(np.exp(half_width) - 1) * 100:+.0f}%"


def reliability_table(holdout: pd.DataFrame, model: ConfidenceModel) -> pd.DataFrame:
    """신뢰도 구간별로 "말한 신뢰도"와 "실제로 맞힌 비율"을 나란히 놓는다."""
    holdout = with_errors(holdout)
    confidence = model.confidence(features_of(holdout))
    band = np.searchsorted(BAND_EDGES, confidence, side="right")
    inside = holdout["로그오차"].to_numpy() <= model.half_width(confidence)
    rows = []
    for k, name in enumerate(BAND_NAMES):
        mask = band == k
        if not mask.any():
            continue
        rows.append({
            "신뢰도 구간": name,
            "건수": int(mask.sum()),
            "평균 신뢰도": round(float(confidence[mask].mean()), 2),
            "실제 20%이내": round(float(holdout["맞힘"].to_numpy()[mask].mean()), 2),
            "중앙값오차율_%": round(float(np.median(holdout["오차율"].to_numpy()[mask])) * 100, 1),
            "가격 구간(추정가 대비)": interval_text(model.half_widths[k]),
            "구간 안에 든 비율": round(float(inside[mask].mean()), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    holdout = pd.read_csv(HOLDOUT_PATH, encoding="utf-8-sig")
    dev, test = holdout[holdout["split"] == "dev"], holdout[holdout["split"] == "test"]
    model = calibrate(dev)
    model.save()

    print("== 신뢰도 식 (dev로 학습). 음수면 신뢰도를 낮추는 조건 ==")
    print(f"기준값 {model.intercept:+.2f}")
    for name, value in model.coefficients.items():
        print(f"{name:<6} {value:+.2f}")

    for name, part in [("dev", dev), ("test", test)]:
        print(f"\n== 신뢰도가 실제와 맞는가: {name} {len(part):,}건 ==")
        print(reliability_table(part, model).to_string(index=False))

    test_errors = with_errors(test)
    test_confidence = model.confidence(features_of(test))
    inside = test_errors["로그오차"].to_numpy() <= model.half_width(test_confidence)
    print(f"\ntest 전체: 평균 신뢰도 {test_confidence.mean():.2f} / 실제 20% 이내 {test_errors['맞힘'].mean():.2f}"
          f" / 가격 구간 안에 든 비율 {inside.mean():.2f} (목표 {INTERVAL_COVERAGE:.2f})")
    print(f"저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
