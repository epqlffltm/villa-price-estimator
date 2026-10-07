# model/resolve.py
"""입력 한 줄(시군구·동·지번·층·호·면적)을 시세 추정에 필요한 값으로 채운다.

입력에 없는 값은 아래 순서로 찾는다. 찾은 방법은 근거 문장(notes)에 남기고,
어림값을 쓸수록 신뢰도에 곱할 값(penalty)을 낮춘다.

  좌표       수집해 둔 geocode 테이블 -> Kakao 주소 검색 -> 없음(같은 건물 거래만 반영)
  전용면적    입력 -> 건축물대장(호, 없으면 같은 층) -> 같은 건물 같은 층 실거래 -> 같은 건물 실거래 -> 동 중앙값
             (호는 "201", "201호", "B동 201", "102-402"처럼 동을 함께 적어도 된다)
  준공년도    같은 지번 실거래 -> 건축물대장 표제부 -> 주변 실거래 중앙값

실거래·좌표·건축물대장 어디에도 없는 지번은 추정하지 않고 실패로 돌려준다.
"""
import re
import statistics
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from model import building
from model.estimate import distance_m, make_target
from model.geocode import GeocodeError, geocode_lot
from model.normalize import parse_float, split_jibun

# 구 이름 -> (시군구코드, 시군구 전체 이름, 이 구에서 다루는 동. None이면 구 전체)
SUPPORTED = {
    "강남구": ("11680", "서울특별시 강남구", None),
    "관악구": ("11620", "서울특별시 관악구", None),
    "강서구": ("11500", "서울특별시 강서구", "화곡동"),
}
NEARBY_YEAR_RADIUS_M = 150.0


class InputError(ValueError):
    """입력만으로는 시세를 낼 수 없다. 메시지가 그대로 status의 실패 사유가 된다."""


@dataclass
class Resolved:
    region: str
    target: dict
    notes: list[str] = field(default_factory=list)
    penalty: float = 1.0


def parse_floor(text) -> int:
    """"3", "3층", "-1", "B1", "지하1층" -> 층 번호(지하는 음수)."""
    raw = str(text or "").strip().upper()
    match = re.search(r"\d+", raw)
    if not match:
        raise InputError(f"층을 읽을 수 없음: '{text}'")
    number = int(match.group())
    if number == 0:
        raise InputError("0층은 없음")
    is_basement = raw.startswith("-") or raw.startswith("B") or "지하" in raw
    return -number if is_basement else number


def canonical_jibun(text) -> tuple[str, bool, int, int]:
    """"123-45" -> ("123-45", False, 123, 45). 부번 0은 "123"으로 쓴다."""
    parts = split_jibun(str(text or ""))
    if parts is None:
        raise InputError(f"지번을 읽을 수 없음: '{text}'")
    is_san, bonbun, bubun = parts
    name = f"{bonbun}-{bubun}" if bubun else f"{bonbun}"
    return ("산 " + name if is_san else name), is_san, bonbun, bubun


class Resolver:
    def __init__(self, trades: pd.DataFrame, geocodes: pd.DataFrame, data_key: str = "", kakao_key: str = "",
                 session=None):
        self.trades = {region: group.reset_index(drop=True) for region, group in trades.groupby("region")}
        self.geocodes = geocodes[geocodes["status"] == "ok"]
        self.data_key = (data_key or "").strip()
        self.kakao_key = (kakao_key or "").strip()
        self.session = session
        self._register_cache: dict[tuple, list[dict] | None] = {}

    def resolve(self, row: dict) -> Resolved:
        gu, sgg_cd, sigungu, only_dong = self._region_of(row.get("sigungu"))
        dong = str(row.get("dong") or "").strip()
        if not dong:
            raise InputError("동이 비어 있음")
        if only_dong and dong != only_dong:
            raise InputError(f"권역 밖: {gu}는 {only_dong}만 지원")
        jibun, is_san, bonbun, bubun = canonical_jibun(row.get("jibun"))
        floor = parse_floor(row.get("floor"))
        ho = str(row.get("ho") or "").strip()

        region = only_dong or gu
        region_trades = self.trades[region]
        lot = f"{sgg_cd} {dong} {jibun}"
        lot_trades = region_trades[region_trades["lot"] == lot]
        resolved = Resolved(region=region, target={})
        if dong not in set(region_trades["dong"]):
            resolved.notes.append(f"{dong}은 수집된 거래가 없는 동")
            resolved.penalty *= 0.8

        lat, lon, b_code = self._locate(sigungu, dong, jibun, bonbun, bubun, resolved)
        register = _Register(self, b_code, bonbun, bubun, is_san)

        # 실거래도, 좌표도, 건축물대장도 없으면 실제로 있는 지번인지 확인할 길이 없다. 지어내지 않고 실패로 남긴다.
        if lot_trades.empty and lat is None and not (register.available and register.title_items()):
            raise InputError("지번을 확인할 수 없음(실거래·좌표·건축물대장 어디에도 없음)")

        area, area_is_guess = self._area(row.get("area_m2"), floor, ho, lot_trades, region_trades, dong, register, resolved)
        build_year = self._build_year(lot_trades, region_trades, dong, lat, lon, register, resolved)
        house_type = self._house_type(lot_trades, register)

        resolved.target = make_target(dong, lot, floor, area, build_year, house_type, lat, lon, area_is_guess)
        return resolved

    def _region_of(self, sigungu) -> tuple[str, str, str, str | None]:
        text = str(sigungu or "")
        for gu, (sgg_cd, full_name, only_dong) in SUPPORTED.items():
            if gu in text:
                return gu, sgg_cd, full_name, only_dong
        raise InputError(f"권역 밖: '{text}'")

    def _locate(self, sigungu, dong, jibun, bonbun, bubun, resolved: Resolved):
        """(위도, 경도, 법정동코드)를 찾는다. 못 찾은 값은 None."""
        same_dong = self.geocodes[(self.geocodes["sigungu"] == sigungu) & (self.geocodes["dong"] == dong)]
        known = same_dong[same_dong["jibun"] == jibun]
        if len(known):
            row = known.iloc[0]
            return float(row["lat"]), float(row["lon"]), str(row["b_code"])
        # 법정동코드는 동마다 하나라서, 같은 동의 다른 지번에서 가져올 수 있다.
        b_code = str(same_dong.iloc[0]["b_code"]) if len(same_dong) else None
        if self.kakao_key:
            try:
                found = geocode_lot(self.kakao_key, sigungu, dong, jibun, bonbun, bubun, session=self.session)
            except GeocodeError:
                found = {"status": "error"}
            if found["status"] == "ok":
                return found["lat"], found["lon"], found["b_code"] or b_code
        resolved.notes.append("좌표를 찾지 못해 같은 건물 거래만 반영")
        return None, None, b_code

    def _area(self, given, floor, ho, lot_trades, region_trades, dong, register, resolved: Resolved):
        """(전용면적, 어림값인지)를 돌려준다."""
        area = parse_float(str(given or ""))
        if area and area > 0:
            return area, False

        # 같은 호 이름이 여러 동에 있을 때 가리는 데 쓴다: 이 지번·이 층에서 실제로 거래된 면적들
        traded_areas = list(lot_trades.loc[lot_trades["floor"] == floor, "area_m2"])
        found = building.find_unit_area(register.area_items(), floor, ho, traded_areas) if register.available else None
        if found:
            resolved.notes.append(f"면적 {found[0]}㎡({found[1]})")
            if building.is_guess(found[1]):  # 호를 하나로 특정하지 못한 경우
                resolved.penalty *= 0.9
                return found[0], True
            return found[0], False

        why = "건축물대장에서 찾지 못함" if register.available else "건축물대장 조회 불가"
        same_floor = lot_trades.loc[lot_trades["floor"] == floor, "area_m2"]
        if len(same_floor):
            distinct = same_floor.round(1).nunique()
            resolved.notes.append(f"면적 {same_floor.median():.2f}㎡(같은 건물 {floor}층 실거래 기준, {why})")
            if distinct > 1:
                resolved.penalty *= 0.8
            return float(same_floor.median()), distinct > 1
        if len(lot_trades):
            resolved.notes.append(f"면적 {lot_trades['area_m2'].median():.2f}㎡(같은 건물 실거래 중앙값, {why})")
            resolved.penalty *= 0.7
            return float(lot_trades["area_m2"].median()), True
        pool = region_trades.loc[region_trades["dong"] == dong, "area_m2"]
        pool = pool if len(pool) else region_trades["area_m2"]
        resolved.notes.append(f"면적을 찾지 못해 {pool.median():.2f}㎡(지역 중앙값)로 가정")
        resolved.penalty *= 0.5
        return float(pool.median()), True

    def _build_year(self, lot_trades, region_trades, dong, lat, lon, register, resolved: Resolved) -> int:
        if len(lot_trades):
            return int(lot_trades["build_year"].mode().iloc[0])
        year = building.built_year(register.title_items()) if register.available else None
        if year:
            resolved.notes.append(f"준공 {year}년(건축물대장)")
            return year
        pool = region_trades.loc[region_trades["dong"] == dong, "build_year"]
        if lat is not None:
            distance = distance_m(lat, lon, region_trades["lat"].to_numpy(), region_trades["lon"].to_numpy())
            near = region_trades.loc[np.nan_to_num(distance, nan=np.inf) <= NEARBY_YEAR_RADIUS_M, "build_year"]
            pool = near if len(near) >= 5 else pool
        pool = pool if len(pool) else region_trades["build_year"]
        year = int(statistics.median(pool))
        resolved.notes.append(f"준공년도를 찾지 못해 주변 중앙값 {year}년으로 가정")
        resolved.penalty *= 0.8
        return year

    def _house_type(self, lot_trades, register) -> str:
        if len(lot_trades):
            return "연립" if (lot_trades["house_type"] == "연립").mean() > 0.5 else "다세대"
        found = building.house_type_of(register.title_items()) if register.available else None
        return found or "다세대"

    def fetch_register(self, operation, b_code, bonbun, bubun, is_san) -> list[dict]:
        key = (operation, b_code, bonbun, bubun, is_san)
        if key not in self._register_cache:
            try:
                self._register_cache[key] = building.fetch_items(
                    operation, self.data_key, b_code, bonbun, bubun, is_san, session=self.session)
            except building.BuildingRegisterError:
                self._register_cache[key] = None
        return self._register_cache[key] or []


class _Register:
    """한 지번의 건축물대장. 실제로 필요해질 때만 조회한다."""

    def __init__(self, resolver: Resolver, b_code, bonbun, bubun, is_san):
        self.resolver, self.args = resolver, (b_code, bonbun, bubun, is_san)
        self.available = bool(resolver.data_key and b_code)

    def area_items(self) -> list[dict]:
        return self.resolver.fetch_register("getBrExposPubuseAreaInfo", *self.args)

    def title_items(self) -> list[dict]:
        return self.resolver.fetch_register("getBrTitleInfo", *self.args)
