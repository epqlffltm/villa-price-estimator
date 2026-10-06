# collect/probe_building.py
"""건축물대장 API가 실제로 어떤 모양의 응답을 주는지 한 지번으로 확인하는 시험 스크립트.

predict.py는 입력에 전용면적이 없을 때 건축물대장에서 찾아야 한다. 그 코드를 쓰기 전에
표제부(준공일·층수)와 전유공용면적(호별 면적) 응답을 그대로 저장해 필드 이름을 확인한다.
응답은 data/raw/probe/ 에 저장된다(커밋되지 않는 폴더).

실행:
    uv run --env-file .env python -m collect.probe_building
    uv run --env-file .env python -m collect.probe_building --dong 화곡동 --jibun 24-21
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from urllib.parse import unquote

import requests

BASE_URL = "https://apis.data.go.kr/1613000/BldRgstHubService"
OPERATIONS = {
    "title": "getBrTitleInfo",               # 표제부: 건물 한 동의 준공일, 층수, 세대수
    "area": "getBrExposPubuseAreaInfo",      # 전유공용면적: 호별 전용·공용 면적
}
DB_PATH = Path("data/trades.db")
OUT_DIR = Path("data/raw/probe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dong", default="신림동")
    parser.add_argument("--jibun", default="598-178")
    args = parser.parse_args()

    key = unquote((os.environ.get("DATA_GO_KR_KEY") or "").strip())
    if not key:
        sys.exit("환경변수 DATA_GO_KR_KEY가 없습니다. .env를 확인하세요.")

    # 좌표 변환 때 받아 둔 법정동코드(10자리)를 쓴다: 앞 5자리 시군구, 뒤 5자리 법정동
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute(
            "SELECT t.sigungu, t.bonbun, t.bubun, g.b_code FROM trades t "
            "JOIN geocode g ON g.sigungu = t.sigungu AND g.dong = t.dong AND g.jibun = t.jibun "
            "WHERE t.dong = ? AND t.jibun = ? LIMIT 1",
            (args.dong, args.jibun),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        sys.exit(f"{args.dong} {args.jibun} 을 trades·geocode 테이블에서 찾지 못했습니다.")
    sigungu, bonbun, bubun, b_code = row
    print(f"대상: {sigungu} {args.dong} {args.jibun} / 법정동코드 {b_code}")

    params = {
        "serviceKey": key,
        "sigunguCd": b_code[:5],
        "bjdongCd": b_code[5:],
        "platGbCd": "0",            # 0 대지, 1 산
        "bun": f"{bonbun:04d}",     # 본번 4자리
        "ji": f"{bubun:04d}",       # 부번 4자리
        "numOfRows": 100,
        "pageNo": 1,
        "_type": "json",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, operation in OPERATIONS.items():
        res = requests.get(f"{BASE_URL}/{operation}", params=params, timeout=30)
        path = OUT_DIR / f"{name}_{args.dong}_{args.jibun}.txt"
        path.write_text(res.text, encoding="utf-8")
        print(f"\n== {operation}: HTTP {res.status_code}, {len(res.text):,}자 -> {path} ==")
        try:
            body = res.json()["response"]["body"]
            items = body.get("items") or {}
            items = items.get("item", []) if isinstance(items, dict) else items
            items = items if isinstance(items, list) else [items]
            print(f"totalCount={body.get('totalCount')} / 이번 응답 {len(items)}건")
            if items:
                print("필드:", ", ".join(items[0].keys()))
                print("첫 항목:", json.dumps(items[0], ensure_ascii=False)[:1200])
        except (ValueError, KeyError, TypeError):
            print("JSON으로 읽지 못했습니다. 응답 앞부분:")
            print(res.text[:800])


if __name__ == "__main__":
    main()
