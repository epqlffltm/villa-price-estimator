# collect/fetch_trades.py
"""국토교통부 연립다세대 매매 실거래가 API를 구·월 단위로 호출해 원본 그대로 CSV로 저장한다.

- 가공하지 않는다. API가 준 필드명·값을 그대로 저장하고, 정제는 다음 단계에서 한다.
- 구·월마다 파일 하나(data/raw/trades/11500_202510.csv). 이미 있으면 건너뛴다(--force로 재수집).
- API가 알려준 totalCount와 실제 저장 건수를 _manifest.csv에 남겨 누락을 확인할 수 있게 한다.

실행 예:
    set DATA_GO_KR_KEY=발급받은_Decoding_키      (PowerShell: $env:DATA_GO_KR_KEY="...")
    python collect/fetch_trades.py --start 202110 --end 202610
"""
import argparse
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import requests

URL = "https://apis.data.go.kr/1613000/RTMSDataSvcRHTrade/getRTMSDataSvcRHTrade"

# 법정동코드 앞 5자리(시군구). 강서구는 구 전체를 받아 두고 화곡동만 쓰는 것은 정제 단계에서 거른다.
LAWD_CODES = {
    "11500": "서울특별시 강서구",
    "11680": "서울특별시 강남구",
    "11620": "서울특별시 관악구",
}

OUT_DIR = Path("data/raw/trades")
PAGE_SIZE = 1000
OK_CODES = {"00", "000"}


def month_range(start: str, end: str):
    """'202110', '202610' -> '202110', '202111', ... '202610'"""
    y, m = int(start[:4]), int(start[4:])
    end_y, end_m = int(end[:4]), int(end[4:])
    while (y, m) <= (end_y, end_m):
        yield f"{y}{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def call_api(key: str, lawd: str, ym: str, page: int) -> ET.Element:
    """한 페이지를 받아 XML 루트를 돌려준다. 일시 오류는 3번까지 다시 시도한다."""
    params = {
        "serviceKey": key,
        "LAWD_CD": lawd,
        "DEAL_YMD": ym,
        "pageNo": page,
        "numOfRows": PAGE_SIZE,
    }
    last_error = None
    for attempt in range(3):
        try:
            res = requests.get(URL, params=params, timeout=30)
            res.raise_for_status()
            root = ET.fromstring(res.content)
            code = (root.findtext(".//resultCode") or "").strip()
            if code not in OK_CODES:
                # 키 오류·호출 한도 초과 등은 다시 시도해도 같으므로 내용을 보여주고 멈춘다.
                msg = root.findtext(".//resultMsg") or root.findtext(".//returnAuthMsg") or res.text[:300]
                raise SystemExit(f"[API 오류] {lawd} {ym} code={code!r} msg={msg}")
            return root
        except (requests.RequestException, ET.ParseError) as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise SystemExit(f"[실패] {lawd} {ym} page={page}: {last_error}")


def fetch_month(key: str, lawd: str, ym: str):
    """한 구·한 달의 모든 페이지를 모아 (행 목록, totalCount)를 돌려준다."""
    rows, page, total = [], 1, 0
    while True:
        root = call_api(key, lawd, ym, page)
        total = int(root.findtext(".//totalCount") or 0)
        items = [{c.tag: (c.text or "").strip() for c in item} for item in root.iter("item")]
        rows.extend(items)
        if not items or len(rows) >= total:
            return rows, total
        page += 1


def save_csv(path: Path, rows: list[dict]) -> None:
    """필드가 행마다 조금 달라도 빠지지 않도록 전체 필드의 합집합을 헤더로 쓴다."""
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_manifest(lawd: str, ym: str, total: int, saved: int) -> None:
    path = OUT_DIR / "_manifest.csv"
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["lawd_cd", "deal_ym", "total_count", "saved_rows", "fetched_at"])
        writer.writerow([lawd, ym, total, saved, datetime.now().isoformat(timespec="seconds")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="시작 계약월 YYYYMM")
    parser.add_argument("--end", required=True, help="끝 계약월 YYYYMM")
    parser.add_argument("--force", action="store_true", help="이미 받은 달도 다시 받는다")
    args = parser.parse_args()

    key = os.environ.get("DATA_GO_KR_KEY")
    if not key:
        sys.exit("환경변수 DATA_GO_KR_KEY가 없습니다. 공공데이터포털의 일반 인증키를 넣어 주세요.")
    # Encoding 키(%2F, %3D가 들어 있는 형태)를 넣어도 동작하도록 원래 문자로 되돌린다.
    # requests가 전송할 때 다시 인코딩하므로, 여기서 풀어 두지 않으면 이중 인코딩되어 인증에 실패한다.
    key = unquote(key.strip())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mismatches = []

    for lawd, name in LAWD_CODES.items():
        for ym in month_range(args.start, args.end):
            path = OUT_DIR / f"{lawd}_{ym}.csv"
            if path.exists() and not args.force:
                continue
            rows, total = fetch_month(key, lawd, ym)
            save_csv(path, rows)
            append_manifest(lawd, ym, total, len(rows))
            flag = "" if total == len(rows) else "  <-- 건수 불일치"
            if flag:
                mismatches.append((lawd, ym, total, len(rows)))
            print(f"{name} {ym}: total={total} saved={len(rows)}{flag}")
            time.sleep(0.2)

    if mismatches:
        print(f"\n건수 불일치 {len(mismatches)}건: {mismatches}")
    else:
        print("\n모든 구·월에서 totalCount와 저장 건수가 일치합니다.")


if __name__ == "__main__":
    main()