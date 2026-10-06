# collect/geocode_trades.py
"""trades 테이블의 고유 지번을 좌표로 바꿔 geocode 테이블에 쌓는다.

- 이미 변환한 지번은 다시 호출하지 않는다. 중간에 끊겨도 같은 명령으로 이어서 한다.
- 실패(not_found)와 불일치(mismatch)도 기록해 둔다. --retry를 붙이면 그 지번만 다시 시도한다.

실행:
    uv run --env-file .env python -m collect.geocode_trades
"""
import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

from model.geocode import GeocodeError, geocode_lot

DB_PATH = Path("data/trades.db")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS geocode (
    sigungu    TEXT NOT NULL,
    dong       TEXT NOT NULL,
    jibun      TEXT NOT NULL,
    status     TEXT NOT NULL,
    lat        REAL,
    lon        REAL,
    b_code     TEXT,
    matched    TEXT,
    fetched_at TEXT,
    PRIMARY KEY (sigungu, dong, jibun)
)
"""

# 아직 geocode 테이블에 없는 지번만 고른다.
TODO_QUERY = """
SELECT DISTINCT t.sgg_cd, t.sigungu, t.dong, t.jibun, t.bonbun, t.bubun
FROM trades t
LEFT JOIN geocode g ON g.sigungu = t.sigungu AND g.dong = t.dong AND g.jibun = t.jibun
WHERE g.jibun IS NULL
ORDER BY t.sgg_cd, t.dong, t.bonbun, t.bubun
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--retry", action="store_true", help="ok가 아닌 지번을 지우고 다시 시도한다")
    parser.add_argument("--limit", type=int, default=0, help="시험용: 앞에서부터 N건만 변환한다")
    args = parser.parse_args()

    key = (os.environ.get("KAKAO_REST_KEY") or "").strip()
    if not key:
        sys.exit("환경변수 KAKAO_REST_KEY가 없습니다. .env를 확인하세요.")

    con = sqlite3.connect(args.db)
    try:
        con.execute(CREATE_TABLE)
        if args.retry:
            con.execute("DELETE FROM geocode WHERE status != 'ok'")
            con.commit()

        todo = con.execute(TODO_QUERY).fetchall()
        if args.limit:
            todo = todo[: args.limit]
        print(f"변환할 지번: {len(todo):,}개")

        for done, (sgg_cd, sigungu, dong, jibun, bonbun, bubun) in enumerate(todo, start=1):
            try:
                row = geocode_lot(key, sigungu, dong, jibun, bonbun, bubun)
            except GeocodeError as e:
                con.commit()
                sys.exit(f"[중단] {sigungu} {dong} {jibun}: {e}")
            # 좌표가 다른 구로 잡혔으면 일치로 보지 않는다. (법정동코드 앞 5자리 = 시군구코드)
            if row["status"] == "ok" and not row["b_code"].startswith(sgg_cd):
                row["status"] = "mismatch"
            con.execute(
                "INSERT OR REPLACE INTO geocode VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sigungu, dong, jibun, row["status"], row["lat"], row["lon"], row["b_code"], row["matched"],
                 datetime.now().isoformat(timespec="seconds")),
            )
            if done % 200 == 0:
                con.commit()
                print(f"  {done:,} / {len(todo):,}")
            time.sleep(0.05)
        con.commit()

        print("\n== geocode 테이블 ==")
        for status, count in con.execute("SELECT status, COUNT(*) FROM geocode GROUP BY status ORDER BY status"):
            print(f"{status:<10} {count:>7,}")
        missing = con.execute(
            "SELECT COUNT(*) FROM trades t LEFT JOIN geocode g "
            "ON g.sigungu = t.sigungu AND g.dong = t.dong AND g.jibun = t.jibun AND g.status = 'ok' "
            "WHERE g.jibun IS NULL"
        ).fetchone()[0]
        total = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        print(f"좌표가 없는 거래: {missing:,} / {total:,}건")
        print("\n== ok가 아닌 지번 (최대 20개) ==")
        for row in con.execute("SELECT sigungu, dong, jibun, status, matched FROM geocode WHERE status != 'ok' LIMIT 20"):
            print("  ", *row)
    finally:
        con.close()


if __name__ == "__main__":
    main()
