# villa-price-estimator

서울 다세대·연립(빌라)의 매매 시세를 추정한다. 지번·층·호를 넣으면 추정 시세, 가격 구간, 신뢰도, 산출 근거를 돌려준다.

- 대상 권역: 서울특별시 강서구 화곡동, 강남구, 관악구
- 산출 기준일: 2026-10-06
- 방식: 요인별 가격식(회귀) + 같은 건물·주변 실거래 보정(비교사례)
- 자체 검증(처음 보는 물건 2,381건): 중앙값 오차율 10.6%, 실제 거래가의 ±20% 안에 든 비율 76.4%

## 실행

Python 3.12 이상이 필요하다.

```bash
# uv를 쓰는 경우
uv sync
uv run python predict.py --input input.csv --output output.csv

# pip를 쓰는 경우
pip install -r requirements.txt
python predict.py --input input.csv --output output.csv
```

예시 입력으로 바로 돌려 볼 수 있다. 20건 기준 몇 초 안에 끝난다.

```bash
uv run python predict.py --input examples/input.csv --output output.csv
```

### 입력 `input.csv`

| 컬럼 | 예시 | 비고 |
|---|---|---|
| id | 1 | 식별자 |
| sigungu | 서울특별시 관악구 | 시군구 |
| dong | 봉천동 | 법정동 |
| jibun | 123-45 | 지번 |
| floor | 3 | 층. 지하는 `-1` 또는 `B1` |
| ho | 301 | 호. 비어 있어도 된다 |
| area_m2 | 42.5 | 전용면적. 비어 있어도 된다 |

### 출력 `output.csv`

| 컬럼 | 예시 | 비고 |
|---|---|---|
| id | 1 | 입력과 같다 |
| price_est | 285000000 | 추정 시세(원) |
| price_low | 260000000 | 가격 구간 하한 |
| price_high | 310000000 | 가격 구간 상한 |
| confidence | 0.72 | 추정가가 실제 거래가의 ±20% 안에 들 확률 |
| basis | 같은 건물 3건·반경 100m 51건 실거래 반영(가격식 대비 -21%) / 2012년 준공 4층 29.87㎡ 다세대 | 산출 근거 |
| status | ok | 산출하지 못하면 `fail: 사유` |

한 줄에서 문제가 생겨도 멈추지 않고, 그 줄만 `fail`로 남긴다. 권역 밖 주소, 읽을 수 없는 지번·층, 실거래·좌표·건축물대장 어디에서도 확인되지 않는 지번이 실패 사유다.

## API 키 (선택)

수집해 둔 `data/trades.db`만으로도 동작한다. 아래 키를 환경변수로 넣으면 DB에 없는 값을 실행 중에 조회한다.

| 환경변수 | 쓰는 곳 | 없을 때 |
|---|---|---|
| `DATA_GO_KR_KEY` | 건축물대장에서 전용면적(호 기준)·준공년도 조회 | 같은 건물 실거래 기록으로 어림하고 신뢰도를 낮춘다 |
| `KAKAO_REST_KEY` | 수집되지 않은 지번의 좌표 조회 | 같은 건물 거래만 반영한다 |

발급 방법

1. **공공데이터포털** (data.go.kr): 로그인 후 `국토교통부_건축HUB_건축물대장정보 서비스`를 검색해 활용신청한다. 자동 승인된다. 마이페이지의 일반 인증키를 쓴다. Encoding 키와 Decoding 키 어느 쪽을 넣어도 된다.
2. **Kakao Developers** (developers.kakao.com): 애플리케이션을 만들고 REST API 키를 쓴다. 앱의 제품 설정에서 카카오맵을 켜야 한다.

넣는 방법

```bash
# .env 파일을 쓰는 경우 (.env.example을 복사해 값을 채운다)
uv run --env-file .env python predict.py --input input.csv --output output.csv

# 직접 넣는 경우
export DATA_GO_KR_KEY=...      # PowerShell: $env:DATA_GO_KR_KEY="..."
export KAKAO_REST_KEY=...
```

## 폴더 구조

```
predict.py            제출 프로그램. 입력 CSV -> 출력 CSV
validate.py           홀드아웃으로 모델별 오차를 잰다
report.py             검증 결과를 reports/validation.md로 정리한다
collect/
  fetch_trades.py     실거래가 API -> data/raw/trades/*.csv
  geocode_trades.py   지번 -> 좌표 (Kakao) -> geocode 테이블
  probe_building.py   건축물대장 API 응답 확인용
model/
  normalize.py        금액·지번·날짜 문자열 변환
  clean.py            원본 CSV 정제 -> data/trades.db
  holdout.py          검증 규칙 (홀드아웃, 같은 호 제외, dev/test)
  baseline.py         기준선: 동별 ㎡당 중앙값 × 면적
  formula.py          가격식: 요인별 회귀
  estimate.py         가격식 + 실거래 보정
  confidence.py       신뢰도와 가격 구간
  confidence.json     학습된 신뢰도 식
  geocode.py          Kakao 주소 검색
  building.py         건축물대장 조회
  resolve.py          입력에 없는 면적·준공년도·좌표 채우기
data/trades.db        정제된 실거래와 좌표 (SQLite)
reports/              검증 결과
examples/input.csv    예시 입력
tests/                테스트 117개
```

## 사용 데이터

| 데이터 | 출처 | 범위 | 건수 |
|---|---|---|---|
| 연립다세대 매매 실거래가 | 국토교통부 (공공데이터포털 API) | 강서구·강남구·관악구, 계약 2021.10 ~ 2026.10 | 원본 26,784건 → 정제 후 21,220건 |
| 지번 좌표, 법정동코드 | Kakao 로컬 API 주소 검색 | 위 거래의 고유 지번 | 7,511개 중 7,508개 변환 |
| 건축물대장 표제부·전유공용면적 | 국토교통부 건축HUB (공공데이터포털 API) | 실행 중 필요한 지번만 조회 | 저장하지 않음 |

정제에서 뺀 것은 해제된 거래 1,609건과 강서구 중 화곡동이 아닌 거래 3,955건이다. 가격이 높거나 낮은 실제 거래는 지우지 않았다. 연립도 포함했다(2,109건). 수집일은 2026-10-06이며, 신고 기한(계약 후 30일) 때문에 마지막 한두 달은 건수가 덜 찼다.

상용 시세 서비스의 값은 입력에도 학습에도 쓰지 않았다. 소유자 등 개인정보는 수집하지 않았다.

## 모델

1. **가격식** (`model/formula.py`): 권역마다 ㎡당 가격을 동·거래 시점·연식·면적·층·주택 유형·거래 방식으로 설명하는 회귀식을 만든다.
2. **실거래 보정** (`model/estimate.py`): 같은 건물과 주변 실거래가 가격식보다 몇 % 비싸게/싸게 팔렸는지를 가중평균해 보정한다. 가까울수록(50m), 최근일수록(24개월), 같은 건물일수록(10배) 크게 반영한다.
3. **신뢰도** (`model/confidence.py`): 추정가가 실제의 ±20% 안에 들 확률. 근거 거래의 양과 흩어짐, 지하층·1층, 신축·노후로 정한다. 가격 구간은 신뢰도 구간별로 실제 거래가의 80%가 들어온 폭이다.

추정 대상과 같은 호로 보이는 거래(같은 지번·층·면적)는 가격식 학습과 보정 양쪽에서 뺀다. 실거래가에는 호가 없어서 이 세 가지가 같으면 같은 호로 본다.

## 자체 검증

기준일로부터 12개월 안의 거래 4,860건을 홀드아웃으로 삼았다. 지번 기준으로 반씩 나눠 dev(2,479건)로 설정값을 고르고 test(2,381건)는 성적 확인에만 썼다.

| 모델 | 중앙값 오차율 | 20% 이내 |
|---|---|---|
| 기준선 (동별 ㎡당 중앙값 × 면적) | 28.6% | 35.3% |
| 가격식 | 17.2% | 56.9% |
| 가격식 + 실거래 보정 | 10.6% | 76.4% |

| 신뢰도 구간 | 말한 신뢰도 | 실제로 20% 이내 | 가격 구간 |
|---|---|---|---|
| 0.6 미만 | 0.47 | 0.53 | ±51% |
| 0.6 ~ 0.7 | 0.66 | 0.60 | ±36% |
| 0.7 ~ 0.8 | 0.76 | 0.74 | ±25% |
| 0.8 이상 | 0.84 | 0.87 | ±19% |

조건별 오차, 크게 틀린 사례, 예시 산출 3건은 [reports/validation.md](reports/validation.md)에 있다.

## 다시 만들기

데이터 수집부터 리포트까지 순서대로 실행하면 같은 결과가 나온다. 수집에는 두 키가 모두 필요하다. 실거래가 수집은 `국토교통부_연립다세대 매매 실거래가 자료` 활용신청이 추가로 필요하다.

```bash
uv run --env-file .env python collect/fetch_trades.py --start 202110 --end 202610
uv run python -m model.clean
uv run --env-file .env python -m collect.geocode_trades
uv run python validate.py --save reports/holdout.csv
uv run python -m model.confidence
uv run python report.py
uv run pytest
```

## 한계

- 직거래는 시세보다 낮게 거래되는 경우가 많은데, 입력으로는 거래 방식을 알 수 없다. 크게 틀린 사례의 38.5%가 직거래다.
- 지하층(중앙값 오차율 19.8%)과 준공 30년 이상(14.3%)에서 오차가 크다. 재개발 기대처럼 실거래가에 없는 요인이 가격을 움직인다.
- 준공 1년 이내 신축은 12.2% 높게 추정하는 경향이 있다.
- 추정은 기준일 시세다. 9~12개월 전 거래와 비교하면 그사이 시세 변화만큼 차이가 난다.
- 면적·준공년도를 어림한 경우의 신뢰도 감점은 검증 표본이 없어 규칙으로 정했다.
