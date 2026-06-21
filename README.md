# AnomaLens — 다변량 시계열 이상탐지 대시보드

임의의 다변량 시계열 CSV 파일을 업로드하면 자동으로 데이터 품질 분석 → 시계열 분해(STL) →
정상성 검정(ADF) → 이상탐지 → 평가지표 시각화까지 수행하는 Streamlit 웹앱입니다.

## 핵심 기능

**기본 파이프라인**
- **완전 자동 분석**: 시간 컬럼, 라벨 컬럼을 자동으로 인식하여 별도 설정 없이 바로 분석
- **파일 교체 시 자동 재분석**: 업로드 파일의 해시값을 캐시 키로 사용하여, 새 파일을 올리면 즉시 새로운 이상탐지가 수행됩니다
- **다변량 지원**: 변수별로 독립적인 이상탐지 모델을 학습하고, 결과를 통합(OR)하여 보여줍니다
- **내장 데이터셋 1-클릭 체험**: 사이드바의 GECCO 2019 카드를 누르면 업로드 없이 바로 샘플 데이터로 전체 파이프라인을 체험할 수 있습니다

**시계열 분석**
- 🗂️ **데이터 품질 분석**: 행/열 수, 분석 기간, 추정 주기, 결측치/중복 통계, 변수별 기술통계
- 📐 **STL 분해**: 각 변수를 추세(Trend)/계절성(Seasonal)/잔차(Residual)로 분리. 계절 주기는 ACF로
  변수마다 자동 추정되며, 직접 입력해 조정할 수도 있습니다
- 📏 **정상성 분석**: ADF 검정 + Ljung-Box 검정으로 정상성 여부 판정

**이상탐지 및 평가**
- **다중 알고리즘 비교**: NormScorer / KMeansScorer / WassersteinScorer (Darts 기반) + Isolation Forest (scikit-learn 기반)
- **정량 평가**: 정답 라벨이 있으면 AUC-ROC, AUC-PR, Precision/Recall/F1 (`is_anomaly` 라벨이 있으면 모델 탐지 비율과 실제 이상 비율을 함께 표시)
- **비지도 평가**: 라벨이 없으면 점수 분포, 이상/정상 분리도 등 보조 지표
- 💡 **이상 원인 자동 설명**: 상위 이상 시점에 대해 평균 대비 시그마, 변화율, 동시 탐지 지표를 자동으로 설명
- 🔁 **Scorer 일치도 분석**: 서로 다른 지표가 같은 시점을 이상으로 판단하는 비율
- 🔍 **자동 확대**: 이상탐지 결과 탭에서 주요 이상 구간 주변으로 그래프가 자동으로 확대됩니다

## 기술 스택

- **웹 프레임워크**: Streamlit
- **이상탐지 엔진**: [Darts](https://unit8co.github.io/darts/) `ForecastingAnomalyModel`(LightGBM) + `NormScorer`/`KMeansScorer`/`WassersteinScorer`, scikit-learn `IsolationForest`
- **시계열 분석**: statsmodels (`STL`, `adfuller`, `acorr_ljungbox`)
- **시각화**: Plotly

## 샘플 데이터 (실제 공개 벤치마크)

샘플은 합성 데이터가 아니라 **출처가 명확한 실제 공개 이상탐지 벤치마크**입니다. 자세한 내용은
`sample_data/ATTRIBUTION.md`를 참고하세요.

| 파일 | 출처 | 설명 |
|---|---|---|
| `gecco2019_sample.csv` | [GECCO Industrial Challenge 2019](https://www.th-koeln.de/informatik-und-ingenieurwissenschaften/gecco-2019-industrial-challenge-detecting-anomalies-in-drinking-water-quality_63959.php) (TH Köln 주최, 비상업적 연구/교육 목적 사용 가능) | 음료수 정수 처리 공정의 수질 센서 6개(수온/산도/전기전도도/탁도/UV흡광도/유량) 다변량 시계열, 6,000행(1분 간격). 실제 이상 라벨 포함, 이상 비율 ~0.95% |

## 로컬 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속합니다.

## 사용 방법

1. 왼쪽 사이드바에서 CSV 파일을 업로드합니다. (또는 "내장 데이터셋" 카드의 GECCO 2019로 바로 체험)
2. 시간 컬럼과 변수 컬럼이 자동으로 인식됩니다.
3. 탐지 구간 비율, 사용할 Scorer, 이상 판정 임계값, Isolation Forest 비교 여부를 조절합니다.
4. 8개 탭(이상탐지 결과·Scorer 비교·정량 평가·일치도 분석·이상 원인 설명·데이터 품질·STL 분해·정상성 검정)을 이동하며 결과를 확인합니다.

### CSV 파일 요구사항

- 시간/날짜 컬럼 1개 (이름은 자유: `timestamp`, `date`, `날짜` 등 자동 인식. 못 찾으면 행 순서를 시간으로 간주)
- 수치형 변수 컬럼 1개 이상 (다변량 권장)
- (선택) 정답 라벨 컬럼: `label`, `is_anomaly`, `anomaly` 등의 이름이거나 0/1 값으로 구성되면 자동 인식되어 정량 평가에 사용됩니다
- 최소 20행 이상 권장 (STL 분해는 최소 8개 시점 필요)

## 배포 (Streamlit Community Cloud)

1. 이 폴더를 GitHub 저장소에 푸시합니다 (`requirements.txt`, `app.py`, `core/`, `sample_data/` 포함).
2. [share.streamlit.io](https://share.streamlit.io)에 접속하여 GitHub 계정으로 로그인합니다.
3. "New app" → 저장소/브랜치/`app.py` 경로를 지정하고 Deploy를 누릅니다.
4. 몇 분 내 `https://[앱이름].streamlit.app` 형태의 공개 URL이 생성됩니다.

## 프로젝트 구조

```
anomaly_app/
├── app.py                          # Streamlit 메인 앱 (UI + 시각화, 8개 탭)
├── core/
│   ├── data_loader.py              # CSV 자동 파싱 (시간/라벨 컬럼 추론)
│   ├── preprocessing.py            # 결측치 처리
│   ├── detection.py                # Darts 기반 이상탐지 엔진 (NormScorer/KMeans/Wasserstein)
│   ├── isolation_forest_detection.py  # Isolation Forest 기반 이상탐지
│   ├── evaluation.py               # 평가지표 계산 (라벨 유/무 모두 지원)
│   ├── quality.py                  # 데이터 품질 분석
│   ├── decomposition.py            # STL 분해 + ACF 기반 계절 주기 자동 추정(estimate_period)
│   ├── stationarity.py             # ADF/Ljung-Box 정상성 검정
│   ├── explain.py                  # 이상 원인 자동 설명
│   └── report.py                   # HTML 리포트 생성 (현재 app.py에서는 사용하지 않음)
├── sample_data/                    # GECCO 2019 수질 이상탐지 샘플 + 출처 문서(ATTRIBUTION.md)
├── scripts/make_sample_data.py     # 샘플 데이터 가공 스크립트 예시 (현재 샘플과는 별개)
├── requirements.txt
└── .streamlit/config.toml          # 테마 설정
```

## 알고리즘 설명 (요약)

**Darts 기반 (예측 오차 기반)**
1. 각 변수를 학습 구간(정상 패턴)과 탐지 구간으로 분리
2. LightGBM 기반 예측 모델로 다음 시점을 예측
3. 예측값과 실제값의 차이를 기준으로 3가지 Scorer가 이상 점수를 계산
4. `QuantileDetector`로 점수 상위 N%를 이상으로 판정, 변수/Scorer 간 결과를 OR로 통합

**Isolation Forest (트리 기반 비지도 학습)**
1. 각 시점 주변 슬라이딩 윈도우 통계(이동평균과의 편차, 이동표준편차, 1차 차분)를 특징으로 추출
2. 다수의 랜덤 트리로 데이터를 분리하는 데 필요한 분리 횟수가 적을수록 이상치로 판정

**STL 분해 / ADF 검정**으로 이상탐지 이전에 데이터 자체의 추세·계절성·정상성 특성을
먼저 파악할 수 있습니다. STL의 계절 주기는 ACF(자기상관함수)에서 가장 두드러진 피크를
찾아 변수마다 자동으로 추정되며, STL 분해 탭에서 직접 값을 조정할 수도 있습니다.

