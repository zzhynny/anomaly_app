# 샘플 데이터 출처 (Attribution)

샘플은 합성 데이터가 아닌, 공개된 시계열 이상탐지 벤치마크에서 가져온 실제 데이터입니다.
원본 값은 변형하지 않았으며, 컬럼명과 형식만 정리했습니다.

---

## gecco2019_sample.csv

- **출처**: GECCO Industrial Challenge 2019 — Water Quality Anomaly Detection
  (Genetic and Evolutionary Computation Conference 2019 산업 챌린지)
- **주최**: TH Köln (Technische Hochschule Köln)
- **공식 페이지**: https://www.th-koeln.de/informatik-und-ingenieurwissenschaften/gecco-2019-industrial-challenge-detecting-anomalies-in-drinking-water-quality_63959.php
- **라이선스**: 공개 대회 데이터 (비상업적 연구/교육 목적 사용 가능)
- **설명**: 음료수 정수 처리 공정에서 수집한 6개 수질 센서의 다변량 시계열
  - `temperature_c`(수온), `ph`(산도), `conductivity_ms`(전기전도도),
    `turbidity_ntu`(탁도), `uv_absorption`(UV 흡광도), `flow_rate_lmin`(유량)
  - 6,000행, 1분 간격 (2017-07-22 ~ 2017-07-26)
  - 실제 이상 라벨(`is_anomaly`) 포함, 이상 비율 약 0.95% (57건)

## 가공 내용

- 원본 데이터에서 이상 구간이 앱의 기본 탐지 구간(뒤쪽 30%) 안에 포함되도록 구간을
  잘라 슬라이싱했습니다 — 그래야 AUC-ROC/Precision/Recall 같은 정량 평가가 실제로
  의미 있게 계산됩니다.
- 컬럼명을 영문 소문자 + 단위 접미사로 정리했습니다 (값 자체는 변형 없음).
