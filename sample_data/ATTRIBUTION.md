# 샘플 데이터 출처 (Attribution)

두 샘플 데이터 모두 합성 데이터가 아닌, 공개된 시계열 이상탐지 벤치마크에서 가져온 실제 데이터입니다.
원본 값은 변형하지 않았으며, 컬럼명과 형식만 정리했습니다.

---

## 1. smd_server_machine.csv

- **출처**: Server Machine Dataset (SMD)
- **논문**: Su et al., "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network", KDD 2019
- **GitHub**: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
- **라이선스**: MIT License
- **원본 파일**: `test/machine-1-1.txt` + `test_label/machine-1-1.txt`
- **설명**: 인터넷 대기업 서버 노드에서 수집한 38개 성능 지표 (CPU, 메모리, 네트워크 등)
  5주치 데이터 중 테스트 구간(약 20일), 분 단위, 이상 비율 ~9.5%

---

## 2. psm_server_metrics.csv

- **출처**: Pooled Server Metrics (PSM), eBay
- **논문**: Abdulaal & Lancewicki, "Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization", KDD 2021
- **GitHub**: https://github.com/eBay/RANSynCoders/tree/main/data
- **라이선스**: Apache License 2.0
- **원본 파일**: `test.csv` + `test_label.csv`
- **설명**: eBay 애플리케이션 서버 노드의 25개 KPI 지표 (분 단위)
  원본 87,841행 중 앞 5,000행, 이상 비율 ~6.6%

---

## 가공 내용

- SMD: 공백 구분자(txt) → 컬럼명 추가 + timestamp 컬럼 생성 + is_anomaly 라벨 병합
- PSM: 정수 분 단위 timestamp → datetime 변환 + is_anomaly 라벨 병합
- 값 자체는 변형 없음
