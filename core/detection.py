"""
detection.py
다변량 시계열에 대해 Darts의 ForecastingAnomalyModel을 컴포넌트(변수)별로 적용하고,
NormScorer / KMeansScorer / WassersteinScorer 세 가지 점수 계산기로 이상 점수를 산출한다.

Scorers: NormScorer(예측오차) / KMeansScorer(패턴) / WassersteinScorer(분포) 기반 이상 점수 계산.

현재 설치된 Darts 버전은 ForecastingAnomalyModel에 GlobalForecastingModel만 허용하므로
(LocalForecastingModel인 ExponentialSmoothing 등은 사용 불가) LightGBMModel을 기본 예측기로 사용한다.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.ad import (
    KMeansScorer,
    NormScorer,
    QuantileDetector,
    WassersteinScorer,
    ForecastingAnomalyModel,
)
from darts.dataprocessing.transformers import Scaler as DartsScaler
from darts.models import LightGBMModel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

SCORER_REGISTRY = {
    "norm": {
        "label": "NormScorer (예측오차 크기)",
        "build": lambda window: NormScorer(),
        "description": "예측값과 실제값의 차이(절댓값)만으로 이상을 판단합니다. 가장 단순하고 해석이 쉬운 지표입니다.",
    },
    "kmeans": {
        "label": "KMeansScorer (패턴 군집)",
        "build": lambda window: KMeansScorer(k=2, window=window),
        "description": "오차의 패턴을 클러스터링하여 판단합니다. 값의 크기뿐 아니라 패턴 이상 탐지에 유리합니다.",
    },
    "wasserstein": {
        "label": "WassersteinScorer (분포 변화)",
        "build": lambda window: WassersteinScorer(window=window),
        "description": "구간별 분포 간 거리를 측정합니다. 전반적인 분포 변화를 감지하는 데 최적화되어 있습니다.",
    },
}


@dataclass
class ComponentResult:
    name: str
    train: TimeSeries
    test: TimeSeries
    scores: dict[str, TimeSeries]          # scorer_key -> anomaly score TimeSeries (test 구간)
    binary: dict[str, TimeSeries]          # scorer_key -> 0/1 이상 판정 TimeSeries
    threshold_quantile: float


@dataclass
class DetectionResult:
    components: dict[str, ComponentResult]
    combined_binary: TimeSeries            # 변수 전체를 OR로 합친 최종 이상 판정 (전체 길이 기준)
    test_start: pd.Timestamp
    scorer_keys: list[str]
    notes: list[str] = field(default_factory=list)


def _decide_lags_and_window(n_train: int) -> tuple[int, int]:
    """데이터 길이에 따라 LightGBM lag 수와 Scorer window 크기를 안전하게 결정한다."""
    lags = max(2, min(14, n_train // 4))
    window = max(2, min(20, n_train // 6))
    return lags, window


def run_multivariate_detection(
    df: pd.DataFrame,
    test_ratio: float = 0.5,
    scorer_keys: list[str] | None = None,
    threshold_quantile: float = 0.95,
) -> DetectionResult:
    """다변량 DataFrame을 받아 컴포넌트별 이상탐지를 수행한다.

    Parameters
    ----------
    df : 시간 인덱스를 가진 수치형 DataFrame (라벨 컬럼은 제외된 상태여야 함)
    test_ratio : 전체 데이터 중 이상탐지를 수행할 비율(뒷부분). 앞부분은 "정상 패턴 학습용"으로 사용.
    scorer_keys : 사용할 scorer 종류 목록. None이면 전부 사용.
    threshold_quantile : 이상 여부를 가르는 점수 분위수 (예: 0.95 -> 상위 5%를 이상으로 판정)
    """
    if scorer_keys is None:
        scorer_keys = list(SCORER_REGISTRY.keys())

    notes: list[str] = []

    # 실제 센서 데이터는 간격이 완벽히 일정하지 않은 경우가 흔하다 (예: 1초 주기인데 일부 구간만 2초로 튐).
    # Darts는 이런 경우 freq를 강제하지 않으면 에러를 내므로, 먼저 시도하고 실패하면
    # 추론된 빈도로 리샘플링(빈틈은 자동 보강)하여 재시도한다. 기존 호출 방식(파라미터 없이 호출)은 그대로 동작한다.
    try:
        full_ts = TimeSeries.from_dataframe(df, fill_missing_dates=False)
    except ValueError:
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is None:
            diffs = df.index.to_series().diff().dropna()
            if len(diffs) > 0:
                inferred_freq = pd.tseries.frequencies.to_offset(diffs.mode().iloc[0]).freqstr
        if inferred_freq is None:
            raise ValueError(
                "시간 인덱스의 주기를 추론할 수 없어 이상탐지를 진행할 수 없습니다. "
                "시간 간격이 너무 불규칙한 데이터일 수 있습니다."
            )
        notes.append(
            f"시간 간격이 완전히 일정하지 않아(예: 일부 구간 결측) 추정 주기 '{inferred_freq}' 기준으로 "
            "빈 시점을 보강한 뒤 분석을 진행했습니다."
        )
        full_ts = TimeSeries.from_dataframe(df, fill_missing_dates=True, freq=inferred_freq)
        # 리샘플링 중 생긴 결측치를 다시 보강 (선형 보간)
        if np.isnan(full_ts.values()).any():
            from darts.dataprocessing.transformers import MissingValuesFiller
            full_ts = MissingValuesFiller().transform(full_ts)

    n_total = len(full_ts)
    if n_total < 20:
        raise ValueError(
            f"이상탐지를 안정적으로 수행하기에 데이터가 너무 적습니다 (현재 {n_total}행). "
            "최소 20행 이상의 데이터를 권장합니다."
        )

    train_ratio = 1 - test_ratio
    train_full, test_full = full_ts.split_before(train_ratio)
    n_train = len(train_full)

    lags, window = _decide_lags_and_window(n_train)
    if n_train < lags + 5:
        notes.append(
            "학습 구간이 짧아 이상탐지 정밀도가 떨어질 수 있습니다. "
            "가능하면 더 긴 시계열을 사용해 주세요."
        )

    components: dict[str, ComponentResult] = {}
    comp_or_list: list[TimeSeries] = []
    test_index = test_full.time_index

    for comp_name in df.columns:
        uni_full = full_ts[comp_name]
        uni_train, uni_test = uni_full.split_before(train_ratio)

        if np.isnan(uni_train.values()).any() or np.isnan(uni_test.values()).any():
            notes.append(
                f"[{comp_name}] 전처리 후에도 결측치가 남아 있어 이 변수는 이상탐지에서 제외했습니다."
            )
            components[comp_name] = ComponentResult(
                name=comp_name, train=uni_train, test=uni_test,
                scores={}, binary={}, threshold_quantile=threshold_quantile,
            )
            continue

        # 변수별 스케일링 (스케일 차이가 큰 다변량에서도 Scorer가 안정적으로 동작하도록)
        scaler = DartsScaler(StandardScaler())
        uni_train_scaled = scaler.fit_transform(uni_train)
        uni_test_scaled = scaler.transform(uni_test)

        model = LightGBMModel(lags=lags, output_chunk_length=1, verbose=-1, random_state=42)

        scores: dict[str, TimeSeries] = {}
        binary: dict[str, TimeSeries] = {}

        for key in scorer_keys:
            scorer = SCORER_REGISTRY[key]["build"](window)
            anomaly_model = ForecastingAnomalyModel(model=model, scorer=[scorer])
            try:
                anomaly_model.fit(uni_train_scaled, allow_model_training=True)
                score_ts = anomaly_model.score(uni_test_scaled)
                if isinstance(score_ts, (list, tuple)):
                    score_ts = score_ts[0]
            except Exception as exc:  # noqa: BLE001
                notes.append(f"[{comp_name}] {SCORER_REGISTRY[key]['label']} 계산 실패: {exc}")
                continue

            scores[key] = score_ts

            detector = QuantileDetector(high_quantile=threshold_quantile)
            detector.fit(score_ts)
            binary_ts = detector.detect(score_ts)
            binary[key] = binary_ts

        components[comp_name] = ComponentResult(
            name=comp_name,
            train=uni_train,
            test=uni_test,
            scores=scores,
            binary=binary,
            threshold_quantile=threshold_quantile,
        )

        # 컴포넌트 내 scorer들을 OR로 통합 (하나라도 이상이면 이상).
        # window 기반 scorer(KMeans/Wasserstein)는 앞부분 일부가 깎여 NormScorer보다 짧으므로,
        # 먼저 모든 scorer 결과를 공통 시간 구간(교집합)으로 정렬한 뒤 numpy max(OR)로 합친다.
        if binary:
            binary_list = list(binary.values())
            common_start = max(b.start_time() for b in binary_list)
            common_end = min(b.end_time() for b in binary_list)
            aligned = [b.slice(common_start, common_end) for b in binary_list]
            stacked = np.stack([b.values().flatten() for b in aligned], axis=0)
            comp_or = stacked.max(axis=0).astype(int)
            comp_or_ts = TimeSeries.from_times_and_values(
                aligned[0].time_index, comp_or.reshape(-1, 1), columns=[comp_name]
            )
            comp_or_list.append(comp_or_ts)

    # 모든 컴포넌트의 OR 결과를 다시 공통 구간으로 정렬한 뒤, 변수 간에도 OR로 통합한다
    # (어느 변수든 하나라도 이상으로 판정되면 해당 시점을 전체 이상으로 표시).
    if comp_or_list:
        common_start = max(c.start_time() for c in comp_or_list)
        common_end = min(c.end_time() for c in comp_or_list)
        aligned_components = [c.slice(common_start, common_end) for c in comp_or_list]
        stacked_components = np.stack(
            [c.values().flatten() for c in aligned_components], axis=0
        )
        combined_binary_arr = stacked_components.max(axis=0).astype(int)
        combined_index = aligned_components[0].time_index
    else:
        combined_binary_arr = np.zeros(len(test_index), dtype=int)
        combined_index = test_index

    combined_binary = TimeSeries.from_times_and_values(
        combined_index, combined_binary_arr.reshape(-1, 1), columns=["any_anomaly"]
    )

    return DetectionResult(
        components=components,
        combined_binary=combined_binary,
        test_start=test_index[0],
        scorer_keys=scorer_keys,
        notes=notes,
    )
