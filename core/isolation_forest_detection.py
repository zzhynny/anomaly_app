"""
isolation_forest_detection.py
Isolation Forest를 이용한 이상탐지.

기존 detection.py(Darts 기반 ForecastingAnomalyModel + Scorer 3종)와는 완전히 독립된 모듈로,
같은 변수에 대해 결과를 나란히 비교할 수 있도록 동일한 인터페이스(컴포넌트별 0/1 이진 결과 +
이상 점수)를 제공한다. 기존 Darts 파이프라인 코드는 전혀 수정하지 않는다.

방법론: 각 시점 주변의 슬라이딩 윈도우 통계(평균, 표준편차, 변화량 등)를 특징으로 추출하여
IsolationForest에 입력한다. 단순히 원본 값만 넣는 것보다 패턴/변화량 기반 이상도 함께 잡아낼 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestComponentResult:
    name: str
    anomaly_score: pd.Series      # 높을수록 이상 (양수로 정규화된 점수)
    binary: pd.Series             # 0/1 이상 판정
    contamination: float


@dataclass
class IsolationForestResult:
    components: dict[str, IsolationForestComponentResult]
    combined_binary: pd.Series    # 변수 중 하나라도 이상이면 1 (OR 통합)


def _build_window_features(series: pd.Series, window: int) -> pd.DataFrame:
    """슬라이딩 윈도우 기반 특징: 원본값, 이동평균과의 차이, 이동표준편차, 1차 차분."""
    rolling_mean = series.rolling(window=window, min_periods=1, center=True).mean()
    rolling_std = series.rolling(window=window, min_periods=1, center=True).std().fillna(0)
    diff = series.diff().fillna(0)

    features = pd.DataFrame(
        {
            "value": series,
            "dev_from_rolling_mean": series - rolling_mean,
            "rolling_std": rolling_std,
            "diff": diff,
        },
        index=series.index,
    )
    return features


def detect_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    window: int = 10,
    random_state: int = 42,
) -> IsolationForestResult:
    """다변량 DataFrame에 컴포넌트별로 Isolation Forest를 적용한다.

    Parameters
    ----------
    df : 시간 인덱스를 가진 수치형 DataFrame (결측치 없어야 함)
    contamination : 전체 데이터 중 이상치로 간주할 비율 추정값 (기존 Darts 파이프라인의
        threshold_quantile과 동일한 역할)
    window : 슬라이딩 윈도우 특징 추출에 사용할 윈도우 크기
    """
    components: dict[str, IsolationForestComponentResult] = {}
    combined_arr = np.zeros(len(df), dtype=int)

    for col in df.columns:
        series = df[col]
        if series.isna().any():
            continue

        features = _build_window_features(series, window)

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        model.fit(features.values)

        # decision_function: 낮을수록 이상치에 가까움 -> 부호를 뒤집어 "높을수록 이상"으로 통일
        raw_score = -model.decision_function(features.values)
        pred = model.predict(features.values)  # -1: 이상, 1: 정상
        binary = (pred == -1).astype(int)

        components[col] = IsolationForestComponentResult(
            name=col,
            anomaly_score=pd.Series(raw_score, index=df.index, name=col),
            binary=pd.Series(binary, index=df.index, name=col),
            contamination=contamination,
        )
        combined_arr = np.maximum(combined_arr, binary)

    combined_binary = pd.Series(combined_arr, index=df.index, name="any_anomaly_iforest")

    return IsolationForestResult(components=components, combined_binary=combined_binary)
