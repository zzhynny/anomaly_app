"""
stationarity.py
정상성 검정을 구현한다.

- ADF 검정 (Augmented Dickey-Fuller test): 단위근 존재 여부 -> 정상성 판정
  귀무가설(H0): 단위근이 존재한다 = 비정상성이다
  p-value < 0.05 이면 귀무가설 기각 -> 정상성 시계열로 판정
- Ljung-Box 검정: 데이터가 백색잡음(자기상관 없음)인지 확인하는 보조 지표
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


@dataclass
class StationarityResult:
    column: str
    adf_statistic: float
    adf_pvalue: float
    adf_critical_values: dict[str, float]
    is_stationary: bool                 # ADF 기준 (p < 0.05)
    n_lags_used: int
    ljungbox_pvalue: float | None       # 시차 10 기준 Ljung-Box p-value
    has_autocorrelation: bool | None    # Ljung-Box p < 0.05 -> 자기상관 존재(패턴 있음)
    interpretation: str                 # 사람이 읽을 수 있는 한 줄 해석


def _interpret(is_stationary: bool, has_autocorrelation: bool | None) -> str:
    if is_stationary:
        base = "정상성 시계열로 판정됩니다 (평균/분산이 시간에 따라 일정). 차분 없이 통계 모형을 적용할 수 있습니다."
    else:
        base = "비정상성 시계열로 판정됩니다 (추세 또는 분산 변화 존재). 차분(differencing) 등으로 정상화 후 분석을 권장합니다."
    if has_autocorrelation is True:
        base += " 시차 간 자기상관이 존재해 예측 가능한 패턴이 있는 것으로 보입니다."
    elif has_autocorrelation is False:
        base += " 백색잡음에 가까워 추가 패턴을 찾기 어려울 수 있습니다."
    return base


def test_stationarity(series: pd.Series, ljungbox_lags: int = 10) -> StationarityResult:
    """단일 변수에 대해 ADF 검정(+ 보조로 Ljung-Box 검정)을 수행한다."""
    clean = series.dropna()
    if len(clean) < 10:
        raise ValueError("정상성 검정을 수행하기에 데이터가 너무 적습니다 (최소 10개 시점 필요).")

    adf_stat, adf_pvalue, n_lags_used, n_obs, critical_values, _ = adfuller(clean, autolag="AIC")
    is_stationary = adf_pvalue < 0.05

    ljungbox_pvalue = None
    has_autocorrelation = None
    try:
        lb_lags = min(ljungbox_lags, max(1, len(clean) // 5))
        lb_result = acorr_ljungbox(clean, lags=[lb_lags], return_df=True)
        ljungbox_pvalue = float(lb_result["lb_pvalue"].iloc[0])
        has_autocorrelation = ljungbox_pvalue < 0.05
    except Exception:
        pass

    return StationarityResult(
        column=str(series.name),
        adf_statistic=float(adf_stat),
        adf_pvalue=float(adf_pvalue),
        adf_critical_values={k: float(v) for k, v in critical_values.items()},
        is_stationary=is_stationary,
        n_lags_used=int(n_lags_used),
        ljungbox_pvalue=ljungbox_pvalue,
        has_autocorrelation=has_autocorrelation,
        interpretation=_interpret(is_stationary, has_autocorrelation),
    )


def test_stationarity_dataframe(df: pd.DataFrame) -> dict[str, StationarityResult]:
    """다변량 DataFrame의 모든 컬럼에 정상성 검정을 적용한다."""
    results: dict[str, StationarityResult] = {}
    for col in df.columns:
        try:
            results[col] = test_stationarity(df[col])
        except Exception:
            continue
    return results
