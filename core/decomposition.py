"""
decomposition.py
STL(Seasonal and Trend decomposition using Loess)을
각 변수에 적용하여 추세(Trend), 계절성(Seasonal), 잔차(Residual) 성분으로 분해한다.

신규 모듈이며 기존 detection/evaluation 로직과는 독립적으로 동작한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# 자주 쓰이는 pandas freq 코드 -> STL에 넘길 기본 계절 주기(period, 데이터 포인트 수) 매핑.
# 예: 일별(D) 데이터는 7일(주간) 주기를, 시간별(h) 데이터는 24시간 주기를 기본으로 가정한다.
FREQ_TO_DEFAULT_PERIOD = {
    "D": 7,
    "B": 5,
    "h": 24,
    "H": 24,
    "min": 60,
    "T": 60,
    "s": 60,
    "S": 60,
    "W": 52,
    "M": 12,
    "ME": 12,
    "MS": 12,
    "Q": 4,
    "QS": 4,
}


@dataclass
class STLResult:
    column: str
    period: int
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    observed: pd.Series
    trend_strength: float      # 추세 강도 (0~1, 1에 가까울수록 추세가 강함)
    seasonal_strength: float   # 계절성 강도 (0~1)


def guess_period_from_freq(freq_guess: str | None, n_obs: int) -> int | None:
    """추론된 데이터 주기(freq)로부터 STL에 사용할 계절 주기(period, 정수)를 결정한다.

    freq가 알려져 있는데도 데이터 길이가 그 계절 주기의 2배에 못 미치면, 주기를 억지로
    줄여서 맞추지 않고 None을 반환해 "분석 불가"임을 호출자에게 알린다.
    """
    if freq_guess:
        base_freq = "".join(ch for ch in freq_guess if not ch.isdigit())
        if base_freq in FREQ_TO_DEFAULT_PERIOD:
            period = FREQ_TO_DEFAULT_PERIOD[base_freq]
            if n_obs >= period * 2:
                return period
            return None
    # freq를 모르는 경우에는 데이터 길이의 1/10 정도를 기본 주기로 근사한다 (최소 2).
    fallback = max(2, n_obs // 10)
    period = min(fallback, max(2, n_obs // 2 - 1))
    return period if n_obs >= period * 2 else None


def estimate_period(series, max_lag=200):
    """ACF(자기상관함수)에서 가장 두드러진 주기적 피크를 찾아 계절 주기를 데이터 기반으로 추정한다."""
    from statsmodels.tsa.stattools import acf
    try:
        series_clean = series.dropna()
        if len(series_clean) < max_lag * 2:
            return max(2, len(series_clean) // 10)
        acf_vals = acf(series_clean, nlags=max_lag, fft=True)
        acf_vals[0] = 0
        peaks = []
        for i in range(2, len(acf_vals) - 1):
            if acf_vals[i] > acf_vals[i-1] and acf_vals[i] > acf_vals[i+1]:
                peaks.append((i, acf_vals[i]))
        if peaks:
            return max(2, max(peaks, key=lambda x: x[1])[0])
        return 12
    except:
        return 12


def _measure_trend_strength(observed: np.ndarray, trend: np.ndarray, resid: np.ndarray) -> float:
    """Var(resid) / Var(detrend) 비율 기반 추세 강도. 1에 가까울수록 추세가 강하다."""
    detrended_var = np.var(observed - trend + resid)
    if detrended_var <= 1e-12:
        return 0.0
    strength = 1 - (np.var(resid) / detrended_var)
    return float(np.clip(strength, 0, 1))


def _measure_seasonal_strength(observed: np.ndarray, seasonal: np.ndarray, resid: np.ndarray) -> float:
    deseasonalized_var = np.var(observed - seasonal + resid)
    if deseasonalized_var <= 1e-12:
        return 0.0
    strength = 1 - (np.var(resid) / deseasonalized_var)
    return float(np.clip(strength, 0, 1))


def decompose_series(
    series: pd.Series,
    freq_guess: str | None,
    period_override: int | None = None,
    robust: bool = False,
) -> STLResult:
    """단일 변수(Series)에 STL 분해를 적용한다."""
    clean = series.dropna()
    n_obs = len(clean)
    if n_obs < 8:
        raise ValueError("데이터 길이가 충분하지 않아 계절성 분석을 생략했습니다.")

    period = period_override or guess_period_from_freq(freq_guess, n_obs)
    if period is None:
        raise ValueError("데이터 길이가 충분하지 않아 계절성 분석을 생략했습니다.")

    stl = STL(clean, period=period, robust=robust)
    fit = stl.fit()

    trend_strength = _measure_trend_strength(clean.values, fit.trend.values, fit.resid.values)
    seasonal_strength = _measure_seasonal_strength(clean.values, fit.seasonal.values, fit.resid.values)

    return STLResult(
        column=str(series.name),
        period=period,
        trend=fit.trend,
        seasonal=fit.seasonal,
        resid=fit.resid,
        observed=clean,
        trend_strength=trend_strength,
        seasonal_strength=seasonal_strength,
    )


def decompose_dataframe(
    df: pd.DataFrame,
    freq_guess: str | None,
    period_override: int | None = None,
) -> dict[str, STLResult]:
    """다변량 DataFrame의 모든 컬럼에 STL 분해를 적용한다. 실패한 컬럼은 결과에서 제외한다."""
    results: dict[str, STLResult] = {}
    for col in df.columns:
        try:
            results[col] = decompose_series(df[col], freq_guess, period_override)
        except Exception:
            continue
    return results
