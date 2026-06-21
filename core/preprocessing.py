"""
preprocessing.py
시계열 전처리: 결측치 보강, 중복 시간 인덱스 정리, 정규화.

원칙:
- 결측치는 절대 단순 삭제(drop)하지 않고, 주변 데이터를 이용해 보강(Imputation)한다.
- 분석 목적(이상탐지)이므로 이상치/극단치는 제거하지 않고 그대로 유지한다
  (이상탐지가 목적일 때는 원본 이상치를 변형하면 안 된다.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    n_missing_before: int
    n_missing_after: int
    method_used: str
    notes: list[str]


def reindex_to_regular_grid(df: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    """불규칙한 시간 인덱스를 일정한 간격의 그리드로 재배열한다.

    freq를 추론하지 못한 경우, 원본 순서를 그대로 유지한다 (그리드화하지 않음).
    """
    if freq is None:
        return df
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    return df.reindex(full_index)


def fill_missing_values(
    df: pd.DataFrame,
    method: str = "interpolate_linear",
) -> PreprocessResult:
    """결측치를 채운다.

    method 옵션:
    - "interpolate_linear": 선형 보간 (기본, 가장 무난함)
    - "ffill": 직전 값으로 채움 (LOCF)
    - "bfill": 직후 값으로 채움 (NOCB)
    - "moving_average": 이동평균 기반 대체
    """
    notes: list[str] = []
    n_missing_before = int(df.isna().sum().sum())

    if n_missing_before == 0:
        return PreprocessResult(
            df=df, n_missing_before=0, n_missing_after=0,
            method_used="없음 (결측치 없음)", notes=notes,
        )

    if method == "interpolate_linear":
        filled = df.interpolate(method="linear", limit_direction="both")
        method_label = "선형 보간 (Interpolation)"
    elif method == "ffill":
        filled = df.ffill().bfill()  # 맨 앞 결측은 bfill로 보완
        method_label = "직전값 유지 (LOCF)"
    elif method == "bfill":
        filled = df.bfill().ffill()
        method_label = "직후값 유지 (NOCB)"
    elif method == "moving_average":
        filled = df.copy()
        for col in df.columns:
            filled[col] = df[col].fillna(
                df[col].rolling(window=5, min_periods=1, center=True).mean()
            )
        filled = filled.interpolate(method="linear", limit_direction="both")
        method_label = "이동평균 기반 대체"
    else:
        raise ValueError(f"알 수 없는 결측치 처리 방법: {method}")

    n_missing_after = int(filled.isna().sum().sum())
    if n_missing_after > 0:
        # 여전히 남아있다면 (예: 컬럼 전체가 NaN) 0으로 최종 보강
        filled = filled.fillna(0.0)
        notes.append(
            "일부 컬럼은 전체가 결측이라 보간이 불가능해 0으로 채웠습니다. "
            "해당 변수의 분석 결과는 참고용으로만 활용하세요."
        )

    notes.insert(
        0,
        f"결측치 {n_missing_before:,}개를 '{method_label}' 방식으로 채웠습니다.",
    )

    return PreprocessResult(
        df=filled,
        n_missing_before=n_missing_before,
        n_missing_after=int(filled.isna().sum().sum()),
        method_used=method_label,
        notes=notes,
    )


def remove_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """분산이 0인(상수) 컬럼은 이상탐지 모델 학습에 방해가 되므로 제외한다."""
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    return df, constant_cols
