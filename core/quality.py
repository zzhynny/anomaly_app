"""
quality.py
업로드된 시계열의 기초 품질 통계를 계산한다 (신규 모듈, 기존 기능에 영향 없음).

제공 정보:
- 총 행 수, 총 컬럼 수, 분석 기간(시작일/종료일), 추정 주기
- 결측치 개수/비율, 중복 데이터 수
- 변수별 기술통계 (평균, 표준편차, 최소/최대, 분위수 등)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    n_rows: int
    n_cols: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_str: str
    freq_guess: str | None
    n_missing_total: int
    missing_ratio_total: float
    missing_by_column: pd.Series           # 컬럼별 결측치 개수
    n_duplicated_timestamps: int
    descriptive_stats: pd.DataFrame         # 변수별 기술통계 (count, mean, std, min, 25%, 50%, 75%, max)


def _format_duration(start: pd.Timestamp, end: pd.Timestamp) -> str:
    delta = end - start
    days = delta.days
    if days >= 365:
        return f"약 {days / 365:.1f}년 ({days:,}일)"
    if days >= 1:
        return f"{days:,}일"
    hours = delta.total_seconds() / 3600
    return f"약 {hours:.1f}시간"


def analyze_quality(
    raw_df_with_possible_dupes_index: pd.DataFrame,
    clean_df: pd.DataFrame,
    freq_guess: str | None,
) -> QualityReport:
    """데이터 품질 리포트를 생성한다.

    Parameters
    ----------
    raw_df_with_possible_dupes_index : 전처리(결측치 보강) 이전의 원본 형태 데이터프레임.
        결측치/중복 통계는 이 원본 기준으로 계산해야 사용자가 실제 데이터 상태를 알 수 있다.
    clean_df : 전처리가 끝난 데이터프레임 (기술통계는 이 기준으로 계산해 분석에 실제로
        사용되는 값의 분포를 보여준다).
    freq_guess : data_loader가 추론한 주기 문자열.
    """
    n_rows = len(raw_df_with_possible_dupes_index)
    n_cols = raw_df_with_possible_dupes_index.shape[1]

    start_date = raw_df_with_possible_dupes_index.index.min()
    end_date = raw_df_with_possible_dupes_index.index.max()

    n_missing_total = int(raw_df_with_possible_dupes_index.isna().sum().sum())
    total_cells = n_rows * n_cols if n_cols > 0 else 1
    missing_ratio_total = n_missing_total / total_cells if total_cells else 0.0
    missing_by_column = raw_df_with_possible_dupes_index.isna().sum()

    n_duplicated_timestamps = int(raw_df_with_possible_dupes_index.index.duplicated().sum())

    descriptive_stats = clean_df.describe().T
    descriptive_stats["missing(전처리전)"] = missing_by_column.reindex(descriptive_stats.index)

    return QualityReport(
        n_rows=n_rows,
        n_cols=n_cols,
        start_date=start_date,
        end_date=end_date,
        duration_str=_format_duration(start_date, end_date),
        freq_guess=freq_guess,
        n_missing_total=n_missing_total,
        missing_ratio_total=missing_ratio_total,
        missing_by_column=missing_by_column,
        n_duplicated_timestamps=n_duplicated_timestamps,
        descriptive_stats=descriptive_stats,
    )
