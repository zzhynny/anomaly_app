"""
data_loader.py
업로드된 CSV를 읽어 시간 인덱스를 자동으로 추론하고,
분석 가능한 pandas DataFrame 형태로 정리한다.

설계 원칙:
- 사용자는 "시간 컬럼 이름"이나 "포맷"을 미리 알 필요가 없어야 한다 (완전 자동화 요구사항).
- 다변량(2개 이상 수치형 컬럼)을 기본으로 다루되, 단변량 CSV가 들어와도 동작해야 한다.
- 라벨(정답 이상 여부) 컬럼이 있을 수 있으므로 이를 분리해서 인식한다.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd


LABEL_NAME_HINTS = {
    "label", "labels", "anomaly", "anomalies", "is_anomaly",
    "target", "y", "ground_truth", "fault", "outlier", "정상", "이상",
    "이상여부", "라벨",
}


@dataclass
class LoadResult:
    df: pd.DataFrame                 # 시간 인덱스 + 수치형 변수만 남은 데이터프레임
    time_col: str | None             # 원본에서 사용된 시간 컬럼명 (없었으면 None -> 합성)
    label_col: str | None            # 라벨 컬럼명 (있으면)
    dropped_cols: list[str] = field(default_factory=list)  # 수치형이 아니라 제외된 컬럼
    freq_guess: str | None = None    # 추론된 주기 (D, H, min 등)
    warnings: list[str] = field(default_factory=list)


def _looks_like_label(col_name: str, series: pd.Series) -> bool:
    """컬럼이 '정답 이상 라벨'일 가능성이 높은지 판단.

    컬럼명에 라벨 관련 키워드가 있으면 확실한 라벨 후보로 간주한다.
    값만으로 판단하는 경우(0/1 이진 컬럼)에는, 전체가 0인 경우는 제외한다
    (SMD처럼 실제 측정값이 우연히 모두 0인 컬럼이 라벨로 오인식되는 것을 방지).
    """
    name_lower = str(col_name).strip().lower()
    if name_lower in LABEL_NAME_HINTS:
        return True
    # 값이 0/1 이진인 경우에만 추가 후보로 간주하되,
    # 전체가 0(또는 1)인 경우는 측정값일 가능성이 높으므로 제외한다.
    unique_vals = series.dropna().unique()
    if (len(unique_vals) == 2
            and pd.Series(unique_vals).isin([0, 1, True, False]).all()
            and 0 in unique_vals and 1 in unique_vals):
        return True
    return False


def _try_parse_datetime_column(series: pd.Series, allow_numeric: bool = False) -> pd.Series | None:
    """주어진 시리즈를 datetime으로 변환 시도. 실패율이 높으면 None.

    숫자형(int/float) 컬럼은 기본적으로 시간 파싱 후보에서 제외한다.
    pandas.to_datetime이 순수 정수를 epoch 나노초로 잘못 해석해
    일반 측정값 컬럼(예: 1,4,7)을 시간 컬럼으로 오인하는 것을 막기 위함이다.
    """
    if pd.api.types.is_numeric_dtype(series) and not allow_numeric:
        return None
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    success_ratio = parsed.notna().mean()
    if success_ratio >= 0.95:
        return parsed
    return None


def detect_time_column(df: pd.DataFrame) -> str | None:
    """가장 그럴듯한 시간 컬럼을 자동으로 찾는다.

    우선순위:
    1) dtype이 이미 datetime인 컬럼
    2) 컬럼명에 시간 관련 키워드가 포함된 컬럼 중 파싱 성공률이 높은 것
    3) 전체 컬럼 중 파싱 성공률이 가장 높은 것 (충분히 높을 때만)
    """
    # 1) 이미 datetime dtype
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    time_keywords = ("time", "date", "timestamp", "datetime", "ds", "일시", "날짜", "시간")
    candidates = []
    for col in df.columns:
        name_lower = str(col).strip().lower()
        has_keyword = any(k in name_lower for k in time_keywords)
        score = 0
        if has_keyword:
            score += 2
        # 숫자형 컬럼은 컬럼명에 시간 관련 키워드가 있을 때만 (예: unix_timestamp) 파싱을 시도한다.
        parsed = _try_parse_datetime_column(df[col], allow_numeric=has_keyword)
        if parsed is not None:
            score += 1
            candidates.append((score, col))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def infer_frequency(index: pd.DatetimeIndex) -> str | None:
    """시간 인덱스의 주기를 추론. 실패 시 가장 흔한 간격으로 근사."""
    if len(index) < 3:
        return None
    try:
        freq = pd.infer_freq(index)
    except ValueError:
        freq = None
    if freq is not None:
        return freq
    diffs = index.to_series().diff().dropna()
    if len(diffs) == 0:
        return None
    most_common = diffs.mode()
    if len(most_common) > 0:
        return pd.tseries.frequencies.to_offset(most_common.iloc[0]).freqstr
    return None


def load_csv(file_obj, max_rows: int | None = None) -> LoadResult:
    """업로드된 파일 객체(Streamlit UploadedFile 또는 경로/버퍼)를 읽어 LoadResult로 반환."""
    warnings: list[str] = []

    if hasattr(file_obj, "getvalue"):
        raw = file_obj.getvalue()
        buf = io.BytesIO(raw)
    else:
        buf = file_obj

    # 구분자 자동 추론 (보통 컴마지만 세미콜론/탭도 있을 수 있음)
    try:
        df = pd.read_csv(buf, sep=None, engine="python")
    except Exception:
        buf.seek(0) if hasattr(buf, "seek") else None
        df = pd.read_csv(buf)

    if max_rows is not None and len(df) > max_rows:
        warnings.append(
            f"데이터가 {len(df):,}행으로 매우 많아 최근 {max_rows:,}행만 사용합니다."
        )
        df = df.tail(max_rows).reset_index(drop=True)

    # 컬럼명 공백 제거
    df.columns = [str(c).strip() for c in df.columns]

    time_col = detect_time_column(df)

    if time_col is not None:
        parsed_time = pd.to_datetime(df[time_col], errors="coerce")
        n_failed = parsed_time.isna().sum()
        if n_failed > 0:
            warnings.append(
                f"시간 컬럼 '{time_col}'에서 {n_failed}개 값을 날짜로 해석할 수 없어 해당 행을 제거했습니다."
            )
        df = df.assign(__parsed_time=parsed_time).dropna(subset=["__parsed_time"])
        df = df.sort_values("__parsed_time")
        df = df.set_index("__parsed_time")
        df.index.name = "timestamp"
    else:
        # 시간 컬럼을 찾지 못한 경우: 행 순서를 그대로 시간축으로 간주 (정수 인덱스 -> 합성 일자)
        warnings.append(
            "시간/날짜로 인식할 수 있는 컬럼을 찾지 못해, 행 순서를 시간 축으로 간주한 가상의 일자(1일 간격)를 생성했습니다."
        )
        df = df.reset_index(drop=True)
        df.index = pd.date_range("2000-01-01", periods=len(df), freq="D")
        df.index.name = "timestamp"

    # 원본 시간 컬럼은 더 이상 필요 없으므로 제거 (남아있다면)
    if time_col is not None and time_col in df.columns:
        df = df.drop(columns=[time_col])

    # 라벨 컬럼 탐지 (수치형 컬럼들 중에서)
    label_col = None
    numeric_df = df.select_dtypes(include="number")
    for col in numeric_df.columns:
        if _looks_like_label(col, numeric_df[col]):
            label_col = col
            break

    dropped_cols = [c for c in df.columns if c not in numeric_df.columns]
    if dropped_cols:
        warnings.append(
            f"수치형이 아닌 컬럼 {dropped_cols}는 분석 대상에서 제외했습니다."
        )

    feature_df = numeric_df.copy()
    if label_col is not None:
        feature_df = feature_df.drop(columns=[label_col])

    if feature_df.shape[1] == 0:
        raise ValueError(
            "분석에 사용할 수 있는 수치형 변수 컬럼을 찾지 못했습니다. "
            "최소 1개 이상의 숫자 컬럼이 필요합니다."
        )

    # 중복 시간 인덱스 처리 (있다면 평균으로 집계)
    if feature_df.index.duplicated().any():
        warnings.append("중복된 시간 값이 발견되어 같은 시점의 값을 평균으로 합쳤습니다.")
        feature_df = feature_df.groupby(feature_df.index).mean()
        if label_col is not None:
            label_series = numeric_df[label_col].groupby(feature_df.index).max()
        else:
            label_series = None
    else:
        label_series = numeric_df[label_col] if label_col is not None else None

    freq_guess = infer_frequency(feature_df.index)

    if label_series is not None:
        feature_df = feature_df.assign(**{label_col: label_series})

    return LoadResult(
        df=feature_df,
        time_col=time_col,
        label_col=label_col,
        dropped_cols=dropped_cols,
        freq_guess=freq_guess,
        warnings=warnings,
    )
