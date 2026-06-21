"""
explain.py
탐지된 이상 시점 중 점수가 높은 상위 N개에 대해, 왜 이상으로 판정되었는지
사람이 읽을 수 있는 자동 설명을 생성한다.

설명에 포함되는 요소:
- 평균 대비 몇 시그마(표준편차) 떨어져 있는지
- 직전 구간 대비 변화량(급격한 상승/하강 여부)
- 어떤 Scorer(들)가 동시에 이 시점을 이상으로 판정했는지 (다중 합치 -> 신뢰도 가늠)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.detection import DetectionResult, SCORER_REGISTRY


@dataclass
class AnomalyExplanation:
    timestamp: pd.Timestamp
    column: str
    value: float
    sigma_from_mean: float           # 학습 구간 평균 대비 몇 표준편차 떨어져 있는지
    pct_change_recent: float | None  # 직전 window 평균 대비 변화율(%)
    direction: str                   # "급격한 상승" / "급격한 하강" / "패턴 변화"
    detected_by: list[str]           # 이 시점을 이상으로 판정한 scorer key 목록
    confidence_label: str            # "이상 가능성 높음" 등
    summary_lines: list[str]         # 화면에 표시할 설명 줄들


def _direction_label(value: float, baseline_mean: float) -> str:
    if value > baseline_mean:
        return "최근 패턴 대비 급격한 상승"
    return "최근 패턴 대비 급격한 하강"


def explain_top_anomalies(
    detection: DetectionResult,
    top_n: int = 5,
) -> list[AnomalyExplanation]:
    """탐지 결과 전체(모든 컴포넌트)를 통틀어, 이상 점수가 가장 높은 상위 N개 시점을 설명한다."""
    candidates: list[tuple[float, str, pd.Timestamp]] = []  # (정규화점수, 컬럼명, 시각)

    # 컴포넌트별로 scorer 점수를 0~1로 정규화한 뒤 평균내어 "종합 이상도"를 만든다.
    for comp_name, comp in detection.components.items():
        if not comp.scores:
            continue
        norm_scores = []
        for key, score_ts in comp.scores.items():
            vals = score_ts.values().flatten()
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if vmax - vmin < 1e-12:
                continue
            norm = (vals - vmin) / (vmax - vmin)
            norm_scores.append(pd.Series(norm, index=score_ts.time_index))

        if not norm_scores:
            continue

        # 공통 구간으로 정렬 후 평균
        common_start = max(s.index.min() for s in norm_scores)
        common_end = min(s.index.max() for s in norm_scores)
        aligned = [s.loc[common_start:common_end] for s in norm_scores]
        combined = pd.concat(aligned, axis=1).mean(axis=1)

        for ts, score in combined.items():
            candidates.append((float(score), comp_name, ts))

    candidates.sort(key=lambda x: x[0], reverse=True)

    explanations: list[AnomalyExplanation] = []
    seen = set()
    for score, comp_name, ts in candidates:
        key = (comp_name, ts)
        if key in seen:
            continue
        seen.add(key)

        comp = detection.components[comp_name]
        full_series = pd.concat([comp.train.to_series(), comp.test.to_series()])
        if ts not in full_series.index:
            continue

        baseline = comp.train.to_series()
        baseline_mean = baseline.mean()
        baseline_std = baseline.std() if baseline.std() > 1e-12 else 1e-12

        value = float(full_series.loc[ts])
        sigma = (value - baseline_mean) / baseline_std

        # 직전 구간(같은 길이의 window) 평균 대비 변화율
        idx_pos = full_series.index.get_loc(ts)
        window = 10
        recent_start = max(0, idx_pos - window)
        recent_mean = full_series.iloc[recent_start:idx_pos].mean() if idx_pos > recent_start else baseline_mean
        pct_change = ((value - recent_mean) / abs(recent_mean) * 100) if abs(recent_mean) > 1e-9 else None

        # 이 시점에서 어떤 scorer가 이상으로 판정했는지
        detected_by = []
        for sk, binary_ts in comp.binary.items():
            b_index = binary_ts.time_index
            if ts in b_index:
                val = binary_ts.values().flatten()[list(b_index).index(ts)]
                if val == 1:
                    detected_by.append(sk)

        n_detect = len(detected_by)
        if n_detect >= 2:
            confidence = "이상 가능성 높음 (다중 지표 동시 탐지)"
        elif n_detect == 1:
            confidence = "이상 가능성 보통 (단일 지표 탐지)"
        else:
            confidence = "참고용 (점수는 높지만 임계값 미달)"

        direction = _direction_label(value, baseline_mean)
        scorer_labels = [SCORER_REGISTRY[k]["label"] for k in detected_by]

        summary_lines = [
            f"{ts}",
            f"변수: {comp_name}",
            f"평균 대비 {sigma:+.1f}\u03c3 (시그마)",
            direction,
        ]
        if pct_change is not None:
            summary_lines.append(f"직전 구간 대비 {pct_change:+.1f}% 변화")
        if scorer_labels:
            summary_lines.append(f"{', '.join(scorer_labels)} 동시 탐지" if n_detect >= 2 else f"{scorer_labels[0]} 탐지")
        summary_lines.append(f"→ {confidence}")

        explanations.append(
            AnomalyExplanation(
                timestamp=ts,
                column=comp_name,
                value=value,
                sigma_from_mean=float(sigma),
                pct_change_recent=float(pct_change) if pct_change is not None else None,
                direction=direction,
                detected_by=detected_by,
                confidence_label=confidence,
                summary_lines=summary_lines,
            )
        )

        if len(explanations) >= top_n:
            break

    return explanations
