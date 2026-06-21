"""
evaluation.py
이상탐지 결과가 "적절한지" 판단할 수 있도록 다양한 평가지표를 계산한다.

두 가지 모드:
1) 라벨(정답 이상 여부)이 있는 경우: AUC-ROC, AUC-PR, Precision/Recall/F1 등 정량 평가
2) 라벨이 없는 경우 (대부분의 실제 사용 시나리오): 점수 분포, 탐지율, scorer 간 일치도 등
   "정답은 모르지만 이 탐지가 그럴듯한가"를 판단하는 데 도움이 되는 보조 지표를 제공
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from core.detection import DetectionResult, SCORER_REGISTRY


@dataclass
class LabeledMetrics:
    scorer_key: str
    auc_roc: float | None
    auc_pr: float | None
    precision: float
    recall: float
    f1: float
    n_true_anomalies: int
    n_detected: int


@dataclass
class UnsupervisedMetrics:
    scorer_key: str
    n_detected: int
    detection_rate: float          # 전체 중 이상으로 판정된 비율
    score_mean: float
    score_std: float
    score_max: float
    separation_ratio: float        # (이상으로 판정된 점수 평균) / (정상 판정 점수 평균) - 클수록 분리가 잘 됨


def _tz_naive(index: pd.Index) -> pd.Index:
    """DatetimeIndex가 tz-aware이면 tz 정보를 제거한다.

    binary_ts/score_ts의 time_index와 label_series.index가 tz 보유 여부에서
    서로 다르면 reindex 결과가 전부 NaN이 되어버리므로, 비교 전에 항상
    tz-naive로 통일한다.
    """
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        return index.tz_localize(None)
    return index


def evaluate_with_labels(
    detection: DetectionResult,
    label_series: pd.Series,
) -> dict[str, dict[str, LabeledMetrics]]:
    """컴포넌트별, scorer별로 정답 라벨 대비 성능을 계산한다.

    label_series는 전체 시간 인덱스를 가진 0/1 시리즈여야 한다 (test 구간 포함).
    """
    results: dict[str, dict[str, LabeledMetrics]] = {}

    label_series = label_series.astype(int)
    label_series = label_series.set_axis(_tz_naive(label_series.index))

    # 탐지 구간 전체의 실제 이상 개수 (모든 scorer/컴포넌트 공통 기준).
    # 컴포넌트/scorer마다 binary_ts의 길이가 달라 reindex 범위가 제각각이면
    # "실제 이상" 개수가 scorer마다 다르게 보이는 문제가 생기므로, detection
    # 전체의 탐지 구간(combined_binary)을 기준으로 통일한다.
    test_idx = _tz_naive(detection.combined_binary.time_index)
    n_true_anomalies = int(label_series.reindex(test_idx).fillna(0).astype(int).sum())

    for comp_name, comp in detection.components.items():
        results[comp_name] = {}
        for key, binary_ts in comp.binary.items():
            idx = _tz_naive(binary_ts.time_index)
            y_pred = binary_ts.values().flatten().astype(int)
            y_true = label_series.reindex(idx).fillna(0).astype(int).values

            score_ts = comp.scores.get(key)
            y_score = None
            if score_ts is not None:
                score_idx = _tz_naive(score_ts.time_index)
                aligned_true_for_score = label_series.reindex(score_idx).fillna(0).astype(int).values
                if len(np.unique(aligned_true_for_score)) > 1:
                    y_score = score_ts.values().flatten()
                    true_for_auc = aligned_true_for_score
                else:
                    true_for_auc = None
            else:
                true_for_auc = None

            auc_roc = None
            auc_pr = None
            if y_score is not None and true_for_auc is not None:
                try:
                    auc_roc = float(roc_auc_score(true_for_auc, y_score))
                    auc_pr = float(average_precision_score(true_for_auc, y_score))
                except ValueError:
                    pass

            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))

            results[comp_name][key] = LabeledMetrics(
                scorer_key=key,
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                precision=precision,
                recall=recall,
                f1=f1,
                n_true_anomalies=n_true_anomalies,
                n_detected=int(y_pred.sum()),
            )

    return results


def evaluate_unsupervised(
    detection: DetectionResult,
) -> dict[str, dict[str, UnsupervisedMetrics]]:
    """라벨이 없을 때, 점수 분포 기반으로 탐지의 그럴듯함을 가늠할 수 있는 보조 지표를 계산한다."""
    results: dict[str, dict[str, UnsupervisedMetrics]] = {}

    for comp_name, comp in detection.components.items():
        results[comp_name] = {}
        for key, score_ts in comp.scores.items():
            binary_ts = comp.binary.get(key)
            scores = score_ts.values().flatten()
            scores = scores[~np.isnan(scores)]

            if binary_ts is not None:
                # score와 binary의 시간축이 동일하므로 그대로 매칭
                common_start = max(score_ts.start_time(), binary_ts.start_time())
                common_end = min(score_ts.end_time(), binary_ts.end_time())
                s_aligned = score_ts.slice(common_start, common_end).values().flatten()
                b_aligned = binary_ts.slice(common_start, common_end).values().flatten().astype(int)
            else:
                s_aligned = scores
                b_aligned = np.zeros_like(scores, dtype=int)

            n_detected = int(b_aligned.sum())
            detection_rate = float(n_detected / len(b_aligned)) if len(b_aligned) > 0 else 0.0

            if n_detected > 0 and n_detected < len(b_aligned):
                mean_anomaly = s_aligned[b_aligned == 1].mean()
                mean_normal = s_aligned[b_aligned == 0].mean()
                separation_ratio = float(mean_anomaly / mean_normal) if mean_normal != 0 else float("inf")
            else:
                separation_ratio = float("nan")

            results[comp_name][key] = UnsupervisedMetrics(
                scorer_key=key,
                n_detected=n_detected,
                detection_rate=detection_rate,
                score_mean=float(np.mean(scores)) if len(scores) else float("nan"),
                score_std=float(np.std(scores)) if len(scores) else float("nan"),
                score_max=float(np.max(scores)) if len(scores) else float("nan"),
                separation_ratio=separation_ratio,
            )

    return results


def scorer_agreement_matrix(detection: DetectionResult, comp_name: str) -> pd.DataFrame:
    """한 컴포넌트 내에서 scorer들끼리 같은 시점을 이상으로 판단하는 정도(일치율)를 행렬로 계산.

    여러 scorer가 같은 시점을 이상으로 본다면 그 탐지에 대한 신뢰도가 높다고 해석할 수 있다.
    """
    comp = detection.components[comp_name]
    keys = list(comp.binary.keys())
    if not keys:
        return pd.DataFrame()

    # 공통 구간으로 정렬
    common_start = max(comp.binary[k].start_time() for k in keys)
    common_end = min(comp.binary[k].end_time() for k in keys)
    aligned = {k: comp.binary[k].slice(common_start, common_end).values().flatten() for k in keys}

    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            agree = (aligned[ki] == aligned[kj]).mean()
            mat[i, j] = agree

    labels = [SCORER_REGISTRY[k]["label"] for k in keys]
    return pd.DataFrame(mat, index=labels, columns=labels)
