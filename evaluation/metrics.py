"""Metrics shared by the diffusion experiments."""

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Scale scores to [0, 1].  A constant vector maps to all-zeros.

    This mirrors what ``test.py`` does before computing AUC.  It is monotone, so
    it does not change a single model's ROC-AUC -- it exists to put the TSR and
    diffusion scores on a common scale before fusion.
    """
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC, or NaN when only one class is present (never a silent 0.5)."""
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (PR-AUC) with abnormal = 1 as the positive class."""
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, np.asarray(scores, dtype=np.float64)))


def summarize(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """ROC-AUC, PR-AUC and the normal/abnormal means for one score vector."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    normal = scores[labels == 0]
    abnormal = scores[labels == 1]
    return {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
        "mean_normal": float(normal.mean()) if normal.size else float("nan"),
        "mean_abnormal": float(abnormal.mean()) if abnormal.size else float("nan"),
        "diff": float(abnormal.mean() - normal.mean()) if normal.size and abnormal.size else float("nan"),
        "n_normal": int(normal.size),
        "n_abnormal": int(abnormal.size),
    }
