"""Classification metrics for loan default prediction.

All metrics are computed on predicted probabilities with an
optional calibrated threshold for classification-based metrics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute full suite of binary classification metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted default probabilities in [0, 1].
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Dict with keys: auc_roc, gini, ks_statistic, f1, precision, recall, accuracy.
    """
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    gini = compute_gini(y_true, y_prob)
    ks = compute_ks_statistic(y_true, y_prob)

    return {
        "auc_roc": float(auc),
        "gini": float(gini),
        "ks_statistic": float(ks),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
    }


def compute_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini coefficient = 2 * AUC - 1.

    Range: [-1, 1]. A perfect classifier has Gini = 1.
    """
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic separating default vs. non-default distributions.

    Measures the maximum separation between the CDF of predicted probabilities
    for positive (default) and negative (repaid) classes.

    Range: [0, 1]. Higher is better.
    """
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]

    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0

    ks_stat, _ = stats.ks_2samp(pos_probs, neg_probs)
    return float(ks_stat)


def compute_fbeta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 2.0,
) -> float:
    """Compute F-beta score.

    With beta=2, recall is weighted twice as heavily as precision —
    appropriate when false negatives (missed defaults) are more costly.
    """
    from sklearn.metrics import fbeta_score
    return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
