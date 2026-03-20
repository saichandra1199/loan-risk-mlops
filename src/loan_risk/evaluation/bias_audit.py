"""Fairness metrics across demographic slices.

Audits model performance across loan_purpose and home_ownership slices
to detect disparate impact. Reports AUC and approval rate per slice.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)

# Slice columns we audit for fairness
AUDIT_COLUMNS = ["loan_purpose", "home_ownership", "loan_term_months"]


def compute_slice_metrics(
    df: pl.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    slice_columns: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Compute AUC and approval rate for each value of each slice column.

    Args:
        df: DataFrame containing slice columns and loan_default column.
        y_prob: Predicted default probabilities (same length as df).
        threshold: Decision threshold for approval rate calculation.
        slice_columns: Columns to slice by. Defaults to AUDIT_COLUMNS.

    Returns:
        Dict mapping slice_column → list of per-value metric dicts.
    """
    slice_columns = slice_columns or AUDIT_COLUMNS
    target = "loan_default"
    results: dict[str, list[dict]] = {}

    for col in slice_columns:
        if col not in df.columns:
            continue

        slice_results = []
        unique_values = df[col].unique().to_list()

        for val in sorted(str(v) for v in unique_values):
            mask = df[col].cast(pl.Utf8) == str(val)
            indices = mask.arg_true().to_numpy()

            if len(indices) < 20:
                continue  # Skip slices too small for reliable metrics

            slice_y_true = df[target].to_numpy()[indices]
            slice_y_prob = y_prob[indices]
            y_pred = (slice_y_prob >= threshold).astype(int)

            # Skip if only one class present in slice
            if len(np.unique(slice_y_true)) < 2:
                continue

            slice_auc = float(roc_auc_score(slice_y_true, slice_y_prob))
            approval_rate = float(1 - y_pred.mean())  # predicted approve = predicted 0

            slice_results.append(
                {
                    "value": str(val),
                    "n_samples": int(len(indices)),
                    "auc_roc": round(slice_auc, 4),
                    "approval_rate": round(approval_rate, 4),
                    "default_rate": round(float(slice_y_true.mean()), 4),
                }
            )

        results[col] = slice_results
        logger.info(
            "bias_audit_slice_complete",
            column=col,
            n_values=len(slice_results),
        )

    return results


def compute_disparate_impact(slice_metrics: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Compute disparate impact ratio for each slice column.

    Disparate impact = min(approval_rate) / max(approval_rate).
    A ratio < 0.8 (the 4/5ths rule) is a potential fairness concern.

    Args:
        slice_metrics: Output of compute_slice_metrics().

    Returns:
        Dict mapping column name → disparate impact info.
    """
    di_results: dict[str, list[dict]] = {}

    for col, slices in slice_metrics.items():
        if len(slices) < 2:
            continue

        approval_rates = [s["approval_rate"] for s in slices]
        max_rate = max(approval_rates)
        min_rate = min(approval_rates)

        if max_rate == 0:
            continue

        di_ratio = min_rate / max_rate
        flagged = di_ratio < 0.80

        di_results[col] = [
            {
                "disparate_impact_ratio": round(di_ratio, 4),
                "max_approval_rate": round(max_rate, 4),
                "min_approval_rate": round(min_rate, 4),
                "flagged_4_5ths_rule": flagged,
            }
        ]

        if flagged:
            logger.warning(
                "disparate_impact_flagged",
                column=col,
                ratio=di_ratio,
            )

    return di_results
