"""Live model performance tracking using delayed ground-truth labels.

Loan defaults are typically known 30-90 days after origination.
When labels arrive, this module computes live AUC against logged predictions.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from loan_risk.config import get_settings
from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)


def log_prediction(
    loan_id: str,
    default_probability: float,
    model_version: str,
    request_id: str,
    timestamp: str | None = None,
) -> None:
    """Append a prediction to the monitoring log.

    The log is used later when ground-truth labels arrive to compute live AUC.

    Args:
        loan_id: Unique loan identifier.
        default_probability: Predicted default probability.
        model_version: Model version used for this prediction.
        request_id: Request trace ID.
        timestamp: ISO timestamp (auto-generated if None).
    """
    cfg = get_settings()
    log_path = Path(cfg.monitoring.prediction_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = timestamp or datetime.datetime.utcnow().isoformat()

    new_row = pl.DataFrame(
        {
            "loan_id": [loan_id],
            "default_probability": [float(default_probability)],
            "model_version": [model_version],
            "request_id": [request_id],
            "timestamp": [timestamp],
            "actual_default": [None],  # filled in when labels arrive
        },
        schema={
            "loan_id": pl.Utf8,
            "default_probability": pl.Float64,
            "model_version": pl.Utf8,
            "request_id": pl.Utf8,
            "timestamp": pl.Utf8,
            "actual_default": pl.Int32,
        },
    )

    if log_path.exists():
        existing = pl.read_parquet(log_path)
        combined = pl.concat([existing, new_row], how="diagonal")
    else:
        combined = new_row

    combined.write_parquet(log_path)


def update_ground_truth(
    labels: pl.DataFrame,
    loan_id_column: str = "loan_id",
    label_column: str = "loan_default",
) -> int:
    """Update the prediction log with newly arrived ground-truth labels.

    Args:
        labels: DataFrame with loan_id and actual default label columns.
        loan_id_column: Name of the ID column in labels DataFrame.
        label_column: Name of the target column in labels DataFrame.

    Returns:
        Number of predictions updated.
    """
    cfg = get_settings()
    log_path = Path(cfg.monitoring.prediction_log_path)

    if not log_path.exists():
        logger.warning("prediction_log_not_found", path=str(log_path))
        return 0

    log_df = pl.read_parquet(log_path)

    # Join on loan_id
    labels_renamed = labels.select(
        pl.col(loan_id_column).alias("loan_id"),
        pl.col(label_column).cast(pl.Int32).alias("actual_default"),
    )

    updated = log_df.join(
        labels_renamed,
        on="loan_id",
        how="left",
        suffix="_new",
    ).with_columns(
        pl.coalesce(["actual_default_new", "actual_default"]).alias("actual_default")
    ).drop("actual_default_new")

    updated.write_parquet(log_path)
    n_updated = int(labels_renamed["loan_id"].is_in(log_df["loan_id"]).sum())

    logger.info("ground_truth_updated", n_updated=n_updated)
    return n_updated


def compute_live_auc(min_samples: int = 100) -> dict[str, Any]:
    """Compute live AUC from the prediction log where labels have arrived.

    Args:
        min_samples: Minimum labeled samples required to compute AUC.

    Returns:
        Dict with live_auc, n_labeled, n_total, model_versions.
        Returns {"live_auc": None} if insufficient labeled data.
    """
    cfg = get_settings()
    log_path = Path(cfg.monitoring.prediction_log_path)

    if not log_path.exists():
        return {"live_auc": None, "reason": "No prediction log found"}

    log_df = pl.read_parquet(log_path)
    labeled = log_df.filter(pl.col("actual_default").is_not_null())

    if len(labeled) < min_samples:
        return {
            "live_auc": None,
            "reason": f"Insufficient labeled samples: {len(labeled)} < {min_samples}",
            "n_labeled": len(labeled),
            "n_total": len(log_df),
        }

    y_true = labeled["actual_default"].to_numpy()
    y_prob = labeled["default_probability"].to_numpy()

    if len(np.unique(y_true)) < 2:
        return {"live_auc": None, "reason": "Only one class present in labeled data"}

    live_auc = float(roc_auc_score(y_true, y_prob))
    model_versions = labeled["model_version"].unique().to_list()

    logger.info("live_auc_computed", live_auc=live_auc, n_labeled=len(labeled))

    return {
        "live_auc": round(live_auc, 4),
        "n_labeled": len(labeled),
        "n_total": len(log_df),
        "model_versions": model_versions,
    }
