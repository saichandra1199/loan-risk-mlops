"""SHAP-based model explainability using TreeExplainer.

TreeExplainer computes exact Shapley values for tree-based models
(LightGBM, XGBoost, Random Forest) in O(TLD^2) time where T=trees,
L=leaves, D=depth — much faster than the KernelExplainer alternative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.pipeline import Pipeline

from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    check_additivity: bool = False,
) -> np.ndarray:
    """Compute SHAP values using TreeExplainer.

    Args:
        model: Fitted tree model (LightGBM or XGBoost).
        X: Feature matrix (numpy array).
        check_additivity: Verify SHAP values sum to model output (slow).

    Returns:
        SHAP values array of shape (n_samples, n_features).
        For binary classification, returns values for the positive class.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=check_additivity)

    # LightGBM returns list [neg_class, pos_class]; take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info(
        "shap_values_computed",
        n_samples=shap_values.shape[0],
        n_features=shap_values.shape[1],
    )
    return shap_values


def get_top_shap_factors(
    shap_values_row: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Extract top N SHAP factors for a single prediction.

    Args:
        shap_values_row: SHAP values for one sample (1D array).
        feature_names: Feature names corresponding to SHAP value indices.
        top_n: Number of top factors to return.

    Returns:
        List of dicts with keys: feature, shap_value, direction.
        Sorted by absolute SHAP value descending.
    """
    indexed = sorted(
        enumerate(shap_values_row),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    factors = []
    for idx, value in indexed[:top_n]:
        if idx < len(feature_names):
            factors.append(
                {
                    "feature": feature_names[idx],
                    "shap_value": round(float(value), 4),
                    "direction": "increases_risk" if value > 0 else "decreases_risk",
                }
            )

    return factors


def save_shap_summary_plot(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_path: str,
    max_display: int = 20,
) -> None:
    """Generate and save a SHAP summary beeswarm plot.

    Args:
        shap_values: SHAP values matrix (n_samples, n_features).
        X: Feature matrix (n_samples, n_features).
        feature_names: Column names for features.
        output_path: File path to save the PNG figure.
        max_display: Maximum features to display.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("shap_summary_plot_saved", path=output_path)
