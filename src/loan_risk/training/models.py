"""Model factory: returns configured (unfitted) model instances.

Supported models:
- lgbm: LightGBMClassifier (primary)
- xgboost: XGBClassifier (challenger)
"""

from __future__ import annotations

from typing import Any, Literal

import lightgbm as lgb
import numpy as np
from xgboost import XGBClassifier

from loan_risk.exceptions import ConfigurationError
from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)

ModelName = Literal["lgbm", "xgboost"]


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight = n_negatives / n_positives for imbalanced data."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0:
        raise ValueError("No positive examples in training data.")
    return n_neg / n_pos


def get_model(
    name: ModelName,
    params: dict[str, Any] | None = None,
    scale_pos_weight: float = 1.0,
    random_seed: int = 42,
) -> lgb.LGBMClassifier | XGBClassifier:
    """Return a configured, unfitted model instance.

    Args:
        name: Model type identifier.
        params: Hyperparameters to override defaults (e.g., from Optuna).
        scale_pos_weight: Class imbalance weight (n_neg/n_pos).
        random_seed: Reproducibility seed.

    Returns:
        Unfitted sklearn-compatible classifier.

    Raises:
        ConfigurationError: If model name is not supported.
    """
    params = params or {}

    if name == "lgbm":
        defaults: dict[str, Any] = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        defaults.update(params)
        model = lgb.LGBMClassifier(**defaults)

    elif name == "xgboost":
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_seed,
            "n_jobs": -1,
            "eval_metric": "auc",
            "tree_method": "hist",
        }
        defaults.update(params)
        model = XGBClassifier(**defaults)

    else:
        raise ConfigurationError(f"Unsupported model name: '{name}'. Choose 'lgbm' or 'xgboost'.")

    logger.info("model_created", name=name, params=defaults)
    return model
