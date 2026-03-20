"""Optuna objective functions for hyperparameter optimisation."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from loan_risk.logging_setup import get_logger
from loan_risk.training.models import get_model

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def lgbm_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float = 1.0,
    random_seed: int = 42,
) -> float:
    """Optuna objective for LightGBM: maximise validation AUC-ROC.

    Args:
        trial: Optuna Trial object.
        X_train: Transformed training features.
        y_train: Training labels.
        X_val: Transformed validation features.
        y_val: Validation labels.
        scale_pos_weight: Class imbalance weight.
        random_seed: Reproducibility seed.

    Returns:
        Validation AUC-ROC (higher is better).
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = get_model(
        "lgbm",
        params=params,
        scale_pos_weight=scale_pos_weight,
        random_seed=random_seed,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)

    logger.debug("trial_complete", trial_number=trial.number, auc=auc)
    return auc


def xgboost_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float = 1.0,
    random_seed: int = 42,
) -> float:
    """Optuna objective for XGBoost: maximise validation AUC-ROC."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    model = get_model(
        "xgboost",
        params=params,
        scale_pos_weight=scale_pos_weight,
        random_seed=random_seed,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_probs)
