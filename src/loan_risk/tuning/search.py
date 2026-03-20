"""Run Optuna hyperparameter search and persist best parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import polars as pl

from loan_risk.config import get_settings
from loan_risk.data.splits import DataSplits
from loan_risk.features.pipeline import build_feature_pipeline, prepare_features
from loan_risk.logging_setup import get_logger
from loan_risk.training.models import compute_scale_pos_weight
from loan_risk.tuning.objective import lgbm_objective, xgboost_objective

logger = get_logger(__name__)


def run_hyperparameter_search(
    splits: DataSplits,
    model_name: str | None = None,
    n_trials: int | None = None,
    random_seed: int | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Run Optuna TPE search and return best hyperparameters.

    Args:
        splits: DataSplits (train + val used for tuning, test held out).
        model_name: "lgbm" or "xgboost". Defaults to config value.
        n_trials: Number of Optuna trials. Defaults to config value.
        random_seed: Random seed for reproducibility.
        study_name: Optuna study name (for SQLite persistence).

    Returns:
        Dict of best hyperparameters for the given model.
    """
    cfg = get_settings()
    model_name = model_name or cfg.model.name
    n_trials = n_trials or cfg.training.n_trials
    seed = random_seed or cfg.training.random_seed
    target = cfg.data.target_column
    id_col = cfg.data.id_column

    # Prepare features for tuning (fit on train, transform val)
    feature_pipeline = build_feature_pipeline()
    drop_cols = [c for c in [target, id_col] if c in splits.train.columns]

    X_train = prepare_features(splits.train.drop(drop_cols), feature_pipeline, fit=True)
    X_val = prepare_features(splits.val.drop(drop_cols), feature_pipeline, fit=False)
    y_train = splits.train[target].to_numpy()
    y_val = splits.val[target].to_numpy()

    spw = compute_scale_pos_weight(y_train)

    # Select objective function
    objective_map: dict[str, Callable] = {
        "lgbm": lgbm_objective,
        "xgboost": xgboost_objective,
    }
    if model_name not in objective_map:
        raise ValueError(f"No objective defined for model '{model_name}'")

    objective_fn = objective_map[model_name]

    def objective(trial: optuna.Trial) -> float:
        return objective_fn(
            trial=trial,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            scale_pos_weight=spw,
            random_seed=seed,
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name or f"{model_name}_tuning",
        storage=None,  # In-memory; set to sqlite:///optuna.db for persistence
        load_if_exists=True,
    )

    logger.info(
        "hyperparameter_search_started",
        model=model_name,
        n_trials=n_trials,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value

    logger.info(
        "hyperparameter_search_complete",
        model=model_name,
        best_auc=best_value,
        best_params=best_params,
        n_trials=len(study.trials),
    )

    # Persist best params
    output_dir = Path("artifacts/best_params")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_best_params.json"
    with open(output_path, "w") as f:
        json.dump({"best_params": best_params, "best_auc": best_value}, f, indent=2)

    logger.info("best_params_saved", path=str(output_path))
    return best_params
