"""ModelTrainer: orchestrates training, evaluation, and MLflow logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from loan_risk.config import get_settings
from loan_risk.data.splits import DataSplits
from loan_risk.evaluation.metrics import compute_classification_metrics
from loan_risk.exceptions import TrainingError
from loan_risk.features.pipeline import (
    build_feature_pipeline,
    prepare_features,
    save_pipeline,
)
from loan_risk.logging_setup import get_logger
from loan_risk.training.models import compute_scale_pos_weight, get_model

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Summary of a completed training run."""

    run_id: str
    model_name: str
    val_auc: float
    test_auc: float
    threshold: float
    params: dict[str, Any]
    feature_pipeline_path: str


class ModelTrainer:
    """End-to-end model trainer with MLflow experiment tracking.

    Handles:
    - Feature pipeline fitting + transformation
    - Model training with early stopping on LightGBM/XGBoost
    - Threshold calibration (optimised for F-beta recall-weighted)
    - MLflow run logging (params, metrics, artifacts)

    Usage:
        trainer = ModelTrainer()
        result = trainer.fit(splits, model_name="lgbm")
    """

    def __init__(self) -> None:
        self.cfg = get_settings()

    def fit(
        self,
        splits: DataSplits,
        model_name: str | None = None,
        best_params: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Train a model end-to-end and log to MLflow.

        Args:
            splits: DataSplits with train/val/test DataFrames.
            model_name: Model type ("lgbm" or "xgboost"). Defaults to config.
            best_params: Optuna-tuned hyperparameters. If None, uses defaults.

        Returns:
            TrainingResult with run_id, metrics, and artifact paths.
        """
        model_name = model_name or self.cfg.model.name

        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("training_started", run_id=run_id, model_name=model_name)

            try:
                result = self._train_run(
                    run_id=run_id,
                    splits=splits,
                    model_name=model_name,
                    best_params=best_params or {},
                )
            except Exception as exc:
                logger.error("training_failed", run_id=run_id, error=str(exc))
                raise TrainingError(f"Training failed: {exc}") from exc

        logger.info(
            "training_complete",
            run_id=result.run_id,
            val_auc=result.val_auc,
            test_auc=result.test_auc,
        )
        return result

    def _train_run(
        self,
        run_id: str,
        splits: DataSplits,
        model_name: str,
        best_params: dict[str, Any],
    ) -> TrainingResult:
        """Inner training logic (runs inside an active MLflow run context)."""
        cfg = self.cfg
        target = cfg.data.target_column
        seed = cfg.training.random_seed

        # --- Feature Engineering ---
        feature_pipeline = build_feature_pipeline()
        drop_cols = [target, cfg.data.id_column]

        train_features = splits.train.drop([c for c in drop_cols if c in splits.train.columns])
        val_features = splits.val.drop([c for c in drop_cols if c in splits.val.columns])
        test_features = splits.test.drop([c for c in drop_cols if c in splits.test.columns])

        X_train = prepare_features(train_features, feature_pipeline, fit=True)
        X_val = prepare_features(val_features, feature_pipeline, fit=False)
        X_test = prepare_features(test_features, feature_pipeline, fit=False)

        y_train = splits.train[target].to_numpy()
        y_val = splits.val[target].to_numpy()
        y_test = splits.test[target].to_numpy()

        # --- Save feature pipeline ---
        pipeline_path = f"artifacts/preprocessor_{run_id[:8]}.pkl"
        Path("artifacts").mkdir(exist_ok=True)
        save_pipeline(feature_pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

        # --- Model ---
        spw = compute_scale_pos_weight(y_train)
        model = get_model(model_name, params=best_params, scale_pos_weight=spw, random_seed=seed)

        # LightGBM early stopping
        fit_kwargs: dict[str, Any] = {}
        if model_name == "lgbm":
            from lightgbm import early_stopping, log_evaluation
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "callbacks": [
                    early_stopping(cfg.training.early_stopping_rounds, verbose=False),
                    log_evaluation(period=-1),
                ],
            }
        elif model_name == "xgboost":
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
            }

        model.fit(X_train, y_train, **fit_kwargs)

        # --- Threshold calibration (F-beta, beta=2) ---
        val_probs = model.predict_proba(X_val)[:, 1]
        threshold = _calibrate_threshold(y_val, val_probs, beta=2.0)

        # --- Metrics ---
        val_metrics = compute_classification_metrics(y_val, val_probs, threshold=threshold)
        test_probs = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_classification_metrics(y_test, test_probs, threshold=threshold)

        # --- MLflow Logging ---
        mlflow.log_params({"model_name": model_name, "threshold": threshold, **best_params})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=cfg.mlflow.registered_model_name,
            input_example=X_train[:1],
        )

        return TrainingResult(
            run_id=run_id,
            model_name=model_name,
            val_auc=val_metrics["auc_roc"],
            test_auc=test_metrics["auc_roc"],
            threshold=threshold,
            params=best_params,
            feature_pipeline_path=pipeline_path,
        )


def _calibrate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0,
) -> float:
    """Find decision threshold that maximises F-beta score (recall-weighted).

    A beta > 1 weights recall higher than precision — appropriate for
    loan default prediction where false negatives (approving risky loans)
    are more costly than false positives.

    Args:
        y_true: True binary labels.
        y_prob: Predicted default probabilities.
        beta: F-beta weight (2.0 = recall twice as important as precision).

    Returns:
        Optimal probability threshold in [0.1, 0.9].
    """
    from sklearn.metrics import fbeta_score

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score, best_threshold = -1.0, 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold
