"""SageMaker Training Job entry point.

SageMaker injects hyperparameters as SM_HP_* environment variables.
Data is provided at SM_CHANNEL_TRAIN, model artifacts must be written to SM_MODEL_DIR.

Environment variables used:
    SM_HP_MODEL_TYPE          — lgbm or xgboost
    SM_HP_MLFLOW_TRACKING_URI — MLflow tracking URI (RDS PostgreSQL)
    SM_CHANNEL_TRAIN          — path to feature data
    SM_MODEL_DIR              — path to write model artifacts
    SM_OUTPUT_DATA_DIR        — path to write evaluation outputs
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# SageMaker Training container provides the source at /opt/ml/code
sys.path.insert(0, "/opt/ml/code/src")
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# SageMaker paths
TRAIN_DIR = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))


def _get_hp(name: str, default: str = "") -> str:
    return os.environ.get(f"SM_HP_{name.upper()}", default)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_type = _get_hp("model_type", "lgbm")
    mlflow_uri = _get_hp("mlflow_tracking_uri", os.environ.get("MLFLOW_TRACKING_URI", ""))
    data_bucket = _get_hp("data_bucket", os.environ.get("AWS_DATA_BUCKET", ""))
    artifacts_bucket = _get_hp("artifacts_bucket", os.environ.get("AWS_ARTIFACTS_BUCKET", ""))

    if mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
        os.environ["MLFLOW__TRACKING_URI"] = mlflow_uri

    import polars as pl  # noqa: PLC0415

    from loan_risk.config import get_settings  # noqa: PLC0415
    from loan_risk.features.pipeline import load_pipeline  # noqa: PLC0415
    from loan_risk.training.trainer import ModelTrainer  # noqa: PLC0415

    cfg = get_settings()

    # Load split data
    train_df = pl.read_parquet(TRAIN_DIR / "train.parquet")
    val_df = pl.read_parquet(TRAIN_DIR / "val.parquet")
    test_df = pl.read_parquet(TRAIN_DIR / "test.parquet")

    print(f"Data loaded — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Load preprocessor from artifacts dir if available
    preprocessor_path = TRAIN_DIR / "preprocessor.pkl"
    feature_pipeline = None
    if preprocessor_path.exists():
        feature_pipeline = load_pipeline(str(preprocessor_path))

    # Override model type from hyperparameter
    import copy  # noqa: PLC0415
    settings_override = copy.deepcopy(cfg)
    settings_override.model.name = model_type  # type: ignore[assignment]

    # Collect SageMaker HPO hyperparameters
    hp_overrides: dict = {}
    for key in ["max_depth", "learning_rate", "n_estimators", "subsample",
                "colsample_bytree", "min_child_weight", "num_leaves", "reg_alpha", "reg_lambda"]:
        val = _get_hp(key)
        if val:
            try:
                hp_overrides[key] = float(val) if "." in val else int(val)
            except ValueError:
                hp_overrides[key] = val

    # Train
    trainer = ModelTrainer(settings=settings_override)
    result = trainer.fit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_pipeline=feature_pipeline,
        hyperparams=hp_overrides or None,
    )

    print(f"Training complete — test AUC: {result.test_auc:.4f}, run_id: {result.run_id}")

    # Write model artifacts to SM_MODEL_DIR (SageMaker tars this up automatically)
    import joblib  # noqa: PLC0415
    joblib.dump(result.model, MODEL_DIR / "model.pkl")

    # Write metrics for downstream EvaluateStep
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps({
        "test_auc": result.test_auc,
        "run_id": result.run_id,
        "model_type": model_type,
        "data_bucket": data_bucket,
        "artifacts_bucket": artifacts_bucket,
    }, indent=2))

    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
