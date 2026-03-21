"""SageMaker ProcessingStep: Full model evaluation.

Reads:
  /opt/ml/processing/model/     — model.pkl (tarball extracted by SageMaker)
  /opt/ml/processing/test_data/ — test.parquet

Writes:
  /opt/ml/processing/output/evaluation_report.json
  /opt/ml/processing/output/metrics.json
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, "/opt/ml/processing/input/code/src")
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

MODEL_DIR = Path("/opt/ml/processing/model")
TEST_DATA_DIR = Path("/opt/ml/processing/test_data")
OUTPUT_DIR = Path("/opt/ml/processing/output")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import joblib  # noqa: PLC0415
    import polars as pl  # noqa: PLC0415

    from loan_risk.config import get_settings  # noqa: PLC0415
    from loan_risk.evaluation.metrics import compute_classification_metrics  # noqa: PLC0415
    from loan_risk.features.pipeline import (  # noqa: PLC0415
        build_feature_pipeline,
        prepare_features,
    )

    cfg = get_settings()

    # Extract model tarball if present (SageMaker packages model as model.tar.gz)
    model_tar = MODEL_DIR / "model.tar.gz"
    if model_tar.exists():
        with tarfile.open(model_tar, "r:gz") as tar:
            tar.extractall(MODEL_DIR)

    model_path = MODEL_DIR / "model.pkl"
    if not model_path.exists():
        pkl_files = list(MODEL_DIR.rglob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No model.pkl found in {MODEL_DIR}")
        model_path = pkl_files[0]

    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}: {type(model).__name__}")

    # Load test data
    test_files = list(TEST_DATA_DIR.glob("test*.parquet")) + list(TEST_DATA_DIR.glob("*.parquet"))
    if not test_files:
        raise FileNotFoundError(f"No parquet files in {TEST_DATA_DIR}")

    test_df = pl.read_parquet(test_files[0])
    print(f"Test data: {test_df.shape}")

    target_col = cfg.data.target_column
    y_test = test_df[target_col].to_numpy()

    # Apply feature pipeline
    preprocessor_files = list(MODEL_DIR.glob("preprocessor.pkl")) + list(MODEL_DIR.rglob("preprocessor.pkl"))
    if preprocessor_files:
        pipeline = joblib.load(preprocessor_files[0])
        X_test, _ = prepare_features(test_df, pipeline, fit=False)
    else:
        pipeline = build_feature_pipeline()
        X_test, _ = prepare_features(test_df, pipeline, fit=True)

    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = compute_classification_metrics(y_test, y_prob)
    print(f"Test AUC: {metrics.get('test_auc', metrics.get('roc_auc', 0)):.4f}")

    # Build evaluation report (matches SageMaker PropertyFile path expectations)
    report = {
        "metrics": {
            "test_auc": metrics.get("test_auc", metrics.get("roc_auc", 0.0)),
            "gini": metrics.get("gini", 0.0),
            "ks_statistic": metrics.get("ks_statistic", 0.0),
        },
        "evaluation_parameters": {
            "n_test_samples": len(y_test),
            "positive_rate": float(y_test.mean()),
        },
    }

    report_path = OUTPUT_DIR / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Evaluation report written to {report_path}")

    # Also write flat metrics.json for DVC compatibility
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
