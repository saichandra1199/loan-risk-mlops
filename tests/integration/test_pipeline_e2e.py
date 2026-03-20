"""End-to-end integration tests for the training pipeline.

These tests run the full pipeline on synthetic data with a small
number of Optuna trials and rows, so they complete quickly while
still exercising all pipeline stages.

NOTE: These tests require MLflow to be running or use a local SQLite backend.
Set MLFLOW_TRACKING_URI=sqlite:///test_mlruns.db before running.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest


# Use local SQLite for tests — no real MLflow server needed
os.environ.setdefault("MLFLOW__TRACKING_URI", "sqlite:///test_mlruns.db")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlruns.db")


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory) -> Path:
    """Generate a small synthetic dataset for pipeline tests."""
    import numpy as np

    tmp = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(42)
    n = 500

    df = pl.DataFrame(
        {
            "loan_id": [f"LOAN_{i:07d}" for i in range(1, n + 1)],
            "loan_amount": rng.uniform(1000, 40000, n).round(2).tolist(),
            "annual_income": rng.uniform(20000, 200000, n).round(2).tolist(),
            "employment_years": rng.integers(0, 30, n).tolist(),
            "credit_score": rng.integers(300, 850, n).tolist(),
            "debt_to_income_ratio": rng.uniform(0.05, 0.90, n).round(4).tolist(),
            "num_open_accounts": rng.integers(1, 15, n).tolist(),
            "num_delinquencies": rng.integers(0, 5, n).tolist(),
            "loan_purpose": rng.choice(
                ["debt_consolidation", "home_improvement", "major_purchase", "medical", "vacation", "other"],
                n,
            ).tolist(),
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n).tolist(),
            "loan_term_months": rng.choice([36, 60], n).tolist(),
            "loan_default": rng.choice([0, 1], n, p=[0.85, 0.15]).tolist(),
        }
    )

    data_path = tmp / "loans.parquet"
    df.write_parquet(data_path)
    return data_path


def test_ingestion_loads_data(small_dataset: Path) -> None:
    """Ingestion loads and returns a non-empty DataFrame."""
    from loan_risk.data.ingestion import load_raw_data

    df = load_raw_data(small_dataset)
    assert len(df) > 0
    assert "loan_default" in df.columns


def test_validation_passes_clean_data(small_dataset: Path) -> None:
    """Clean synthetic data passes schema validation."""
    import polars as pl
    from loan_risk.data.schemas import validate_raw

    df = pl.read_parquet(small_dataset)
    validated = validate_raw(df)
    assert len(validated) == len(df)


def test_feature_pipeline_produces_numeric_matrix(small_dataset: Path) -> None:
    """Feature pipeline transforms raw data to a numeric numpy array."""
    import numpy as np
    import polars as pl
    from loan_risk.data.splits import stratified_split
    from loan_risk.features.pipeline import build_feature_pipeline, prepare_features

    df = pl.read_parquet(small_dataset)
    splits = stratified_split(df, test_size=0.2, val_size=0.1, random_seed=42)

    pipeline = build_feature_pipeline()
    drop_cols = ["loan_default", "loan_id"]
    train_features = splits.train.drop(drop_cols)
    X = prepare_features(train_features, pipeline, fit=True)

    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert X.shape[0] == len(splits.train)
    assert X.shape[1] > 10  # Should have many features after OHE


def test_model_trains_and_predicts(small_dataset: Path) -> None:
    """Model trains without error and produces probabilities in [0, 1]."""
    import numpy as np
    import polars as pl
    from loan_risk.data.splits import stratified_split
    from loan_risk.features.pipeline import build_feature_pipeline, prepare_features
    from loan_risk.training.models import compute_scale_pos_weight, get_model

    df = pl.read_parquet(small_dataset)
    splits = stratified_split(df, test_size=0.2, val_size=0.1, random_seed=42)

    pipeline = build_feature_pipeline()
    drop_cols = ["loan_default", "loan_id"]
    X_train = prepare_features(splits.train.drop(drop_cols), pipeline, fit=True)
    X_test = prepare_features(splits.test.drop(drop_cols), pipeline, fit=False)
    y_train = splits.train["loan_default"].to_numpy()

    spw = compute_scale_pos_weight(y_train)
    model = get_model("lgbm", scale_pos_weight=spw, random_seed=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    assert len(probs) == len(splits.test)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_evaluation_metrics_reasonable(small_dataset: Path) -> None:
    """AUC on synthetic data is above 0.5 (better than random)."""
    import numpy as np
    import polars as pl
    from loan_risk.data.splits import stratified_split
    from loan_risk.evaluation.metrics import compute_classification_metrics
    from loan_risk.features.pipeline import build_feature_pipeline, prepare_features
    from loan_risk.training.models import compute_scale_pos_weight, get_model

    df = pl.read_parquet(small_dataset)
    splits = stratified_split(df, test_size=0.2, val_size=0.1, random_seed=42)

    pipeline = build_feature_pipeline()
    drop_cols = ["loan_default", "loan_id"]
    X_train = prepare_features(splits.train.drop(drop_cols), pipeline, fit=True)
    X_test = prepare_features(splits.test.drop(drop_cols), pipeline, fit=False)
    y_train = splits.train["loan_default"].to_numpy()
    y_test = splits.test["loan_default"].to_numpy()

    model = get_model("lgbm", scale_pos_weight=compute_scale_pos_weight(y_train))
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_test, probs)

    # Should be better than random on meaningful synthetic data
    assert metrics["auc_roc"] > 0.5
    assert 0.0 <= metrics["gini"] <= 1.0
