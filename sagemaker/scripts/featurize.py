"""SageMaker ProcessingStep: Feature engineering and train/val/test split.

Reads preprocessed Parquet from /opt/ml/processing/input/,
writes split Parquets + preprocessor.pkl to /opt/ml/processing/output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/opt/ml/processing/input/code/src")
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

INPUT_DIR = Path("/opt/ml/processing/input")
DATA_OUTPUT_DIR = Path("/opt/ml/processing/output/data")
ARTIFACT_OUTPUT_DIR = Path("/opt/ml/processing/output/artifacts")


def main() -> None:
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import polars as pl  # noqa: PLC0415

    from loan_risk.config import get_settings  # noqa: PLC0415
    from loan_risk.data.splits import split_data  # noqa: PLC0415
    from loan_risk.features.pipeline import (  # noqa: PLC0415
        build_feature_pipeline,
        prepare_features,
    )

    cfg = get_settings()

    # Load preprocessed data
    parquet_files = list(INPUT_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")

    df = pl.read_parquet(parquet_files[0])
    print(f"Loaded {len(df)} rows")

    # Split data
    train_df, val_df, test_df = split_data(
        df,
        test_size=cfg.training.test_size,
        val_size=cfg.training.val_size,
        random_seed=cfg.training.random_seed,
        target_col=cfg.data.target_column,
    )

    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Fit feature pipeline on training data, transform all splits
    pipeline = build_feature_pipeline()
    X_train, pipeline = prepare_features(train_df, pipeline, fit=True)
    X_val, _ = prepare_features(val_df, pipeline, fit=False)
    X_test, _ = prepare_features(test_df, pipeline, fit=False)

    # Save splits
    target = cfg.data.target_column
    train_df.write_parquet(DATA_OUTPUT_DIR / "train.parquet")
    val_df.write_parquet(DATA_OUTPUT_DIR / "val.parquet")
    test_df.write_parquet(DATA_OUTPUT_DIR / "test.parquet")

    # Save preprocessor
    import joblib  # noqa: PLC0415
    joblib.dump(pipeline, ARTIFACT_OUTPUT_DIR / "preprocessor.pkl")

    print(f"Saved splits and preprocessor to {DATA_OUTPUT_DIR}, {ARTIFACT_OUTPUT_DIR}")
    print(f"Feature shape: {X_train.shape}")

    _ = (X_val, X_test, target)  # suppress unused warning


if __name__ == "__main__":
    main()
