"""Build the sklearn feature engineering pipeline.

The pipeline:
1. LoanToIncomeRatioTransformer  — adds loan_to_income_ratio
2. LogTransformer                — adds log_loan_amount, log_annual_income
3. CreditScoreBinner             — adds credit_score_band (ordinal string)
4. DelinquencyRiskFlag           — adds high_delinquency_risk binary flag
5. ColumnTransformer             — OHE categoricals, StandardScaler numerics
6. Output: dense numpy array ready for model training

Usage:
    pipeline = build_feature_pipeline()
    X_train = pipeline.fit_transform(train_df.to_pandas(), y_train)
    X_test  = pipeline.transform(test_df.to_pandas())
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_risk.features.definitions import (
    CATEGORICAL_FEATURES,
    CREDIT_SCORE_BINS,
    CREDIT_SCORE_LABELS,
    LOG_TRANSFORM_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from loan_risk.features.transformers import (
    CreditScoreBinner,
    DelinquencyRiskFlag,
    LogTransformer,
    LoanToIncomeRatioTransformer,
)
from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)


def build_feature_pipeline() -> Pipeline:
    """Return an unfitted sklearn Pipeline for loan feature engineering.

    The pipeline is stateless until fit() is called. Call fit_transform()
    on the training DataFrame and transform() on validation/test DataFrames.

    Returns:
        sklearn Pipeline ready to be fit.
    """
    # Post-engineering numeric features (original + engineered)
    numeric_features_final = NUMERIC_FEATURES + [
        "loan_to_income_ratio",
        "log_loan_amount",
        "log_annual_income",
    ]

    # Categorical features after engineering (original + binned credit score)
    categorical_features_final = CATEGORICAL_FEATURES + ["credit_score_band"]

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features_final,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features_final,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("loan_to_income", LoanToIncomeRatioTransformer()),
            ("log_transform", LogTransformer(columns=LOG_TRANSFORM_FEATURES)),
            ("credit_binner", CreditScoreBinner(bins=CREDIT_SCORE_BINS, labels=CREDIT_SCORE_LABELS)),
            ("delinquency_flag", DelinquencyRiskFlag(threshold=2)),
            ("column_transformer", column_transformer),
        ]
    )

    return pipeline


def prepare_features(
    df: pl.DataFrame | pd.DataFrame,
    pipeline: Pipeline,
    fit: bool = False,
) -> np.ndarray:
    """Apply feature pipeline to a DataFrame.

    Args:
        df: Input data (Polars or pandas). Must NOT contain the target column.
        pipeline: sklearn Pipeline (fitted or unfitted).
        fit: If True, call fit_transform(); if False, call transform().

    Returns:
        Numpy array of shape (n_samples, n_features).
    """
    if isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df

    if fit:
        X = pipeline.fit_transform(pdf)
    else:
        X = pipeline.transform(pdf)

    logger.info(
        "features_prepared",
        n_rows=X.shape[0],
        n_features=X.shape[1],
        fit=fit,
    )
    return X


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer.

    Returns:
        List of feature names matching the pipeline output columns.
    """
    ct: ColumnTransformer = pipeline.named_steps["column_transformer"]
    numeric_names = list(ct.transformers_[0][2])
    ohe: OneHotEncoder = ct.transformers_[1][1]
    cat_names = list(ohe.get_feature_names_out(ct.transformers_[1][2]))
    return numeric_names + cat_names


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Persist a fitted pipeline to disk with joblib."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("pipeline_saved", path=path)


def load_pipeline(path: str) -> Pipeline:
    """Load a fitted pipeline from disk."""
    pipeline = joblib.load(path)
    logger.info("pipeline_loaded", path=path)
    return pipeline
