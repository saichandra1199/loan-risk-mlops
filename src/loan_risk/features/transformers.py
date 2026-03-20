"""Custom sklearn-compatible transformers for loan feature engineering.

All transformers follow the sklearn transformer protocol:
fit(X, y=None), transform(X), fit_transform(X, y=None).
They operate on pandas DataFrames (as required by sklearn Pipeline internals)
but accept Polars input via a Polars→pandas bridge in pipeline.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LoanToIncomeRatioTransformer(TransformerMixin, BaseEstimator):
    """Adds loan_to_income_ratio = loan_amount / annual_income."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> LoanToIncomeRatioTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["loan_to_income_ratio"] = X["loan_amount"] / X["annual_income"].replace(0, np.nan)
        X["loan_to_income_ratio"] = X["loan_to_income_ratio"].fillna(0.0).clip(0.0, 10.0)
        return X


class LogTransformer(TransformerMixin, BaseEstimator):
    """Applies log1p to specified columns to reduce right skew.

    Args:
        columns: List of column names to log-transform.
        prefix: Prefix added to transformed column names (e.g., "log_").
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        prefix: str = "log_",
    ) -> None:
        self.columns = columns or ["loan_amount", "annual_income"]
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> LogTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[f"{self.prefix}{col}"] = np.log1p(X[col].clip(lower=0))
        return X


class CreditScoreBinner(TransformerMixin, BaseEstimator):
    """Bins credit_score into FICO tier bands as a categorical feature.

    Args:
        bins: Bin edges (monotonically increasing).
        labels: One label per bin interval.
        output_column: Name of the new binned column.
    """

    def __init__(
        self,
        bins: list[int] | None = None,
        labels: list[str] | None = None,
        output_column: str = "credit_score_band",
    ) -> None:
        self.bins = bins or [300, 580, 670, 740, 800, 851]
        self.labels = labels or ["Poor", "Fair", "Good", "Very_Good", "Exceptional"]
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CreditScoreBinner:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.output_column] = pd.cut(
            X["credit_score"],
            bins=self.bins,
            labels=self.labels,
            right=False,
            include_lowest=True,
        ).astype(str)
        return X


class DelinquencyRiskFlag(TransformerMixin, BaseEstimator):
    """Adds binary high_delinquency_risk flag (1 if num_delinquencies >= threshold)."""

    def __init__(self, threshold: int = 2) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DelinquencyRiskFlag:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["high_delinquency_risk"] = (X["num_delinquencies"] >= self.threshold).astype(int)
        return X
