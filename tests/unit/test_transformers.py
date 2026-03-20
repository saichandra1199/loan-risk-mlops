"""Unit tests for custom feature transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_risk.features.transformers import (
    CreditScoreBinner,
    DelinquencyRiskFlag,
    LogTransformer,
    LoanToIncomeRatioTransformer,
)


@pytest.fixture
def sample_pd(sample_df) -> pd.DataFrame:
    """Convert shared polars fixture to pandas for sklearn transformer tests."""
    return sample_df.to_pandas()


class TestLoanToIncomeRatio:
    def test_adds_column(self, sample_pd: pd.DataFrame) -> None:
        t = LoanToIncomeRatioTransformer()
        out = t.fit_transform(sample_pd)
        assert "loan_to_income_ratio" in out.columns

    def test_ratio_calculation(self) -> None:
        df = pd.DataFrame({"loan_amount": [10000.0], "annual_income": [50000.0]})
        out = LoanToIncomeRatioTransformer().fit_transform(df)
        assert abs(out["loan_to_income_ratio"].iloc[0] - 0.2) < 1e-6

    def test_handles_zero_income(self) -> None:
        df = pd.DataFrame({"loan_amount": [5000.0], "annual_income": [0.0]})
        out = LoanToIncomeRatioTransformer().fit_transform(df)
        assert out["loan_to_income_ratio"].iloc[0] == 0.0

    def test_does_not_modify_original(self, sample_pd: pd.DataFrame) -> None:
        original_cols = list(sample_pd.columns)
        LoanToIncomeRatioTransformer().fit_transform(sample_pd)
        assert list(sample_pd.columns) == original_cols


class TestLogTransformer:
    def test_adds_log_columns(self, sample_pd: pd.DataFrame) -> None:
        t = LogTransformer(columns=["loan_amount", "annual_income"])
        out = t.fit_transform(sample_pd)
        assert "log_loan_amount" in out.columns
        assert "log_annual_income" in out.columns

    def test_log_values_positive(self, sample_pd: pd.DataFrame) -> None:
        t = LogTransformer(columns=["loan_amount"])
        out = t.fit_transform(sample_pd)
        assert (out["log_loan_amount"] >= 0).all()

    def test_log1p_correctness(self) -> None:
        df = pd.DataFrame({"loan_amount": [np.e - 1]})
        out = LogTransformer(columns=["loan_amount"]).fit_transform(df)
        assert abs(out["log_loan_amount"].iloc[0] - 1.0) < 1e-6


class TestCreditScoreBinner:
    def test_adds_band_column(self, sample_pd: pd.DataFrame) -> None:
        t = CreditScoreBinner()
        out = t.fit_transform(sample_pd)
        assert "credit_score_band" in out.columns

    def test_poor_credit_band(self) -> None:
        # 500 < 580, so falls in the first bin: "Poor"
        df = pd.DataFrame({"credit_score": [500]})
        out = CreditScoreBinner().fit_transform(df)
        assert out["credit_score_band"].iloc[0] == "Poor"

    def test_exceptional_credit_band(self) -> None:
        df = pd.DataFrame({"credit_score": [820]})
        out = CreditScoreBinner().fit_transform(df)
        assert out["credit_score_band"].iloc[0] == "Exceptional"

    def test_all_rows_binned(self, sample_pd: pd.DataFrame) -> None:
        out = CreditScoreBinner().fit_transform(sample_pd)
        assert out["credit_score_band"].notna().all()


class TestDelinquencyRiskFlag:
    def test_adds_flag_column(self, sample_pd: pd.DataFrame) -> None:
        out = DelinquencyRiskFlag(threshold=2).fit_transform(sample_pd)
        assert "high_delinquency_risk" in out.columns

    def test_flag_is_binary(self, sample_pd: pd.DataFrame) -> None:
        out = DelinquencyRiskFlag().fit_transform(sample_pd)
        assert set(out["high_delinquency_risk"].unique()).issubset({0, 1})

    def test_threshold_applied(self) -> None:
        df = pd.DataFrame({"num_delinquencies": [0, 1, 2, 3]})
        out = DelinquencyRiskFlag(threshold=2).fit_transform(df)
        expected = [0, 0, 1, 1]
        assert out["high_delinquency_risk"].tolist() == expected
