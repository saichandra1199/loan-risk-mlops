"""Shared pytest fixtures for all tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_df() -> pl.DataFrame:
    """200-row synthetic loan DataFrame for unit tests."""
    rng = np.random.default_rng(42)
    n = 200

    return pl.DataFrame(
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


@pytest.fixture(scope="session")
def inference_df(sample_df: pl.DataFrame) -> pl.DataFrame:
    """Sample df without the target column — for testing inference paths."""
    return sample_df.drop("loan_default")
