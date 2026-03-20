"""Unit tests for Pandera data validation schemas."""

from __future__ import annotations

import polars as pl
import pytest

from loan_risk.data.schemas import (
    RawLoanSchema,
    validate_raw,
    validate_inference_input,
)
from loan_risk.exceptions import DataValidationError


def test_raw_schema_valid(sample_df: pl.DataFrame) -> None:
    """Valid data passes schema without errors."""
    validated = validate_raw(sample_df)
    assert len(validated) == len(sample_df)


def test_raw_schema_rejects_invalid_credit_score(sample_df: pl.DataFrame) -> None:
    """Credit score outside 300-850 triggers validation error."""
    bad_df = sample_df.with_columns(pl.lit(200).alias("credit_score"))
    with pytest.raises(DataValidationError):
        validate_raw(bad_df)


def test_raw_schema_rejects_invalid_loan_default(sample_df: pl.DataFrame) -> None:
    """Target column only allows 0 or 1."""
    bad_df = sample_df.with_columns(pl.lit(2).alias("loan_default"))
    with pytest.raises(DataValidationError):
        validate_raw(bad_df)


def test_raw_schema_rejects_invalid_loan_purpose(sample_df: pl.DataFrame) -> None:
    """Unknown loan purpose fails isin check."""
    bad_df = sample_df.with_columns(pl.lit("crypto").alias("loan_purpose"))
    with pytest.raises(DataValidationError):
        validate_raw(bad_df)


def test_inference_schema_no_target(inference_df: pl.DataFrame) -> None:
    """Inference schema works without loan_default column."""
    validated = validate_inference_input(inference_df)
    assert "loan_default" not in validated.columns


def test_inference_schema_rejects_bad_home_ownership(inference_df: pl.DataFrame) -> None:
    """Invalid home ownership value raises DataValidationError."""
    bad_df = inference_df.with_columns(pl.lit("CONDO").alias("home_ownership"))
    with pytest.raises(DataValidationError):
        validate_inference_input(bad_df)
