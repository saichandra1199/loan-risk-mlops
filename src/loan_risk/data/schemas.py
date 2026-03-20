"""Pandera schemas for raw and processed loan data.

These schemas are the single source of truth for the data contract
throughout the pipeline. Any schema violation raises DataValidationError.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

from loan_risk.exceptions import DataValidationError
from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)


class RawLoanSchema(pa.DataFrameModel):
    """Schema for raw loan application data (post-ingestion, pre-feature-engineering)."""

    loan_id: str = pa.Field(str_matches=r"^LOAN_\d+$", nullable=False)
    loan_amount: float = pa.Field(ge=100.0, le=100_000.0, nullable=False)
    annual_income: float = pa.Field(ge=1_000.0, le=10_000_000.0, nullable=False)
    employment_years: int = pa.Field(ge=0, le=50, nullable=False)
    credit_score: int = pa.Field(ge=300, le=850, nullable=False)
    debt_to_income_ratio: float = pa.Field(ge=0.0, le=2.0, nullable=False)
    num_open_accounts: int = pa.Field(ge=0, le=100, nullable=False)
    num_delinquencies: int = pa.Field(ge=0, le=50, nullable=False)
    loan_purpose: str = pa.Field(
        isin=[
            "debt_consolidation",
            "home_improvement",
            "major_purchase",
            "medical",
            "vacation",
            "other",
        ],
        nullable=False,
    )
    home_ownership: str = pa.Field(
        isin=["RENT", "MORTGAGE", "OWN"],
        nullable=False,
    )
    loan_term_months: int = pa.Field(isin=[36, 60], nullable=False)
    loan_default: int = pa.Field(isin=[0, 1], nullable=False)

    class Config:
        coerce = True
        drop_invalid_rows = False


class InferenceInputSchema(pa.DataFrameModel):
    """Schema for inference requests (no target column required)."""

    loan_amount: float = pa.Field(ge=100.0, le=100_000.0, nullable=False)
    annual_income: float = pa.Field(ge=1_000.0, le=10_000_000.0, nullable=False)
    employment_years: int = pa.Field(ge=0, le=50, nullable=False)
    credit_score: int = pa.Field(ge=300, le=850, nullable=False)
    debt_to_income_ratio: float = pa.Field(ge=0.0, le=2.0, nullable=False)
    num_open_accounts: int = pa.Field(ge=0, le=100, nullable=False)
    num_delinquencies: int = pa.Field(ge=0, le=50, nullable=False)
    loan_purpose: str = pa.Field(
        isin=[
            "debt_consolidation",
            "home_improvement",
            "major_purchase",
            "medical",
            "vacation",
            "other",
        ],
        nullable=False,
    )
    home_ownership: str = pa.Field(isin=["RENT", "MORTGAGE", "OWN"], nullable=False)
    loan_term_months: int = pa.Field(isin=[36, 60], nullable=False)

    class Config:
        coerce = True


def validate_raw(df: pl.DataFrame) -> pl.DataFrame:
    """Validate raw data against RawLoanSchema.

    Args:
        df: Raw loan DataFrame.

    Returns:
        Validated DataFrame (same object if valid).

    Raises:
        DataValidationError: If any schema constraint is violated.
    """
    logger.info("validating_raw_data", n_rows=len(df))
    try:
        validated = RawLoanSchema.validate(df)
        logger.info("raw_validation_passed", n_rows=len(validated))
        return validated
    except pa.errors.SchemaError as exc:
        errors = [str(exc)]
        logger.error("raw_validation_failed", errors=errors)
        raise DataValidationError(
            f"Raw data failed schema validation: {exc}", errors=errors
        ) from exc


def validate_inference_input(df: pl.DataFrame) -> pl.DataFrame:
    """Validate inference input (no target column).

    Raises:
        DataValidationError: If input fails schema constraints.
    """
    try:
        return InferenceInputSchema.validate(df)
    except pa.errors.SchemaError as exc:
        raise DataValidationError(
            f"Inference input failed validation: {exc}", errors=[str(exc)]
        ) from exc
