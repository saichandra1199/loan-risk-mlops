"""Pydantic v2 request/response schemas for the prediction API.

These are the public contract of the /predict endpoint.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoanApplicationRequest(BaseModel):
    """Loan application input for default risk prediction."""

    loan_amount: float = Field(
        ..., ge=100.0, le=100_000.0, description="Requested loan amount in USD"
    )
    annual_income: float = Field(
        ..., ge=1_000.0, le=10_000_000.0, description="Applicant gross annual income"
    )
    employment_years: int = Field(
        ..., ge=0, le=50, description="Years at current employer"
    )
    credit_score: int = Field(
        ..., ge=300, le=850, description="FICO-equivalent credit score"
    )
    debt_to_income_ratio: float = Field(
        ..., ge=0.0, le=2.0, description="Monthly debt payments / monthly income"
    )
    num_open_accounts: int = Field(
        ..., ge=0, le=100, description="Number of active credit lines"
    )
    num_delinquencies: int = Field(
        default=0, ge=0, le=50, description="30+ day late payments in past 2 years"
    )
    loan_purpose: Literal[
        "debt_consolidation",
        "home_improvement",
        "major_purchase",
        "medical",
        "vacation",
        "other",
    ] = Field(..., description="Purpose of the loan")
    home_ownership: Literal["RENT", "MORTGAGE", "OWN"] = Field(
        ..., description="Applicant's housing situation"
    )
    loan_term_months: Literal[36, 60] = Field(
        ..., description="Loan repayment term in months"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "loan_amount": 15000,
            "annual_income": 55000,
            "employment_years": 3,
            "credit_score": 680,
            "debt_to_income_ratio": 0.35,
            "num_open_accounts": 4,
            "num_delinquencies": 0,
            "loan_purpose": "debt_consolidation",
            "home_ownership": "RENT",
            "loan_term_months": 36,
        }
    }}


class RiskFactor(BaseModel):
    """A single SHAP-based risk factor in the prediction explanation."""

    feature: str
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    """Full prediction response with explanation and metadata."""

    prediction: Literal["APPROVE", "REJECT"]
    default_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    risk_tier: Literal["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK", "VERY_HIGH_RISK"]
    top_factors: list[RiskFactor]
    model_version: str
    request_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_name: str
    model_version: str
    model_alias: str
    training_date: str | None
    validation_auc: float | None
    feature_count: int | None


def compute_confidence(probability: float) -> str:
    """Map default probability to a confidence label."""
    distance_from_boundary = abs(probability - 0.5)
    if distance_from_boundary >= 0.30:
        return "HIGH"
    elif distance_from_boundary >= 0.15:
        return "MEDIUM"
    return "LOW"


def compute_risk_tier(probability: float) -> str:
    """Map default probability to a risk tier label."""
    if probability < 0.20:
        return "LOW_RISK"
    elif probability < 0.45:
        return "MEDIUM_RISK"
    elif probability < 0.70:
        return "HIGH_RISK"
    return "VERY_HIGH_RISK"
