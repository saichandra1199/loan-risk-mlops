"""Feature group definitions and constants for the loan risk pipeline.

This module is the single source of truth for which features exist,
how they are grouped, and what type they have. Import from here
rather than scattering magic strings across the codebase.
"""

from __future__ import annotations

# Raw numeric features (before any transformation)
NUMERIC_FEATURES: list[str] = [
    "loan_amount",
    "annual_income",
    "employment_years",
    "credit_score",
    "debt_to_income_ratio",
    "num_open_accounts",
    "num_delinquencies",
    "loan_term_months",
]

# Categorical features (will be one-hot encoded)
CATEGORICAL_FEATURES: list[str] = [
    "loan_purpose",
    "home_ownership",
]

# Engineered features added during feature pipeline
ENGINEERED_FEATURES: list[str] = [
    "loan_to_income_ratio",
    "log_loan_amount",
    "log_annual_income",
    "credit_score_band",
]

# Features used for model training (numeric + categorical dummies + engineered)
# This is the full feature set the model sees.
ALL_FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES

# Features to log-transform (right-skewed distributions)
LOG_TRANSFORM_FEATURES: list[str] = ["loan_amount", "annual_income"]

# Credit score bins aligned with FICO tiers
CREDIT_SCORE_BINS: list[int] = [300, 580, 670, 740, 800, 851]
CREDIT_SCORE_LABELS: list[str] = ["Poor", "Fair", "Good", "Very_Good", "Exceptional"]

# Target and ID columns (excluded from features)
TARGET_COLUMN: str = "loan_default"
ID_COLUMN: str = "loan_id"

# Valid values for categorical features
VALID_LOAN_PURPOSES: list[str] = [
    "debt_consolidation",
    "home_improvement",
    "major_purchase",
    "medical",
    "vacation",
    "other",
]

VALID_HOME_OWNERSHIP: list[str] = ["RENT", "MORTGAGE", "OWN"]
VALID_LOAN_TERMS: list[int] = [36, 60]

# Feature groups for SHAP analysis and bias auditing
FEATURE_GROUPS: dict[str, list[str]] = {
    "creditworthiness": ["credit_score", "num_delinquencies", "credit_score_band"],
    "income_capacity": ["annual_income", "log_annual_income", "employment_years"],
    "loan_characteristics": [
        "loan_amount",
        "log_loan_amount",
        "loan_to_income_ratio",
        "loan_term_months",
    ],
    "debt_burden": ["debt_to_income_ratio", "num_open_accounts"],
    "demographics": ["loan_purpose", "home_ownership"],
}
