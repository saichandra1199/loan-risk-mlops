"""Domain exceptions for the loan risk pipeline."""

from __future__ import annotations


class LoanRiskBaseError(Exception):
    """Base exception for all loan risk pipeline errors."""


class DataValidationError(LoanRiskBaseError):
    """Raised when input data fails Pandera schema validation."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        super().__init__(message)


class DataIngestionError(LoanRiskBaseError):
    """Raised when data cannot be loaded or parsed."""


class FeatureEngineeringError(LoanRiskBaseError):
    """Raised when feature transformation fails."""


class ModelNotFoundError(LoanRiskBaseError):
    """Raised when a requested model version is not in the registry."""

    def __init__(self, model_name: str, version: str | None = None) -> None:
        self.model_name = model_name
        self.version = version
        msg = f"Model '{model_name}'"
        if version:
            msg += f" version '{version}'"
        msg += " not found in registry."
        super().__init__(msg)


class ModelPromotionError(LoanRiskBaseError):
    """Raised when a model fails the promotion gate (e.g., AUC too low)."""

    def __init__(self, model_name: str, auc: float, threshold: float) -> None:
        self.auc = auc
        self.threshold = threshold
        super().__init__(
            f"Model '{model_name}' AUC {auc:.4f} < threshold {threshold:.4f}. "
            "Promotion rejected."
        )


class TrainingError(LoanRiskBaseError):
    """Raised when model training fails."""


class PredictionError(LoanRiskBaseError):
    """Raised when inference fails."""


class MonitoringError(LoanRiskBaseError):
    """Raised when drift detection or monitoring fails."""


class ConfigurationError(LoanRiskBaseError):
    """Raised when configuration is invalid or missing required values."""
