"""Central configuration using pydantic-settings.

All config is loaded from:
1. config/settings.yaml (base defaults)
2. Environment variables (override any setting)
3. .env file (loaded automatically if present)

Usage:
    from loan_risk.config import get_settings
    cfg = get_settings()
    print(cfg.mlflow.tracking_uri)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent.parent


class MLflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "loan-risk"
    registered_model_name: str = "loan-risk-classifier"


class ModelConfig(BaseModel):
    name: Literal["lgbm", "xgboost", "catboost"] = "lgbm"
    promotion_auc_threshold: float = 0.80


class TrainingConfig(BaseModel):
    test_size: float = Field(0.20, ge=0.05, le=0.40)
    val_size: float = Field(0.10, ge=0.05, le=0.30)
    n_trials: int = Field(100, ge=1)
    random_seed: int = 42
    early_stopping_rounds: int = 50


class ServingConfig(BaseModel):
    port: int = Field(8000, ge=1024, le=65535)
    model_alias: str = "champion"
    max_batch_size: int = 100


class MonitoringConfig(BaseModel):
    drift_psi_threshold: float = 0.15
    drift_ks_pvalue_threshold: float = 0.05
    performance_auc_threshold: float = 0.75
    reference_data_path: str = "data/reference/reference.parquet"
    prediction_log_path: str = "data/monitoring/predictions.parquet"


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    target_column: str = "loan_default"
    id_column: str = "loan_id"


class FeaturesConfig(BaseModel):
    credit_score_bins: list[int] = [300, 580, 670, 740, 800, 850]
    credit_score_labels: list[str] = ["Poor", "Fair", "Good", "Very Good", "Exceptional"]
    max_dti_ratio: float = 1.0
    log_transform_columns: list[str] = ["annual_income", "loan_amount"]


class AWSConfig(BaseModel):
    region: str = "us-east-1"
    data_bucket: str = "loan-risk-data"
    artifacts_bucket: str = "loan-risk-artifacts"
    mlflow_bucket: str = "loan-risk-mlflow"
    cloudwatch_namespace: str = "LoanRisk"
    sagemaker_model_package_group: str = "loan-risk-classifier"
    prediction_log_prefix: str = "monitoring/predictions/"
    sns_alert_topic_arn: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    mlflow: MLflowConfig = MLflowConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    serving: ServingConfig = ServingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    aws: AWSConfig = AWSConfig()

    @classmethod
    def from_yaml(cls, yaml_path: Path | None = None) -> Settings:
        """Load settings from YAML file, then apply env var overrides."""
        path = yaml_path or (PROJECT_ROOT / "config" / "settings.yaml")
        if path.exists():
            with open(path) as f:
                yaml_data = yaml.safe_load(f) or {}
            # Flatten nested dict for pydantic-settings
            return cls(**yaml_data)
        return cls()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance (loaded from YAML + env vars)."""
    return Settings.from_yaml()
