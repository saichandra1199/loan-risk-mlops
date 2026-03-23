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
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent.parent


class _YamlSource(PydanticBaseSettingsSource):
    """Lowest-priority source: reads config/settings.yaml."""

    def __init__(self, settings_cls: type[BaseSettings], yaml_path: Path) -> None:
        super().__init__(settings_cls)
        self._data: dict = {}
        if yaml_path.exists():
            with open(yaml_path) as f:
                self._data = yaml.safe_load(f) or {}

    def get_field_value(self, field, field_name):  # type: ignore[override]
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict:
        return self._data


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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = PROJECT_ROOT / "config" / "settings.yaml"
        return (
            init_settings,      # highest priority: explicit kwargs
            env_settings,       # process env vars
            dotenv_settings,    # .env file
            _YamlSource(settings_cls, yaml_path),  # lowest: settings.yaml
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | None = None) -> Settings:
        """Return a Settings instance; sources are ordered by settings_customise_sources."""
        return cls()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance (loaded from YAML + env vars)."""
    return Settings.from_yaml()
