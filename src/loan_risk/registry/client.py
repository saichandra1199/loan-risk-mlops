"""MLflow model registry client.

Provides a typed interface for:
- Promoting a run to the model registry
- Transitioning model versions (Staging → Production → Archived)
- Fetching the latest Production model
- Aliasing models (champion/challenger pattern)
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from loan_risk.config import get_settings
from loan_risk.exceptions import ModelNotFoundError, ModelPromotionError
from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)


class MLflowRegistryClient:
    """Typed wrapper around MlflowClient for model lifecycle management."""

    def __init__(self) -> None:
        self.cfg = get_settings()
        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        self._client = MlflowClient()
        self._model_name = self.cfg.mlflow.registered_model_name

    def promote_if_passes_gate(
        self,
        run_id: str,
        test_auc: float,
        artifact_path: str = "model",
    ) -> ModelVersion | None:
        """Register and promote a model if it passes the AUC gate.

        Args:
            run_id: MLflow run ID containing the model artifact.
            test_auc: Held-out test AUC of this run.
            artifact_path: MLflow artifact path (default: "model").

        Returns:
            ModelVersion if promoted, None if rejected.

        Raises:
            ModelPromotionError: If AUC is below threshold.
        """
        threshold = self.cfg.model.promotion_auc_threshold

        if test_auc < threshold:
            raise ModelPromotionError(self._model_name, test_auc, threshold)

        model_uri = f"runs:/{run_id}/{artifact_path}"
        version = mlflow.register_model(
            model_uri=model_uri,
            name=self._model_name,
        )

        # Set champion alias
        self._client.set_registered_model_alias(
            name=self._model_name,
            alias="champion",
            version=version.version,
        )

        # Archive previous champion (all other versions)
        for mv in self._client.search_model_versions(f"name='{self._model_name}'"):
            if mv.version != version.version:
                self._client.set_model_version_tag(
                    name=self._model_name,
                    version=mv.version,
                    key="status",
                    value="archived",
                )

        self._client.set_model_version_tag(
            name=self._model_name,
            version=version.version,
            key="test_auc",
            value=str(test_auc),
        )

        logger.info(
            "model_promoted",
            run_id=run_id,
            version=version.version,
            test_auc=test_auc,
        )
        return version

    def get_champion_model(self) -> Any:
        """Load the current champion model from the registry.

        Returns:
            Loaded sklearn-compatible model.

        Raises:
            ModelNotFoundError: If no champion model exists.
        """
        alias = self.cfg.serving.model_alias
        try:
            model_uri = f"models:/{self._model_name}@{alias}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("champion_model_loaded", alias=alias, model_name=self._model_name)
            return model
        except Exception as exc:
            raise ModelNotFoundError(self._model_name, version=f"@{alias}") from exc

    def list_versions(self) -> list[dict[str, Any]]:
        """Return summary of all registered model versions."""
        versions = self._client.search_model_versions(f"name='{self._model_name}'")
        return [
            {
                "version": mv.version,
                "run_id": mv.run_id,
                "status": mv.status,
                "tags": mv.tags,
                "creation_timestamp": mv.creation_timestamp,
            }
            for mv in versions
        ]

    def get_version_info(self, version: str) -> dict[str, Any]:
        """Get details for a specific model version."""
        try:
            mv = self._client.get_model_version(name=self._model_name, version=version)
            return {
                "version": mv.version,
                "run_id": mv.run_id,
                "status": mv.status,
                "tags": mv.tags,
            }
        except Exception as exc:
            raise ModelNotFoundError(self._model_name, version=version) from exc
