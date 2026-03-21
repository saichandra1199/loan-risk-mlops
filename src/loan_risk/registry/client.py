"""Model registry clients.

Provides typed interfaces for:
- MLflowRegistryClient: MLflow model registry (champion alias pattern)
- SageMakerRegistryClient: SageMaker Model Registry (model package groups)

Both can be used simultaneously: MLflow manages the serving alias (@champion)
while SageMaker tracks approval status for audit and lineage.
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

    def get_champion_model(self) -> tuple[Any, str]:
        """Load the current champion model from the registry.

        Returns:
            Tuple of (loaded sklearn-compatible model, version string).

        Raises:
            ModelNotFoundError: If no champion model exists.
        """
        alias = self.cfg.serving.model_alias
        try:
            model_uri = f"models:/{self._model_name}@{alias}"
            model = mlflow.sklearn.load_model(model_uri)
            mv = self._client.get_model_version_by_alias(self._model_name, alias)
            version = str(mv.version)
            logger.info("champion_model_loaded", alias=alias, model_name=self._model_name, version=version)
            return model, version
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


class SageMakerRegistryClient:
    """Typed wrapper around boto3 SageMaker client for model package lifecycle."""

    def __init__(self) -> None:
        self.cfg = get_settings()
        self._group_name = self.cfg.aws.sagemaker_model_package_group
        self._region = self.cfg.aws.region

    def _get_client(self):
        import boto3  # noqa: PLC0415
        return boto3.client("sagemaker", region_name=self._region)

    def promote_to_sagemaker_registry(
        self,
        model_uri: str,
        test_auc: float,
        inference_image: str = "",
        approval_status: str = "Approved",
    ) -> str:
        """Create a ModelPackage in the group and set its approval status.

        Args:
            model_uri: S3 URI to model artifacts (e.g. s3://bucket/prefix/model.tar.gz).
            test_auc: Test AUC to record as a model card metric.
            inference_image: ECR image URI for inference container.
            approval_status: "Approved" or "PendingManualApproval".

        Returns:
            ModelPackage ARN.
        """
        client = self._get_client()

        create_kwargs: dict[str, Any] = {
            "ModelPackageGroupName": self._group_name,
            "ModelApprovalStatus": approval_status,
            "InferenceSpecification": {
                "Containers": [{
                    "Image": inference_image or "763104351884.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
                    "ModelDataUrl": model_uri,
                }],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
                "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.xlarge", "ml.m5.2xlarge"],
            },
            "ModelCardStatus": "Draft",
            "AdditionalInferenceSpecificationsToAdd": [],
        }

        response = client.create_model_package(**create_kwargs)
        arn = response["ModelPackageArn"]

        logger.info(
            "sagemaker_model_package_created",
            arn=arn,
            group=self._group_name,
            test_auc=test_auc,
        )
        return arn

    def get_champion_from_sagemaker(self) -> dict[str, Any] | None:
        """Return the latest Approved model package in the group.

        Returns:
            Dict with arn, creation_time, and metadata; or None if no approved package.
        """
        client = self._get_client()

        response = client.list_model_packages(
            ModelPackageGroupName=self._group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )

        packages = response.get("ModelPackageSummaryList", [])
        if not packages:
            return None

        pkg = packages[0]
        return {
            "arn": pkg["ModelPackageArn"],
            "version": pkg.get("ModelPackageVersion"),
            "creation_time": str(pkg.get("CreationTime", "")),
            "status": pkg.get("ModelApprovalStatus"),
        }

    def list_packages(self, max_results: int = 20) -> list[dict[str, Any]]:
        """List model packages in the group."""
        client = self._get_client()
        response = client.list_model_packages(
            ModelPackageGroupName=self._group_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=max_results,
        )
        return [
            {
                "arn": p["ModelPackageArn"],
                "version": p.get("ModelPackageVersion"),
                "status": p.get("ModelApprovalStatus"),
                "creation_time": str(p.get("CreationTime", "")),
            }
            for p in response.get("ModelPackageSummaryList", [])
        ]
