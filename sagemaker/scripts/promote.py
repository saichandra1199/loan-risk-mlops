"""Promote a SageMaker Model Package to Approved status and update MLflow champion alias.

Usage:
    python sagemaker/scripts/promote.py \
        --metrics-path s3://loan-risk-artifacts/evaluation/evaluation_report.json \
        --model-package-arn arn:aws:sagemaker:ap-south-1:123456789:model-package/loan-risk-classifier/1 \
        --mlflow-run-id abc123
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_metrics(metrics_path: str) -> dict:
    """Load evaluation metrics from S3 or local path."""
    if metrics_path.startswith("s3://"):
        import boto3  # noqa: PLC0415
        s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"))
        bucket, key = metrics_path[5:].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    else:
        return json.loads(Path(metrics_path).read_text())


def promote_sagemaker_model(model_package_arn: str, auc: float, threshold: float = 0.80) -> bool:
    """Set SageMaker model package approval status to Approved if AUC passes gate."""
    import boto3  # noqa: PLC0415

    if auc < threshold:
        print(f"AUC {auc:.4f} < threshold {threshold:.4f} — not promoting")
        return False

    client = boto3.client("sagemaker", region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"))
    client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription=f"Auto-approved: AUC={auc:.4f} >= {threshold}",
    )
    print(f"Model package approved: {model_package_arn}")
    return True


def update_mlflow_champion(run_id: str, test_auc: float) -> None:
    """Register model version and set champion alias in MLflow."""
    from loan_risk.registry.client import MLflowRegistryClient  # noqa: PLC0415

    registry = MLflowRegistryClient()
    version = registry.promote_if_passes_gate(run_id=run_id, test_auc=test_auc)
    if version:
        print(f"MLflow champion updated to version {version.version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote model to champion")
    parser.add_argument("--metrics-path", required=True, help="Path to evaluation_report.json")
    parser.add_argument("--model-package-arn", required=True, help="SageMaker Model Package ARN")
    parser.add_argument("--mlflow-run-id", required=True, help="MLflow run ID")
    parser.add_argument("--auc-threshold", type=float, default=0.80)
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_path)
    test_auc = metrics.get("metrics", metrics).get("test_auc", 0.0)
    print(f"Loaded metrics — test AUC: {test_auc:.4f}")

    promoted = promote_sagemaker_model(
        model_package_arn=args.model_package_arn,
        auc=test_auc,
        threshold=args.auc_threshold,
    )

    if promoted:
        update_mlflow_champion(run_id=args.mlflow_run_id, test_auc=test_auc)
        print("Promotion complete.")
    else:
        print("Promotion skipped — AUC below threshold.")
        sys.exit(1)


if __name__ == "__main__":
    main()
