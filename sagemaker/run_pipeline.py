"""CLI to trigger or manage a SageMaker Pipeline execution.

Usage:
    uv run python sagemaker/run_pipeline.py --stage all
    uv run python sagemaker/run_pipeline.py --stage train --skip-tuning
    uv run python sagemaker/run_pipeline.py --stage all --model-type xgboost --n-trials 100
    uv run python sagemaker/run_pipeline.py --status <execution-arn>
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import boto3

PIPELINE_NAME = "loan-risk-training-pipeline"
REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")


def _get_client() -> boto3.client:
    return boto3.client("sagemaker", region_name=REGION)


def start_execution(
    model_type: str = "lgbm",
    skip_tuning: bool = True,
    n_trials: int = 50,
    wait: bool = False,
) -> str:
    """Start a SageMaker Pipeline execution.

    Returns:
        Execution ARN.
    """
    client = _get_client()

    params = [
        {"Name": "model_type", "Value": model_type},
        {"Name": "skip_tuning", "Value": str(skip_tuning)},
        {"Name": "n_trials", "Value": str(n_trials)},
    ]

    response = client.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineParameters=params,
    )
    execution_arn = response["PipelineExecutionArn"]
    print(f"Execution started: {execution_arn}")
    print(
        f"Track at: https://console.aws.amazon.com/sagemaker/home?region={REGION}"
        f"#/pipelines/{PIPELINE_NAME}/executions"
    )

    if wait:
        _wait_for_execution(client, execution_arn)

    return execution_arn


def get_execution_status(execution_arn: str) -> dict:
    """Return the current status of a pipeline execution."""
    client = _get_client()
    response = client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
    return {
        "status": response["PipelineExecutionStatus"],
        "created": str(response.get("CreationTime", "")),
        "last_modified": str(response.get("LastModifiedTime", "")),
        "failure_reason": response.get("FailureReason", ""),
    }


def _wait_for_execution(client: boto3.client, execution_arn: str) -> None:
    """Poll until execution reaches a terminal state."""
    terminal = {"Succeeded", "Failed", "Stopped"}
    print("Waiting for execution to complete (polling every 30s)...")

    while True:
        resp = client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
        status = resp["PipelineExecutionStatus"]
        print(f"  Status: {status}")

        if status in terminal:
            if status == "Succeeded":
                print("Execution completed successfully.")
            else:
                reason = resp.get("FailureReason", "unknown")
                print(f"Execution {status.lower()}: {reason}", file=sys.stderr)
                sys.exit(1)
            return

        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SageMaker training pipeline")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── start ──────────────────────────────────────────────────────────────────
    start_parser = subparsers.add_parser("start", help="Start a pipeline execution")
    start_parser.add_argument("--stage", default="all", choices=["all", "train", "evaluate"],
                              help="Pipeline stage to run (informational — all stages run)")
    start_parser.add_argument("--model-type", default="lgbm", choices=["lgbm", "xgboost"])
    start_parser.add_argument("--skip-tuning", action="store_true", default=True)
    start_parser.add_argument("--no-skip-tuning", dest="skip_tuning", action="store_false")
    start_parser.add_argument("--n-trials", type=int, default=50)
    start_parser.add_argument("--wait", action="store_true", help="Block until execution completes")

    # ── status ─────────────────────────────────────────────────────────────────
    status_parser = subparsers.add_parser("status", help="Check execution status")
    status_parser.add_argument("execution_arn", help="Pipeline execution ARN")

    # Legacy: support --stage as top-level arg (matches old training.yml interface)
    parser.add_argument("--stage", help=argparse.SUPPRESS)
    parser.add_argument("--skip-tuning", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--model-type", default="lgbm", help=argparse.SUPPRESS)
    parser.add_argument("--n-trials", type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument("--wait", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.command == "start" or args.command is None:
        start_execution(
            model_type=args.model_type,
            skip_tuning=args.skip_tuning,
            n_trials=args.n_trials,
            wait=args.wait,
        )
    elif args.command == "status":
        info = get_execution_status(args.execution_arn)
        for key, value in info.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
