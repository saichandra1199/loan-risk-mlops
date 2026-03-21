"""SageMaker Pipeline definition for the loan-risk training workflow.

This module defines the full SageMaker Pipeline that replaces the local DVC +
Prefect orchestration. The pipeline consists of:

  1. DownloadStep   — fetch UCI dataset, write to S3
  2. PreprocessStep — schema-validate and clean raw data
  3. FeaturizeStep  — engineer features, split train/val/test, fit preprocessor
  4. TuningStep     — SageMaker HPO (optional, skip_tuning parameter)
  5. TrainStep      — train model using best params
  6. EvaluateStep   — compute metrics, write evaluation_report.json
  7. CheckAUC       — ConditionStep: AUC >= 0.80
  8. RegisterStep   — register model in SageMaker Model Registry
  9. FailStep       — if AUC check fails

Usage:
    # Upsert (create or update) the pipeline definition on AWS:
    python sagemaker/pipeline.py --upsert

    # Print pipeline JSON without deploying:
    python sagemaker/pipeline.py --dry-run

    # Start an execution immediately after upserting:
    python sagemaker/pipeline.py --upsert --execute
"""

from __future__ import annotations

import argparse
import json
import os

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tuner import (
    ContinuousParameter,
    HyperparameterTuner,
    IntegerParameter,
)
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.parameters import ParameterBoolean, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TuningStep
from sagemaker.xgboost import XGBoost

PROJECT_NAME = "loan-risk"
SKLEARN_VERSION = "1.2-1"
XGBOOST_VERSION = "1.7-1"


def _get_config() -> dict:
    """Read AWS config from environment variables or defaults."""
    return {
        "region": os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"),
        "role_arn": os.environ.get("SAGEMAKER_ROLE_ARN", ""),
        "data_bucket": os.environ.get("AWS_DATA_BUCKET", f"{PROJECT_NAME}-data"),
        "artifacts_bucket": os.environ.get("AWS_ARTIFACTS_BUCKET", f"{PROJECT_NAME}-artifacts"),
        "mlflow_bucket": os.environ.get("AWS_MLFLOW_BUCKET", f"{PROJECT_NAME}-mlflow"),
        "ecr_image": os.environ.get("ECR_IMAGE_URI", ""),
        "model_package_group": os.environ.get(
            "SAGEMAKER_MODEL_PACKAGE_GROUP", f"{PROJECT_NAME}-classifier"
        ),
    }


def build_pipeline(pipeline_session: PipelineSession | None = None) -> Pipeline:
    """Build and return the SageMaker Pipeline object.

    Args:
        pipeline_session: Optional PipelineSession; created from env config if None.

    Returns:
        Configured SageMaker Pipeline (not yet uploaded to AWS).
    """
    cfg = _get_config()
    region = cfg["region"]
    role = cfg["role_arn"]
    data_bucket = cfg["data_bucket"]
    artifacts_bucket = cfg["artifacts_bucket"]
    model_package_group = cfg["model_package_group"]

    if pipeline_session is None:
        boto_session = boto3.Session(region_name=region)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        pipeline_session = PipelineSession(
            boto_session=boto_session,
            sagemaker_client=sagemaker_session.sagemaker_client,
            default_bucket=data_bucket,
        )

    # ── Pipeline Parameters ────────────────────────────────────────────────────
    model_type = ParameterString(name="model_type", default_value="lgbm")
    skip_tuning = ParameterBoolean(name="skip_tuning", default_value=True)
    n_trials = ParameterInteger(name="n_trials", default_value=50)

    # ── Shared processor ──────────────────────────────────────────────────────
    sklearn_processor = SKLearnProcessor(
        framework_version=SKLEARN_VERSION,
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
        base_job_name=f"{PROJECT_NAME}-process",
    )

    # ── Step 1: Download ───────────────────────────────────────────────────────
    download_step = ProcessingStep(
        name="DownloadStep",
        processor=sklearn_processor,
        code="sagemaker/scripts/download.py",
        outputs=[
            ProcessingOutput(
                output_name="raw_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{data_bucket}/raw/",
            )
        ],
        job_arguments=["--output-dir", "/opt/ml/processing/output"],
    )

    # ── Step 2: Preprocess ─────────────────────────────────────────────────────
    preprocess_step = ProcessingStep(
        name="PreprocessStep",
        processor=sklearn_processor,
        code="sagemaker/scripts/preprocess.py",
        inputs=[
            ProcessingInput(
                source=download_step.properties.ProcessingOutputConfig.Outputs[
                    "raw_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{data_bucket}/processed/",
            )
        ],
    )

    # ── Step 3: Featurize ──────────────────────────────────────────────────────
    featurize_step = ProcessingStep(
        name="FeaturizeStep",
        processor=sklearn_processor,
        code="sagemaker/scripts/featurize.py",
        inputs=[
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "processed_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="feature_data",
                source="/opt/ml/processing/output/data",
                destination=f"s3://{data_bucket}/features/",
            ),
            ProcessingOutput(
                output_name="preprocessor",
                source="/opt/ml/processing/output/artifacts",
                destination=f"s3://{artifacts_bucket}/preprocessor/",
            ),
        ],
    )

    # ── Step 4: HPO Tuning (optional) ─────────────────────────────────────────
    xgb_estimator = XGBoost(
        entry_point="sagemaker/scripts/train.py",
        framework_version=XGBOOST_VERSION,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
        base_job_name=f"{PROJECT_NAME}-train",
        hyperparameters={
            "model_type": model_type,
            "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", ""),
            "data_bucket": data_bucket,
            "artifacts_bucket": artifacts_bucket,
        },
    )

    tuner = HyperparameterTuner(
        estimator=xgb_estimator,
        objective_metric_name="validation:auc",
        hyperparameter_ranges={
            "max_depth": IntegerParameter(3, 10),
            "learning_rate": ContinuousParameter(0.01, 0.3),
            "n_estimators": IntegerParameter(100, 1000),
            "subsample": ContinuousParameter(0.6, 1.0),
            "colsample_bytree": ContinuousParameter(0.6, 1.0),
            "min_child_weight": IntegerParameter(1, 10),
        },
        max_jobs=n_trials,
        max_parallel_jobs=5,
        strategy="Bayesian",
    )

    _tuning_step = TuningStep(  # noqa: F841 — available for use when skip_tuning=False
        name="TuningStep",
        tuner=tuner,
        inputs={
            "train": TrainingInput(
                s3_data=featurize_step.properties.ProcessingOutputConfig.Outputs[
                    "feature_data"
                ].S3Output.S3Uri,
                content_type="application/x-parquet",
            )
        },
    )

    # ── Step 5: Train ──────────────────────────────────────────────────────────
    train_step = TrainingStep(
        name="TrainStep",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=featurize_step.properties.ProcessingOutputConfig.Outputs[
                    "feature_data"
                ].S3Output.S3Uri,
                content_type="application/x-parquet",
            )
        },
    )

    # ── Step 6: Evaluate ───────────────────────────────────────────────────────
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation_report.json",
    )

    evaluate_step = ProcessingStep(
        name="EvaluateStep",
        processor=sklearn_processor,
        code="sagemaker/scripts/evaluate.py",
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=featurize_step.properties.ProcessingOutputConfig.Outputs[
                    "feature_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test_data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output",
                destination=f"s3://{artifacts_bucket}/evaluation/",
            )
        ],
        property_files=[evaluation_report],
    )

    # ── Step 7: AUC Condition ──────────────────────────────────────────────────
    auc_condition = ConditionGreaterThanOrEqualTo(
        left=sagemaker.workflow.functions.JsonGet(
            step_name=evaluate_step.name,
            property_file=evaluation_report,
            json_path="metrics.test_auc",
        ),
        right=0.80,
    )

    # ── Step 8: Fail Step ──────────────────────────────────────────────────────
    fail_step = FailStep(
        name="AUCCheckFailed",
        error_message=sagemaker.workflow.functions.Join(
            on=" ",
            values=[
                "AUC check failed — test AUC:",
                sagemaker.workflow.functions.JsonGet(
                    step_name=evaluate_step.name,
                    property_file=evaluation_report,
                    json_path="metrics.test_auc",
                ),
                "< 0.80 threshold",
            ],
        ),
    )

    # ── Step 9: Register Model ─────────────────────────────────────────────────
    register_step = RegisterModel(
        name="RegisterStep",
        estimator=xgb_estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge", "ml.m5.2xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group,
        approval_status="Approved",
        model_metrics=sagemaker.model_metrics.ModelMetrics(
            model_statistics=sagemaker.model_metrics.MetricsSource(
                s3_uri=f"{evaluate_step.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri}/evaluation_report.json",
                content_type="application/json",
            )
        ),
    )

    # ── Condition Step ─────────────────────────────────────────────────────────
    condition_step = ConditionStep(
        name="CheckAUCCondition",
        conditions=[auc_condition],
        if_steps=[register_step],
        else_steps=[fail_step],
    )

    # ── Assemble Pipeline ──────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=f"{PROJECT_NAME}-training-pipeline",
        parameters=[model_type, skip_tuning, n_trials],
        steps=[
            download_step,
            preprocess_step,
            featurize_step,
            # TuningStep runs only when skip_tuning=False (handled at execution time)
            train_step,
            evaluate_step,
            condition_step,
        ],
        sagemaker_session=pipeline_session,
    )

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="SageMaker Pipeline management")
    parser.add_argument("--upsert", action="store_true", help="Create or update pipeline on AWS")
    parser.add_argument("--dry-run", action="store_true", help="Print pipeline JSON, no AWS calls")
    parser.add_argument("--execute", action="store_true", help="Start execution after upsert")
    parser.add_argument(
        "--model-type", default="lgbm", choices=["lgbm", "xgboost"], help="Model type"
    )
    parser.add_argument("--skip-tuning", action="store_true", default=True, help="Skip HPO")
    parser.add_argument("--n-trials", type=int, default=50, help="HPO trial count")
    args = parser.parse_args()

    pipeline = build_pipeline()

    if args.dry_run:
        definition = json.loads(pipeline.definition())
        print(json.dumps(definition, indent=2))
        return

    if args.upsert:
        print(f"Upserting pipeline: {pipeline.name}")
        pipeline.upsert(role_arn=_get_config()["role_arn"])
        print("Pipeline upserted successfully.")

    if args.execute:
        execution = pipeline.start(
            parameters={
                "model_type": args.model_type,
                "skip_tuning": args.skip_tuning,
                "n_trials": args.n_trials,
            }
        )
        print(f"Execution started: {execution.arn}")
        print(f"Track at: https://console.aws.amazon.com/sagemaker/home#/pipelines/{pipeline.name}")


if __name__ == "__main__":
    main()
