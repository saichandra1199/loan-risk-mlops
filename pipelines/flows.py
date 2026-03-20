"""Prefect @flow definitions for the full training and monitoring pipelines."""

from __future__ import annotations

from typing import Any

from prefect import flow
from prefect.logging import get_run_logger

from pipelines.tasks import (
    engineer_features_task,
    evaluate_and_register_task,
    ingest_data_task,
    run_monitoring_task,
    train_model_task,
    tune_hyperparameters_task,
    validate_data_task,
)


@flow(name="loan-risk-full-pipeline", log_prints=True)
def full_pipeline_flow(
    input_path: str = "data/raw/loans.parquet",
    model_name: str | None = None,
    n_trials: int = 20,
    skip_tuning: bool = False,
) -> dict[str, Any]:
    """Full ML pipeline: ingest → validate → features → tune → train → evaluate → register.

    Args:
        input_path: Path to raw loan data file.
        model_name: Model type to train. Defaults to config setting.
        n_trials: Number of Optuna trials for hyperparameter search.
        skip_tuning: If True, skip tuning and use default model params.

    Returns:
        Dict with training and registration results.
    """
    logger = get_run_logger()
    logger.info(f"Starting full pipeline with input: {input_path}")

    # Stage 1: Ingest
    parquet_path = ingest_data_task(input_path)

    # Stage 2: Validate
    validated_path = validate_data_task(parquet_path)

    # Stage 3: Feature Engineering
    processed_dir, pipeline_path = engineer_features_task(validated_path)

    # Stage 4: Hyperparameter Tuning (optional)
    best_params: dict[str, Any] = {}
    if not skip_tuning:
        best_params = tune_hyperparameters_task(
            processed_dir=processed_dir,
            model_name=model_name,
            n_trials=n_trials,
        )

    # Stage 5: Train
    training_result = train_model_task(
        processed_dir=processed_dir,
        best_params=best_params,
        model_name=model_name,
    )

    # Stage 6: Evaluate + Register
    registration_result = evaluate_and_register_task(training_result)

    result = {
        "training": training_result,
        "registration": registration_result,
        "pipeline_path": pipeline_path,
    }

    logger.info(f"Pipeline complete. Promoted: {registration_result['promoted']}")
    return result


@flow(name="loan-risk-daily-monitor", log_prints=True)
def daily_monitor_flow(
    reference_path: str = "data/reference/reference.parquet",
    current_path: str = "data/monitoring/recent.parquet",
    auto_retrain: bool = True,
) -> dict[str, Any]:
    """Daily monitoring flow: check data drift and trigger retraining if needed.

    Args:
        reference_path: Baseline reference dataset Parquet path.
        current_path: Current production data window.
        auto_retrain: If True and drift detected, trigger full pipeline.

    Returns:
        Monitoring summary with optional retraining result.
    """
    logger = get_run_logger()
    logger.info("Starting daily monitoring check")

    monitoring_result = run_monitoring_task(reference_path, current_path)

    result: dict[str, Any] = {"monitoring": monitoring_result}

    if monitoring_result.get("retrain_triggered") and auto_retrain:
        logger.warning("Drift detected — triggering retraining flow")
        retrain_result = full_pipeline_flow(skip_tuning=True)
        result["retraining"] = retrain_result

    logger.info(f"Monitoring complete. Retrain triggered: {monitoring_result.get('retrain_triggered')}")
    return result
