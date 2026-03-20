"""Prefect @task wrappers for each pipeline stage.

Each task wraps one pipeline stage function, adds retry logic,
and emits structured logs for Prefect UI visibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import task

from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)


@task(retries=2, retry_delay_seconds=30, name="ingest-data")
def ingest_data_task(input_path: str) -> str:
    """Load raw data and save to Parquet in data/raw/.

    Args:
        input_path: Path to raw CSV or Parquet file.

    Returns:
        Path to the saved Parquet file.
    """
    from loan_risk.config import get_settings
    from loan_risk.data.ingestion import load_raw_data

    cfg = get_settings()
    df = load_raw_data(input_path)

    output_dir = Path(cfg.data.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "loans.parquet"
    df.write_parquet(output_path)

    logger.info("ingest_task_complete", output_path=str(output_path), n_rows=len(df))
    return str(output_path)


@task(retries=1, name="validate-data")
def validate_data_task(parquet_path: str) -> str:
    """Validate raw Parquet data against RawLoanSchema.

    Args:
        parquet_path: Path to raw Parquet file.

    Returns:
        Same path (pass-through on success).

    Raises:
        DataValidationError: On schema violation.
    """
    import polars as pl

    from loan_risk.data.schemas import validate_raw

    df = pl.read_parquet(parquet_path)
    validate_raw(df)
    logger.info("validation_task_complete", path=parquet_path, n_rows=len(df))
    return parquet_path


@task(retries=1, name="engineer-features")
def engineer_features_task(parquet_path: str) -> tuple[str, str]:
    """Build and fit feature pipeline, save processed splits.

    Args:
        parquet_path: Path to validated raw Parquet.

    Returns:
        Tuple of (processed_dir, pipeline_path).
    """
    import polars as pl

    from loan_risk.config import get_settings
    from loan_risk.data.splits import stratified_split
    from loan_risk.features.pipeline import build_feature_pipeline, save_pipeline

    cfg = get_settings()
    df = pl.read_parquet(parquet_path)
    splits = stratified_split(
        df,
        test_size=cfg.training.test_size,
        val_size=cfg.training.val_size,
        random_seed=cfg.training.random_seed,
    )

    output_dir = Path(cfg.data.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits.train.write_parquet(output_dir / "train.parquet")
    splits.val.write_parquet(output_dir / "val.parquet")
    splits.test.write_parquet(output_dir / "test.parquet")

    pipeline = build_feature_pipeline()
    drop_cols = [c for c in [cfg.data.target_column, cfg.data.id_column] if c in splits.train.columns]
    from loan_risk.features.pipeline import prepare_features
    prepare_features(splits.train.drop(drop_cols), pipeline, fit=True)

    pipeline_path = "artifacts/preprocessor.pkl"
    Path("artifacts").mkdir(exist_ok=True)
    save_pipeline(pipeline, pipeline_path)

    # Save reference dataset for monitoring
    ref_dir = Path("data/reference")
    ref_dir.mkdir(parents=True, exist_ok=True)
    splits.train.write_parquet(ref_dir / "reference.parquet")

    logger.info(
        "feature_engineering_task_complete",
        processed_dir=str(output_dir),
        train=len(splits.train),
        val=len(splits.val),
        test=len(splits.test),
    )
    return str(output_dir), pipeline_path


@task(retries=1, name="tune-hyperparameters")
def tune_hyperparameters_task(
    processed_dir: str,
    model_name: str | None = None,
    n_trials: int = 20,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search.

    Args:
        processed_dir: Directory with train.parquet, val.parquet.
        model_name: Model type to tune.
        n_trials: Number of Optuna trials (reduced for orchestration runs).

    Returns:
        Best hyperparameters dict.
    """
    import polars as pl

    from loan_risk.config import get_settings
    from loan_risk.data.splits import DataSplits
    from loan_risk.tuning.search import run_hyperparameter_search

    cfg = get_settings()
    processed_path = Path(processed_dir)

    splits = DataSplits(
        train=pl.read_parquet(processed_path / "train.parquet"),
        val=pl.read_parquet(processed_path / "val.parquet"),
        test=pl.read_parquet(processed_path / "test.parquet"),
    )

    best_params = run_hyperparameter_search(
        splits=splits,
        model_name=model_name or cfg.model.name,
        n_trials=n_trials,
    )

    logger.info("tuning_task_complete", model_name=model_name, best_params=best_params)
    return best_params


@task(retries=1, name="train-model")
def train_model_task(
    processed_dir: str,
    best_params: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Train a model and log to MLflow.

    Args:
        processed_dir: Directory with train/val/test parquet files.
        best_params: Hyperparameters from tuning step (or None for defaults).
        model_name: Model type to train.

    Returns:
        TrainingResult serialised as dict.
    """
    import dataclasses

    import polars as pl

    from loan_risk.config import get_settings
    from loan_risk.data.splits import DataSplits
    from loan_risk.training.trainer import ModelTrainer

    cfg = get_settings()
    processed_path = Path(processed_dir)

    splits = DataSplits(
        train=pl.read_parquet(processed_path / "train.parquet"),
        val=pl.read_parquet(processed_path / "val.parquet"),
        test=pl.read_parquet(processed_path / "test.parquet"),
    )

    trainer = ModelTrainer()
    result = trainer.fit(
        splits=splits,
        model_name=model_name or cfg.model.name,
        best_params=best_params or {},
    )

    logger.info(
        "training_task_complete",
        run_id=result.run_id,
        val_auc=result.val_auc,
        test_auc=result.test_auc,
    )
    return dataclasses.asdict(result)


@task(retries=1, name="evaluate-and-register")
def evaluate_and_register_task(training_result: dict[str, Any]) -> dict[str, Any]:
    """Run evaluation and attempt model promotion.

    Args:
        training_result: Output from train_model_task.

    Returns:
        Dict with promoted flag, version, and report path.
    """
    from loan_risk.evaluation.report import EvaluationReport
    from loan_risk.exceptions import ModelPromotionError
    from loan_risk.registry.client import MLflowRegistryClient

    run_id = training_result["run_id"]
    test_auc = training_result["test_auc"]
    threshold = training_result["threshold"]

    report = EvaluationReport(
        run_id=run_id,
        model_name=training_result["model_name"],
        test_metrics={"auc_roc": test_auc, "threshold": threshold},
        val_metrics={"auc_roc": training_result["val_auc"]},
        params=training_result.get("params", {}),
        threshold=threshold,
    )

    report_path = report.save()
    promoted = False
    version = None

    try:
        registry = MLflowRegistryClient()
        mv = registry.promote_if_passes_gate(run_id=run_id, test_auc=test_auc)
        if mv is not None:
            promoted = True
            version = mv.version
    except ModelPromotionError as exc:
        logger.warning("promotion_rejected", reason=str(exc))

    result = {
        "promoted": promoted,
        "version": version,
        "report_path": str(report_path),
        "test_auc": test_auc,
    }
    logger.info("evaluate_register_task_complete", **result)
    return result


@task(retries=1, name="run-monitoring")
def run_monitoring_task(reference_path: str, current_path: str) -> dict[str, Any]:
    """Compare current data against reference for drift.

    Args:
        reference_path: Reference (training) dataset Parquet path.
        current_path: Current production data Parquet path.

    Returns:
        MonitoringResult summary dict.
    """
    import polars as pl

    from loan_risk.monitoring.alerts import run_monitoring_checks
    from loan_risk.monitoring.drift import generate_drift_report

    reference_df = pl.read_parquet(reference_path)
    current_df = pl.read_parquet(current_path)

    drift_report_path = "reports/monitoring/drift_report.html"
    generate_drift_report(reference_df, current_df, output_path=drift_report_path)

    monitoring_result = run_monitoring_checks(reference_df, current_df)
    summary = monitoring_result.summary()

    logger.info("monitoring_task_complete", **summary)
    return summary
