#!/usr/bin/env python3
"""CLI to run pipeline stages individually or all at once.

Usage:
    uv run python scripts/run_pipeline.py --stage all
    uv run python scripts/run_pipeline.py --stage ingest --input data/raw/loans.csv
    uv run python scripts/run_pipeline.py --stage validate
    uv run python scripts/run_pipeline.py --stage features
    uv run python scripts/run_pipeline.py --stage tune --n-trials 50
    uv run python scripts/run_pipeline.py --stage train
    uv run python scripts/run_pipeline.py --stage evaluate
    uv run python scripts/run_pipeline.py --stage monitor
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path


def run_ingest(input_path: str, cfg) -> str:
    from loan_risk.data.ingestion import load_raw_data

    df = load_raw_data(input_path)
    output_dir = Path(cfg.data.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "loans.parquet"
    df.write_parquet(output_path)
    print(f"[ingest] Loaded {len(df):,} rows → {output_path}")
    return str(output_path)


def run_validate(parquet_path: str) -> None:
    import polars as pl

    from loan_risk.data.schemas import validate_raw

    df = pl.read_parquet(parquet_path)
    validate_raw(df)
    print(f"[validate] Schema validation passed for {len(df):,} rows")


def run_features(parquet_path: str, cfg) -> tuple[str, str]:
    import polars as pl

    from loan_risk.data.splits import stratified_split
    from loan_risk.features.pipeline import build_feature_pipeline, prepare_features, save_pipeline

    df = pl.read_parquet(parquet_path)
    splits = stratified_split(
        df,
        test_size=cfg.training.test_size,
        val_size=cfg.training.val_size,
        random_seed=cfg.training.random_seed,
    )

    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits.train.write_parquet(processed_dir / "train.parquet")
    splits.val.write_parquet(processed_dir / "val.parquet")
    splits.test.write_parquet(processed_dir / "test.parquet")

    # Fit and save pipeline
    pipeline = build_feature_pipeline()
    drop_cols = [c for c in [cfg.data.target_column, cfg.data.id_column] if c in splits.train.columns]
    prepare_features(splits.train.drop(drop_cols), pipeline, fit=True)

    Path("artifacts").mkdir(exist_ok=True)
    pipeline_path = "artifacts/preprocessor.pkl"
    save_pipeline(pipeline, pipeline_path)

    # Save reference for monitoring
    ref_dir = Path("data/reference")
    ref_dir.mkdir(parents=True, exist_ok=True)
    splits.train.write_parquet(ref_dir / "reference.parquet")

    summary = splits.summary()
    print(
        f"[features] train={summary['train']['n_rows']:,} "
        f"val={summary['val']['n_rows']:,} "
        f"test={summary['test']['n_rows']:,}"
    )
    return str(processed_dir), pipeline_path


def run_tune(processed_dir: str, model_name: str, n_trials: int, cfg) -> dict:
    from loan_risk.data.splits import DataSplits
    from loan_risk.tuning.search import run_hyperparameter_search

    splits = DataSplits.from_dir(processed_dir)

    best_params = run_hyperparameter_search(
        splits=splits,
        model_name=model_name,
        n_trials=n_trials,
    )
    print(f"[tune] Best params: {json.dumps(best_params, indent=2)}")
    return best_params


def run_train(processed_dir: str, model_name: str, best_params: dict, cfg):
    from loan_risk.data.splits import DataSplits
    from loan_risk.training.trainer import ModelTrainer

    splits = DataSplits.from_dir(processed_dir)

    trainer = ModelTrainer()
    result = trainer.fit(splits=splits, model_name=model_name, best_params=best_params)
    print(f"[train] run_id={result.run_id}")
    print(f"[train] val_auc={result.val_auc:.4f}  test_auc={result.test_auc:.4f}")

    # Save result for downstream stages (DVC evaluate stage)
    result_path = Path("artifacts/last_training_result.json")
    result_path.parent.mkdir(exist_ok=True)
    result_path.write_text(json.dumps(dataclasses.asdict(result), indent=2))

    return result


def run_evaluate(training_result, cfg) -> None:
    from loan_risk.evaluation.report import EvaluationReport
    from loan_risk.exceptions import ModelPromotionError
    from loan_risk.registry.client import MLflowRegistryClient

    report = EvaluationReport(
        run_id=training_result.run_id,
        model_name=training_result.model_name,
        test_metrics={"auc_roc": training_result.test_auc},
        val_metrics={"auc_roc": training_result.val_auc},
        params=training_result.params,
        threshold=training_result.threshold,
    )
    report_path = report.save()
    print(f"[evaluate] Report saved to {report_path}")

    registry = MLflowRegistryClient()
    promoted = False
    try:
        mv = registry.promote_if_passes_gate(
            run_id=training_result.run_id,
            test_auc=training_result.test_auc,
        )
        if mv:
            promoted = True
            print(f"[evaluate] Model promoted to champion. Version: {mv.version}")
    except ModelPromotionError as exc:
        print(f"[evaluate] Promotion rejected: {exc}")

    # Write DVC-trackable metrics file
    metrics_path = Path("reports/evaluation/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_auc": float(training_result.test_auc),
        "val_auc": float(training_result.val_auc),
        "threshold": float(training_result.threshold),
        "promoted": promoted,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] DVC metrics saved to {metrics_path}")


def run_monitor(cfg) -> None:
    import polars as pl

    from loan_risk.monitoring.alerts import run_monitoring_checks
    from loan_risk.monitoring.drift import generate_drift_report

    ref_path = Path(cfg.monitoring.reference_data_path)
    # Use test split as current-window features (same schema as reference)
    cur_path = Path(cfg.data.processed_dir) / "test.parquet"

    if not ref_path.exists():
        print(f"[monitor] Reference data not found: {ref_path}")
        return
    if not cur_path.exists():
        print(f"[monitor] Current feature data not found: {cur_path}. Run --stage features first.")
        return

    ref_df = pl.read_parquet(ref_path)
    cur_df = pl.read_parquet(cur_path)

    generate_drift_report(ref_df, cur_df)
    result = run_monitoring_checks(ref_df, cur_df)
    print(f"[monitor] Alerts: {len(result.alerts)}, Retrain triggered: {result.retrain_triggered}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Loan risk pipeline CLI")
    parser.add_argument(
        "--stage",
        choices=["all", "ingest", "validate", "features", "tune", "train", "evaluate", "monitor"],
        required=True,
        help="Pipeline stage to run",
    )
    parser.add_argument("--input", default="data/raw/loans.parquet", help="Input data path")
    parser.add_argument("--model", default=None, help="Model type: lgbm | xgboost")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trial count")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning")
    args = parser.parse_args()

    from loan_risk.config import get_settings
    from loan_risk.logging_setup import configure_logging
    configure_logging(json_output=False)
    cfg = get_settings()

    model_name = args.model or cfg.model.name
    n_trials = args.n_trials or cfg.training.n_trials
    raw_parquet = f"{cfg.data.raw_dir}/loans.parquet"

    if args.stage in ("all", "ingest"):
        run_ingest(args.input, cfg)

    if args.stage in ("all", "validate"):
        run_validate(raw_parquet)

    processed_dir = cfg.data.processed_dir
    pipeline_path = "artifacts/preprocessor.pkl"

    if args.stage in ("all", "features"):
        processed_dir, pipeline_path = run_features(raw_parquet, cfg)

    best_params: dict = {}
    if args.stage in ("all", "tune") and not args.skip_tuning:
        best_params = run_tune(processed_dir, model_name, n_trials, cfg)
    elif args.stage in ("all",) and args.skip_tuning:
        # Load previously saved params if available
        param_file = Path(f"artifacts/best_params/{model_name}_best_params.json")
        if param_file.exists():
            with open(param_file) as f:
                best_params = json.load(f).get("best_params", {})
        else:
            print(f"[tune] --skip-tuning: no saved params at {param_file}, using model defaults")

    training_result = None
    if args.stage in ("all", "train"):
        training_result = run_train(processed_dir, model_name, best_params, cfg)

    if args.stage in ("all", "evaluate"):
        if training_result is None:
            # Load from previous train stage
            result_path = Path("artifacts/last_training_result.json")
            if result_path.exists():
                from loan_risk.training.trainer import TrainingResult
                data = json.loads(result_path.read_text())
                training_result = TrainingResult(**data)
                print(f"[evaluate] Loaded training result from {result_path}")
            else:
                print("[evaluate] No training result found. Run --stage train first.")
                return 1
        run_evaluate(training_result, cfg)

    if args.stage in ("all", "monitor"):
        run_monitor(cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
