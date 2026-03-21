"""Evidently-based data drift detection.

Generates HTML drift reports and structured drift metrics
comparing a reference dataset against current production data.

Reports are uploaded to S3 and PSI metrics are emitted to CloudWatch
when AWS config is available.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from loan_risk.config import get_settings
from loan_risk.exceptions import MonitoringError
from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)


def _upload_report_to_s3(html_path: str) -> None:
    """Upload an Evidently HTML drift report to S3. Best-effort."""
    try:
        import boto3  # noqa: PLC0415
        cfg = get_settings()
        if not cfg.aws.data_bucket or cfg.aws.data_bucket == "loan-risk-data":
            return
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        key = f"monitoring/drift_reports/{timestamp}.html"
        s3 = boto3.client("s3", region_name=cfg.aws.region)
        s3.upload_file(html_path, cfg.aws.data_bucket, key)
        logger.info("drift_report_uploaded_to_s3", bucket=cfg.aws.data_bucket, key=key)
    except Exception as exc:
        logger.warning("drift_report_s3_upload_failed", error=str(exc))


def _emit_psi_to_cloudwatch(psi_values: dict[str, float]) -> None:
    """Emit PSI per feature to CloudWatch. Best-effort."""
    try:
        import boto3  # noqa: PLC0415
        cfg = get_settings()
        if not psi_values:
            return
        cw = boto3.client("cloudwatch", region_name=cfg.aws.region)
        metric_data = [
            {
                "MetricName": "PSI",
                "Dimensions": [{"Name": "Feature", "Value": feature}],
                "Value": psi,
                "Unit": "None",
            }
            for feature, psi in psi_values.items()
        ]
        # CloudWatch accepts max 20 metrics per call
        for i in range(0, len(metric_data), 20):
            cw.put_metric_data(
                Namespace=cfg.aws.cloudwatch_namespace,
                MetricData=metric_data[i:i + 20],
            )
    except Exception as exc:
        logger.warning("cloudwatch_psi_emit_failed", error=str(exc))


def generate_drift_report(
    reference_df: pl.DataFrame,
    current_df: pl.DataFrame,
    output_path: str = "reports/monitoring/drift_report.html",
    column_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate Evidently drift report comparing reference vs. current data.

    Args:
        reference_df: Baseline reference dataset (e.g., training data).
        current_df: Current production data window.
        output_path: Path to save the HTML report.
        column_mapping: Optional Evidently ColumnMapping for target/feature hints.

    Returns:
        Dict with drift summary: dataset_drift, drifted_columns, share_of_drifted_columns.

    Raises:
        MonitoringError: If the report cannot be generated.
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError as exc:
        raise MonitoringError("Evidently not installed. Run: pip install evidently") from exc

    cfg = get_settings()
    target_col = cfg.data.target_column
    id_col = cfg.data.id_column

    ref_cols_to_drop = [id_col] if id_col in reference_df.columns else []
    ref_pd = reference_df.drop(ref_cols_to_drop).to_pandas()

    cur_cols_to_drop = [id_col] if id_col in current_df.columns else []
    cur_pd = current_df.drop(cur_cols_to_drop).to_pandas()

    if column_mapping is None:
        # Auto-detect categorical columns
        cat_cols = [
            c for c in ref_pd.columns
            if ref_pd[c].dtype == object and c != target_col
        ]
        column_mapping = ColumnMapping(
            target=target_col if target_col in ref_pd.columns else None,
            categorical_features=cat_cols,
        )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_pd, current_data=cur_pd, column_mapping=column_mapping)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path_obj))

    # Extract structured summary
    report_dict = report.as_dict()
    drift_summary = _extract_drift_summary(report_dict)

    logger.info(
        "drift_report_generated",
        output_path=output_path,
        dataset_drift=drift_summary.get("dataset_drift"),
        drifted_columns=drift_summary.get("drifted_columns"),
    )

    # Upload HTML report to S3 (best-effort)
    _upload_report_to_s3(str(output_path_obj))

    return drift_summary


def _extract_drift_summary(report_dict: dict[str, Any]) -> dict[str, Any]:
    """Pull key metrics out of the Evidently report dict."""
    try:
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            if "DatasetDriftMetric" in metric.get("metric", ""):
                result = metric.get("result", {})
                return {
                    "dataset_drift": result.get("dataset_drift", False),
                    "drifted_columns": result.get("number_of_drifted_columns", 0),
                    "share_of_drifted_columns": result.get("share_of_drifted_columns", 0.0),
                    "n_features_tested": result.get("number_of_columns", 0),
                }
    except Exception as exc:
        logger.warning("drift_summary_extraction_failed", error=str(exc))

    return {
        "dataset_drift": False,
        "drifted_columns": 0,
        "share_of_drifted_columns": 0.0,
        "n_features_tested": 0,
    }


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI < 0.10: No significant change
    PSI 0.10–0.20: Moderate change, monitor
    PSI > 0.20: Significant change, investigate

    Args:
        reference: Reference distribution (baseline).
        current: Current distribution.
        n_bins: Number of bins for histogram.

    Returns:
        PSI value (float, non-negative).
    """
    import numpy as np

    # Build bins on reference, apply to both
    bins = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    # Avoid division by zero / log(0)
    ref_pct = (ref_counts + 0.0001) / (len(reference) + 0.0001 * n_bins)
    cur_pct = (cur_counts + 0.0001) / (len(current) + 0.0001 * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi)


def compute_feature_psi_all(
    reference_df: pl.DataFrame,
    current_df: pl.DataFrame,
    numeric_features: list[str] | None = None,
) -> dict[str, float]:
    """Compute PSI for all numeric features.

    Args:
        reference_df: Reference Polars DataFrame.
        current_df: Current Polars DataFrame.
        numeric_features: Feature names to compute PSI for.

    Returns:
        Dict mapping feature name -> PSI value.
    """
    if numeric_features is None:
        from loan_risk.features.definitions import NUMERIC_FEATURES
        numeric_features = NUMERIC_FEATURES

    ref_pd = reference_df.to_pandas()
    cur_pd = current_df.to_pandas()

    psi_values: dict[str, float] = {}
    for feature in numeric_features:
        if feature in ref_pd.columns and feature in cur_pd.columns:
            psi = compute_psi(ref_pd[feature].dropna(), cur_pd[feature].dropna())
            psi_values[feature] = round(psi, 4)

    # Emit PSI metrics to CloudWatch (best-effort)
    _emit_psi_to_cloudwatch(psi_values)

    return psi_values
