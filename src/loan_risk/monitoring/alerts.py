"""Drift and performance threshold checks — triggers retraining signal.

A "retraining signal" is a structured dict that downstream orchestration
(SageMaker Pipelines / EventBridge) can act on to kick off a retraining run.
Critical alerts are published to SNS when a topic ARN is configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl

from loan_risk.config import get_settings
from loan_risk.logging_setup import get_logger
from loan_risk.monitoring.drift import compute_feature_psi_all
from loan_risk.monitoring.performance import compute_live_auc

logger = get_logger(__name__)


def _publish_to_sns(topic_arn: str, subject: str, message: dict) -> None:
    """Publish a structured alert payload to SNS. Best-effort."""
    if not topic_arn:
        return
    try:
        import boto3  # noqa: PLC0415
        cfg = get_settings()
        sns = boto3.client("sns", region_name=cfg.aws.region)
        sns.publish(
            TopicArn=topic_arn,
            Subject=subject[:100],
            Message=json.dumps(message, indent=2),
        )
        logger.info("sns_alert_published", topic_arn=topic_arn, subject=subject)
    except Exception as exc:
        logger.warning("sns_publish_failed", error=str(exc))


@dataclass
class MonitoringAlert:
    """A single monitoring alert."""

    alert_type: str  # drift_psi | performance_degradation | data_quality
    severity: str    # warning | critical
    feature: str | None
    value: float | None
    threshold: float | None
    message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MonitoringResult:
    """Full monitoring check result."""

    alerts: list[MonitoringAlert] = field(default_factory=list)
    retrain_triggered: bool = False
    psi_values: dict[str, float] = field(default_factory=dict)
    live_auc: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def has_critical_alerts(self) -> bool:
        return any(a.severity == "critical" for a in self.alerts)

    def summary(self) -> dict[str, Any]:
        return {
            "n_alerts": len(self.alerts),
            "retrain_triggered": self.retrain_triggered,
            "has_critical": self.has_critical_alerts,
            "psi_values": self.psi_values,
            "live_auc": self.live_auc,
            "timestamp": self.timestamp,
        }


def run_monitoring_checks(
    reference_df: pl.DataFrame,
    current_df: pl.DataFrame,
    check_performance: bool = True,
) -> MonitoringResult:
    """Run all monitoring checks and return a result with triggered alerts.

    Args:
        reference_df: Reference (baseline) dataset.
        current_df: Current production data window.
        check_performance: Whether to also compute live AUC.

    Returns:
        MonitoringResult with alerts and retrain_triggered flag.
    """
    cfg = get_settings()
    result = MonitoringResult()

    # --- PSI Drift Check ---
    psi_values = compute_feature_psi_all(reference_df, current_df)
    result.psi_values = psi_values

    psi_threshold = cfg.monitoring.drift_psi_threshold

    for feature, psi in psi_values.items():
        if psi > psi_threshold * 2:  # 2x threshold = critical
            alert = MonitoringAlert(
                alert_type="drift_psi",
                severity="critical",
                feature=feature,
                value=psi,
                threshold=psi_threshold * 2,
                message=f"Critical PSI drift on '{feature}': {psi:.3f} > {psi_threshold * 2:.3f}",
            )
            result.alerts.append(alert)
            logger.warning("critical_drift_alert", feature=feature, psi=psi)

        elif psi > psi_threshold:
            alert = MonitoringAlert(
                alert_type="drift_psi",
                severity="warning",
                feature=feature,
                value=psi,
                threshold=psi_threshold,
                message=f"Warning PSI drift on '{feature}': {psi:.3f} > {psi_threshold:.3f}",
            )
            result.alerts.append(alert)
            logger.info("warning_drift_alert", feature=feature, psi=psi)

    # --- Performance Check ---
    if check_performance:
        perf = compute_live_auc()
        result.live_auc = perf.get("live_auc")

        if result.live_auc is not None:
            auc_threshold = cfg.monitoring.performance_auc_threshold
            if result.live_auc < auc_threshold:
                alert = MonitoringAlert(
                    alert_type="performance_degradation",
                    severity="critical",
                    feature=None,
                    value=result.live_auc,
                    threshold=auc_threshold,
                    message=(
                        f"Live AUC {result.live_auc:.4f} < threshold {auc_threshold:.4f}"
                    ),
                )
                result.alerts.append(alert)
                logger.warning(
                    "performance_degradation_alert",
                    live_auc=result.live_auc,
                    threshold=auc_threshold,
                )

    # --- Retrain Decision ---
    result.retrain_triggered = result.has_critical_alerts

    if result.retrain_triggered:
        n_critical = sum(1 for a in result.alerts if a.severity == "critical")
        logger.warning("retraining_triggered", n_critical_alerts=n_critical)

        # Publish critical alerts to SNS
        sns_arn = cfg.aws.sns_alert_topic_arn
        if sns_arn:
            _publish_to_sns(
                topic_arn=sns_arn,
                subject=f"[{cfg.model.name}] Retraining triggered: {n_critical} critical alerts",
                message={
                    "retrain_triggered": True,
                    "n_critical_alerts": n_critical,
                    "alerts": [
                        {
                            "type": a.alert_type,
                            "severity": a.severity,
                            "feature": a.feature,
                            "value": a.value,
                            "threshold": a.threshold,
                            "message": a.message,
                        }
                        for a in result.alerts
                        if a.severity == "critical"
                    ],
                    "psi_values": result.psi_values,
                    "live_auc": result.live_auc,
                    "timestamp": result.timestamp,
                },
            )

    return result
