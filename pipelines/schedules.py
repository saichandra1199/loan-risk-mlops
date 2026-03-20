"""Prefect deployment schedules for automated pipeline runs."""

from __future__ import annotations

from prefect.client.schemas.schedules import CronSchedule

# Nightly retraining at 02:00 UTC
NIGHTLY_RETRAIN_SCHEDULE = CronSchedule(cron="0 2 * * *", timezone="UTC")

# Daily monitoring at 06:00 UTC (after nightly data arrives)
DAILY_MONITOR_SCHEDULE = CronSchedule(cron="0 6 * * *", timezone="UTC")

# Weekly full retraining with tuning on Sundays at 03:00 UTC
WEEKLY_RETRAIN_SCHEDULE = CronSchedule(cron="0 3 * * 0", timezone="UTC")


def deploy_flows() -> None:
    """Deploy flows with their schedules to the Prefect server.

    Run once to register deployments:
        uv run python pipelines/schedules.py
    """
    from pipelines.flows import daily_monitor_flow, full_pipeline_flow

    # Nightly retrain (no tuning, fast)
    full_pipeline_flow.serve(
        name="nightly-retrain",
        schedule=NIGHTLY_RETRAIN_SCHEDULE,
        parameters={"skip_tuning": True},
    )

    # Weekly full retrain with tuning
    full_pipeline_flow.serve(
        name="weekly-retrain-with-tuning",
        schedule=WEEKLY_RETRAIN_SCHEDULE,
        parameters={"skip_tuning": False, "n_trials": 100},
    )

    # Daily monitoring
    daily_monitor_flow.serve(
        name="daily-monitor",
        schedule=DAILY_MONITOR_SCHEDULE,
    )


if __name__ == "__main__":
    deploy_flows()
