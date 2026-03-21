locals {
  pipeline_arn = "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:pipeline/${var.project_name}-training-pipeline"
}

# ── Nightly retraining (no tuning) ────────────────────────────────────────────
resource "aws_scheduler_schedule" "nightly_retrain" {
  name                         = "${var.project_name}-nightly-retrain"
  description                  = "Nightly retraining without HPO — Mon–Sat 02:00 UTC"
  group_name                   = "default"
  schedule_expression          = "cron(0 2 ? * MON-SAT *)"
  schedule_expression_timezone = "UTC"

  flexible_time_window {
    mode                      = "FLEXIBLE"
    maximum_window_in_minutes = 15
  }

  target {
    arn      = var.sagemaker_pipeline_arn
    role_arn = var.eventbridge_role_arn

    sagemaker_pipeline_parameters {
      pipeline_parameter {
        name  = "skip_tuning"
        value = "True"
      }
    }
  }
}

# ── Weekly retraining with HPO ─────────────────────────────────────────────────
resource "aws_scheduler_schedule" "weekly_retrain_hpo" {
  name                         = "${var.project_name}-weekly-retrain-hpo"
  description                  = "Weekly retraining with HPO — Sundays 03:00 UTC"
  group_name                   = "default"
  schedule_expression          = "cron(0 3 ? * SUN *)"
  schedule_expression_timezone = "UTC"

  flexible_time_window {
    mode                      = "FLEXIBLE"
    maximum_window_in_minutes = 30
  }

  target {
    arn      = var.sagemaker_pipeline_arn
    role_arn = var.eventbridge_role_arn

    sagemaker_pipeline_parameters {
      pipeline_parameter {
        name  = "skip_tuning"
        value = "False"
      }
      pipeline_parameter {
        name  = "n_trials"
        value = "50"
      }
    }
  }
}

# ── Daily monitoring (06:00 UTC) ───────────────────────────────────────────────
# Monitoring is triggered via EventBridge → Lambda (Lambda not defined here).
# Create as a disabled placeholder; enable when Lambda is deployed.
resource "aws_scheduler_schedule" "daily_monitor" {
  name                         = "${var.project_name}-daily-monitor"
  description                  = "Daily drift + performance monitoring — 06:00 UTC"
  group_name                   = "default"
  schedule_expression          = "cron(0 6 * * ? *)"
  schedule_expression_timezone = "UTC"
  state                        = "DISABLED"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    # Placeholder — replace with Lambda ARN when monitoring Lambda is deployed
    arn      = "arn:aws:lambda:${var.aws_region}:${var.account_id}:function:${var.project_name}-monitor"
    role_arn = var.eventbridge_role_arn
    input    = jsonencode({ action = "run_monitoring_checks" })
  }
}
