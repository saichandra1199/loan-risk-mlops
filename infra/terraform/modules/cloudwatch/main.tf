locals {
  namespace = "LoanRisk"
}

# ── SNS Topic ──────────────────────────────────────────────────────────────────
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
  tags = { Name = "${var.project_name}-alerts" }
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ── Log Groups ─────────────────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "ecs_serving" {
  name              = "/ecs/${var.project_name}-serving"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "sagemaker_pipeline" {
  name              = "/sagemaker/${var.project_name}-pipeline"
  retention_in_days = 90
}

# ── CloudWatch Dashboard ───────────────────────────────────────────────────────
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-dashboard"
  dashboard_body = templatefile("${path.module}/dashboard.json.tpl", {
    namespace   = local.namespace
    region      = var.aws_region
    ecs_cluster = var.ecs_cluster
    ecs_service = var.ecs_service
    alb_suffix  = var.alb_arn_suffix
    tg_suffix   = var.tg_arn_suffix
  })
}

# ── Alarms ─────────────────────────────────────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "psi_high" {
  alarm_name          = "${var.project_name}-psi-high"
  alarm_description   = "PSI exceeds 0.15 — data drift detected, consider retraining"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "PSI"
  namespace           = local.namespace
  period              = 3600
  statistic           = "Maximum"
  threshold           = 0.15
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "auc_low" {
  alarm_name          = "${var.project_name}-auc-low"
  alarm_description   = "Live AUC dropped below 0.75 — model performance degradation"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "LiveAUC"
  namespace           = local.namespace
  period              = 3600
  statistic           = "Average"
  threshold           = 0.75
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "ecs_5xx" {
  alarm_name          = "${var.project_name}-ecs-5xx-rate"
  alarm_description   = "ALB 5xx error rate > 1%"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 1.0
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  metric_query {
    id          = "error_rate"
    expression  = "100 * errors / MAX([errors, requests])"
    label       = "5xx Error Rate (%)"
    return_data = true
  }

  metric_query {
    id = "errors"
    metric {
      metric_name = "HTTPCode_Target_5XX_Count"
      namespace   = "AWS/ApplicationELB"
      period      = 60
      stat        = "Sum"
      dimensions = {
        LoadBalancer = var.alb_arn_suffix
        TargetGroup  = var.tg_arn_suffix
      }
    }
  }

  metric_query {
    id = "requests"
    metric {
      metric_name = "RequestCount"
      namespace   = "AWS/ApplicationELB"
      period      = 60
      stat        = "Sum"
      dimensions = {
        LoadBalancer = var.alb_arn_suffix
        TargetGroup  = var.tg_arn_suffix
      }
    }
  }
}
