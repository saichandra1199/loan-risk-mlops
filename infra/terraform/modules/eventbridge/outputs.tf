output "nightly_retrain_arn"    { value = aws_scheduler_schedule.nightly_retrain.arn }
output "weekly_hpo_retrain_arn" { value = aws_scheduler_schedule.weekly_retrain_hpo.arn }
output "daily_monitor_arn"      { value = aws_scheduler_schedule.daily_monitor.arn }
