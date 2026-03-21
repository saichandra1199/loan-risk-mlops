output "data_bucket_name" {
  description = "S3 bucket for raw/processed data, DVC cache, and prediction logs"
  value       = module.s3.data_bucket_name
}

output "artifacts_bucket_name" {
  description = "S3 bucket for preprocessor pkl, best_params, and reports"
  value       = module.s3.artifacts_bucket_name
}

output "mlflow_bucket_name" {
  description = "S3 bucket for MLflow artifact store"
  value       = module.s3.mlflow_bucket_name
}

output "ecr_repository_url" {
  description = "ECR repository URI for the serving image"
  value       = module.ecr.repository_url
}

output "alb_dns_name" {
  description = "ALB DNS name — use this as the prediction API endpoint"
  value       = module.alb.dns_name
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint for MLflow tracking server"
  value       = module.rds.endpoint
  sensitive   = true
}

output "sagemaker_execution_role_arn" {
  description = "IAM role ARN for SageMaker Pipeline execution"
  value       = module.iam.sagemaker_role_arn
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = module.ecs.service_name
}

output "sagemaker_model_package_group" {
  description = "SageMaker Model Package Group name"
  value       = module.sagemaker.model_package_group_name
}

output "sns_alert_topic_arn" {
  description = "SNS topic ARN for monitoring alerts"
  value       = module.cloudwatch.sns_topic_arn
}
