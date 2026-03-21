resource "aws_secretsmanager_secret" "rds_password" {
  name                    = "${var.project_name}/rds-password"
  description             = "RDS PostgreSQL master password for MLflow tracking backend"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "rds_password" {
  secret_id     = aws_secretsmanager_secret.rds_password.id
  secret_string = var.db_password
}

resource "aws_secretsmanager_secret" "mlflow_config" {
  name                    = "${var.project_name}/mlflow-config"
  description             = "MLflow connection config for ECS tasks and SageMaker"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "mlflow_config" {
  secret_id = aws_secretsmanager_secret.mlflow_config.id
  secret_string = jsonencode({
    MLFLOW_TRACKING_URI = "postgresql://mlflow:${var.db_password}@${var.rds_endpoint}/mlflow"
    MLFLOW_S3_BUCKET    = var.mlflow_bucket
  })
}
