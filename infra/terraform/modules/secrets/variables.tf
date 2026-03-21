variable "project_name"   { type = string }
variable "environment"    { type = string }
variable "db_password"    { type = string; sensitive = true }
variable "rds_endpoint"   { type = string; sensitive = true }
variable "mlflow_bucket"  { type = string }
