locals {
  data_bucket      = "${var.project_name}-data-${var.account_id}"
  artifacts_bucket = "${var.project_name}-artifacts-${var.account_id}"
  mlflow_bucket    = "${var.project_name}-mlflow-${var.account_id}"
}

resource "aws_s3_bucket" "data" {
  bucket = local.data_bucket
  tags   = { Name = local.data_bucket, Purpose = "raw-data-dvc-monitoring" }
}

resource "aws_s3_bucket" "artifacts" {
  bucket = local.artifacts_bucket
  tags   = { Name = local.artifacts_bucket, Purpose = "preprocessor-params-reports" }
}

resource "aws_s3_bucket" "mlflow" {
  bucket = local.mlflow_bucket
  tags   = { Name = local.mlflow_bucket, Purpose = "mlflow-artifacts-sagemaker-models" }
}

# Block all public access
resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "mlflow" {
  bucket                  = aws_s3_bucket.mlflow.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Versioning on all buckets
resource "aws_s3_bucket_versioning" "data"      { bucket = aws_s3_bucket.data.id;      versioning_configuration { status = "Enabled" } }
resource "aws_s3_bucket_versioning" "artifacts" { bucket = aws_s3_bucket.artifacts.id; versioning_configuration { status = "Enabled" } }
resource "aws_s3_bucket_versioning" "mlflow"    { bucket = aws_s3_bucket.mlflow.id;    versioning_configuration { status = "Enabled" } }

# Server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule { apply_server_side_encryption_by_default { sse_algorithm = "AES256" } }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule { apply_server_side_encryption_by_default { sse_algorithm = "AES256" } }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id
  rule { apply_server_side_encryption_by_default { sse_algorithm = "AES256" } }
}
