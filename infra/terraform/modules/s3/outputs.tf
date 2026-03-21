output "data_bucket_name"      { value = aws_s3_bucket.data.bucket }
output "artifacts_bucket_name" { value = aws_s3_bucket.artifacts.bucket }
output "mlflow_bucket_name"    { value = aws_s3_bucket.mlflow.bucket }
output "data_bucket_arn"       { value = aws_s3_bucket.data.arn }
output "artifacts_bucket_arn"  { value = aws_s3_bucket.artifacts.arn }
output "mlflow_bucket_arn"     { value = aws_s3_bucket.mlflow.arn }
