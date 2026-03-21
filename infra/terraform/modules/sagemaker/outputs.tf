locals {
  pipeline_name = "${var.project_name}-training-pipeline"
}

output "model_package_group_name" {
  value = aws_sagemaker_model_package_group.main.model_package_group_name
}

output "pipeline_name" {
  value = local.pipeline_name
}

# Constructed ARN — the pipeline is created by the Python SDK, not Terraform.
# This lets EventBridge reference it without Terraform owning the resource.
output "pipeline_arn" {
  value = "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:pipeline/${local.pipeline_name}"
}
