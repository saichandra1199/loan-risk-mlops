resource "aws_sagemaker_model_package_group" "main" {
  model_package_group_name        = "${var.project_name}-classifier"
  model_package_group_description = "Loan risk classifier models — managed by the training pipeline"
}

# SageMaker Pipeline is defined via the Python SDK (sagemaker/pipeline.py).
# This placeholder data source lets us reference its ARN in outputs/EventBridge.
# After running `python sagemaker/pipeline.py --upsert`, fill in the name below
# or pass it as a variable.
resource "aws_sagemaker_pipeline" "training" {
  pipeline_name         = "${var.project_name}-training-pipeline"
  pipeline_display_name = "Loan Risk Training Pipeline"
  role_arn              = var.sagemaker_role_arn

  # Pipeline definition is managed by the Python SDK; use a minimal placeholder
  # that will be overwritten on first `python sagemaker/pipeline.py --upsert`.
  pipeline_definition = jsonencode({
    Version = "2020-12-01"
    Metadata = {}
    Parameters = []
    Steps = []
  })
}
