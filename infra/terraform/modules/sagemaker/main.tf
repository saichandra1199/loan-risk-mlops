resource "aws_sagemaker_model_package_group" "main" {
  model_package_group_name        = "${var.project_name}-classifier"
  model_package_group_description = "Loan risk classifier models — managed by the training pipeline"
}

# The SageMaker Pipeline is defined and managed via the Python SDK (sagemaker/pipeline.py).
# Run `python sagemaker/pipeline.py --upsert` to create/update it.
# Terraform only manages the Model Package Group here; the pipeline ARN is derived
# from the expected name so EventBridge can reference it without owning the resource.
