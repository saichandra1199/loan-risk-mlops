output "model_package_group_name" { value = aws_sagemaker_model_package_group.main.model_package_group_name }
output "pipeline_name"           { value = aws_sagemaker_pipeline.training.pipeline_name }
output "pipeline_arn"            { value = aws_sagemaker_pipeline.training.arn }
