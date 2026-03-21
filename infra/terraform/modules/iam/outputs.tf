output "ecs_task_role_arn"      { value = aws_iam_role.ecs_task.arn }
output "ecs_execution_role_arn" { value = aws_iam_role.ecs_execution.arn }
output "sagemaker_role_arn"     { value = aws_iam_role.sagemaker.arn }
output "eventbridge_role_arn"   { value = aws_iam_role.eventbridge.arn }
