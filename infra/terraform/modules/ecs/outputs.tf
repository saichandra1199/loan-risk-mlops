output "cluster_name"  { value = aws_ecs_cluster.main.name }
output "cluster_arn"   { value = aws_ecs_cluster.main.arn }
output "service_name"  { value = aws_ecs_service.serving.name }
output "service_arn"   { value = aws_ecs_service.serving.id }
