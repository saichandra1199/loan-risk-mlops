resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Name = "${var.project_name}-cluster" }
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}-serving"
  retention_in_days = 30
}

resource "aws_ecs_task_definition" "serving" {
  family                   = "${var.project_name}-serving"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  task_role_arn            = var.ecs_task_role_arn
  execution_role_arn       = var.ecs_exec_role_arn

  container_definitions = jsonencode([{
    name      = "serving"
    image     = var.ecr_image_uri
    essential = true

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "AWS_DEFAULT_REGION",           value = var.aws_region },
      { name = "AWS_DATA_BUCKET",              value = var.data_bucket },
      { name = "AWS_ARTIFACTS_BUCKET",         value = var.artifacts_bucket },
      { name = "AWS_CLOUDWATCH_NAMESPACE",     value = "LoanRisk" },
    ]

    secrets = [{
      name      = "MLFLOW_TRACKING_URI"
      valueFrom = "${var.mlflow_secret_arn}:MLFLOW_TRACKING_URI::"
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${var.project_name}-serving"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\""]
      interval    = 30
      timeout     = 10
      retries     = 3
      startPeriod = 60
    }
  }])
}

resource "aws_ecs_service" "serving" {
  name            = "${var.project_name}-serving"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.serving.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.ecs_sg_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = var.alb_target_group_arn
    container_name   = "serving"
    container_port   = 8000
  }

  deployment_controller {
    type = "ECS"
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  lifecycle {
    ignore_changes = [task_definition, desired_count]
  }
}
