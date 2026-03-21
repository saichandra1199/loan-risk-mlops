resource "aws_ecr_repository" "serving" {
  name                 = "${var.project_name}-serving"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = { Name = "${var.project_name}-serving" }
}

resource "aws_ecr_lifecycle_policy" "serving" {
  repository = aws_ecr_repository.serving.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}
