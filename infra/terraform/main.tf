terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "loan-risk-tf-state-512491905847"
    key            = "prod/terraform.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "loan-risk-tf-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

# ── VPC ──────────────────────────────────────────────────────────────────────
module "vpc" {
  source       = "./modules/vpc"
  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = var.vpc_cidr
}

# ── S3 Buckets ────────────────────────────────────────────────────────────────
module "s3" {
  source      = "./modules/s3"
  account_id  = data.aws_caller_identity.current.account_id
  project_name = var.project_name
  environment  = var.environment
}

# ── ECR ───────────────────────────────────────────────────────────────────────
module "ecr" {
  source       = "./modules/ecr"
  project_name = var.project_name
  environment  = var.environment
}

# ── RDS ───────────────────────────────────────────────────────────────────────
module "rds" {
  source             = "./modules/rds"
  project_name       = var.project_name
  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  rds_sg_id          = module.vpc.rds_sg_id
  db_password        = var.db_password
}

# ── IAM Roles ─────────────────────────────────────────────────────────────────
module "iam" {
  source              = "./modules/iam"
  project_name        = var.project_name
  environment         = var.environment
  account_id          = data.aws_caller_identity.current.account_id
  aws_region          = var.aws_region
  data_bucket_arn     = module.s3.data_bucket_arn
  artifacts_bucket_arn = module.s3.artifacts_bucket_arn
  mlflow_bucket_arn   = module.s3.mlflow_bucket_arn
}

# ── Secrets Manager ───────────────────────────────────────────────────────────
module "secrets" {
  source           = "./modules/secrets"
  project_name     = var.project_name
  environment      = var.environment
  db_password      = var.db_password
  rds_endpoint     = module.rds.endpoint
  mlflow_bucket    = module.s3.mlflow_bucket_name
}

# ── ECS Cluster + Service ─────────────────────────────────────────────────────
module "ecs" {
  source              = "./modules/ecs"
  project_name        = var.project_name
  environment         = var.environment
  aws_region          = var.aws_region
  ecr_image_uri       = "${module.ecr.repository_url}:latest"
  ecs_task_role_arn   = module.iam.ecs_task_role_arn
  ecs_exec_role_arn   = module.iam.ecs_execution_role_arn
  private_subnet_ids  = module.vpc.private_subnet_ids
  ecs_sg_id           = module.vpc.ecs_sg_id
  alb_target_group_arn = module.alb.target_group_arn
  mlflow_secret_arn   = module.secrets.mlflow_secret_arn
  data_bucket         = module.s3.data_bucket_name
  artifacts_bucket    = module.s3.artifacts_bucket_name
}

# ── ALB ───────────────────────────────────────────────────────────────────────
module "alb" {
  source              = "./modules/alb"
  project_name        = var.project_name
  environment         = var.environment
  vpc_id              = module.vpc.vpc_id
  public_subnet_ids   = module.vpc.public_subnet_ids
  alb_sg_id           = module.vpc.alb_sg_id
  acm_certificate_arn = var.acm_certificate_arn
}

# ── SageMaker ─────────────────────────────────────────────────────────────────
module "sagemaker" {
  source               = "./modules/sagemaker"
  project_name         = var.project_name
  environment          = var.environment
  sagemaker_role_arn   = module.iam.sagemaker_role_arn
  aws_region           = var.aws_region
  account_id           = data.aws_caller_identity.current.account_id
}

# ── CloudWatch ────────────────────────────────────────────────────────────────
module "cloudwatch" {
  source       = "./modules/cloudwatch"
  project_name = var.project_name
  environment  = var.environment
  alert_email  = var.alert_email
  aws_region   = var.aws_region
  ecs_cluster  = module.ecs.cluster_name
  ecs_service  = module.ecs.service_name
  alb_arn_suffix = module.alb.alb_arn_suffix
  tg_arn_suffix  = module.alb.target_group_arn_suffix
}

# ── EventBridge ───────────────────────────────────────────────────────────────
module "eventbridge" {
  source                    = "./modules/eventbridge"
  project_name              = var.project_name
  environment               = var.environment
  aws_region                = var.aws_region
  account_id                = data.aws_caller_identity.current.account_id
  eventbridge_role_arn      = module.iam.eventbridge_role_arn
  sagemaker_pipeline_arn    = module.sagemaker.pipeline_arn
}
