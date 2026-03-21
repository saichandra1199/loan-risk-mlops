variable "project_name"  { type = string }
variable "environment"   { type = string }
variable "aws_region"    { type = string }
variable "alert_email"   { type = string; default = "" }
variable "ecs_cluster"   { type = string }
variable "ecs_service"   { type = string }
variable "alb_arn_suffix" { type = string }
variable "tg_arn_suffix"  { type = string }
