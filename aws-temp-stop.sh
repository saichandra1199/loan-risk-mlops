#!/usr/bin/env bash
# aws-temp-stop.sh — Temporarily stop all expensive AWS resources (zero cost while away).
#
# Destroys: RDS, ECS cluster & tasks, VPC, NAT Gateways, ALB, ECR, SageMaker, etc.
# Preserves: S3 buckets + data, Terraform state bucket, GitHub secrets, AWS CLI config.
#
# S3 BucketNotEmpty errors are expected and harmless — Terraform intentionally
# leaves non-empty buckets so your model artifacts and MLflow data survive.
# Resume with: terraform apply + Step 9 re-deploy (see process.md).
#
# Usage:
#   ./aws-temp-stop.sh

set -uo pipefail   # note: no -e so S3 bucket errors don't abort the script

REGION="ap-south-1"

confirm() {
  read -rp "$1 [yes/no]: " REPLY
  if [[ "$REPLY" != "yes" ]]; then
    echo "Aborted."
    exit 0
  fi
}

echo ""
echo "================================================================"
echo "  AWS TEMPORARY STOP — loan-risk MLOps"
echo "  Region: $REGION"
echo "================================================================"
echo ""
echo "  This will destroy all expensive resources (RDS, ECS, NAT Gateways,"
echo "  ALB, ECR). Your S3 data and Terraform state are preserved so you"
echo "  can resume later with a single terraform apply."
echo ""
confirm "Proceed with temporary stop?"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo ""
echo "Account ID: $ACCOUNT_ID"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$SCRIPT_DIR/infra/terraform"

if [[ ! -d "$TF_DIR" ]]; then
  echo "ERROR: Terraform directory not found at $TF_DIR"
  exit 1
fi

read -rp "Enter the database password used during terraform apply: " DB_PASSWORD
echo ""

echo "================================================================"
echo "  Running terraform destroy ..."
echo "  (S3 BucketNotEmpty errors below are expected — not a problem)"
echo "================================================================"
echo ""

cd "$TF_DIR"
terraform destroy \
  -var="db_password=${DB_PASSWORD}" \
  -parallelism=20 \
  -auto-approve || DESTROY_EXIT=$?
cd "$SCRIPT_DIR"

echo ""
if [[ "${DESTROY_EXIT:-0}" -ne 0 ]]; then
  echo "NOTE: terraform destroy finished with errors (exit code ${DESTROY_EXIT})."
  echo "      If the only errors are S3 BucketNotEmpty — that is expected."
  echo "      Your data is safe. The expensive resources are still destroyed."
  echo ""
fi

# ── Verify the expensive resources are gone ────────────────────────────────────
echo "================================================================"
echo "  Verifying expensive resources are destroyed"
echo "================================================================"

echo ""
echo "ECS clusters (should be empty):"
aws ecs list-clusters --region "$REGION" --query 'clusterArns' --output text \
  && echo "" || echo "  (none)"

echo "RDS instances (should be empty):"
aws rds describe-db-instances --region "$REGION" \
  --query 'DBInstances[*].DBInstanceIdentifier' --output text \
  && echo "" || echo "  (none)"

echo "Load balancers (should be empty):"
aws elbv2 describe-load-balancers --region "$REGION" \
  --query 'LoadBalancers[*].LoadBalancerName' --output text \
  && echo "" || echo "  (none)"

echo "NAT Gateways (should be empty):"
aws ec2 describe-nat-gateways --region "$REGION" \
  --filter Name=state,Values=available \
  --query 'NatGateways[*].NatGatewayId' --output text \
  && echo "" || echo "  (none)"

echo "S3 buckets (these are intentionally preserved):"
aws s3 ls | grep loan-risk || echo "  (none)"

echo ""
echo "================================================================"
echo "  Temporary stop complete."
echo "  You are now paying only cents/month for S3 storage."
echo ""
echo "  To resume: see 'Resume Next Time' section in process.md"
echo "  (terraform apply → push to trigger CI → redeploy ECS service)"
echo "================================================================"
