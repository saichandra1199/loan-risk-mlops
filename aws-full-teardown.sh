#!/usr/bin/env bash
# aws-full-teardown.sh — Permanently remove all AWS resources for loan-risk MLOps.
# WARNING: This is irreversible. All S3 data (model artifacts, MLflow runs) will be deleted.
#
# Usage:
#   ./aws-full-teardown.sh                  # full teardown (skips IAM admin user deletion)
#   ./aws-full-teardown.sh --delete-iam-user # also deletes the loan-risk-admin IAM user
#
# Requirements: aws CLI, terraform, jq (optional), configured AWS credentials

set -euo pipefail

REGION="ap-south-1"
DELETE_IAM_USER=false

for arg in "$@"; do
  case "$arg" in
    --delete-iam-user) DELETE_IAM_USER=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── helpers ────────────────────────────────────────────────────────────────────
confirm() {
  read -rp "$1 [yes/no]: " REPLY
  if [[ "$REPLY" != "yes" ]]; then
    echo "Aborted."
    exit 0
  fi
}

echo ""
echo "================================================================"
echo "  AWS FULL TEARDOWN — loan-risk MLOps"
echo "  Region: $REGION"
echo "================================================================"
echo ""
echo "  WARNING: This will permanently delete ALL AWS resources and data."
echo "  S3 buckets (model artifacts, MLflow data) will be wiped."
echo ""
confirm "Are you sure you want to proceed?"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo ""
echo "Account ID: $ACCOUNT_ID"
echo ""

# ── Step 1 — Terraform destroy ─────────────────────────────────────────────────
echo "================================================================"
echo "  Step 1 — Destroy all Terraform-managed resources"
echo "================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$SCRIPT_DIR/infra/terraform"

if [[ ! -d "$TF_DIR" ]]; then
  echo "ERROR: Terraform directory not found at $TF_DIR"
  exit 1
fi

read -rp "Enter the database password used during terraform apply: " DB_PASSWORD

# ── Step 1a — Empty S3 buckets BEFORE terraform destroy ───────────────────────
# Terraform cannot delete non-empty versioned buckets; empty them first so
# terraform destroy succeeds in one pass.
echo ""
echo "Pre-emptying S3 buckets so terraform destroy can delete them ..."

empty_bucket() {
  local BUCKET="$1"
  # Check the bucket exists first
  aws s3api head-bucket --bucket "$BUCKET" --region "$REGION" 2>/dev/null || return 0

  echo "  Emptying $BUCKET ..."

  # Delete all versioned objects (paginate until none remain)
  while true; do
    VERSIONS=$(aws s3api list-object-versions \
      --bucket "$BUCKET" \
      --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
      --max-items 1000 \
      --output json 2>/dev/null)
    [[ $(echo "$VERSIONS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('Objects') or []))") -eq 0 ]] && break
    aws s3api delete-objects --bucket "$BUCKET" --delete "$VERSIONS" \
      --region "$REGION" > /dev/null 2>&1 || true
  done

  # Delete all delete markers
  while true; do
    MARKERS=$(aws s3api list-object-versions \
      --bucket "$BUCKET" \
      --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' \
      --max-items 1000 \
      --output json 2>/dev/null)
    [[ $(echo "$MARKERS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('Objects') or []))") -eq 0 ]] && break
    aws s3api delete-objects --bucket "$BUCKET" --delete "$MARKERS" \
      --region "$REGION" > /dev/null 2>&1 || true
  done

  # Remove any remaining objects (non-versioned)
  aws s3 rm "s3://${BUCKET}" --recursive --region "$REGION" 2>/dev/null || true
  echo "  $BUCKET emptied ✓"
}

for BUCKET in \
  "loan-risk-data-${ACCOUNT_ID}" \
  "loan-risk-artifacts-${ACCOUNT_ID}" \
  "loan-risk-mlflow-${ACCOUNT_ID}"; do
  empty_bucket "$BUCKET"
done
echo "Buckets emptied — proceeding with terraform destroy"
echo ""

cd "$TF_DIR"
terraform destroy -var="db_password=${DB_PASSWORD}" -parallelism=20 -auto-approve
echo "Terraform destroy complete ✓"
cd "$SCRIPT_DIR"

# ── Step 2 — Delete data S3 buckets (now empty, just remove the shells) ────────
echo ""
echo "================================================================"
echo "  Step 2 — Delete S3 data/artifact buckets"
echo "================================================================"

for BUCKET in \
  "loan-risk-data-${ACCOUNT_ID}" \
  "loan-risk-artifacts-${ACCOUNT_ID}" \
  "loan-risk-mlflow-${ACCOUNT_ID}"; do

  echo "Deleting bucket: $BUCKET"
  aws s3 rb "s3://${BUCKET}" --force --region "$REGION" 2>/dev/null \
    && echo "  Deleted ✓" || echo "  Already gone or skipped"
done

# ── Step 3 — Delete Terraform state bucket ────────────────────────────────────
echo ""
echo "================================================================"
echo "  Step 3 — Delete Terraform state bucket"
echo "================================================================"

TF_BUCKET="loan-risk-tf-state-${ACCOUNT_ID}"
echo "Deleting Terraform state bucket: $TF_BUCKET"
aws s3 rb "s3://${TF_BUCKET}" --force --region "$REGION" 2>/dev/null \
  && echo "Deleted ✓" || echo "Already gone or skipped"

# ── Step 4 — Delete DynamoDB lock table ───────────────────────────────────────
echo ""
echo "================================================================"
echo "  Step 4 — Delete Terraform state lock table (DynamoDB)"
echo "================================================================"

aws dynamodb delete-table \
  --table-name loan-risk-tf-locks \
  --region "$REGION" 2>/dev/null \
  && echo "DynamoDB lock table deleted ✓" || echo "Already gone or skipped"

# ── Step 5 — Delete GitHub Actions IAM role and OIDC provider ─────────────────
echo ""
echo "================================================================"
echo "  Step 5 — Delete GitHub Actions IAM role and OIDC provider"
echo "================================================================"

# Detach policy
aws iam detach-role-policy \
  --role-name loan-risk-github-actions-role \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess 2>/dev/null || true

# Delete role
aws iam delete-role --role-name loan-risk-github-actions-role 2>/dev/null \
  && echo "IAM role deleted ✓" || echo "IAM role already gone or skipped"

# Delete OIDC provider
OIDC_ARN=$(aws iam list-open-id-connect-providers \
  --query "OpenIDConnectProviderList[?ends_with(Arn,'token.actions.githubusercontent.com')].Arn" \
  --output text 2>/dev/null || true)

if [[ -n "$OIDC_ARN" ]]; then
  aws iam delete-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN"
  echo "OIDC provider deleted ✓"
else
  echo "OIDC provider already gone or not found"
fi

# ── Step 6 — Delete IAM admin user (optional) ─────────────────────────────────
if [[ "$DELETE_IAM_USER" == "true" ]]; then
  echo ""
  echo "================================================================"
  echo "  Step 6 — Delete IAM admin user (loan-risk-admin)"
  echo "================================================================"

  # Delete access keys
  for KEY_ID in $(aws iam list-access-keys --user-name loan-risk-admin \
    --query 'AccessKeyMetadata[*].AccessKeyId' --output text 2>/dev/null); do
    aws iam delete-access-key --user-name loan-risk-admin --access-key-id "$KEY_ID"
    echo "  Access key $KEY_ID deleted"
  done

  # Detach policies
  aws iam detach-user-policy \
    --user-name loan-risk-admin \
    --policy-arn arn:aws:iam::aws:policy/AdministratorAccess 2>/dev/null || true

  # Delete user
  aws iam delete-user --user-name loan-risk-admin 2>/dev/null \
    && echo "IAM user loan-risk-admin deleted ✓" || echo "Already gone or skipped"
else
  echo ""
  echo "Step 6 — Skipped (pass --delete-iam-user to also remove loan-risk-admin)"
fi

# ── Step 7 — Verification ──────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Step 7 — Verifying remaining resources"
echo "================================================================"

echo "ECS clusters:"
aws ecs list-clusters --region "$REGION" --query 'clusterArns' --output text || echo "  (none)"

echo "RDS instances:"
aws rds describe-db-instances --region "$REGION" \
  --query 'DBInstances[*].DBInstanceIdentifier' --output text || echo "  (none)"

echo "Load balancers:"
aws elbv2 describe-load-balancers --region "$REGION" \
  --query 'LoadBalancers[*].LoadBalancerName' --output text || echo "  (none)"

echo "NAT Gateways:"
aws ec2 describe-nat-gateways --region "$REGION" \
  --filter Name=state,Values=available \
  --query 'NatGateways[*].NatGatewayId' --output text || echo "  (none)"

echo "S3 buckets (loan-risk*):"
aws s3 ls | grep loan-risk || echo "  (none)"

echo ""
echo "================================================================"
echo "  Teardown complete. Your AWS account has zero ongoing costs."
echo "================================================================"
