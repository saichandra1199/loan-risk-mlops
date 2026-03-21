#!/usr/bin/env bash
# Bootstrap script — run ONCE before terraform init
# Creates:
#   1. S3 bucket for Terraform state
#   2. DynamoDB table for Terraform state locks
#   3. GitHub OIDC provider in IAM
#   4. IAM role that GitHub Actions can assume
#
# Usage:
#   chmod +x infra/bootstrap.sh
#   ./infra/bootstrap.sh
#
# Prerequisites: aws CLI configured with AdministratorAccess

set -euo pipefail

# ── Configuration — edit these ────────────────────────────────────────────────
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
GITHUB_ORG="saichandra1199"           # your GitHub username or org
GITHUB_REPO="loan-risk-mlops"         # your GitHub repo name
PROJECT="loan-risk"
# ─────────────────────────────────────────────────────────────────────────────

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
TF_STATE_BUCKET="${PROJECT}-tf-state-${ACCOUNT_ID}"
TF_LOCK_TABLE="${PROJECT}-tf-locks"
GITHUB_ACTIONS_ROLE="${PROJECT}-github-actions-role"

echo "=== Bootstrap for account: $ACCOUNT_ID, region: $AWS_REGION ==="
echo ""

# ── 1. S3 bucket for Terraform state ──────────────────────────────────────────
echo "1/4  Creating Terraform state bucket: $TF_STATE_BUCKET"
if aws s3api head-bucket --bucket "$TF_STATE_BUCKET" 2>/dev/null; then
  echo "     Already exists — skipping"
else
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$TF_STATE_BUCKET" --region "$AWS_REGION"
  else
    aws s3api create-bucket --bucket "$TF_STATE_BUCKET" --region "$AWS_REGION" \
      --create-bucket-configuration LocationConstraint="$AWS_REGION"
  fi
  aws s3api put-bucket-versioning --bucket "$TF_STATE_BUCKET" \
    --versioning-configuration Status=Enabled
  aws s3api put-bucket-encryption --bucket "$TF_STATE_BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-public-access-block --bucket "$TF_STATE_BUCKET" \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
  echo "     Created ✓"
fi

# ── 2. DynamoDB table for Terraform locks ─────────────────────────────────────
echo "2/4  Creating Terraform lock table: $TF_LOCK_TABLE"
if aws dynamodb describe-table --table-name "$TF_LOCK_TABLE" --region "$AWS_REGION" 2>/dev/null; then
  echo "     Already exists — skipping"
else
  aws dynamodb create-table \
    --table-name "$TF_LOCK_TABLE" \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "$AWS_REGION"
  echo "     Created ✓"
fi

# ── 3. GitHub OIDC provider ────────────────────────────────────────────────────
echo "3/4  Creating GitHub OIDC identity provider"
GITHUB_OIDC_URL="https://token.actions.githubusercontent.com"
GITHUB_THUMBPRINT="6938fd4d98bab03faadb97b34396831e3780aea1"

EXISTING=$(aws iam list-open-id-connect-providers --query \
  "OpenIDConnectProviderList[?ends_with(Arn, 'token.actions.githubusercontent.com')].Arn" \
  --output text)

if [ -n "$EXISTING" ]; then
  echo "     Already exists: $EXISTING — skipping"
  OIDC_ARN="$EXISTING"
else
  OIDC_ARN=$(aws iam create-open-id-connect-provider \
    --url "$GITHUB_OIDC_URL" \
    --client-id-list "sts.amazonaws.com" \
    --thumbprint-list "$GITHUB_THUMBPRINT" \
    --query OpenIDConnectProviderArn --output text)
  echo "     Created: $OIDC_ARN ✓"
fi

# ── 4. IAM role for GitHub Actions ────────────────────────────────────────────
echo "4/4  Creating GitHub Actions IAM role: $GITHUB_ACTIONS_ROLE"

TRUST_POLICY=$(cat <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "$OIDC_ARN"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:${GITHUB_ORG}/${GITHUB_REPO}:*"
        },
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
POLICY
)

ROLE_ARN=$(aws iam get-role --role-name "$GITHUB_ACTIONS_ROLE" \
  --query Role.Arn --output text 2>/dev/null || true)

if [ -n "$ROLE_ARN" ]; then
  echo "     Already exists: $ROLE_ARN — skipping"
else
  ROLE_ARN=$(aws iam create-role \
    --role-name "$GITHUB_ACTIONS_ROLE" \
    --assume-role-policy-document "$TRUST_POLICY" \
    --query Role.Arn --output text)
  # Attach AdministratorAccess for Terraform (narrow this down after initial setup)
  aws iam attach-role-policy \
    --role-name "$GITHUB_ACTIONS_ROLE" \
    --policy-arn "arn:aws:iam::aws:policy/AdministratorAccess"
  echo "     Created: $ROLE_ARN ✓"
fi

# ── Update Terraform backend config ───────────────────────────────────────────
BACKEND_FILE="infra/terraform/main.tf"
sed -i "s|loan-risk-tf-state|${TF_STATE_BUCKET}|g" "$BACKEND_FILE"
sed -i "s|\"us-east-1\"|\"${AWS_REGION}\"|g" "$BACKEND_FILE"

# ── Print summary ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Bootstrap complete!"
echo "================================================================"
echo ""
echo "  Account ID:          $ACCOUNT_ID"
echo "  TF State Bucket:     $TF_STATE_BUCKET"
echo "  TF Lock Table:       $TF_LOCK_TABLE"
echo "  GitHub Actions Role: $ROLE_ARN"
echo ""
echo "  NEXT STEPS:"
echo ""
echo "  1. Set these GitHub Actions secrets in your repo"
echo "     (Settings → Secrets and variables → Actions):"
echo ""
echo "     Secret name       Value"
echo "     ─────────────     ─────"
echo "     AWS_ROLE_ARN      $ROLE_ARN"
echo "     TF_DB_PASSWORD    <choose a strong password>"
echo ""
echo "  2. Set these GitHub Actions variables:"
echo ""
echo "     Variable name          Value"
echo "     ─────────────────      ─────"
echo "     AWS_REGION             $AWS_REGION"
echo "     AWS_DATA_BUCKET        ${PROJECT}-data-${ACCOUNT_ID}"
echo "     AWS_ARTIFACTS_BUCKET   ${PROJECT}-artifacts-${ACCOUNT_ID}"
echo "     ECS_CLUSTER            ${PROJECT}-cluster"
echo "     ECS_SERVICE            ${PROJECT}-serving"
echo "     SAGEMAKER_ROLE_ARN     (filled in after terraform apply)"
echo ""
echo "  3. Run Terraform:"
echo "     cd infra/terraform"
echo "     terraform init"
echo "     terraform plan -var='db_password=<your-password>'"
echo "     terraform apply -var='db_password=<your-password>'"
echo ""
echo "  4. After apply, get SAGEMAKER_ROLE_ARN:"
echo "     terraform output sagemaker_execution_role_arn"
echo "     → set this as GitHub variable SAGEMAKER_ROLE_ARN"
echo ""
echo "================================================================"
