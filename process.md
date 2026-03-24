# AWS Deployment Guide — Loan Risk MLOps Pipeline

This document walks you through every step needed to deploy the loan-risk MLOps
pipeline on AWS — from creating an account to running your first prediction.

**Estimated time:** 2–3 hours (most of it is waiting for AWS to provision resources)
**Estimated monthly cost:** ~$100/month (see cost breakdown at the bottom)

---

## Prerequisites

- A computer running Linux or WSL2 (Windows Subsystem for Linux)
- A GitHub account with access to this repository
- A credit card (required by AWS even for free tier)

---

## Overview of Steps

```
1. Create AWS Account
2. Install Tools (AWS CLI, Terraform, Docker)
3. Create IAM Admin User
4. Configure AWS CLI
5. Run Bootstrap Script (one-time setup)
6. Set GitHub Secrets & Variables
7. Run Terraform (provision all AWS resources)
8. Push Code to Trigger First Docker Build
9. Run the Training Pipeline
10. Verify Everything Works
```

---

## Step 1 — Create an AWS Account

1. Go to **https://aws.amazon.com**
2. Click **"Create an AWS Account"**
3. Enter your email address and choose a root password
4. Select account type: **Personal**
5. Enter your credit card details (required — you won't be charged unless you exceed free tier)
6. Verify your phone number
7. Select **"Basic support"** (free)
8. Sign in to the AWS Console

### Set a Billing Alert (Important — Do This First)

This protects you from unexpected charges:

1. In the AWS Console, click your name (top right) → **Billing and Cost Management**
2. Click **Budgets** → **Create budget**
3. Choose **"Use a template"** → **"Monthly cost budget"**
4. Set amount: `$50`
5. Enter your email address for alerts
6. Click **Create budget**

> You will receive an email if your spending exceeds $50 in any month.

---

## Step 2 — Install Required Tools

Open your terminal and run the following commands:

### AWS CLI

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version
# Expected output: aws-cli/2.x.x Python/3.x.x ...
```

### Terraform

```bash
sudo apt-get update && sudo apt-get install -y gnupg software-properties-common curl

curl -fsSL https://apt.releases.hashicorp.com/gpg | \
  sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
  https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/hashicorp.list

sudo apt-get update && sudo apt-get install -y terraform

# Verify installation
terraform --version
# Expected output: Terraform v1.6.x
```

### Docker

```bash
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER

# Log out and log back in, then verify
docker --version
# Expected output: Docker version 24.x.x ...
```

---

## Step 3 — Create an IAM Admin User

> **Why?** You should never use your AWS root account for day-to-day work.
> An IAM user with admin permissions is safer.

1. In the AWS Console, search for **IAM** in the top search bar
2. Click **Users** in the left sidebar → **Create user**
3. Set username: `loan-risk-admin`
4. Check **"Provide user access to the AWS Management Console"**
5. Select **"I want to create an IAM user"**
6. Set a console password → click **Next**
7. Select **"Attach policies directly"**
8. Search for and select **AdministratorAccess**
9. Click **Next** → **Create user**

### Create Access Keys for the CLI

1. Click on the user `loan-risk-admin` you just created
2. Go to the **Security credentials** tab
3. Scroll down to **Access keys** → click **Create access key**
4. Select **"Command Line Interface (CLI)"** → check the confirmation box → **Next**
5. Click **Create access key**
6. **IMPORTANT:** Copy both values now — you cannot see the secret again:
   - Access key ID (looks like: `AKIAIOSFODNN7EXAMPLE`)
   - Secret access key (looks like: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`)

---

## Step 4 — Configure AWS CLI

Run this in your terminal and paste the values from Step 3:

```bash
aws configure
```

You will be prompted for:

```
AWS Access Key ID:      <paste your Access Key ID>
AWS Secret Access Key:  <paste your Secret Access Key>
Default region name:    ap-south-1
Default output format:  json
```

### Verify It Works

```bash
aws sts get-caller-identity
```

Expected output (your account ID will be different):
```json
{
    "UserId": "AIDAIOSFODNN7EXAMPLE",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/loan-risk-admin"
}
```

If you see your account ID, your CLI is configured correctly.

---

## Step 5 — Run the Bootstrap Script

> **What this does:** Creates 4 things that must exist before Terraform can run:
> 1. An S3 bucket to store Terraform's state file
> 2. A DynamoDB table to prevent concurrent Terraform runs
> 3. A GitHub OIDC identity provider so GitHub Actions can authenticate to AWS
> 4. An IAM role that GitHub Actions will assume to deploy resources

Navigate to the project directory and run:

```bash
cd /path/to/your/loan-risk-mlops   # replace with your actual path

chmod +x infra/bootstrap.sh
./infra/bootstrap.sh
```

The script takes about 1–2 minutes. At the end it prints a summary like this:

```
================================================================
  Bootstrap complete!
================================================================

  Account ID:          123456789012
  TF State Bucket:     loan-risk-tf-state-123456789012
  TF Lock Table:       loan-risk-tf-locks
  GitHub Actions Role: arn:aws:iam::123456789012:role/loan-risk-github-actions-role

  NEXT STEPS:
  ...
================================================================
```

**Save this output** — you need the values in the next step.

---

## Step 6 — Set GitHub Secrets and Variables

Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**

### Secrets (sensitive — hidden from logs)

Click **"New repository secret"** for each:

| Secret Name | Value |
|-------------|-------|
| `AWS_ROLE_ARN` | The role ARN from bootstrap output (e.g. `arn:aws:iam::123456789012:role/loan-risk-github-actions-role`) |
| `TF_DB_PASSWORD` | Choose a strong password for the database (e.g. `MyDbPass123`) — **do not use** `/`, `@`, `"`, or spaces (RDS rejects them). Write it down, you will need it in Step 7. |

### Variables (non-sensitive — visible in logs)

Click the **"Variables"** tab → **"New repository variable"** for each:

| Variable Name | Value |
|---------------|-------|
| `AWS_REGION` | `ap-south-1` |
| `AWS_DATA_BUCKET` | `loan-risk-data-<your-account-id>` (replace with your account ID from bootstrap output) |
| `AWS_ARTIFACTS_BUCKET` | `loan-risk-artifacts-<your-account-id>` |
| `ECS_CLUSTER` | `loan-risk-cluster` |
| `ECS_SERVICE` | `loan-risk-serving` |

> `SAGEMAKER_ROLE_ARN` will be added after Step 7 — Terraform outputs it for you.

---

## Step 7 — Run Terraform to Provision AWS Resources

> **What this creates:** VPC, subnets, RDS database, ECS cluster, load balancer,
> ECR image registry, SageMaker model group, CloudWatch dashboards and alarms,
> IAM roles, and Secrets Manager entries — everything the pipeline needs.

```bash
cd infra/terraform

# Download the AWS Terraform provider (~100MB, takes a minute)
terraform init

# Preview what will be created (no changes made yet)
terraform plan -var="db_password=MLopsClaude1"
# Replace MyDbPass123 with the password you chose in Step 6

# Create all resources (takes 10–15 minutes)
terraform apply -var="db_password=MLopsClaude1"
```

When prompted `Do you want to perform these actions?` — type `yes` and press Enter.

### After Apply — Get the SageMaker Role ARN

```bash
terraform output sagemaker_execution_role_arn
```

Copy the output value. Go back to GitHub → **Settings → Secrets and variables → Actions → Variables** and add:

| Variable Name | Value |
|---------------|-------|
| `SAGEMAKER_ROLE_ARN` | The ARN from the terraform output |

### Verify Resources Were Created

```bash
# Check S3 buckets
aws s3 ls | grep loan-risk

# Check ECS cluster
aws ecs list-clusters

# Check RDS instance
aws rds describe-db-instances --query "DBInstances[*].DBInstanceIdentifier"
```

---

## Step 8 — Push Code to Trigger the First Docker Build

The CI workflow automatically builds and pushes the Docker image to ECR whenever
code is pushed to the `main` branch (or the `aws` branch, if that is your main branch).

```bash
# From the project root
git checkout aws          # make sure you're on the aws branch
git push origin aws       # push triggers the CI workflow
```

Go to GitHub → **Actions** tab → you should see the **"CI — Lint & Unit Tests"** workflow running.

Or watch it from your terminal:

```bash
gh run watch --repo <your-github-username>/loan-risk-mlops
```

Once the workflow shows a green checkmark, verify the image landed in ECR:

```bash
aws ecr describe-images \
  --repository-name loan-risk-serving \
  --region ap-south-1 \
  --query 'imageDetails[-1].{tag:imageTags[0],pushed:imagePushedAt}' \
  --output table
```

You should see a row with tag `latest` and a recent timestamp. If the table is empty, the CI push step failed — check the Actions logs.

---

## Step 9 — Deploy the Serving Layer

> **Note:** `terraform apply` creates the ECS service immediately, but the Docker
> image doesn't exist in ECR until the first CI build finishes (Step 8). ECS tasks
> will fail with `CannotPullContainerError` until the image is pushed. This is
> expected — run the commands below after Step 8 completes.

> **Important:** The ECS service has `ignore_changes = [task_definition]` in Terraform,
> which means Terraform will create a new task definition revision but will NOT
> automatically update the running service to use it. You must do this manually
> every time Terraform changes the task definition.

Once the Docker image is in ECR, deploy the latest task definition:

```bash
# Get the latest task definition revision created by Terraform
TASK_DEF=$(aws ecs list-task-definitions \
  --family-prefix loan-risk-serving \
  --region ap-south-1 \
  --sort DESC \
  --query 'taskDefinitionArns[0]' \
  --output text)

echo "Deploying: $TASK_DEF"

# Update the service to use it
aws ecs update-service \
  --cluster loan-risk-cluster \
  --service loan-risk-serving \
  --task-definition $TASK_DEF \
  --region ap-south-1 > /dev/null

# Wait for it to stabilise (2–3 minutes)
aws ecs wait services-stable \
  --cluster loan-risk-cluster \
  --services loan-risk-serving \
  --region ap-south-1

echo "ECS service is running!"

# If any quota change is done, Then run this

 aws service-quotas  list-requested-service-quota-change-history \
    --service-code sagemaker \
    --region ap-south-1 \
    --query 'RequestedQuotas[*].{name:QuotaName,status:Status}' \
    --output table
```

---

## Step 10 — Run the Training Pipeline

Go to GitHub → **Actions** → **"Training Pipeline (SageMaker)"** → **Run workflow**

Or from your terminal:

```bash
cd /path/to/your/loan-risk-mlops

# First, upsert the pipeline definition to SageMaker
export AWS_DEFAULT_REGION=ap-south-1
export SAGEMAKER_ROLE_ARN=$(cd infra/terraform && terraform output -raw sagemaker_execution_role_arn)
export AWS_DATA_BUCKET=loan-risk-data-$(aws sts get-caller-identity --query Account --output text)
export AWS_ARTIFACTS_BUCKET=loan-risk-artifacts-$(aws sts get-caller-identity --query Account --output text)
export PYTHONPATH=src

uv run python sagemaker/pipeline.py --upsert

# Start a training run (skip tuning for the first run — it's faster)
uv run python sagemaker/pipeline.py --upsert --execute --skip-tuning
```

Monitor the pipeline in the AWS Console:
1. Go to **Amazon SageMaker** → **Pipelines**
2. Click **loan-risk-training-pipeline**
3. Click on the latest execution to see each step's status

---

## Step 11 — Verify the Prediction API Works

```bash
# Get the ALB DNS name
ALB_DNS=$(cd infra/terraform && terraform output -raw alb_dns_name)
echo "API endpoint: http://$ALB_DNS"

# Health check
curl http://$ALB_DNS/health

# Test a prediction
curl -X POST http://$ALB_DNS/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_id": "test-001",
    "annual_income": 75000,
    "loan_amount": 15000,
    "loan_term_months": 36,
    "loan_purpose": "debt_consolidation",
    "credit_score": 720,
    "employment_years": 5,
    "home_ownership": "RENT",
    "debt_to_income_ratio": 0.25,
    "num_open_accounts": 4
  }'
```

Expected response:
```json
{
  "prediction": "APPROVE",
  "default_probability": 0.12,
  "confidence": "HIGH",
  "risk_tier": "LOW",
  "model_version": "1",
  "request_id": "...",
  "latency_ms": 45.2
}
```

---

## Monitoring

After the pipeline runs and predictions are served, you can view metrics in CloudWatch:

1. Go to **AWS Console → CloudWatch → Dashboards**
2. Click **loan-risk-dashboard**

You will see:
- Prediction volume and latency
- Default probability distribution
- PSI drift per feature (alerts trigger at PSI > 0.15)
- Live AUC (alerts trigger if AUC drops below 0.75)
- ECS CPU and memory utilisation
- ALB error rates

Alerts are sent by email via SNS. To receive them, go to:
**SNS → Topics → loan-risk-alerts → Create subscription → Protocol: Email → your email**

---

## How Automated Retraining Works

The pipeline automatically retrains via EventBridge schedules (set up by Terraform):

| Schedule | When | What |
|----------|------|------|
| Nightly retrain | Mon–Sat 02:00 UTC | Retrain without HPO (fast, ~30 min) |
| Weekly HPO retrain | Sundays 03:00 UTC | Retrain with hyperparameter tuning (slow, ~2–3 hrs) |

You can also trigger manually from GitHub Actions → **Training Pipeline (SageMaker)** → **Run workflow**.

---

## Stop Now / Resume Next Time (Zero Cost While Away)

### Stop Everything (run this now)

```bash
cd infra/terraform
terraform destroy -var="db_password=MLopsClaude1" -parallelism=20
# When prompted: type 'yes' and press Enter
# Takes ~8–10 minutes
```

> `-parallelism=20` doubles concurrent deletions (default is 10). RDS deletion itself
> takes ~5 min regardless — that is an AWS minimum, not something Terraform can speed up.

> **If destroy fails on S3 buckets** — that means a bucket has data in it. That is fine.
> S3 costs less than $0.10/month for small amounts of data. The expensive resources
> (RDS, ECS, NAT Gateways, ALB) are still destroyed. Your data is safe.

After destroy, you are paying **nothing** except a few cents for any S3 data that remains.

#### What survives the destroy

| Resource | Preserved? | Why |
|----------|-----------|-----|
| S3 buckets + data | ✅ Yes | Terraform won't delete non-empty buckets |
| Terraform state bucket | ✅ Yes | Created by bootstrap, not managed by `terraform destroy` |
| GitHub secrets & variables | ✅ Yes | Stored in GitHub |
| AWS CLI config | ✅ Yes | On your machine |
| RDS database | ❌ Destroyed | Recreated on next apply |
| ECS cluster & tasks | ❌ Destroyed | Recreated on next apply |
| ECR Docker images | ❌ Destroyed | Rebuilt automatically by CI on next push |
| VPC, NAT Gateways, ALB | ❌ Destroyed | Recreated on next apply |

---

### Resume Next Time (start here tomorrow)

Steps 1–6 are already done. Skip them entirely. Pick up from Step 7:

**Step 7 — Recreate infrastructure:**
```bash
cd infra/terraform
terraform apply -var="db_password=MLopsClaude1"
# Takes 10–15 minutes — type 'yes' when prompted
```

**Step 8 — Rebuild the Docker image:**
```bash
git push origin aws   # triggers CI which pushes the image to ECR

# Watch progress
gh run watch --repo saichandra1199/loan-risk-mlops

# Verify image is in ECR
aws ecr describe-images \
  --repository-name loan-risk-serving \
  --region ap-south-1 \
  --query 'imageDetails[-1].{tag:imageTags[0],pushed:imagePushedAt}' \
  --output table
```

**Step 9 — Redeploy the serving layer:**
```bash
# Get the latest task definition (Terraform creates a new revision on each apply)
TASK_DEF=$(aws ecs list-task-definitions \
  --family-prefix loan-risk-serving \
  --region ap-south-1 \
  --sort DESC \
  --query 'taskDefinitionArns[0]' \
  --output text)

echo "Deploying: $TASK_DEF"

aws ecs update-service \
  --cluster loan-risk-cluster \
  --service loan-risk-serving \
  --task-definition $TASK_DEF \
  --region ap-south-1 > /dev/null

aws ecs wait services-stable \
  --cluster loan-risk-cluster \
  --services loan-risk-serving \
  --region ap-south-1

echo "Back online!"
```

> **No need to re-run the training pipeline** unless you want to retrain. The model
> artifacts are in S3 and MLflow will reconnect to them after the RDS schema is rebuilt.

---

## Full Teardown — Stop All AWS Costs Permanently

Follow this if you want to shut down the project completely and stop all AWS charges.
After these steps, your AWS account will have no running resources and no ongoing costs.

> This is **irreversible for data**. S3 buckets with model artifacts and MLflow data
> will be deleted permanently. Only do this if you are done with the project.

### Step 1 — Destroy all Terraform-managed resources

```bash
cd infra/terraform
terraform destroy -var="db_password=MLopsClaude1" -parallelism=20
# Type 'yes' when prompted — takes ~8–10 minutes
```

### Step 2 — Delete S3 buckets (Terraform leaves these behind if they have data)

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

for BUCKET in \
  "loan-risk-data-${ACCOUNT_ID}" \
  "loan-risk-artifacts-${ACCOUNT_ID}" \
  "loan-risk-mlflow-${ACCOUNT_ID}"; do
  echo "Deleting $BUCKET ..."
  # Delete all object versions (required for versioned buckets)
  aws s3api delete-objects \
    --bucket "$BUCKET" \
    --delete "$(aws s3api list-object-versions \
      --bucket "$BUCKET" \
      --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
      --output json 2>/dev/null)" \
    --region ap-south-1 2>/dev/null || true
  # Delete any remaining delete markers
  aws s3api delete-objects \
    --bucket "$BUCKET" \
    --delete "$(aws s3api list-object-versions \
      --bucket "$BUCKET" \
      --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' \
      --output json 2>/dev/null)" \
    --region ap-south-1 2>/dev/null || true
  # Now delete the bucket
  aws s3 rb s3://"$BUCKET" --force --region ap-south-1 2>/dev/null \
    && echo "  Deleted ✓" || echo "  Already gone or skipped"
done
```

### Step 3 — Delete the Terraform state bucket

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
TF_BUCKET="loan-risk-tf-state-${ACCOUNT_ID}"

echo "Deleting Terraform state bucket: $TF_BUCKET"
aws s3 rb s3://"$TF_BUCKET" --force --region ap-south-1
echo "Deleted ✓"
```

### Step 4 — Delete the Terraform state lock table (DynamoDB)

```bash
aws dynamodb delete-table \
  --table-name loan-risk-tf-locks \
  --region ap-south-1
echo "DynamoDB lock table deleted ✓"
```

### Step 5 — Delete the GitHub Actions IAM role and OIDC provider

```bash
# Detach policy from role
aws iam detach-role-policy \
  --role-name loan-risk-github-actions-role \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

# Delete the role
aws iam delete-role --role-name loan-risk-github-actions-role
echo "IAM role deleted ✓"

# Delete the GitHub OIDC provider
OIDC_ARN=$(aws iam list-open-id-connect-providers \
  --query "OpenIDConnectProviderList[?ends_with(Arn,'token.actions.githubusercontent.com')].Arn" \
  --output text)

if [ -n "$OIDC_ARN" ]; then
  aws iam delete-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN"
  echo "OIDC provider deleted ✓"
fi
```

### Step 6 — Delete the IAM admin user (optional)

Only do this if you created the `loan-risk-admin` user solely for this project and no longer need it.

```bash
# Delete access keys first
for KEY_ID in $(aws iam list-access-keys --user-name loan-risk-admin \
  --query 'AccessKeyMetadata[*].AccessKeyId' --output text); do
  aws iam delete-access-key --user-name loan-risk-admin --access-key-id "$KEY_ID"
done

# Detach policies
aws iam detach-user-policy \
  --user-name loan-risk-admin \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

# Delete the user
aws iam delete-user --user-name loan-risk-admin
echo "IAM user deleted ✓"
```

### Step 7 — Verify nothing is running

```bash
# Should return empty lists for all of these
echo "=== Checking for remaining resources ==="

echo "ECS clusters:"
aws ecs list-clusters --region ap-south-1 --query 'clusterArns' --output text

echo "RDS instances:"
aws rds describe-db-instances --region ap-south-1 \
  --query 'DBInstances[*].DBInstanceIdentifier' --output text

echo "Load balancers:"
aws elbv2 describe-load-balancers --region ap-south-1 \
  --query 'LoadBalancers[*].LoadBalancerName' --output text

echo "NAT Gateways:"
aws ec2 describe-nat-gateways --region ap-south-1 \
  --filter Name=state,Values=available \
  --query 'NatGateways[*].NatGatewayId' --output text

echo "S3 buckets:"
aws s3 ls | grep loan-risk || echo "None"
```

All lists should be empty. If anything remains, check the AWS Console → **Billing → Cost Explorer** to confirm $0 projected cost.

### What is now deleted

| Resource | Deleted? |
|---|---|
| ECS cluster, tasks, service | ✅ |
| RDS PostgreSQL database | ✅ |
| VPC, subnets, NAT Gateways, ALB | ✅ |
| ECR repository and images | ✅ |
| SageMaker model package group | ✅ |
| CloudWatch dashboards, alarms, log groups | ✅ |
| Secrets Manager entries | ✅ |
| SNS topic | ✅ |
| EventBridge schedules | ✅ |
| S3 buckets and all data | ✅ |
| Terraform state bucket and lock table | ✅ |
| GitHub Actions IAM role and OIDC provider | ✅ |
| IAM admin user (if Step 6 was run) | ✅ |

Your AWS account is now clean with **zero ongoing costs**.

---

## Cost Breakdown (ap-south-1 / Mumbai, approximate monthly)

| Service | What it does | Cost/month |
|---------|-------------|-----------|
| RDS PostgreSQL db.t3.micro | MLflow tracking database | ~$15 |
| ECS Fargate (2 tasks, 1vCPU/2GB each) | Runs the prediction API | ~$30 |
| NAT Gateways (2) | Private subnet internet access | ~$32 |
| Application Load Balancer | Routes traffic to ECS | ~$16 |
| SageMaker Processing + Training | Runs during pipeline only | ~$5–20 per run |
| S3 (3 buckets) | Data, artifacts, MLflow store | ~$2–5 |
| CloudWatch | Logs, metrics, dashboards | ~$3–5 |
| Secrets Manager | Stores DB credentials | ~$1 |
| **Total (idle)** | | **~$100/month** |

> **Tip:** The NAT Gateways are the most expensive item at $32/month. If cost is a
> concern, you can modify the VPC module to use a single NAT Gateway (reduces HA
> but saves ~$16/month).

---

## Troubleshooting

### `terraform init` fails with "bucket does not exist"
The bootstrap script must be run before `terraform init`. Run `./infra/bootstrap.sh` first.

### GitHub Actions fails with "Not authorized to perform sts:AssumeRoleWithWebIdentity"
- Check that `AWS_ROLE_ARN` secret is set correctly in GitHub
- Verify the OIDC provider was created: AWS Console → IAM → Identity providers
- Verify the role trust policy references your exact GitHub org/repo name

### ECS tasks are failing to start
- Check ECS task logs: AWS Console → ECS → Clusters → loan-risk-cluster → Tasks → (task ID) → Logs
- Common cause: `MLFLOW__TRACKING_URI` secret not injected — verify Secrets Manager has the `loan-risk/mlflow-config` secret with key `MLFLOW_TRACKING_URI` set to the RDS PostgreSQL URI

### ECS service is running but model is not loaded (`"Model not loaded"` response)
- The service may be using a stale task definition revision. Run:
  ```bash
  TASK_DEF=$(aws ecs list-task-definitions --family-prefix loan-risk-serving \
    --region ap-south-1 --sort DESC --query 'taskDefinitionArns[0]' --output text)
  aws ecs update-service --cluster loan-risk-cluster --service loan-risk-serving \
    --task-definition $TASK_DEF --region ap-south-1
  ```
- The `ignore_changes = [task_definition]` lifecycle rule means Terraform never auto-updates the running service — you must run the above after every `terraform apply`

### SageMaker pipeline fails with "service limit 0 Instances"
Your account needs quota increases before the first run. Request all three at once:
```bash
# ml.m5.large for processing jobs (DownloadStep, PreprocessStep, etc.)
aws service-quotas request-service-quota-increase \
  --service-code sagemaker --quota-code L-8541302D \
  --desired-value 2 --region ap-south-1

# ml.m5.xlarge for processing jobs (FeaturizeStep, EvaluateStep)
aws service-quotas request-service-quota-increase \
  --service-code sagemaker --quota-code L-0307F515 \
  --desired-value 2 --region ap-south-1

# ml.m5.xlarge for training jobs (TrainStep)
aws service-quotas request-service-quota-increase \
  --service-code sagemaker --quota-code L-CCE2AFA6 \
  --desired-value 2 --region ap-south-1
```
Wait for approval email (usually minutes, up to 24 hours), then re-run the pipeline.

### SageMaker pipeline step fails for another reason
- Go to SageMaker → Pipelines → loan-risk-training-pipeline → (execution) → (failed step)
- Click "View logs" to see the CloudWatch log output

### `aws configure` credentials not working
- Make sure you created an **IAM user** access key, not a root access key
- Root access keys are visible under your name → Security credentials → Access keys (not recommended)

---

## Next Steps After Setup

Once everything is running, come back and we will:

1. Set up an MLflow server on EC2 (currently using SQLite via the RDS URI — we will add a proper UI)
2. Configure a custom domain for the prediction API
3. Set up the daily monitoring Lambda function
4. Add model A/B testing (champion/challenger)
5. Set up VPC endpoints to reduce NAT gateway costs

---

*Generated by Claude Code — reach out if any step is unclear before proceeding.*
