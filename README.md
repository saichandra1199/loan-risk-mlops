# 🏦 Loan Risk MLOps Pipeline — AWS Edition

[![CI — Lint & Tests](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/ci.yml/badge.svg?branch=aws)](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/ci.yml)
[![Training Pipeline](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/training.yml/badge.svg?branch=aws)](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/training.yml)
[![Test Serving Layer](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/serve.yml/badge.svg?branch=aws)](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/serve.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Terraform](https://img.shields.io/badge/Terraform-1.6+-purple.svg)](https://www.terraform.io/)
[![AWS](https://img.shields.io/badge/AWS-ap--south--1-orange.svg)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A **production-grade Machine Learning system** that predicts whether a credit card
> customer will default on their next payment — fully deployed on AWS.
> This branch migrates the local pipeline (DVC + Prefect + Prometheus + Docker Compose)
> to managed AWS services: SageMaker Pipelines, ECS Fargate, RDS, CloudWatch, and more.

---

## Table of Contents

1. [What Changed from the Local Branch?](#1-what-changed-from-the-local-branch)
2. [The Dataset](#2-the-dataset)
3. [AWS Architecture](#3-aws-architecture)
4. [AWS Service Mapping — What Replaced What](#4-aws-service-mapping--what-replaced-what)
5. [Tech Stack](#5-tech-stack)
6. [Project Structure — Every File Explained](#6-project-structure--every-file-explained)
7. [How to Deploy to AWS](#7-how-to-deploy-to-aws)
8. [How to Run the Training Pipeline](#8-how-to-run-the-training-pipeline)
9. [API Reference](#9-api-reference)
10. [Monitoring with CloudWatch](#10-monitoring-with-cloudwatch)
11. [Stop, Resume, and Cost Management](#11-stop-resume-and-cost-management)
12. [Complete Removal — Cut All Ties with AWS](#12-complete-removal--cut-all-ties-with-aws)
13. [Troubleshooting](#13-troubleshooting)
14. [FAQ](#14-faq)

---

## 1. What Changed from the Local Branch?

The `main` branch runs everything locally — GitHub Actions runners, Docker Compose,
SQLite, local file storage, and Prometheus. This `aws` branch replaces every one of
those with a managed AWS equivalent so the system can run reliably in production
without keeping your laptop on.

The Python application code (`src/loan_risk/`) is preserved almost entirely as-is.
Only the infrastructure wiring, monitoring sinks, and orchestration layer changed.

**What is the same:**
- The dataset, ML model, feature engineering, and evaluation logic
- The FastAPI prediction API (`src/loan_risk/serving/`)
- All unit and integration tests
- The Pandera schemas, sklearn transformers, and MLflow tracking calls

**What changed:**

| Component | main branch (local) | aws branch (this) |
|-----------|--------------------|--------------------|
| Orchestration | Prefect flows | SageMaker Pipelines |
| Scheduling | Prefect schedules | EventBridge Scheduler |
| Metrics | Prometheus | CloudWatch |
| Storage | Local files | S3 |
| Database | SQLite | RDS PostgreSQL |
| Serving | Docker Compose | ECS Fargate + ALB |
| Image registry | Docker Hub / local | ECR |
| Alerts | Custom scripts | CloudWatch Alarms + SNS |

---

## 2. The Dataset

Same dataset as the `main` branch — UCI Default of Credit Card Clients.

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository (via OpenML) |
| Size | 30,000 rows, 24 columns |
| Target | `default_payment_next_month` (1 = default, 0 = no default) |
| Default rate | ~22% |
| Task | Binary classification |

On AWS, raw data is downloaded to `s3://loan-risk-data-{account_id}/raw/` instead of
`data/raw/` locally. DVC is configured to use that S3 bucket as its remote cache.

---

## 3. AWS Architecture

```
                     ┌──────────────────────────────────────┐
                     │          GitHub Actions               │
                     │  push → CI (lint + test + ECR push)  │
                     │  dispatch → Training Pipeline         │
                     └────────────┬─────────────────────────┘
                                  │  OIDC (no passwords)
               ┌──────────────────▼──────────────────────┐
               │               AWS (ap-south-1)           │
               │                                          │
 Internet ──► ALB (port 80/443)                           │
               │                                          │
               ▼                                          │
     ┌──────────────────┐    ┌──────────────────────┐     │
     │   ECS Fargate    │    │  SageMaker Pipelines  │     │
     │   (FastAPI API)  │    │  (training + HPO)     │     │
     │   2 Fargate tasks│    │  9-step DAG           │     │
     └────────┬─────────┘    └──────────┬───────────┘     │
              │                         │                  │
              ▼                         ▼                  │
     ┌──────────────────────────────────────────────┐      │
     │                  S3 Buckets                   │      │
     │  loan-risk-data      → raw, processed, logs  │      │
     │  loan-risk-artifacts → preprocessor.pkl       │      │
     │  loan-risk-mlflow    → MLflow + model files   │      │
     └──────────────────────────────────────────────┘      │
              │                         │                  │
              ▼                         ▼                  │
     ┌────────────────┐      ┌──────────────────────┐      │
     │  RDS PostgreSQL│      │  SageMaker Model     │      │
     │  (MLflow DB)   │      │  Package Group       │      │
     └────────────────┘      └──────────────────────┘      │
              │                                            │
              ▼                                            │
     ┌──────────────────────────────────────────────┐      │
     │               CloudWatch                      │      │
     │  Custom metrics: PredictionCount, Latency,   │      │
     │                  PSI, LiveAUC                 │      │
     │  Alarms → SNS → Email                        │      │
     │  Dashboard: loan-risk-dashboard               │      │
     └──────────────────────────────────────────────┘      │
                                                           │
     ┌──────────────────────────────────────────────┐      │
     │           EventBridge Scheduler               │      │
     │  Mon–Sat 02:00 UTC → nightly retrain          │      │
     │  Sun    03:00 UTC → weekly HPO retrain        │      │
     └──────────────────────────────────────────────┘      │
               └────────────────────────────────────────┘
```

All of the above is declared in `infra/terraform/` and created with one command.
See [process.md](process.md) for the full deployment walkthrough.

---

## 4. AWS Service Mapping — What Replaced What

| Local / main branch | AWS (this branch) | Why |
|--------------------|--------------------|-----|
| DVC local cache | S3 (`loan-risk-data/dvc-cache/`) | Shared, persistent, versioned |
| MLflow SQLite | RDS PostgreSQL + S3 artifact store | Durable, queryable, multi-user |
| Prefect flows | SageMaker Pipelines (`sagemaker/pipeline.py`) | Managed compute, no server to maintain |
| Prefect schedules | EventBridge Scheduler | Serverless cron, no worker process |
| Optuna HPO | SageMaker Automatic Model Tuning | Parallel trials on managed instances |
| MLflow Model Registry | SageMaker Model Package Group + MLflow alias | Approval workflow + champion alias |
| GitHub Actions training runner | SageMaker Pipeline execution | GPU/CPU on demand, not tied to CI minutes |
| FastAPI on Docker Compose | ECS Fargate + ALB | Managed containers, auto-scaling |
| Docker Hub / local image | ECR (`loan-risk-serving`) | Private, regional, IAM-gated |
| Prometheus metrics | CloudWatch (`LoanRisk` namespace) | No server, pay-per-metric |
| Grafana dashboards | CloudWatch Dashboard | Integrated with alarms |
| Local parquet prediction log | S3 (`monitoring/predictions/`) | Durable, queryable |
| Custom alert scripts | CloudWatch Alarms + SNS | Managed, no polling needed |
| Local `artifacts/` pkl files | S3 (`loan-risk-artifacts/`) | Shared across pipeline steps |

---

## 5. Tech Stack

### Infrastructure & Orchestration

**Terraform** — every AWS resource is declared as code in `infra/terraform/`. One
`terraform apply` creates everything from scratch; one `terraform destroy` removes it.
State is stored in S3 so the team can share it.

**SageMaker Pipelines** — replaces Prefect. A pipeline is a DAG of processing and
training steps running on managed compute. Defined in `sagemaker/pipeline.py` using
the Python SDK. No server to keep running between executions.

**EventBridge Scheduler** — serverless cron that triggers SageMaker Pipeline
executions on a schedule. Replaces Prefect schedules entirely.

**ECS Fargate** — runs the FastAPI serving container without managing EC2 instances.
Tasks are replaced automatically if they crash, and billed per second of actual use.

### Data & Storage

**S3 (3 buckets)** — `loan-risk-data` for raw data, DVC cache, and prediction logs;
`loan-risk-artifacts` for preprocessor pkl and best hyperparameters; `loan-risk-mlflow`
for MLflow artifacts and SageMaker model files.

**RDS PostgreSQL 16** — MLflow tracking database. Stores experiment runs, metrics, and
the model registry. Lives in a private subnet; only ECS and SageMaker can reach it.

**DVC (S3 remote)** — same role as `main` branch but data cached to S3 instead of
locally. `dvc repro` still works for local runs.

### Application (unchanged from main)

**LightGBM / XGBoost** — gradient-boosted tree models. Same training code as `main`.

**scikit-learn pipeline** — feature preprocessing saved as `preprocessor.pkl`.

**MLflow** — experiment tracking and model registry. Same tracking calls as `main`;
backend is now RDS instead of SQLite.

**FastAPI** — prediction API. Same routes as `main`; metrics now go to CloudWatch.

### Monitoring

**CloudWatch** — receives custom metrics (`PredictionCount`, `PredictionLatency`,
`DefaultProbability`, `PSI`, `LiveAUC`). Alarms fire automatically when thresholds
are crossed. No Prometheus server needed.

**SNS** — delivers alarm notifications by email. Subscribe at:
AWS Console → SNS → Topics → `loan-risk-alerts` → Create subscription.

**Evidently** — still generates Evidently drift reports; HTML files are uploaded to S3.

### CI/CD

**GitHub Actions + OIDC** — no long-lived AWS credentials in GitHub. CI authenticates
by assuming an IAM role via the GitHub OIDC identity provider (created by bootstrap).

---

## 6. Project Structure — Every File Explained

```
loan-risk-mlops/  (aws branch)
│
├── src/loan_risk/                ← Python package (mostly unchanged from main)
│   ├── config.py                 ← Settings singleton; added AWSConfig section
│   ├── data/                     ← Pandera schemas, train/val/test splits
│   ├── features/                 ← Transformers, feature definitions, pipeline
│   ├── training/                 ← ModelTrainer — fits and logs to MLflow
│   ├── tuning/                   ← Optuna objectives (used locally; SageMaker AMT used in pipeline)
│   ├── evaluation/               ← Metrics, SHAP, bias audit, EvaluationReport
│   ├── registry/
│   │   └── client.py             ← MLflowRegistryClient + new SageMakerRegistryClient
│   ├── serving/
│   │   ├── app.py                ← FastAPI factory (unchanged)
│   │   └── predictor.py          ← ModelPredictor; Prometheus → CloudWatch metrics
│   └── monitoring/
│       ├── drift.py              ← Evidently drift + PSI; uploads reports to S3
│       ├── performance.py        ← Logs predictions to S3; emits AUC to CloudWatch
│       └── alerts.py             ← Publishes critical alerts to SNS
│
├── sagemaker/                    ← NEW: replaces pipelines/ entirely
│   ├── pipeline.py               ← Full 9-step SageMaker Pipeline definition
│   ├── run_pipeline.py           ← CLI to trigger/monitor pipeline executions
│   └── scripts/                  ← Entry-point scripts run inside SageMaker steps
│       ├── download.py           ← Step 1: download dataset to S3
│       ├── preprocess.py         ← Step 2: validate and clean → parquet
│       ├── featurize.py          ← Step 3: feature engineering → train/val/test splits
│       ├── train.py              ← Step 4: train model, log to MLflow
│       ├── evaluate.py           ← Step 5: compute metrics, write metrics.json
│       └── promote.py            ← Step 6: approve model package if AUC gate passes
│
├── infra/                        ← NEW: all AWS infrastructure as Terraform code
│   ├── bootstrap.sh              ← Run ONCE before terraform init (S3 state bucket,
│   │                                DynamoDB lock table, GitHub OIDC + IAM role)
│   └── terraform/
│       ├── main.tf               ← AWS provider, S3 backend, all module calls
│       ├── variables.tf          ← aws_region, project_name, db_password, etc.
│       ├── outputs.tf            ← ALB DNS, ECR URI, SageMaker role ARN, etc.
│       └── modules/
│           ├── vpc/              ← VPC, 2 public + 2 private subnets, NAT gateways, SGs
│           ├── s3/               ← 3 buckets with encryption and versioning
│           ├── ecr/              ← ECR repo for the serving Docker image
│           ├── rds/              ← RDS PostgreSQL 16 in private subnets
│           ├── iam/              ← 4 roles: ECS task, ECS execution, SageMaker, EventBridge
│           ├── secrets/          ← Secrets Manager: DB password, MLflow URI
│           ├── ecs/              ← ECS cluster, Fargate task definition, service
│           ├── alb/              ← ALB, target group, HTTP/HTTPS listeners
│           ├── sagemaker/        ← SageMaker Model Package Group (pipeline = Python SDK)
│           ├── cloudwatch/       ← Log groups, dashboard, 3 alarms, SNS topic
│           └── eventbridge/      ← 3 schedules: nightly retrain, weekly HPO, daily monitor
│
├── ecs/
│   └── task-definition.json      ← Fargate task definition template
│
├── config/
│   ├── settings.yaml             ← App config; added aws: section (buckets, namespace)
│   └── cloudwatch_dashboard.json ← Dashboard widget JSON (used by Terraform)
│
├── .github/workflows/
│   ├── ci.yml                    ← Lint + tests on every push; ECR push on aws/main
│   ├── serve.yml                 ← Integration tests; ECS deploy on serving changes
│   ├── training.yml              ← Triggers SageMaker Pipeline (replaces dvc repro)
│   └── deploy-infra.yml          ← Manual: terraform plan or apply
│
├── scripts/                      ← Local pipeline scripts (still work without AWS)
├── tests/                        ← Unit + integration tests (unchanged from main)
├── Dockerfile                    ← Multi-stage build; copies uv.lock for reproducible installs
├── docker-compose.yml            ← LOCAL DEV ONLY: MLflow + optional Grafana
├── pyproject.toml                ← boto3 + sagemaker SDK added; prefect + prometheus removed
├── .dvc/config                   ← DVC remote → s3://loan-risk-data-{account_id}/dvc-cache
├── process.md                    ← 📋 FULL DEPLOYMENT GUIDE — start here
└── README.md                     ← This file
```

---

## 7. How to Deploy to AWS

> Complete step-by-step instructions with exact commands, expected outputs, and
> troubleshooting tips are in **[process.md](process.md)**.

The high-level sequence:

```
Step 1  Create AWS account + billing alert
Step 2  Install tools (AWS CLI, Terraform, Docker)
Step 3  Create IAM admin user + access keys
Step 4  aws configure  (set region: ap-south-1)
Step 5  ./infra/bootstrap.sh  (one-time: state bucket, OIDC, GitHub IAM role)
Step 6  Set GitHub secrets + variables
Step 7  terraform apply  →  all AWS resources created  (~15 min)
Step 8  git push origin aws  →  CI builds + pushes Docker image to ECR
Step 9  aws ecs update-service --force-new-deployment  →  API goes live
```

Steps 1–6 are one-time setup. On subsequent runs, start from Step 7.

**See [process.md](process.md) for every command, variable name, and expected output.**

---

## 8. How to Run the Training Pipeline

The pipeline runs on SageMaker with 9 steps:

```
Download → Preprocess → Featurize → [HPO] → Train → Evaluate → AUC Gate → Register / Fail
```

### Option A — GitHub Actions (recommended)

GitHub → **Actions** → **"Training Pipeline (SageMaker)"** → **Run workflow**

> The "Run workflow" button only appears when the workflow file is on the default branch.
> If you don't see it, trigger from the CLI:
> ```bash
> gh workflow run training.yml --ref aws
> ```

### Option B — CLI

```bash
export AWS_DEFAULT_REGION=ap-south-1
export SAGEMAKER_ROLE_ARN=$(cd infra/terraform && terraform output -raw sagemaker_execution_role_arn)
export AWS_DATA_BUCKET=loan-risk-data-$(aws sts get-caller-identity --query Account --output text)
export AWS_ARTIFACTS_BUCKET=loan-risk-artifacts-$(aws sts get-caller-identity --query Account --output text)
export PYTHONPATH=src

# Upsert pipeline definition and start execution (skip HPO for first run)
uv run python sagemaker/pipeline.py --upsert --execute --skip-tuning
```

### Monitoring a running execution

```bash
# Check status by execution ARN
uv run python sagemaker/run_pipeline.py status <execution-arn>
```

Or: AWS Console → **SageMaker** → **Pipelines** → `loan-risk-training-pipeline`
→ click the execution → see each step's status and logs.

### Pipeline parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `lgbm` | `lgbm` or `xgboost` |
| `skip_tuning` | `true` | Skip HPO (uses last saved best params) |
| `n_trials` | `0` | HPO trials when tuning is enabled |

The weekly cron (Sundays 03:00 UTC) automatically sets `skip_tuning=false, n_trials=50`.

---

## 9. API Reference

Once deployed, get the endpoint:

```bash
ALB_DNS=$(cd infra/terraform && terraform output -raw alb_dns_name)
echo "API: http://$ALB_DNS"
```

### POST /predict

**Request body:**
```json
{
  "loan_id": "test-001",
  "annual_income": 75000,
  "loan_amount": 15000,
  "loan_term": 36,
  "loan_purpose": "debt_consolidation",
  "credit_score": 720,
  "employment_length": 5,
  "home_ownership": "RENT",
  "debt_to_income": 0.25
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `loan_id` | string | Unique identifier for the request |
| `annual_income` | float | Applicant annual income in USD |
| `loan_amount` | float | Requested loan amount in USD |
| `loan_term` | int | Loan term in months (12, 24, 36, 48, 60) |
| `loan_purpose` | string | One of: `debt_consolidation`, `credit_card`, `home_improvement`, `other` |
| `credit_score` | int | FICO score (300–850) |
| `employment_length` | int | Years employed (0–10+) |
| `home_ownership` | string | `RENT`, `OWN`, or `MORTGAGE` |
| `debt_to_income` | float | Monthly debt payments / monthly gross income |

**Response:**
```json
{
  "prediction": "APPROVE",
  "default_probability": 0.12,
  "confidence": "HIGH",
  "risk_tier": "LOW",
  "model_version": "1",
  "request_id": "a3f2...",
  "latency_ms": 45.2
}
```

**Risk tiers:**

| Tier | Probability | Meaning |
|------|-------------|---------|
| LOW | < 0.20 | Approve — strong borrower |
| MEDIUM | 0.20 – 0.40 | Approve with conditions |
| HIGH | 0.40 – 0.60 | Manual review recommended |
| VERY_HIGH | > 0.60 | Reject |

### GET /health

```bash
curl http://$ALB_DNS/health
# {"status": "healthy", "model_loaded": true, "model_version": "1"}
```

### GET /metrics

Returns a plain-text snapshot of prediction counts and latency percentiles.
Full metrics are in CloudWatch under the `LoanRisk` namespace.

### GET /docs

FastAPI interactive docs at `http://$ALB_DNS/docs`.

---

## 10. Monitoring with CloudWatch

All monitoring moved from Prometheus + Grafana to CloudWatch.

### Dashboard

AWS Console → **CloudWatch** → **Dashboards** → **loan-risk-dashboard**

| Widget | Metric | Alert threshold |
|--------|--------|----------------|
| Prediction volume | `PredictionCount` | — |
| P99 latency | `PredictionLatency` | — |
| Default probability distribution | `DefaultProbability` | — |
| PSI per feature | `PSI` | > 0.15 → retrain alert |
| Live AUC | `LiveAUC` | < 0.75 → degradation alert |
| ECS CPU + memory | `AWS/ECS` | — |
| ALB 5xx rate | `AWS/ApplicationELB` | > 1% → alert |

### Receiving alerts by email

1. AWS Console → **SNS** → **Topics** → `loan-risk-alerts`
2. **Create subscription** → Protocol: **Email** → enter your address
3. Confirm via the email link you receive

### Checking metrics from the CLI

```bash
# Prediction volume in the last hour
aws cloudwatch get-metric-statistics \
  --namespace LoanRisk \
  --metric-name PredictionCount \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 3600 \
  --statistics Sum \
  --region ap-south-1
```

---

## 11. Stop, Resume, and Cost Management

**Approximate cost when running 24/7: ~$93/month**
(NAT Gateways $32 + ECS Fargate $30 + ALB $16 + RDS $15 + misc $5)

> For exact stop/resume commands, what data is preserved, and which steps to
> re-run tomorrow — see the **"Stop Now / Resume Next Time"** section in
> **[process.md](process.md)**.

**Quick reference:**

```bash
# Stop everything now (takes ~8–10 min, then you pay nothing)
cd infra/terraform && terraform destroy -var="db_password=YourPassword" -parallelism=20

# Resume tomorrow — re-run Steps 7, 8, 9 from process.md
terraform apply -var="db_password=YourPassword"
git push origin aws
aws ecs update-service --cluster loan-risk-cluster --service loan-risk-serving \
  --force-new-deployment --region ap-south-1
```

After `terraform destroy`, S3 buckets with data are preserved (they cost pennies).
Everything else is gone and you are billed nothing until the next `terraform apply`.

---

## 12. Complete Removal — Cut All Ties with AWS

Use this when you want to permanently delete everything and pay nothing ever again.
**This is irreversible. All data, models, and logs will be gone.**

### Step 1 — Destroy Terraform-managed resources

```bash
cd infra/terraform
terraform destroy -var="db_password=YourPassword" -parallelism=20
# Type 'yes' when prompted — takes ~8–10 minutes
```

### Step 2 — Empty and delete the S3 buckets

`terraform destroy` skips non-empty S3 buckets. Delete them manually:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

for BUCKET in \
  loan-risk-data-$ACCOUNT_ID \
  loan-risk-artifacts-$ACCOUNT_ID \
  loan-risk-mlflow-$ACCOUNT_ID \
  loan-risk-tf-state-$ACCOUNT_ID; do

  echo "Emptying $BUCKET ..."

  # Remove all object versions (required for versioned buckets)
  VERSIONS=$(aws s3api list-object-versions --bucket $BUCKET \
    --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
    --output json 2>/dev/null)
  [ "$VERSIONS" != "null" ] && [ "$VERSIONS" != "" ] && \
    aws s3api delete-objects --bucket $BUCKET --delete "$VERSIONS" \
      --region ap-south-1 2>/dev/null || true

  # Remove delete markers
  MARKERS=$(aws s3api list-object-versions --bucket $BUCKET \
    --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' \
    --output json 2>/dev/null)
  [ "$MARKERS" != "null" ] && [ "$MARKERS" != "" ] && \
    aws s3api delete-objects --bucket $BUCKET --delete "$MARKERS" \
      --region ap-south-1 2>/dev/null || true

  aws s3 rb s3://$BUCKET --force --region ap-south-1 2>/dev/null || true
  echo "  done: $BUCKET"
done
```

### Step 3 — Delete the DynamoDB lock table

```bash
aws dynamodb delete-table \
  --table-name loan-risk-tf-locks \
  --region ap-south-1
```

### Step 4 — Remove the GitHub OIDC provider and IAM role

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Detach policy and delete the GitHub Actions role
aws iam detach-role-policy \
  --role-name loan-risk-github-actions-role \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess 2>/dev/null || true
aws iam delete-role --role-name loan-risk-github-actions-role 2>/dev/null || true

# Delete the OIDC provider
aws iam delete-open-id-connect-provider \
  --open-id-connect-provider-arn \
  arn:aws:iam::$ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com \
  2>/dev/null || true

echo "OIDC provider and GitHub Actions role removed"
```

### Step 5 — Delete ECR repository (if still present)

```bash
aws ecr delete-repository \
  --repository-name loan-risk-serving \
  --force \
  --region ap-south-1 2>/dev/null || echo "Already deleted"
```

### Step 6 — Revoke IAM access keys

1. AWS Console → **IAM** → **Users** → `loan-risk-admin`
2. **Security credentials** tab → **Access keys** → **Deactivate** → **Delete**
3. Delete the user if you no longer need it

### Step 7 — Verify nothing is running

```bash
# All four should return empty results
aws ec2 describe-nat-gateways \
  --filter Name=state,Values=available \
  --region ap-south-1 \
  --query 'NatGateways[*].NatGatewayId'

aws rds describe-db-instances \
  --region ap-south-1 \
  --query 'DBInstances[*].DBInstanceIdentifier'

aws ecs list-clusters --region ap-south-1

aws s3 ls | grep loan-risk
```

If all four return empty — you are done. Zero AWS charges from this point forward.

> Also cancel your billing alert: AWS Console → **Billing** → **Budgets** →
> delete the `loan-risk` budget to stop receiving notification emails.

---

## 13. Troubleshooting

### `terraform init` — "bucket does not exist"
Run `./infra/bootstrap.sh` before `terraform init`. The bootstrap creates the S3 state bucket.

### `terraform apply` — "ResourceAlreadyExistsException" on a log group
A previous partial apply created it. Import it into state:
```bash
terraform import module.ecs.aws_cloudwatch_log_group.ecs /ecs/loan-risk-serving
```

### `terraform apply` — "MasterUserPassword is not valid"
Password contains an illegal character (`/`, `@`, `"`, or space).
Use only letters and digits, e.g. `MyDbPass123`.

### `terraform apply` — "secret with this name is already scheduled for deletion"
This happens if you ran `terraform destroy` with an older version of the config that used
a 7-day Secrets Manager recovery window. Force-delete the pending secrets, then re-apply:
```bash
aws secretsmanager delete-secret --secret-id loan-risk/rds-password --force-delete-without-recovery
aws secretsmanager delete-secret --secret-id loan-risk/mlflow-config --force-delete-without-recovery
terraform apply -var="db_password=YourPassword"
```
The current config uses `recovery_window_in_days = 0`, so this won't recur on future destroy/apply cycles.

### ECS tasks stuck in `CannotPullContainerError`
The Docker image hasn't been pushed to ECR yet. Wait for the CI
`Build & Push to ECR` job to finish, then:
```bash
aws ecs update-service --cluster loan-risk-cluster --service loan-risk-serving \
  --force-new-deployment --region ap-south-1
```

### ECS service never stabilises (`aws ecs wait` times out)
```bash
# See what's happening
aws ecs describe-services \
  --cluster loan-risk-cluster --services loan-risk-serving \
  --region ap-south-1 --query 'services[0].events[:5]'
```
Common causes: wrong `MLFLOW_TRACKING_URI` in Secrets Manager, or the IAM task role
is missing S3 permissions.

### GitHub Actions — "Not authorized to perform sts:AssumeRoleWithWebIdentity"
- Check `AWS_ROLE_ARN` secret is set in GitHub → Settings → Secrets
- Verify the OIDC provider exists: IAM → Identity providers
- Verify the role trust policy references your exact `repo:owner/repo:*`

### Training Pipeline workflow not visible in GitHub Actions UI
The "Run workflow" button only appears for workflows on the default branch.
```bash
gh workflow run training.yml --ref aws
```

### SageMaker pipeline step fails
AWS Console → **SageMaker** → **Pipelines** → `loan-risk-training-pipeline`
→ click the execution → click the failed step → **View logs**

---

## 14. FAQ

**Q: Why does this branch exist separately instead of replacing main?**
A: The `main` branch is a self-contained local stack — no AWS account needed. This
`aws` branch is the production deployment. Having both lets you learn the local
patterns first, then see exactly how each piece maps to a managed service.

**Q: Do I need to run the training pipeline before the API works?**
A: Yes. The API loads the champion model from MLflow, which is populated by the
training pipeline. Run at least one pipeline execution after deploying.

**Q: Why two model registries — MLflow and SageMaker?**
A: MLflow's `@champion` alias is what the FastAPI predictor uses to load the model
(`mlflow.pyfunc.load_model`). The SageMaker Model Package Group provides an approval
workflow and audit trail. Both are updated by `sagemaker/scripts/promote.py`.

**Q: What is the AUC gate?**
A: After training, `evaluate.py` computes test-set AUC. If it falls below 0.80,
the pipeline routes to a `FailStep` and the model is not registered. This prevents a
degraded model from being promoted automatically.

**Q: Can I run the pipeline locally without AWS?**
A: Yes. Set `MLFLOW_TRACKING_URI=sqlite:///mlruns.db` and run:
```bash
uv run python scripts/run_pipeline.py --stage all --skip-tuning
```
This uses SQLite instead of RDS and saves artifacts to `artifacts/` instead of S3.

**Q: How do I add a new feature?**
A: Same process as `main` — edit `src/loan_risk/features/definitions.py` and
`features/transformers.py`. The SageMaker featurize step calls the same
`build_feature_pipeline()` function, so changes propagate automatically.

**Q: What does the nightly retrain actually do?**
A: EventBridge triggers the SageMaker Pipeline with `skip_tuning=true` every night.
It re-downloads data, re-featurizes, retrains with the last known best hyperparameters,
evaluates, and promotes if AUC >= 0.80. The whole run takes ~30 minutes.

---

## License

MIT — see `LICENSE`. Dataset: UCI Default of Credit Card Clients (CC BY 4.0).
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques.
*Expert Systems with Applications*, 36(2), 2473–2480.
