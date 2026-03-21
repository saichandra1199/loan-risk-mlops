# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v

# Run only unit tests (fast, no external services)
uv run pytest tests/unit/ -v

# Run integration tests (API tests with mocked predictor)
uv run pytest tests/integration/ -v

# Run a single test file
uv run pytest tests/unit/test_transformers.py -v

# Run a single test by name
uv run pytest tests/unit/test_transformers.py::TestCreditScoreBinner::test_poor_credit_band -v

# Lint and format
uv run ruff check src/ scripts/ sagemaker/
uv run ruff check --fix src/ scripts/ sagemaker/
uv run ruff format src/

# Run local pipeline stages (no AWS needed)
uv run python scripts/run_pipeline.py --stage features
uv run python scripts/run_pipeline.py --stage train --skip-tuning
uv run python scripts/run_pipeline.py --stage evaluate
uv run python scripts/run_pipeline.py --stage all --skip-tuning

# Start the prediction API
uv run uvicorn loan_risk.serving.app:create_app --factory --port 8000 --reload

# MLflow UI (no Docker)
MLFLOW_TRACKING_URI=sqlite:///mlruns.db mlflow ui --port 5000

# Local dev stack (MLflow only — Prefect and Prometheus removed from this branch)
docker-compose up -d

# SageMaker pipeline — upsert definition + start execution
uv run python sagemaker/pipeline.py --upsert --execute --skip-tuning

# SageMaker pipeline — dry-run (prints pipeline JSON, no AWS call)
uv run python sagemaker/pipeline.py --dry-run

# Check a running SageMaker execution
uv run python sagemaker/run_pipeline.py status <execution-arn>

# Terraform (run from infra/terraform/)
terraform init
terraform plan  -var="db_password=<pass>"
terraform apply -var="db_password=<pass>"
terraform destroy -var="db_password=<pass>"
```

## Environment

- All Python commands must be prefixed with `uv run` — the package is installed in `.venv/` managed by uv.
- Set `MLFLOW_TRACKING_URI=sqlite:///mlruns.db` to run locally without RDS. CI always uses this.
- `PYTHONPATH=src` is required when running scripts directly without `uv run`. With `uv run` it is handled automatically.
- Config loads in this order: `config/settings.yaml` → `.env` → environment variables. Use `__` as the nested delimiter (e.g. `MLFLOW__TRACKING_URI=...`, `AWS__DATA_BUCKET=...`).
- `get_settings()` in `src/loan_risk/config.py` is an `@lru_cache` singleton. If you mutate settings in tests, call `get_settings.cache_clear()` after.
- SageMaker pipeline scripts require: `AWS_DEFAULT_REGION`, `SAGEMAKER_ROLE_ARN`, `AWS_DATA_BUCKET`, `AWS_ARTIFACTS_BUCKET`.

## Architecture

### Two execution paths

The same Python application code supports two runtime environments:

**Local path** — `scripts/run_pipeline.py` orchestrates directly, MLflow uses SQLite, artifacts go to `artifacts/`, data goes to `data/`. No AWS account needed.

**AWS path** — `sagemaker/pipeline.py` defines a SageMaker Pipeline DAG that runs each stage on managed compute. MLflow uses RDS PostgreSQL as backend and S3 as artifact store. The FastAPI serving container runs on ECS Fargate behind an ALB.

### Data flow (AWS path)

```
UCI OpenML
  → sagemaker/scripts/download.py    → s3://loan-risk-data/raw/
  → sagemaker/scripts/preprocess.py  → s3://loan-risk-data/processed/
  → sagemaker/scripts/featurize.py   → s3://loan-risk-data/processed/{train,val,test}.parquet
                                        s3://loan-risk-artifacts/preprocessor.pkl
  → sagemaker/scripts/train.py       → MLflow run (RDS backend + s3://loan-risk-mlflow/)
  → sagemaker/scripts/evaluate.py    → metrics.json + evaluation_report.json
  → AUC >= 0.80 gate
  → sagemaker/scripts/promote.py     → SageMaker Model Package (Approved) + MLflow @champion alias
  → ECS Fargate (loan_risk.serving.app) → POST /predict (loads @champion via MLflow URI → S3)
```

### Package layout (`src/loan_risk/`)

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic-settings `Settings` singleton. Includes `AWSConfig` (buckets, CloudWatch namespace, SNS ARN). Single source of truth for all config. |
| `data/schemas.py` | Pandera `RawLoanSchema` and `InferenceInputSchema`. Contract between preprocessing and the rest of the pipeline. |
| `data/splits.py` | Stratified 70/10/20 train/val/test split preserving the ~22% default rate. |
| `features/definitions.py` | Constants: `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, `ENGINEERED_FEATURES`, `CREDIT_SCORE_BINS`. **Start here when adding/removing features.** |
| `features/transformers.py` | Custom sklearn `BaseEstimator`/`TransformerMixin` classes. All are stateless or fit on training data only. |
| `features/pipeline.py` | `build_feature_pipeline()` assembles transformers into a single sklearn `Pipeline`. `prepare_features()` calls fit or transform. Saved to `preprocessor.pkl`. |
| `training/trainer.py` | `ModelTrainer.fit()` — fits feature pipeline, trains LightGBM/XGBoost, calibrates threshold (F-beta β=2), logs to MLflow. Returns `TrainingResult`. |
| `tuning/` | Optuna objectives for LightGBM and XGBoost. `run_hyperparameter_search()` saves best params to `artifacts/best_params/`. Used locally; AWS path uses SageMaker AMT. |
| `evaluation/` | `compute_classification_metrics()` (AUC, Gini, KS), SHAP `TreeExplainer`, bias audit, `EvaluationReport` dataclass. |
| `registry/client.py` | `MLflowRegistryClient` (promotes/archives via `@champion` alias) + `SageMakerRegistryClient` (creates model packages, lists approved versions). Both are updated on promotion. |
| `serving/predictor.py` | `ModelPredictor` singleton (`get_predictor()`). Loaded once at startup. Emits `PredictionCount`, `PredictionLatency`, `DefaultProbability` to CloudWatch. **Never call `predictor.load()` inside request handlers.** |
| `serving/app.py` | `create_app()` factory — registers routes, middleware, lifespan (model warm-up). Also exposes module-level `app = create_app()` for direct uvicorn invocation. |
| `monitoring/drift.py` | Evidently drift reports + PSI per feature. PSI > 0.15 triggers alert. Uploads HTML reports to S3; emits PSI metrics to CloudWatch. |
| `monitoring/performance.py` | Logs predictions to S3 (`monitoring/predictions/{date}/`). `compute_live_auc()` emits `LiveAUC` to CloudWatch. |
| `monitoring/alerts.py` | `run_monitoring_checks()` publishes structured JSON to SNS topic for critical alerts. CloudWatch Alarms handle threshold-based alerting independently. |

### Key design decisions

**Polars vs pandas**: Pipeline ingests and stores data in Polars (`pl.DataFrame`), but sklearn transformers work on pandas. `prepare_features()` in `features/pipeline.py` converts with `.to_pandas()` before calling the sklearn pipeline.

**MLflow without a server**: Set `MLFLOW_TRACKING_URI=sqlite:///mlruns.db`. SQLite supports both tracking and the model registry. This is what CI and local dev use. In production, the URI points to RDS via a Secrets Manager entry injected into ECS.

**FastAPI dependency injection and mocking**: Routes use `Depends(get_predictor)`. To mock in tests, use `app.dependency_overrides[get_predictor] = lambda: mock`. Do **not** use `unittest.mock.patch` on the route — `Depends` captures the function reference at decoration time. The app's lifespan calls `get_predictor()` directly (bypasses DI), so also patch `loan_risk.serving.app.get_predictor` to prevent real model loading in tests.

**Dual model registry**: MLflow `@champion` alias is the serving contract — the FastAPI predictor resolves the model URI through it. The SageMaker Model Package Group provides the approval workflow. `sagemaker/scripts/promote.py` writes to both on a successful pipeline execution.

**Structlog**: Uses `structlog.stdlib.LoggerFactory()`. All logs are JSON in production (`json_output=True`). In tests, logs appear in captured stdout.

**CloudWatch metrics**: `predictor.py` and `monitoring/` emit custom metrics to the `LoanRisk` namespace via `boto3.client("cloudwatch").put_metric_data`. CloudWatch Alarms (defined in Terraform) fire to SNS independently of the application code. The `GET /metrics` endpoint returns in-process counters only.

### Infrastructure (`infra/terraform/`)

All AWS resources are declared as Terraform modules. Run `terraform apply` once to create everything. Key modules:
- `modules/vpc/` — VPC, public/private subnets, NAT gateways, security groups
- `modules/ecs/` — Fargate cluster, task definition (1 vCPU / 2 GB), service (desired 2), CloudWatch log group
- `modules/cloudwatch/` — Dashboard, 3 alarms (PSI > 0.15, AUC < 0.75, 5xx > 1%), SNS topic
- `modules/eventbridge/` — nightly retrain (Mon–Sat 02:00 UTC) + weekly HPO (Sun 03:00 UTC) schedules
- `modules/sagemaker/` — Model Package Group only; the Pipeline DAG is managed by `sagemaker/pipeline.py --upsert`

Before `terraform init`, run `./infra/bootstrap.sh` once to create the S3 state bucket, DynamoDB lock table, and GitHub OIDC provider.

### Testing conventions

- **Unit tests** (`tests/unit/`) use the `sample_df` fixture (200-row Polars DataFrame in `tests/conftest.py`). No external services.
- **Integration tests** (`tests/integration/test_api.py`) use a `mock_predictor` fixture and `app.dependency_overrides`. No trained model required.
- pytest is configured with `asyncio_mode = "auto"` and coverage on `src/loan_risk/`.
- `N803`/`N806` ruff rules are suppressed globally — `X`, `X_train`, `X_val` are accepted ML conventions.

### GitHub Actions

- `ci.yml` — every push: ruff (`src/ scripts/ sagemaker/`) + 30 unit tests. On `aws`/`main` branch push: also builds and pushes Docker image to ECR using OIDC.
- `serve.yml` — when `src/loan_risk/serving/` or integration tests change: 7 API tests + ECS rolling deploy.
- `training.yml` — manual dispatch + weekly cron (Sundays 02:00 UTC): triggers SageMaker Pipeline via `sagemaker/pipeline.py --upsert --execute`. Replaces the old DVC runner.
- `deploy-infra.yml` — manual dispatch: `terraform plan` or `terraform apply`.

All workflows use `MLFLOW_TRACKING_URI: "sqlite:///mlruns.db"` and `PYTHONPATH: "src"`. AWS workflows authenticate via OIDC (`AWS_ROLE_ARN` secret, no stored credentials).
