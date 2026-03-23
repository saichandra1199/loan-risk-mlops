# 🏦 Loan Risk MLOps Pipeline

[![CI — Lint & Tests](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/ci.yml)
[![Training Pipeline](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/training.yml/badge.svg)](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/training.yml)
[![Test Serving Layer](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/serve.yml/badge.svg)](https://github.com/saichandra1199/loan-risk-mlops/actions/workflows/serve.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![DVC](https://img.shields.io/badge/DVC-pipeline-purple)](https://dvc.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A **production-grade Machine Learning system** that predicts whether a credit card
> customer will default on their next payment.
> Built to teach real-world MLOps practices: data versioning, automated pipelines,
> model tracking, and CI/CD — using free, open-source tools only.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [Why do we need MLOps?](#2-why-do-we-need-mlops)
3. [The Dataset](#3-the-dataset)
4. [System Architecture](#4-system-architecture)
5. [Tech Stack Explained](#5-tech-stack-explained)
6. [Project Structure — Every File Explained](#6-project-structure--every-file-explained)
7. [Quick Start (from zero to running)](#7-quick-start-from-zero-to-running)
8. [Understanding DVC (Data Version Control)](#8-understanding-dvc-data-version-control)
9. [Understanding GitHub Actions (CI/CD)](#9-understanding-github-actions-cicd)
10. [Current Pipeline Status — All Green](#10-current-pipeline-status--all-green)
11. [Pipeline Stages — What Happens at Each Step](#11-pipeline-stages--what-happens-at-each-step)
12. [API Reference](#12-api-reference)
13. [Configuration Guide](#13-configuration-guide)
14. [Running Experiments](#14-running-experiments)
15. [How to Implement New Features](#15-how-to-implement-new-features)
16. [Monitoring the Pipeline](#16-monitoring-the-pipeline)
17. [Debugging Guide](#17-debugging-guide)
18. [Troubleshooting](#18-troubleshooting)
19. [FAQ for Beginners](#19-faq-for-beginners)
20. [Observability Tools — MLflow, Prefect, Prometheus, Grafana](#20-observability-tools--mlflow-prefect-prometheus-grafana)

---

## 1. What is this project?

This project builds a **credit default prediction system** — a machine learning model that
looks at a customer's financial history and predicts: **"Will this person fail to pay their
credit card bill next month?"**

The prediction output is binary:
- **REJECT (1)** — The model predicts the customer will default (not pay)
- **APPROVE (0)** — The model predicts the customer will pay on time

But this is not just a model — it's a complete **production system** that includes:

| What it does | Why it matters |
|---|---|
| Downloads real financial data | Uses actual UCI research data, not toy examples |
| Validates data quality | Catches bad data before it corrupts your model |
| Engineers features | Transforms raw numbers into signals the model can learn from |
| Trains & evaluates the model | LightGBM with AUC, KS statistic, Gini coefficient |
| Tracks all experiments | MLflow records every run so you can compare results |
| Serves predictions via API | FastAPI endpoint with real-time SHAP explanations |
| Monitors for data drift | Alerts when real-world data shifts away from training data |
| Orchestrates everything | Prefect runs the full pipeline on a schedule |
| Versions data & pipeline | DVC ensures anyone can reproduce your exact results |
| Automates CI/CD | GitHub Actions tests and trains on every push |

### Who is this for?

- **Students** learning MLOps by seeing a complete, real project
- **Data scientists** wanting to learn how to productionize models
- **Engineers** understanding the full ML lifecycle
- **Teams** needing a reference architecture for financial ML

---

## 2. Why do we need MLOps?

### The problem without MLOps

Imagine you build a loan default model in a Jupyter notebook:

```
Day 1: You train a model. It gets 85% accuracy.
Day 30: Someone asks "what data did you use?" You forgot.
Day 60: The model accuracy drops. You don't know why.
Day 90: A colleague tries to retrain. It fails. Different library versions.
Day 120: You need to deploy. You don't know how.
```

This is the reality for most ML projects. They work on a laptop but fail in production.

### The solution: MLOps

MLOps (Machine Learning Operations) solves this by treating ML like software engineering:

```
Data → Version Control → Automated Pipeline → Model Registry → Serving → Monitoring
```

With this project:
- **Every run is tracked**: MLflow logs every model's parameters, metrics, and artifacts
- **Data is versioned**: DVC tracks exactly which data was used for each model
- **Pipeline is reproducible**: `dvc repro` runs the exact same pipeline on any machine
- **Tests run automatically**: GitHub Actions runs tests on every code change
- **Deployment is automated**: A new model is only deployed if it passes the quality gate (AUC ≥ 0.75)

---

## 3. The Dataset

### Source: UCI Default of Credit Card Clients

- **Where**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Download**: Automatic via `python scripts/download_dataset.py` (no account needed)
- **Size**: 30,000 customers × 24 features
- **Target**: Will the customer default on next month's payment? (1=Yes, 0=No)
- **Default rate**: ~22% of customers defaulted

This is a real dataset used in academic research, collected from a Taiwanese bank.

### Raw UCI Columns

| Column | What it means |
|---|---|
| `LIMIT_BAL` | Credit limit given to the customer (New Taiwan Dollar) |
| `SEX` | Gender (1=male, 2=female) |
| `EDUCATION` | Education level (1=grad school, 2=university, 3=high school, 4=other) |
| `MARRIAGE` | Marital status (1=married, 2=single, 3=other) |
| `AGE` | Age in years |
| `PAY_0` | Repayment status in September 2005 (-1=paid on time, 1=1 month late, 2=2 months late, ...) |
| `PAY_2` to `PAY_6` | Same for August 2005 back to April 2005 |
| `BILL_AMT1` to `BILL_AMT6` | Bill statement amount for each month |
| `PAY_AMT1` to `PAY_AMT6` | Previous payment amount for each month |
| `default.payment.next.month` | **TARGET**: Did they default? (1=yes, 0=no) |

### How we transform the data

Our preprocessing script (`scripts/preprocess_dataset.py`) maps UCI columns to
a canonical loan schema that our pipeline understands:

| Our Feature | Source | How Derived |
|---|---|---|
| `loan_amount` | `LIMIT_BAL` | Credit limit = loan exposure; clipped to [100, 100k] |
| `annual_income` | `LIMIT_BAL × 3` | Credit limit is roughly 1/3 of annual income (proxy) |
| `employment_years` | `AGE - 22` | Assumes work started at age 22; clipped to [0, 50] |
| `credit_score` | `PAY_0..PAY_6` | 300 + 550 × (fraction of months paid on time) → FICO-like scale |
| `debt_to_income_ratio` | `BILL_AMT1 / monthly_income` | Bill utilisation rate |
| `num_open_accounts` | `count(PAY_AMT > 0)` | Months where a payment was actually made |
| `num_delinquencies` | `count(PAY_X > 0)` | Months with a payment delay |
| `loan_purpose` | `EDUCATION` | 1→home_improvement, 2→debt_consolidation, 3→major_purchase, etc. |
| `home_ownership` | `MARRIAGE` | 1→MORTGAGE (married), 2→RENT (single), 3→OWN |
| `loan_term_months` | `LIMIT_BAL` | 36 months if limit ≤ 200k, else 60 months |
| `loan_default` | Target column | Direct copy |

> **Note for beginners**: These mappings are approximations, not exact relationships.
> Real financial data pipelines often use proxies like this when exact features are unavailable.
> The key is to document the assumptions clearly (which we do here).

---

## 4. System Architecture

### Data Flow

```
Internet
  │
  ▼ (1) Download
┌─────────────────────────────────┐
│  UCI Credit Card Default Dataset │  30,000 rows × 24 UCI columns
│  data/raw/credit_default_raw.csv │
└─────────────────┬───────────────┘
                  │
                  ▼ (2) Preprocess
┌─────────────────────────────────┐
│  Canonical Loan Schema          │  30,000 rows × 12 loan features
│  data/raw/loans.parquet         │  Validated by Pandera
└─────────────────┬───────────────┘
                  │
                  ▼ (3) Feature Engineering
┌─────────────────────────────────┐  ┌────────────────────────────┐
│  Train / Val / Test Splits      │  │ Fitted sklearn Pipeline     │
│  data/processed/                │  │ artifacts/preprocessor.pkl  │
│  (stratified by default rate)   │  │ (StandardScaler + OHE)      │
└─────────────────┬───────────────┘  └────────────────────────────┘
                  │
                  ▼ (4) Model Training
┌─────────────────────────────────┐  ┌────────────────────────────┐
│  LightGBM Classifier            │  │ MLflow Tracking            │
│  scale_pos_weight for imbalance │  │ (logs params, metrics,     │
│  F-beta threshold calibration   │  │  model artifact, SHAP)     │
└─────────────────┬───────────────┘  └────────────────────────────┘
                  │
                  ▼ (5) Evaluation + Gate
┌─────────────────────────────────┐
│  AUC-ROC / Gini / KS statistic  │
│  SHAP feature importance        │  If AUC ≥ 0.75 → promote
│  Bias audit (by education/age)  │  If AUC < 0.75 → reject
└─────────────────┬───────────────┘
                  │
                  ▼ (6) Model Registry (if promoted)
┌─────────────────────────────────┐
│  MLflow Model Registry          │
│  alias: "champion"              │
│  Previous versions → archived   │
└─────────────────┬───────────────┘
                  │
                  ▼ (7) Serving
┌─────────────────────────────────┐
│  FastAPI Prediction Server      │
│  POST /predict → JSON response  │
│  GET  /health  → readiness      │
│  GET  /metrics → Prometheus     │
└─────────────────┬───────────────┘
                  │
                  ▼ (8) Monitoring
┌─────────────────────────────────┐
│  Evidently Data Drift Reports   │  Compares live data vs training
│  PSI on all numeric features    │  PSI > 0.15 → retrain alert
│  Live AUC when labels arrive    │  AUC < 0.70 → retrain alert
└─────────────────────────────────┘
```

### CI/CD Flow (GitHub Actions)

```
You push code to GitHub
        │
        ▼
┌─────────────────────┐
│  ci.yml triggers    │
│  ✓ ruff lint        │
│  ✓ unit tests       │
└─────────┬───────────┘
          │ (if on main/PR, also triggers)
          ▼
┌─────────────────────┐     Every Sunday 02:00 UTC
│  training.yml       │ ◄──────────────────────────
│  ✓ dvc repro        │
│  ✓ download data    │
│  ✓ preprocess       │
│  ✓ featurize        │
│  ✓ train model      │
│  ✓ metrics report   │
└─────────────────────┘
```

---

## 5. Tech Stack Explained

Here's why each tool was chosen — in plain English:

### Package Management: `uv`
**What it is**: A Python package manager (replaces pip/conda for this project)
**Why we use it**: 10-100x faster than pip. `uv sync` installs all 40+ dependencies in ~10 seconds. It also creates a lockfile (`uv.lock`) so everyone on the team installs the exact same versions.

### Data Processing: `Polars`
**What it is**: A DataFrame library (like pandas, but much faster)
**Why we use it**: For a 30,000-row dataset the difference isn't huge, but for 1M+ rows, Polars is 5-20x faster than pandas. It uses Apache Arrow under the hood and supports lazy evaluation (compute only when needed).

### Data Validation: `Pandera`
**What it is**: A library that validates DataFrames against a schema
**Why we use it**: Without Pandera, bad data silently corrupts your model. With Pandera, if `credit_score` is suddenly 1500 (impossible), the pipeline immediately fails with a clear error message instead of producing a garbage model.

### Experiment Tracking: `MLflow`
**What it is**: A tool that records every training run
**Why we use it**: Every time you train a model, MLflow logs:
- The hyperparameters you used
- The metrics (AUC, F1, etc.)
- The actual model file
- The feature importance plot
This lets you compare 100 experiments and say "run #47 was the best because..."

### Hyperparameter Tuning: `Optuna`
**What it is**: Automated hyperparameter search
**Why we use it**: Instead of manually trying learning_rate=0.01, then 0.05, then 0.1, Optuna intelligently searches the parameter space using Bayesian optimization (TPE sampler). It finds good parameters faster than random search.

### Orchestration: `Prefect`
**What it is**: A workflow scheduler
**Why we use it**: The pipeline needs to run every Sunday at 2am. Prefect handles: running tasks in order, retrying failed tasks, alerting on failures, and showing a visual dashboard of what ran when.

### Model Training: `LightGBM`
**What it is**: A gradient boosting decision tree library
**Why we use it**: For tabular data (spreadsheet-style), gradient boosting consistently outperforms neural networks. LightGBM is particularly fast (uses histograms instead of exact splits) and handles categorical features natively.

### Serving: `FastAPI`
**What it is**: A Python web framework for building APIs
**Why we use it**: FastAPI automatically generates API documentation, validates request data with Pydantic, and is async-native (handles concurrent requests efficiently). `/predict` returns a prediction in <50ms.

### Monitoring: `Evidently`
**What it is**: An ML monitoring library
**Why we use it**: When the real world changes (e.g., a recession raises default rates), the model's training data is no longer representative. Evidently detects this "data drift" by comparing new data against the training data distribution using PSI (Population Stability Index).

### Data Versioning: `DVC`
**What it is**: Git for data and ML pipelines (explained further in Section 8)
**Why we use it**: Git tracks code changes. DVC tracks data and pipeline changes. `dvc repro` replays the exact pipeline that produced a past model result.

---

## 6. Project Structure — Every File Explained

```
loan-risk-mlops/
│
├── README.md                   ← You are here
├── pyproject.toml              ← Project metadata + all 40+ dependencies
├── .python-version             ← Tells uv to use Python 3.11
├── .env.example                ← Template for secret configuration (copy to .env)
├── .gitignore                  ← Files git should NOT track (venv, secrets, data)
├── .dvcignore                  ← Files DVC should NOT cache (same idea)
├── .pre-commit-config.yaml     ← Auto-run linting before every git commit
├── Makefile                    ← Shortcut commands (make train, make test, etc.)
├── Dockerfile                  ← How to containerize the prediction server
├── docker-compose.yml          ← Starts MLflow + Prefect + Prometheus + Grafana
│
├── dvc.yaml                    ← THE PIPELINE: defines 4 stages (download→train)
├── params.yaml                 ← Tunable parameters (model type, train/test split %)
│
├── .github/
│   └── workflows/
│       ├── ci.yml              ← GitHub Actions: run tests on every push
│       ├── training.yml        ← GitHub Actions: run full pipeline weekly
│       └── serve.yml           ← GitHub Actions: test the prediction API
│
├── config/
│   ├── settings.yaml           ← Non-secret config (MLflow URL, model name, etc.)
│   ├── monitoring_thresholds.yaml  ← When to trigger retraining alerts
│   └── prometheus.yml          ← Metrics scraping config for Prometheus
│
├── scripts/                    ← Standalone scripts you can run directly
│   ├── download_dataset.py     ← Downloads UCI dataset (30K rows) from OpenML
│   ├── preprocess_dataset.py   ← Maps UCI columns → our loan feature schema
│   ├── run_pipeline.py         ← CLI to run any pipeline stage: --stage all
│   ├── generate_sample_data.py ← Creates synthetic data for quick tests
│   └── promote_model.py        ← Manually promote a model to Production
│
├── src/loan_risk/              ← The main Python package (all ML code lives here)
│   ├── __init__.py             ← Package marker, exports __version__
│   ├── config.py               ← Loads config from settings.yaml + env vars
│   ├── logging_setup.py        ← Structured JSON logging with structlog
│   ├── exceptions.py           ← Custom exception classes (DataValidationError, etc.)
│   │
│   ├── data/
│   │   ├── ingestion.py        ← load_raw_data() reads CSV or Parquet → pl.DataFrame
│   │   ├── schemas.py          ← Pandera schemas: what shape the data must have
│   │   └── splits.py           ← stratified_split() → train/val/test DataFrames
│   │
│   ├── features/
│   │   ├── definitions.py      ← Constants: which columns are features, targets, IDs
│   │   ├── transformers.py     ← Custom sklearn transforms (log, bin credit score, etc.)
│   │   └── pipeline.py         ← build_feature_pipeline() → sklearn Pipeline
│   │
│   ├── training/
│   │   ├── models.py           ← get_model("lgbm") → unfitted LGBMClassifier
│   │   └── trainer.py          ← ModelTrainer.fit() → trains + logs to MLflow
│   │
│   ├── tuning/
│   │   ├── objective.py        ← Optuna objective: try these params, return AUC
│   │   └── search.py           ← run_hyperparameter_search() → best params dict
│   │
│   ├── evaluation/
│   │   ├── metrics.py          ← compute_classification_metrics() → AUC, Gini, KS
│   │   ├── explainability.py   ← SHAP values for feature importance
│   │   ├── bias_audit.py       ← Fairness metrics across education/marital status
│   │   └── report.py           ← EvaluationReport dataclass → JSON file
│   │
│   ├── registry/
│   │   └── client.py           ← MLflowRegistryClient: promote/archive/fetch models
│   │
│   ├── serving/
│   │   ├── app.py              ← create_app() → FastAPI application factory
│   │   ├── routes.py           ← /predict /health /metrics /model-info endpoints
│   │   ├── schemas.py          ← Pydantic v2 request/response schemas
│   │   ├── predictor.py        ← ModelPredictor singleton (loads model once)
│   │   └── middleware.py       ← Logs every request with latency + request ID
│   │
│   └── monitoring/
│       ├── drift.py            ← Evidently drift reports + PSI computation
│       ├── performance.py      ← Tracks live AUC when labels arrive (30-90 days later)
│       └── alerts.py           ← run_monitoring_checks() → retrain signal
│
├── pipelines/                  ← Prefect orchestration
│   ├── tasks.py                ← @task wrappers for each pipeline stage
│   ├── flows.py                ← @flow: full_pipeline_flow(), daily_monitor_flow()
│   └── schedules.py            ← CronSchedule for nightly and weekly runs
│
├── data/                       ← All data (tracked by DVC, not git)
│   ├── raw/                    ← Original/downloaded data
│   │   ├── credit_default_raw.csv  ← UCI raw dataset (downloaded by pipeline)
│   │   └── loans.parquet           ← Preprocessed canonical format
│   ├── processed/              ← Train/val/test splits after feature engineering
│   ├── reference/              ← Training data saved for drift comparison
│   └── monitoring/             ← Live prediction logs (for delayed-label AUC)
│
├── artifacts/                  ← Model artifacts (tracked by DVC)
│   ├── preprocessor.pkl        ← Fitted sklearn Pipeline (StandardScaler + OHE)
│   ├── best_params/            ← Optuna best hyperparameters (JSON)
│   └── last_training_result.json  ← Metadata from most recent training run
│
├── reports/                    ← Generated reports
│   ├── evaluation/
│   │   └── metrics.json        ← AUC, Gini, KS, F1 (DVC-tracked metric)
│   ├── monitoring/
│   │   └── drift_report.html   ← Evidently HTML drift report
│   └── validation/
│       └── validation_report.json  ← Schema validation results
│
├── tests/
│   ├── conftest.py             ← Shared fixtures (200-row synthetic DataFrame)
│   ├── unit/                   ← Tests that run in <1 second, no external services
│   │   ├── test_schemas.py     ← Test Pandera validation rules
│   │   ├── test_transformers.py ← Test custom sklearn transformers
│   │   ├── test_splits.py      ← Test stratified splitting
│   │   └── test_metrics.py     ← Test AUC/Gini/KS computation
│   └── integration/            ← Tests that need more setup
│       ├── test_api.py         ← Test FastAPI endpoints (with mocked model)
│       └── test_pipeline_e2e.py ← End-to-end pipeline on 500-row synthetic data
│
└── notebooks/
    ├── 01_eda.ipynb            ← Exploratory data analysis
    └── 02_model_comparison.ipynb ← Compare LightGBM vs XGBoost runs
```

---

## 7. Quick Start (from zero to running)

### Prerequisites

You need:
- **Python 3.11+** ([download](https://www.python.org/downloads/))
- **Git** ([download](https://git-scm.com/downloads))
- **uv** (install below)

Optional (for the full stack):
- **Docker** (for MLflow UI, Prefect, Prometheus, Grafana)
- **8GB RAM** recommended for training

### Step 1: Clone the repository

```bash
git clone https://github.com/saichandra1199/loan-risk-mlops.git
cd loan-risk-mlops
```

### Step 2: Install uv (fast Python package manager)

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

### Option A — One-command shortcuts (recommended)

Two shell scripts automate the entire workflow:

```bash
# Full training flow (install → test → download → preprocess → train → promote)
bash run.sh

# Serve the model and make a prediction
bash predict.sh
```

`run.sh` must complete successfully before `predict.sh` will work.
See the step-by-step breakdown below if you want to run stages individually.

---

### Option B — Step by step

### Step 3: Install all dependencies

```bash
uv sync --extra dev
```

This installs all packages (including `openpyxl` and `xlrd` for Excel dataset reading)
into `.venv/` in ~30 seconds.

### Step 4: Configure environment

Create a `.env` file with the SQLite MLflow backend (no server required):

```bash
echo 'MLFLOW__TRACKING_URI=sqlite:///mlruns.db' > .env
```

> **Why not `.env.example`?** The example file points MLflow at `http://localhost:5000`
> (a Docker server). The SQLite backend works locally with zero setup.

### Step 5: Verify the installation

```bash
uv run python -c "import loan_risk; print(f'loan_risk v{loan_risk.__version__} installed OK')"
```

**Expected output:** `loan_risk v0.1.0 installed OK`

### Step 6: Run unit tests (no data or server needed)

```bash
uv run pytest tests/unit/ -v
```

**Expected output:** All 30+ tests should pass.

### Step 7: Download the dataset

```bash
uv run python scripts/download_dataset.py --output data/raw/credit_default_raw.csv
```

This downloads 30,000 rows from the UCI/OpenML repository (~2MB).
OpenML is tried first; if it fails, the script falls back to a direct UCI ZIP download
(requires `openpyxl` + `xlrd`, both included in the project dependencies).

**Expected output:**
```
Downloading UCI Credit Card Default dataset via OpenML...
  Source: https://www.openml.org/d/350
  Downloaded 30,000 rows × 25 columns
  Saved to: data/raw/credit_default_raw.csv
  Default rate: 22.1%
Download complete.
```

> If the file already exists, the script skips the download and exits immediately.

### Step 8: Preprocess the data

```bash
uv run python scripts/preprocess_dataset.py
```

This maps UCI columns to our loan schema and validates with Pandera.

**Expected output:**
```
Loading raw data from: data/raw/credit_default_raw.csv
  Loaded 30,000 rows × 25 columns
Applying feature transformations...
Validating against Pandera schema...
  Schema validation PASSED: 30,000 rows
Saved preprocessed data to: data/raw/loans.parquet
```

### Step 9: Run the full training pipeline

```bash
MLFLOW_TRACKING_URI=sqlite:///mlruns.db uv run python scripts/run_pipeline.py --stage all --skip-tuning
```

This runs: feature engineering → train LightGBM → evaluate → register model in MLflow.

**Expected output (abbreviated):**
```
[features] train=21,000 val=3,000 test=6,000
[train] run_id=b57185d4...
[train] val_auc=0.7691  test_auc=0.7548
[evaluate] Report saved to reports/evaluation/report_b57185d4_2026-03-23.json
[evaluate] Promotion rejected: AUC 0.7548 < threshold 0.8000
[evaluate] DVC metrics saved to reports/evaluation/metrics.json
```

> **Note on promotion:** The default AUC threshold is 0.80. The UCI dataset typically
> yields ~0.75 AUC, so automatic promotion is rejected. Promote manually after training:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlruns.db uv run python - <<'EOF'
from mlflow import MlflowClient
mc = MlflowClient()
versions = mc.search_model_versions("name='loan-risk-classifier'")
latest = max(versions, key=lambda v: int(v.version))
mc.set_registered_model_alias("loan-risk-classifier", "champion", latest.version)
print(f"Promoted version {latest.version} to @champion")
EOF
```

### Step 10: Start the prediction API

```bash
MLFLOW_TRACKING_URI=sqlite:///mlruns.db uv run uvicorn loan_risk.serving.app:create_app --factory --port 8000 --reload
```

**Expected output:**
```
INFO: Started server process [12345]
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Step 11: Make your first prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amount": 15000,
    "annual_income": 45000,
    "employment_years": 3,
    "credit_score": 620,
    "debt_to_income_ratio": 0.45,
    "num_open_accounts": 3,
    "num_delinquencies": 2,
    "loan_purpose": "debt_consolidation",
    "home_ownership": "RENT",
    "loan_term_months": 36
  }'
```

**Expected response:**
```json
{
  "prediction": "REJECT",
  "default_probability": 0.6834,
  "confidence": "HIGH",
  "risk_tier": "HIGH_RISK",
  "top_factors": [
    {"feature": "num_delinquencies", "shap_value": 0.21, "direction": "increases_risk"},
    {"feature": "credit_score", "shap_value": -0.15, "direction": "decreases_risk"},
    {"feature": "debt_to_income_ratio", "shap_value": 0.12, "direction": "increases_risk"}
  ],
  "model_version": "1",
  "request_id": "req_a7f3b2c1",
  "latency_ms": 23.4
}
```

### Optional: Start the full stack with Docker

```bash
docker-compose up -d
```

This starts:
- **MLflow UI**: http://localhost:5000 (view all training runs)
- **Prefect UI**: http://localhost:4200 (schedule and monitor pipelines)
- **Prometheus**: http://localhost:9090 (raw metrics)
- **Grafana**: http://localhost:3000 (metrics dashboard, user: admin, password: admin)

---

## 8. Understanding DVC (Data Version Control)

### What problem does DVC solve?

Consider this scenario:
1. You train Model A on `loans_jan.csv`. It gets AUC=0.76.
2. Six months later, you train Model B on `loans_jul.csv`. It gets AUC=0.73.
3. Was the drop because of new data? Different code? Different parameters?

Without DVC, you can't answer this. **DVC makes it answerable**.

### DVC is like Git, but for data

| Git | DVC |
|---|---|
| Tracks code changes | Tracks data changes |
| `git commit` | `dvc commit` |
| `git checkout v1.0` | `dvc checkout v1.0` |
| `.gitignore` | `.dvcignore` |

### The DVC pipeline (`dvc.yaml`)

Instead of running scripts manually, DVC defines a pipeline:

```
download → preprocess → featurize → train
```

Each stage knows:
- **What it needs** (dependencies: scripts, data files, params)
- **What it produces** (outputs: processed files, model artifacts)
- **What parameters it uses** (from `params.yaml`)

When you run `dvc repro`, DVC:
1. Checks which stages have changed inputs
2. Only re-runs changed stages (like `make`)
3. Caches outputs so you don't recompute what hasn't changed

### Common DVC commands

```bash
# Run the full pipeline (only re-runs changed stages)
dvc repro

# Show the pipeline as a diagram
dvc dag

# Show current metrics
dvc metrics show

# Compare metrics between git commits
dvc metrics diff HEAD~1 HEAD

# See what changed since last run
dvc status

# Compare parameters between experiments
dvc params diff
```

### How DVC tracks data

When DVC tracks a file, it:
1. Computes the file's hash (like a fingerprint)
2. Stores the file in a local cache (`.dvc/cache/`)
3. Creates a small `.dvc` file (committed to git) that records the hash

This means you commit only the `.dvc` file (a few bytes) to git, not the data.
To share data with teammates, you push to a DVC remote (S3, GCS, SFTP, etc.).

---

## 9. Understanding GitHub Actions (CI/CD)

### What is CI/CD?

- **CI** (Continuous Integration): Automatically test code every time someone pushes
- **CD** (Continuous Delivery): Automatically deploy or train when tests pass

Without CI/CD, bugs slip through because nobody remembers to run tests before pushing.

### Our workflows

#### `ci.yml` — Runs on every push

```
You push code
     ↓
GitHub spins up an Ubuntu server
     ↓
Installs Python 3.11 + all dependencies
     ↓
Runs ruff (checks code style)
     ↓
Runs pytest tests/unit/ (checks logic)
     ↓
Green ✓ or Red ✗ shown on your PR/commit
```

The whole thing takes ~2-3 minutes. If tests fail, you know immediately.

#### `training.yml` — Runs weekly or manually

```
Every Sunday 02:00 UTC (or manual trigger)
     ↓
Checks out the latest code
     ↓
Installs dependencies
     ↓
Runs: dvc repro
  → downloads data
  → preprocesses it
  → engineers features
  → trains model
  → evaluates model
     ↓
Publishes metrics to the GitHub Actions summary
     ↓
Uploads training reports as downloadable artifacts
```

### Viewing results

After a training run:
1. Go to your repo on GitHub
2. Click **Actions** tab
3. Click the latest **Training Pipeline** run
4. See the metrics table in the job summary
5. Download the reports artifact for detailed results

---

## 10. Current Pipeline Status — All Green

This section shows the exact state of all three CI/CD workflows as of the initial setup.
Every workflow badge at the top of this README is live — click any badge to see the latest run.

### The three workflows

| Workflow | Trigger | What it checks | Typical duration |
|---|---|---|---|
| **CI — Lint & Unit Tests** | Every push to any branch | Code style (ruff) + 30 unit tests | ~35 seconds |
| **Test Serving Layer** | Push touching `src/serving/` or `test_api.py` | 7 FastAPI endpoint integration tests | ~45 seconds |
| **Training Pipeline (DVC)** | Every Sunday 02:00 UTC + manual trigger | Full 4-stage DVC pipeline (download → train) | ~5 minutes |

### What "green" means for each workflow

#### CI — Lint & Unit Tests

```
✓ Checkout code
✓ Set up Python 3.11
✓ Install uv                    (fast package manager, ~10s)
✓ Install dependencies          (uv sync --extra dev, uses cached uv.lock)
✓ Lint with ruff                (checks code style — 0 errors)
✓ Check formatting              (warning only)
✓ Run unit tests                (30/30 passed in ~3s)
✓ Upload test results
```

All 30 unit tests cover:
- `test_schemas.py` — Pandera validates correct data, rejects bad credit scores / invalid purposes
- `test_transformers.py` — Each custom sklearn transformer produces correct output
- `test_splits.py` — Stratified split preserves default rate, no overlap between sets
- `test_metrics.py` — AUC, Gini, KS statistic calculations are mathematically correct

#### Test Serving Layer

```
✓ Checkout code
✓ Set up Python 3.11
✓ Install uv + dependencies
✓ Run API integration tests     (7/7 passed in ~3s)
```

The 7 integration tests cover:
- `POST /predict` returns HTTP 200 with valid input
- `POST /predict` response matches the `PredictionResponse` schema
- `POST /predict` returns HTTP 422 (validation error) for credit_score < 300
- `POST /predict` returns HTTP 422 for unknown `loan_purpose`
- `GET /health` returns `model_loaded: true`
- `GET /metrics` returns Prometheus-format text
- `GET /model-info` returns model name + version

The tests use a **mock predictor** — no real MLflow model needed.
This is done with FastAPI's `app.dependency_overrides` system.

#### Training Pipeline (DVC)

```
✓ Checkout code
✓ Set up Python 3.11
✓ Install uv + DVC
✓ Create required directories
✓ dvc repro --no-lock           (runs all 4 pipeline stages)
  ✓ download   → data/raw/credit_default_raw.csv   (30K rows from UCI)
  ✓ preprocess → data/raw/loans.parquet            (Pandera validated)
  ✓ featurize  → data/processed/ + preprocessor.pkl
  ✓ train      → reports/evaluation/metrics.json
✓ Display metrics               (AUC, Gini, KS shown in job summary)
✓ Upload reports artifact       (downloadable for 30 days)
```

### How to view live results

1. Go to https://github.com/saichandra1199/loan-risk-mlops
2. Click the **Actions** tab
3. Click any workflow run to see step-by-step logs
4. For the Training Pipeline: click the run → click the job → scroll to
   **"Display metrics"** to see the AUC table, or download the **reports** artifact

### How to trigger the training pipeline manually

From the GitHub web UI:
1. Actions → **Training Pipeline (DVC)**
2. Click **Run workflow** (top right)
3. Set inputs:
   - `model_name`: `lgbm` or `xgboost`
   - `n_trials`: `50` (number of Optuna tuning trials)
   - `skip_tuning`: `false` (set `true` for a fast run without tuning)

From the command line (requires gh CLI):
```bash
gh workflow run training.yml \
  -f model_name=lgbm \
  -f n_trials=50 \
  -f skip_tuning=false
```

### Watching workflow results from the terminal

```bash
# Install gh CLI (if not already installed)
# Download from: https://github.com/cli/cli/releases/latest

# List recent workflow runs
gh run list --repo saichandra1199/loan-risk-mlops --limit 10

# Watch a run live (replace RUN_ID with the number from the list)
gh run watch RUN_ID --repo saichandra1199/loan-risk-mlops

# See logs from a failed run
gh run view RUN_ID --repo saichandra1199/loan-risk-mlops --log-failed

# Download artifacts from a training run
gh run download RUN_ID --repo saichandra1199/loan-risk-mlops
```

---

## 11. Pipeline Stages — What Happens at Each Step

### Stage 1: Download (`python scripts/download_dataset.py`)

**Input**: Nothing (downloads from internet)
**Output**: `data/raw/credit_default_raw.csv`
**What it does**: Uses scikit-learn's `fetch_openml()` to download the UCI dataset from OpenML (no authentication required). Falls back to direct UCI HTTP download if OpenML is unavailable.
**Why DVC caches it**: After the first run, DVC stores the CSV in its local cache. Subsequent `dvc repro` calls skip this stage unless the script itself changes.

### Stage 2: Preprocess (`python scripts/preprocess_dataset.py`)

**Input**: `data/raw/credit_default_raw.csv`
**Output**: `data/raw/loans.parquet` + `reports/validation/validation_report.json`
**What it does**: Maps 24 UCI columns to 12 canonical loan features (see Section 3). Validates output against Pandera schema. Fails loudly if data is wrong.
**Why Parquet**: Parquet is a columnar binary format. Reading 30K rows from Parquet is ~10x faster than CSV and uses ~3x less disk space.

### Stage 3: Feature Engineering (`run_pipeline.py --stage features`)

**Input**: `data/raw/loans.parquet`
**Output**: `data/processed/` (train/val/test) + `artifacts/preprocessor.pkl`
**What it does**:
1. Stratified 70/10/20 split (preserving 22% default rate in each split)
2. Fits sklearn Pipeline on training data: log-transforms loan_amount and annual_income, bins credit_score into 5 FICO bands, adds loan_to_income ratio, one-hot encodes categoricals, standard-scales numerics
3. Saves fitted pipeline as `preprocessor.pkl`

**Why stratified split**: If you split randomly, the train set might have 25% defaults and test set might have 19%. Stratified split ensures each split reflects the true 22% default rate.

### Stage 4: Train + Evaluate (`run_pipeline.py --stage train && --stage evaluate`)

**Input**: `data/processed/` + `artifacts/preprocessor.pkl`
**Output**: `reports/evaluation/metrics.json` + MLflow run

**Training**:
- Computes `scale_pos_weight = n_negative / n_positive = 23,400/6,600 ≈ 3.5` (tells LightGBM to weight defaults 3.5× more)
- Trains LightGBM with early stopping on validation set
- Calibrates decision threshold to maximise F-beta (β=2, recall-weighted): better to flag a risky customer as REJECT than miss them

**Evaluation**:
- AUC-ROC: Overall ranking ability (0.5=random, 1.0=perfect)
- Gini = 2×AUC - 1: Industry-standard credit scoring metric
- KS statistic: Maximum separation between default and non-default probability distributions
- Bias audit: Checks AUC across education levels and marital status

**Promotion gate**: Model is only promoted to "champion" in the MLflow registry if AUC ≥ 0.75.

---

## 12. API Reference

### POST /predict

Predict default risk for a loan application.

**Request body:**
```json
{
  "loan_amount": 15000,
  "annual_income": 45000,
  "employment_years": 3,
  "credit_score": 680,
  "debt_to_income_ratio": 0.35,
  "num_open_accounts": 4,
  "num_delinquencies": 0,
  "loan_purpose": "debt_consolidation",
  "home_ownership": "RENT",
  "loan_term_months": 36
}
```

**Fields:**

| Field | Type | Range | Description |
|---|---|---|---|
| `loan_amount` | float | 100–100,000 | Requested loan amount (USD) |
| `annual_income` | float | 1,000–10,000,000 | Gross annual income |
| `employment_years` | int | 0–50 | Years at current employer |
| `credit_score` | int | 300–850 | FICO-equivalent credit score |
| `debt_to_income_ratio` | float | 0.0–2.0 | Monthly debt / monthly income |
| `num_open_accounts` | int | 0–100 | Active credit lines |
| `num_delinquencies` | int | 0–50 | 30+ day late payments (past 2y) |
| `loan_purpose` | string | see below | Purpose of the loan |
| `home_ownership` | string | RENT, MORTGAGE, OWN | Housing situation |
| `loan_term_months` | int | 36 or 60 | Repayment term |

Valid `loan_purpose` values: `debt_consolidation`, `home_improvement`, `major_purchase`, `medical`, `vacation`, `other`

**Response:**
```json
{
  "prediction": "REJECT",
  "default_probability": 0.6834,
  "confidence": "HIGH",
  "risk_tier": "HIGH_RISK",
  "top_factors": [
    {
      "feature": "num_delinquencies",
      "shap_value": 0.21,
      "direction": "increases_risk"
    }
  ],
  "model_version": "1",
  "request_id": "req_a7f3b2c1d4e5",
  "latency_ms": 23.4
}
```

**Risk tiers:**
- `LOW_RISK` — default probability < 20%
- `MEDIUM_RISK` — 20%–45%
- `HIGH_RISK` — 45%–70%
- `VERY_HIGH_RISK` — > 70%

### GET /health

Check if the service is running and the model is loaded.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1",
  "uptime_seconds": 3621.4
}
```

### GET /metrics

Prometheus-format metrics for monitoring (latency, prediction counts, etc.)

```bash
curl http://localhost:8000/metrics
```

### GET /model-info

Current model metadata.

```bash
curl http://localhost:8000/model-info
```

### GET /docs

Auto-generated interactive API documentation (Swagger UI). Open in browser: http://localhost:8000/docs

---

## 13. Configuration Guide

### The two configuration files

**`config/settings.yaml`** — base configuration:
```yaml
mlflow:
  tracking_uri: "http://localhost:5000"  # Change to "sqlite:///mlruns.db" for no-server mode
  experiment_name: "loan-risk"

model:
  name: "lgbm"                    # Change to "xgboost" to try XGBoost
  promotion_auc_threshold: 0.75   # Lower if AUC doesn't reach 0.75 on your data

training:
  test_size: 0.20      # 20% of data for final test evaluation
  val_size: 0.10       # 10% of data for validation during training
  n_trials: 50         # More trials = better hyperparameters but slower
  random_seed: 42      # Change this to get different random splits
```

**`params.yaml`** — DVC-tracked parameters (changing these triggers re-runs):
```yaml
training:
  model_name: lgbm    # lgbm or xgboost
  random_seed: 42
```

### Using environment variables

Any config key can be overridden with an environment variable:
```bash
# Override MLflow tracking URI
MLFLOW__TRACKING_URI=sqlite:///local.db uv run python scripts/run_pipeline.py --stage train

# Override model type
TRAINING__MODEL_NAME=xgboost uv run python scripts/run_pipeline.py --stage train
```

The `__` separates nested keys: `MLFLOW__TRACKING_URI` maps to `mlflow.tracking_uri`.

---

## 14. Running Experiments

### How to compare two model types

1. Train with LightGBM (default):
```bash
uv run python scripts/run_pipeline.py --stage train --skip-tuning
```

2. Change the model in `params.yaml`:
```yaml
training:
  model_name: xgboost
```

3. Train again:
```bash
uv run python scripts/run_pipeline.py --stage train --skip-tuning
```

4. Compare with DVC:
```bash
dvc params diff     # See what parameters changed
dvc metrics diff    # Compare AUC between runs
```

5. Or compare in MLflow UI:
```bash
docker-compose up -d mlflow
# Open: http://localhost:5000
```

### How to run hyperparameter tuning

```bash
# Run 50 Optuna trials (takes ~10 minutes)
uv run python scripts/run_pipeline.py --stage tune --n-trials 50

# Then train with the best parameters
uv run python scripts/run_pipeline.py --stage train --stage evaluate
```

---

## 15. How to Implement New Features

This section is a practical guide for extending the pipeline.
Every change follows the same pattern: **code → schema → pipeline → test → push**.

---

### Pattern A: Add a new input feature to the model

**Example**: You want to add `loan_age_days` (how old the loan application is) as a feature.

**Step 1 — Add the column name to the feature definitions**

Open `src/loan_risk/features/definitions.py`:
```python
# Before:
NUMERIC_FEATURES = [
    "loan_amount", "annual_income", "employment_years",
    ...
]

# After — add your new feature:
NUMERIC_FEATURES = [
    "loan_amount", "annual_income", "employment_years",
    "loan_age_days",   # ← add here
    ...
]
```

**Step 2 — Update the Pandera schema**

Open `src/loan_risk/data/schemas.py` and add the column:
```python
class RawLoanSchema(pa.DataFrameModel):
    ...
    loan_age_days: Series[int] = pa.Field(ge=0, le=3650)  # 0 to 10 years
```

**Step 3 — Produce the column in the preprocessing script**

Open `scripts/preprocess_dataset.py`, find the `preprocess()` function, and add:
```python
result["loan_age_days"] = 0  # placeholder if not in UCI data
# Or if you have a real source column:
result["loan_age_days"] = pd.to_numeric(df["some_date_column"], errors="coerce").fillna(0)
```

**Step 4 — The pipeline handles the rest automatically**

Since `loan_age_days` is in `NUMERIC_FEATURES`, the sklearn `ColumnTransformer` in
`src/loan_risk/features/pipeline.py` will automatically:
- Apply `StandardScaler` to it
- Include it in the training matrix

**Step 5 — Write a unit test**

Add to `tests/unit/test_schemas.py`:
```python
def test_schema_accepts_valid_loan_age():
    df = make_valid_df()
    df["loan_age_days"] = 365
    validate_raw(pl.from_pandas(df))  # should not raise

def test_schema_rejects_negative_loan_age():
    df = make_valid_df()
    df["loan_age_days"] = -1
    with pytest.raises(DataValidationError):
        validate_raw(pl.from_pandas(df))
```

**Step 6 — Re-run the pipeline**

```bash
dvc repro          # DVC detects schemas.py changed → re-runs preprocess + featurize + train
```

---

### Pattern B: Add a custom feature transformer

**Example**: You want to add a `IncomeToDebtFlag` transformer that marks
customers whose income is very low relative to their debt.

**Step 1 — Write the transformer**

Open `src/loan_risk/features/transformers.py` and add:
```python
class IncomeToDebtFlag(BaseEstimator, TransformerMixin):
    """Flags customers with income < 3× their monthly debt payment."""

    def __init__(self, multiplier: float = 3.0) -> None:
        self.multiplier = multiplier

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        monthly_income = out["annual_income"] / 12
        monthly_debt = out["loan_amount"] / out.get("loan_term_months", 36)
        out["low_income_flag"] = (monthly_income < self.multiplier * monthly_debt).astype(int)
        return out
```

**Step 2 — Add it to the feature pipeline**

Open `src/loan_risk/features/pipeline.py`, find `build_feature_pipeline()`, and insert your step:
```python
from loan_risk.features.transformers import (
    ...
    IncomeToDebtFlag,   # ← import it
)

def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        ("loan_to_income", LoanToIncomeRatioTransformer()),
        ("log_transforms", LogTransformer(...)),
        ("income_debt_flag", IncomeToDebtFlag(multiplier=3.0)),   # ← add step
        ...
    ])
```

**Step 3 — Add to feature definitions**

```python
# In src/loan_risk/features/definitions.py
ENGINEERED_FEATURES = [
    "loan_to_income_ratio",
    "log_loan_amount",
    "log_annual_income",
    "credit_score_band",
    "high_delinquency_risk",
    "low_income_flag",          # ← add here
]
```

**Step 4 — Write a unit test**

```python
# In tests/unit/test_transformers.py
class TestIncomeToDebtFlag:
    def test_flags_low_income(self):
        df = pd.DataFrame({"annual_income": [12000], "loan_amount": [50000], "loan_term_months": [36]})
        out = IncomeToDebtFlag(multiplier=3.0).fit_transform(df)
        assert out["low_income_flag"].iloc[0] == 1

    def test_does_not_flag_high_income(self):
        df = pd.DataFrame({"annual_income": [120000], "loan_amount": [10000], "loan_term_months": [36]})
        out = IncomeToDebtFlag(multiplier=3.0).fit_transform(df)
        assert out["low_income_flag"].iloc[0] == 0
```

---

### Pattern C: Add a new API endpoint

**Example**: Add `GET /predict/batch` that accepts a list of applicants and returns all predictions at once.

**Step 1 — Add request/response schemas**

Open `src/loan_risk/serving/schemas.py` and add:
```python
class BatchPredictionRequest(BaseModel):
    applications: list[LoanApplicationRequest]

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    failed: int
```

**Step 2 — Add the route**

Open `src/loan_risk/serving/routes.py` and add:
```python
from loan_risk.serving.schemas import BatchPredictionRequest, BatchPredictionResponse

@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    predictor: ModelPredictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    results, failed = [], 0
    for i, app in enumerate(request.applications):
        try:
            pred = predictor.predict(app, request_id=f"batch_{i}")
            results.append(pred)
        except Exception:
            failed += 1
    return BatchPredictionResponse(predictions=results, count=len(results), failed=failed)
```

**Step 3 — Add an integration test**

```python
# In tests/integration/test_api.py
def test_batch_predict(client):
    response = client.post("/predict/batch", json={"applications": [SAMPLE_REQUEST, SAMPLE_REQUEST]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["failed"] == 0
```

**Step 4 — Push and watch CI turn green**

```bash
git add -A && git commit -m "Add batch prediction endpoint"
git push origin main
# → serve.yml triggers because serving code changed
# → All tests run automatically
```

---

### Pattern D: Add a new evaluation metric

**Example**: You want to track `precision_at_30pct` (precision when approving the top 30% lowest-risk applicants).

**Step 1 — Add the metric function**

Open `src/loan_risk/evaluation/metrics.py`:
```python
def compute_precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.30) -> float:
    """Precision when approving the lowest-risk k% of applicants."""
    n_approve = int(len(y_prob) * k)
    # Sort by probability ascending (lowest risk first), take top k%
    top_k_indices = np.argsort(y_prob)[:n_approve]
    # Precision = fraction of approved who actually didn't default
    precision = 1.0 - y_true[top_k_indices].mean()
    return float(precision)
```

**Step 2 — Include it in compute_classification_metrics()**

```python
def compute_classification_metrics(...) -> dict:
    return {
        "auc_roc": ...,
        "gini": ...,
        "ks_statistic": ...,
        "precision_at_30pct": compute_precision_at_k(y_true, y_prob, k=0.30),   # ← add
    }
```

**Step 3 — Log it to MLflow in the trainer**

In `src/loan_risk/training/trainer.py`, find where `mlflow.log_metrics()` is called:
```python
mlflow.log_metrics({
    "val_auc": val_metrics["auc_roc"],
    "val_gini": val_metrics["gini"],
    "val_precision_at_30pct": val_metrics["precision_at_30pct"],   # ← add
    ...
})
```

**Step 4 — Add it to the DVC metrics output**

In `scripts/run_pipeline.py`, find where `metrics.json` is written:
```python
metrics_path.write_text(json.dumps({
    "test_auc": training_result.test_auc,
    "test_precision_at_30pct": training_result.test_metrics.get("precision_at_30pct"),  # ← add
    ...
}))
```

Now `dvc metrics show` and the GitHub Actions job summary will include this metric.

---

### Pattern E: Change the ML model

**Example**: Try XGBoost instead of LightGBM.

**Option 1 — Quick experiment (no code change)**

```bash
# Edit params.yaml:
# training:
#   model_name: xgboost

dvc repro   # DVC sees params.yaml changed, re-runs train stage only
```

**Option 2 — Add a brand new model type**

In `src/loan_risk/training/models.py`, add your model to `get_model()`:
```python
def get_model(name: str, params: dict, ...) -> Any:
    if name == "lgbm":
        return LGBMClassifier(...)
    elif name == "xgboost":
        return XGBClassifier(...)
    elif name == "catboost":
        from catboost import CatBoostClassifier     # ← new model
        return CatBoostClassifier(**params, ...)
    else:
        raise ValueError(f"Unknown model: {name}")
```

Add the new tuning objective in `src/loan_risk/tuning/objective.py`:
```python
def catboost_objective(trial, X_train, y_train, X_val, y_val, ...):
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 100, 1000),
    }
    model = get_model("catboost", params, ...)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
```

---

### Quick reference: which file to change for what

| What you want to do | File(s) to change |
|---|---|
| Add/remove an input feature | `features/definitions.py` + `data/schemas.py` + `preprocess_dataset.py` |
| Change how a feature is transformed | `features/transformers.py` + `features/pipeline.py` |
| Try a different model | `params.yaml` (no code) or `training/models.py` (new type) |
| Add a new evaluation metric | `evaluation/metrics.py` + `training/trainer.py` |
| Change the promotion threshold | `params.yaml` → `model.promotion_auc_threshold` |
| Add a new API endpoint | `serving/routes.py` + `serving/schemas.py` |
| Change drift alert thresholds | `config/monitoring_thresholds.yaml` |
| Change the training schedule | `.github/workflows/training.yml` (cron expression) |
| Add a new Prefect flow | `pipelines/flows.py` + `pipelines/tasks.py` |

---

## 16. Monitoring the Pipeline

Monitoring means watching what is happening right now and catching problems early.
There are three levels of monitoring in this project:

---

### Level 1: Pipeline health (did it run? did it pass?)

**GitHub Actions** — pipeline-level health:
```bash
# From terminal
gh run list --repo saichandra1199/loan-risk-mlops --limit 10

# Or watch the live badges at the top of this README
```

Go to: https://github.com/saichandra1199/loan-risk-mlops/actions

What to look for:
- Red X on `CI — Lint & Unit Tests` → a code change broke a test or style rule
- Red X on `Training Pipeline` → the model failed to train or missed the AUC gate
- Yellow circle → the run is still in progress

---

### Level 2: Experiment metrics (is the model getting better or worse?)

**MLflow UI** — experiment-level tracking:

```bash
# Start MLflow UI pointing at your local SQLite database
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
# Open: http://localhost:5000
```

Or with Docker (full stack):
```bash
docker-compose up -d mlflow
# Open: http://localhost:5000
```

What you can do in the MLflow UI:
- **Experiments view**: See every training run with its AUC, Gini, KS side by side
- **Compare runs**: Select 2+ runs → "Compare" → see parameter differences + metric plots
- **Artifacts**: Download the model, feature importance plots, SHAP summary plots
- **Model Registry**: See which version is "champion", which are archived

Key metrics to watch:
```
test_auc     → should be ≥ 0.75 (promotion gate)
val_auc      → should be close to test_auc (large gap = overfitting)
val_gini     → 2×AUC - 1; ≥ 0.50 is good for credit scoring
val_ks       → KS statistic; ≥ 0.35 is good
threshold    → calibrated decision boundary (typically 0.30-0.50)
```

**From DVC**:
```bash
# Show current metrics
dvc metrics show

# Compare current run to the previous git commit
dvc metrics diff HEAD~1

# Compare across multiple git commits
dvc metrics diff main~3 main

# See which parameters changed
dvc params diff HEAD~1
```

Example output of `dvc metrics show`:
```
Path                              auc_roc    gini    ks_stat
reports/evaluation/metrics.json  0.764      0.528   0.387
```

---

### Level 3: Live API health (is the service responding?)

**Prometheus + Grafana** — real-time API metrics:

```bash
docker-compose up -d prometheus grafana
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (login: admin / admin)
```

The FastAPI server automatically exposes metrics at `GET /metrics`:
```bash
curl http://localhost:8000/metrics
```

Key metrics exported:
```
loan_risk_predictions_total{prediction="APPROVE"}   # Count of approvals
loan_risk_predictions_total{prediction="REJECT"}    # Count of rejections
loan_risk_prediction_latency_seconds{quantile="0.99"}  # 99th percentile latency
loan_risk_default_probability{bucket="..."}         # Histogram of predicted probabilities
```

**Health endpoint for quick checks**:
```bash
curl http://localhost:8000/health

# Response tells you:
# - Is the model loaded?
# - Which model version is running?
# - How long has the server been up?
```

---

### Level 4: Data drift (is the world changing?)

**Evidently reports** — statistical drift detection:

```python
# Run the drift check (after collecting enough live data)
from loan_risk.monitoring.drift import generate_drift_report
import polars as pl

reference_df = pl.read_parquet("data/reference/train.parquet")
current_df   = pl.read_parquet("data/monitoring/live_predictions.parquet")

report = generate_drift_report(reference_df, current_df, output_path="reports/monitoring/drift_report.html")
print(report)  # Shows PSI for each feature
```

Or via the Prefect pipeline:
```bash
uv run python -c "
from pipelines.flows import daily_monitor_flow
daily_monitor_flow(
    reference_path='data/reference/train.parquet',
    current_path='data/monitoring/live_predictions.parquet',
    auto_retrain=False
)
"
```

**Understanding PSI (Population Stability Index)**:

| PSI Value | Meaning | Action |
|---|---|---|
| < 0.10 | No drift | No action needed |
| 0.10 – 0.15 | Slight drift | Monitor more closely |
| 0.15 – 0.25 | Moderate drift | Investigate + consider retraining |
| > 0.25 | Significant drift | Retrain immediately |

The drift report is an HTML file (`reports/monitoring/drift_report.html`).
Open it in a browser to see per-feature drift visualisations.

---

### Monitoring checklist — what to check each week

```
□ GitHub Actions: all workflows green this week?
□ MLflow: latest training run AUC ≥ 0.75?
□ MLflow: val_auc and test_auc within 0.03 of each other? (no overfitting)
□ /health endpoint: model_loaded = true?
□ Drift report: any feature PSI > 0.15?
□ Live AUC (if labels available): AUC ≥ 0.70?
□ Prediction distribution: approval rate stable (not suddenly 90% REJECT)?
```

---

## 17. Debugging Guide

This section explains how to diagnose the most common problems in each layer of the pipeline.

---

### How to read structured logs

All logs use JSON format in production. Every log line looks like:
```json
{
  "event": "model_trained",
  "run_id": "a3f8b2c1",
  "val_auc": 0.7682,
  "test_auc": 0.7591,
  "level": "info",
  "logger": "loan_risk.training.trainer",
  "timestamp": "2026-01-15T14:23:01.234Z"
}
```

Key fields:
- `event` — what happened (grep for this to find specific events)
- `level` — `debug`, `info`, `warning`, `error`
- `logger` — which module logged this (tells you exactly where in code)
- `timestamp` — when it happened

**Useful grep patterns**:
```bash
# Watch all log output in real time
uv run python scripts/run_pipeline.py --stage all 2>&1 | jq .

# Find all errors
uv run python scripts/run_pipeline.py --stage train 2>&1 | jq 'select(.level == "error")'

# Find model training events only
... | jq 'select(.logger | startswith("loan_risk.training"))'

# Find the final AUC
... | jq 'select(.event == "model_trained") | {auc: .test_auc, run_id: .run_id}'
```

If you don't have `jq` installed: `sudo apt install jq` or `brew install jq`.

---

### Debugging the data pipeline

**Problem: Pandera schema validation fails**

The error message tells you exactly which column failed and why:
```
DataValidationError: Schema validation failed:
  Column 'credit_score': 15 values out of range [300, 850]
  Failing values: [1500, 0, -99, ...]
```

How to investigate:
```python
import pandas as pd
df = pd.read_parquet("data/raw/loans.parquet")

# Check the failing column
print(df["credit_score"].describe())
print(df["credit_score"].value_counts().head(20))
print(df[df["credit_score"] > 850])   # Find the out-of-range rows
```

The preprocessing script auto-clips most values, but if something is structurally wrong,
you need to trace back to `scripts/preprocess_dataset.py` and find where `credit_score` is derived.

**Problem: Data splits have wrong default rates**

```python
import polars as pl
train = pl.read_parquet("data/processed/train.parquet")
val   = pl.read_parquet("data/processed/val.parquet")
test  = pl.read_parquet("data/processed/test.parquet")

# Expected: all three ~22%
print("Train default rate:", train["loan_default"].mean())
print("Val default rate:  ", val["loan_default"].mean())
print("Test default rate: ", test["loan_default"].mean())
```

If the rates differ by more than 3%, the stratification failed. Check `random_seed` in `params.yaml`.

**Problem: Feature count mismatch between training and inference**

```python
import joblib
pipeline = joblib.load("artifacts/preprocessor.pkl")
feature_names = pipeline.get_feature_names_out()
print(f"Pipeline expects {len(feature_names)} features:")
print(list(feature_names))
```

The model was trained with a different set of features than what the API is receiving.
Always re-run the full `dvc repro` after changing features — never train and serve with
a mismatched preprocessor.

---

### Debugging the model training

**Problem: val_auc much lower than expected**

Common causes and checks:
```python
import numpy as np

# Check class balance in training data
y_train = np.load("data/processed/y_train.npy")
print(f"Default rate in training: {y_train.mean():.1%}")
print(f"scale_pos_weight should be: {(1-y_train.mean()) / y_train.mean():.2f}")

# Check if scale_pos_weight in config matches actual imbalance
# It should be ~3.5 for this dataset
```

**Problem: Training hangs at Optuna trials**

Optuna prints a progress table. If it's stuck on a single trial for >5 minutes:
```python
# In src/loan_risk/tuning/objective.py, the early stopping should kick in
# Add more verbose logging by temporarily changing:
import optuna
optuna.logging.set_verbosity(optuna.logging.DEBUG)
```

Or just skip tuning for debugging:
```bash
uv run python scripts/run_pipeline.py --stage train --skip-tuning
```

**Problem: Model doesn't pass the AUC gate**

```bash
# Check what AUC was achieved
cat reports/evaluation/metrics.json

# Check what the threshold is
cat params.yaml | grep promotion_auc_threshold

# Temporarily lower the threshold to test the pipeline end-to-end:
# In params.yaml, change:
#   model:
#     promotion_auc_threshold: 0.60   # Lower for testing only
```

---

### Debugging the API

**Problem: 500 Internal Server Error on `/predict`**

Enable detailed error messages:
```bash
# Run the server with detailed tracebacks
PYTHONPATH=src uv run uvicorn loan_risk.serving.app:create_app \
  --factory --port 8000 --reload --log-level debug
```

Then try the failing request again. The server logs will show the full Python traceback.

**Problem: 422 Unprocessable Entity**

The request body fails Pydantic validation. The response body tells you exactly what's wrong:
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "credit_score"],
      "msg": "Input should be less than or equal to 850",
      "input": 900
    }
  ]
}
```

**Problem: `Model not loaded` error**

```bash
# Check if a champion model exists in the registry
MLFLOW_TRACKING_URI=sqlite:///mlruns.db \
  uv run python -c "
from loan_risk.registry.client import MLflowRegistryClient
client = MLflowRegistryClient()
print(client.list_versions())
"
```

If the list is empty, train and promote a model first:
```bash
uv run python scripts/run_pipeline.py --stage all --skip-tuning
```

---

### Debugging GitHub Actions failures

**Step 1: Read the error, don't just look at the red X**

Click the failing job → expand the failing step → read the actual error message.
The most common causes:

| Error in CI | Likely cause | Fix |
|---|---|---|
| `No module named 'X'` | Dependency missing from `pyproject.toml` | Add it to `[project.dependencies]` |
| `assert X == Y` in tests | A test expected a different value | Read the test + fix the logic |
| `ruff: Found N errors` | Code style violations | Run `uv run ruff check --fix` locally |
| `dvc repro: stage failed` | A pipeline script crashed | Run the failing script locally and read the error |
| `Permission denied` | File/directory doesn't exist | Add `mkdir -p` before writing to the path |

**Step 2: Reproduce locally**

The CI environment is just Ubuntu + Python 3.11. Reproduce it:
```bash
# Exact same commands as CI:
uv sync --extra dev
MLFLOW_TRACKING_URI=sqlite:///mlruns.db PYTHONPATH=src uv run pytest tests/unit/ -v
```

**Step 3: Check the full log**

```bash
gh run view RUN_ID --repo saichandra1199/loan-risk-mlops --log-failed 2>&1 | less
```

**Step 4: Add a debug print to narrow down**

If a test is failing with a confusing error, add a `print()` statement before the assertion:
```python
def test_something():
    result = compute_thing()
    print(f"DEBUG: result = {result}")   # ← will appear in CI logs
    assert result == expected_value
```

Then push. The CI logs will show your debug output.

---

### Common debugging workflow (step by step)

```
1. Something is wrong (red CI, bad AUC, API error)
         ↓
2. Read the error message carefully
   - What file/line?
   - What was expected vs actual?
         ↓
3. Reproduce locally
   - Run the exact failing command
   - Add -v or --log-level debug for more output
         ↓
4. Isolate the problem
   - Comment out code until the error disappears
   - Binary search: is the problem in step A or step B?
         ↓
5. Check the data at each stage
   - print(df.shape), print(df.dtypes), print(df.head())
   - Check for NaN: print(df.isnull().sum())
   - Check ranges: print(df.describe())
         ↓
6. Fix the issue
         ↓
7. Write a test that would have caught it
         ↓
8. Push — verify CI turns green
```

---

## 18. Troubleshooting

### "No module named 'loan_risk'"

The `src/` directory needs to be in Python's path:
```bash
PYTHONPATH=src uv run python scripts/run_pipeline.py --stage train
# Or use the uv run prefix which handles this automatically
uv run python scripts/run_pipeline.py --stage train
```

### "MLflow connection refused" / retrying localhost:5000

The training pipeline works WITHOUT a running MLflow server.
The `.env.example` file points to `http://localhost:5000` (Docker). Override it:
```bash
# In .env (use double-underscore for nested keys):
MLFLOW__TRACKING_URI=sqlite:///mlruns.db

# Or prefix any command:
MLFLOW_TRACKING_URI=sqlite:///mlruns.db uv run python scripts/run_pipeline.py --stage all
```

### "Model not loaded. Call predictor.load() first." (500 on /predict)

The API started but no `@champion` model alias exists in the registry.
Either the pipeline AUC fell below the promotion threshold, or training hasn't run yet.
Promote the latest trained version manually:
```bash
MLFLOW_TRACKING_URI=sqlite:///mlruns.db uv run python - <<'EOF'
from mlflow import MlflowClient
mc = MlflowClient()
versions = mc.search_model_versions("name='loan-risk-classifier'")
latest = max(versions, key=lambda v: int(v.version))
mc.set_registered_model_alias("loan-risk-classifier", "champion", latest.version)
print(f"Promoted version {latest.version} to @champion")
EOF
```
Then **restart the API** — the model is loaded once at startup, not per request.

### "Promotion rejected: AUC < threshold 0.8000"

This is expected — the UCI credit dataset yields ~0.75 AUC with default hyperparameters.
The threshold exists to prevent bad models reaching production. For local dev, either:
- Use the manual promotion command above, or
- Lower the threshold in `.env`: `MODEL__PROMOTION_AUC_THRESHOLD=0.70`

### "ERROR: pandas not installed" during UCI dataset download

Despite pandas being installed, this error can appear when `openpyxl` or `xlrd` are
missing (needed to read the `.xls` file inside the UCI ZIP). Both are now declared as
project dependencies — run `uv sync` to install them:
```bash
uv sync
```

### "Pandera schema validation failed: credit_score out of range"

The preprocessed data doesn't match the expected schema. Re-run preprocessing:
```bash
rm data/raw/loans.parquet
uv run python scripts/preprocess_dataset.py
```

### "fetch_openml download failed"

The OpenML download sometimes times out. Try the direct UCI download:
```bash
uv run python scripts/download_dataset.py --method uci
```

### "dvc: command not found"

DVC is installed as part of the project dependencies:
```bash
uv run dvc repro    # Use uv run prefix
# or add .venv/bin to PATH:
source .venv/bin/activate
dvc repro
```

### "Port 8000 already in use"

Another process is using port 8000. Either kill it or use a different port:
```bash
uv run uvicorn loan_risk.serving.app:create_app --factory --port 8001
```

### "Model not found in registry" / version '@champion' not found

The champion model hasn't been promoted yet. Run the full pipeline and then promote:
```bash
bash run.sh
# run.sh handles promotion automatically as the final step
```

---

## 19. FAQ for Beginners

**Q: What's the difference between `data/raw/` and `data/processed/`?**

A: `data/raw/` contains the original data as-downloaded (credit_default_raw.csv) and the preprocessed canonical format (loans.parquet). `data/processed/` contains the train/val/test splits after feature engineering — the actual arrays the model trains on.

**Q: Why is the data not in git?**

A: Git is designed for code (text files), not data. A 30MB CSV would make every `git clone` slow and bloat the repository. DVC tracks data separately in a cache, and only the `.dvc` metadata files (a few bytes each) go into git.

**Q: What is AUC-ROC and why do we use it?**

A: AUC (Area Under the ROC Curve) measures a model's ability to rank positive examples above negative ones. A random model has AUC=0.5; a perfect model has AUC=1.0. For imbalanced datasets like ours (22% defaults), AUC is better than accuracy because accuracy would be high just by always predicting "no default".

**Q: Why LightGBM and not a neural network?**

A: For tabular data (rows and columns), gradient boosting typically outperforms neural networks unless you have millions of rows. LightGBM is also much faster to train, more interpretable (SHAP values are exact for tree models), and easier to tune.

**Q: What is SHAP and why does the API return it?**

A: SHAP (SHapley Additive exPlanations) measures how much each feature contributed to a specific prediction. This is required in regulated industries (like banking) where you must explain why a loan was rejected. "The model said REJECT because your debt-to-income ratio is 0.45 (which increases risk by +0.21 in log-odds)."

**Q: What's the difference between DVC and MLflow?**

A: They solve different problems. DVC versions **data and pipelines** (what data + code produced this model?). MLflow tracks **experiments** (what hyperparameters and metrics did each training run have?). They complement each other.

**Q: The model only gets AUC=0.76. Is that good?**

A: For credit default prediction with this dataset, 0.75-0.78 is typical. A naive model (predict no default for everyone) gets AUC=0.5. A Gini coefficient of 0.5 (=AUC of 0.75) is considered acceptable in credit scoring. The UCI dataset itself is cited with AUC around 0.77-0.79 in academic papers.

**Q: How do I deploy this to production?**

A: 1) Build the Docker image (`docker build -t loan-risk-api .`), 2) Push to a container registry, 3) Deploy to Kubernetes/ECS/Cloud Run. The Dockerfile is already set up for production (non-root user, multi-stage build, health check).

**Q: How do I add a new feature?**

A: 1) Add the column name to `src/loan_risk/features/definitions.py`, 2) Update the transformer in `transformers.py` if it needs a custom transform, 3) Update the Pandera schema in `data/schemas.py`, 4) Update the preprocessing script to produce the new column, 5) Re-run `dvc repro`.

**Q: Where do I see all the training runs?**

A: Start MLflow UI with `docker-compose up -d mlflow` then open http://localhost:5000. You'll see every run with its parameters, metrics, and artifacts. Without Docker, run `mlflow ui --backend-store-uri sqlite:///mlruns.db`.

---

---

## 20. Observability Tools — MLflow, Prefect, Prometheus, Grafana

This section is a hands-on guide for beginners. For each tool you will learn:
- **What the problem was** before this tool existed
- **What the tool does** in plain English
- **How to start it**
- **What URL to open**
- **What to actually look at** once it's open — screen by screen

All four tools are part of the `docker-compose` stack and are completely optional for
local development. They become essential once you want to track experiments, schedule
pipelines, or monitor a live service.

---

### Starting the full observability stack

```bash
docker-compose up -d
```

This starts all four tools in the background. Check they are all healthy:

```bash
docker-compose ps
```

All four services should show `running`. If one shows `exited`, check its logs:

```bash
docker-compose logs mlflow
docker-compose logs prefect
docker-compose logs prometheus
docker-compose logs grafana
```

To stop everything:

```bash
docker-compose down
```

---

## MLflow — Experiment Tracking & Model Registry

### The problem it solves

Without MLflow, every training run is a mystery:
- You run the pipeline Monday — AUC is 0.76
- You change a parameter Tuesday — AUC drops to 0.71
- You can't remember what you changed
- Three months later, "which model is in production?" — nobody knows

MLflow is a logbook. Every training run is automatically recorded with:
- Every parameter (`n_estimators`, `learning_rate`, etc.)
- Every metric (`val_auc`, `test_auc`, `gini`, `ks_stat`)
- Every artifact (the trained model file, SHAP plots, feature importance charts)
- When it ran, how long it took, and whether it succeeded

### Option A — Local (no Docker, recommended for development)

```bash
MLFLOW_TRACKING_URI=sqlite:///mlruns.db mlflow ui --port 5000
```

Open: **http://localhost:5000**

This reads from the same `mlruns.db` SQLite file that the training pipeline writes to.
No server setup required.

### Option B — Docker

```bash
docker-compose up -d mlflow
```

Open: **http://localhost:5000**

### When to open MLflow

- **After every training run** — verify the AUC was recorded correctly
- **When comparing hyperparameters** — which settings gave the best AUC?
- **When something breaks** — read the logged parameters and figure out what changed
- **Before deploying** — confirm the model version you want is in the registry

### What to do after opening http://localhost:5000

**Screen 1 — Experiments list (the home page)**

You will see a table with one row per experiment (ours is called `loan-risk`).
Click on `loan-risk` to enter it.

**Screen 2 — Runs list**

Each row is one training run. Columns show:
- `Start Time` — when it ran
- `val_auc`, `test_auc` — the key metrics to compare
- `Duration` — how long training took

What to do:
1. Click the `test_auc` column header to sort — highest AUC run is now at the top
2. Click any run ID (the blue link) to open the run detail page

**Screen 3 — Run detail page**

This has three tabs:

| Tab | What you see | What to look for |
|---|---|---|
| **Overview** | All logged parameters (learning_rate, n_estimators, etc.) | What settings produced this AUC? |
| **Metrics** | Charts of val_auc, test_auc, threshold over training | Is val close to test? (large gap = overfitting) |
| **Artifacts** | Model file, SHAP plots, feature importance | Click `feature_importance.png` to see which features matter most |

**Screen 4 — Comparing two runs**

1. Tick the checkbox on 2 or more runs in the runs list
2. Click the **Compare** button that appears above the table
3. You see a side-by-side diff of every parameter and metric

This is how you answer "Run A beat Run B — what was different?"

**Screen 5 — Model Registry**

Click **Models** in the top navigation bar.
You will see `loan-risk-classifier` listed.
Click it to see all registered versions, which one has the `@champion` alias, and when
each version was registered.

To manually promote a version here:
1. Click a version number (e.g. `Version 2`)
2. Click the pencil icon next to **Aliases**
3. Type `champion` and save

---

## Prefect — Pipeline Orchestration & Scheduling

### The problem it solves

Without Prefect, you run the pipeline manually:
```bash
uv run python scripts/run_pipeline.py --stage all
```

This is fine once. But real systems need:
- The pipeline to run **automatically every week** (new data arrives weekly)
- **Alerts** when it fails — not discovering it a week later
- **Visibility** into what ran, when, and how long each step took
- **Retry logic** — if the download fails due to a network blip, retry 3 times before alerting

Prefect handles all of this. It turns your Python functions into observable, schedulable,
retryable workflows.

### Start Prefect

```bash
docker-compose up -d prefect
```

Open: **http://localhost:4200**

### When to open Prefect

- **To schedule the training pipeline** — set it to run every Sunday at 2am automatically
- **To check if a scheduled run succeeded** — did last night's job work?
- **When a run failed** — Prefect shows exactly which task failed and why
- **To manually trigger a run** — without opening a terminal

### What to do after opening http://localhost:4200

**Screen 1 — Dashboard (home page)**

Shows a summary of recent flow runs:
- Green = succeeded
- Red = failed
- Yellow = running

If anything is red, click on it immediately.

**Screen 2 — Flows**

Click **Flows** in the left sidebar.
A *flow* is a named pipeline (e.g. `weekly-training-flow`, `daily-monitor-flow`).
Each flow can have multiple *deployments* (different schedules or configurations).

**Screen 3 — Flow Runs**

Click any flow to see its run history. Click a specific run to open the task graph:
- Each box is one task (download → preprocess → featurize → train → evaluate)
- Colour tells you: green = done, red = failed, grey = skipped
- Click a red task to see its full error traceback — no need to dig through terminal logs

**Screen 4 — Creating a schedule**

1. Click **Deployments** in the left sidebar
2. Find your deployment (e.g. `weekly-training-flow/production`)
3. Click **Edit**
4. Under **Schedule**, click **Add schedule**
5. Choose **Cron** and enter `0 2 * * 0` (every Sunday at 2am)
6. Click **Save**

The pipeline will now run automatically. You can also click **Quick Run** to trigger it
immediately without waiting for the schedule.

**Screen 5 — Work Pools**

Prefect uses *work pools* to decide where to run your code (local machine, Docker
container, Kubernetes pod, etc.). For local development, a `Process` work pool runs
flows directly on your machine. You need a worker running to actually execute flows:

```bash
# In a separate terminal — keeps the worker alive
uv run prefect worker start --pool default-agent-pool
```

---

## Prometheus — Metrics Collection

### The problem it solves

Your API is running. But:
- How many predictions per second is it handling?
- Are any requests taking longer than 2 seconds?
- Is the model returning 90% REJECT suddenly? (possible data drift)
- Did the server crash at 3am?

Prometheus answers these questions by **scraping** (pulling) metrics from your API
every 15 seconds and storing them in a time-series database.

Your FastAPI server already exposes metrics at `GET /metrics`:
```bash
curl http://localhost:8000/metrics
```

Prometheus visits that URL every 15 seconds and records every counter and histogram.

### Start Prometheus

```bash
docker-compose up -d prometheus
```

Open: **http://localhost:9090**

### When to open Prometheus

- **For ad-hoc queries** — "how many REJECT predictions in the last hour?"
- **To debug a Grafana panel** — Prometheus has a query editor to test expressions
- **To verify metrics are being scraped** — confirm the API is being monitored

### What to do after opening http://localhost:9090

**Screen 1 — Query page (home page)**

There is a text box at the top. This is where you write PromQL queries.

Try these queries one by one — type them in and press **Execute**:

```promql
# Total predictions made (all time)
loan_risk_predictions_total

# Predictions broken down by outcome (APPROVE vs REJECT)
sum by (prediction) (loan_risk_predictions_total)

# Request rate over the last 5 minutes (requests per second)
rate(loan_risk_predictions_total[5m])

# 99th percentile latency in the last 5 minutes (ms)
histogram_quantile(0.99, rate(loan_risk_prediction_latency_seconds_bucket[5m])) * 1000

# Is the model loaded? (1 = yes, 0 = no / degraded)
loan_risk_model_loaded
```

After clicking **Execute**, switch between:
- **Table** tab — raw numbers right now
- **Graph** tab — how the metric changed over time (drag the time range to zoom)

**Screen 2 — Targets (verifying the API is being scraped)**

Click **Status** → **Targets** in the top navigation.

You will see one row per scrape target. Look for `loan_risk_api` with State = `UP`.
If it shows `DOWN`, your API is not running or Prometheus cannot reach it.

**Screen 3 — Alerts**

Click **Alerts** in the top navigation.
Alert rules are defined in `infra/prometheus/rules.yml`. Green = no alerts firing.
Red = something needs your attention.

---

## Grafana — Dashboards & Visualisation

### The problem it solves

Prometheus stores metrics but its UI is designed for engineers writing queries.
Grafana turns those metrics into **visual dashboards** — charts, gauges, and alerts —
that anyone can understand at a glance.

Think of Prometheus as the database and Grafana as the BI tool on top of it.

### Start Grafana

```bash
docker-compose up -d grafana
```

Open: **http://localhost:3000**

Login: `admin` / `admin` (you will be prompted to change the password — you can skip this
for local development).

### When to open Grafana

- **For your daily/weekly health check** — open the dashboard and scan for red
- **When investigating an incident** — zoom into the time window when it happened
- **To share with non-engineers** — the dashboard is self-explanatory, no PromQL needed
- **To set up email/Slack alerts** — so you are notified before a problem becomes an outage

### What to do after opening http://localhost:3000

**Step 1 — Add Prometheus as a data source (first time only)**

1. Click the hamburger menu (☰) in the top-left
2. Go to **Connections** → **Data sources**
3. Click **Add data source**
4. Choose **Prometheus**
5. In the URL field enter: `http://prometheus:9090`
   (use `prometheus` not `localhost` — this is the Docker service name)
6. Click **Save & Test** — you should see "Data source is working"

**Step 2 — Import the pre-built dashboard**

A dashboard JSON is included at `infra/grafana/dashboards/loan_risk.json`:

1. Click the hamburger menu (☰)
2. Go to **Dashboards** → **Import**
3. Click **Upload dashboard JSON file**
4. Select `infra/grafana/dashboards/loan_risk.json`
5. Choose your Prometheus data source
6. Click **Import**

**Step 3 — Reading the dashboard**

The dashboard has four rows:

| Row | Panels | What to check |
|---|---|---|
| **Traffic** | Requests/sec, total predictions | Is the API receiving traffic? |
| **Latency** | p50 / p95 / p99 response times | Is anything slower than 200ms? |
| **Model health** | APPROVE vs REJECT rate, probability distribution | Is the rejection rate stable? A sudden spike may mean drift |
| **System** | Model loaded status, uptime | Is the model currently loaded? |

**Step 4 — Setting a time range**

The default view shows the last 1 hour. Change it with the time picker in the top-right
corner (e.g. "Last 24 hours", "Last 7 days", or a custom range like "last Sunday 9am–11am").

**Step 5 — Setting up alerts (optional)**

1. Click any panel title → **Edit**
2. Click the **Alert** tab on the left
3. Click **Create alert rule from this panel**
4. Set a condition, e.g.: "if `reject_rate` > 0.70 for 10 minutes, send an alert"
5. Add a notification channel (email, Slack, PagerDuty) under **Alerting** → **Contact points**

---

### Quick reference — all four tools

| Tool | URL | Start command | Open when... |
|---|---|---|---|
| **MLflow** | http://localhost:5000 | `mlflow ui --backend-store-uri sqlite:///mlruns.db` (or Docker) | After training; to compare runs; to manage the model registry |
| **Prefect** | http://localhost:4200 | `docker-compose up -d prefect` | To schedule pipelines; to check if a scheduled run succeeded; to debug a failed task |
| **Prometheus** | http://localhost:9090 | `docker-compose up -d prometheus` | To query raw metrics; to verify the API is being scraped; to test PromQL expressions |
| **Grafana** | http://localhost:3000 | `docker-compose up -d grafana` | For daily health checks; to investigate incidents; to share dashboards with the team |

---

## License

MIT License — see [LICENSE](LICENSE) file. The UCI dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgements

- **Dataset**: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications.
- **OpenML**: Vanschoren, J., et al. (2014). OpenML: Networked science in machine learning. SIGKDD Explorations.
