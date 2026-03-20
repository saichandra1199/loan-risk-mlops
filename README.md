# 🏦 Loan Risk MLOps Pipeline

[![CI — Lint & Tests](https://github.com/YOUR_USERNAME/loan-risk-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/loan-risk-mlops/actions/workflows/ci.yml)
[![Training Pipeline](https://github.com/YOUR_USERNAME/loan-risk-mlops/actions/workflows/training.yml/badge.svg)](https://github.com/YOUR_USERNAME/loan-risk-mlops/actions/workflows/training.yml)
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
10. [Pipeline Stages — What Happens at Each Step](#10-pipeline-stages--what-happens-at-each-step)
11. [API Reference](#11-api-reference)
12. [Configuration Guide](#12-configuration-guide)
13. [Running Experiments](#13-running-experiments)
14. [Troubleshooting](#14-troubleshooting)
15. [FAQ for Beginners](#15-faq-for-beginners)

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
git clone https://github.com/YOUR_USERNAME/loan-risk-mlops.git
cd loan-risk-mlops
```

### Step 2: Install uv (fast Python package manager)

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 3: Install all dependencies

```bash
uv sync --extra dev
```

This installs all 40+ packages in ~30 seconds. `uv` reads `pyproject.toml` and
creates a `.venv/` directory with everything you need.

**Expected output:**
```
Resolved 87 packages in 2.34s
Installed 87 packages in 18.45s
```

### Step 4: Configure environment

```bash
cp .env.example .env
```

The defaults in `.env.example` work for local development without any changes.
Edit `.env` only if you want to use a real MLflow server or different paths.

### Step 5: Verify the installation

```bash
uv run python -c "import loan_risk; print(f'loan_risk v{loan_risk.__version__} installed OK')"
```

**Expected output:** `loan_risk v0.1.0 installed OK`

### Step 6: Run unit tests (no data or server needed)

```bash
uv run pytest tests/unit/ -v
```

**Expected output:** All 25+ tests should pass.

### Step 7: Download the dataset

```bash
uv run python scripts/download_dataset.py
```

This downloads 30,000 rows from the UCI/OpenML repository (~2MB).
The file is saved to `data/raw/credit_default_raw.csv`.

**Expected output:**
```
Downloading UCI Credit Card Default dataset via OpenML...
  Source: https://www.openml.org/d/350
  Downloaded 30,000 rows × 25 columns
  Saved to: data/raw/credit_default_raw.csv
  Default rate: 22.1%
Download complete.
```

### Step 8: Preprocess the data

```bash
uv run python scripts/preprocess_dataset.py
```

This maps UCI columns to our loan schema and validates it with Pandera.

**Expected output:**
```
Loading raw data from: data/raw/credit_default_raw.csv
  Loaded 30,000 rows × 25 columns

Applying feature transformations...
  loan_id: LOAN_XXXXXXX format
  loan_amount: from 'limit_bal' → clipped to [100, 100k]
  ...
  loan_default: default rate = 22.1%

Validating against Pandera schema...
  Schema validation PASSED: 30,000 rows
Saved preprocessed data to: data/raw/loans.parquet
```

### Step 9: Run the full training pipeline

```bash
uv run python scripts/run_pipeline.py --stage all --skip-tuning
```

This runs: validate → feature engineering → train LightGBM → evaluate → register model.

**Expected output (abbreviated):**
```
[validate] Schema validation passed for 30,000 rows
[features] train=21,000 val=3,000 test=6,000
[train] run_id=a3f8b2c1...
[train] val_auc=0.7682  test_auc=0.7591
[evaluate] Report saved to reports/evaluation/report_a3f8b2c1_2024-01-15.json
[evaluate] Model promoted to champion. Version: 1
```

### Step 10: Start the prediction API

```bash
uv run uvicorn loan_risk.serving.app:create_app --factory --port 8000 --reload
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

## 10. Pipeline Stages — What Happens at Each Step

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

## 11. API Reference

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

## 12. Configuration Guide

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

## 13. Running Experiments

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

## 14. Troubleshooting

### "No module named 'loan_risk'"

The `src/` directory needs to be in Python's path:
```bash
PYTHONPATH=src uv run python scripts/run_pipeline.py --stage train
# Or use the uv run prefix which handles this automatically
uv run python scripts/run_pipeline.py --stage train
```

### "MLflow connection refused"

The training pipeline works WITHOUT a running MLflow server.
Set this environment variable to use a local SQLite file instead:
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
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

### "Model not found in registry"

The champion model hasn't been trained yet. Run the full pipeline first:
```bash
uv run python scripts/run_pipeline.py --stage all --skip-tuning
```

---

## 15. FAQ for Beginners

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

## License

MIT License — see [LICENSE](LICENSE) file. The UCI dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgements

- **Dataset**: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications.
- **OpenML**: Vanschoren, J., et al. (2014). OpenML: Networked science in machine learning. SIGKDD Explorations.
