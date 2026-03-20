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
uv run ruff check src/ scripts/ pipelines/
uv run ruff check --fix src/ scripts/ pipelines/
uv run ruff format src/

# Full DVC pipeline (download → preprocess → featurize → train)
dvc repro

# Run individual pipeline stages
uv run python scripts/run_pipeline.py --stage features
uv run python scripts/run_pipeline.py --stage train --skip-tuning
uv run python scripts/run_pipeline.py --stage evaluate
uv run python scripts/run_pipeline.py --stage all --skip-tuning

# Start the prediction API
uv run uvicorn loan_risk.serving.app:create_app --factory --port 8000 --reload

# MLflow UI (no Docker)
MLFLOW_TRACKING_URI=sqlite:///mlruns.db mlflow ui --port 5000

# Full stack (MLflow + Prefect + Prometheus + Grafana)
docker-compose up -d
```

## Environment

- All Python commands must be prefixed with `uv run` — the package is installed in `.venv/` managed by uv.
- Set `MLFLOW_TRACKING_URI=sqlite:///mlruns.db` to use a local SQLite file instead of a running MLflow server. CI always uses this.
- `PYTHONPATH=src` is required when running scripts directly without `uv run`. With `uv run` it is handled automatically.
- Config loads in this order: `config/settings.yaml` → `.env` → environment variables. Use `__` as the nested delimiter (e.g. `MLFLOW__TRACKING_URI=...`).
- `get_settings()` in `src/loan_risk/config.py` is an `@lru_cache` singleton. If you mutate settings in tests, call `get_settings.cache_clear()` after.

## Architecture

### Data flow

```
UCI OpenML (30K rows)
  → scripts/download_dataset.py     → data/raw/credit_default_raw.csv
  → scripts/preprocess_dataset.py   → data/raw/loans.parquet       (Pandera-validated)
  → run_pipeline.py --stage features → data/processed/{train,val,test}.parquet
                                       artifacts/preprocessor.pkl
  → run_pipeline.py --stage train   → MLflow run + model registry (champion alias)
  → run_pipeline.py --stage evaluate → reports/evaluation/metrics.json
  → loan_risk.serving.app           → POST /predict (loads champion via MLflow registry)
```

DVC (`dvc.yaml`) wraps these four stages (download / preprocess / featurize / train) and tracks which inputs changed to avoid re-running unchanged stages.

### Package layout (`src/loan_risk/`)

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic-settings `Settings` singleton. Single source of truth for all config values. |
| `data/schemas.py` | Pandera `RawLoanSchema` and `InferenceInputSchema`. The schema is the contract between preprocessing and the rest of the pipeline. |
| `data/splits.py` | Stratified 70/10/20 train/val/test split preserving the ~22% default rate. |
| `features/definitions.py` | Constants: `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, `ENGINEERED_FEATURES`, `CREDIT_SCORE_BINS`. **Start here when adding/removing features.** |
| `features/transformers.py` | Custom sklearn `BaseEstimator` / `TransformerMixin` classes. All are stateless or fit on training data only. |
| `features/pipeline.py` | `build_feature_pipeline()` assembles transformers into a single sklearn `Pipeline`. `prepare_features()` calls fit or transform. Saved as `artifacts/preprocessor.pkl`. |
| `training/trainer.py` | `ModelTrainer.fit()` — fits the feature pipeline on training splits, trains LightGBM/XGBoost, calibrates the classification threshold (F-beta β=2), logs everything to MLflow. Returns a `TrainingResult` dataclass. |
| `tuning/` | Optuna objectives for LightGBM and XGBoost. `run_hyperparameter_search()` saves best params to `artifacts/best_params/`. |
| `evaluation/` | `compute_classification_metrics()` (AUC, Gini, KS), SHAP `TreeExplainer`, bias audit, `EvaluationReport` dataclass. |
| `registry/client.py` | `MLflowRegistryClient` — wraps `MlflowClient` to promote/archive models and load the `@champion` alias. `get_champion_model()` is called at API startup. |
| `serving/predictor.py` | `ModelPredictor` singleton (`get_predictor()`). Loaded once at startup via the FastAPI lifespan. **Never call `predictor.load()` inside request handlers.** |
| `serving/app.py` | `create_app()` factory — registers routes, middleware, and the lifespan (model warm-up). Also exposes module-level `app = create_app()` for direct uvicorn invocation. |
| `monitoring/drift.py` | Evidently drift reports + PSI per feature. PSI > 0.15 triggers a retrain alert. |

### Key design decisions

**Polars vs pandas**: The pipeline ingests and stores data in Polars (`pl.DataFrame`), but sklearn transformers work on pandas. `prepare_features()` in `features/pipeline.py` converts with `.to_pandas()` before calling the sklearn pipeline.

**MLflow without a server**: Set `MLFLOW_TRACKING_URI=sqlite:///mlruns.db`. The SQLite backend supports both tracking and the model registry. This is what CI uses.

**FastAPI dependency injection and mocking**: Routes use `Depends(get_predictor)`. To mock in tests, use `app.dependency_overrides[get_predictor] = lambda: mock`. Do **not** use `unittest.mock.patch` on the route — `Depends` captures the function reference at decoration time, so the patch has no effect on route handlers. The app's lifespan calls `get_predictor()` directly (bypasses DI), so also patch `loan_risk.serving.app.get_predictor` if you need to prevent real model loading.

**Structlog**: Uses `structlog.stdlib.LoggerFactory()` (not `PrintLoggerFactory`). All logs are JSON in production (`json_output=True`). In tests, logs appear in captured stdout.

**DVC metrics**: `reports/evaluation/metrics.json` is the DVC-tracked metrics file (written by `run_pipeline.py --stage evaluate`). `dvc metrics show` and the GitHub Actions job summary both read from it.

### Testing conventions

- **Unit tests** (`tests/unit/`) use the `sample_df` fixture (200-row Polars DataFrame, defined in `tests/conftest.py`). They need no external services.
- **Integration tests** (`tests/integration/test_api.py`) use a `mock_predictor` fixture and `app.dependency_overrides`. They do not require a trained model.
- pytest is configured with `asyncio_mode = "auto"` and coverage on `src/loan_risk/`.
- `N803`/`N806` ruff rules are suppressed globally — `X`, `X_train`, `X_val` are accepted ML conventions.

### GitHub Actions

- `ci.yml` — runs on every push: ruff + 30 unit tests (~35s).
- `serve.yml` — runs when `src/loan_risk/serving/` or `tests/integration/test_api.py` change: 7 API integration tests (~45s).
- `training.yml` — manual dispatch + weekly cron: full `dvc repro`, publishes metrics to job summary, uploads `reports/` as artifact.

All workflows use `MLFLOW_TRACKING_URI: "sqlite:///mlruns.db"` and `PYTHONPATH: "src"`.
