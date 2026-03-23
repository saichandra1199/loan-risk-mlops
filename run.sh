#!/usr/bin/env bash
set -euo pipefail

export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

echo "=== 1. Install dependencies ==="
uv sync --extra dev

echo "=== 2. Verify install ==="
uv run python -c "import loan_risk; print('loan_risk installed OK')"

echo "=== 3. Lint ==="
uv run ruff check src/ scripts/ pipelines/

echo "=== 4. Unit tests ==="
uv run pytest tests/unit/ -v

echo "=== 5. Download dataset ==="
uv run python scripts/download_dataset.py --output data/raw/credit_default_raw.csv

echo "=== 6. Preprocess ==="
uv run python scripts/preprocess_dataset.py

echo "=== 7. Featurize + Train + Evaluate ==="
uv run python scripts/run_pipeline.py --stage all --skip-tuning

echo "=== 8. Promote best model to @champion ==="
uv run python - <<'EOF'
from mlflow import MlflowClient
mc = MlflowClient()
versions = mc.search_model_versions("name='loan-risk-classifier'")
if not versions:
    raise SystemExit("No registered model versions found. Did training succeed?")
latest = max(versions, key=lambda v: int(v.version))
mc.set_registered_model_alias("loan-risk-classifier", "champion", latest.version)
print(f"Promoted version {latest.version} (run {latest.run_id[:8]}) to @champion")
EOF

echo "=== Done — model is ready for serving ==="
