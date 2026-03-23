#!/usr/bin/env bash
set -euo pipefail

export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
PORT=8000

echo "=== 1. Check @champion alias ==="
uv run python - <<'EOF'
from mlflow import MlflowClient
mc = MlflowClient()
try:
    v = mc.get_model_version_by_alias("loan-risk-classifier", "champion")
    print(f"Champion: version {v.version} (run {v.run_id[:8]})")
except Exception:
    raise SystemExit(
        "No @champion model found. Run run.sh first, or promote a version manually:\n"
        "  MLFLOW_TRACKING_URI=sqlite:///mlruns.db uv run python -c \"\n"
        "  from mlflow import MlflowClient; mc = MlflowClient()\n"
        "  mc.set_registered_model_alias('loan-risk-classifier', 'champion', '<version>')\""
    )
EOF

echo "=== 2. Start API server ==="
uv run uvicorn loan_risk.serving.app:create_app --factory --port "$PORT" &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "=== 3. Wait for server to be ready ==="
for i in $(seq 1 20); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    if [ "$i" -eq 20 ]; then
        echo "ERROR: Server did not start within 20s"
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo "=== 4. Health check ==="
curl -s "http://localhost:${PORT}/health" | python3 -m json.tool

echo ""
echo "=== 5. Predict ==="
curl -s -X POST "http://localhost:${PORT}/predict" \
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
  }' | python3 -m json.tool

echo ""
echo "Server is still running on port $PORT (PID $SERVER_PID)."
echo "Stop it with:  kill $SERVER_PID"
