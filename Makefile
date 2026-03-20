.PHONY: install test lint format train serve docker-up docker-down clean help

PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
MYPY := uv run mypy

help:
	@echo "Loan Risk MLOps Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  install     Install all dependencies (uv sync --extra dev)"
	@echo "  test        Run test suite"
	@echo "  lint        Run ruff + mypy"
	@echo "  format      Auto-format with ruff"
	@echo "  train       Run full training pipeline"
	@echo "  serve       Start inference API on port 8000"
	@echo "  docker-up   Start MLflow + Prefect + Prometheus + Grafana"
	@echo "  docker-down Stop all containers"
	@echo "  clean       Remove cache and local artifacts"

install:
	uv sync --extra dev

test:
	$(PYTEST) tests/ -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v

test-cov:
	$(PYTEST) tests/ --cov-fail-under=80

lint:
	$(RUFF) check src/
	$(MYPY) src/

format:
	$(RUFF) format src/
	$(RUFF) check --fix src/

train:
	$(PYTHON) scripts/run_pipeline.py --stage all

data:
	$(PYTHON) scripts/generate_sample_data.py --n-rows 50000

serve:
	uv run uvicorn loan_risk.serving.app:create_app --factory --port 8000 --reload

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf mlruns/ mlartifacts/ .coverage htmlcov/ 2>/dev/null || true

# DVC targets
dvc-init:
	dvc init

dvc-repro:
	$(PYTHON) -m dvc repro

dvc-dag:
	$(PYTHON) -m dvc dag

dvc-status:
	$(PYTHON) -m dvc status

dvc-metrics:
	$(PYTHON) -m dvc metrics show

dvc-params:
	$(PYTHON) -m dvc params diff

# Data targets
download:
	$(PYTHON) scripts/download_dataset.py

preprocess:
	$(PYTHON) scripts/preprocess_dataset.py

# Git setup helper
git-setup:
	git init -b main
	git add .
	git commit -m "Initial commit: Loan Risk MLOps Pipeline"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Create a GitHub repo: gh repo create loan-risk-mlops --public"
	@echo "  2. Push: git remote add origin https://github.com/USERNAME/loan-risk-mlops.git && git push -u origin main"
