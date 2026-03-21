# Multi-stage build for the loan risk inference service
# Deployed to ECS Fargate via ECR (see ecs/task-definition.json and infra/terraform/).
# Runtime deps include boto3 for CloudWatch metrics and S3 prediction logging.
# Stage 1: dependency builder
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .python-version ./
COPY src/ ./src/

# Install only runtime deps (no dev extras)
RUN uv sync --no-dev --no-editable

# Stage 2: runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY .env.example ./.env.example

# Create directories that the app writes to at runtime
RUN mkdir -p artifacts reports/evaluation reports/monitoring data/monitoring

# Set PATH to use .venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "loan_risk.serving.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
