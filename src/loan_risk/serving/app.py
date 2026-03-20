"""FastAPI application factory.

Usage:
    # Direct run
    uv run uvicorn loan_risk.serving.app:create_app --factory --port 8000

    # Or import the app directly
    from loan_risk.serving.app import create_app
    app = create_app()
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from loan_risk.config import get_settings
from loan_risk.logging_setup import configure_logging, get_logger
from loan_risk.serving.middleware import RequestLoggingMiddleware
from loan_risk.serving.predictor import get_predictor
from loan_risk.serving.routes import router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: warm up model on startup, clean up on shutdown."""
    cfg = get_settings()

    configure_logging(json_output=True)
    logger.info("startup", port=cfg.serving.port, model_alias=cfg.serving.model_alias)

    predictor = get_predictor()
    try:
        predictor.load()
        logger.info("model_warmup_complete")
    except Exception as exc:
        # Don't crash on startup — health endpoint will report degraded
        logger.warning("model_warmup_failed", error=str(exc))

    yield  # <- app runs here

    logger.info("shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance with all routes and middleware.
    """
    app = FastAPI(
        title="Loan Risk Prediction API",
        description=(
            "Binary classifier for loan default risk. "
            "Predicts whether an applicant will default (REJECT) or repay (APPROVE)."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Middleware (executed in reverse order of registration)
    app.add_middleware(RequestLoggingMiddleware)

    # Routes
    app.include_router(router)

    return app


# Allow `uvicorn loan_risk.serving.app:app` as an alternative invocation
app = create_app()
