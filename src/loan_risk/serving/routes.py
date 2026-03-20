"""FastAPI route definitions for the loan risk prediction API."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from loan_risk.exceptions import ModelNotFoundError, PredictionError
from loan_risk.logging_setup import get_logger
from loan_risk.serving.predictor import ModelPredictor, get_predictor
from loan_risk.serving.schemas import (
    HealthResponse,
    LoanApplicationRequest,
    ModelInfoResponse,
    PredictionResponse,
)

logger = get_logger(__name__)
router = APIRouter()

# Track server start time for uptime metric
_START_TIME = time.time()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict loan default risk",
    tags=["Prediction"],
)
async def predict(
    request: LoanApplicationRequest,
    predictor: ModelPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Predict default probability for a loan application.

    Returns the prediction (APPROVE/REJECT), default probability,
    confidence level, risk tier, and top SHAP explanatory factors.
    """
    request_id = f"req_{uuid.uuid4().hex[:16]}"

    try:
        response = predictor.predict(request, request_id=request_id)
        return response
    except ModelNotFoundError as exc:
        logger.error("model_not_found", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except PredictionError as exc:
        logger.error("prediction_error", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness check",
    tags=["Operations"],
)
async def health(predictor: ModelPredictor = Depends(get_predictor)) -> HealthResponse:
    """Return service health including model load status."""
    uptime = time.time() - _START_TIME

    return HealthResponse(
        status="healthy" if predictor.is_ready else "degraded",
        model_loaded=predictor.is_ready,
        model_version=predictor._model_version if predictor.is_ready else None,
        uptime_seconds=round(uptime, 1),
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    tags=["Operations"],
    response_class=None,
)
async def metrics():
    """Expose Prometheus metrics for scraping."""
    from fastapi.responses import Response as FastAPIResponse
    return FastAPIResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Current model version metadata",
    tags=["Operations"],
)
async def model_info(predictor: ModelPredictor = Depends(get_predictor)) -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    return ModelInfoResponse(
        model_name="loan-risk-classifier",
        model_version=predictor._model_version,
        model_alias="champion",
        training_date=None,
        validation_auc=None,
        feature_count=len(predictor._feature_names) if predictor._feature_names else None,
    )
