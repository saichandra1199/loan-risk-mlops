"""Model predictor singleton: loads the champion model once, serves predictions.

The predictor handles:
- Lazy model loading (first request or explicit warm-up)
- Feature pipeline application
- SHAP value computation for real-time explainability
- Prometheus metric tracking
"""

from __future__ import annotations

import datetime
import time
from typing import Any

import polars as pl
import shap
from prometheus_client import Counter, Histogram

from loan_risk.config import get_settings
from loan_risk.evaluation.explainability import get_top_shap_factors
from loan_risk.exceptions import PredictionError
from loan_risk.features.pipeline import build_feature_pipeline, load_pipeline
from loan_risk.logging_setup import get_logger
from loan_risk.monitoring.performance import log_prediction
from loan_risk.registry.client import MLflowRegistryClient
from loan_risk.serving.schemas import (
    LoanApplicationRequest,
    PredictionResponse,
    RiskFactor,
    compute_confidence,
    compute_risk_tier,
)

logger = get_logger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "loan_risk_predictions_total",
    "Total predictions served",
    ["prediction", "risk_tier"],
)
PREDICTION_LATENCY = Histogram(
    "loan_risk_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
DEFAULT_PROBABILITY = Histogram(
    "loan_risk_default_probability",
    "Distribution of predicted default probabilities",
    buckets=[0.1 * i for i in range(11)],
)


class ModelPredictor:
    """Singleton predictor that wraps the champion model."""

    def __init__(self) -> None:
        self.cfg = get_settings()
        self._model: Any = None
        self._feature_pipeline = None
        self._feature_names: list[str] = []
        self._explainer: shap.TreeExplainer | None = None
        self._model_version: str = "unknown"
        self._load_time: float = 0.0

    def load(self) -> None:
        """Load the champion model and feature pipeline from registry/disk."""
        logger.info("loading_model", alias=self.cfg.serving.model_alias)

        registry = MLflowRegistryClient()
        self._model, self._model_version = registry.get_champion_model()

        # Try loading persisted pipeline; fall back to building a new one
        pipeline_path = "artifacts/preprocessor.pkl"
        try:
            self._feature_pipeline = load_pipeline(pipeline_path)
        except Exception:
            logger.warning("pipeline_not_found_building_new", path=pipeline_path)
            self._feature_pipeline = build_feature_pipeline()

        # Build SHAP TreeExplainer
        try:
            self._explainer = shap.TreeExplainer(self._model)
        except Exception as exc:
            logger.warning("shap_explainer_failed", error=str(exc))
            self._explainer = None

        self._load_time = time.time()
        logger.info("model_loaded", version=self._model_version)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(
        self,
        request: LoanApplicationRequest,
        request_id: str = "unknown",
    ) -> PredictionResponse:
        """Run a single prediction with SHAP explanation.

        Args:
            request: Validated loan application data.
            request_id: Request ID for tracing.

        Returns:
            PredictionResponse with probability, decision, and explanation.
        """
        if not self.is_ready:
            raise PredictionError("Model not loaded. Call predictor.load() first.")

        start = time.time()

        try:
            # Convert request to DataFrame
            row_dict = request.model_dump()
            df = pl.DataFrame([row_dict])
            pdf = df.to_pandas()

            # Apply feature pipeline
            if self._feature_pipeline is not None:
                try:
                    X = self._feature_pipeline.transform(pdf)
                except Exception:
                    # Pipeline not fitted — apply transform directly
                    from loan_risk.features.pipeline import prepare_features
                    X = prepare_features(df, self._feature_pipeline, fit=False)
            else:
                X = pdf.values

            # Predict
            prob = float(self._model.predict_proba(X)[0, 1])
            threshold = getattr(self._model, "_threshold", 0.5)

            decision = "REJECT" if prob >= threshold else "APPROVE"
            confidence = compute_confidence(prob)
            risk_tier = compute_risk_tier(prob)

            # SHAP explanation
            top_factors: list[RiskFactor] = []
            if self._explainer is not None and len(self._feature_names) > 0:
                try:
                    shap_vals = self._explainer.shap_values(X)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]
                    factors = get_top_shap_factors(shap_vals[0], self._feature_names, top_n=5)
                    top_factors = [RiskFactor(**f) for f in factors]
                except Exception as exc:
                    logger.warning("shap_failed", error=str(exc))

            latency_ms = (time.time() - start) * 1000

            # Prometheus
            PREDICTION_COUNTER.labels(prediction=decision, risk_tier=risk_tier).inc()
            PREDICTION_LATENCY.observe(latency_ms / 1000)
            DEFAULT_PROBABILITY.observe(prob)

            # Log prediction for live AUC monitoring (best-effort)
            try:
                log_prediction(
                    loan_id=request_id,
                    default_probability=prob,
                    model_version=self._model_version,
                    request_id=request_id,
                    timestamp=datetime.datetime.utcnow().isoformat(),
                )
            except Exception:
                pass

            return PredictionResponse(
                prediction=decision,
                default_probability=round(prob, 4),
                confidence=confidence,
                risk_tier=risk_tier,
                top_factors=top_factors,
                model_version=self._model_version,
                request_id=request_id,
                latency_ms=round(latency_ms, 2),
            )

        except PredictionError:
            raise
        except Exception as exc:
            logger.error("prediction_error", request_id=request_id, error=str(exc))
            raise PredictionError(f"Prediction failed: {exc}") from exc


# Module-level singleton
_predictor: ModelPredictor | None = None


def get_predictor() -> ModelPredictor:
    """Return (or create) the module-level predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = ModelPredictor()
    return _predictor
