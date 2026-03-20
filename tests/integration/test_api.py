"""Integration tests for the FastAPI prediction endpoint.

These tests use httpx's AsyncClient to hit the app without starting
a real server. The model predictor is mocked so tests don't need
MLflow infrastructure.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from loan_risk.serving.app import create_app
from loan_risk.serving.schemas import (
    PredictionResponse,
)


SAMPLE_REQUEST = {
    "loan_amount": 15000,
    "annual_income": 55000,
    "employment_years": 3,
    "credit_score": 680,
    "debt_to_income_ratio": 0.35,
    "num_open_accounts": 4,
    "num_delinquencies": 0,
    "loan_purpose": "debt_consolidation",
    "home_ownership": "RENT",
    "loan_term_months": 36,
}


@pytest.fixture
def mock_predictor():
    """Mock predictor that returns a canned prediction without needing MLflow."""
    predictor = MagicMock()
    predictor.is_ready = True
    predictor._model_version = "1"
    predictor._feature_names = ["feature_1", "feature_2"]
    predictor.predict.return_value = PredictionResponse(
        prediction="REJECT",
        default_probability=0.73,
        confidence="HIGH",
        risk_tier="HIGH_RISK",
        top_factors=[],
        model_version="1",
        request_id="req_test",
        latency_ms=5.0,
    )
    return predictor


@pytest.fixture
def client(mock_predictor):
    """Test client with mocked predictor using FastAPI dependency overrides."""
    from loan_risk.serving.predictor import get_predictor

    app = create_app()
    # FastAPI dependency override: replace get_predictor with a function
    # that always returns our mock — this is the correct way to mock Depends()
    app.dependency_overrides[get_predictor] = lambda: mock_predictor
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


def test_predict_returns_200(client) -> None:
    response = client.post("/predict", json=SAMPLE_REQUEST)
    assert response.status_code == 200


def test_predict_response_schema(client) -> None:
    response = client.post("/predict", json=SAMPLE_REQUEST)
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ("APPROVE", "REJECT")
    assert "default_probability" in data
    assert 0.0 <= data["default_probability"] <= 1.0


def test_predict_invalid_credit_score(client) -> None:
    bad_request = {**SAMPLE_REQUEST, "credit_score": 200}
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422


def test_predict_invalid_loan_purpose(client) -> None:
    bad_request = {**SAMPLE_REQUEST, "loan_purpose": "crypto"}
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422


def test_health_returns_200(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True


def test_metrics_returns_prometheus_format(client) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"loan_risk" in response.content or b"# HELP" in response.content


def test_model_info_returns_metadata(client) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "model_name" in data
