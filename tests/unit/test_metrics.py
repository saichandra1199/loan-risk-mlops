"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from loan_risk.evaluation.metrics import (
    compute_classification_metrics,
    compute_gini,
    compute_ks_statistic,
)


@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
    return y_true, y_prob


@pytest.fixture
def random_predictions() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], 200, p=[0.85, 0.15])
    y_prob = rng.uniform(0, 1, 200)
    return y_true, y_prob


def test_perfect_auc(perfect_predictions) -> None:
    y_true, y_prob = perfect_predictions
    metrics = compute_classification_metrics(y_true, y_prob)
    assert metrics["auc_roc"] == pytest.approx(1.0, abs=1e-6)


def test_random_auc_near_half(random_predictions) -> None:
    y_true, y_prob = random_predictions
    metrics = compute_classification_metrics(y_true, y_prob)
    # Random classifier should be near 0.5
    assert 0.3 < metrics["auc_roc"] < 0.7


def test_gini_from_auc(perfect_predictions) -> None:
    y_true, y_prob = perfect_predictions
    gini = compute_gini(y_true, y_prob)
    assert gini == pytest.approx(1.0, abs=1e-6)


def test_ks_statistic(perfect_predictions) -> None:
    y_true, y_prob = perfect_predictions
    ks = compute_ks_statistic(y_true, y_prob)
    assert ks > 0.5


def test_all_metrics_returned(perfect_predictions) -> None:
    y_true, y_prob = perfect_predictions
    metrics = compute_classification_metrics(y_true, y_prob)
    expected_keys = {"auc_roc", "gini", "ks_statistic", "f1", "precision", "recall", "accuracy"}
    assert expected_keys.issubset(set(metrics.keys()))


def test_metrics_with_custom_threshold(perfect_predictions) -> None:
    y_true, y_prob = perfect_predictions
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert "f1" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0
