"""Tests for the healthcare claims prediction API."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from src.app.main import app, risk_guidance


class FakeScaler:
    def transform(self, features):
        return features


class FakeModel:
    def predict_proba(self, features):
        return np.array([[0.18, 0.82]])


def test_health_reports_ready_with_loaded_artifacts() -> None:
    with TestClient(app) as client:
        app.state.model = FakeModel()
        app.state.scaler = FakeScaler()
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model": "loaded"}


def test_predict_returns_high_risk_guidance() -> None:
    with TestClient(app) as client:
        app.state.model = FakeModel()
        app.state.scaler = FakeScaler()
        response = client.post(
            "/predict",
            json={
                "charge_amount": 1250.0,
                "code_count": 2,
                "has_modifier_25": 1,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "denial_probability": 0.82,
        "risk_level": "High",
        "recommended_action": (
            "Route to human billing expert for clinical documentation review."
        ),
    }


def test_predict_validates_claim_input() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"charge_amount": 0, "code_count": 0, "has_modifier_25": 3},
        )

    assert response.status_code == 422


def test_risk_thresholds() -> None:
    assert risk_guidance(0.76)[0] == "High"
    assert risk_guidance(0.75)[0] == "Medium"
    assert risk_guidance(0.40)[0] == "Low"
