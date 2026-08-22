"""FastAPI service for healthcare claim denial-risk predictions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/xgboost_denial_v1.joblib")
SCALER_PATH = Path("models/scaler_v1.joblib")
FEATURE_COLUMNS = ["charge_amount", "code_count", "has_modifier_25"]


class ClaimInput(BaseModel):
    """Validated claim features accepted by the scoring endpoint."""

    charge_amount: float = Field(
        ..., gt=0, description="Total billed amount for the claim"
    )
    code_count: int = Field(
        ..., ge=1, description="Number of distinct procedure codes"
    )
    has_modifier_25: int = Field(
        ..., ge=0, le=1, description="Modifier 25 usage flag (0 or 1)"
    )


class ClaimPredictionResponse(BaseModel):
    """Denial probability and corresponding operational guidance."""

    denial_probability: float
    risk_level: str
    recommended_action: str


def load_model_artifacts() -> tuple[Any, Any]:
    """Load the trained classifier and scaler from the models directory."""

    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once when the API starts."""

    try:
        app.state.model, app.state.scaler = load_model_artifacts()
        app.state.model_error = None
    except Exception as exc:
        app.state.model = None
        app.state.scaler = None
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="Healthcare Claims Denial Prediction API",
    version="1.0.0",
    description="Real-time scoring endpoint for medical billing claims risk assessment.",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Report whether the service and trained artifacts are ready."""

    ready = request.app.state.model is not None
    return {
        "status": "ready" if ready else "degraded",
        "model": "loaded" if ready else "unavailable",
    }


def risk_guidance(probability: float) -> tuple[str, str]:
    """Map a probability to the operational tier specified in the brief."""

    if probability > 0.75:
        return (
            "High",
            "Route to human billing expert for clinical documentation review.",
        )
    if probability > 0.40:
        return (
            "Medium",
            "Verify procedure codes and modifier justification before submission.",
        )
    return "Low", "Auto-approve for clearinghouse submission."


@app.post("/predict", response_model=ClaimPredictionResponse)
def predict_denial(claim: ClaimInput, request: Request) -> ClaimPredictionResponse:
    """Score one claim and return its denial-risk recommendation."""

    model = request.app.state.model
    scaler = request.app.state.scaler
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are unavailable. Run the training pipeline first.",
        )

    try:
        features = pd.DataFrame(
            [[claim.charge_amount, claim.code_count, claim.has_modifier_25]],
            columns=FEATURE_COLUMNS,
        )
        scaled_features = scaler.transform(features)
        probability = float(model.predict_proba(scaled_features)[:, 1][0])
        risk_level, action = risk_guidance(probability)
        return ClaimPredictionResponse(
            denial_probability=round(probability, 4),
            risk_level=risk_level,
            recommended_action=action,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
