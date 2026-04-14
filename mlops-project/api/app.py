"""FastAPI application for serving Production model predictions."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.utils import (
    configure_logging,
    get_feature_template,
    load_environment,
    to_dataframe,
    validate_feature_payload,
)

load_environment()
settings = get_settings()
EXPECTED_FEATURES = get_feature_template()
LOGGER = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    """Payload schema for the prediction endpoint."""

    features: dict[str, Any] = Field(..., description="Feature values keyed by training column name")


@lru_cache(maxsize=1)
def load_model():
    """Load the current Production model from MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{settings.model_name}/Production")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize logging and warm the model cache when possible."""
    configure_logging(settings.log_level)
    try:
        load_model()
    except Exception as exc:  # pragma: no cover - exercised by API request path
        LOGGER.warning("Production model is not available yet: %s", exc)
    yield


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    """Simple health endpoint."""
    model_loaded = True
    try:
        load_model()
    except Exception:
        model_loaded = False
    return {"status": "ok", "model_loaded": model_loaded, "model_name": settings.model_name}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    """Run a single prediction request."""
    missing, extra = validate_feature_payload(request.features)
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "extra_features": extra},
        )

    try:
        model = load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Production model is unavailable") from exc

    dataframe = to_dataframe(request.features)
    prediction = int(model.predict(dataframe)[0])
    probability = float(model.predict_proba(dataframe)[0][1])
    return {
        "prediction": prediction,
        "churn_probability": round(probability, 6),
        "model_name": settings.model_name,
    }
