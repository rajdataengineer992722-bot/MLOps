"""FastAPI service exposing production model inference endpoint."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils import configure_logging, get_feature_template, infer_tracking_uri, to_dataframe


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature map by column name")


app = FastAPI(title="MLOps Prediction API", version="1.0.0")
EXPECTED_FEATURES = get_feature_template()
MODEL_NAME = os.getenv("MODEL_NAME", "breast-cancer-classifier")


@lru_cache(maxsize=1)
def load_model():
    mlflow.set_tracking_uri(infer_tracking_uri())
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")


@app.on_event("startup")
def startup_event() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    received_features = request.features

    missing = sorted(set(EXPECTED_FEATURES) - set(received_features.keys()))
    extra = sorted(set(received_features.keys()) - set(EXPECTED_FEATURES))
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "extra_features": extra},
        )

    dataframe = to_dataframe(received_features)
    model = load_model()
    pred = int(model.predict(dataframe)[0])
    return {"prediction": pred, "model_name": MODEL_NAME}
