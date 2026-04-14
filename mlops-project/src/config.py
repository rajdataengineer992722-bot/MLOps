"""Centralized configuration helpers for runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.utils import load_environment


@dataclass(frozen=True)
class Settings:
    """Environment-driven application settings."""

    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    model_name: str
    log_level: str
    api_host: str
    api_port: int


def get_settings() -> Settings:
    """Load settings from the environment."""
    load_environment()
    return Settings(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn-experiment"),
        model_name=os.getenv("MODEL_NAME", "customer-churn-classifier"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
    )
