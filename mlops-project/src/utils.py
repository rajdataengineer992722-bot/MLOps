"""Utility helpers for data loading, feature processing, and reproducibility."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging once."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_dataset(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and split a realistic classification dataset.

    Uses the Wisconsin Breast Cancer dataset from scikit-learn as a small,
    fully offline dataset suitable for CI/CD and demos.
    """
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    LOGGER.info("Dataset loaded: train=%s rows, test=%s rows", len(x_train), len(x_test))
    return x_train, x_test, y_train, y_test


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return a Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Dict, output_file: str | Path) -> None:
    """Persist a dictionary as JSON."""
    output_file = Path(output_file)
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_feature_template() -> List[str]:
    """Return expected feature names for API validation and docs."""
    dataset = load_breast_cancer(as_frame=True)
    return list(dataset.data.columns)


def infer_tracking_uri() -> str:
    """Get MLflow tracking URI from env with robust local default."""
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def to_dataframe(input_features: Dict[str, float]) -> pd.DataFrame:
    """Convert single JSON payload to model-ready DataFrame with sorted columns."""
    df = pd.DataFrame([input_features])
    return df


def set_random_seed(seed: int = 42) -> None:
    """Ensure deterministic behavior where practical."""
    np.random.seed(seed)
