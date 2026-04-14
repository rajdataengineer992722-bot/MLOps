"""Shared utilities for configuration, dataset preparation, and IO."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_90d",
    "avg_call_minutes",
    "payment_delay_days",
    "nps_score",
]
BOOLEAN_FEATURES = ["is_premium_plan", "autopay_enabled"]
CATEGORICAL_FEATURES = ["contract_type", "region", "internet_service"]
TARGET_COLUMN = "churned"

FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
FEATURE_DEFAULTS: dict[str, Any] = {
    "age": 42,
    "tenure_months": 18,
    "monthly_charges": 89.5,
    "total_charges": 1611.0,
    "support_tickets_90d": 2,
    "avg_call_minutes": 245.0,
    "payment_delay_days": 3,
    "nps_score": 24,
    "is_premium_plan": "yes",
    "autopay_enabled": "no",
    "contract_type": "month-to-month",
    "region": "west",
    "internet_service": "fiber",
}


def load_environment() -> None:
    """Load environment variables from the project .env file when available."""
    load_dotenv(ROOT_DIR / ".env", override=False)


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide logging once."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=False,
    )


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return its resolved path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def infer_tracking_uri() -> str:
    """Resolve the MLflow tracking URI from the environment."""
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def set_random_seed(seed: int = 42) -> None:
    """Set deterministic random state where practical."""
    np.random.seed(seed)


def save_json(payload: dict[str, Any], output_file: str | Path) -> Path:
    """Persist a dictionary as indented JSON."""
    destination = Path(output_file)
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def load_json(input_file: str | Path) -> dict[str, Any]:
    """Load a JSON document from disk."""
    return json.loads(Path(input_file).read_text(encoding="utf-8"))


def get_feature_template() -> list[str]:
    """Return the exact request schema expected by the model."""
    return FEATURE_COLUMNS.copy()


def get_sample_payload() -> dict[str, Any]:
    """Return a representative single prediction payload."""
    return FEATURE_DEFAULTS.copy()


def to_dataframe(input_features: dict[str, Any]) -> pd.DataFrame:
    """Convert a single request payload into a model-ready DataFrame."""
    return pd.DataFrame([{column: input_features[column] for column in FEATURE_COLUMNS}])


def validate_feature_payload(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return missing and extra feature names for a request payload."""
    expected = set(FEATURE_COLUMNS)
    received = set(payload)
    missing = sorted(expected - received)
    extra = sorted(received - expected)
    return missing, extra


def dataset_path() -> Path:
    """Location of the generated churn dataset."""
    return DATA_DIR / "customer_churn.csv"


def _build_customer_churn_dataset(rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic customer churn dataset."""
    features, target = make_classification(
        n_samples=rows,
        n_features=len(NUMERIC_FEATURES),
        n_informative=6,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.62, 0.38],
        class_sep=1.1,
        random_state=random_state,
    )
    frame = pd.DataFrame(features, columns=NUMERIC_FEATURES)

    frame["age"] = np.clip((frame["age"] * 11 + 42).round(), 18, 85).astype(int)
    frame["tenure_months"] = np.clip((frame["tenure_months"] * 10 + 24).round(), 1, 72).astype(int)
    frame["monthly_charges"] = np.clip(frame["monthly_charges"] * 18 + 85, 25, 185).round(2)
    frame["support_tickets_90d"] = np.clip(
        np.round(np.abs(frame["support_tickets_90d"]) * 1.8 + target * 1.5),
        0,
        12,
    ).astype(int)
    frame["avg_call_minutes"] = np.clip(frame["avg_call_minutes"] * 75 + 260, 20, 1200).round(2)
    frame["payment_delay_days"] = np.clip(
        np.round(np.abs(frame["payment_delay_days"]) * 4 + target * 2),
        0,
        30,
    ).astype(int)
    frame["nps_score"] = np.clip((frame["nps_score"] * 12 + 22).round(), -100, 100).astype(int)

    premium_probability = np.clip(0.35 + target * 0.15, 0.1, 0.9)
    autopay_probability = np.clip(0.65 - target * 0.18, 0.1, 0.95)
    frame["is_premium_plan"] = np.where(np.random.rand(rows) < premium_probability, "yes", "no")
    frame["autopay_enabled"] = np.where(np.random.rand(rows) < autopay_probability, "yes", "no")

    contract_options = np.array(["month-to-month", "one-year", "two-year"])
    region_options = np.array(["north", "south", "east", "west"])
    internet_options = np.array(["fiber", "dsl", "wireless"])

    contract_probabilities = np.where(
        target[:, None] == 1,
        np.array([0.72, 0.18, 0.10]),
        np.array([0.34, 0.31, 0.35]),
    )
    region_probabilities = np.array(
        [
            [0.18, 0.26, 0.27, 0.29] if label == 1 else [0.29, 0.22, 0.25, 0.24]
            for label in target
        ]
    )
    internet_probabilities = np.where(
        target[:, None] == 1,
        np.array([0.56, 0.18, 0.26]),
        np.array([0.33, 0.37, 0.30]),
    )

    frame["contract_type"] = [
        np.random.choice(contract_options, p=probabilities) for probabilities in contract_probabilities
    ]
    frame["region"] = [
        np.random.choice(region_options, p=probabilities) for probabilities in region_probabilities
    ]
    frame["internet_service"] = [
        np.random.choice(internet_options, p=probabilities) for probabilities in internet_probabilities
    ]

    frame["total_charges"] = (
        frame["monthly_charges"] * frame["tenure_months"]
        + np.where(frame["is_premium_plan"] == "yes", 140, 40)
        - frame["payment_delay_days"] * 3
    ).round(2)
    frame[TARGET_COLUMN] = target.astype(int)
    return frame


def ensure_dataset() -> Path:
    """Materialize the dataset to disk if it does not already exist."""
    destination = dataset_path()
    if not destination.exists():
        ensure_dir(DATA_DIR)
        dataset = _build_customer_churn_dataset()
        dataset.to_csv(destination, index=False)
    return destination


def load_dataset(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the churn dataset and return a train/test split."""
    data_file = ensure_dataset()
    dataset = pd.read_csv(data_file)
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def save_monitoring_datasets(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Persist reference and current datasets used by the monitoring job."""
    monitoring_dir = ensure_dir(DATA_DIR / "monitoring")
    reference_data.to_csv(monitoring_dir / "reference.csv", index=False)
    current_data.to_csv(monitoring_dir / "current.csv", index=False)
