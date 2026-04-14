"""Evaluation script for the current Production model."""

from __future__ import annotations

import logging
import os

import mlflow
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.train import compute_metrics
from src.utils import configure_logging, infer_tracking_uri, load_dataset, load_environment, save_json

MODEL_NAME = os.getenv("MODEL_NAME", "customer-churn-classifier")


def evaluate() -> dict[str, object]:
    """Evaluate the current Production model and save a JSON report."""
    load_environment()
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(infer_tracking_uri())
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")

    _, x_test, _, y_test = load_dataset()
    metrics = compute_metrics(model, x_test, y_test)
    predictions = model.predict(x_test)
    output = {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }

    save_json(output, "reports/evaluation.json")
    logger.info("Saved evaluation report to reports/evaluation.json")
    print(pd.Series(metrics).to_string())
    return output


if __name__ == "__main__":
    evaluate()
