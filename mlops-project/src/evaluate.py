"""Model evaluation script for the latest Production model in MLflow registry."""

from __future__ import annotations

import logging
import os

import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from src.utils import configure_logging, infer_tracking_uri, load_dataset, save_json

MODEL_NAME = os.getenv("MODEL_NAME", "breast-cancer-classifier")


def evaluate() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(infer_tracking_uri())
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.sklearn.load_model(model_uri)

    _, x_test, _, y_test = load_dataset()
    predictions = model.predict(x_test)

    probs = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_test, probs))

    report = classification_report(y_test, predictions, output_dict=True)
    output = {"metrics": metrics, "report": report}

    save_json(output, "reports/evaluation.json")
    logger.info("Evaluation saved to reports/evaluation.json")

    # Human-friendly output for pipelines.
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    evaluate()
