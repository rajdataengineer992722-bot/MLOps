"""Batch/single inference utility for local CLI usage."""

from __future__ import annotations

import argparse
import json
import os

import mlflow

from src.utils import get_feature_template, infer_tracking_uri, to_dataframe


MODEL_NAME = os.getenv("MODEL_NAME", "breast-cancer-classifier")


def predict(payload: dict) -> dict:
    mlflow.set_tracking_uri(infer_tracking_uri())
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")

    df = to_dataframe(payload)
    prediction = int(model.predict(df)[0])
    return {"prediction": prediction}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict using production MLflow model")
    parser.add_argument(
        "--json",
        required=True,
        help="JSON string containing model features",
    )
    args = parser.parse_args()

    payload = json.loads(args.json)
    missing = set(get_feature_template()) - set(payload.keys())
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    result = predict(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
