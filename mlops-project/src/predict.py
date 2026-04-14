"""CLI utility for local predictions with the Production model."""

from __future__ import annotations

import argparse
import json
import os

import mlflow

from src.utils import (
    configure_logging,
    get_sample_payload,
    infer_tracking_uri,
    load_environment,
    to_dataframe,
    validate_feature_payload,
)

MODEL_NAME = os.getenv("MODEL_NAME", "customer-churn-classifier")


def predict(payload: dict[str, object]) -> dict[str, object]:
    """Run inference against the current Production model."""
    mlflow.set_tracking_uri(infer_tracking_uri())
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    dataframe = to_dataframe(payload)
    prediction = int(model.predict(dataframe)[0])
    probability = float(model.predict_proba(dataframe)[0][1])
    return {
        "prediction": prediction,
        "churn_probability": round(probability, 6),
        "model_name": MODEL_NAME,
    }


def main() -> None:
    """CLI entrypoint."""
    load_environment()
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="Predict customer churn using the Production model")
    parser.add_argument("--json", help="JSON string with feature values")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use the built-in sample payload instead of providing --json",
    )
    args = parser.parse_args()

    if not args.sample and not args.json:
        raise SystemExit("Provide either --json or --sample")

    payload = get_sample_payload() if args.sample else json.loads(args.json)
    missing, extra = validate_feature_payload(payload)
    if missing or extra:
        raise ValueError(f"Invalid payload. missing={missing}, extra={extra}")

    print(json.dumps(predict(payload), indent=2))


if __name__ == "__main__":
    main()
