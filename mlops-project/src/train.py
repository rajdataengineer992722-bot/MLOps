"""Model training pipeline with MLflow tracking and model registry promotion."""

from __future__ import annotations

import logging
import os
import time

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import configure_logging, infer_tracking_uri, load_dataset, set_random_seed

MODEL_NAME = os.getenv("MODEL_NAME", "breast-cancer-classifier")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "breast-cancer-experiment")


def train() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)
    set_random_seed(42)

    mlflow.set_tracking_uri(infer_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    x_train, x_test, y_train, y_test = load_dataset()

    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "random_state": 42,
    }

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(**params)),
        ]
    )

    with mlflow.start_run(run_name="rf-baseline") as run:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("Run %s complete with metrics=%s", run.info.run_id, metrics)

        model_uri = f"runs:/{run.info.run_id}/model"
        register_and_promote_model(model_uri=model_uri, model_name=MODEL_NAME)


def register_and_promote_model(model_uri: str, model_name: str) -> None:
    """Register a model and transition latest version to Production stage."""
    logger = logging.getLogger(__name__)
    client = MlflowClient()

    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    registered = wait_for_model_version_ready(client, model_name, registered.version)

    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )

    logger.info("Model %s version %s moved to Production", model_name, registered.version)


def wait_for_model_version_ready(
    client: MlflowClient, model_name: str, version: str, timeout_seconds: int = 60
):
    """Wait until a newly registered model version is READY before stage transition."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        model_version = client.get_model_version(name=model_name, version=version)
        if model_version.status == ModelVersionStatus.READY:
            return model_version
        time.sleep(1)
    raise TimeoutError(f"Model version {model_name} v{version} was not READY within {timeout_seconds}s")


if __name__ == "__main__":
    train()
