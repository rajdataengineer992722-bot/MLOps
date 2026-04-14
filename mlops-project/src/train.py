"""Training pipeline with MLflow tracking and model registry promotion."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

import mlflow
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    configure_logging,
    get_feature_template,
    get_sample_payload,
    infer_tracking_uri,
    load_dataset,
    load_environment,
    save_json,
    save_monitoring_datasets,
    set_random_seed,
)

MODEL_NAME = os.getenv("MODEL_NAME", "customer-churn-classifier")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn-experiment")


@dataclass
class TrainingConfig:
    """Model hyperparameters and pipeline options."""

    n_estimators: int = 250
    max_depth: int = 10
    min_samples_leaf: int = 3
    random_state: int = 42


def build_pipeline(config: TrainingConfig) -> Pipeline:
    """Construct the preprocessing and training pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("boolean", categorical_transformer, BOOLEAN_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def compute_metrics(model: Pipeline, features, target) -> dict[str, float]:
    """Compute classification metrics for a trained model."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions)),
        "recall": float(recall_score(target, predictions)),
        "f1": float(f1_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    """Create the registered model if needed."""
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)


def get_current_production_version(client: MlflowClient, model_name: str) -> ModelVersion | None:
    """Return the current Production version if one exists."""
    for version in client.search_model_versions(f"name='{model_name}'"):
        if version.current_stage == "Production":
            return version
    return None


def get_version_f1(version: ModelVersion | None) -> float:
    """Extract the validation F1 metric from a registered model version."""
    if version is None:
        return -1.0
    return float(version.tags.get("validation_f1", "-1"))


def register_and_promote_model(
    *,
    model_uri: str,
    model_name: str,
    metrics: dict[str, float],
) -> ModelVersion:
    """Register a new model version and promote it when it outperforms production."""
    logger = logging.getLogger(__name__)
    client = MlflowClient()
    ensure_registered_model(client, model_name)

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.set_model_version_tag(model_name, registered.version, "validation_f1", f"{metrics['f1']:.6f}")
    client.set_model_version_tag(model_name, registered.version, "validation_roc_auc", f"{metrics['roc_auc']:.6f}")
    client.set_model_version_tag(model_name, registered.version, "candidate_status", "pending")

    current_production = get_current_production_version(client, model_name)
    if metrics["f1"] >= get_version_f1(current_production):
        client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        client.set_model_version_tag(model_name, registered.version, "candidate_status", "promoted")
        logger.info("Promoted model version %s to Production", registered.version)
    else:
        client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        client.set_model_version_tag(model_name, registered.version, "candidate_status", "staged")
        logger.info(
            "Kept existing Production model because candidate F1 %.4f did not exceed %.4f",
            metrics["f1"],
            get_version_f1(current_production),
        )
    return registered


def train() -> dict[str, object]:
    """Run the end-to-end training workflow."""
    load_environment()
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)
    set_random_seed(42)

    mlflow.set_tracking_uri(infer_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    x_train, x_test, y_train, y_test = load_dataset()
    config = TrainingConfig()
    model = build_pipeline(config)

    with mlflow.start_run(run_name="customer-churn-random-forest") as run:
        model.fit(x_train, y_train)
        metrics = compute_metrics(model, x_test, y_test)

        mlflow.log_params(asdict(config))
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "model_name": MODEL_NAME,
                "dataset": "synthetic_customer_churn",
                "problem_type": "binary_classification",
            }
        )
        mlflow.log_dict(
            {
                "feature_columns": get_feature_template(),
                "sample_payload": get_sample_payload(),
            },
            "artifacts/model_schema.json",
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=x_test.head(3),
            registered_model_name=None,
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        registered_version = register_and_promote_model(
            model_uri=model_uri,
            model_name=MODEL_NAME,
            metrics=metrics,
        )

        current_data = x_test.copy()
        current_data["monthly_charges"] = (current_data["monthly_charges"] * 1.08).round(2)
        current_data["support_tickets_90d"] = (current_data["support_tickets_90d"] + 1).clip(upper=12)
        save_monitoring_datasets(reference_data=x_train, current_data=current_data)

        summary = {
            "run_id": run.info.run_id,
            "model_name": MODEL_NAME,
            "registered_version": registered_version.version,
            "metrics": metrics,
        }
        save_json(summary, "reports/training_summary.json")
        logger.info("Training completed: %s", summary)
        return summary


if __name__ == "__main__":
    train()
