# Project Summary

## Title

Production-Grade Customer Churn MLOps Pipeline

## Overview

This project is an end-to-end MLOps implementation for predicting telecom customer churn using a Scikit-learn classification model. It is designed to demonstrate the complete machine learning lifecycle in a production-oriented setup, from training and experiment tracking to model serving, CI/CD automation, and drift monitoring.

The system uses a realistic synthetic customer churn dataset with numeric, boolean, and categorical features, making it more representative of real business prediction problems than a toy single-table example.

## Business Problem

Customer churn is a major business concern for subscription-based companies. Predicting which customers are likely to leave helps teams prioritize retention campaigns, reduce revenue loss, and improve customer satisfaction.

This project models that problem as a binary classification task:

- `1` = likely to churn
- `0` = likely to stay

## Objectives

- Build a reproducible training pipeline
- Track experiments and metrics with MLflow
- Register and promote the best model to Production
- Serve predictions through a FastAPI application
- Automate validation through GitHub Actions
- Detect data drift with Evidently AI
- Provide an interactive dashboard for manual scoring

## Tech Stack

- Python 3.10+
- Scikit-learn
- MLflow
- FastAPI
- Docker
- GitHub Actions
- Evidently AI
- Streamlit

## Architecture

```text
Data Generation -> Training Pipeline -> MLflow Tracking + Registry -> Production Model
                                                              |
                                                              v
                                            FastAPI /predict API + Streamlit UI
                                                              |
                                                              v
                                             CI/CD Validation + Drift Monitoring
```

## Key Components

### 1. Training Pipeline

The training workflow:

- generates a deterministic customer churn dataset
- preprocesses numeric and categorical features
- trains a Random Forest classifier
- logs metrics and parameters to MLflow
- registers the trained model
- promotes the best candidate to the `Production` stage

Main file:
- [src/train.py](/e:/mlops/mlops-project/src/train.py)

### 2. Evaluation

The evaluation stage loads the Production model and generates:

- classification metrics
- confusion matrix
- classification report

Main file:
- [src/evaluate.py](/e:/mlops/mlops-project/src/evaluate.py)

### 3. Prediction API

The FastAPI service exposes:

- `GET /health`
- `POST /predict`

It validates input features, loads the current Production model from MLflow, and returns both a prediction and churn probability.

Main file:
- [api/app.py](/e:/mlops/mlops-project/api/app.py)

### 4. Monitoring

The monitoring workflow compares reference data against current data and generates drift reports using Evidently.

Outputs:

- HTML drift report
- JSON drift report

Main file:
- [monitoring/drift.py](/e:/mlops/mlops-project/monitoring/drift.py)

### 5. CI/CD

GitHub Actions automatically:

- installs dependencies
- runs tests
- trains the model
- evaluates the model
- generates drift reports
- builds the Docker image
- performs a smoke deployment test

Workflow file:
- [.github/workflows/ci.yml](/e:/mlops/.github/workflows/ci.yml)

## Model Performance

Latest verified local metrics:

- Accuracy: `0.8625`
- Precision: `0.8452`
- Recall: `0.7802`
- F1 Score: `0.8114`
- ROC AUC: `0.9394`

These values come from the generated evaluation report after training the Production model locally.

## Validation Performed

The project was validated with:

- local unit/API tests via `pytest`
- successful model training and registration
- successful evaluation report generation
- successful drift report generation
- successful live API prediction
- successful GitHub Actions CI workflow run

## Why This Project Is Strong

- It covers the full machine learning lifecycle, not just model training.
- It uses modular, production-style code organization.
- It includes experiment tracking and model registry promotion logic.
- It exposes a serving API suitable for integration.
- It includes monitoring and CI automation, which are often missing in basic ML demos.
- It is fully runnable and GitHub-ready.

## Possible Next Enhancements

- Replace mock deployment with real cloud deployment
- Add authentication and rate limiting to the API
- Add model explainability with SHAP
- Add scheduled retraining
- Persist prediction logs for observability
- Add infrastructure-as-code for deployment

## Repository

- GitHub: https://github.com/rajdataengineer992722-bot/MLOps

## Final Outcome

This project demonstrates practical ML engineering skills across data preparation, model training, registry management, API deployment, monitoring, testing, and CI/CD. It is suitable for portfolio use, interview discussion, and academic or professional submission.
