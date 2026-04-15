# Complete Code Flow

This document explains the complete step-by-step code flow of the project and the purpose of every file currently present in this repository checkout.

## High-Level Flow

The project is an end-to-end MLOps workflow for a customer churn classifier.

1. Configuration is loaded from `.env`.
2. The dataset is generated if it does not already exist.
3. The training pipeline builds preprocessing and model steps.
4. The trained model is logged to MLflow.
5. The model is registered and optionally promoted to `Production`.
6. Monitoring reference/current datasets are saved.
7. Evaluation loads the current `Production` model and writes a report.
8. CLI inference, FastAPI inference, and Streamlit inference all load the `Production` model from MLflow.
9. Drift monitoring compares reference and current data with Evidently and writes HTML/JSON reports.
10. Tests verify API behavior and basic project structure.

## Runtime Flow Step by Step

### 1. Environment and Configuration

Shared configuration starts in `src/utils.py` and `src/config.py`.

- `src/utils.py`
  - defines project paths such as `ROOT_DIR`, `DATA_DIR`, and `REPORTS_DIR`
  - defines the feature groups and the expected feature order
  - loads environment variables from `.env`
  - provides helpers for JSON IO, logging, dataset generation, dataset loading, payload validation, and monitoring dataset persistence
- `src/config.py`
  - wraps environment variables in a `Settings` dataclass
  - centralizes runtime values such as MLflow URI, model name, log level, and API host/port

Main config values come from:

- `.env`
- `.env.example`

### 2. Dataset Creation and Loading

Dataset behavior lives in `src/utils.py`.

1. `ensure_dataset()` checks whether `data/customer_churn.csv` exists.
2. If not, `_build_customer_churn_dataset()` creates a synthetic churn dataset using `make_classification`.
3. Numeric fields are transformed into business-like ranges such as age, tenure, monthly charges, and NPS.
4. Boolean and categorical fields are generated with churn-dependent probabilities.
5. `total_charges` is derived from monthly charges, tenure, premium status, and payment delay.
6. The dataset is written to `data/customer_churn.csv`.
7. `load_dataset()` reads the CSV, splits features and target, and returns a stratified train/test split.

### 3. Training Flow

Training starts in `src/train.py`.

1. `train()` loads environment variables and configures logging.
2. It sets a deterministic seed with `set_random_seed(42)`.
3. It points MLflow to the configured tracking URI.
4. It creates or uses the configured MLflow experiment.
5. It loads `x_train`, `x_test`, `y_train`, and `y_test` from `load_dataset()`.
6. It creates a `TrainingConfig` dataclass with model hyperparameters.
7. `build_pipeline()` constructs the sklearn pipeline:
   - numeric columns use median imputation and standard scaling
   - boolean columns use most-frequent imputation and one-hot encoding
   - categorical columns use most-frequent imputation and one-hot encoding
   - the final estimator is `RandomForestClassifier`
8. Inside an MLflow run, the model is fitted on `x_train` and `y_train`.
9. `compute_metrics()` evaluates the trained model on `x_test` and `y_test`.
10. MLflow logs:
   - parameters from `TrainingConfig`
   - metrics such as accuracy, precision, recall, F1, and ROC AUC
   - tags like model name and problem type
   - a schema artifact containing feature columns and a sample payload
   - the sklearn model artifact itself
11. The model URI is built from the MLflow run ID.
12. `register_and_promote_model()` registers the model version in MLflow.
13. The code checks the current `Production` model version.
14. If the new model's F1 is better than or equal to the current `Production` model, it is promoted to `Production`.
15. Otherwise, it is moved to `Staging`.
16. Training also prepares monitoring datasets:
   - reference data = `x_train`
   - current data = modified `x_test`
17. These monitoring datasets are saved to:
   - `data/monitoring/reference.csv`
   - `data/monitoring/current.csv`
18. A training summary is written to `reports/training_summary.json`.

### 4. Evaluation Flow

Evaluation starts in `src/evaluate.py`.

1. `evaluate()` loads environment variables and logging.
2. It points MLflow to the tracking backend.
3. It loads the current `Production` model from MLflow.
4. It reloads the dataset using `load_dataset()`.
5. It scores the test set using `compute_metrics()` imported from `src/train.py`.
6. It also computes:
   - confusion matrix
   - full sklearn classification report
7. The evaluation output is written to `reports/evaluation.json`.
8. Key metric values are printed to the console.

### 5. CLI Prediction Flow

CLI inference starts in `src/predict.py`.

1. `main()` loads environment variables and logging.
2. It accepts either:
   - `--sample`
   - `--json '{...}'`
3. If neither argument is provided, the script exits.
4. If `--sample` is used, `get_sample_payload()` supplies default features.
5. If `--json` is used, the JSON string is parsed.
6. `validate_feature_payload()` checks for missing and extra features.
7. `predict()` sets the MLflow tracking URI.
8. It loads the current `Production` model from MLflow.
9. `to_dataframe()` converts the payload into a single-row DataFrame with the exact training feature order.
10. The model returns:
    - class prediction
    - churn probability
11. The script prints the prediction result as formatted JSON.

### 6. FastAPI Serving Flow

API serving starts in `api/app.py`.

1. Module import loads environment variables and reads settings through `get_settings()`.
2. `EXPECTED_FEATURES` is created from `get_feature_template()`.
3. `load_model()` is decorated with `lru_cache(maxsize=1)`, so the Production model is loaded once and then reused.
4. During FastAPI lifespan startup:
   - logging is configured
   - the API tries to warm the model cache
   - if the Production model is unavailable, startup continues and logs a warning
5. `GET /health`
   - tries to load the model
   - returns `{status, model_loaded, model_name}`
6. `POST /predict`
   - validates the `features` payload using `validate_feature_payload()`
   - returns HTTP `422` if fields are missing or extra
   - attempts to load the Production model
   - returns HTTP `503` if the model is unavailable
   - converts the request payload to a DataFrame
   - computes prediction and probability
   - returns JSON with `prediction`, `churn_probability`, and `model_name`

### 7. Drift Monitoring Flow

Monitoring starts in `monitoring/drift.py`.

1. `main()` parses `--output-dir`.
2. It calls `generate_drift_report()`.
3. `generate_drift_report()` loads environment variables.
4. It calls `load_monitoring_frames()`.
5. `load_monitoring_frames()` checks whether the monitoring CSVs already exist.
6. If they exist, it reads:
   - `data/monitoring/reference.csv`
   - `data/monitoring/current.csv`
7. If they do not exist:
   - it uses `load_dataset()` to get train/test data
   - it uses `x_train` as reference data
   - it creates a synthetic current snapshot from `x_test`
   - it intentionally shifts distributions, including `monthly_charges` and `contract_type`
   - it persists the generated monitoring datasets
8. Evidently builds a `Report(metrics=[DataDriftPreset()])`.
9. The report compares reference data against current data.
10. Output files are written to the selected output directory:
    - `drift_report.html`
    - `drift_report.json`

### 8. Streamlit Dashboard Flow

Interactive UI starts in `streamlit/dashboard.py`.

1. The module loads environment variables and reads `MODEL_NAME`.
2. Streamlit page metadata is configured.
3. The current `Production` model is loaded from MLflow.
4. Default values are loaded using `get_sample_payload()`.
5. The sidebar offers a button to load a sample customer.
6. Two main columns are rendered:
   - left column: numeric inputs
   - right column: boolean and categorical selectors
7. Input widgets are generated from the feature lists defined in `src/utils.py`.
8. When the user clicks the prediction button:
   - the entered values are converted to a DataFrame
   - the model predicts the class and probability
   - the UI displays a risk label and churn probability metric

### 9. Test Flow

Test code lives in `tests/`.

- `tests/test_sanity.py`
  - checks that key project files exist
  - acts as a minimal structure sanity test
- `tests/test_api.py`
  - skips cleanly if `fastapi` or `mlflow` are unavailable
  - uses `TestClient` to exercise the API
  - defines a `DummyModel` so tests do not rely on a real MLflow registry model
  - tests `/health`
  - tests successful `/predict`
  - tests validation behavior for bad payloads
  - tests that the API returns `503` if model loading fails

## Purpose of Every File

### Root Files

- `README.md`
  - main project documentation
  - setup, usage, architecture, and workflow summary
- `PROJECT_SUMMARY.md`
  - shorter presentation-style summary of the repository
- `requirements.txt`
  - Python dependency list
- `Dockerfile`
  - builds the API image and runs `uvicorn api.app:app`
- `docker-compose.yml`
  - starts MLflow, FastAPI, and Streamlit together
- `.env.example`
  - sample runtime configuration
- `.env`
  - active local runtime configuration for this checkout
- `.gitignore`
  - ignores caches, virtual env, logs, MLflow outputs, and reports
- `CODE_FLOW.md`
  - this code-flow document
- `mlflow.db`
  - local MLflow SQLite backend store
- `mlflow-ui.out.log`, `mlflow-ui.err.log`
  - local logs from running the MLflow UI/server
- `uvicorn-8765.out.log`, `uvicorn-8765.err.log`
  - local logs from running the FastAPI server

### Source Package

- `src/__init__.py`
  - marks `src` as a package
- `src/utils.py`
  - shared utilities for configuration, data, payload validation, file IO, and monitoring snapshots
- `src/config.py`
  - runtime settings dataclass
- `src/train.py`
  - end-to-end training, MLflow logging, registration, and promotion
- `src/evaluate.py`
  - loads Production model and writes evaluation results
- `src/predict.py`
  - CLI prediction utility

### API

- `api/app.py`
  - FastAPI application for prediction serving

### Monitoring

- `monitoring/drift.py`
  - Evidently-based drift reporting workflow

### Streamlit

- `streamlit/dashboard.py`
  - interactive scoring dashboard

### Tests

- `tests/test_api.py`
  - API tests with a dummy model
- `tests/test_sanity.py`
  - repository structure checks

### Data Files

- `data/.gitkeep`
  - keeps the data directory in git when empty
- `data/customer_churn.csv`
  - generated synthetic churn dataset
- `data/monitoring/reference.csv`
  - reference monitoring dataset
- `data/monitoring/current.csv`
  - current monitoring dataset used for drift checks

### Reports

- `reports/training_summary.json`
  - summary from training
- `reports/evaluation.json`
  - evaluation output for the Production model
- `reports/drift/drift_report.html`
  - human-readable drift report
- `reports/drift/drift_report.json`
  - machine-readable drift report

### MLflow Artifact Store

- `mlruns/...`
  - MLflow run artifacts, including:
    - logged model files
    - schema artifacts
    - environment metadata
    - input examples

### Placeholder/Generated Files

- `notebooks/.gitkeep`
  - placeholder for notebooks directory
- `__pycache__/...`
  - compiled Python cache files generated by the interpreter

## Key File Relationships

- `src/utils.py` is the shared dependency for almost everything.
- `src/train.py` creates the model and saves monitoring inputs.
- `src/evaluate.py`, `src/predict.py`, `api/app.py`, and `streamlit/dashboard.py` all consume the current MLflow `Production` model.
- `monitoring/drift.py` consumes saved monitoring datasets and generates drift reports.
- `tests/test_api.py` targets `api/app.py`.

## Most Important Entry Points

- Training: `python src/train.py`
- Evaluation: `python src/evaluate.py`
- CLI prediction: `python src/predict.py --sample`
- API serving: `uvicorn api.app:app --host 0.0.0.0 --port 8000`
- Drift monitoring: `python monitoring/drift.py --output-dir reports/drift`
- Dashboard: `streamlit run streamlit/dashboard.py`
- Tests: `pytest tests -q`

## Notes About This Checkout

This local checkout includes generated artifacts and runtime files such as:

- `.env`
- `mlflow.db`
- `mlruns/`
- `reports/`
- local log files

Also, the docs mention a GitHub Actions workflow, but no `.github/workflows/ci.yml` file is present inside this local `mlops-project` checkout. That means CI is described in documentation, but the workflow file itself is not currently part of this working tree.
