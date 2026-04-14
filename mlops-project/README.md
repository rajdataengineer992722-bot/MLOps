# Production-Grade MLOps Project

This repository contains a complete, runnable MLOps workflow for a customer churn classification use case. It covers model training, experiment tracking, model registry promotion, API serving, CI/CD automation, and drift monitoring in a single GitHub-ready project.

## Use Case

The project predicts whether a telecom customer is likely to churn using a realistic synthetic dataset with mixed feature types:

- Numeric features such as `monthly_charges`, `tenure_months`, and `support_tickets_90d`
- Boolean service flags such as `autopay_enabled`
- Categorical profile fields such as `contract_type`, `region`, and `internet_service`

The dataset is generated deterministically and stored locally in `data/customer_churn.csv`, which keeps the project fully runnable offline and reproducible in CI.

## Project Structure

```text
mlops-project/
|
|-- data/
|-- notebooks/
|-- src/
|   |-- config.py
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|   `-- utils.py
|
|-- api/
|   `-- app.py
|
|-- monitoring/
|   `-- drift.py
|
|-- streamlit/
|   `-- dashboard.py
|
|-- tests/
|   |-- test_api.py
|   `-- test_sanity.py
|
|-- .github/workflows/
|   `-- ci.yml
|
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- .env.example
`-- README.md
```

## Architecture

```text
                    +--------------------------------------+
                    |          GitHub Actions CI           |
                    | test -> train -> evaluate -> build   |
                    | -> smoke deploy                      |
                    +------------------+-------------------+
                                       |
                                       v
+------------------+         +---------+-------------------+
| src/train.py     |         | MLflow Tracking + Registry  |
| - generate data  |-------> | experiments, artifacts,     |
| - preprocess     | metrics | model versions, Production  |
| - train model    | model   +---------+-------------------+
+---------+--------+                   |
          |                            | load latest Production model
          | save monitoring snapshots   v
          v                    +--------+-------------------+
+---------+--------+           | FastAPI Serving Layer      |
| Evidently Drift   |          | api/app.py                |
| monitoring/drift  |          | POST /predict             |
+---------+---------+          +--------+------------------+
          |                               |
          v                               v
    HTML and JSON reports         Streamlit dashboard / clients
```

## Tech Stack

- Python 3.10+
- Scikit-learn
- MLflow
- FastAPI
- Docker
- GitHub Actions
- Evidently AI
- Streamlit

## Local Setup

### 1. Create a virtual environment and install dependencies

```bash
cd mlops-project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

PowerShell:

```powershell
cd mlops-project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### 2. Train and register the model

```bash
python src/train.py
```

This step:

- generates `data/customer_churn.csv` if it does not exist
- logs parameters and metrics to MLflow
- registers a model version
- promotes the best version to `Production`
- writes monitoring reference and current datasets

### 3. Evaluate the Production model

```bash
python src/evaluate.py
```

Outputs:

- `reports/training_summary.json`
- `reports/evaluation.json`

### 4. Run CLI prediction

```bash
python src/predict.py --sample
```

Or provide a custom payload:

```bash
python src/predict.py --json '{"age": 52, "tenure_months": 8, "monthly_charges": 118.5, "total_charges": 948.0, "support_tickets_90d": 4, "avg_call_minutes": 180.0, "payment_delay_days": 9, "nps_score": -12, "is_premium_plan": "no", "autopay_enabled": "no", "contract_type": "month-to-month", "region": "south", "internet_service": "fiber"}'
```

## Running the API

### Local API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /predict`

Sample request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"age":52,"tenure_months":8,"monthly_charges":118.5,"total_charges":948.0,"support_tickets_90d":4,"avg_call_minutes":180.0,"payment_delay_days":9,"nps_score":-12,"is_premium_plan":"no","autopay_enabled":"no","contract_type":"month-to-month","region":"south","internet_service":"fiber"}}'
```

## Docker

### Start MLflow, API, and Streamlit

```bash
docker compose up --build
```

Services:

- MLflow UI: `http://localhost:5000`
- FastAPI docs: `http://localhost:8000/docs`
- Streamlit dashboard: `http://localhost:8501`

If you want the API to serve a Production model inside Docker, run training once locally before building the image so the local MLflow database and artifacts are present in the project directory.

## Monitoring and Drift Detection

Generate drift reports with Evidently:

```bash
python monitoring/drift.py --output-dir reports/drift
```

Outputs:

- `reports/drift/drift_report.html`
- `reports/drift/drift_report.json`

The monitoring job compares reference training data against a current dataset snapshot with intentionally shifted distributions to demonstrate drift detection.

## Streamlit Dashboard

Run locally:

```bash
streamlit run streamlit/dashboard.py
```

The dashboard loads the current Production model and provides a simple operator UI for interactive scoring.

## Testing

Run the API and sanity tests:

```bash
pytest tests -q
```

The API tests use dependency-aware imports so minimal environments can skip optional runtime tests without breaking the whole suite.

## CI/CD Workflow

The workflow in `.github/workflows/ci.yml` does the following on pushes and pull requests:

1. Checks out the repository
2. Installs Python dependencies
3. Runs pytest
4. Trains and registers the model in a local MLflow SQLite backend
5. Evaluates the Production model
6. Generates an Evidently drift report
7. Builds the Docker image
8. Executes a mock push step
9. Starts the API container and runs a health-check smoke test as a lightweight deployment verification

To integrate with a real registry or platform, replace the mock push step and local smoke deployment with your registry login and infrastructure deployment commands.

## Production Practices Included

- modular code organization
- environment-driven configuration
- reproducible data generation
- train and evaluation artifacts
- MLflow experiment tracking and model versioning
- best-model promotion logic for Production stage
- API schema validation and graceful `503` handling when no model is registered
- containerized serving and dashboard services
- automated CI verification
- Evidently-based drift reporting
