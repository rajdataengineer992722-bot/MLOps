# Production-Grade MLOps Project (Breast Cancer Classification)

This repository contains an end-to-end MLOps pipeline:

- **Training** with Scikit-learn
- **Experiment tracking + model registry** with MLflow
- **Model serving API** with FastAPI
- **Containerization** with Docker and docker-compose
- **CI/CD automation** with GitHub Actions
- **Data drift monitoring** with Evidently
- **Optional dashboard** with Streamlit

---

## 1) Project Structure

```text
mlops-project/
│
├── data/
├── notebooks/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── api/
│   └── app.py
│
├── monitoring/
│   └── drift.py
│
├── streamlit/
│   └── dashboard.py
│
├── tests/
│   └── test_api.py
│
├── .github/workflows/
│   └── ci.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 2) Architecture Diagram (ASCII)

```text
                    +------------------------------+
                    |      GitHub Actions CI       |
                    | tests -> train -> build      |
                    +-------------+----------------+
                                  |
                                  v
+-------------+      logs   +-----+--------------------+
|  Training   +------------->      MLflow Server       |
| (src/train) | metrics/model| Experiments + Registry  |
+------+------+             +-----------+--------------+
       |                                  |
       | register/promote                 | load Production model
       v                                  v
+------+-------------------+      +-------+----------------+
| Evidently Drift Monitor  |      | FastAPI /predict API   |
| (monitoring/drift.py)    |      | (api/app.py)           |
+--------------------------+      +------------------------+
                                             |
                                             v
                                      Client / Streamlit
```

---

## 3) Setup Instructions

### Prerequisites

- Python 3.10+
- Docker + Docker Compose

### Install locally

```bash
cd mlops-project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `PYTHONPATH` to project root for script execution:

```bash
export PYTHONPATH=$(pwd)
```

---

## 4) Run Locally

### Step A: Start MLflow + API + Streamlit with Docker Compose

```bash
docker compose up --build
```

- MLflow UI: http://localhost:5000
- API docs: http://localhost:8000/docs
- Streamlit dashboard: http://localhost:8501

### Step B: Train and register model

From another terminal:

```bash
cd mlops-project
export PYTHONPATH=$(pwd)
python src/train.py
```

This will:
- Load breast cancer dataset
- Train model
- Log params + metrics to MLflow
- Register model and transition latest version to **Production**
- Wait for model version to become READY before stage transition

### Step C: Evaluate

```bash
python src/evaluate.py
```

Saves output to `reports/evaluation.json`.

### Step D: Predict (CLI)

```bash
python src/predict.py --json '{"mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8, "mean area": 1001.0, "mean smoothness": 0.1184, "mean compactness": 0.2776, "mean concavity": 0.3001, "mean concave points": 0.1471, "mean symmetry": 0.2419, "mean fractal dimension": 0.07871, "radius error": 1.095, "texture error": 0.9053, "perimeter error": 8.589, "area error": 153.4, "smoothness error": 0.006399, "compactness error": 0.04904, "concavity error": 0.05373, "concave points error": 0.01587, "symmetry error": 0.03003, "fractal dimension error": 0.006193, "worst radius": 25.38, "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0, "worst smoothness": 0.1622, "worst compactness": 0.6656, "worst concavity": 0.7119, "worst concave points": 0.2654, "worst symmetry": 0.4601, "worst fractal dimension": 0.1189}'
```

### Step E: API prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8, "mean area": 1001.0, "mean smoothness": 0.1184, "mean compactness": 0.2776, "mean concavity": 0.3001, "mean concave points": 0.1471, "mean symmetry": 0.2419, "mean fractal dimension": 0.07871, "radius error": 1.095, "texture error": 0.9053, "perimeter error": 8.589, "area error": 153.4, "smoothness error": 0.006399, "compactness error": 0.04904, "concavity error": 0.05373, "concave points error": 0.01587, "symmetry error": 0.03003, "fractal dimension error": 0.006193, "worst radius": 25.38, "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0, "worst smoothness": 0.1622, "worst compactness": 0.6656, "worst concavity": 0.7119, "worst concave points": 0.2654, "worst symmetry": 0.4601, "worst fractal dimension": 0.1189}}'
```

---

## 5) Monitoring and Drift Detection

Generate Evidently report:

```bash
export PYTHONPATH=$(pwd)
python monitoring/drift.py --output-dir reports/drift
```

Output: `reports/drift/drift_report.html`

---

## 6) CI/CD Workflow

The workflow in `.github/workflows/ci.yml` performs:

1. Checkout code
2. Install dependencies
3. Run API tests with `pytest`
4. Train and register model
5. Generate drift report
6. Build Docker image
7. Mock image push step
8. Mock deploy step

To make deployment fully real, replace mock steps with your registry/Kubernetes/ECS commands.

---

## 7) Production Best Practices Included

- Modular code (`src`, `api`, `monitoring`)
- Logging configuration utility
- Environment-driven config (`.env.example`)
- Registry stage promotion to Production
- API schema validation and strict feature checks
- Graceful `503` response when no Production model is available
- Automated testing and CI pipeline
