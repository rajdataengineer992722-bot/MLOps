from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from api.app import EXPECTED_FEATURES, app


class DummyModel:
    def predict(self, df):
        return [1]


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr("api.app.load_model", lambda: DummyModel())

    payload = {"features": {feature: 0.1 for feature in EXPECTED_FEATURES}}
    client = TestClient(app)
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1


def test_predict_validation(monkeypatch):
    monkeypatch.setattr("api.app.load_model", lambda: DummyModel())
    client = TestClient(app)

    response = client.post("/predict", json={"features": {"bad_feature": 1.0}})
    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_predict_returns_503_when_model_unavailable(monkeypatch):
    def _missing_model():
        raise MlflowException("not found")

    monkeypatch.setattr("api.app.load_model", _missing_model)
    client = TestClient(app)

    payload = {"features": {feature: 0.1 for feature in EXPECTED_FEATURES}}
    response = client.post("/predict", json=payload)

    assert response.status_code == 503
