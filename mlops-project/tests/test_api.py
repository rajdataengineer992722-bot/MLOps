import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mlflow")

from fastapi.testclient import TestClient

from api.app import EXPECTED_FEATURES, app


class DummyModel:
    def predict(self, dataframe):
        return [1]

    def predict_proba(self, dataframe):
        return [[0.09, 0.91]]


def build_payload():
    payload = {"features": {feature: 1 for feature in EXPECTED_FEATURES}}
    payload["features"]["is_premium_plan"] = "yes"
    payload["features"]["autopay_enabled"] = "no"
    payload["features"]["contract_type"] = "month-to-month"
    payload["features"]["region"] = "west"
    payload["features"]["internet_service"] = "fiber"
    return payload


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr("api.app.load_model", lambda: DummyModel())
    client = TestClient(app)

    response = client.post("/predict", json=build_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert 0 <= body["churn_probability"] <= 1


def test_predict_validation(monkeypatch):
    monkeypatch.setattr("api.app.load_model", lambda: DummyModel())
    client = TestClient(app)

    response = client.post("/predict", json={"features": {"bad_feature": 1.0}})
    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_predict_returns_503_when_model_unavailable(monkeypatch):
    def raise_error():
        raise RuntimeError("registry empty")

    monkeypatch.setattr("api.app.load_model", raise_error)
    client = TestClient(app)

    response = client.post("/predict", json=build_payload())
    assert response.status_code == 503
