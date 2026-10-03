import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.serving.app as app_mod

PAYLOAD = {
    "RevolvingUtilizationOfUnsecuredLines": 0.4,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.3,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
}


@pytest.fixture
def client():
    return TestClient(app_mod.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_503_when_no_production_model(client, monkeypatch):
    def boom():
        raise RuntimeError("no model is currently aliased @production")

    monkeypatch.setattr(app_mod._cache, "get", boom)
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 503


def test_predict_uses_configured_threshold_and_reports_version(client, monkeypatch):
    monkeypatch.setattr(app_mod._cache, "get", lambda: ("model", "9"))
    monkeypatch.setattr(app_mod, "score", lambda model, df: np.array([0.51]))
    monkeypatch.setattr(app_mod, "_DECISION_THRESHOLD", 0.5)

    body = client.post("/predict", json=PAYLOAD).json()

    assert body["model_version"] == "9"
    assert body["predicted_label"] == 1
    assert body["predicted_prob"] == pytest.approx(0.51)


def test_predict_passes_hyphenated_column_names_to_the_model(client, monkeypatch):
    seen = {}

    def fake_score(model, df):
        seen["columns"] = list(df.columns)
        return np.array([0.1])

    monkeypatch.setattr(app_mod._cache, "get", lambda: ("model", "1"))
    monkeypatch.setattr(app_mod, "score", fake_score)

    client.post("/predict", json=PAYLOAD)

    assert "NumberOfTime30-59DaysPastDueNotWorse" in seen["columns"]


def test_predict_rejects_invalid_age(client):
    response = client.post("/predict", json={**PAYLOAD, "age": 5})
    assert response.status_code == 422


def test_model_info(client, monkeypatch):
    monkeypatch.setattr(app_mod._cache, "get", lambda: ("model", "4"))
    assert client.get("/model-info").json()["production_version"] == "4"
