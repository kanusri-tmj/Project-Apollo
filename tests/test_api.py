"""Tests for the Flask deployment API.

These are skipped automatically when no trained artifact is present (run
``python -m src.train`` to create one).
"""

from __future__ import annotations

import pytest

from src import config, predict

pytestmark = pytest.mark.skipif(
    not list(config.MODELS_DIR.glob("*_latest.joblib")),
    reason="No trained model artifact found; run `python -m src.train` first.",
)


@pytest.fixture(scope="module")
def client():
    flask_app = pytest.importorskip("app.flask_app")
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


def _record() -> dict:
    return {
        "age": 57,
        "sex": 1,
        "chest_pain_type": 4,
        "resting_bp": 140,
        "cholesterol": 260,
        "fasting_blood_sugar": 0,
        "resting_ecg": 1,
        "max_heart_rate": 140,
        "exercise_angina": 1,
        "st_depression": 2.0,
        "st_slope": 2,
        "num_major_vessels": 2,
        "thalassemia": 7,
    }


def test_health_reports_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_index_renders_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Patient measurements" in response.data


def test_predict_returns_probability(client):
    response = client.post("/predict", json={"features": _record()})
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_out_of_range(client):
    bad = _record()
    bad["age"] = 500
    response = client.post("/predict", json={"features": bad})
    assert response.status_code == 422
    assert response.get_json()["details"]


def test_predict_rejects_bad_threshold(client):
    response = client.post("/predict", json={"features": _record(), "threshold": 2})
    assert response.status_code == 400


def test_explain_returns_real_internals(client):
    response = client.post("/explain", json={"features": _record()})
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) >= {"variants", "verdict", "primary", "threshold"}

    primary = body["variants"][body["primary"]]
    # The decomposed terms must reconstruct the logit exactly.
    total = primary["intercept"] + sum(f["contribution"] for f in primary["features"])
    assert abs(total - primary["logit"]) < 1e-4
    # And the verdict must agree with the primary variant's probability.
    assert abs(body["verdict"]["probability"] - primary["probability"]) < 1e-6


def test_explain_contrasts_ridge_and_lasso(client):
    body = client.post("/explain", json={"features": _record()}).get_json()
    variants = body["variants"]
    if "ridge_l2" in variants and "lasso_l1" in variants:
        assert variants["lasso_l1"]["n_selected"] <= variants["ridge_l2"]["n_selected"]


def test_explain_rejects_invalid_input(client):
    bad = _record()
    bad["thalassemia"] = 5
    assert client.post("/explain", json={"features": bad}).status_code == 422


def test_home_page_has_all_three_sections(client):
    body = client.get("/").data
    for heading in (b"Home", b"Live Simulation", b"About Us", b"Abstract", b"Team details"):
        assert heading in body


def test_batch_endpoint(client):
    response = client.post("/api/predict/batch", json={"patients": [_record(), _record()]})
    assert response.status_code == 200
    assert response.get_json()["count"] == 2


def test_model_info_exposes_input_spec(client):
    body = client.get("/model-info").get_json()
    assert body["version"]
    assert len(body["input_spec"]) == len(config.FEATURES)


def test_metrics_endpoint(client):
    client.get("/health")
    body = client.get("/metrics").get_json()
    assert "requests_total" in body
