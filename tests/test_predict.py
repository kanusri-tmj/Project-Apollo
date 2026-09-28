"""Tests for the inference helpers used by both deployment apps."""

from __future__ import annotations

import pytest

from src import config, predict


def _valid_record() -> dict:
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


def test_validate_accepts_well_formed_record():
    assert predict.validate_record(_valid_record()) == []


def test_validate_rejects_out_of_range_numeric():
    record = _valid_record()
    record["age"] = 200
    errors = predict.validate_record(record)
    assert any("age" in e for e in errors)


def test_validate_rejects_unknown_category_code():
    record = _valid_record()
    record["thalassemia"] = 5  # not a valid UCI code
    errors = predict.validate_record(record)
    assert any("thalassemia" in e for e in errors)


def test_validate_reports_non_numeric():
    record = _valid_record()
    record["cholesterol"] = "high"
    errors = predict.validate_record(record)
    assert any("cholesterol" in e for e in errors)


def test_predict_patient_returns_expected_keys(fitted_pipeline):
    result = predict.predict_patient(_valid_record(), model=fitted_pipeline)
    assert set(result) >= {"prediction", "label", "probability", "risk_percent", "warnings"}
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in ("Disease present", "No disease")


def test_threshold_moves_prediction(fitted_pipeline):
    record = _valid_record()
    low = predict.predict_patient(record, model=fitted_pipeline, threshold=0.05)
    high = predict.predict_patient(record, model=fitted_pipeline, threshold=0.95)
    assert low["probability"] == high["probability"]
    assert low["prediction"] >= high["prediction"]


def test_predict_batch_handles_missing_fields(fitted_pipeline):
    partial = {"age": 60, "sex": 1, "cholesterol": 250}
    results = predict.predict_batch([partial], model=fitted_pipeline)
    assert len(results) == 1
    assert results[0]["prediction"] in (0, 1)


def test_predict_batch_empty_returns_empty(fitted_pipeline):
    assert predict.predict_batch([], model=fitted_pipeline) == []


def test_feature_list_matches_config():
    assert len(config.FEATURES) == 13
    assert config.TARGET not in config.FEATURES
