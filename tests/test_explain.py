"""Tests for :mod:`src.explain` — the maths behind the Live Simulation UI.

These assert that what the frontend animates is genuinely what the model
computes, not a re-derivation that could drift.
"""

from __future__ import annotations

import pytest

from src import config, explain, predict, preprocessing, train


@pytest.fixture(scope="module")
def variant_models(synthetic_frame):
    """Baseline / Ridge / Lasso pipelines fitted on the same synthetic split."""
    X, y = preprocessing.split_features_target(synthetic_frame)
    models = {}
    for variant in config.MODEL_VARIANTS:
        model = train.build_model_pipeline(variant, C=1.0)
        model.fit(X, y)
        models[variant] = model
    return models


def _record() -> dict:
    return {
        "age": 57, "sex": 1, "chest_pain_type": 4, "resting_bp": 140,
        "cholesterol": 260, "fasting_blood_sugar": 0, "resting_ecg": 1,
        "max_heart_rate": 140, "exercise_angina": 1, "st_depression": 2.0,
        "st_slope": 2, "num_major_vessels": 2, "thalassemia": 7,
    }


def test_probability_matches_predict_proba(variant_models):
    frame = predict.to_feature_frame([_record()])
    for variant, model in variant_models.items():
        result = explain.explain_record(_record(), model, variant=variant)
        expected = float(model.predict_proba(frame)[:, 1][0])
        assert abs(result["probability"] - expected) < 1e-6, variant


def test_contributions_reconstruct_the_logit(variant_models):
    result = explain.explain_record(_record(), variant_models["ridge_l2"])
    total = result["intercept"] + sum(f["contribution"] for f in result["features"])
    assert abs(total - result["logit"]) < 1e-5


def test_sigmoid_is_stable_at_extremes():
    assert explain.sigmoid(0) == pytest.approx(0.5)
    assert explain.sigmoid(1000) == pytest.approx(1.0)
    assert explain.sigmoid(-1000) == pytest.approx(0.0, abs=1e-9)


def test_lasso_selects_fewer_features_than_ridge(variant_models):
    lasso = explain.explain_record(_record(), variant_models["lasso_l1"], variant="lasso_l1")
    ridge = explain.explain_record(_record(), variant_models["ridge_l2"], variant="ridge_l2")
    assert lasso["n_selected"] < lasso["n_total"]
    assert lasso["n_selected"] <= ridge["n_selected"]


def test_zeroed_lasso_features_have_zero_contribution(variant_models):
    result = explain.explain_record(_record(), variant_models["lasso_l1"], variant="lasso_l1")
    dropped = [f for f in result["features"] if not f["selected"]]
    assert dropped, "expected Lasso to drop at least one coefficient"
    for feature in dropped:
        assert feature["coefficient"] == 0
        assert feature["contribution"] == 0


def test_standardised_block_has_expected_length(variant_models):
    result = explain.explain_record(_record(), variant_models["ridge_l2"])
    # 5 continuous features + the engineered age x cholesterol interaction.
    assert len(result["standardized"]) == len(config.NUMERIC_FEATURES) + 1
    names = {item["name"] for item in result["standardized"]}
    assert "age_chol" in names


def test_base_feature_mapping(variant_models):
    result = explain.explain_record(_record(), variant_models["ridge_l2"])
    for feature in result["features"]:
        assert feature["base"] in config.FEATURES or feature["base"] in {"age_band", "age_chol"}


def test_explain_patient_picks_primary_and_builds_verdict(variant_models):
    payload = explain.explain_patient(_record(), variant_models, primary="lasso_l1", threshold=0.5)
    assert payload["primary"] == "lasso_l1"
    assert set(payload["variants"]) == set(variant_models)
    verdict = payload["verdict"]
    assert verdict["prediction"] in (0, 1)
    assert verdict["label"] in ("Disease present", "No disease")
    assert 0.0 <= verdict["probability"] <= 1.0
    assert verdict["variant"] == "lasso_l1"


def test_explain_patient_falls_back_when_primary_missing(variant_models):
    payload = explain.explain_patient(_record(), variant_models, primary="does_not_exist")
    assert payload["primary"] in payload["variants"]


def test_explain_patient_requires_models():
    with pytest.raises(ValueError):
        explain.explain_patient(_record(), {})


def test_threshold_only_affects_class_not_probability(variant_models):
    low = explain.explain_record(_record(), variant_models["ridge_l2"], threshold=0.01)
    high = explain.explain_record(_record(), variant_models["ridge_l2"], threshold=0.99)
    assert low["probability"] == high["probability"]
    assert low["prediction"] >= high["prediction"]
