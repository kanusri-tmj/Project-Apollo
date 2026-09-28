"""Tests for cleaning, validation, feature engineering and the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, preprocessing


def test_clean_dataframe_drops_duplicates(synthetic_frame):
    duplicated = pd.concat([synthetic_frame, synthetic_frame.iloc[[0, 1]]], ignore_index=True)
    cleaned = preprocessing.clean_dataframe(duplicated)
    assert len(cleaned) == len(synthetic_frame)


def test_report_missing_flags_injected_nans(synthetic_frame):
    report = preprocessing.report_missing(synthetic_frame)
    assert "num_major_vessels" in report.index
    assert "thalassemia" in report.index


def test_detect_outliers_iqr_finds_extreme_value():
    series = pd.Series([1.0] * 20 + [1000.0])
    mask = preprocessing.detect_outliers_iqr(series)
    assert mask.sum() == 1
    assert bool(mask.iloc[-1]) is True


def test_detect_outliers_zscore_threshold():
    series = pd.Series(np.r_[np.zeros(50), 50.0])
    mask = preprocessing.detect_outliers_zscore(series, threshold=3.0)
    assert mask.sum() == 1


def test_validate_ranges_flags_impossible_values(synthetic_frame):
    frame = synthetic_frame.copy()
    frame.loc[0, "cholesterol"] = 1000.0  # above the plausible maximum
    report = preprocessing.validate_ranges(frame)
    row = report.set_index("column").loc["cholesterol"]
    assert row["above_range"] == 1
    assert row["valid"] is np.False_ or row["valid"] == False  # noqa: E712


def test_engineer_adds_expected_columns(synthetic_frame):
    engineer = preprocessing.ClinicalFeatureEngineer(enable=True)
    out = engineer.fit_transform(synthetic_frame[config.FEATURES])
    assert "age_band" in out.columns
    assert "age_chol" in out.columns
    assert len(out) == len(synthetic_frame)


def test_engineer_disabled_is_passthrough(synthetic_frame):
    engineer = preprocessing.ClinicalFeatureEngineer(enable=False)
    out = engineer.fit_transform(synthetic_frame[config.FEATURES])
    assert list(out.columns) == config.FEATURES


def test_pipeline_output_is_finite_and_named(synthetic_frame):
    pipeline = preprocessing.build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(synthetic_frame[config.FEATURES])
    assert transformed.shape[0] == len(synthetic_frame)
    assert np.isfinite(transformed).all(), "imputers should leave no NaNs"
    names = preprocessing.feature_names(pipeline)
    assert len(names) == transformed.shape[1]


def test_numeric_features_are_standardised(synthetic_frame):
    pipeline = preprocessing.build_preprocessing_pipeline(engineer=False)
    transformed = pipeline.fit_transform(synthetic_frame[config.FEATURES])
    names = list(preprocessing.feature_names(pipeline))
    # The first len(NUMERIC_FEATURES) columns are the scaled numeric block.
    numeric_block = transformed[:, : len(config.NUMERIC_FEATURES)]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-6)


def test_correlation_helpers(synthetic_frame):
    corr = preprocessing.feature_target_correlation(synthetic_frame)
    assert set(corr.index) == set(config.FEATURES)
    # No engineered features are perfectly collinear in the raw schema.
    pairs = preprocessing.correlation_redundancy(synthetic_frame, threshold=0.99)
    assert pairs == []


def test_drop_redundant_features_removes_collinear_column(synthetic_frame):
    frame = synthetic_frame.copy()
    frame["chol_copy"] = frame["cholesterol"]  # perfectly correlated
    _, dropped = preprocessing.drop_redundant_features(frame, threshold=0.99)
    assert "chol_copy" in dropped or "cholesterol" in dropped
