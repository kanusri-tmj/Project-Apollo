"""Shared pytest fixtures.

Tests avoid depending on a pre-trained artifact where possible: where a model
is required it is trained in-memory on a small synthetic dataset so the suite
runs fast and offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, preprocessing


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    """A small synthetic frame with the project's schema and both classes."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = 120
    frame = pd.DataFrame(
        {
            "age": rng.integers(29, 78, n),
            "sex": rng.integers(0, 2, n),
            "chest_pain_type": rng.integers(1, 5, n),
            "resting_bp": rng.integers(94, 200, n),
            "cholesterol": rng.integers(126, 420, n),
            "fasting_blood_sugar": rng.integers(0, 2, n),
            "resting_ecg": rng.integers(0, 3, n),
            "max_heart_rate": rng.integers(71, 202, n),
            "exercise_angina": rng.integers(0, 2, n),
            "st_depression": np.round(rng.uniform(0, 6.2, n), 1),
            "st_slope": rng.integers(1, 4, n),
            "num_major_vessels": rng.integers(0, 4, n),
            "thalassemia": rng.choice([3, 6, 7], n),
            "target": rng.integers(0, 2, n),
        }
    )
    # Inject a couple of missing values to exercise the imputers.
    frame.loc[0, "num_major_vessels"] = np.nan
    frame.loc[1, "thalassemia"] = np.nan
    return frame


@pytest.fixture(scope="session")
def fitted_pipeline(synthetic_frame: pd.DataFrame):
    """A preprocessing + logistic-regression pipeline fitted on synthetic data."""
    from src.train import build_model_pipeline

    X, y = preprocessing.split_features_target(synthetic_frame)
    model = build_model_pipeline("ridge_l2", C=1.0)
    model.fit(X, y)
    return model
