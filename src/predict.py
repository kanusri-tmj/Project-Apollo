"""Inference: load a saved artifact and score patients.

Shared by the Flask API and the Streamlit UI so both behave identically. The
saved pipeline already contains imputation, encoding and scaling, so callers
only ever supply the 13 **raw** clinical measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config, model_card


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained artifact can be located."""


def available_models() -> list[Path]:
    """Return every ``*_latest.joblib`` artifact found in ``models/``."""
    if not config.MODELS_DIR.exists():
        return []
    return sorted(config.MODELS_DIR.glob("*_latest.joblib"))


def default_model_path() -> Path:
    """Resolve the artifact the app should serve.

    Preference order:
    1. ``models/model_latest.joblib`` (stable deploy pointer, if present),
    2. the most recently modified ``*_latest.joblib``,
    3. any ``*.joblib`` file.
    """
    pointer = config.MODELS_DIR / "model_latest.joblib"
    if pointer.exists():
        return pointer

    candidates = available_models()
    if not candidates:
        candidates = sorted(config.MODELS_DIR.glob("*.joblib"))
    if not candidates:
        raise ModelNotFoundError(
            "No trained model found in 'models/'. Run `python -m src.train` first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(path: str | Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Load the model and its metadata info dict."""
    resolved = Path(path) if path else default_model_path()
    return model_card.load_artifact(resolved)


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems with a patient record.

    Missing optional fields fall back to population medians at scoring time,
    but out-of-range or unknown categorical codes are hard errors because they
    almost always indicate a UI/API bug.
    """
    errors: list[str] = []

    for column in config.NUMERIC_FEATURES:
        if column not in record or record[column] is None:
            continue
        try:
            value = float(record[column])
        except (TypeError, ValueError):
            errors.append(f"{column} must be numeric.")
            continue
        low, high = config.VALID_RANGES.get(column, (None, None))
        if low is not None and not (low <= value <= high):
            errors.append(f"{column}={value} is outside the plausible range {low}-{high}.")

    for column in config.CATEGORICAL_FEATURES:
        if column not in record or record[column] is None:
            continue
        allowed = set(config.CATEGORY_LEVELS.get(column, {}).keys())
        try:
            code = int(float(record[column]))
        except (TypeError, ValueError):
            errors.append(f"{column} must be an integer code.")
            continue
        if allowed and code not in allowed:
            errors.append(
                f"{column}={code} is not a valid code. Valid: {sorted(allowed)}."
            )

    return errors


def _to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a feature frame, filling absent fields with NaN for imputation.

    Every column is coerced to a numeric dtype so the imputers/encoders in the
    saved pipeline see a consistent schema even for partially-specified records.
    """
    frame = pd.DataFrame(records)
    for column in config.FEATURES:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[list(config.FEATURES)]
    for column in config.FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def to_feature_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Public wrapper around the internal feature-frame construction.

    Used by :mod:`src.explain` so model introspection sees exactly the same
    frame the scorer does.
    """
    return _to_frame(records)


def predict_batch(
    records: list[dict[str, Any]],
    model: Any | None = None,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Score a list of patient records, returning prediction dicts."""
    if not records:
        return []
    if model is None:
        model, _ = load_model()

    frame = _to_frame(records)
    probabilities = model.predict_proba(frame)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    outputs: list[dict[str, Any]] = []
    for probability, prediction in zip(probabilities, predictions):
        outputs.append(
            {
                "prediction": int(prediction),
                "label": "Disease present" if prediction == 1 else "No disease",
                "probability": round(float(probability), 4),
                "risk_percent": round(float(probability) * 100, 2),
                "threshold": threshold,
            }
        )
    return outputs


def predict_patient(
    record: dict[str, Any],
    model: Any | None = None,
    threshold: float = 0.5,
    validate: bool = True,
) -> dict[str, Any]:
    """Score a single patient and attach any validation warnings."""
    warnings = validate_record(record) if validate else []
    result = predict_batch([record], model=model, threshold=threshold)[0]
    result["warnings"] = warnings
    result["valid"] = not warnings
    return result
