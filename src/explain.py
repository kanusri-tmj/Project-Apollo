"""Model introspection for the "Live Simulation" section of the web app.

This module turns a fitted pipeline plus one patient record into the *actual*
numbers behind a prediction so the frontend can animate real maths instead of
decorative fake animation:

* the standardised numeric inputs (output of ``StandardScaler``),
* every transformed feature value, its learned coefficient and their product,
* the intercept, the linear logit and the sigmoid probability,
* per-variant summaries so Ridge's dampening and Lasso's exact zeros can be
  contrasted side by side.

The arithmetic here is verified against ``predict_proba`` in the test suite, so
what the UI animates is guaranteed to be what the model actually computes.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from src import config, predict, preprocessing

# Coefficients below this magnitude are treated as exactly zero (Lasso drops
# features by driving them to 0; floating point may leave ~1e-16 residues).
ZERO_TOLERANCE = 1e-8


def sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exponential = math.exp(z)
    return exponential / (1.0 + exponential)


def _base_feature(name: str) -> str:
    """Map a one-hot column name back to its clinical feature.

    ``num_major_vessels_3.0`` -> ``num_major_vessels``; numeric columns are
    returned unchanged.
    """
    if name in config.NUMERIC_FEATURES or name in {"age_chol", "age_band"}:
        return name
    return name.rsplit("_", 1)[0]


def _raw_value(frame_row: Any, name: str) -> float | None:
    """Raw clinical value for a numeric/engineered feature, if available."""
    if name == "age_chol":
        try:
            return round(float(frame_row["age"]) * float(frame_row["cholesterol"]) / 1000.0, 4)
        except (KeyError, TypeError, ValueError):
            return None
    try:
        return round(float(frame_row[name]), 4)
    except (KeyError, TypeError, ValueError):
        return None


def explain_record(
    record: dict[str, Any],
    pipeline: Any,
    threshold: float = 0.5,
    variant: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Decompose one prediction into its exact constituent numbers."""
    frame = predict.to_feature_frame([record])

    # The nested preprocessing pipeline applies feature engineering + encoding
    # + scaling, so ``transformed`` is precisely what the classifier consumes.
    preprocess = pipeline.named_steps["preprocess"]
    transformed = np.asarray(preprocess.transform(frame), dtype=float)[0]

    names = list(preprocessing.feature_names(pipeline))
    classifier = pipeline.named_steps["classifier"]
    coefficients = np.ravel(classifier.coef_).astype(float)
    intercept = float(np.ravel(classifier.intercept_)[0])

    contributions = transformed * coefficients
    logit = float(intercept + contributions.sum())
    probability = sigmoid(logit)

    features = [
        {
            "name": name,
            "base": _base_feature(name),
            "value": round(float(value), 6),
            "coefficient": round(float(coef), 6),
            "contribution": round(float(value * coef), 6),
            "selected": bool(abs(coef) > ZERO_TOLERANCE),
        }
        for name, value, coef in zip(names, transformed, coefficients)
    ]

    # Standardised numeric block (the first N numeric outputs of the scaler).
    numeric_names = list(config.NUMERIC_FEATURES)
    if "age_chol" in names:
        numeric_names.append("age_chol")
    standardized = []
    for name in numeric_names:
        if name not in names:
            continue
        index = names.index(name)
        standardized.append(
            {
                "name": name,
                "value": round(float(transformed[index]), 6),
                "raw": _raw_value(frame.iloc[0], name),
            }
        )

    selected_count = int(sum(1 for coef in coefficients if abs(coef) > ZERO_TOLERANCE))
    return {
        "variant": variant,
        "label": label or (config.VARIANT_LABELS.get(variant) if variant else None),
        "features": features,
        "standardized": standardized,
        "intercept": round(intercept, 6),
        "logit": round(logit, 6),
        "probability": round(probability, 6),
        "prediction": int(probability >= threshold),
        "n_selected": selected_count,
        "n_total": len(coefficients),
        "n_zeroed": len(coefficients) - selected_count,
        "max_abs_coefficient": round(float(np.abs(coefficients).max()), 6) if len(coefficients) else 0.0,
        "max_abs_contribution": round(float(np.abs(contributions).max()), 6) if len(contributions) else 0.0,
    }


def explain_patient(
    record: dict[str, Any],
    models: Mapping[str, Any],
    primary: str | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Explain a patient across every available model variant.

    Parameters
    ----------
    record:
        Raw clinical measurements (the 13 features).
    models:
        Mapping of variant name -> fitted pipeline.
    primary:
        Variant whose probability drives the headline verdict. Falls back to
        ``config.PRIMARY_VARIANT`` and then to any available variant.
    """
    if not models:
        raise ValueError("No models supplied to explain_patient().")

    explanations = {
        variant: explain_record(
            record,
            pipeline,
            threshold=threshold,
            variant=variant,
            label=config.VARIANT_LABELS.get(variant, variant),
        )
        for variant, pipeline in models.items()
    }

    if primary not in explanations:
        primary = config.PRIMARY_VARIANT if config.PRIMARY_VARIANT in explanations else next(iter(explanations))

    primary_explanation = explanations[primary]
    return {
        "input": record,
        "threshold": threshold,
        "primary": primary,
        "variants": explanations,
        "verdict": {
            "prediction": primary_explanation["prediction"],
            "label": "Disease present" if primary_explanation["prediction"] == 1 else "No disease",
            "probability": primary_explanation["probability"],
            "risk_percent": round(primary_explanation["probability"] * 100, 2),
            "threshold": threshold,
            "variant": primary,
            "variant_label": config.VARIANT_LABELS.get(primary, primary),
        },
    }
