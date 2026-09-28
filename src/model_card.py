"""Versioned model persistence and model-card generation.

Every training run produces

* a timestamped, versioned ``.joblib`` artifact (immutable audit trail),
* a stable ``*_latest.joblib`` pointer used by the deployment app,
* ``model_metadata.json`` with everything needed to serve the model,
* ``MODEL_CARD.md`` — a short human-readable card (metrics + expected inputs).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src import config


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def versioned_filename(name: str, version: str = config.MODEL_VERSION) -> str:
    """Build the immutable artifact filename, e.g. ``ridge_v1.0.0_20260926T1015Z``."""
    return f"{name}_v{version}_{_timestamp()}.joblib"


def save_artifact(
    pipeline: Any,
    name: str,
    metadata: dict[str, Any],
    version: str = config.MODEL_VERSION,
) -> dict[str, Path]:
    """Persist a fitted pipeline together with its metadata.

    Returns a mapping with the versioned path, the ``latest`` pointer path and
    the metadata JSON path.
    """
    config.ensure_directories()

    versioned_path = config.MODELS_DIR / versioned_filename(name, version)
    latest_path = config.MODELS_DIR / f"{name}_latest.joblib"

    payload = {
        "name": name,
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
    }

    # The artifact carries a small header so the app can sanity-check it.
    joblib.dump({"model": pipeline, "info": payload}, versioned_path)
    joblib.dump({"model": pipeline, "info": payload}, latest_path)

    metadata_path = config.MODELS_DIR / f"{name}_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"[model_card] Saved versioned artifact -> {versioned_path.name}")
    print(f"[model_card] Updated latest pointer   -> {latest_path.name}")
    return {
        "versioned_path": versioned_path,
        "latest_path": latest_path,
        "metadata_path": metadata_path,
    }


def load_artifact(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a saved artifact, returning ``(pipeline, info)``.

    Accepts either the ``{model, info}`` wrapper written by :func:`save_artifact`
    or a bare pickled estimator (backwards compatibility).
    """
    payload = joblib.load(Path(path))
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("info", {})
    return payload, {}


def render_model_card(
    name: str,
    version: str,
    metrics: dict[str, float],
    best_params: dict[str, Any],
    feature_list: list[str],
    input_spec: list[dict[str, Any]],
    notes: str = "",
    versioned_path: Path | None = None,
) -> str:
    """Render the Markdown model card body."""
    metric_lines = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())
    params = ", ".join(f"{k}={v}" for k, v in best_params.items()) or "default"
    features = ", ".join(feature_list)
    input_rows = "\n".join(
        f"| {spec['name']} | {spec['type']} | {spec.get('range', spec.get('options', ''))} | {spec['description']} |"
        for spec in input_spec
    )

    return f"""# Model Card — Heart Disease Risk (Logistic Regression, {name})

**Version:** `{version}`
**Artifact:** `{versioned_path.name if versioned_path else 'n/a'}`
**Generated:** {datetime.now(timezone.utc).isoformat()}

## Intended use
Decision-support for clinicians: estimate the probability that a patient has
coronary artery disease from routine, non-invasive clinical measurements.
**Not a diagnostic device.** Outputs must be interpreted by a qualified
clinician alongside the full clinical picture.

## Model
- Algorithm: **{name}** — logistic regression with {'L2 (Ridge)' if 'ridge' in name.lower() else 'L1 (Lasso)' if 'lasso' in name.lower() else 'no'} penalty.
- Selected hyper-parameters: `{params}`
- Preprocessing: median/mode imputation, one-hot encoding, standard scaling, optional engineered features.
- Trained on: UCI Cleveland Heart Disease dataset (303 records), binary target.

## Performance (held-out test set, 20% stratified split)
| Metric | Value |
| --- | --- |
{metric_lines}

## Inputs
| Field | Type | Range / options | Description |
| --- | --- | --- | --- |
{input_rows}

## Features used
{features}

## Limitations & ethical notes
- Trained on a small historical cohort (Cleveland, 1988) — not demographically representative.
- Risk probabilities are not calibrated to any local patient population.
- {notes or 'Do not use as the sole basis for any clinical decision.'}
"""
