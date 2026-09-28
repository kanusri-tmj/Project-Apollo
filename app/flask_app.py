"""Flask deployment: REST API + minimal clinician-facing web form.

Endpoints
---------
``GET  /``              Single-page site: Home / Live Simulation / About Us.
``POST /predict``       JSON prediction endpoint (used by the site's JS).
``POST /explain``       Real model internals for the Live Simulation section.
``POST /api/predict``   Alias for the same endpoint under a namespaced path.
``POST /api/predict/batch``  Score a list of patients in one call.
``GET  /health``        Liveness/readiness probe (also reports model version).
``GET  /model-info``    Metrics, version and expected input specification.
``GET  /metrics``       Basic in-process monitoring counters.

Run locally::

    python app/flask_app.py
    # or: flask --app app.flask_app run --port 5000

Production (Render / any WSGI host)::

    gunicorn --bind 0.0.0.0:$PORT wsgi:app

``wsgi.py`` at the project root re-exports the ``app`` object below, and the
server binds to the platform-injected ``$PORT`` (never a hardcoded 5000). See
``render.yaml`` for the deployed configuration.

Operational note: put the ``/metrics`` counters behind auth or swap them for
Prometheus instrumentation before exposing them publicly.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

# Make the project root importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from src import config, explain, predict  # noqa: E402

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("heart-disease.api")

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_variants() -> tuple[dict, dict]:
    """Load every persisted variant so the simulation can contrast Ridge/Lasso."""
    models: dict = {}
    infos: dict = {}
    for variant in config.MODEL_VARIANTS:
        path = config.MODELS_DIR / f"{variant}_latest.joblib"
        if not path.exists():
            continue
        try:
            model, info = predict.load_model(path)
            models[variant] = model
            infos[variant] = info
            logger.info("Loaded variant '%s' (v%s) from %s", variant, info.get("version"), path.name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load variant '%s': %s", variant, exc)
    return models, infos


MODELS, MODEL_INFOS = _load_variants()

# The headline verdict comes from the variant with the best test ROC-AUC.
PRIMARY_VARIANT = (
    max(MODELS, key=lambda v: MODEL_INFOS[v].get("metadata", {}).get("metrics", {}).get("roc_auc", 0.0))
    if MODELS
    else None
)
MODEL = MODELS.get(PRIMARY_VARIANT) if PRIMARY_VARIANT else None
MODEL_INFO = MODEL_INFOS.get(PRIMARY_VARIANT, {})
MODEL_VERSION = MODEL_INFO.get("version", "unknown")

if MODEL is None:  # pragma: no cover - startup guard
    logger.error("No model could be loaded. Train it first: python -m src.train")
else:
    logger.info("Serving primary variant '%s' (v%s)", PRIMARY_VARIANT, MODEL_VERSION)


# --------------------------------------------------------------------------- #
# Lightweight monitoring (in-process, per-worker)
# --------------------------------------------------------------------------- #
class Monitoring:
    """Thread-safe counters and latency tracking for basic observability."""

    def __init__(self, window: int = 200) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests_total = 0
        self.predictions_total = 0
        self.errors_total = 0
        self.positive_predictions = 0
        self.latencies_ms: deque[float] = deque(maxlen=window)

    def observe_request(self, latency_ms: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.latencies_ms.append(latency_ms)

    def observe_prediction(self, prediction: int) -> None:
        with self._lock:
            self.predictions_total += 1
            self.positive_predictions += int(prediction == 1)

    def observe_error(self) -> None:
        with self._lock:
            self.errors_total += 1

    def snapshot(self) -> dict:
        with self._lock:
            latencies = list(self.latencies_ms)
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "requests_total": self.requests_total,
            "predictions_total": self.predictions_total,
            "positive_predictions": self.positive_predictions,
            "errors_total": self.errors_total,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2)
            if len(latencies) >= 20
            else None,
            "model_version": MODEL_VERSION,
        }


monitor = Monitoring()


@app.before_request
def _start_timer() -> None:
    request._start_time = time.perf_counter()  # type: ignore[attr-defined]


@app.after_request
def _record_latency(response):
    start = getattr(request, "_start_time", None)
    if start is not None:
        latency_ms = (time.perf_counter() - start) * 1000
        monitor.observe_request(latency_ms)
        logger.info("%s %s -> %s (%.1f ms)", request.method, request.path, response.status_code, latency_ms)
    return response


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_model():
    """Return an error response when the model is unavailable."""
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Train it first: python -m src.train"}), 503
    return None


def _score(payload: dict) -> tuple[dict, int]:
    """Validate and score a single patient payload."""
    if not isinstance(payload, dict):
        return {"error": "Request body must be a JSON object."}, 400

    # Accept both {"features": {...}} and a flat record.
    record = payload.get("features", payload)
    if not isinstance(record, dict):
        return {"error": "'features' must be a JSON object."}, 400

    threshold = float(payload.get("threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        return {"error": "threshold must be between 0 and 1."}, 400

    errors = predict.validate_record(record)
    if errors:
        # Out-of-range vitals are rejected so clinicians see actionable feedback.
        return {"error": "Invalid input.", "details": errors}, 422

    result = predict.predict_patient(record, model=MODEL, threshold=threshold, validate=False)
    monitor.observe_prediction(result["prediction"])
    result["model_version"] = MODEL_VERSION
    return result, 200


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    """Clinician-facing form."""
    return render_template(
        "index.html",
        numeric_features=config.NUMERIC_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES,
        category_levels=config.CATEGORY_LEVELS,
        ranges=config.VALID_RANGES,
        model_version=MODEL_VERSION,
        available_variants=list(MODELS.keys()),
        variant_labels=config.VARIANT_LABELS,
        primary_variant=PRIMARY_VARIANT,
        metrics=MODEL_INFO.get("metadata", {}).get("metrics", {}),
    )


@app.post("/explain")
@app.post("/api/explain")
def api_explain():
    """Return the real model internals for one patient (Live Simulation)."""
    guard = _require_model()
    if guard:
        return guard

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    record = payload.get("features", payload)
    if not isinstance(record, dict):
        return jsonify({"error": "'features' must be a JSON object."}), 400

    try:
        threshold = float(payload.get("threshold", 0.5))
    except (TypeError, ValueError):
        return jsonify({"error": "threshold must be a number."}), 400
    if not 0.0 < threshold < 1.0:
        return jsonify({"error": "threshold must be between 0 and 1."}), 400

    errors = predict.validate_record(record)
    if errors:
        monitor.observe_error()
        return jsonify({"error": "Invalid input.", "details": errors}), 422

    try:
        result = explain.explain_patient(record, MODELS, primary=PRIMARY_VARIANT, threshold=threshold)
    except ValueError as exc:
        monitor.observe_error()
        return jsonify({"error": str(exc)}), 503

    result["model_version"] = MODEL_VERSION
    monitor.observe_prediction(result["verdict"]["prediction"])
    return jsonify(result)


@app.post("/predict")
@app.post("/api/predict")
def api_predict():
    """Score one patient."""
    guard = _require_model()
    if guard:
        return guard
    payload = request.get_json(silent=True) or {}
    body, status = _score(payload)
    if status >= 400:
        monitor.observe_error()
    return jsonify(body), status


@app.post("/api/predict/batch")
def api_predict_batch():
    """Score many patients in a single request."""
    guard = _require_model()
    if guard:
        return guard

    payload = request.get_json(silent=True) or {}
    records = payload.get("patients")
    if not isinstance(records, list) or not records:
        return jsonify({"error": "'patients' must be a non-empty list."}), 400

    threshold = float(payload.get("threshold", 0.5))
    results = predict.predict_batch(records, model=MODEL, threshold=threshold)
    for result in results:
        monitor.observe_prediction(result["prediction"])
    return jsonify({"count": len(results), "results": results, "model_version": MODEL_VERSION})


@app.get("/health")
def health():
    """Kubernetes-style probe."""
    return jsonify(
        {
            "status": "ok" if MODEL is not None else "degraded",
            "model_loaded": MODEL is not None,
            "model_version": MODEL_VERSION,
        }
    ), (200 if MODEL is not None else 503)


@app.get("/model-info")
def model_info():
    """Expose the model card metadata so clients can render dynamic forms."""
    return jsonify(
        {
            "version": MODEL_VERSION,
            "name": MODEL_INFO.get("name"),
            "created_utc": MODEL_INFO.get("created_utc"),
            "metrics": MODEL_INFO.get("metadata", {}).get("metrics", {}),
            "features": config.FEATURES,
            "input_spec": MODEL_INFO.get("metadata", {}).get("input_spec", []),
            "primary_variant": PRIMARY_VARIANT,
            "available_variants": {
                variant: MODEL_INFOS[variant].get("metadata", {}).get("metrics", {})
                for variant in MODELS
            },
        }
    )


@app.get("/metrics")
def metrics():
    """Basic monitoring counters (see module docstring for production notes)."""
    return jsonify(monitor.snapshot())


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(500)
def server_error(_):  # pragma: no cover
    monitor.observe_error()
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    # Local development only: the Flask development server. In production the
    # app is served by gunicorn through ``wsgi.py`` (see render.yaml).
    # PORT is honoured so the container/host can pick the port, defaulting to
    # 5000 for local runs.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
