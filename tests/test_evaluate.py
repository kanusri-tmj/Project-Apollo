"""Tests for the evaluation diagnostics.

Covers the three layers added on top of the headline metrics:

* specificity / MCC in :func:`evaluate.compute_metrics`,
* bootstrap confidence intervals (and their reproducibility),
* calibration metrics and the raw-vs-Platt-vs-isotonic comparison,
* permutation importance, including that it actually finds a planted signal.

The tests are deliberately fast: they run on the shared synthetic frame or on
tiny toy frames rather than the real cohort, so they need no trained artifact.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src import config, evaluate, preprocessing
from src.train import build_model_pipeline


# --------------------------------------------------------------------------- #
# Headline metrics: specificity and MCC
# --------------------------------------------------------------------------- #
def test_compute_metrics_reports_specificity_and_mcc():
    # 8 patients: 3 false positives -> specificity = 5/8.
    y_true = [0] * 8 + [1] * 4
    y_pred = [0] * 5 + [1] * 3 + [1] * 4
    y_proba = [0.1, 0.1, 0.2, 0.3, 0.2, 0.9, 0.8, 0.7, 0.8, 0.7, 0.6, 0.9]

    metrics = evaluate.compute_metrics(y_true, y_pred, y_proba)

    assert metrics["specificity"] == pytest.approx(5 / 8)
    assert metrics["recall"] == pytest.approx(1.0)
    assert -1.0 <= metrics["mcc"] <= 1.0
    assert set(evaluate.METRIC_ORDER) <= set(metrics)


def test_compute_metrics_specificity_is_zero_when_no_negatives():
    """Guards the division guard: an all-positive test set must not raise."""
    with warnings.catch_warnings():
        # Degenerate input: sklearn quite rightly warns that AUC/MCC are
        # undefined with one class. The behaviour under test is the 0.0 fallback.
        warnings.simplefilter("ignore")
        metrics = evaluate.compute_metrics([1, 1, 1], [1, 1, 1], [0.6, 0.7, 0.8])
    assert metrics["specificity"] == 0.0


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def _toy_results():
    """Two models with clearly different separation on a toy problem."""
    rng = np.random.default_rng(0)
    n = 120
    y_true = rng.integers(0, 2, n)
    strong = np.where(y_true == 1, rng.uniform(0.6, 0.95, n), rng.uniform(0.05, 0.4, n))
    weak = np.where(y_true == 1, rng.uniform(0.45, 0.85, n), rng.uniform(0.15, 0.55, n))
    return y_true, {"strong": {"y_proba": strong}, "weak": {"y_proba": weak}}


def test_bootstrap_ci_brackets_the_point_estimate():
    y_true, results = _toy_results()
    point, low, high = evaluate.bootstrap_metric_ci(
        y_true, results["strong"]["y_proba"], evaluate.BOOTSTRAP_METRICS["roc_auc"], n_boot=300
    )
    assert low <= point <= high
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_ci_is_reproducible():
    """The project promises reproducibility; the CI must not move between runs."""
    y_true, results = _toy_results()
    args = (y_true, results["strong"]["y_proba"], evaluate.BOOTSTRAP_METRICS["roc_auc"])
    assert evaluate.bootstrap_metric_ci(*args, n_boot=200) == evaluate.bootstrap_metric_ci(*args, n_boot=200)


def test_metric_confidence_intervals_covers_every_model_and_metric():
    y_true, results = _toy_results()
    table = evaluate.metric_confidence_intervals(results, y_true, n_boot=100)
    assert len(table) == len(results) * len(evaluate.BOOTSTRAP_METRICS)
    assert set(table["model"]) == set(results)
    assert (table["ci_low"] <= table["ci_high"]).all()
    assert (table["ci_width"] >= 0).all()


def test_intervals_overlap_detects_separation():
    table = pd.DataFrame(
        [
            {"model": "a", "metric": "roc_auc", "value": 0.60, "ci_low": 0.50, "ci_high": 0.70, "ci_width": 0.20},
            {"model": "b", "metric": "roc_auc", "value": 0.95, "ci_low": 0.90, "ci_high": 1.00, "ci_width": 0.10},
        ]
    )
    assert evaluate.intervals_overlap(table, "a", "b", "roc_auc") is False

    overlapping = table.copy()
    overlapping.loc[1, ["ci_low", "ci_high"]] = [0.65, 0.90]
    assert evaluate.intervals_overlap(overlapping, "a", "b", "roc_auc") is True

    # A model missing from the table is treated as inseparable rather than crashing.
    assert evaluate.intervals_overlap(table, "a", "missing", "roc_auc") is True


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_calibration_metrics_penalise_confident_mistakes():
    y_true = [0, 1, 0, 1]
    calibrated = evaluate.calibration_metrics(y_true, [0.1, 0.9, 0.05, 0.8])
    overconfident_wrong = evaluate.calibration_metrics(y_true, [0.99, 0.02, 0.99, 0.01])

    assert calibrated["brier"] < overconfident_wrong["brier"]
    assert calibrated["log_loss"] < overconfident_wrong["log_loss"]
    for value in calibrated.values():
        assert np.isfinite(value)


def test_compare_calibration_returns_all_three_variants(synthetic_frame):
    X, y = preprocessing.split_features_target(synthetic_frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=config.RANDOM_STATE, stratify=y
    )
    table, curves = evaluate.compare_calibration(
        lambda: build_model_pipeline("ridge_l2", C=1.0),
        X_train,
        y_train,
        X_test,
        y_test,
        variant="ridge_l2",
    )

    assert set(table["calibration"]) == {"uncalibrated", "sigmoid", "isotonic"}
    assert (table["brier"].between(0, 1)).all()
    assert (table["log_loss"] >= 0).all()
    for method, (predicted, observed) in curves.items():
        assert method in {"uncalibrated", "sigmoid", "isotonic"}
        assert len(predicted) == len(observed) > 0


# --------------------------------------------------------------------------- #
# Permutation importance
# --------------------------------------------------------------------------- #
def test_permutation_importance_finds_a_planted_signal():
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {"signal": signal, "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n)}
    )
    y = pd.Series((signal + rng.normal(scale=0.3, size=n) > 0).astype(int))
    model = LogisticRegression(max_iter=1000).fit(X, y)

    importance = evaluate.permutation_importance_report({"toy": {"model": model}}, X, y, n_repeats=10)

    assert len(importance) == len(X.columns)
    best = importance.sort_values("importance_mean", ascending=False).iloc[0]
    assert best["feature"] == "signal"
    assert best["importance_mean"] > 0


def test_permutation_importance_is_reproducible(fitted_pipeline, synthetic_frame):
    """Fixed random_state must make the diagnostics repeatable."""
    X, y = preprocessing.split_features_target(synthetic_frame)
    X_test, y_test = X.iloc[:40], y.iloc[:40]
    results = {"ridge_l2": {"model": fitted_pipeline}}

    first = evaluate.permutation_importance_report(results, X_test, y_test, n_repeats=5)
    second = evaluate.permutation_importance_report(results, X_test, y_test, n_repeats=5)
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- #
# Figures are produced (a plotting bug would otherwise only surface on a
# full training run)
# --------------------------------------------------------------------------- #
def test_diagnostic_figures_are_written(tmp_path, monkeypatch):
    """Figures must render — and must not clobber the real report figures.

    ``FIGURES_DIR`` is redirected to a temp directory so running the suite never
    overwrites the diagrams a training run produced in ``reports/figures``.
    """
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)

    y_true, results = _toy_results()
    ci_table = evaluate.metric_confidence_intervals(results, y_true, n_boot=50)

    importance = pd.DataFrame(
        [
            {"model": model, "feature": feature, "importance_mean": value, "importance_std": 0.01}
            for model in ("a", "b")
            for feature, value in (("age", 0.12), ("sex", 0.04))
        ]
    )
    curves = {
        "uncalibrated": (np.array([0.2, 0.6]), np.array([0.1, 0.7])),
        "sigmoid": (np.array([0.2, 0.6]), np.array([0.2, 0.6])),
    }
    calibration = pd.DataFrame(
        [
            {"variant": "a", "calibration": name, "brier": 0.1, "log_loss": 0.3,
             "calibration_slope": 1.0, "calibration_intercept": 0.0}
            for name in curves
        ]
    )

    produced = [
        evaluate.plot_metric_confidence_intervals(ci_table),
        evaluate.plot_calibration_curves(curves, calibration, "a"),
        evaluate.plot_permutation_importance(importance, order_by="a"),
    ]

    names = {"metric_confidence_intervals.png", "calibration_curves.png", "permutation_importance.png"}
    assert {Path(p).name for p in produced} == names
    for path in produced:
        figure = Path(path)
        assert figure.parent == tmp_path
        assert figure.exists() and figure.stat().st_size > 0
