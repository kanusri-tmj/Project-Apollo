"""Model evaluation helpers.

Three layers of assessment, in increasing order of rigour:

1. **Headline metrics** — accuracy / precision / recall / specificity / F1 / MCC /
   ROC-AUC, confusion matrices, the comparison table, and the confusion-matrix and
   ROC-curve figures.
2. **Statistical robustness** — bootstrap confidence intervals for every metric,
   so a difference between two models is only reported as a difference when the
   intervals actually separate.
3. **Probability quality** — Brier score, log loss, calibration slope/intercept and
   reliability curves for the raw model against Platt (sigmoid) and isotonic
   recalibration.

Plus coefficient inspection and permutation importance, which together show
Lasso's built-in feature selection (coefficients shrunk exactly to zero) is
borne out by an independent, model-agnostic measure.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config  # noqa: E402

sns.set_theme(style="whitegrid")

METRIC_ORDER = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "mcc",
    "roc_auc",
]

# Metric name -> callable(y_true, y_score). Every entry is a function of the
# *probability*, so the same table can drive both point estimates and bootstrap
# resampling without recomputing predictions per resample.
BOOTSTRAP_METRICS = {
    "accuracy": lambda yt, ys: accuracy_score(yt, (ys >= 0.5).astype(int)),
    "precision": lambda yt, ys: precision_score(yt, (ys >= 0.5).astype(int), zero_division=0),
    "recall": lambda yt, ys: recall_score(yt, (ys >= 0.5).astype(int), zero_division=0),
    "specificity": lambda yt, ys: recall_score(1 - np.asarray(yt), 1 - (ys >= 0.5).astype(int)),
    "f1": lambda yt, ys: f1_score(yt, (ys >= 0.5).astype(int), zero_division=0),
    "mcc": lambda yt, ys: matthews_corrcoef(yt, (ys >= 0.5).astype(int)),
    "roc_auc": lambda yt, ys: roc_auc_score(yt, ys),
    "brier": lambda yt, ys: brier_score_loss(yt, ys),
}

# Palette (matches the web UI) so the figures and the site read as one product.
RED = "#d81f36"
BLUE = "#1d4ed8"
INK = "#16191f"
GREY = "#8a93a3"


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Return the headline classification metrics for a single model.

    ``specificity`` is included because for a screening aid the false-positive
    rate matters as much as recall: it is what a clinician pays in unnecessary
    follow-up for every case caught.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict[str, object]:
    """Score a fitted model on the held-out test set."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["threshold"] = threshold
    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def cross_validation_scores(cv_results: Mapping[str, float]) -> dict[str, float]:
    """Best CV score and the corresponding hyper-parameter from a ``cv_results_``."""
    best = int(np.argmax(cv_results["mean_test_score"]))
    return {
        "cv_best_score": float(cv_results["mean_test_score"][best]),
        "cv_best_std": float(cv_results["std_test_score"][best]),
        "cv_best_params": cv_results["params"][best],
        "cv_best_rank": int(cv_results["rank_test_score"][best]),
    }


def build_comparison_table(results: Mapping[str, dict]) -> pd.DataFrame:
    """Assemble a model-by-metric comparison table, best value per column bolded."""
    rows = {}
    for name, result in results.items():
        row = {metric: round(result["metrics"][metric], 4) for metric in METRIC_ORDER}
        row["best_C"] = result.get("best_params", {}).get("classifier__C", "n/a")
        row["cv_roc_auc"] = round(result.get("cv_best_score", float("nan")), 4)
        rows[name] = row
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    return table.reset_index()


def highlight_best(table: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Return a Styler that highlights the best value in each metric column."""
    numeric_cols = [c for c in METRIC_ORDER if c in table.columns]

    def _best(s: pd.Series) -> list[str]:
        is_best = s == s.max()
        return ["background-color: #c6efce; font-weight: bold" if v else "" for v in is_best]

    styler = table.style
    try:
        return styler.apply(_best, subset=numeric_cols)
    except Exception:  # pragma: no cover - depends on optional jinja2
        return styler


def coefficient_table(model, feature_names: list[str]) -> pd.DataFrame:
    """Coefficients of the final classifier, sorted by absolute magnitude.

    With L1 (Lasso) regularisation many coefficients are exactly zero — this
    table makes the implicit feature selection visible.
    """
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    coefs = np.ravel(classifier.coef_)
    table = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    table["selected"] = table["coefficient"].abs() > 1e-8
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def plot_confusion_matrices(results: Mapping[str, dict]) -> str:
    """Grid of confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    axes = np.atleast_1d(axes)
    for ax, (name, result) in zip(axes, results.items()):
        sns.heatmap(
            result["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["No disease", "Disease"],
            yticklabels=["No disease", "Disease"],
        )
        ax.set_title(f"{name}\n(acc={result['metrics']['accuracy']:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / "confusion_matrices.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_roc_curves(results: Mapping[str, dict], y_test) -> str:
    """Overlaid ROC curves with AUC in the legend."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
        auc = result["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — baseline vs Ridge vs Lasso")
    ax.legend(loc="lower right")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / "roc_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_lasso_path(path: pd.DataFrame) -> str:
    """Number of non-zero Lasso coefficients as a function of ``C``."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogx(path["C"], path["n_selected"], marker="o", linewidth=2, color="darkorange")
    ax.set_xlabel("C (inverse regularisation strength, log scale)")
    ax.set_ylabel("Features with non-zero coefficient")
    ax.set_title("Lasso (L1) embedded feature selection path")
    ax.grid(True, which="both", alpha=0.3)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "lasso_feature_path.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def threshold_sensitivity(model, X_test, y_test, thresholds=None) -> pd.DataFrame:
    """Recall/precision trade-off across decision thresholds (clinical utility)."""
    thresholds = thresholds if thresholds is not None else np.arange(0.1, 0.91, 0.1)
    proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": round(precision_score(y_test, pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, pred, zero_division=0), 4),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Statistical robustness: bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_metric_ci(
    y_true,
    y_proba,
    metric_fn: Callable,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = config.RANDOM_STATE,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a single metric.

    The held-out set holds ~61 patients, so a point estimate such as AUC 0.9654
    is printed with far more precision than the data can support. Resampling with
    replacement shows the range the metric would plausibly take on another draw
    from the same population, which is what makes model comparisons honest.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    point = float(metric_fn(y_true, y_proba))

    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: list[float] = []
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        resampled = y_true[index]
        # A resample containing a single class cannot score AUC / precision / etc.
        if len(np.unique(resampled)) < 2:
            continue
        samples.append(float(metric_fn(resampled, y_proba[index])))

    if not samples:  # pragma: no cover - degenerate input
        return point, float("nan"), float("nan")
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(low), float(high)


def metric_confidence_intervals(
    results: Mapping[str, dict],
    y_test,
    metrics: Mapping[str, Callable] | None = None,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Bootstrap CIs for every model x metric, as a tidy frame.

    One row per (model, metric): point estimate, interval, and interval width.
    """
    metrics = metrics or BOOTSTRAP_METRICS
    rows = []
    for name, result in results.items():
        for metric, fn in metrics.items():
            point, low, high = bootstrap_metric_ci(y_test, result["y_proba"], fn, n_boot=n_boot)
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "value": round(point, 4),
                    "ci_low": round(low, 4),
                    "ci_high": round(high, 4),
                    "ci_width": round(high - low, 4),
                }
            )
    return pd.DataFrame(rows)


def intervals_overlap(ci_table: pd.DataFrame, model_a: str, model_b: str, metric: str) -> bool:
    """True when two models' CIs for ``metric`` overlap, i.e. are not separable."""
    subset = ci_table[ci_table["metric"] == metric].set_index("model")
    if model_a not in subset.index or model_b not in subset.index:
        return True
    a, b = subset.loc[model_a], subset.loc[model_b]
    return not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])


def plot_metric_confidence_intervals(
    ci_table: pd.DataFrame,
    metrics: Sequence[str] = ("accuracy", "precision", "recall", "specificity", "roc_auc", "brier"),
) -> str:
    """Dot-and-whisker panels: each metric per model, with its 95% interval."""
    metrics = [m for m in metrics if m in set(ci_table["metric"])]
    models = list(dict.fromkeys(ci_table["model"]))
    ncols = 3
    nrows = max(1, int(np.ceil(len(metrics) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    ypos = np.arange(len(models))

    for ax, metric in zip(axes, metrics):
        subset = ci_table[ci_table["metric"] == metric].set_index("model").loc[models]
        value = subset["value"].to_numpy(dtype=float)
        low = subset["ci_low"].to_numpy(dtype=float)
        high = subset["ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            value,
            ypos,
            xerr=[value - low, high - value],
            fmt="o",
            color=RED,
            ecolor=GREY,
            capsize=4,
            markersize=7,
            linewidth=1.6,
        )
        ax.set_yticks(ypos)
        ax.set_yticklabels(models)
        ax.set_title(metric)
        ax.grid(True, axis="x", alpha=0.3)

    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("95% bootstrap confidence intervals (2000 resamples of the held-out set)")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / "metric_confidence_intervals.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# Probability quality: Brier score, log loss and calibration
# --------------------------------------------------------------------------- #
def calibration_metrics(y_true, y_proba) -> dict[str, float]:
    """Brier score, log loss, and the Cox calibration slope / intercept.

    A perfectly calibrated model has slope 1 and intercept 0. Slope below 1 means
    predictions are too extreme (over-confident); an intercept away from 0 means
    they are systematically too high or too low.
    """
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(y_proba, dtype=float), 1e-6, 1 - 1e-6)
    log_odds = np.log(proba / (1 - proba)).reshape(-1, 1)

    # Unpenalised logistic regression of the outcome on the model's own log-odds.
    probe = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    probe.fit(log_odds, y_true)
    return {
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba)),
        "calibration_slope": float(probe.coef_[0, 0]),
        "calibration_intercept": float(probe.intercept_[0]),
    }


def compare_calibration(
    build_model: Callable,
    X_train,
    y_train,
    X_test,
    y_test,
    variant: str,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Compare the raw model against Platt (sigmoid) and isotonic recalibration.

    ``build_model`` is a zero-argument callable returning a *fresh, unfitted*
    pipeline; it is injected by :mod:`src.train` so this module does not import
    ``train`` (which would be circular). Both recalibrators fit inside
    cross-validation on the training split only, so the test set stays untouched
    and the comparison remains honest.
    """
    probabilities: dict[str, np.ndarray] = {
        "uncalibrated": build_model().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    }
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(build_model(), method=method, cv=5)
        calibrated.fit(X_train, y_train)
        probabilities[method] = calibrated.predict_proba(X_test)[:, 1]

    rows, curves = [], {}
    for name, proba in probabilities.items():
        rows.append({"variant": variant, "calibration": name, **calibration_metrics(y_test, proba)})
        # Quantile bins: with ~61 test patients, equal-width bins can come out empty.
        observed, predicted = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
        curves[name] = (predicted, observed)
    return pd.DataFrame(rows), curves


def plot_calibration_curves(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    table: pd.DataFrame,
    variant: str,
) -> str:
    """Reliability diagram: raw vs Platt vs isotonic against the ideal diagonal."""
    style = {
        "uncalibrated": (INK, "Uncalibrated"),
        "sigmoid": (RED, "Platt scaling (sigmoid)"),
        "isotonic": (BLUE, "Isotonic"),
    }
    brier = table.set_index("calibration")["brier"]

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1.2, label="Perfect calibration")
    for name, (predicted, observed) in curves.items():
        colour, label = style.get(name, (INK, name))
        ax.plot(
            predicted,
            observed,
            marker="o",
            linewidth=2,
            color=colour,
            label=f"{label} (Brier={brier.get(name, float('nan')):.3f})",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration (reliability) — {variant}")
    ax.legend(loc="upper left", fontsize=9)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / "calibration_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# Independence check on the coefficients: permutation importance
# --------------------------------------------------------------------------- #
def permutation_importance_report(
    results: Mapping[str, dict],
    X_test,
    y_test,
    n_repeats: int = 30,
) -> pd.DataFrame:
    """Permutation importance on the *raw* clinical features, per model.

    Coefficient magnitude is not comparable across variants (they shrink on
    different scales) and describes the encoded feature space. Shuffling a raw
    column and re-scoring asks the model-agnostic question instead: how much does
    ROC-AUC actually depend on this measurement? For Lasso it is also an
    independent check on which features it dropped.
    """
    rows = []
    for name, result in results.items():
        outcome = permutation_importance(
            result["model"],
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=config.RANDOM_STATE,
            n_jobs=1,
        )
        for feature, mean, std in zip(
            X_test.columns, outcome.importances_mean, outcome.importances_std
        ):
            rows.append(
                {
                    "model": name,
                    "feature": feature,
                    "importance_mean": round(float(mean), 5),
                    "importance_std": round(float(std), 5),
                }
            )
    return pd.DataFrame(rows)


def plot_permutation_importance(importance: pd.DataFrame, order_by: str) -> str:
    """Grouped horizontal bars, features ordered by one model's importance."""
    table = importance.pivot(index="feature", columns="model", values="importance_mean")
    spread = importance.pivot(index="feature", columns="model", values="importance_std")
    if order_by in table:
        order = table[order_by].sort_values().index
    else:  # pragma: no cover - defensive
        order = table.max(axis=1).sort_values().index
    table, spread = table.loc[order], spread.loc[order]

    models = list(table.columns)
    colours = [GREY, RED, BLUE]
    # Cycle the palette rather than indexing it: the plot must still work if a
    # variant is ever added to or removed from config.MODEL_VARIANTS.
    palette = {model: colours[i % len(colours)] for i, model in enumerate(models)}
    height = 0.8 / len(models)
    ypos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    for offset, model in enumerate(models):
        ax.barh(
            ypos + (offset - (len(models) - 1) / 2) * height,
            table[model].to_numpy(dtype=float),
            height=height,
            xerr=spread[model].to_numpy(dtype=float),
            color=palette.get(model, GREY),
            label=model,
            error_kw={"elinewidth": 0.9, "ecolor": GREY, "capsize": 2},
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(order)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("Permutation importance — repeated shuffles of the test set")
    ax.legend(loc="lower right")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / "permutation_importance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)
