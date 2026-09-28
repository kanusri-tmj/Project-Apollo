"""Model evaluation helpers.

Computes accuracy / precision / recall / F1 / ROC-AUC plus confusion matrices,
builds the comparison table, and renders confusion-matrix and ROC-curve
figures. Also exposes coefficient inspection, which is how we demonstrate
Lasso's built-in feature selection (coefficients shrunk exactly to zero).
"""

from __future__ import annotations

from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config  # noqa: E402

sns.set_theme(style="whitegrid")

METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Return the headline classification metrics for a single model."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
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
