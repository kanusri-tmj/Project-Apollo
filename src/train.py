"""Model training, cross-validated tuning and persistence.

Trains three logistic-regression variants on the *same* stratified split:

===================  =========================================================
Variant              Purpose
===================  =========================================================
``baseline``         Unpenalised logistic regression — the reference point.
``ridge_l2``         L2 penalty: shrinks coefficients, keeps all features.
``lasso_l1``         L1 penalty: drives weak coefficients to exactly zero,
                     acting as a built-in feature selector.
===================  =========================================================

Regularisation strength is tuned with stratified k-fold cross-validation
(ROC-AUC). The best model by test ROC-AUC is serialised with a model card.

Run end-to-end with::

    python -m src.train
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src import config, data_loader, eda, evaluate, model_card, preprocessing

# Cross-validation folds occasionally omit a rare category (e.g. a thalassemia
# code), which makes OneHotEncoder warn about unknown categories. Because we set
# ``handle_unknown='ignore'`` the behaviour is correct and the warning is just
# noise across dozens of fold fits, so silence only that message.
warnings.filterwarnings(
    "ignore", message="Found unknown categories", category=UserWarning
)

# Human-readable variant names used throughout the reports (see src/config.py).
VARIANTS = config.MODEL_VARIANTS

VARIANT_LABELS = config.VARIANT_LABELS


def make_classifier(variant: str, C: float = 1.0) -> LogisticRegression:
    """Construct the classifier for a given variant."""
    if variant == "baseline":
        # No penalty at all — C is irrelevant, kept for API symmetry.
        return LogisticRegression(penalty=None, max_iter=5000, random_state=config.RANDOM_STATE)
    if variant == "ridge_l2":
        return LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=5000, random_state=config.RANDOM_STATE
        )
    if variant == "lasso_l1":
        return LogisticRegression(
            penalty="l1", C=C, solver="liblinear", max_iter=5000, random_state=config.RANDOM_STATE
        )
    raise ValueError(f"Unknown variant: {variant!r}")


def build_model_pipeline(variant: str, C: float = 1.0) -> Pipeline:
    """Preprocessing pipeline + classifier for one variant."""
    return Pipeline(
        [
            ("preprocess", preprocessing.build_preprocessing_pipeline()),
            ("classifier", make_classifier(variant, C)),
        ]
    )


def tune_variant(
    variant: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: config.TrainingConfig | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Cross-validate the regularisation strength ``C`` for a penalised variant."""
    cfg = cfg or config.TrainingConfig()
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    if variant == "baseline":
        # Nothing to tune; fit directly and report the CV score for the record.
        model = build_model_pipeline(variant)
        scores = _cross_val_score(model, X_train, y_train, cv, cfg.scoring, cfg.n_jobs)
        model.fit(X_train, y_train)
        return model, {
            "cv_best_score": float(np.mean(scores)),
            "cv_best_std": float(np.std(scores)),
            "cv_best_params": {},
            "cv_best_rank": 1,
            "cv_results": None,
        }

    search = GridSearchCV(
        estimator=build_model_pipeline(variant),
        param_grid={"classifier__C": cfg.c_grid},
        scoring=cfg.scoring,
        cv=cv,
        n_jobs=cfg.n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)
    info = evaluate.cross_validation_scores(search.cv_results_)
    info["cv_results"] = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, info


def _cross_val_score(model, X, y, cv, scoring, n_jobs: int = 1) -> np.ndarray:
    from sklearn.model_selection import cross_val_score

    return cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)


def train_all_variants(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: config.TrainingConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Train, tune and evaluate every variant on the shared split."""
    results: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        print(f"[train] Training '{variant}' ...")
        model, cv_info = tune_variant(variant, X_train, y_train, cfg)
        evaluation = evaluate.evaluate_model(model, X_test, y_test)
        evaluation["model"] = model
        evaluation["variant"] = variant
        evaluation["label"] = VARIANT_LABELS[variant]
        evaluation.update(
            {
                "cv_best_score": cv_info.get("cv_best_score"),
                "cv_best_std": cv_info.get("cv_best_std"),
                "best_params": cv_info.get("cv_best_params", {}),
            }
        )
        results[variant] = evaluation
        m = evaluation["metrics"]
        print(
            f"    acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
            f"roc_auc={m['roc_auc']:.3f}  (cv={cv_info.get('cv_best_score', float('nan')):.3f})"
        )
    return results


def select_best_model(results: dict[str, dict[str, Any]]) -> str:
    """Pick the winner by test ROC-AUC, breaking ties on F1."""
    return max(
        results,
        key=lambda k: (results[k]["metrics"]["roc_auc"], results[k]["metrics"]["f1"]),
    )


def feature_selection_report(model: Pipeline) -> pd.DataFrame:
    """Coefficient table for the penalised model, showing L1 zeroing."""
    names = preprocessing.feature_names(model)
    return evaluate.coefficient_table(model, names)


def lasso_feature_path(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    c_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Trace how many features survive L1 regularisation as ``C`` grows.

    Demonstrates Lasso's role as an embedded feature selector: small ``C``
    (strong penalty) keeps only the strongest predictors; as ``C`` increases the
    penalty weakens and more coefficients become non-zero.
    """
    c_values = np.asarray(c_values if c_values is not None else config.C_GRID)

    # Transform once with the shared preprocessing pipeline.
    preprocess = preprocessing.build_preprocessing_pipeline()
    X_transformed = preprocess.fit_transform(X_train)
    names = np.asarray(preprocessing.feature_names(preprocess), dtype=object)

    rows = []
    for c in c_values:
        clf = LogisticRegression(
            penalty="l1", C=float(c), solver="liblinear", max_iter=5000,
            random_state=config.RANDOM_STATE,
        )
        clf.fit(X_transformed, y_train)
        coefs = np.ravel(clf.coef_)
        mask = np.abs(coefs) > 1e-8
        rows.append(
            {
                "C": float(c),
                "n_selected": int(mask.sum()),
                "n_zeroed": int((~mask).sum()),
                "train_accuracy": float(clf.score(X_transformed, y_train)),
                "selected_features": ";".join(names[mask]),
            }
        )
    return pd.DataFrame(rows)


def input_spec() -> list[dict[str, Any]]:
    """Machine-readable description of the raw inputs the model expects."""
    spec: list[dict[str, Any]] = []
    for column in config.NUMERIC_FEATURES:
        low, high = config.VALID_RANGES.get(column, (None, None))
        spec.append(
            {
                "name": column,
                "type": "number",
                "range": f"{low} - {high}",
                "description": {
                    "age": "Age in years",
                    "resting_bp": "Resting blood pressure (mm Hg)",
                    "cholesterol": "Serum cholesterol (mg/dl)",
                    "max_heart_rate": "Maximum heart rate achieved",
                    "st_depression": "ST depression induced by exercise (oldpeak)",
                }[column],
            }
        )
    for column in config.CATEGORICAL_FEATURES:
        options = config.CATEGORY_LEVELS.get(column, {})
        spec.append(
            {
                "name": column,
                "type": "categorical",
                "options": {str(k): v for k, v in options.items()},
                "description": column.replace("_", " ").capitalize(),
            }
        )
    return spec


def run_training(
    run_eda: bool = True,
    save: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full pipeline: load -> clean -> EDA -> split -> train -> evaluate -> save."""
    config.ensure_directories()

    # ---- 1. Load & clean -------------------------------------------------- #
    df = data_loader.load_dataset(binarize=True, save_snapshot=True)
    df = preprocessing.clean_dataframe(df)
    df.to_csv(config.CLEAN_CSV, index=False)
    print(f"[train] Dataset shape after cleaning: {df.shape}")

    # ---- 2. EDA ----------------------------------------------------------- #
    eda_output = eda.run_eda(df) if run_eda else None

    # ---- 3. Feature diagnostics ------------------------------------------ #
    redundancy = preprocessing.correlation_redundancy(df)
    target_corr = preprocessing.feature_target_correlation(df)
    if verbose:
        print(f"[train] Highly-correlated pairs (>={config.CORR_REDUNDANCY_THRESHOLD}): {redundancy or 'none'}")
        print("[train] Absolute feature-target correlation:")
        print(target_corr.round(3).to_string())

    # ---- 4. Stratified split --------------------------------------------- #
    X, y = preprocessing.split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(
        f"[train] Train={X_train.shape[0]} Test={X_test.shape[0]} | "
        f"train prevalence={y_train.mean():.3f} test prevalence={y_test.mean():.3f}"
    )

    # ---- 5. Train & evaluate --------------------------------------------- #
    results = train_all_variants(X_train, y_train, X_test, y_test)
    table = evaluate.build_comparison_table(results)
    best_name = select_best_model(results)
    print(f"[train] Best model: {best_name}")

    # ---- 6. Diagnostic figures ------------------------------------------- #
    evaluate.plot_confusion_matrices(results)
    evaluate.plot_roc_curves(results, y_test)

    # Diagnostic reports on disk.
    table.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    redundancy_df = pd.DataFrame(redundancy, columns=["feature_a", "feature_b", "correlation"])
    redundancy_df.to_csv(config.REPORTS_DIR / "feature_redundancy.csv", index=False)
    target_corr.rename("abs_correlation").to_csv(config.REPORTS_DIR / "feature_target_correlation.csv")

    coef_frames = {}
    for variant in VARIANTS:
        coefs = feature_selection_report(results[variant]["model"])
        coefs.to_csv(config.REPORTS_DIR / f"coefficients_{variant}.csv", index=False)

        coef_frames[variant] = coefs

    # Lasso regularisation path (embedded feature selection).
    lasso_path = lasso_feature_path(X_train, y_train)
    lasso_path.to_csv(config.REPORTS_DIR / "lasso_feature_selection.csv", index=False)
    evaluate.plot_lasso_path(lasso_path)

    # Threshold sensitivity for the winning model (clinical trade-off).
    sensitivity = evaluate.threshold_sensitivity(results[best_name]["model"], X_test, y_test)
    sensitivity.to_csv(config.REPORTS_DIR / f"threshold_sensitivity_{best_name}.csv", index=False)

    # ---- Statistical robustness: bootstrap confidence intervals ----------- #
    # A ~61-patient test set cannot support five-decimal comparisons, so every
    # metric is reported with the range it would plausibly take on a re-draw.
    ci_table = evaluate.metric_confidence_intervals(results, y_test)
    ci_table.to_csv(config.REPORTS_DIR / "metric_confidence_intervals.csv", index=False)
    evaluate.plot_metric_confidence_intervals(ci_table)

    headline_ci = ci_table[ci_table["metric"].isin(("roc_auc", "recall", "specificity", "brier"))]
    print("[train] Bootstrap 95% confidence intervals:")
    print(headline_ci.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Say plainly when the headline difference is not statistically conclusive.
    for other in VARIANTS:
        if other == best_name:
            continue
        if evaluate.intervals_overlap(ci_table, best_name, other, "roc_auc"):
            print(
                f"[train] NOTE: '{best_name}' vs '{other}' ROC-AUC intervals overlap - "
                "at this test size the difference is not statistically conclusive."
            )

    # ---- Probability quality: calibration --------------------------------- #
    # Rebuild the winning pipeline (same variant and tuned C) so the raw model
    # can be compared against Platt and isotonic recalibration.
    best_c = results[best_name]["best_params"].get("classifier__C")
    build_best = (
        partial(build_model_pipeline, best_name)
        if best_c is None
        else partial(build_model_pipeline, best_name, C=float(best_c))
    )
    calibration_table, calibration_curves = evaluate.compare_calibration(
        build_best, X_train, y_train, X_test, y_test, variant=best_name
    )
    calibration_table.to_csv(config.REPORTS_DIR / "calibration_metrics.csv", index=False)
    evaluate.plot_calibration_curves(calibration_curves, calibration_table, best_name)

    print("[train] Calibration (Brier, log loss, Cox slope/intercept):")
    print(calibration_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- Independent check on the coefficient story ----------------------- #
    # Coefficients are not comparable across variants; permutation importance is
    # a model-agnostic second opinion on which measurements actually matter.
    importance = evaluate.permutation_importance_report(results, X_test, y_test)
    importance.to_csv(config.REPORTS_DIR / "permutation_importance.csv", index=False)
    evaluate.plot_permutation_importance(importance, order_by=best_name)

    output = {
        "results": results,
        "comparison": table,
        "best_model": best_name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "eda": eda_output,
        "coefficients": coef_frames,
        "lasso_path": lasso_path,
        "confidence_intervals": ci_table,
        "calibration": calibration_table,
        "permutation_importance": importance,
    }

    # ---- 7. Persist every variant, then the best one's model card --------- #
    if save:
        artifacts = persist_all_variants(results, X_train, y_train)
        artifact_info = persist_best_model(
            results[best_name], X_train, y_train, paths=artifacts[best_name]
        )
        output["artifacts"] = artifacts
        output["artifact"] = artifact_info

    return output


def build_metadata(
    result: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    """Build the metadata block stored alongside a serialised model."""
    variant = result["variant"]
    return {
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "metrics": {k: float(v) for k, v in result["metrics"].items()},
        "best_params": {k: str(v) for k, v in result.get("best_params", {}).items()},
        "cv_best_score": float(result.get("cv_best_score", float("nan"))),
        "features": config.FEATURES,
        "transformed_features": preprocessing.feature_names(result["model"]),
        "train_size": int(len(X_train)),
        "train_prevalence": float(y_train.mean()),
        "input_spec": input_spec(),
    }


def persist_all_variants(
    results: dict[str, dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, dict[str, Any]]:
    """Serialise *every* variant so the web app can contrast Ridge and Lasso."""
    artifacts: dict[str, dict[str, Any]] = {}
    for variant, result in results.items():
        artifacts[variant] = model_card.save_artifact(
            result["model"], variant, build_metadata(result, X_train, y_train)
        )
    return artifacts


def persist_best_model(
    result: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialise the winning model and write its model card.

    ``paths`` lets the caller reuse artifacts already written by
    :func:`persist_all_variants` instead of writing a duplicate.
    """
    variant = result["variant"]
    metrics = result["metrics"]

    if paths is None:
        paths = model_card.save_artifact(
            result["model"], variant, build_metadata(result, X_train, y_train)
        )

    card = model_card.render_model_card(
        name=VARIANT_LABELS[variant],
        version=config.MODEL_VERSION,
        metrics=metrics,
        best_params=result.get("best_params", {}),
        feature_list=config.FEATURES,
        input_spec=input_spec(),
        versioned_path=paths["versioned_path"],
    )
    card_path = config.MODELS_DIR / "MODEL_CARD.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"[train] Model card written -> {card_path}")

    return {**paths, "model_card": card_path}


if __name__ == "__main__":
    run_training()
