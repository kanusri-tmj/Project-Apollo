"""Cleaning, validation, encoding and scaling.

The module is split into two concerns:

1. **Data-quality helpers** (:func:`clean_dataframe`, :func:`report_missing`,
   :func:`detect_outliers_iqr`, :func:`detect_outliers_zscore`,
   :func:`validate_ranges`, :func:`unique_value_report`) that *inspect* and
   lightly tidy the raw frame. These are used by EDA and the training script.

2. **The scikit-learn preprocessing pipeline** built by
   :func:`build_preprocessing_pipeline`. It is a :class:`ColumnTransformer`
   that imputes missing values, one-hot encodes categoricals, standardises
   numeric variables and (optionally) adds engineered features. Because the
   whole thing is a single fitted object it can be serialised and reused
   verbatim by the deployment app.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


# --------------------------------------------------------------------------- #
# Data-quality helpers
# --------------------------------------------------------------------------- #
def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Return a tidied copy of ``df``.

    * drops exact duplicate rows (a known artefact of the Cleveland extract),
    * coerces every expected column to a numeric dtype,
    * leaves missing values in place so the imputers in the pipeline handle
      them consistently for both training and inference.
    """
    df = df.copy()

    expected = [c for c in config.FEATURES + ["target", "target_severity"] if c in df.columns]
    for column in expected:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before = len(df)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[preprocessing] Removed {removed} duplicate row(s).")
    return df


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value counts and percentages (non-zero rows only)."""
    missing = df.isna().sum()
    report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": (missing / len(df) * 100).round(2),
        }
    )
    return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)


def unique_value_report(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Unique values (and counts) for the categorical columns."""
    columns = list(columns or [c for c in config.CATEGORICAL_FEATURES if c in df.columns])
    rows = []
    for column in columns:
        values = sorted(df[column].dropna().unique().tolist())
        rows.append(
            {
                "column": column,
                "n_unique": len(values),
                "values": values,
                "counts": df[column].value_counts().sort_index().to_dict(),
            }
        )
    return pd.DataFrame(rows)


def detect_outliers_iqr(series: pd.Series, factor: float = config.OUTLIER_IQR_FACTOR) -> pd.Series:
    """Boolean mask of IQR-based outliers (outside Q1-1.5*IQR .. Q3+1.5*IQR)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(
    series: pd.Series, threshold: float = config.OUTLIER_ZSCORE_THRESHOLD
) -> pd.Series:
    """Boolean mask of z-score outliers (|z| > threshold)."""
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=series.index)
    z = (series - series.mean()) / std
    return z.abs() > threshold


def report_outliers(
    df: pd.DataFrame, columns: Iterable[str] = ("cholesterol", "resting_bp")
) -> pd.DataFrame:
    """Compare IQR and z-score outlier counts for the requested columns.

    Outliers in clinical measurements are *flagged*, not silently deleted:
    they can represent genuine high-risk patients. The returned table lets a
    clinician decide whether winsorising is appropriate.
    """
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column].dropna()
        iqr_mask = detect_outliers_iqr(series)
        z_mask = detect_outliers_zscore(series)
        rows.append(
            {
                "column": column,
                "n": len(series),
                "iqr_outliers": int(iqr_mask.sum()),
                "zscore_outliers": int(z_mask.sum()),
                "min": float(series.min()),
                "max": float(series.max()),
                "iqr_lower": float(series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25))),
                "iqr_upper": float(series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25))),
            }
        )
    return pd.DataFrame(rows)


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Check that continuous features fall inside plausible physiological ranges."""
    rows = []
    for column, (low, high) in config.VALID_RANGES.items():
        if column not in df.columns:
            continue
        series = df[column].dropna()
        below = int((series < low).sum())
        above = int((series > high).sum())
        rows.append(
            {
                "column": column,
                "valid_min": low,
                "valid_max": high,
                "observed_min": float(series.min()) if len(series) else np.nan,
                "observed_max": float(series.max()) if len(series) else np.nan,
                "below_range": below,
                "above_range": above,
                "valid": below == 0 and above == 0,
            }
        )
    return pd.DataFrame(rows)


def _candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns of ``df`` excluding the target columns."""
    excluded = {config.TARGET, config.TARGET_RAW, "target_severity"}
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def correlation_redundancy(
    df: pd.DataFrame,
    threshold: float = config.CORR_REDUNDANCY_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Return feature pairs whose absolute correlation exceeds ``threshold``.

    Used to justify dropping redundant predictors before modelling. Considers
    every numeric non-target column, so engineered or imported extras are
    covered too.
    """
    columns = _candidate_feature_columns(df)
    numeric = df[columns].corr(numeric_only=True)
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = numeric.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((left, right, float(value)))
    return pairs


def feature_target_correlation(df: pd.DataFrame) -> pd.Series:
    """Absolute Pearson correlation of each feature with the binary target."""
    if config.TARGET not in df.columns:
        return pd.Series(dtype=float)
    corr = df[[*config.FEATURES, config.TARGET]].corr(numeric_only=True)[config.TARGET]
    return corr.drop(config.TARGET).sort_values(key=np.abs, ascending=False)


def drop_redundant_features(
    df: pd.DataFrame, threshold: float = config.CORR_REDUNDANCY_THRESHOLD
) -> tuple[pd.DataFrame, list[str]]:
    """Drop one feature from each highly-correlated pair, keeping the one with
    the stronger absolute correlation to the target."""
    pairs = correlation_redundancy(df, threshold)
    if not pairs:
        return df, []

    target_corr = feature_target_correlation(df).abs().to_dict()
    to_drop: set[str] = set()
    for left, right, _ in pairs:
        if left in to_drop or right in to_drop:
            continue
        weaker = left if target_corr.get(left, 0) < target_corr.get(right, 0) else right
        to_drop.add(weaker)
    kept = df.drop(columns=sorted(to_drop))
    return kept, sorted(to_drop)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add clinically-motivated derived features to the raw patient frame.

    * ``age_band`` — binned age group (categorical),
    * ``age_chol`` — age x cholesterol interaction scaled to a sane magnitude.

    Implemented as a scikit-learn transformer so it is fitted and serialised
    together with the rest of the pipeline; the deployment app therefore only
    ever needs to send the *raw* clinical measurements.
    """

    def __init__(
        self,
        enable: bool = config.ENABLE_FEATURE_ENGINEERING,
        age_bins: list[int] | None = None,
        age_labels: list[str] | None = None,
    ) -> None:
        self.enable = enable
        self.age_bins = age_bins or config.AGE_BINS
        self.age_labels = age_labels or config.AGE_BIN_LABELS

    def fit(self, X: pd.DataFrame, y=None) -> "ClinicalFeatureEngineer":
        # Record the incoming schema so ``get_feature_names_out`` can return
        # correctly-named output even when sklearn does not pass it a list.
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not self.enable:
            return X
        if "age" in X.columns:
            X["age_band"] = pd.cut(
                X["age"], bins=self.age_bins, labels=self.age_labels, right=False
            ).astype(object)
        if "age" in X.columns and "cholesterol" in X.columns:
            X["age_chol"] = X["age"] * X["cholesterol"] / 1000.0
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            base = [str(f) for f in input_features]
        elif hasattr(self, "feature_names_in_"):
            base = [str(f) for f in self.feature_names_in_]
        else:
            base = [f"x{i}" for i in range(getattr(self, "n_features_in_", 0))]
        if self.enable:
            base += ["age_band", "age_chol"]
        return np.asarray(base, dtype=object)


def feature_groups(engineer: bool = config.ENABLE_FEATURE_ENGINEERING) -> tuple[list[str], list[str]]:
    """Return ``(numeric_columns, categorical_columns)`` after engineering."""
    numeric = list(config.NUMERIC_FEATURES)
    categorical = list(config.CATEGORICAL_FEATURES)
    if engineer:
        numeric.append("age_chol")
        categorical.append("age_band")
    return numeric, categorical


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def build_preprocessing_pipeline(
    engineer: bool = config.ENABLE_FEATURE_ENGINEERING,
    scale: bool = True,
) -> Pipeline:
    """Build the reusable preprocessing pipeline.

    Steps
    -----
    * :class:`ClinicalFeatureEngineer` — optional derived features,
    * numeric branch: median imputation + standardisation (required for fair
      Ridge/Lasso penalisation),
    * categorical branch: mode imputation + one-hot encoding (unknown
      categories at inference time are ignored rather than raising).
    """
    numeric, categorical = feature_groups(engineer)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("engineer", ClinicalFeatureEngineer(enable=engineer)),
            ("preprocess", column_transformer),
        ]
    )


def feature_names(pipeline: Pipeline) -> list[str]:
    """Extract the post-transform feature names from a fitted pipeline."""
    pre = pipeline.named_steps["preprocess"]
    return list(pre.get_feature_names_out())


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a Clean frame into ``(X, y)`` using the configured feature list."""
    X = df[list(config.FEATURES)].copy()
    y = df[config.TARGET].astype(int).copy()
    return X, y
