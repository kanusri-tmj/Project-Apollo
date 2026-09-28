"""Exploratory Data Analysis.

Generates the required visualisations and summary tables and writes them to
``reports/figures`` and ``reports`` respectively. All plotting uses the
non-interactive ``Agg`` backend so the module runs head-less in CI.

Charts produced
---------------
* ``histograms_continuous.png`` — age / cholesterol / resting BP distributions
* ``class_balance.png``         — count plot of the binary target
* ``categorical_vs_target.png`` — disease counts per categorical level
* ``boxplots_continuous.png``   — cholesterol & max-heart-rate spread/outliers
* ``scatter_age_vs_hr.png``     — age vs max heart rate coloured by outcome
* ``correlation_heatmap.png``   — correlation of all features
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # head-less rendering

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config, preprocessing  # noqa: E402

sns.set_theme(style="whitegrid")
FIGURE_DPI = 120


def _save(fig: plt.Figure, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics (mean/median/std/min/max/skew) for numeric features."""
    numeric = df[[c for c in config.NUMERIC_FEATURES if c in df.columns]]
    summary = numeric.describe().T[["mean", "50%", "std", "min", "max"]]
    summary = summary.rename(columns={"50%": "median"})
    summary["range"] = summary["max"] - summary["min"]
    summary["skew"] = numeric.skew()
    return summary.round(3)


def plot_histograms(df: pd.DataFrame) -> str:
    """Distributions of age, cholesterol and resting blood pressure."""
    columns = [c for c in ["age", "cholesterol", "resting_bp"] if c in df.columns]
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
    axes = [axes] if len(columns) == 1 else axes
    for ax, column in zip(axes, columns):
        sns.histplot(df[column].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {column} (skew={df[column].skew():.2f})")
        ax.set_xlabel(column)
    return _save(fig, "histograms_continuous.png")


def plot_class_balance(df: pd.DataFrame) -> str:
    """Count plot showing the balance of the binary target."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=config.TARGET, ax=ax, hue=config.TARGET, legend=False,
                  palette="Set2")
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Class balance: no disease (0) vs disease (1)")
    ax.set_xlabel("Heart disease")
    ax.set_ylabel("Patients")
    return _save(fig, "class_balance.png")


def plot_categorical_vs_target(df: pd.DataFrame) -> str:
    """Disease counts for each categorical feature."""
    columns = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]
    n_cols = 4
    n_rows = -(-len(columns) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()
    for ax, column in zip(axes, columns):
        sns.countplot(data=df, x=column, hue=config.TARGET, ax=ax, palette="Set1")
        ax.set_title(column)
        ax.set_xlabel("")
        ax.legend(title="disease", fontsize=8)
    for ax in axes[len(columns):]:
        ax.axis("off")
    return _save(fig, "categorical_vs_target.png")


def plot_boxplots(df: pd.DataFrame) -> str:
    """Spread and outliers for cholesterol and max heart rate."""
    columns = [c for c in ["cholesterol", "max_heart_rate", "resting_bp"] if c in df.columns]
    fig, axes = plt.subplots(1, len(columns), figsize=(4.5 * len(columns), 4))
    axes = [axes] if len(columns) == 1 else axes
    for ax, column in zip(axes, columns):
        sns.boxplot(x=df[column].dropna(), ax=ax, color="lightcoral")
        ax.set_title(f"{column} (box plot)")
        ax.set_xlabel(column)
    return _save(fig, "boxplots_continuous.png")


def plot_age_vs_heart_rate(df: pd.DataFrame) -> str:
    """Scatter plot of age vs max heart rate, coloured by outcome."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.scatterplot(
        data=df,
        x="age",
        y="max_heart_rate",
        hue=config.TARGET,
        palette="coolwarm",
        alpha=0.8,
        ax=ax,
    )
    ax.set_title("Age vs max heart rate by heart-disease status")
    ax.legend(title="disease")
    return _save(fig, "scatter_age_vs_hr.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Correlation heatmap across every clinical feature plus the target."""
    columns = [c for c in config.FEATURES if c in df.columns]
    if config.TARGET in df.columns:
        columns = [*columns, config.TARGET]
    corr = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap of clinical features")
    return _save(fig, "correlation_heatmap.png")


def run_eda(df: pd.DataFrame, save_tables: bool = True) -> dict[str, object]:
    """Run the whole EDA suite and return the generated artifact paths/tables."""
    config.ensure_directories()

    summary = statistical_summary(df)
    missing = preprocessing.report_missing(df)
    uniques = preprocessing.unique_value_report(df)
    outliers = preprocessing.report_outliers(df)
    ranges = preprocessing.validate_ranges(df)
    redundancy = preprocessing.correlation_redundancy(df)
    target_corr = preprocessing.feature_target_correlation(df)

    figures = {
        "histograms": plot_histograms(df),
        "class_balance": plot_class_balance(df),
        "categorical_vs_target": plot_categorical_vs_target(df),
        "boxplots": plot_boxplots(df),
        "age_vs_hr": plot_age_vs_heart_rate(df),
        "correlation_heatmap": plot_correlation_heatmap(df),
    }

    if save_tables:
        summary.to_csv(config.REPORTS_DIR / "statistical_summary.csv")
        if not missing.empty:
            missing.to_csv(config.REPORTS_DIR / "missing_values.csv")
        outliers.to_csv(config.REPORTS_DIR / "outlier_report.csv", index=False)
        ranges.to_csv(config.REPORTS_DIR / "range_validation.csv", index=False)
        uniques.to_csv(config.REPORTS_DIR / "unique_values.csv", index=False)

    return {
        "figures": figures,
        "summary": summary,
        "missing": missing,
        "uniques": uniques,
        "outliers": outliers,
        "ranges": ranges,
        "redundancy": redundancy,
        "target_correlation": target_corr,
    }
