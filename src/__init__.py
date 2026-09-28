"""Heart Disease Prediction — a regularised logistic-regression project.

Modules
-------
config          Central paths, schema and model settings.
data_loader     Dataset download / loading.
preprocessing   Cleaning, validation, encoding and scaling.
eda             Exploratory plots.
train           Model training and cross-validation.
evaluate        Metrics, comparison table and diagnostic plots.
predict         Loading a saved artifact and scoring new patients.
model_card      Model card + versioned artifact persistence.
explain         Model introspection powering the Live Simulation UI.
"""

__all__ = [
    "config",
    "data_loader",
    "preprocessing",
    "eda",
    "evaluate",
    "train",
    "predict",
    "model_card",
    "explain",
]
