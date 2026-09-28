"""Central configuration for the Heart Disease Prediction project.

Everything that other modules need to agree on (paths, column names, feature
groups, hyper-parameter grids, model versioning) lives here so that there is a
single source of truth. Import this module instead of hard-coding values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
RAW_UCI_ARCHIVE = DATA_RAW_DIR / "heart_uci.zip"
RAW_UCI_MEMBER = "processed.cleveland.data"  # 303 records, the "Cleveland" subset
RAW_CSV = DATA_RAW_DIR / "heart_disease_raw.csv"
CLEAN_CSV = DATA_PROCESSED_DIR / "heart_disease_clean.csv"

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
# Column order as it appears in processed.cleveland.data (headerless file).
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Friendly, clinician-readable column names.
RENAME_MAP = {
    "age": "age",
    "sex": "sex",
    "cp": "chest_pain_type",
    "trestbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_ecg",
    "thalach": "max_heart_rate",
    "exang": "exercise_angina",
    "oldpeak": "st_depression",
    "slope": "st_slope",
    "ca": "num_major_vessels",
    "thal": "thalassemia",
    "num": "target",
}

# Raw numeric-coded categorical columns (needed for encoding / validation).
CATEGORICAL_FEATURES = [
    "sex",
    "chest_pain_type",
    "fasting_blood_sugar",
    "resting_ecg",
    "exercise_angina",
    "st_slope",
    "num_major_vessels",
    "thalassemia",
]

# Continuous clinical measurements.
NUMERIC_FEATURES = [
    "age",
    "resting_bp",
    "cholesterol",
    "max_heart_rate",
    "st_depression",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # 13 clinical features
TARGET = "target"
TARGET_RAW = "num"  # raw UCI severity column (0-4) before binarisation

# Original UCI codes for the categorical variables (used by the UI / validation).
CATEGORY_LEVELS = {
    "sex": {0: "Female", 1: "Male"},
    "chest_pain_type": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fasting_blood_sugar": {0: "<= 120 mg/dl", 1: "> 120 mg/dl"},
    "resting_ecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exercise_angina": {0: "No", 1: "Yes"},
    "st_slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "num_major_vessels": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thalassemia": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}

# Plausible physiological ranges used for data validation.
VALID_RANGES = {
    "age": (18, 100),
    "resting_bp": (80, 220),
    "cholesterol": (100, 600),
    "max_heart_rate": (60, 220),
    "st_depression": (0.0, 7.0),
}

# --------------------------------------------------------------------------- #
# Modelling configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Regularisation strengths explored during cross-validation.
C_GRID = np.logspace(-3, 3, 13)

# Correlation threshold above which a feature pair is flagged as redundant.
CORR_REDUNDANCY_THRESHOLD = 0.85
# Correlation of a feature with the target below which it is "near-zero signal".
CORR_WEAK_SIGNAL_THRESHOLD = 0.02

OUTLIER_IQR_FACTOR = 1.5
OUTLIER_ZSCORE_THRESHOLD = 3.0

# Whether to add engineered features (age bands, age x cholesterol interaction).
ENABLE_FEATURE_ENGINEERING = True
AGE_BINS = [0, 39, 49, 59, 69, 120]
AGE_BIN_LABELS = ["<40", "40-49", "50-59", "60-69", "70+"]

# The three trained variants. All are persisted so the Live Simulation can
# contrast Ridge's dampening with Lasso's exact-zero feature selection.
MODEL_VARIANTS = ("baseline", "ridge_l2", "lasso_l1")
VARIANT_LABELS = {
    "baseline": "Plain logistic regression",
    "ridge_l2": "Ridge (L2)",
    "lasso_l1": "Lasso (L1)",
}
# Variant used as the headline verdict (overridden at runtime by test ROC-AUC).
PRIMARY_VARIANT = "ridge_l2"

MODEL_VERSION = "1.0.0"


@dataclass
class SplitConfig:
    """Settings for the stratified train/test split."""

    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE


@dataclass
class TrainingConfig:
    """Settings shared by every trained model variant."""

    cv_folds: int = 5
    scoring: str = "roc_auc"
    c_grid: np.ndarray = field(default_factory=lambda: C_GRID)
    random_state: int = RANDOM_STATE
    # The grid is tiny (13 values x 5 folds x 2 variants), so single-process is
    # fast. Set to -1 to parallelise; on Windows the loky backend may then print
    # harmless resource-tracker tracebacks at interpreter shutdown.
    n_jobs: int = 1


def ensure_directories() -> None:
    """Create every directory the pipeline writes to."""
    for directory in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
