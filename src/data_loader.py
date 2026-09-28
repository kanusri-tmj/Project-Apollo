"""Dataset acquisition and loading.

Downloads the UCI *Cleveland* Heart Disease dataset (303 records, 13 clinical
features + target) and exposes it as a tidy :class:`pandas.DataFrame` with
clinician-friendly column names.

The UCI file is headerless and uses ``?`` for missing values (only in the
``ca``/``thal`` columns), so we read those as ``NaN`` up front.

Running this module directly performs the download and writes a raw CSV
snapshot to ``data/raw/heart_disease_raw.csv``::

    python -m src.data_loader
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src import config

# Original target is disease severity 0-4; binarise to 0 = absent, 1 = present.
DISEASE_PRESENT_THRESHOLD = 1


def download_raw(force: bool = False) -> Path:
    """Download and extract ``processed.cleveland.data`` into ``data/raw``.

    Returns the path to the extracted data file. The download is skipped when
    the file already exists unless ``force=True``.
    """
    config.ensure_directories()
    extracted = config.DATA_RAW_DIR / config.RAW_UCI_MEMBER

    if extracted.exists() and not force:
        return extracted

    if force or not config.RAW_UCI_ARCHIVE.exists():
        print(f"[data_loader] Downloading dataset from {config.RAW_DATA_URL} ...")
        urllib.request.urlretrieve(config.RAW_DATA_URL, config.RAW_UCI_ARCHIVE)

    with zipfile.ZipFile(config.RAW_UCI_ARCHIVE) as archive:
        if config.RAW_UCI_MEMBER not in archive.namelist():
            raise FileNotFoundError(
                f"{config.RAW_UCI_MEMBER} not found inside {config.RAW_UCI_ARCHIVE}"
            )
        archive.extract(config.RAW_UCI_MEMBER, config.DATA_RAW_DIR)

    print(f"[data_loader] Saved raw data to {extracted}")
    return extracted


def load_raw_dataframe() -> pd.DataFrame:
    """Load the raw UCI file into a DataFrame with readable column names."""
    path = download_raw()
    df = pd.read_csv(
        path,
        header=None,
        names=config.UCI_COLUMNS,
        na_values=["?"],
        # Force every column numeric (the file is fully numeric but may be
        # parsed as strings when missing markers are present).
        dtype="float64",
    )
    df = df.rename(columns=config.RENAME_MAP)
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 0-4 severity ``target`` into a binary present/absent label.

    ``0`` -> no disease, anything ``>= 1`` -> disease present. The original
    severity column is retained as ``target_severity`` for reference.
    """
    df = df.copy()
    if config.TARGET_RAW in df.columns and config.TARGET not in df.columns:
        df = df.rename(columns={config.TARGET_RAW: config.TARGET})
    df["target_severity"] = df[config.TARGET]
    df[config.TARGET] = (df[config.TARGET] >= DISEASE_PRESENT_THRESHOLD).astype(int)
    return df


def load_dataset(binarize: bool = True, save_snapshot: bool = False) -> pd.DataFrame:
    """Convenience loader returning the analysis-ready raw DataFrame.

    Parameters
    ----------
    binarize:
        When ``True`` (default) the 0-4 severity target is collapsed into a
        binary 0/1 disease-present column.
    save_snapshot:
        When ``True`` the loaded frame is written to ``data/raw`` as CSV.
    """
    df = load_raw_dataframe()
    if binarize:
        df = binarize_target(df)
    if save_snapshot:
        config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.RAW_CSV, index=False)
        print(f"[data_loader] Raw snapshot written to {config.RAW_CSV}")
    return df


if __name__ == "__main__":
    frame = load_dataset(save_snapshot=True)
    print(frame.head())
    print(f"\nShape: {frame.shape}")
    print(f"Target counts:\n{frame[config.TARGET].value_counts()}")
