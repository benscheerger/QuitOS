"""Load and validate the raw EMASENS file.

The dataset is not part of this repository (R1). It must be obtained from the
EMASENS authors and placed in ``data/raw/`` (D10).
"""

from pathlib import Path

import pandas as pd
import pyreadr
from pandas.api.types import is_numeric_dtype

from src import config


def load_raw(path: str | Path = config.RAW_DATA_PATH, validate: bool = True) -> pd.DataFrame:
    """Read the raw ``.rds`` file and return it as a DataFrame.

    Implements D10 (raw input read from ``data/raw/``) and supports R1 (the data
    is never shipped with the repo, so a missing file is an expected situation
    with an explanatory error). ``id`` is cast to ``str`` so participant keys
    behave the same everywhere (R6 relies on them). Nothing is fitted (R7).

    Args:
        path: Location of ``analytic_sample_1.rds``.
        validate: If True (default), run :func:`validate_raw_schema` before returning.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``validate`` is True and the schema check fails.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. The EMASENS dataset is not "
            "redistributed in this repository (design R1). Obtain "
            f"'{path.name}' from the EMASENS authors (Leppin et al., 2026) and "
            f"place it in {config.RAW_DATA_DIR}."
        )

    df = pyreadr.read_r(str(path))[None]
    # pandas >= 3: astype(str) keeps missing values missing, so null ids stay
    # detectable by validate_raw_schema instead of becoming the text "nan".
    df[config.ID_COL] = df[config.ID_COL].astype(str)

    if validate:
        validate_raw_schema(df)
    return df


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Check the raw table before any processing. Does not modify ``df``.

    Guards the assumptions later steps rely on:
      - required columns exist (R2 needs ``EMA_date_time``; R4 needs the
        timing columns; D7 needs ``EMA_number``);
      - ``id`` and ``EMA_id`` are present and ``EMA_id`` is unique, so rows
        and participants can be identified (R4, R6);
      - ``craving_imp`` is a whole number in [0, 10] on every row (T1/T2
        targets, D1).

    Raises:
        ValueError: Describing the first problem found. Messages report counts
            and column names only, never data values.
    """
    missing = [c for c in config.REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")

    for col in (config.ID_COL, config.EMA_ID_COL):
        n_null = int(df[col].isna().sum())
        if n_null:
            raise ValueError(f"Column '{col}' has {n_null} null value(s).")

    n_dup = int(df[config.EMA_ID_COL].duplicated().sum())
    if n_dup:
        raise ValueError(f"Column '{config.EMA_ID_COL}' has {n_dup} duplicate value(s).")

    craving = df[config.CRAVING_COL]
    if not is_numeric_dtype(craving):
        raise ValueError(f"Column '{config.CRAVING_COL}' must be numeric, got {craving.dtype}.")
    n_null = int(craving.isna().sum())
    if n_null:
        raise ValueError(f"Column '{config.CRAVING_COL}' has {n_null} null value(s).")
    n_out = int(((craving < config.CRAVING_MIN) | (craving > config.CRAVING_MAX)).sum())
    if n_out:
        raise ValueError(
            f"Column '{config.CRAVING_COL}' has {n_out} value(s) outside "
            f"[{config.CRAVING_MIN}, {config.CRAVING_MAX}]."
        )
    n_frac = int((craving % 1 != 0).sum())
    if n_frac:
        raise ValueError(f"Column '{config.CRAVING_COL}' has {n_frac} non-whole value(s).")
