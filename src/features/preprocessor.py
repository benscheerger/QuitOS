"""Fold-fitted preprocessing for the linear baselines (R7)."""

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def make_preprocessor() -> ColumnTransformer:
    """Return an **unfitted** preprocessor for ``config.FEATURE_COLUMNS``.

    - ``NUMERIC_FEATURES``: ``VarianceThreshold(0.0)`` drops columns that are
      constant in the fitting data, then ``StandardScaler()``.
    - ``CATEGORICAL_FEATURES``: ``OneHotEncoder(handle_unknown="ignore")``,
      dense output, no dropped level. A level not seen during fitting becomes
      an all-zero block instead of raising.
    - Any other column is dropped. Output is a pandas DataFrame with the
      original numeric names and ``<column>_<level>`` one-hot names.

    R7: everything here learns from data (which columns are constant, means,
    standard deviations, category levels). It must **only be fit on
    training-fold rows** and then used to transform that fold's test rows.
    Never fit it on the full table or on test participants.

    Interpretation: because every level gets a one-hot column (no
    ``drop="first"``), each categorical block sums to 1 and is collinear with
    the intercept. Individual coefficients are therefore not interpretable on
    their own; only predictions (and, with regularisation, their combined
    effect) are meaningful.
    """
    numeric = Pipeline(
        [
            ("drop_constant", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, list(config.NUMERIC_FEATURES)),
            ("cat", categorical, list(config.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
