"""Tests for src/data/load_data.py. Synthetic data only; no real files are read (R1)."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load_data
from src.data.load_data import load_raw, validate_raw_schema


def test_valid_fixture_passes(raw_df):
    validate_raw_schema(raw_df)


@pytest.mark.parametrize("col", config.REQUIRED_RAW_COLUMNS)
def test_missing_required_column(raw_df, col):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_schema(raw_df.drop(columns=col))


@pytest.mark.parametrize("col", [config.ID_COL, config.EMA_ID_COL])
def test_null_key(raw_df, col):
    raw_df.loc[0, col] = None
    with pytest.raises(ValueError, match=f"'{col}' has 1 null"):
        validate_raw_schema(raw_df)


def test_duplicate_ema_id(raw_df):
    raw_df.loc[1, config.EMA_ID_COL] = raw_df.loc[0, config.EMA_ID_COL]
    with pytest.raises(ValueError, match="1 duplicate"):
        validate_raw_schema(raw_df)


@pytest.mark.parametrize(
    "value, message",
    [
        (np.nan, "null"),
        (-1.0, "outside"),
        (11.0, "outside"),
        (4.5, "non-whole"),
    ],
)
def test_bad_craving(raw_df, value, message):
    raw_df.loc[0, config.CRAVING_COL] = value
    with pytest.raises(ValueError, match=message):
        validate_raw_schema(raw_df)


def test_non_numeric_craving(raw_df):
    raw_df[config.CRAVING_COL] = raw_df[config.CRAVING_COL].astype(str)
    with pytest.raises(ValueError, match="must be numeric"):
        validate_raw_schema(raw_df)


def test_negative_zero_craving_is_allowed(raw_df):
    # The real file stores some zeros as -0.0.
    raw_df.loc[0, config.CRAVING_COL] = -0.0
    validate_raw_schema(raw_df)


def test_validate_does_not_modify_input(raw_df):
    before = raw_df.copy()
    validate_raw_schema(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_load_raw_missing_file_explains(tmp_path):
    with pytest.raises(FileNotFoundError, match="not redistributed"):
        load_raw(tmp_path / "analytic_sample_1.rds")


@pytest.fixture
def fake_rds(tmp_path, monkeypatch):
    """Point load_raw at an empty file and make pyreadr return a given frame."""
    path = tmp_path / "fake.rds"
    path.touch()
    holder = {}
    monkeypatch.setattr(load_data.pyreadr, "read_r", lambda _p: {None: holder["df"].copy()})

    def _set(df):
        holder["df"] = df
        return path

    return _set


def test_load_raw_casts_id_to_str(raw_df, fake_rds):
    raw_df[config.ID_COL] = raw_df[config.ID_COL].astype("category")  # as in the real file
    out = load_raw(fake_rds(raw_df))
    assert all(isinstance(v, str) for v in out[config.ID_COL])


def test_load_raw_validates_by_default(raw_df, fake_rds):
    path = fake_rds(raw_df.drop(columns=config.CRAVING_COL))
    with pytest.raises(ValueError, match="missing required columns"):
        load_raw(path)
    load_raw(path, validate=False)  # opt-out skips the check
