"""Tests for src/data/preprocess.py (R2, R3, R4, D7, D11)."""

import pandas as pd
import pytest

from src import config
from src.data.load_data import load_raw
from src.data.preprocess import preprocess

EXPECTED_ORDER = ["p1_8", "p1_9", "p1_11", "p1_10", "p2_1", "p2_2"]
ADDED_COLUMNS = {
    config.IS_OBSERVED_COL,
    config.WAS_FILLED_IN_COL,
    config.IS_SELF_INITIATED_COL,
    config.EVENT_TIME_COL,
}


@pytest.fixture
def processed(raw_df):
    return preprocess(raw_df)


def _by_ema_id(df):
    return df.set_index(config.EMA_ID_COL)


# --- Flags (R2, D7, D11) --------------------------------------------------------


def test_observed_and_filled_in_flags(processed):
    rows = _by_ema_id(processed)
    filled = rows[config.WAS_FILLED_IN_COL]
    assert filled[filled].index.tolist() == ["p1_9"]
    assert (rows[config.IS_OBSERVED_COL] == ~rows[config.WAS_FILLED_IN_COL]).all()
    assert rows[config.IS_OBSERVED_COL].dtype == bool


def test_self_initiated_flag_survives_dropping_ema_number(processed):
    assert config.EMA_NUMBER_COL not in processed.columns
    rows = _by_ema_id(processed)
    flagged = rows[config.IS_SELF_INITIATED_COL]
    assert flagged[flagged].index.tolist() == ["p1_11"]


# --- Exclusions (R3) ------------------------------------------------------------


def test_excluded_columns_are_dropped(processed):
    assert not set(config.EXCLUDED_COLUMNS) & set(processed.columns)


def test_excluded_columns_already_absent_is_fine(raw_df):
    out = preprocess(raw_df.drop(columns=["mpath_data", "fitbit_data"]))
    assert not set(config.EXCLUDED_COLUMNS) & set(out.columns)


# --- Event time and ordering (R4) -----------------------------------------------


def test_event_time_uses_timestamp_or_scheduled_hour(processed):
    rows = _by_ema_id(processed)
    assert rows.loc["p1_9", config.EVENT_TIME_COL] == pd.Timestamp(2026, 1, 5, 10)
    observed = rows[rows[config.IS_OBSERVED_COL]]
    assert (observed[config.EVENT_TIME_COL] == observed[config.EMA_DATETIME_COL]).all()
    assert processed[config.EVENT_TIME_COL].notna().all()
    assert processed[config.EVENT_TIME_COL].dtype == processed[config.EMA_DATETIME_COL].dtype


def test_fixture_contains_the_inversion(raw_df):
    # Sanity check: by EMA_id, p1_10 comes first, but p1_11 happened earlier.
    rows = _by_ema_id(raw_df)
    assert rows.loc["p1_11", config.EMA_DATETIME_COL] < rows.loc["p1_10", config.EMA_DATETIME_COL]


def test_rows_sorted_by_id_then_time(processed):
    assert processed[config.EMA_ID_COL].tolist() == EXPECTED_ORDER


def test_self_initiated_inversion_is_fixed(processed):
    order = processed[config.EMA_ID_COL].tolist()
    assert order.index("p1_11") < order.index("p1_10")


def test_event_time_increases_within_participant(processed):
    monotonic = processed.groupby(config.ID_COL)[config.EVENT_TIME_COL].apply(
        lambda s: s.is_monotonic_increasing
    )
    assert monotonic.all()


def test_filled_in_row_sits_between_neighbours(processed):
    order = processed[config.EMA_ID_COL].tolist()
    i = order.index("p1_9")
    assert order[i - 1 : i + 2] == ["p1_8", "p1_9", "p1_11"]


def test_ties_broken_by_numeric_ema_id_suffix(raw_df):
    # Make p1_10 a filled-in 10:00 row, tying with p1_9 on event_time.
    # A text sort would wrongly put "p1_10" before "p1_9".
    idx = raw_df.index[raw_df[config.EMA_ID_COL] == "p1_10"][0]
    raw_df.loc[idx, config.EMA_DATETIME_COL] = pd.NaT
    raw_df.loc[idx, config.EMA_HOUR_COL] = 10.0
    order = preprocess(raw_df)[config.EMA_ID_COL].tolist()
    assert order.index("p1_9") < order.index("p1_10")


def test_malformed_ema_id_raises(raw_df):
    raw_df.loc[0, config.EMA_ID_COL] = "p2-2"
    with pytest.raises(ValueError, match="integer suffix"):
        preprocess(raw_df)


# --- Shape and purity -----------------------------------------------------------


def test_all_rows_kept_including_filled_in(raw_df, processed):
    assert len(processed) == len(raw_df)
    assert processed[config.WAS_FILLED_IN_COL].sum() == raw_df[config.EMA_DATETIME_COL].isna().sum()


def test_original_column_names_kept(raw_df, processed):
    expected = (set(raw_df.columns) - set(config.EXCLUDED_COLUMNS)) | ADDED_COLUMNS
    assert set(processed.columns) == expected
    assert set(config.KEY_COLUMNS) <= set(processed.columns)


def test_index_is_reset(processed):
    assert processed.index.equals(pd.RangeIndex(len(processed)))


def test_input_is_not_modified(raw_df):
    before = raw_df.copy()
    preprocess(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


# --- Real data (skipped when the file is absent, e.g. CI / fresh clone; R1) -----


@pytest.mark.skipif(not config.RAW_DATA_PATH.is_file(), reason="real EMASENS file not present")
def test_real_data_counts():
    # Asserts on counts only; never prints rows.
    df = preprocess(load_raw())
    assert df[config.ID_COL].nunique() == 38
    assert len(df) == 6124
    assert int(df[config.IS_OBSERVED_COL].sum()) == 4722
    assert int(df[config.IS_SELF_INITIATED_COL].sum()) == 44
