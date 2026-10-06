"""Tests for src/features/build_features.py (R2, R3, R5, D1, D2, D3, D7).

Synthetic data only, except one count-only real-data test that is skipped
when the file is absent (R1).
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.load_data import load_raw
from src.data.preprocess import preprocess
from src.features.build_features import build_modeling_table

FEATURES = list(config.FEATURE_COLUMNS)
TARGETS = list(config.TARGET_COLUMNS)


@pytest.fixture
def processed(features_raw_df):
    return preprocess(features_raw_df)


@pytest.fixture
def table(processed):
    return build_modeling_table(processed)


def _rows(df):
    return df.set_index(config.EMA_ID_COL)


def _append_answer(raw, ema_id, day, hour, minute, craving, lapse=0.0):
    """Return a copy of ``raw`` with one more scheduled, answered p1 prompt."""
    row = raw.loc[raw[config.EMA_ID_COL] == "p1_8"].copy()
    row[config.EMA_ID_COL] = ema_id
    row[config.EMA_DATE_COL] = [dt.date(2026, 1, 4 + day)]
    row[config.EMA_HOUR_COL] = float(hour)
    row[config.STUDY_DAY_COL] = np.int32(day)
    row[config.EMA_DATETIME_COL] = pd.Timestamp(2026, 1, 4 + day, hour, minute)
    row[config.CRAVING_COL] = craving
    row[config.LAPSE_COL] = lapse
    return pd.concat([raw, row], ignore_index=True)


def _randomise(df, mask, rng):
    """Copy of ``df`` with every value a feature could read replaced on ``mask`` rows.

    Times, ids and observed/filled-in status are kept, so the row set and
    ordering stay the same and only the *values* differ.
    """
    out = df.copy()
    n = int(mask.sum())
    for col in config.EMA_ITEM_COLUMNS:
        out.loc[mask, col] = rng.integers(0, 11, n).astype(float)
    for col in config.BASELINE_NUMERIC:
        out.loc[mask, col] = rng.uniform(18, 80, n)
    for col in config.CATEGORICAL_FEATURES:
        out.loc[mask, col] = [f"random_{v}" for v in rng.integers(0, 1000, n)]
    out.loc[mask, config.IS_SELF_INITIATED_COL] = rng.integers(0, 2, n).astype(bool)
    return out


# --- Leakage: the future and filled-in rows must not matter (R5, D3) ------------


def test_future_changes_do_not_affect_past_features(processed):
    base = build_modeling_table(processed)
    rng = np.random.default_rng(0)
    later_rows_changed = False

    for cutoff in sorted(processed[config.EVENT_TIME_COL].unique()):
        future = processed[config.EVENT_TIME_COL] > cutoff
        if not future.any():
            continue
        out = build_modeling_table(_randomise(processed, future, rng))

        assert out[config.EMA_ID_COL].tolist() == base[config.EMA_ID_COL].tolist()
        past = base[config.EVENT_TIME_COL] <= cutoff
        pd.testing.assert_frame_equal(
            out.loc[past, FEATURES], base.loc[past, FEATURES], obj=f"features at cutoff {cutoff}"
        )
        if not out.loc[~past, FEATURES].equals(base.loc[~past, FEATURES]):
            later_rows_changed = True

    # The randomisation must actually bite, or the test above proves nothing.
    assert later_rows_changed


def test_filled_in_rows_change_nothing(processed):
    filled = processed[config.WAS_FILLED_IN_COL]
    assert filled.any()
    mutated = _randomise(processed, filled, np.random.default_rng(1))
    pd.testing.assert_frame_equal(build_modeling_table(mutated), build_modeling_table(processed))


# --- Target alignment (D2, R2) ----------------------------------------------------


def test_expected_rows_and_targets(table):
    assert table[config.EMA_ID_COL].tolist() == ["p1_8", "p1_11", "p2_1"]
    assert table[config.TARGET_EMA_ID_COL].tolist() == ["p1_11", "p1_10", "p2_2"]
    assert table[config.Y_CRAVING_NEXT].tolist() == [9.0, 7.0, 3.0]
    assert table[config.Y_HIGH_NEXT].tolist() == [1, 1, 0]
    assert table[config.TARGET_IS_SELF_INITIATED_COL].tolist() == [1, 0, 0]
    assert table[config.TARGET_GAP_H_COL].tolist() == pytest.approx([125 / 60, 0.5, 55 / 60])


def test_last_answer_of_day_has_no_target(table):
    # p1_10's next real answer (p1_12) is on study day 2 (D2).
    assert "p1_10" not in table[config.EMA_ID_COL].tolist()
    assert "p1_12" not in table[config.EMA_ID_COL].tolist()


def test_target_is_next_real_same_day_answer(table, processed):
    src = _rows(processed)
    assert (table[config.TARGET_EVENT_TIME_COL] > table[config.EVENT_TIME_COL]).all()

    real = processed[processed[config.IS_OBSERVED_COL]]
    next_real = {}
    for _, person in real.groupby(config.ID_COL):
        ids = person[config.EMA_ID_COL].tolist()
        next_real.update(zip(ids[:-1], ids[1:]))

    for _, row in table.iterrows():
        target = row[config.TARGET_EMA_ID_COL]
        assert target == next_real[row[config.EMA_ID_COL]]
        assert src.loc[target, config.IS_OBSERVED_COL]
        assert src.loc[target, config.STUDY_DAY_COL] == row[config.STUDY_DAY_COL]
        assert src.loc[target, config.ID_COL] == row[config.ID_COL]
        assert src.loc[target, config.CRAVING_COL] == row[config.Y_CRAVING_NEXT]
        assert src.loc[target, config.EVENT_TIME_COL] == row[config.TARGET_EVENT_TIME_COL]


def test_high_craving_threshold_boundary(processed):
    src = processed.copy()
    src.loc[src[config.EMA_ID_COL] == "p1_11", config.CRAVING_COL] = 6.0
    src.loc[src[config.EMA_ID_COL] == "p1_10", config.CRAVING_COL] = 7.0
    rows = _rows(build_modeling_table(src))
    assert rows.loc["p1_8", config.Y_CRAVING_NEXT] == 6.0
    assert rows.loc["p1_8", config.Y_HIGH_NEXT] == 0
    assert rows.loc["p1_11", config.Y_CRAVING_NEXT] == 7.0
    assert rows.loc["p1_11", config.Y_HIGH_NEXT] == 1


# --- Hand-computed feature values (R5) ------------------------------------------


def test_hand_computed_p1_day1(features_raw_df):
    # Add a later day-1 answer so p1_10 gets a target and appears in the table.
    # Its own features cannot depend on this later row (see the future test).
    raw = _append_answer(features_raw_df, "p1_13", day=1, hour=12, minute=30, craving=5.0)
    rows = _rows(build_modeling_table(preprocess(raw))).loc[["p1_8", "p1_11", "p1_10"]]

    assert rows[config.CRAVING_COL].tolist() == [4.0, 9.0, 7.0]
    assert rows["craving_mean_so_far"].tolist() == pytest.approx([4.0, 6.5, 20 / 3])
    assert rows["craving_roll_mean"].tolist() == pytest.approx([4.0, 6.5, 20 / 3])
    assert rows["craving_dev_from_mean"].tolist() == pytest.approx([0.0, 2.5, 7 - 20 / 3])
    assert rows["n_prev_obs"].tolist() == [0, 1, 2]
    assert rows["hours_since_prev_obs"].tolist() == pytest.approx([0.0, 125 / 60, 0.5])
    assert rows["prev_lapse"].tolist() == [0.0, 0.0, 1.0]
    assert rows["is_first_obs"].tolist() == [1, 0, 0]
    assert rows[config.IS_SELF_INITIATED_COL].tolist() == [0, 1, 0]
    assert rows["hour_of_day"].tolist() == pytest.approx([9 + 5 / 60, 11 + 10 / 60, 11 + 40 / 60])


def test_history_spans_days_and_window_rolls(features_raw_df):
    # Add a second day-2 answer so p1_12 (4th real answer) gets a target.
    raw = _append_answer(features_raw_df, "p1_14", day=2, hour=10, minute=30, craving=3.0)
    row = _rows(build_modeling_table(preprocess(raw))).loc["p1_12"]

    assert row["craving_mean_so_far"] == pytest.approx((4 + 9 + 7 + 2) / 4)  # 5.5
    assert row["craving_roll_mean"] == pytest.approx((9 + 7 + 2) / 3)  # last 3 only: 6.0
    assert row["n_prev_obs"] == 3
    assert row["hours_since_prev_obs"] == pytest.approx(21 + 50 / 60)  # 11:40 -> 09:30 next day
    assert row["is_first_obs"] == 0
    assert row[config.STUDY_DAY_COL] == 2


# --- Column hygiene (R3, R5) ----------------------------------------------------


def test_feature_columns_exclude_forbidden():
    forbidden = (
        set(config.EXCLUDED_COLUMNS)
        | {
            config.ID_COL,
            config.EMA_ID_COL,
            config.EMA_DATE_COL,
            config.EMA_DATETIME_COL,
            config.EVENT_TIME_COL,
            config.IS_OBSERVED_COL,
            config.WAS_FILLED_IN_COL,
        }
        | set(config.TARGET_COLUMNS)
        | set(config.METADATA_COLUMNS)
    )
    assert not forbidden & set(config.FEATURE_COLUMNS)
    assert config.TARGET_GAP_H_COL not in config.FEATURE_COLUMNS
    assert "lapse_prior" in config.EXCLUDED_COLUMNS
    assert len(set(config.FEATURE_COLUMNS)) == len(config.FEATURE_COLUMNS)
    assert len(config.EMA_ITEM_COLUMNS) == 48


def test_output_columns_and_order(table):
    expected = [
        config.ID_COL,
        config.EMA_ID_COL,
        config.EVENT_TIME_COL,
        *config.FEATURE_COLUMNS,
        *config.TARGET_COLUMNS,
        *config.METADATA_COLUMNS,
    ]
    assert list(table.columns) == expected
    assert table.index.equals(pd.RangeIndex(len(table)))


def test_categorical_features_stay_strings(table):
    for col in config.CATEGORICAL_FEATURES:
        assert all(isinstance(v, str) for v in table[col])


def test_no_nan(table):
    assert not table[FEATURES + TARGETS].isna().any().any()


# --- Input validation and purity --------------------------------------------------


def test_missing_column_raises(processed):
    with pytest.raises(ValueError, match="missing required columns"):
        build_modeling_table(processed.drop(columns="age"))


def test_unsorted_input_raises(processed):
    with pytest.raises(ValueError, match="not sorted"):
        build_modeling_table(processed.iloc[::-1].reset_index(drop=True))


def test_nan_feature_raises(processed):
    src = processed.copy()
    src.loc[src[config.EMA_ID_COL] == "p1_8", "excited_imp"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        build_modeling_table(src)


def test_input_is_not_modified(processed):
    before = processed.copy()
    build_modeling_table(processed)
    pd.testing.assert_frame_equal(processed, before)


# --- Real data (skipped when the file is absent; counts only, R1) ---------------


@pytest.mark.skipif(not config.RAW_DATA_PATH.is_file(), reason="real EMASENS file not present")
def test_real_data_counts():
    df = build_modeling_table(preprocess(load_raw()))
    assert len(df) == 4343
    assert df[config.ID_COL].nunique() == 38
    assert int(df[config.Y_HIGH_NEXT].sum()) == 1665
    assert not df[FEATURES + TARGETS].isna().any().any()
    assert len(config.EMA_ITEM_COLUMNS) == 48
    assert set(config.EMA_ITEM_COLUMNS) <= set(df.columns)
