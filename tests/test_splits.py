"""Tests for src/utils/splits.py (R3, R5, R6, D4).

Synthetic data, except one count-only real-data test that is skipped when the
file is absent (R1).
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.load_data import load_raw
from src.data.preprocess import preprocess
from src.features.build_features import build_modeling_table
from src.utils.splits import fold_assignments, get_xy, participant_folds


def _ids(table, idx):
    return set(table[config.ID_COL].iloc[idx])


# --- participant_folds (R6, D4) ---------------------------------------------------


def test_train_and_test_participants_are_disjoint(modeling_table):
    for train_idx, test_idx in participant_folds(modeling_table):
        assert not _ids(modeling_table, train_idx) & _ids(modeling_table, test_idx)


def test_each_participant_and_row_in_exactly_one_test_fold(modeling_table):
    folds = participant_folds(modeling_table)
    assert len(folds) == config.N_SPLITS

    test_rows = np.concatenate([test for _, test in folds])
    assert sorted(test_rows.tolist()) == list(range(len(modeling_table)))

    test_ids = [pid for _, test in folds for pid in _ids(modeling_table, test)]
    assert sorted(test_ids) == sorted(modeling_table[config.ID_COL].unique())

    for train_idx, test_idx in folds:
        # Within a fold, train + test cover every row once.
        assert sorted(np.concatenate([train_idx, test_idx]).tolist()) == list(
            range(len(modeling_table))
        )


def test_folds_are_deterministic(modeling_table):
    first = participant_folds(modeling_table)
    second = participant_folds(modeling_table)
    for (tr1, te1), (tr2, te2) in zip(first, second, strict=True):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)


@pytest.mark.parametrize("n_splits", [-1, 0, 1, 11])
def test_bad_n_splits_raises(modeling_table, n_splits):
    # modeling_table has 10 participants.
    with pytest.raises(ValueError, match="n_splits"):
        participant_folds(modeling_table, n_splits=n_splits)


def test_n_splits_equal_to_participants_is_allowed(modeling_table):
    folds = participant_folds(modeling_table, n_splits=10)
    assert all(len(_ids(modeling_table, test)) == 1 for _, test in folds)


# --- fold_assignments -------------------------------------------------------------


def test_fold_assignments_match_participant_folds(modeling_table):
    assignments = fold_assignments(modeling_table)
    folds = participant_folds(modeling_table)

    assert assignments.name == "fold"
    assert assignments.index.is_monotonic_increasing
    assert set(assignments.index) == set(modeling_table[config.ID_COL])
    for fold, (_, test_idx) in enumerate(folds):
        for pid in _ids(modeling_table, test_idx):
            assert assignments[pid] == fold
    assert assignments.value_counts().sort_index().tolist() == [
        len(_ids(modeling_table, test)) for _, test in folds
    ]


# --- get_xy (R3, R5) --------------------------------------------------------------


@pytest.mark.parametrize("target", config.TARGET_COLUMNS)
def test_get_xy_returns_exactly_the_features(modeling_table, target):
    X, y, groups = get_xy(modeling_table, target)

    assert list(X.columns) == list(config.FEATURE_COLUMNS)
    forbidden = {
        config.ID_COL,
        config.EMA_ID_COL,
        config.EVENT_TIME_COL,
        *config.TARGET_COLUMNS,
        *config.METADATA_COLUMNS,
    }
    assert not forbidden & set(X.columns)
    pd.testing.assert_series_equal(y, modeling_table[target])
    pd.testing.assert_series_equal(groups, modeling_table[config.ID_COL])


def test_get_xy_returns_copies(modeling_table):
    before = modeling_table.copy()
    X, y, _ = get_xy(modeling_table, config.Y_CRAVING_NEXT)
    X.iloc[:, 0] = -1.0
    y.iloc[:] = -1.0
    pd.testing.assert_frame_equal(modeling_table, before)


def test_get_xy_bad_target_raises(modeling_table):
    with pytest.raises(ValueError, match="Unknown target"):
        get_xy(modeling_table, config.TARGET_GAP_H_COL)


def test_get_xy_missing_feature_raises(modeling_table):
    with pytest.raises(ValueError, match="missing feature columns"):
        get_xy(modeling_table.drop(columns="craving_roll_mean"), config.Y_HIGH_NEXT)


# --- Real data (skipped when the file is absent; counts only, R1) ---------------


@pytest.mark.skipif(not config.RAW_DATA_PATH.is_file(), reason="real EMASENS file not present")
def test_real_data_folds():
    table = build_modeling_table(preprocess(load_raw()))
    folds = participant_folds(table)

    assert [len(_ids(table, test)) for _, test in folds] == [8, 8, 8, 7, 7]
    assert [len(test) for _, test in folds] == [907, 903, 901, 817, 815]
    # pd.Index wrapper only for the type checker (bundled pandas stubs vs venv numpy).
    positives = [int(table.iloc[pd.Index(test)][config.Y_HIGH_NEXT].sum()) for _, test in folds]
    assert positives == [352, 381, 337, 266, 329]
    for train_idx, test_idx in folds:
        assert not _ids(table, train_idx) & _ids(table, test_idx)
