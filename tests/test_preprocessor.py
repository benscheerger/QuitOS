"""Tests for src/features/preprocessor.py (R7).

Synthetic data, except one real-data test that is skipped when the file is
absent and never prints rows (R1).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src import config
from src.data.load_data import load_raw
from src.data.preprocess import preprocess
from src.features.build_features import build_modeling_table
from src.features.preprocessor import make_preprocessor
from src.utils.splits import get_xy, participant_folds

NUMERIC = list(config.NUMERIC_FEATURES)
TRAIN_IDS = {"p00", "p01", "p02", "p03", "p04", "p05"}


@pytest.fixture
def xy_split(modeling_table):
    """Train = p00..p05, test = p06..p09; test numerics shifted so means differ."""
    X, _, groups = get_xy(modeling_table, config.Y_CRAVING_NEXT)
    is_train = groups.isin(TRAIN_IDS)
    X_train = X.loc[is_train].copy()
    X_test = X.loc[~is_train].copy()
    X_test[NUMERIC] = X_test[NUMERIC] + 100.0
    return X_train, X_test


def _num_steps(pre):
    num = pre.named_transformers_["num"]
    return num.named_steps["drop_constant"], num.named_steps["scale"]


def _transform(pre, X) -> pd.DataFrame:
    """Transform and confirm the pandas output promised by ``set_output``."""
    out = pre.transform(X)
    assert isinstance(out, pd.DataFrame)
    return out


def _rows(df: pd.DataFrame, idx) -> pd.DataFrame:
    # pd.Index wrapper only for the type checker: pyrefly's bundled pandas stubs
    # use a different numpy stub than the venv, so raw ndarrays fail to match.
    return df.iloc[pd.Index(idx)]


def _kept_numeric(pre):
    drop_constant, _ = _num_steps(pre)
    return list(drop_constant.get_feature_names_out(NUMERIC))


# --- Fit on training rows only (R7) ---------------------------------------------


def test_scaler_statistics_come_from_fitted_subset(xy_split):
    X_train, X_test = xy_split
    pre = make_preprocessor().fit(X_train)
    _, scaler = _num_steps(pre)
    kept = _kept_numeric(pre)

    np.testing.assert_allclose(scaler.mean_, X_train[kept].mean().to_numpy())
    np.testing.assert_allclose(scaler.scale_, X_train[kept].std(ddof=0).to_numpy())
    full_means = pd.concat([X_train, X_test])[kept].mean().to_numpy()
    assert not np.allclose(scaler.mean_, full_means)


def test_changing_test_rows_does_not_change_fitted_params(modeling_table):
    X, _, groups = get_xy(modeling_table, config.Y_CRAVING_NEXT)
    train = groups.isin(TRAIN_IDS)
    rng = np.random.default_rng(7)

    mutated = X.copy()
    n_test = (~train).sum()
    for col in NUMERIC:
        mutated.loc[~train, col] = rng.normal(500, 50, n_test).tolist()
    for col in config.CATEGORICAL_FEATURES:
        mutated.loc[~train, col] = "never_seen_in_train"

    a = make_preprocessor().fit(X.loc[train])
    b = make_preprocessor().fit(mutated.loc[train])
    (vt_a, sc_a), (vt_b, sc_b) = _num_steps(a), _num_steps(b)
    np.testing.assert_array_equal(vt_a.get_support(), vt_b.get_support())
    np.testing.assert_array_equal(sc_a.mean_, sc_b.mean_)
    np.testing.assert_array_equal(sc_a.scale_, sc_b.scale_)
    cats_a = a.named_transformers_["cat"].categories_
    cats_b = b.named_transformers_["cat"].categories_
    assert [list(c) for c in cats_a] == [list(c) for c in cats_b]
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


def test_unseen_category_gives_all_zero_block(xy_split):
    X_train, X_test = xy_split
    X_test.iloc[0, X_test.columns.get_loc("motivation_to_stop")] = "Brand new level"
    pre = make_preprocessor().fit(X_train)
    out = _transform(pre, X_test)

    block = [c for c in out.columns if c.startswith("motivation_to_stop_")]
    assert block and "motivation_to_stop_Brand new level" not in out.columns
    assert out.iloc[0][block].sum() == 0

    seen = set(pre.named_transformers_["cat"].categories_[
        list(config.CATEGORICAL_FEATURES).index("motivation_to_stop")
    ])
    is_seen = X_test["motivation_to_stop"].isin(seen).to_numpy()
    row_sums = out[block].sum(axis=1).to_numpy()
    np.testing.assert_array_equal(row_sums, is_seen.astype(float))


def test_column_constant_in_train_is_dropped(xy_split):
    X_train, X_test = xy_split
    X_train["excited_imp"] = 5.0
    assert X_test["excited_imp"].nunique() > 1

    pre = make_preprocessor().fit(X_train)
    out = _transform(pre, X_test)
    assert "excited_imp" not in out.columns
    assert "calm_imp" in out.columns


# --- Output shape and names -----------------------------------------------------


def test_output_is_named_dataframe_without_nan(xy_split):
    X_train, X_test = xy_split
    pre = make_preprocessor().fit(X_train)
    out = _transform(pre, X_test)

    assert isinstance(out, pd.DataFrame)
    assert not out.isna().any().any()
    assert out.index.equals(X_test.index)

    cats = pre.named_transformers_["cat"].categories_
    expected = _kept_numeric(pre) + [
        f"{col}_{level}" for col, levels in zip(config.CATEGORICAL_FEATURES, cats) for level in levels
    ]
    assert list(out.columns) == expected
    assert len(set(out.columns)) == len(out.columns)


def test_transform_before_fit_raises(modeling_table):
    X, _, _ = get_xy(modeling_table, config.Y_CRAVING_NEXT)
    with pytest.raises(NotFittedError):
        make_preprocessor().transform(X)


def test_each_call_returns_a_new_unfitted_object():
    assert make_preprocessor() is not make_preprocessor()


# --- Real data (skipped when the file is absent; no rows printed, R1) -----------


@pytest.mark.skipif(not config.RAW_DATA_PATH.is_file(), reason="real EMASENS file not present")
def test_real_data_every_fold_fits_on_train_and_transforms_test():
    # Folds 0 and 2 contain motivation_to_stop levels never seen in training.
    table = build_modeling_table(preprocess(load_raw()))
    X, _, _ = get_xy(table, config.Y_HIGH_NEXT)
    for train_idx, test_idx in participant_folds(table):
        pre = make_preprocessor().fit(_rows(X, train_idx))
        out = _transform(pre, _rows(X, test_idx))
        assert len(out) == len(test_idx)
        assert not out.isna().any().any()
        assert list(out.columns) == list(pre.get_feature_names_out())
