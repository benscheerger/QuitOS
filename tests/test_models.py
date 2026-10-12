"""Tests for src/models/cv.py, train_linear.py and train_logistic.py (R1, R6, R7, D8, D9, D12).

Synthetic data, except one real-data test that is skipped when the file is
absent and checks aggregate numbers only (R1).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from src import config
from src.data.load_data import load_raw
from src.data.preprocess import preprocess
from src.features.build_features import build_modeling_table
from src.models import train_linear, train_logistic
from src.models.cv import oof_predict
from src.utils.splits import fold_assignments, participant_folds


def _balanced(table: pd.DataFrame) -> pd.DataFrame:
    """Alternate y_high_next within each participant so every inner-CV
    validation fold has both classes (needed for log-loss on tiny data)."""
    out = table.copy()
    out[config.Y_HIGH_NEXT] = (out.groupby(config.ID_COL).cumcount() % 2).astype(int)
    return out


def _fold_ids(table, idx):
    return set(table[config.ID_COL].iloc[pd.Index(idx)])


# --- Leakage: test targets never influence that fold's predictions --------------


@pytest.mark.parametrize(
    "name, target, models, proba",
    [
        ("ridge", config.Y_CRAVING_NEXT, train_linear.T1_MODELS, False),
        ("logistic_l2", config.Y_HIGH_NEXT, train_logistic.T2_MODELS, True),
    ],
)
def test_test_target_shuffle_leaves_fold_predictions_unchanged(
    modeling_table, name, target, models, proba
):
    table = _balanced(modeling_table)
    factory, grid = models[name]
    base = oof_predict(table, target, factory, param_grid=grid, proba=proba)

    test_idx = participant_folds(table)[0][1]
    scrambled = table.copy()
    values = scrambled[target].to_numpy(copy=True)
    rng = np.random.default_rng(0)
    # Shuffle, then flip, so the fold-0 test targets are guaranteed to change.
    flipped = rng.permutation(values[test_idx])
    values[test_idx] = 1 - flipped if target == config.Y_HIGH_NEXT else 10 - flipped
    scrambled[target] = values
    out = oof_predict(scrambled, target, factory, param_grid=grid, proba=proba)

    in_fold = base.fold == 0
    np.testing.assert_array_equal(out.pred[in_fold], base.pred[in_fold])
    # The changed rows are *training* rows for other folds, so those must move.
    assert not np.allclose(out.pred[~in_fold], base.pred[~in_fold])


# --- Recording estimator: what every fit and predict actually saw ----------------

FIT_LOG: list[tuple[int, tuple]] = []
PREDICT_LOG: list[tuple[int, tuple]] = []
CURRENT_FOLD = {"fold": -1}


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Predicts the training mean and logs the row labels of every fit/predict.

    ``alpha`` exists only so GridSearchCV has something to tune. Logs are
    module-level because GridSearchCV clones (deep-copies) estimators.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        FIT_LOG.append((CURRENT_FOLD["fold"], tuple(X.index)))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        PREDICT_LOG.append((CURRENT_FOLD["fold"], tuple(X.index)))
        return np.full(len(X), self.mean_)


def test_every_fit_and_predict_sees_only_allowed_rows(modeling_table):
    FIT_LOG.clear()
    PREDICT_LOG.clear()
    calls = []

    def make_estimator():
        CURRENT_FOLD["fold"] = len(calls)
        calls.append(1)
        return RecordingRegressor()

    grid = {"alpha": [1.0, 2.0, 3.0]}
    result = oof_predict(modeling_table, config.Y_CRAVING_NEXT, make_estimator, param_grid=grid)
    folds = participant_folds(modeling_table)
    ids = modeling_table[config.ID_COL]

    assert len(calls) == config.N_SPLITS  # one fresh estimator per outer fold
    # 5 inner splits x 3 grid values + 1 refit, per outer fold
    assert len(FIT_LOG) == config.N_SPLITS * (config.INNER_SPLITS * len(grid["alpha"]) + 1)
    for fold, rows in FIT_LOG:
        train_ids = _fold_ids(modeling_table, folds[fold][0])
        assert set(ids.loc[list(rows)]) <= train_ids, f"fit in fold {fold} saw test participants"

    # Predict calls: inner scoring stays inside training rows; outer test rows
    # are predicted exactly once, in their own fold.
    outer_test_predictions = []
    for fold, rows in PREDICT_LOG:
        train_idx, test_idx = folds[fold]
        if set(rows) <= set(test_idx.tolist()):
            outer_test_predictions.extend(rows)
        else:
            assert set(rows) <= set(train_idx.tolist())
    assert sorted(outer_test_predictions) == list(range(len(modeling_table)))

    assert all(params.keys() == {"alpha"} for params in result.best_params)


def test_fold_matches_fold_assignments(modeling_table):
    result = oof_predict(modeling_table, config.Y_CRAVING_NEXT, train_linear.T1_MODELS["linear"][0])
    expected = modeling_table[config.ID_COL].map(fold_assignments(modeling_table)).to_numpy()
    np.testing.assert_array_equal(result.fold, expected)
    assert not np.isnan(result.pred).any()
    assert result.best_params == [{}] * config.N_SPLITS


# --- Baselines ------------------------------------------------------------------


def test_t1_baselines(modeling_table):
    base = train_linear.baseline_predictions(modeling_table)
    np.testing.assert_array_equal(base["persistence"], modeling_table[config.CRAVING_COL])
    np.testing.assert_array_equal(base["person_mean"], modeling_table["craving_mean_so_far"])

    y = modeling_table[config.Y_CRAVING_NEXT].to_numpy()
    for train_idx, test_idx in participant_folds(modeling_table):
        np.testing.assert_allclose(base["train_mean"][test_idx], y[train_idx].mean())


def test_t2_baselines(modeling_table):
    base = train_logistic.baseline_predictions(modeling_table)
    score, is_proba = base["craving_score"]
    np.testing.assert_array_equal(score, modeling_table[config.CRAVING_COL])
    assert not is_proba

    prevalence, is_proba = base["prevalence"]
    assert is_proba
    y = modeling_table[config.Y_HIGH_NEXT].to_numpy()
    for train_idx, test_idx in participant_folds(modeling_table):
        np.testing.assert_allclose(prevalence[test_idx], y[train_idx].mean())


# --- Clipping (D9) --------------------------------------------------------------


def test_clip_t1():
    clipped, n = train_linear.clip_t1(np.array([-1.5, 0.0, 5.0, 10.0, 12.0]))
    np.testing.assert_array_equal(clipped, [0.0, 0.0, 5.0, 10.0, 10.0])
    assert n == 2


def test_all_evaluated_t1_predictions_are_in_range(modeling_table, monkeypatch):
    # A step-shaped target makes the linear fits overshoot [0, 10] at the extremes.
    table = modeling_table.copy()
    table[config.Y_CRAVING_NEXT] = np.where(table[config.CRAVING_COL] >= 5, 10.0, 0.0)

    seen = []
    original = train_linear.regression_metrics

    def recording_metrics(y, pred):
        seen.append(np.asarray(pred))
        return original(y, pred)

    monkeypatch.setattr(train_linear, "regression_metrics", recording_metrics)
    results = train_linear.run_t1(table)

    all_pred = np.concatenate(seen)
    assert all_pred.min() >= config.CRAVING_MIN
    assert all_pred.max() <= config.CRAVING_MAX
    assert results.metrics["n_clipped"].sum() > 0  # clipping was actually exercised


# --- main(): aggregate files only (R1) ------------------------------------------

EXPECTED_FILES = {
    "t1_metrics.csv": {"model", "kind", "mae", "rmse", "r2", "n_clipped",
                       "beats_persistence_mae", "beats_persistence_rmse"},
    "t1_by_day.csv": {"model", "study_day", "n", "mae", "rmse", "r2"},
    "t2_metrics.csv": {"model", "kind", "roc_auc", "pr_auc", "brier", "auc_pp_n_excluded",
                       "beats_craving_score_auc", "beats_prevalence_brier"},
    "t2_by_day.csv": {"model", "study_day", "n", "roc_auc", "pr_auc", "brier"},
    "t2_calibration.csv": {"model", "bin", "mean_pred", "obs_rate", "count"},
    "cv_params.csv": {"task", "model", "fold", "param", "value"},
}


def test_main_writes_aggregate_files_only(modeling_table, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    table = _balanced(modeling_table)
    train_linear.main(table)
    train_logistic.main(table)
    capsys.readouterr()

    assert {p.name for p in tmp_path.iterdir()} == set(EXPECTED_FILES)
    ids = set(table[config.ID_COL]) | set(table[config.EMA_ID_COL])
    for name, required in EXPECTED_FILES.items():
        df = pd.read_csv(tmp_path / name)
        assert required <= set(df.columns), name
        assert not {config.ID_COL, config.EMA_ID_COL} & set(df.columns), name
        assert not df.astype(str).isin(ids).any().any(), f"{name} contains ids"
        assert len(df) < len(table) or name == "t1_by_day.csv" and len(df) < 10 * len(table)

    t1 = pd.read_csv(tmp_path / "t1_metrics.csv")
    t2 = pd.read_csv(tmp_path / "t2_metrics.csv")
    assert set(t1["model"]) == {*train_linear.T1_MODELS, "persistence", "train_mean", "person_mean"}
    assert set(t2["model"]) == {*train_logistic.T2_MODELS, "prevalence", "craving_score"}

    params = pd.read_csv(tmp_path / "cv_params.csv")
    assert params.groupby("task")["model"].unique().map(list).to_dict() == {
        "t1": ["ridge"],
        "t2": ["logistic_l2"],
    }
    assert len(params) == 2 * config.N_SPLITS

    train_linear.main(table)  # rerun replaces T1 rows instead of stacking them
    capsys.readouterr()
    assert len(pd.read_csv(tmp_path / "cv_params.csv")) == 2 * config.N_SPLITS


# --- Real data (skipped when the file is absent; aggregates only, R1) -----------


@pytest.fixture(scope="module")
def real_results():
    table = build_modeling_table(preprocess(load_raw()))
    t1 = train_linear.run_t1(table).metrics.set_index("model")
    t2 = train_logistic.run_t2(table).metrics.set_index("model")
    return t1, t2


@pytest.mark.skipif(not config.RAW_DATA_PATH.is_file(), reason="real EMASENS file not present")
def test_real_data_baselines_and_sanity(real_results):
    t1, t2 = real_results
    tol = 1e-4
    assert t1.loc["persistence", "mae"] == pytest.approx(1.4847, abs=tol)
    assert t1.loc["persistence", "rmse"] == pytest.approx(2.4111, abs=tol)
    assert t1.loc["train_mean", "mae"] == pytest.approx(2.7872, abs=tol)
    assert t1.loc["person_mean", "mae"] == pytest.approx(1.8299, abs=tol)
    assert t2.loc["prevalence", "brier"] == pytest.approx(0.2370, abs=tol)
    assert t2.loc["craving_score", "roc_auc"] == pytest.approx(0.8370, abs=tol)

    # Models: sanity bounds only, no pinned scores.
    assert t1.loc["ridge", "rmse"] < t1.loc["persistence", "rmse"]
    assert t2.loc["logistic_l2", "brier"] < t2.loc["prevalence", "brier"]
