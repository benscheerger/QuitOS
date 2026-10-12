"""Tests for src/models/evaluate.py on small hand-made arrays (no real data)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.models.evaluate import (
    by_study_day,
    calibration_table,
    classification_metrics,
    per_participant,
    r2,
    regression_metrics,
    roc_auc,
)

# Series (not ndarrays) so the direct sklearn reference calls type-check with
# pyrefly's bundled stubs; evaluate.py accepts both.
Y_REG = pd.Series([0.0, 3.0, 5.0, 7.0, 10.0, 6.0])
P_REG = pd.Series([1.0, 2.5, 5.0, 8.0, 9.0, 4.0])
Y_BIN = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
P_BIN = pd.Series([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.7])


def test_regression_metrics_match_sklearn():
    out = regression_metrics(Y_REG, P_REG)
    assert out["mae"] == pytest.approx(mean_absolute_error(Y_REG, P_REG))
    assert out["rmse"] == pytest.approx(np.sqrt(mean_squared_error(Y_REG, P_REG)))
    assert out["r2"] == pytest.approx(r2_score(Y_REG, P_REG))


def test_classification_metrics_match_sklearn():
    out = classification_metrics(Y_BIN, P_BIN)
    assert out["roc_auc"] == pytest.approx(roc_auc_score(Y_BIN, P_BIN))
    assert out["pr_auc"] == pytest.approx(average_precision_score(Y_BIN, P_BIN))
    assert out["brier"] == pytest.approx(brier_score_loss(Y_BIN, P_BIN))


def test_brier_is_nan_for_ranking_scores():
    out = classification_metrics(Y_BIN, P_BIN * 10, proba=False)
    assert np.isnan(out["brier"])
    assert out["roc_auc"] == pytest.approx(roc_auc_score(Y_BIN, P_BIN))


def test_undefined_metrics_are_nan_not_errors():
    assert np.isnan(roc_auc(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])))
    assert np.isnan(r2(np.full(3, 5.0), np.array([4.0, 5.0, 6.0])))
    out = classification_metrics(np.ones(3), np.array([0.2, 0.5, 0.9]))
    assert np.isnan(out["roc_auc"]) and np.isnan(out["pr_auc"])


def test_per_participant_auc_skips_single_class():
    groups = np.array(["a", "a", "a", "a", "b", "b", "c", "c"])
    y = np.array([0, 1, 0, 1, 1, 1, 0, 1])  # b is single-class
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.5, 0.6, 0.7, 0.3])
    out = per_participant(roc_auc, y, p, groups)
    assert out["n_used"] == 2
    assert out["n_excluded"] == 1
    expected = np.array(
        [
            float(roc_auc_score([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])),
            float(roc_auc_score([0, 1], [0.7, 0.3])),
        ]
    )
    assert out["mean"] == pytest.approx(expected.mean())  # (1.0 + 0.0) / 2
    assert out["sd"] == pytest.approx(expected.std(ddof=1))


def test_per_participant_r2_skips_constant_target():
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    y = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0])  # b constant
    p = np.array([1.0, 2.5, 2.5, 3.0, 4.0, 5.0])
    out = per_participant(r2, y, p, groups)
    assert (out["n_used"], out["n_excluded"]) == (1, 1)
    assert out["mean"] == pytest.approx(r2_score([1, 2, 3], [1, 2.5, 2.5]))
    assert np.isnan(out["sd"])  # only one participant used


def test_by_study_day_groups_correctly():
    day = pd.Series([2, 1, 2, 1, 3, 2])
    out = by_study_day(regression_metrics, Y_REG, P_REG, day)
    assert out["study_day"].tolist() == [1, 2, 3]
    assert out["n"].tolist() == [2, 3, 1]
    day1 = day == 1
    assert out.loc[0, "mae"] == pytest.approx(mean_absolute_error(Y_REG[day1], P_REG[day1]))
    assert {"rmse", "r2"} <= set(out.columns)


def test_calibration_counts_and_values():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < p).astype(int)
    table = calibration_table(y, p, n_bins=10)

    assert len(table) == 10
    assert table["count"].sum() == 200
    assert table["mean_pred"].is_monotonic_increasing
    frame = pd.DataFrame({"p": p, "y": y})
    first = pd.qcut(frame["p"], 10, labels=False) == 0
    assert table.loc[0, "mean_pred"] == pytest.approx(frame.loc[first, "p"].mean())
    assert table.loc[0, "obs_rate"] == pytest.approx(frame.loc[first, "y"].mean())


def test_calibration_merges_bins_for_few_distinct_values():
    p = np.repeat([0.3, 0.4, 0.5], [4, 3, 3])
    y = np.array([0, 0, 1, 0, 1, 0, 0, 1, 1, 0])
    table = calibration_table(y, p, n_bins=10)
    assert len(table) <= 3
    assert table["count"].sum() == len(p)
