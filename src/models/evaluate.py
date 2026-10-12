"""Shared regression and classification metrics (§6, D12).

All functions return aggregates only; nothing here keeps per-row values or ids.
"""

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src import config

ArrayLike = np.ndarray | pd.Series
MetricFn = Callable[[ArrayLike, ArrayLike], float]


def _arr(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _pair(y: ArrayLike, pred: ArrayLike) -> tuple[pd.Series, pd.Series]:
    """Positional float columns for sklearn (indexes of Series inputs are dropped).

    Going through a DataFrame also keeps pyrefly happy: its bundled sklearn and
    pandas stubs use a different numpy than the venv, so plain venv arrays fail
    their signatures while pandas columns satisfy both.
    """
    frame = pd.DataFrame({"y": _arr(y), "pred": _arr(pred)})
    return frame["y"], frame["pred"]


# --- Single metrics (NaN instead of errors for undefined cases) -----------------


def mae(y: ArrayLike, pred: ArrayLike) -> float:
    return float(mean_absolute_error(*_pair(y, pred)))


def rmse(y: ArrayLike, pred: ArrayLike) -> float:
    return float(np.sqrt(mean_squared_error(*_pair(y, pred))))


def r2(y: ArrayLike, pred: ArrayLike) -> float:
    """R²; NaN when ``y`` is constant or has < 2 values (undefined), so
    per-participant averages skip it."""
    values = _arr(y)
    if values.size < 2 or np.unique(values).size < 2:
        return float("nan")
    return float(r2_score(*_pair(y, pred)))


def roc_auc(y: ArrayLike, score: ArrayLike) -> float:
    """ROC-AUC; NaN when ``y`` has a single class (undefined)."""
    if np.unique(_arr(y)).size < 2:
        return float("nan")
    return float(roc_auc_score(*_pair(y, score)))


def pr_auc(y: ArrayLike, score: ArrayLike) -> float:
    """Average precision (PR-AUC); NaN when ``y`` has a single class."""
    if np.unique(_arr(y)).size < 2:
        return float("nan")
    return float(average_precision_score(*_pair(y, score)))


def brier(y: ArrayLike, p: ArrayLike) -> float:
    return float(brier_score_loss(*_pair(y, p)))


# --- Metric bundles -------------------------------------------------------------


def regression_metrics(y: ArrayLike, pred: ArrayLike) -> dict[str, float]:
    """T1 pooled metrics: MAE and RMSE (co-primary, D12) and R²."""
    return {"mae": mae(y, pred), "rmse": rmse(y, pred), "r2": r2(y, pred)}


def classification_metrics(y: ArrayLike, score: ArrayLike, proba: bool = True) -> dict[str, float]:
    """T2 pooled metrics: ROC-AUC, PR-AUC and, for probabilities only, Brier.

    ``proba=False`` marks ``score`` as a ranking score (e.g. ``craving_imp``),
    for which the Brier score is meaningless and reported as NaN.
    """
    return {
        "roc_auc": roc_auc(y, score),
        "pr_auc": pr_auc(y, score),
        "brier": brier(y, score) if proba else float("nan"),
    }


# --- Breakdowns -----------------------------------------------------------------


def per_participant(
    metric_fn: MetricFn, y: ArrayLike, pred: ArrayLike, groups: ArrayLike
) -> dict[str, float]:
    """Compute ``metric_fn`` per participant and summarise across participants (§6).

    Participants whose metric is undefined (NaN), e.g. single-class participants
    for AUC or constant ``y`` for R², are excluded and counted.

    Returns:
        ``{"mean", "sd", "n_used", "n_excluded"}``; ``sd`` uses ddof=1 (spread
        across participants) and is NaN if fewer than 2 participants are used.
    """
    frame = pd.DataFrame({"y": _arr(y), "pred": _arr(pred), "g": np.asarray(groups)})
    values = np.array(
        [metric_fn(part["y"], part["pred"]) for _, part in frame.groupby("g")]
    )
    used = values[~np.isnan(values)]
    return {
        "mean": float(used.mean()) if used.size else float("nan"),
        "sd": float(used.std(ddof=1)) if used.size > 1 else float("nan"),
        "n_used": int(used.size),
        "n_excluded": int(values.size - used.size),
    }


def by_study_day(
    metrics_fn: Callable[[ArrayLike, ArrayLike], Mapping[str, float]],
    y: ArrayLike,
    pred: ArrayLike,
    study_day: ArrayLike,
) -> pd.DataFrame:
    """Pooled metrics within each study day of the test participants (D5, §6).

    Returns:
        One row per study day: ``study_day``, ``n`` and the keys of ``metrics_fn``.
    """
    frame = pd.DataFrame({"y": _arr(y), "pred": _arr(pred), "day": np.asarray(study_day)})
    rows = []
    for day, part in frame.groupby("day", sort=True):
        rows.append(
            {
                "study_day": day,
                "n": len(part),
                **metrics_fn(part["y"], part["pred"]),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(
    y: ArrayLike, p: ArrayLike, n_bins: int = config.CALIBRATION_BINS
) -> pd.DataFrame:
    """Reliability table with quantile bins of the predicted probability (§6).

    Bins with identical edges are merged (``duplicates="drop"``), so a model
    with few distinct predictions (e.g. the prevalence baseline) gets fewer bins.

    Returns:
        ``bin`` (0-based), ``mean_pred``, ``obs_rate``, ``count``.
    """
    frame = pd.DataFrame({"y": _arr(y), "p": _arr(p)})
    frame["bin"] = pd.qcut(frame["p"], q=n_bins, labels=False, duplicates="drop")
    table = (
        frame.groupby("bin", sort=True)
        .agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), count=("y", "size"))
        .reset_index()
    )
    table["bin"] = table["bin"].astype(int)
    return table
