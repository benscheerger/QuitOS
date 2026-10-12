"""Out-of-fold predictions over participant folds, with grouped inner tuning (R6, R7, D4, D8)."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from src import config
from src.utils.splits import get_xy, participant_folds


@dataclass(frozen=True)
class OOFResult:
    """Out-of-fold predictions for every row of the modelling table.

    Attributes:
        pred: Predictions aligned by position with the table (one per row).
        fold: Outer fold in which each row was a test row.
        best_params: One dict per outer fold with the inner-CV choice; ``{}``
            when no grid was searched.
    """

    pred: np.ndarray
    fold: np.ndarray
    best_params: list[dict[str, Any]]


def oof_predict(
    table: pd.DataFrame,
    target: str,
    make_estimator: Callable[[], Any],
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    proba: bool = False,
) -> OOFResult:
    """Predict every row once, from a model that never saw that row's participant.

    For each outer fold from :func:`participant_folds` (D4, R6):
      1. ``make_estimator()`` builds a fresh, unfitted estimator (usually a
         Pipeline with the preprocessor, so scaling/encoding is fit on the
         training fold only, R7).
      2. If ``param_grid`` is given, the estimator is wrapped in
         ``GridSearchCV(cv=GroupKFold(INNER_SPLITS))`` and fitted with
         ``groups`` = the **training fold's** participant ids, so tuning never
         touches the outer test fold (D8, R7). Scoring is ``neg_log_loss`` for
         ``y_high_next`` and ``neg_mean_squared_error`` otherwise.
      3. The fitted model predicts the test rows (``predict_proba[:, 1]`` if
         ``proba``).

    Raises:
        ValueError: Propagated from :func:`get_xy` / :func:`participant_folds`.
    """
    X, y, groups = get_xy(table, target)
    scoring = "neg_log_loss" if target == config.Y_HIGH_NEXT else "neg_mean_squared_error"

    pred = np.full(len(table), np.nan)
    fold_of_row = np.full(len(table), -1, dtype=int)
    best_params: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(participant_folds(table)):
        X_train, y_train = X.iloc[pd.Index(train_idx)], y.iloc[pd.Index(train_idx)]
        X_test = X.iloc[pd.Index(test_idx)]

        estimator = make_estimator()
        if param_grid is not None:
            search = GridSearchCV(
                estimator,
                dict(param_grid),
                scoring=scoring,
                cv=GroupKFold(n_splits=config.INNER_SPLITS),
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train, groups=groups.iloc[pd.Index(train_idx)])
            model = search.best_estimator_
            best_params.append(dict(search.best_params_))
        else:
            model = estimator.fit(X_train, y_train)
            best_params.append({})

        if np.any(fold_of_row[test_idx] != -1):
            raise RuntimeError("A row was assigned to more than one test fold.")
        pred[test_idx] = model.predict_proba(X_test)[:, 1] if proba else model.predict(X_test)
        fold_of_row[test_idx] = fold

    if np.isnan(pred).any() or (fold_of_row < 0).any():
        raise RuntimeError("Some rows did not receive an out-of-fold prediction.")
    return OOFResult(pred=pred, fold=fold_of_row, best_params=best_params)


# --- Aggregate results helpers ----------------------------------------------------

CV_PARAMS_FILE = "cv_params.csv"
CV_PARAMS_COLUMNS = ["task", "model", "fold", "param", "value"]


@dataclass(frozen=True)
class TaskResults:
    """Aggregate outputs of one task. No per-row predictions or ids (R1)."""

    metrics: pd.DataFrame
    by_day: pd.DataFrame
    params: pd.DataFrame
    calibration: pd.DataFrame | None = None


def params_frame(task: str, oof: Mapping[str, OOFResult]) -> pd.DataFrame:
    """Long table of inner-CV choices: one row per (model, outer fold, parameter)."""
    rows = [
        {"task": task, "model": name, "fold": fold, "param": param, "value": value}
        for name, result in oof.items()
        for fold, params in enumerate(result.best_params)
        for param, value in params.items()
    ]
    return pd.DataFrame(rows, columns=CV_PARAMS_COLUMNS)


def update_cv_params(results_dir: Path, params: pd.DataFrame, task: str) -> Path:
    """Write ``params`` into ``cv_params.csv``, replacing earlier rows of ``task``.

    T1 and T2 share the file; replacing (not appending blindly) keeps reruns
    from stacking duplicate rows.
    """
    path = results_dir / CV_PARAMS_FILE
    if path.is_file():
        existing = pd.read_csv(path)
        existing = existing[existing["task"] != task]
        params = pd.concat([existing, params], ignore_index=True)
    params[CV_PARAMS_COLUMNS].to_csv(path, index=False)
    return path
