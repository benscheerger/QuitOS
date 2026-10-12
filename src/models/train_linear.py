"""T1: next craving intensity (0–10) with linear models and baselines (§6, D8, D9, D12).

Run with ``python -m src.models.train_linear``. Writes aggregate results only
(R1): no per-row predictions, ``id`` or ``EMA_id`` ever leave this module.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from src import config
from src.data.load_data import load_raw
from src.data.preprocess import preprocess
from src.features.build_features import build_modeling_table
from src.features.preprocessor import make_preprocessor
from src.models.cv import OOFResult, TaskResults, oof_predict, params_frame, update_cv_params
from src.models.evaluate import by_study_day, mae, per_participant, r2, regression_metrics, rmse

TASK = "t1"
TARGET = config.Y_CRAVING_NEXT


def _full_pipeline(model: Any) -> Pipeline:
    """All features, preprocessed inside the fold (R7)."""
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def _craving_only_pipeline(model: Any) -> Pipeline:
    """Current craving only: the simplest learned baseline (§6)."""
    select = ColumnTransformer([("craving", "passthrough", [config.CRAVING_COL])], remainder="drop")
    return Pipeline([("select", select), ("model", model)])


# name -> (factory for a fresh unfitted estimator, inner-CV grid or None)
T1_MODELS: dict[str, tuple[Callable[[], Pipeline], dict[str, Any] | None]] = {
    "linear": (lambda: _full_pipeline(LinearRegression()), None),
    "ridge": (lambda: _full_pipeline(Ridge()), {"model__alpha": config.RIDGE_ALPHAS}),
    "craving_only_linear": (lambda: _craving_only_pipeline(LinearRegression()), None),
}


def clip_t1(pred: np.ndarray) -> tuple[np.ndarray, int]:
    """Clip predictions to the craving range [0, 10] (D9) and count how many moved."""
    pred = np.asarray(pred, dtype=float)
    n_clipped = int(((pred < config.CRAVING_MIN) | (pred > config.CRAVING_MAX)).sum())
    return np.clip(pred, config.CRAVING_MIN, config.CRAVING_MAX), n_clipped


def baseline_predictions(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """T1 baselines (§6). Only ``train_mean`` is fitted, on training folds (R7)."""
    train_mean = oof_predict(table, TARGET, DummyRegressor).pred
    return {
        "persistence": np.asarray(table[config.CRAVING_COL], dtype=float),
        "train_mean": train_mean,
        "person_mean": np.asarray(table["craving_mean_so_far"], dtype=float),
    }


def model_predictions(table: pd.DataFrame) -> dict[str, OOFResult]:
    """Out-of-fold predictions for every T1 model (unclipped)."""
    return {
        name: oof_predict(table, TARGET, factory, param_grid=grid)
        for name, (factory, grid) in T1_MODELS.items()
    }


def _metrics_row(
    name: str, kind: str, y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n_clipped: int
) -> dict[str, Any]:
    row: dict[str, Any] = {"model": name, "kind": kind, "n_rows": len(y)}
    row.update(regression_metrics(y, pred))
    for metric, fn in (("mae", mae), ("rmse", rmse), ("r2", r2)):
        pp = per_participant(fn, y, pred, groups)
        row[f"{metric}_pp_mean"] = pp["mean"]
        row[f"{metric}_pp_sd"] = pp["sd"]
        if metric == "r2":
            row["r2_pp_n_used"] = pp["n_used"]
            row["r2_pp_n_excluded"] = pp["n_excluded"]
    row["n_clipped"] = n_clipped
    return row


def run_t1(table: pd.DataFrame) -> TaskResults:
    """Fit/evaluate all T1 models and baselines; return aggregate tables only."""
    y = np.asarray(table[TARGET], dtype=float)
    groups = np.asarray(table[config.ID_COL])
    study_day = np.asarray(table[config.STUDY_DAY_COL])

    oof = model_predictions(table)
    candidates = [(name, "model", result.pred) for name, result in oof.items()]
    candidates += [(name, "baseline", pred) for name, pred in baseline_predictions(table).items()]

    metric_rows, day_frames = [], []
    for name, kind, raw in candidates:
        pred, n_clipped = clip_t1(raw)  # D9: every T1 prediction is clipped
        metric_rows.append(_metrics_row(name, kind, y, pred, groups, n_clipped))
        day = by_study_day(regression_metrics, y, pred, study_day)
        day.insert(0, "model", name)
        day_frames.append(day)

    metrics = pd.DataFrame(metric_rows)
    # D12: MAE and RMSE are co-primary; a model beats persistence only where it wins.
    reference = metrics.set_index("model").loc["persistence"]
    for metric in ("mae", "rmse"):
        metrics[f"beats_persistence_{metric}"] = metrics[metric] < reference[metric]

    return TaskResults(
        metrics=metrics,
        by_day=pd.concat(day_frames, ignore_index=True),
        params=params_frame(TASK, oof),
    )


def main(table: pd.DataFrame | None = None) -> TaskResults:
    """Run T1 and write ``t1_metrics.csv``, ``t1_by_day.csv`` and ``cv_params.csv``.

    ``table`` defaults to the real modelling table. ``config.RESULTS_DIR`` is
    read at call time so tests can redirect it.
    """
    if table is None:
        table = build_modeling_table(preprocess(load_raw()))
    results = run_t1(table)

    out_dir = config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results.metrics.to_csv(out_dir / "t1_metrics.csv", index=False)
    results.by_day.to_csv(out_dir / "t1_by_day.csv", index=False)
    update_cv_params(out_dir, results.params, TASK)

    shown = ["model", "kind", "mae", "rmse", "r2", "mae_pp_mean", "r2_pp_mean", "n_clipped",
             "beats_persistence_mae", "beats_persistence_rmse"]
    print(results.metrics[shown].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return results


if __name__ == "__main__":
    main()
