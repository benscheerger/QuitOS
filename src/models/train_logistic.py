"""T2: next craving is high (>= 7) with logistic models and baselines (§6, D1, D8).

Run with ``python -m src.models.train_logistic``. Writes aggregate results only
(R1): no per-row predictions, ``id`` or ``EMA_id`` ever leave this module.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data.load_data import load_raw
from src.data.preprocess import preprocess
from src.features.build_features import build_modeling_table
from src.features.preprocessor import make_preprocessor
from src.models.cv import OOFResult, TaskResults, oof_predict, params_frame, update_cv_params
from src.models.evaluate import (
    brier,
    by_study_day,
    calibration_table,
    classification_metrics,
    per_participant,
    pr_auc,
    roc_auc,
)

TASK = "t2"
TARGET = config.Y_HIGH_NEXT
MAX_ITER = 5000


def _full_pipeline(model: Any) -> Pipeline:
    """All features, preprocessed inside the fold (R7)."""
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def _craving_only_pipeline(model: Any) -> Pipeline:
    """Current craving only: the simplest learned baseline (§6)."""
    select = ColumnTransformer([("craving", "passthrough", [config.CRAVING_COL])], remainder="drop")
    return Pipeline([("select", select), ("model", model)])


def _unpenalised() -> LogisticRegression:
    # C=np.inf is sklearn's replacement for penalty=None (deprecated in 1.8,
    # removed in 1.10); it fits the same unpenalised model.
    return LogisticRegression(C=np.inf, max_iter=MAX_ITER)


# name -> (factory for a fresh unfitted estimator, inner-CV grid or None)
T2_MODELS: dict[str, tuple[Callable[[], Pipeline], dict[str, Any] | None]] = {
    "logistic": (lambda: _full_pipeline(_unpenalised()), None),
    "logistic_l2": (
        lambda: _full_pipeline(LogisticRegression(max_iter=MAX_ITER)),
        {"model__C": config.LOGISTIC_CS},
    ),
    "craving_only_logistic": (lambda: _craving_only_pipeline(_unpenalised()), None),
}


def baseline_predictions(table: pd.DataFrame) -> dict[str, tuple[np.ndarray, bool]]:
    """T2 baselines (§6) as ``name -> (score, is_probability)``.

    ``prevalence`` is the training-fold rate of high craving (fitted on
    training folds only, R7). ``craving_score`` ranks rows by current craving;
    it is not a probability, so only AUC and PR-AUC apply.
    """
    prevalence = oof_predict(
        table, TARGET, lambda: DummyClassifier(strategy="prior"), proba=True
    ).pred
    return {
        "prevalence": (prevalence, True),
        "craving_score": (np.asarray(table[config.CRAVING_COL], dtype=float), False),
    }


def model_predictions(table: pd.DataFrame) -> dict[str, OOFResult]:
    """Out-of-fold predicted probabilities for every T2 model."""
    return {
        name: oof_predict(table, TARGET, factory, param_grid=grid, proba=True)
        for name, (factory, grid) in T2_MODELS.items()
    }


def _metrics_row(
    name: str, kind: str, y: np.ndarray, score: np.ndarray, groups: np.ndarray, is_proba: bool
) -> dict[str, Any]:
    row: dict[str, Any] = {"model": name, "kind": kind, "n_rows": len(y)}
    row.update(classification_metrics(y, score, proba=is_proba))
    for metric, fn in (("auc", roc_auc), ("pr_auc", pr_auc)):
        pp = per_participant(fn, y, score, groups)
        for key in ("mean", "sd", "n_used", "n_excluded"):
            row[f"{metric}_pp_{key}"] = pp[key]
    if is_proba:
        pp = per_participant(brier, y, score, groups)
        row["brier_pp_mean"], row["brier_pp_sd"] = pp["mean"], pp["sd"]
    else:
        row["brier_pp_mean"] = row["brier_pp_sd"] = float("nan")
    return row


def run_t2(table: pd.DataFrame) -> TaskResults:
    """Fit/evaluate all T2 models and baselines; return aggregate tables only."""
    y = np.asarray(table[TARGET], dtype=float)
    groups = np.asarray(table[config.ID_COL])
    study_day = np.asarray(table[config.STUDY_DAY_COL])

    oof = model_predictions(table)
    candidates = [(name, "model", result.pred, True) for name, result in oof.items()]
    candidates += [
        (name, "baseline", score, is_proba)
        for name, (score, is_proba) in baseline_predictions(table).items()
    ]

    metric_rows, day_frames, calibration_frames = [], [], []
    for name, kind, score, is_proba in candidates:
        metric_rows.append(_metrics_row(name, kind, y, score, groups, is_proba))

        def day_metrics(y_day, s_day, is_proba=is_proba):
            return classification_metrics(y_day, s_day, proba=is_proba)

        day = by_study_day(day_metrics, y, score, study_day)
        day.insert(0, "model", name)
        day_frames.append(day)
        if is_proba:
            calibration = calibration_table(y, score)
            calibration.insert(0, "model", name)
            calibration_frames.append(calibration)

    metrics = pd.DataFrame(metric_rows)
    # Same rule as D12: a model beats a baseline only on the metric where it wins.
    ref = metrics.set_index("model")
    metrics["beats_craving_score_auc"] = metrics["roc_auc"] > ref.loc["craving_score", "roc_auc"]
    metrics["beats_craving_score_pr_auc"] = metrics["pr_auc"] > ref.loc["craving_score", "pr_auc"]
    metrics["beats_prevalence_brier"] = metrics["brier"] < ref.loc["prevalence", "brier"]

    return TaskResults(
        metrics=metrics,
        by_day=pd.concat(day_frames, ignore_index=True),
        params=params_frame(TASK, oof),
        calibration=pd.concat(calibration_frames, ignore_index=True),
    )


def main(table: pd.DataFrame | None = None) -> TaskResults:
    """Run T2 and write ``t2_metrics.csv``, ``t2_by_day.csv``,
    ``t2_calibration.csv`` and ``cv_params.csv``.

    ``table`` defaults to the real modelling table. ``config.RESULTS_DIR`` is
    read at call time so tests can redirect it.
    """
    if table is None:
        table = build_modeling_table(preprocess(load_raw()))
    results = run_t2(table)

    out_dir = config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results.metrics.to_csv(out_dir / "t2_metrics.csv", index=False)
    results.by_day.to_csv(out_dir / "t2_by_day.csv", index=False)
    if results.calibration is not None:
        results.calibration.to_csv(out_dir / "t2_calibration.csv", index=False)
    update_cv_params(out_dir, results.params, TASK)

    shown = ["model", "kind", "roc_auc", "pr_auc", "brier", "auc_pp_mean", "auc_pp_n_excluded",
             "beats_craving_score_auc", "beats_prevalence_brier"]
    print(results.metrics[shown].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return results


if __name__ == "__main__":
    main()
