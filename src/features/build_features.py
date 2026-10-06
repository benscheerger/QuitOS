"""Build the modelling table: one row per real answer that has a same-day next real answer.

Everything here is row-wise or a per-participant cumulative/shift operation
over that participant's *own past*, so it can run on all participants before
the train/test split without leaking information. Nothing is fitted: no
encoding, scaling or column removal (R7, R8).
"""

import pandas as pd

from src import config

_REQUIRED_COLUMNS = (
    config.ID_COL,
    config.EMA_ID_COL,
    config.EVENT_TIME_COL,
    config.STUDY_DAY_COL,
    config.IS_OBSERVED_COL,
    config.IS_SELF_INITIATED_COL,
    *config.EMA_ITEM_COLUMNS,
    *config.BASELINE_NUMERIC,
    *config.CATEGORICAL_FEATURES,
)


def build_modeling_table(processed: pd.DataFrame) -> pd.DataFrame:
    """Turn ``preprocess()`` output into features at time t and targets at t+1.

    Pure function: ``processed`` is not modified.

    Rows and targets
      - Feature rows are real answers only (R2, D3). Filled-in rows are
        dropped first and influence nothing: not features, not history, not
        targets.
      - Target = the participant's next real answer in ``event_time`` order,
        kept only if it is on the same ``study_day`` (D2); otherwise the row
        is dropped. ``y_craving_next`` is its craving; ``y_high_next`` is
        ``int(y >= HIGH_CRAVING_THRESHOLD)`` (D1). Self-initiated reports are
        valid rows and targets (D7).

    Features (``config.FEATURE_COLUMNS``), all known at t (R5, §5)
      - Current answers: all ``EMA_ITEM_COLUMNS``.
      - History over the participant's real answers up to and including t,
        across study days: ``craving_mean_so_far`` (running mean),
        ``craving_roll_mean`` (last ``HISTORY_WINDOW`` answers),
        ``craving_dev_from_mean``, ``n_prev_obs``, ``prev_lapse`` (previous real
        answer's lapse, 0 if none; replaces the authors' ``lapse_prior``, R3).
      - Time: ``hour_of_day`` (hour + minute/60), ``study_day``,
        ``hours_since_prev_obs`` (0 for the first answer), ``is_first_obs``,
        ``is_self_initiated``.
      - Baseline: ``age``, ``cpd``; ``sex``, ``time_to_first_cig``,
        ``motivation_to_stop`` left as strings (encoding is fitted later, R7).

    Metadata (``config.METADATA_COLUMNS``) describes the target row and is
    **never** a feature. ``target_gap_h`` in particular reveals whether the
    next prompt was skipped, which is future information (R5).

    Returns:
        Columns ``id, EMA_id, event_time, *FEATURE_COLUMNS, *TARGET_COLUMNS,
        *METADATA_COLUMNS``, index reset. Booleans are returned as 0/1 ints.

    Raises:
        ValueError: If required columns are missing, ``event_time`` has nulls,
            rows are not sorted by (``id``, ``event_time``) (R4), or any
            feature/target is NaN.
    """
    _validate_input(processed)

    # Real answers only (R2, D3). .loc returns a new frame; the input is untouched.
    obs = processed.loc[processed[config.IS_OBSERVED_COL].astype(bool)].reset_index(drop=True)
    by_person = obs.groupby(config.ID_COL, sort=False)

    craving = obs[config.CRAVING_COL]
    event_time = obs[config.EVENT_TIME_COL]
    clock = pd.DatetimeIndex(event_time)
    one_hour = pd.Timedelta(hours=1)
    n_prev_obs = by_person.cumcount()
    mean_so_far = by_person[config.CRAVING_COL].cumsum() / (n_prev_obs + 1)

    features = pd.DataFrame(
        {
            **{col: obs[col] for col in config.EMA_ITEM_COLUMNS},
            # History (R5): only answers at or before t
            "craving_mean_so_far": mean_so_far,
            "craving_roll_mean": by_person[config.CRAVING_COL].transform(
                lambda s: s.rolling(config.HISTORY_WINDOW, min_periods=1).mean()
            ),
            "craving_dev_from_mean": craving - mean_so_far,
            "n_prev_obs": n_prev_obs,
            "prev_lapse": by_person[config.LAPSE_COL].shift(1).fillna(0.0),
            # Time
            "hour_of_day": (clock.hour + clock.minute / 60).to_numpy(),
            config.STUDY_DAY_COL: obs[config.STUDY_DAY_COL],
            "hours_since_prev_obs": (
                by_person[config.EVENT_TIME_COL].diff() / one_hour
            ).fillna(0.0),
            "is_first_obs": (n_prev_obs == 0).astype(int),
            config.IS_SELF_INITIATED_COL: obs[config.IS_SELF_INITIATED_COL].astype(int),
            # Baseline
            **{col: obs[col] for col in config.BASELINE_NUMERIC},
            **{col: obs[col].astype(str) for col in config.CATEGORICAL_FEATURES},
        }
    )[list(config.FEATURE_COLUMNS)]

    # Target: next real answer of the same participant, same study day (D2)
    next_time = by_person[config.EVENT_TIME_COL].shift(-1)
    next_day = by_person[config.STUDY_DAY_COL].shift(-1)
    has_target = next_time.notna() & (next_day == obs[config.STUDY_DAY_COL])

    y_next = by_person[config.CRAVING_COL].shift(-1)
    targets = pd.DataFrame(
        {
            config.Y_CRAVING_NEXT: y_next,
            config.Y_HIGH_NEXT: y_next >= config.HIGH_CRAVING_THRESHOLD,
        }
    )
    metadata = pd.DataFrame(
        {
            config.TARGET_EMA_ID_COL: by_person[config.EMA_ID_COL].shift(-1),
            config.TARGET_EVENT_TIME_COL: next_time,
            config.TARGET_GAP_H_COL: (next_time - event_time) / one_hour,
            config.TARGET_IS_SELF_INITIATED_COL: by_person[config.IS_SELF_INITIATED_COL].shift(-1),
        }
    )

    keys = obs[[config.ID_COL, config.EMA_ID_COL, config.EVENT_TIME_COL]]
    out = pd.concat([keys, features, targets, metadata], axis=1).loc[has_target]
    out = out.reset_index(drop=True)
    out[config.Y_HIGH_NEXT] = out[config.Y_HIGH_NEXT].astype(int)
    out[config.TARGET_IS_SELF_INITIATED_COL] = out[config.TARGET_IS_SELF_INITIATED_COL].astype(int)

    nan_cols = [
        c for c in (*config.FEATURE_COLUMNS, *config.TARGET_COLUMNS) if out[c].isna().any()
    ]
    if nan_cols:
        raise ValueError(f"NaN in modelling table columns: {nan_cols}")
    return out


def _validate_input(df: pd.DataFrame) -> None:
    """Check required columns and the R4 ordering that ``preprocess()`` guarantees."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    ids = df[config.ID_COL]
    times = df[config.EVENT_TIME_COL]
    n_null = times.isna().sum()
    if n_null:
        raise ValueError(f"Column '{config.EVENT_TIME_COL}' has {n_null} null value(s).")

    prev_ids = ids.shift()
    same_person = ids == prev_ids
    out_of_order = (ids < prev_ids) | (same_person & (times < times.shift()))
    n_bad = out_of_order.sum()
    if n_bad:
        raise ValueError(
            f"Rows are not sorted by ({config.ID_COL}, {config.EVENT_TIME_COL}): "
            f"{n_bad} row(s) out of order (R4). Run preprocess() first."
        )
