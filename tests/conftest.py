"""Shared synthetic fixtures. No real EMASENS data is used here (R1)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src import config

DAY = dt.date(2026, 1, 5)


def _ts(hour: int, minute: int) -> pd.Timestamp:
    return pd.Timestamp(2026, 1, 5, hour, minute)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small raw-shaped table, with rows deliberately out of order.

    Participant p1 (true time order: p1_8, p1_9, p1_11, p1_10):
      - p1_8   scheduled, answered 09:05
      - p1_9   scheduled, **filled in** (no timestamp) -> event_time 10:00
      - p1_10  scheduled 11:00 prompt, answered 11:40
      - p1_11  **self-initiated lapse report** (EMA_number missing), submitted
               11:10, i.e. *before* the 11:00 prompt was answered, but listed
               *after* it by EMA_id (the R4 inversion)
    Participant p2: two ordinary answered prompts.

    EMA_id suffixes 8..11 cross the 9 -> 10 boundary, so a text sort of
    EMA_id would order them wrongly. Every R3 excluded column is present.
    """
    rows = [
        # EMA_id, EMA_number, hour, EMA_date_time, craving, lapse_event
        ("p2_2", 2.0, 10.0, _ts(10, 15), 3.0, 0.0),
        ("p1_11", np.nan, 11.0, _ts(11, 10), 9.0, 1.0),
        ("p1_8", 1.0, 9.0, _ts(9, 5), 4.0, 0.0),
        ("p2_1", 1.0, 9.0, _ts(9, 20), 2.0, 0.0),
        ("p1_10", 3.0, 11.0, _ts(11, 40), 7.0, 0.0),
        ("p1_9", 2.0, 10.0, pd.NaT, 5.0, 0.0),
    ]
    df = pd.DataFrame(
        {
            config.ID_COL: [r[0].split("_")[0] for r in rows],
            config.EMA_ID_COL: [r[0] for r in rows],
            config.EMA_NUMBER_COL: [r[1] for r in rows],
            config.EMA_DATE_COL: [DAY] * len(rows),
            config.EMA_HOUR_COL: [r[2] for r in rows],
            config.STUDY_DAY_COL: np.ones(len(rows), dtype="int32"),
            config.EMA_DATETIME_COL: pd.Series(
                [r[3] for r in rows], dtype="datetime64[s]"
            ),
            config.CRAVING_COL: [r[4] for r in rows],
            "lapse_event_imp": [r[5] for r in rows],
            "lapse_prior": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    df[config.ID_COL] = df[config.ID_COL].astype(str)
    df[config.EMA_ID_COL] = df[config.EMA_ID_COL].astype(str)
    for col in config.EXCLUDED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


BASELINE = {
    "p1": {"age": 34.0, "cpd": 12.0, "sex": "female",
           "time_to_first_cig": "Within 5 minutes", "motivation_to_stop": "Very high"},
    "p2": {"age": 51.0, "cpd": 20.0, "sex": "male",
           "time_to_first_cig": "6-30 minutes", "motivation_to_stop": "Medium"},
}


@pytest.fixture
def features_raw_df(raw_df) -> pd.DataFrame:
    """``raw_df`` plus what Step 2 needs. Run it through ``preprocess()`` first.

    Adds per-participant baseline columns, every missing ``EMA_ITEM_COLUMNS``
    entry (fixed 0.0), and one more p1 row:
      - p1_12  scheduled, **study_day 2**, answered 09:30 next morning, craving 2

    Real answers in time order: p1_8, p1_11, p1_10 | p1_12 (day 2); p2_1, p2_2.
    Expected modelling rows: p1_8 -> p1_11, p1_11 -> p1_10, p2_1 -> p2_2.
    p1_10 has no same-day next answer, so it is dropped (D2).
    """
    df = raw_df.copy()
    for col in config.EMA_ITEM_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    day2 = df.loc[df[config.EMA_ID_COL] == "p1_8"].copy()
    day2[config.EMA_ID_COL] = "p1_12"
    day2[config.EMA_NUMBER_COL] = 1.0
    day2[config.EMA_DATE_COL] = [DAY + dt.timedelta(days=1)]
    day2[config.EMA_HOUR_COL] = 9.0
    day2[config.STUDY_DAY_COL] = np.int32(2)
    day2[config.EMA_DATETIME_COL] = pd.Timestamp(2026, 1, 6, 9, 30)
    day2[config.CRAVING_COL] = 2.0
    day2[config.LAPSE_COL] = 0.0
    df = pd.concat([df, day2], ignore_index=True)

    for col in (*config.BASELINE_NUMERIC, *config.CATEGORICAL_FEATURES):
        df[col] = df[config.ID_COL].map({pid: vals[col] for pid, vals in BASELINE.items()})
    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    return df


CATEGORY_LEVELS = {
    "sex": ["female", "male"],
    "time_to_first_cig": ["Within 5 minutes", "6-30 minutes", "31-60 minutes"],
    "motivation_to_stop": ["Low", "Medium", "High"],
}


@pytest.fixture
def modeling_table() -> pd.DataFrame:
    """A synthetic table shaped like ``build_modeling_table()`` output.

    10 participants (p00..p09) with uneven row counts (3, 4, ..., 12 rows), all
    ``FEATURE_COLUMNS``, targets and metadata. Values are random but seeded.
    Baseline columns are constant within each participant, as in the real data.
    """
    rng = np.random.default_rng(42)
    parts = []
    for i in range(10):
        pid = f"p{i:02d}"
        n = 3 + i
        times = pd.date_range("2026-01-05 09:00", periods=n, freq="h").astype("datetime64[s]")
        y = rng.integers(0, 11, n).astype(float).tolist()
        cols: dict[str, object] = {
            config.ID_COL: [pid] * n,
            config.EMA_ID_COL: [f"{pid}_{k}" for k in range(1, n + 1)],
            config.EVENT_TIME_COL: times,
        }
        for col in config.NUMERIC_FEATURES:
            cols[col] = rng.integers(0, 11, n).astype(float).tolist()
        for col in config.BASELINE_NUMERIC:
            cols[col] = [float(rng.integers(18, 70))] * n
        for col in config.CATEGORICAL_FEATURES:
            cols[col] = [str(rng.choice(CATEGORY_LEVELS[col]))] * n
        cols[config.Y_CRAVING_NEXT] = y
        cols[config.Y_HIGH_NEXT] = [int(v >= config.HIGH_CRAVING_THRESHOLD) for v in y]
        cols[config.TARGET_EMA_ID_COL] = [f"{pid}_{k}" for k in range(2, n + 2)]
        cols[config.TARGET_EVENT_TIME_COL] = times + pd.Timedelta(hours=1)
        cols[config.TARGET_GAP_H_COL] = [1.0] * n
        cols[config.TARGET_IS_SELF_INITIATED_COL] = [0] * n
        parts.append(pd.DataFrame(cols))
    table = pd.concat(parts, ignore_index=True)
    columns = [
        config.ID_COL,
        config.EMA_ID_COL,
        config.EVENT_TIME_COL,
        *config.FEATURE_COLUMNS,
        *config.TARGET_COLUMNS,
        *config.METADATA_COLUMNS,
    ]
    return table[columns]
