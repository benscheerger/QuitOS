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
