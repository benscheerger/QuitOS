"""Turn the validated raw table into a time-ordered table of observations.

No statistics are computed and nothing is fitted here (R7, R8). Every
transformation is row-wise or a sort, so it can run on all participants
before the train/test split without leaking information.
"""

import pandas as pd

from src import config

_SEQ_COL = "_ema_seq"  # temporary sort key, removed before returning


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add flags, drop excluded columns, build ``event_time`` and sort.

    Pure function: ``df`` is not modified and a new DataFrame is returned.

    Steps:
      1. Flags, computed before any column is dropped:
         - ``is_observed`` = ``EMA_date_time`` is not null (R2: only these rows
           may be test targets);
         - ``was_filled_in`` = not ``is_observed`` (R2, D3, D11);
         - ``is_self_initiated`` = ``EMA_number`` is null (D7). ``EMA_number``
           is dropped in step 2, so this must come first.
      2. Drop ``EXCLUDED_COLUMNS`` (R3). Columns already absent are ignored.
      3. ``event_time`` = ``EMA_date_time``; for filled-in rows (no timestamp)
         it falls back to ``EMA_date`` + ``EMA_hour`` hours (R4).
      4. Sort by ``id``, ``event_time``, then the integer sequence number in
         ``EMA_id`` (R4: ``EMA_id`` order alone disagrees with real time).
      5. Keep every row, including filled-in rows (D11 needs them for the
         comparison variant; R2 filtering happens later), and keep the
         original column names. The index is reset to 0..n-1.

    Raises:
        ValueError: If an ``EMA_id`` has no integer suffix after its last ``_``.
    """
    out = df.copy()

    # 1. Flags (R2, D7, D11)
    out[config.IS_OBSERVED_COL] = out[config.EMA_DATETIME_COL].notna()
    out[config.WAS_FILLED_IN_COL] = ~out[config.IS_OBSERVED_COL]
    out[config.IS_SELF_INITIATED_COL] = out[config.EMA_NUMBER_COL].isna()

    # 2. Exclusions (R3)
    out = out.drop(columns=list(config.EXCLUDED_COLUMNS), errors="ignore")

    # 3. Event time (R4)
    observed_time = out[config.EMA_DATETIME_COL]
    scheduled_time = pd.to_datetime(out[config.EMA_DATE_COL]) + pd.to_timedelta(
        out[config.EMA_HOUR_COL], unit="h"
    )
    # Match EMA_date_time's resolution (datetime64[s] in the real file) so the
    # fillna below does not silently change the column's dtype.
    if pd.api.types.is_datetime64_dtype(observed_time):
        scheduled_time = scheduled_time.astype(observed_time.dtype)
    out[config.EVENT_TIME_COL] = observed_time.fillna(scheduled_time)

    # 4. Ordering (R4)
    out[_SEQ_COL] = _ema_sequence(out[config.EMA_ID_COL])
    out = out.sort_values(
        [config.ID_COL, config.EVENT_TIME_COL, _SEQ_COL], kind="mergesort"
    ).drop(columns=_SEQ_COL)

    # 5. All rows kept, fresh index
    return out.reset_index(drop=True)


def _ema_sequence(ema_id: pd.Series) -> pd.Series:
    """Return the integer after the last ``_`` in each ``EMA_id`` (e.g. ``p1_10`` -> 10).

    Used as the final tie-breaker in the R4 sort. Parsed as an integer because
    a text sort would put ``p1_10`` before ``p1_9``.
    """
    as_text = ema_id.astype(str)
    n_bad = (~as_text.str.fullmatch(r".*_\d+")).sum()
    if n_bad:
        raise ValueError(
            f"{n_bad} '{config.EMA_ID_COL}' value(s) lack an integer suffix after '_'."
        )
    return as_text.str.rsplit("_", n=1).str[-1].astype("int64")
