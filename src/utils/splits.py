"""Participant-level cross-validation splits (R6, D4).

Folds are built once from the modelling table and shared by every model, so
all comparisons use the same test participants.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config

IndexArray = npt.NDArray[np.intp]
Fold = tuple[IndexArray, IndexArray]


def get_xy(table: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split the modelling table into features, target and participant groups.

    ``X`` is selected from ``config.FEATURE_COLUMNS`` only, in config order, so
    keys (``id``, ``EMA_id``, ``event_time``), targets and metadata such as
    ``target_gap_h`` can never enter it (R3, R5). ``groups`` is the participant
    id used for participant-level splitting (R6).

    Raises:
        ValueError: If ``target`` is not in ``TARGET_COLUMNS`` or a feature
            column is missing.
    """
    if target not in config.TARGET_COLUMNS:
        raise ValueError(f"Unknown target '{target}'; expected one of {config.TARGET_COLUMNS}.")
    missing = [c for c in config.FEATURE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"Table is missing feature columns: {missing}")

    X = table[list(config.FEATURE_COLUMNS)].copy()
    y = table[target].copy()
    groups = table[config.ID_COL].copy()
    return X, y, groups


def participant_folds(table: pd.DataFrame, n_splits: int = config.N_SPLITS) -> list[Fold]:
    """Return ``(train_idx, test_idx)`` positional index arrays, grouped by participant.

    Uses sklearn ``GroupKFold`` with default settings (no shuffle), so folds are
    deterministic and balanced by row count (D4). No participant appears in
    both train and test of the same fold (R6). Use the arrays with ``.iloc``.

    Raises:
        ValueError: If ``n_splits`` < 2 or exceeds the number of participants.
    """
    n_participants = table[config.ID_COL].nunique()
    if n_splits < 2 or n_splits > n_participants:
        raise ValueError(
            f"n_splits must be between 2 and the number of participants "
            f"({n_participants}); got {n_splits}."
        )
    splitter = GroupKFold(n_splits=n_splits)
    return [
        (np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp))
        for train, test in splitter.split(table, groups=table[config.ID_COL])
    ]


def fold_assignments(table: pd.DataFrame, n_splits: int = config.N_SPLITS) -> pd.Series:
    """Map each participant id to the fold in which it is a test participant (D4, D5).

    Returns:
        Integer ``Series`` named ``"fold"``, indexed by participant id, sorted by id.
    """
    ids = table[config.ID_COL].to_numpy()
    assignment: dict[str, int] = {}
    for fold, (_, test_idx) in enumerate(participant_folds(table, n_splits)):
        for pid in np.unique(ids[test_idx]):
            assignment[str(pid)] = fold
    series = pd.Series(assignment, name="fold", dtype="int64")
    series.index.name = config.ID_COL
    return series.sort_index()
