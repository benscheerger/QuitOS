"""Shared constants for the QuitOS v1 pipeline.

Every value here is fixed up front (see docs/design_v1.md). Nothing is derived
from the data, so importing this module cannot leak information (R7, R8).
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # D10: source files live here, never committed (R1)
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # reserved for our own pipeline outputs
RAW_DATA_PATH = RAW_DATA_DIR / "analytic_sample_1.rds"

# --- Key column names ----------------------------------------------------------
ID_COL = "id"
EMA_ID_COL = "EMA_id"
EMA_NUMBER_COL = "EMA_number"
EMA_DATE_COL = "EMA_date"
EMA_HOUR_COL = "EMA_hour"
STUDY_DAY_COL = "study_day"
EMA_DATETIME_COL = "EMA_date_time"
EVENT_TIME_COL = "event_time"  # derived in preprocess (R4)
CRAVING_COL = "craving_imp"

# --- Flag columns added by preprocess -----------------------------------------
IS_OBSERVED_COL = "is_observed"  # R2: real answer (EMA_date_time not null)
WAS_FILLED_IN_COL = "was_filled_in"  # R2, D11: filled in by the dataset authors
IS_SELF_INITIATED_COL = "is_self_initiated"  # D7: lapse report submitted by participant

# --- Accepted decisions --------------------------------------------------------
HIGH_CRAVING_THRESHOLD = 7  # D1: "high" means craving >= 7
NEXT_SAME_DAY_ONLY = True  # D2: next real answer, same study day only
N_SPLITS = 5  # D4: GroupKFold(k=5) over participants
RANDOM_STATE = 42

CRAVING_MIN = 0
CRAVING_MAX = 10

# --- Column groups -------------------------------------------------------------
# R3: never used as features. Leakage (lapse_lagged = next prompt's lapse),
# post-study follow-up / CO readings, row identifiers and study-admin fields.
EXCLUDED_COLUMNS = (
    "lapse_lagged",
    "smoking_fup",
    "last_time_smoke_fup",
    "CO_reading",
    "CO_reading1",
    "CO_reading2",
    "nr",
    "EMA_number",
    "start_date",
    "end_date",
    "time_first_prompt",
    "time_last_prompt",
    "start_datetime",
    "end_datetime",
    "altered_prompt",
    "altered_prompt_start_time",
    "altered_prompt_study_day",
    "participant_specific_variable_1",
    "participant_specific_variable_2",
    "mpath_data",
    "fitbit_data",
)

# Identifiers and timing columns kept for joins, ordering and splits.
# Not features: id/EMA_id are removed from the feature set in Step 2 (R3).
KEY_COLUMNS = (
    ID_COL,
    EMA_ID_COL,
    EMA_DATE_COL,
    EMA_HOUR_COL,
    STUDY_DAY_COL,
    EMA_DATETIME_COL,
    EVENT_TIME_COL,
)

# Columns the raw file must contain. event_time is derived, so it is not required;
# EMA_number is needed to flag self-initiated reports before it is dropped (D7).
REQUIRED_RAW_COLUMNS = tuple(c for c in KEY_COLUMNS if c != EVENT_TIME_COL) + (
    EMA_NUMBER_COL,
    CRAVING_COL,
)
