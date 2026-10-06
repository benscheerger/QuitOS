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
    # Authors' lapse_prior is built from filled-in rows in EMA_id order and
    # disagrees with the real-answer history; replaced by prev_lapse.
    "lapse_prior",
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

# --- Step 2: features and targets ------------------------------------------------
LAPSE_COL = "lapse_event_imp"
HISTORY_WINDOW = 3  # rolling mean over the last k real answers (§5)

# All 48 survey items in the raw file, in file order. "_imp" = filled in where
# missing by the dataset authors; on feature rows only real answers are used (D3).
EMA_ITEM_COLUMNS = (
    "excited_imp",
    "cigarette_availability_imp",
    "calm_imp",
    "bored_imp",
    "enthusiastic_imp",
    "irritable_imp",
    "anxious_imp",
    "contented_imp",
    "lapse_event_imp",
    "stressed_imp",
    "sad_imp",
    "motivation_imp",
    "craving_imp",
    "caffeine_imp",
    "happy_imp",
    "confidence_imp",
    "pain_imp",
    "alcohol_imp",
    "nicotine_imp",
    "location_home_imp",
    "location_school_work_imp",
    "location_outside_imp",
    "location_restaurant_imp",
    "location_public_place_imp",
    "location_public_transport_imp",
    "location_private_vehicle_imp",
    "location_others_home_imp",
    "location_other_imp",
    "activity_eating_imp",
    "activity_tv_imp",
    "activity_music_imp",
    "activity_reading_imp",
    "activity_working_imp",
    "activity_walking_imp",
    "activity_child_care_imp",
    "activity_socialising_imp",
    "activity_social_media_imp",
    "activity_relaxing_imp",
    "activity_chores_imp",
    "activity_other_imp",
    "social_context_alone_imp",
    "social_context_partner_imp",
    "social_context_friend_imp",
    "social_context_child_imp",
    "social_context_relative_imp",
    "social_context_colleague_imp",
    "social_context_stranger_imp",
    "social_context_other_imp",
)

# Person-history features (R5: past and current answer only)
HISTORY_FEATURES = (
    "craving_mean_so_far",
    "craving_roll_mean",
    "craving_dev_from_mean",
    "n_prev_obs",
    "prev_lapse",
)
# Time and context features known at t. No "time until next prompt": that
# depends on whether future prompts are answered.
TIME_FEATURES = (
    "hour_of_day",
    STUDY_DAY_COL,
    "hours_since_prev_obs",
    "is_first_obs",
    IS_SELF_INITIATED_COL,
)
# Baseline traits, fixed per person (§5)
BASELINE_NUMERIC = ("age", "cpd")
CATEGORICAL_FEATURES = ("sex", "time_to_first_cig", "motivation_to_stop")

NUMERIC_FEATURES = EMA_ITEM_COLUMNS + HISTORY_FEATURES + TIME_FEATURES + BASELINE_NUMERIC
# Fixed order. Categorical columns stay as strings; encoding is fitted later on
# training participants only (R7).
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Targets (D1, D2)
Y_CRAVING_NEXT = "y_craving_next"
Y_HIGH_NEXT = "y_high_next"
TARGET_COLUMNS = (Y_CRAVING_NEXT, Y_HIGH_NEXT)

# Metadata about the target row. Never features: target_gap_h, for example,
# reveals whether the next prompt was skipped (future information, R5).
TARGET_EMA_ID_COL = "target_EMA_id"
TARGET_EVENT_TIME_COL = "target_event_time"
TARGET_GAP_H_COL = "target_gap_h"
TARGET_IS_SELF_INITIATED_COL = "target_is_self_initiated"
METADATA_COLUMNS = (
    TARGET_EMA_ID_COL,
    TARGET_EVENT_TIME_COL,
    TARGET_GAP_H_COL,
    TARGET_IS_SELF_INITIATED_COL,
)
