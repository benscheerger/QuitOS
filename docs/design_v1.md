# QuitOS — v1 ML Pipeline Design

**Status:** Accepted for v1 (decisions D1–D11 approved 2026-10-06) · **Last updated:** 2026-10-06 · **Source of findings:** `notebooks/01_eda.ipynb`

## 1. Goal and scope

Build a reproducible, leakage-safe baseline pipeline that predicts a participant's **next** craving from information available **now**.

| Task | Target | Model |
|---|---|---|
| T1 Regression | Next craving intensity (0–10) | Linear regression |
| T2 Classification | Next craving is high (binary) | Logistic regression |

**Out of scope for v1:** mobile app, backend/API, database, Fitbit features, advanced ML (boosting, mixed-effects, sequence models), interventions.

## 2. Dataset

EMASENS (Leppin et al., 2026). Files are not redistributed in this repo.

| Property | Value |
|---|---|
| File used | `analytic_sample_1.rds`. `analytic_sample_2` holds 30 of the same participants with identical survey answers, plus Fitbit columns. |
| Participants | 38 |
| Design | 10 study days × 16 hourly prompts, plus 44 lapse reports participants submitted themselves |
| Rows | 6,124 total; 4,722 real answers (77%) |
| Craving | 0–10 integer; on real answers mean 4.9, SD 3.2 |
| Between-person share of variance (ICC) | ≈ 0.48 |
| Correlation between consecutive answers | ≈ 0.71 |

## 3. Data restrictions (hard rules)

These are not up for debate in v1. Each one should be enforced by code or a test.

| ID | Restriction | Reason |
|---|---|---|
| R1 | Dataset files and notebook outputs are never committed. | Public repo; the data is not ours to redistribute. |
| R2 | **Test targets come only from real answers** (`EMA_date_time` not null), in every fold. Filled-in rows may be used for *training* only as described in D11. | 23% of rows (`*_imp` columns) were filled in by the dataset authors and look identical to real answers. They are smoother (SD 2.63 vs. 3.18) and make the task look easier: "next = current" MAE drops from 1.54 to 1.27 when they are included. The fill-in method may also have used future data. Scoring on them would inflate every metric. |
| R3 | Excluded as features: `lapse_lagged`, `lapse_prior`, `smoking_fup`, `last_time_smoke_fup`, `CO_reading*`, IDs (`nr`, `id`, `EMA_id`, `EMA_number`), study-admin columns, `participant_specific_variable_*`. | `lapse_lagged` is the **next** prompt's lapse. `lapse_prior` was built by the authors from filled-in rows in `EMA_id` order; it differs from the real-answer history in 85 real rows and is replaced by our own `prev_lapse`. Follow-up/CO columns were measured after the study period. |
| R4 | Rows are ordered by `id`, then `EMA_date_time`, **not** `EMA_id`. | `EMA_id` order disagrees with actual time in 4 cases. |
| R5 | Features at time *t* use only that participant's information at or before *t*. Per-person statistics use only the *past* (running mean so far, rolling windows). | Prevents temporal leakage. |
| R6 | Participant-level split: no participant appears in both train and test. | ICC ≈ 0.5, so a row-level split would reward memorising individuals. |
| R7 | Anything fitted (scalers, encoders, feature selection, any threshold derived from data) is fit on **training participants only**. | Prevents test-set leakage through preprocessing. |
| R8 | EDA statistics are descriptive only; they must not drive feature selection. | The EDA was computed on all participants, including future test participants. |

## 4. Decisions

The defaults below were accepted on 2026-10-06. The other options stay listed so they can be revisited later.

| ID | Decision | Options | Accepted default | Rationale |
|---|---|---|---|---|
| D1 | "High craving" cutoff | ≥7 (38% positive) · ≥8 (24%) · relative to each person's own level | **≥7, fixed in `config.py`** | Balanced classes; fixed up front, so no leakage. |
| D2 | Meaning of "next" | Next real answer · same study day only · within 3h | **Next real answer, same study day** | Gives a clean ~1h-ahead task. 4,343 pairs, all 38 participants. |
| D3 | Filled-in rows as feature inputs | Exclude · include with a "was filled in" flag · carry forward the last real answer plus "hours since last real answer" | **Exclude**; history features use real answers only, plus hours since the last real answer | Safest given the unknown fill-in method. Carrying forward the last real answer keeps history unbroken using only the past. |
| D4 | Validation scheme | Single holdout · GroupKFold · GroupKFold + final holdout | **GroupKFold (k=5)**, plus an optional final holdout | 38 people is too few for one stable split. |
| D5 | Evaluation scenario | New user (participant unseen) · warm start, **simulated offline on this dataset**: each test participant's first 2–3 days are treated as "onboarding" and added to training, then their later days are scored. No deployed app or real users needed. | **v1: new user only** (the D4 GroupKFold), with test scores also **broken down by study day**. Warm start deferred. | One scheme keeps v1 simple. Person-history features already use each test participant's own earlier answers (no fitting, no leakage), so the by-day breakdown shows how much history helps. Refitting on each person's early days (warm start) is only worth it if that breakdown shows gains, or once models have per-person parameters. |
| D6 | v1 features | See §5 | Person-history plus current survey answers | History captures most of the signal. |
| D7 | Self-submitted lapse reports | Include as rows · scheduled prompts only | **Include**, plus an `is_self_initiated` flag | These are real answers. |
| D8 | Regularisation | Plain · Ridge (L2) | **Plain + Ridge variant**, strength tuned with participant-grouped cross-validation | Many related features, few people. |
| D9 | Regression output range | Raw · clipped to [0, 10] | **Clip** | Craving is bounded. |
| D10 | File location | Keep in `data/processed/` · move to `data/raw/` | **`data/raw/`**: done 2026-10-06; `.gitignore` covers all of `data/` | They are our raw input; matches the README. `data/processed/` is reserved for our own pipeline outputs. |
| D11 | Filled-in rows as **training** targets | Real answers only · real answers plus filled-in rows, with a `was_filled_in` flag | **Real answers only**; the filled-in variant is run as a comparison | More rows add no new information about craving and may copy the authors' fill-in model. Because test scores always use real answers only (R2), the comparison is fair and the effect is measured, not assumed. The variant must keep the **test pairs identical** to the default; filled-in rows may only *add* training pairs. |

## 5. v1 features (all computed at time *t*)

- **Current survey answers:** craving, mood/affect items, motivation, confidence, pain, cigarette availability, substances, location/activity/social context (constant columns dropped), `lapse_event`.
- **Person history (past only):** running mean of craving so far, rolling mean of the last *k* answers, current craving minus the running mean, number of earlier answers, `prev_lapse` (the previous real answer's lapse; replaces `lapse_prior`, see R3).
- **Time:** hour of day, study day, hours since the previous real answer.
- **Baseline traits (fixed per person):** age, sex, cigarettes per day (`cpd`), time to first cigarette, motivation to stop.

**Not features:** `target_gap_h` (hours from *t* to the target answer) is kept as metadata only. It reveals whether the next prompt was skipped, which is future information: 610 of 4,343 pairs have a gap > 1.5 h, and 625 span at least one skipped prompt. "Hours until the next prompt" was dropped for the same reason.

## 6. Evaluation

| Task | Metrics | Comparison baselines |
|---|---|---|
| T1 | MAE, RMSE, R² (pooled **and** averaged per participant) | Persistence (next = current); training-set mean; person's running mean |
| T2 | ROC-AUC, PR-AUC, Brier score, calibration curve (pooled and per participant) | Training-set rate of high craving; persistence (`current ≥ threshold`) |

A model is only useful if it beats **persistence**. On same-day pairs of real answers, persistence has MAE 1.48 vs. 2.75 for the mean.

All metrics are also reported **by study day** of the test participant, to show how much accumulated history helps (see D5).

**Metric notes**
- **T1:** MAE (mean absolute error) is the headline metric. It is in craving points, easy to read, and directly comparable to the baselines. RMSE (the loss linear regression minimises) penalises big misses. Pooled R² is inflated by between-person variance, so per-participant averages are also reported.
- **T2:**
  - ROC-AUC measures how well moments are ranked (0.5 = random).
  - PR-AUC measures the precision/recall trade-off. Its random baseline is the share of high moments (≈ 0.38), not 0.5. Precision maps to alert fatigue, recall to missed cravings.
  - Brier score measures probability accuracy. Always predicting the training rate scores ≈ 0.235.
  - Calibration checks that predicted probabilities match how often high craving actually happens.
- **Per-participant AUC** checks whether the model finds *when* a person craves, not just *who* craves. It can't be computed for participants who are never or always high (1 person at ≥7); report how many were excluded.
- **Accuracy is not used:** always predicting "not high" already scores ≈ 62%.
- **The alert cutoff** (when to intervene) is a later product decision. When chosen, it is set on training folds only and reported as precision and recall at that cutoff.

## 7. Planned leakage tests

| Test file | Asserts |
|---|---|
| `test_splits.py` | No participant ID is in both train and test, in every fold. |
| `test_features.py` | Each target's timestamp is later than its feature row's; **test-fold targets come only from real answers**; filled-in training targets appear only in the D11 variant and always carry `was_filled_in`; excluded columns (R3) are absent; history features match a version computed only from past rows. |
| `test_preprocess.py` | Ordering by `EMA_date_time`; filled-in rows are flagged correctly; scalers are fit on training data only. |

## 8. Known unknowns

- How the dataset authors filled in missing survey answers. This affects D3 and D11. Their paper reportedly used a Kalman filter for the Fitbit data; that kind of method uses values both before and after each gap. The method for the survey answers is unconfirmed.
- Definitions and time alignment of the Fitbit `pd_*` windows. Needed before using `analytic_sample_2`.
- When the CO readings were taken. Assumed to be follow-up; excluded either way.

## 9. Future work (post-v1)

Ordinal or two-part models for the 0–10 target · mixed-effects models · gradient boosting · Fitbit features · warm-start personalisation · intervention timing.
