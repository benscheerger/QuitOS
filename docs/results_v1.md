# QuitOS — v1 Results

**Status:** v1 baseline results · **Date:** 2026-10-11 · **Design:** [design_v1.md](design_v1.md) (R1–R8, D1–D12)
**Data:** EMASENS `analytic_sample_1` · 4,343 same-day pairs of real answers · 38 participants · 1,665 high next cravings (38.3%)
**Validation:** participant-grouped 5-fold CV (D4). Every score is out-of-fold, for participants the model never saw. Ridge and L2 penalties were tuned by inner CV grouped by participant, on training folds only (D8).

## 1. Summary

1. **Next craving ≈ current craving is the main signal.** 37.9% of next answers exactly equal the current one, so the persistence baseline ("next = current") is very hard to beat.
2. **T1 (next craving intensity): the models beat persistence on RMSE, not MAE (D12).** Ridge cuts RMSE from 2.41 to 2.11, and large misses (off by 3+ points) from 19.9% to 13.6% of predictions. But its typical error is worse: MAE 1.59 vs 1.48.
3. **The other 62 features add a little on T1 and nothing on T2.** On T1, Ridge beats a linear model on current craving alone. On T2, a logistic model on current craving alone matches the full L2 model on ROC-AUC, PR-AUC and Brier. The full model's one advantage is calibration.
4. **Within a person, prediction is weak.** Pooled R² (0.56) and pooled AUC (0.83) are lifted by differences *between* people. Averaged per participant, Ridge R² is 0.08 and AUC is 0.71.
5. **Accumulated history does not help.** Ridge does not catch up with persistence as a participant's study days pass (D5). This makes warm start a low priority.

## 2. T1: next craving intensity (0–10)

Predictions are clipped to [0, 10] (D9). Clipping changed 45 linear and 21 Ridge predictions.

| Model | Type | MAE | RMSE | R² | MAE, mean per person | R², mean per person (SD) |
|---|---|---|---|---|---|---|
| Persistence (next = current) | baseline | **1.485** | 2.411 | 0.425 | **1.476** | −0.14 (0.48) |
| Training-fold mean | baseline | 2.787 | 3.209 | −0.020 | 2.776 | −4.18 (15.3) |
| Person's running mean | baseline | 1.830 | 2.378 | 0.440 | 1.814 | −0.10 (0.10) |
| Linear, current craving only | model | 1.659 | 2.242 | 0.502 | 1.650 | −0.29 (1.70) |
| Linear, all features | model | 1.646 | 2.155 | 0.540 | 1.631 | −0.11 (1.06) |
| **Ridge, all features** (α = 316 in every fold) | model | 1.587 | **2.105** | **0.561** | 1.575 | **0.08** (0.37) |

**Ridge vs. persistence (D12: MAE and RMSE co-primary):**
- **RMSE:** Ridge wins: pooled 2.105 vs 2.411, better in all 5 folds and for 31 of 38 participants.
- **MAE:** Ridge loses: 1.587 vs 1.485, worse in 3 of 5 folds and for 22 of 38 participants.

Why the two metrics disagree:

| | Persistence | Ridge |
|---|---|---|
| Predictions within 0.5 points | **37.9%** | 22.1% |
| Predictions off by 3+ points | 19.9% | **13.6%** |
| MAE on stable pairs (change < 3; 80% of pairs) | **0.69** | 1.16 |
| MAE on big changes (change ≥ 3; 20% of pairs) | 4.68 | **3.31** |

Ridge pulls predictions toward the mean. It gives up exact hits on stable pairs in exchange for smaller misses when craving moves a lot. From a product angle, large swings are the moments that matter. But Ridge still misses them by 3.3 points on average.

**Do the extra features help?** Yes, modestly. Ridge beats the current-craving-only model on RMSE (2.105 vs 2.242) and on MAE (1.587 vs 1.659), although the MAE gain is not clearly resolved (§5). The penalty itself helps too: Ridge beats unpenalised linear regression on RMSE (2.105 vs 2.155).

## 3. T2: next craving is high (≥ 7)

| Model | Type | ROC-AUC | PR-AUC | Brier | AUC, mean per person (SD) |
|---|---|---|---|---|---|
| Training-fold rate of high craving | baseline | 0.463 | 0.362 | 0.237 | 0.50 (0.00) |
| Current craving as a ranking score | baseline | **0.837** | 0.739 | n/a | 0.71 (0.14) |
| Logistic, current craving only | model | 0.833 | **0.752** | **0.159** | 0.71 (0.14) |
| Logistic, all features, no penalty | model | 0.764 | 0.639 | 0.201 | 0.71 (0.15) |
| **Logistic, all features, L2** (C = 0.003–0.01) | model | 0.828 | **0.752** | 0.163 | 0.71 (0.15) |

Per-person AUC excludes 2 of 38 participants, whose next cravings were all high or all not high. Brier scores of ranking scores are not meaningful, so none is given. The PR-AUC of random guessing is about 0.38.

What the table shows:
- **The extra features add nothing to ranking.** The full L2 model and the current-craving-only model are indistinguishable:
  - AUC difference −0.005 (95% interval −0.037 to +0.021);
  - the full model has the better AUC for 21 of 36 participants;
  - the fold-level AUC difference is about +0.02 in 4 folds and −0.06 in the fifth.
- **Without a penalty, logistic regression overfits.** The plain model is overconfident: when it predicts 0.91, the observed rate is 0.69. L2 fixes this (Brier 0.163 vs 0.201).
- **Calibration is the full model's one advantage.** The L2 model stays within about 0.05 of the observed rate in all 10 probability bins (largest miss 0.053). The current-craving-only model misses by up to 0.17 in the middle range (in one bin it predicts 0.33 where 0.15 is observed).

  The reason is the shape of the data. The observed rate of high next craving barely moves across current cravings 2–5 (0.15–0.20), then jumps at 6–7 (0.34, then 0.58). A logistic curve on raw craving can't follow a plateau followed by a jump.

<details>
<summary>Calibration table (10 quantile bins)</summary>

| Bin | L2: mean predicted | L2: observed | Craving-only: mean predicted | Craving-only: observed |
|---|---|---|---|---|
| 0 | 0.038 | 0.044 | 0.038 | 0.084 |
| 1 | 0.074 | 0.085 | 0.064 | 0.095 |
| 2 | 0.131 | 0.124 | 0.112 | 0.145 |
| 3 | 0.213 | 0.212 | 0.185 | 0.206 |
| 4 | 0.311 | 0.294 | 0.327 | 0.153 |
| 5 | 0.407 | 0.392 | 0.454 | 0.337 |
| 6 | 0.498 | 0.449 | 0.568 | 0.568 |
| 7 | 0.600 | 0.622 | 0.640 | 0.742 |
| 8 | 0.698 | 0.751 | 0.717 | 0.804 |
| 9 | 0.826 | 0.860 | 0.837 | 0.844 |

L2 bins hold 434–435 pairs each. The craving-only model gives one prediction per craving level in each fold, so it has few distinct values and its bins are uneven (378–512 pairs).
</details>

## 4. By study day (D5)

| Study day | Pairs | MAE: persistence | MAE: Ridge | RMSE: persistence | RMSE: Ridge | AUC: craving score | AUC: L2 |
|---|---|---|---|---|---|---|---|
| 1 | 451 | 1.77 | 1.85 | 2.71 | 2.40 | 0.79 | 0.80 |
| 2 | 452 | 1.64 | 1.74 | 2.57 | 2.27 | 0.84 | 0.84 |
| 3 | 449 | 1.64 | 1.71 | 2.65 | 2.25 | 0.79 | 0.81 |
| 4 | 451 | 1.48 | 1.53 | 2.39 | 2.07 | 0.84 | 0.82 |
| 5 | 446 | 1.44 | 1.54 | 2.44 | 2.03 | 0.82 | 0.82 |
| 6 | 432 | 1.39 | 1.49 | 2.26 | 1.99 | 0.83 | 0.78 |
| 7 | 434 | 1.36 | 1.50 | 2.26 | 2.01 | 0.88 | 0.86 |
| 8 | 416 | 1.37 | 1.43 | 2.27 | 1.92 | 0.85 | 0.85 |
| 9 | 422 | 1.46 | 1.52 | 2.27 | 2.01 | 0.85 | 0.83 |
| 10 | 390 | 1.27 | 1.53 | 2.16 | 2.02 | 0.89 | 0.86 |

- Every method gets better later in the study, because craving becomes more stable.
- **History features don't close the gap.** Ridge trails persistence on MAE on every day, and its RMSE advantage shrinks from 0.31 on day 1 to 0.14 on day 10.
- The person's running mean, which grows with history, never beats persistence on MAE.
- **Implication:** warm start (refitting on each person's early days, D5) is unlikely to pay off with these linear models.

## 5. Robustness

These numbers come from a one-off analysis of the same out-of-fold predictions. They are **not yet part of the pipeline code**. The 95% intervals come from a participant-level bootstrap: 2,000 resamples of the 38 participants.

| Comparison (first minus second) | Difference | 95% interval | Participants where the first is better |
|---|---|---|---|
| MAE, Ridge − persistence | +0.102 | −0.017 to +0.225 | 16 / 38 |
| RMSE, Ridge − persistence | −0.306 | −0.422 to −0.180 | 31 / 38 |
| RMSE, Ridge − craving-only linear | −0.137 | −0.206 to −0.068 | 27 / 38 |
| MAE, Ridge − craving-only linear | −0.071 | −0.151 to +0.008 | 25 / 38 |
| AUC, L2 − craving-only logistic | −0.005 | −0.037 to +0.021 | 21 / 36 |
| Brier, L2 − craving-only logistic | +0.004 | −0.008 to +0.016 | 22 / 38 |
| Brier, L2 − unpenalised logistic | −0.038 | −0.061 to −0.018 | — |

**Clear results:** Ridge beats persistence and the craving-only model on RMSE, and L2 beats unpenalised logistic regression.
**Not clearly resolved:** the MAE loss to persistence, and every T2 difference between the full and craving-only models.

Results also vary a lot between folds. For example, pooled RMSE for Ridge ranges from 1.59 to 2.46 across folds, and AUC for L2 from 0.70 to 0.90. With 38 participants, single-split results would be unreliable.

## 6. Limitations

- **Small sample:** 38 participants from one study. Results may not carry over to other populations or apps.
- **Short horizon:** "next" means the next real answer on the same study day, typically about 1 hour ahead (D2). Longer horizons are untested.
- **Filled-in rows:** rows the authors filled in are excluded as inputs and test targets (D3, R2). The training-target variant (D11) hasn't been run yet.
- **Decisions made after seeing the prototype:**
  - The co-primary metric rule (D12) and the craving-only baselines were added after a scratch prototype showed the MAE/RMSE split.
  - They make the comparison stricter, not easier. Nevertheless, D12 was not fixed before any results were seen.
- **No holdout:** there is no untouched holdout set (D4), so all numbers are cross-validated estimates on one fixed partition (default `GroupKFold`, no repeats).
- **Self-reports:** craving is self-reported on a 0–10 scale, with a pile-up at 0 (15% of the current answers in the pairs).

## 7. What this means for v1.1

| Option | Motivation from these results |
|---|---|
| Current craving as categories or a spline in T2 | Fixes the plateau-then-jump calibration problem cheaply, and tests whether simple non-linearity matches the full model |
| Median (quantile) regression for T1 | Optimises absolute error directly; the only way a linear model may approach persistence on MAE |
| Predict the *change* in craving | Persistence is the natural reference; this models departures from it directly |
| Run the D11 variant (filled-in rows as extra training targets) | Planned comparison; cheap with the current CV code |
| Move the bootstrap and per-participant win counts into the pipeline | Makes §5 reproducible |
| Warm start | Deprioritised: §4 shows history doesn't help |

## 8. Reproduce

```bash
.venv/bin/python -m src.models.train_linear     # writes results/t1_metrics.csv, t1_by_day.csv, cv_params.csv
.venv/bin/python -m src.models.train_logistic   # writes results/t2_metrics.csv, t2_by_day.csv, t2_calibration.csv
```

Each script takes about 15 s. Results contain aggregates only (R1) and are git-ignored. The dataset must be placed in `data/raw/` (see README).
