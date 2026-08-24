# XGBoost hyperparameter sweep (P5)

**Status:** shipped, current. Implemented, unit-tested, adversarially reviewed, and executed
(v26.10).

## What shipped

A small grid search (6 candidates including the original config as baseline) over `max_depth`,
`min_child_weight`, `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`, run once on the
**first** rolling window's (`T`, not `T_last`) internal validation split — before the main rolling
loop. The winning config is then applied fixed across all rolling iterations; it is not re-tuned
per rolling day. See `decisions.md` D-02 for why `T` and not `T_last`, and why not per-day retuning.

## Why

A direct pipeline run measured XGBoost's overfit gap (train 4.54% → test 7.35% MAPE at the time,
2.82 pp) at roughly 4x the DNN's (0.73 pp) for nearly identical test MAPE — meaning XGBoost was
fitting training noise the DNN wasn't. The original config (`max_depth=5, reg_alpha=0.1,
reg_lambda=1.0`) was more permissive than the data supported; this sweep searches for a more
regularized config using only data provably before the test period.

## Evidence

Selected `max_depth=4, min_child_weight=5, reg_alpha=0.3, reg_lambda=2.0, subsample=0.8,
colsample_bytree=0.8` — lower validation MAPE (4.658% vs 4.680%) and a much smaller train/val gap
(0.694 pp vs 1.299 pp) than the original config. Applied across all rolling iterations at the time,
this improved XGBoost on all four test metrics (MAPE 7.35%→7.10%, WAPE 6.20%→6.07%, MAE 70.7→69.3
MW, RMSE 102.3→99.2 MW), pulling the flat Ensemble to a new best at the time.

The config remains in place through every subsequent leakage/alignment fix this project has done;
current honest numbers (post weather hour-convention fix) are in `MACHINE LEARNING RD CONTEXT.md`
§12.11.

## Design correction made *before* running anything (worth its own record)

The first draft of this sweep (and the weighted-ensemble weight search built the same way) used the
**last** rolling window's (`T_last`) validation split. Caught during unit-test-writing, before any
execution, that `T_last`'s training window overlaps 98.3% of the reported test period's rows —
tuning against it would leak actual test-period outcomes into hyperparameter selection. Redesigned
to use `T` instead. See `mistakes.md` M-08 for the full incident record.

## Verification

Verified with a standalone unit test that reproduces both the leak (rejected `T_last` design) and
its absence (shipped `T` design) side by side. Independently adversarially reviewed: confirmed
clean on all 6 checked points (no leakage, correct row/scaling alignment, `.py`/`.ipynb` parity,
nothing downstream broken, sweep baseline sanity) — flagged two pre-existing, out-of-scope
observations inherited from the original rolling-loop design, not introduced by this change.

## Revert

Reverting the sweep restores the original hardcoded config (`max_depth=5, min_child_weight=1,
reg_alpha=0.1, reg_lambda=1.0`) for the rolling loop. See `MACHINE LEARNING RD CONTEXT.md` §11 for
the exact commit sequencing (P5 and the weighted-ensemble were implemented in the same commit,
`d473e98`, since they share one pipeline run — see that section for how to isolate just this piece).

## Source

`MACHINE LEARNING RD CONTEXT.md` §8 rows 16–17; §9.2; §10 (P5); §11. See also `decisions.md` D-02,
`mistakes.md` M-08.
