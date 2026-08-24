# Flat 0.5/0.5 DNN+XGBoost ensemble (P3)

**Status:** shipped, current primary model. First implemented v26.x, confirmed on three separate
runs including post-P5 (see `specs/completed/xgboost-hyperparameter-sweep.md`).

## What shipped

`pred_ens = 0.5 * pred_dnn + 0.5 * pred_xgb`, computed directly on the two models' already-stored
test-set forecasts — zero retraining, one line. Added as a fourth row (`Ensemble`) in
`results_summary`, alongside DNN, XGBoost, and Naive.

## Why

The two base models' error bursts don't fully coincide (different hours miss differently in the
forecast-vs-actual plots), and their individual test MAPEs sit within a fraction of a point of each
other — the classic setup where a 50/50 average beats both.

## Evidence

Confirmed on three separate full pipeline runs, including after the P5 XGBoost regularization
sweep: the Ensemble beat both individual models on **every** metric each time — not just
MAPE/WAPE (relative) but also MAE and RMSE (absolute, MW), meaning it's not just better on average,
it's more robust on the worst days too (RMSE penalizes large misses more).

Current honest numbers (post weather hour-convention fix, oracle scenario, 59 rolling days):
Ensemble 8.46% MAPE / 7.14% WAPE / 81.3 MW MAE / 115.8 MW RMSE — still beating both DNN (8.81%) and
XGBoost (8.53%) individually. See `MACHINE LEARNING RD CONTEXT.md` §12.11 for the full table and
the leakage-fix history behind why these numbers moved from earlier (leakier) figures.

A weighted-ensemble refinement (validation-tuned weight, not flat 50/50) was tried, tested,
adversarially reviewed, and executed — it did not beat the flat 0.5/0.5 on the actual test set. See
`decisions.md` D-01 for the full story; that code was removed per user request after the negative
result was recorded.

## Verification

Adversarially reviewed (slices/shapes/leakage all confirmed clean) at the point it was first added.
Re-verified end-to-end on every subsequent full pipeline rerun this project has done since
(leakage-fix reruns, lag-semantics rerun, weather hour-fix rerun) — the Ensemble has never lost its
"best model" position in the oracle scenario across any of those reruns.

## Revert

No dedicated revert — this is a two-line computation with no dependencies beyond the two base
models' stored forecasts. Removing the `Ensemble` row from `_summary_rows` reverts it.

## Source

`MACHINE LEARNING RD CONTEXT.md` §8 rows 9, 17, 18; §9.1; §9.2; §10 (P3); §12.11. See also
`decisions.md` D-01.
