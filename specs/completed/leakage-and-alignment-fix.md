# Leakage and time-alignment correctness batch (external review #3, #11-demand-half, #12)

**Status:** shipped, current. Three model-affecting correctness fixes, done together and validated
in one rerun (2026-07-23).

## What shipped

1. **Solar calibration leakage fixed (#3, critical).** Replaced the full-year per-hour
   `real_solar/ninja_solar` ratio (which used target-hour real solar) with
   `rebuild_calibration_features(df_in, T)` — refits the ratio from training rows (`< T`) only, per
   rolling window, applied using only ninja inputs. Raw ninja helper columns structurally excluded
   from the model's feature list. See `mistakes.md` M-01.
2. **Demand/generation hour-ending alignment fixed (#11, demand half).** Demand loader remapped
   `H_k → k:00` (was `(k-1):00`), matching the generation file's hour-ending convention. See
   `mistakes.md` M-02. (The weather side of this same convention question was left open at this
   point — see `specs/completed/weather-hour-convention-fix.md` for how that was closed later.)
3. **Index integrity checks added (#12).** Explicit `assert`s for duplicate and missing hourly
   timestamps, placed before any row-count-based shift/lag feature. See `mistakes.md` M-09.

## Why

The external reviewer's #1 instruction: "fix the leakage and time alignment first, then rerun both
models on a clean holdout period." None of the three fixes is a score-chasing tuning choice — each
is justified by the data-generating process independent of whether the resulting score went up or
down (it went up, i.e. got worse, which is itself evidence the fixes weren't chosen to flatter the
score) — see `decisions.md` D-08 for the holdout-discipline reasoning this established.

## Evidence — honest before/after (same 58-day test window, corrected data)

| Model | MAPE% before → after | WAPE% | MAE MW | RMSE MW |
|---|---|---|---|---|
| DNN | 7.01 → 8.65 | 6.11 → 7.26 | 69.7 → 82.8 | 96.6 → 115.3 |
| XGBoost | 7.10 → 8.36 | 6.07 → 7.00 | 69.3 → 79.8 | 99.2 → 111.6 |
| **Ensemble** | 6.78 → 8.33 | 5.83 → 6.95 | 66.5 → 79.3 | 94.3 → 110.7 |
| Naive | 11.57 → 11.66 | — | — | 165.0 → 163.9 |

Numbers went up ~1.5 pp MAPE, exactly as the reviewer predicted the leak would make the reported
scores too optimistic. The flat Ensemble remained the best model and still beat naive by ~28%.
(These numbers were themselves later superseded by the weather hour-convention fix — see
`MACHINE LEARNING RD CONTEXT.md` §12.11 for current figures. Kept here as the honest record of
*this* batch's specific effect.)

## Verification

Independent adversarial subagent review: clean on all 6 checked points — (1) calibration factor fit
strictly on rows `< T`; (2) `_h24` alignment sound given the enforced gap-free index; (3) raw-ninja
helpers structurally excluded from the model feature list; (4) `rebuild_calibration_features` called
per window with the correct `T`; (5) the `shift(-HORIZON)` NaN tail fully trimmed; (6) the demand
hour-ending remap self-consistent, nothing downstream assumed the old mapping. Plus a standalone
unit test (`scratchpad/test_calibration_leakage.py`) proving the calibrated irradiance is invariant
to perturbing test-period real solar, and that the removed full-year ratio would have leaked.

## Revert

`git revert d7c875d` restores the leaky calibration, the hour-beginning demand mapping, and removes
the index checks — do not, except to reproduce the old inflated numbers for a comparison table. The
three fixes are intertwined in one commit (validated in one rerun together); to isolate just one,
revert and re-apply selectively.

## Source

`MACHINE LEARNING RD CONTEXT.md` §12.6. See also `mistakes.md` M-01, M-02, M-09; `decisions.md`
D-08.
