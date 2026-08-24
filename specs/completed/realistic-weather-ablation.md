# Realistic (non-oracle) weather experiment (external review #4)

**Status:** shipped, current. New standalone script (2026-07-23), not merged into the main
pipeline.

## What shipped

`experiments/realistic_weather_ablation.py` — the main pipeline with exactly one structural
change: the four `*_h24` weather features are dropped entirely (26 model features instead of 30).
Reuses the main pipeline's leakage-safe calibration rebuild, hydro-profile rebuild, hour-ending
demand alignment, and index-integrity check verbatim. Current (non-shifted) weather stays — that's
legitimately observed "now" data, same class as `demanda_residual` or `hidro_mw`. Results write to
`results_summary_realistic.csv`/`forecasts_realistic.csv`, distinct from the main pipeline's output
files so running both doesn't clobber either.

## Why

The main pipeline's h+24 weather features use the true future MERRA-2 value, not an operational
forecast — a disclosed but unquantified "perfect-foresight" upper bound. The external review asked
for a genuine second experiment that does *not* use future observed weather, so the oracle number
has something honest to compare against. See `decisions.md` D-03 and D-13 for why this ablation
omits the features entirely rather than simulating forecast error.

## Evidence — current numbers (post weather hour-convention fix, 59 rolling days)

| Model | Oracle MAPE/WAPE/MAE/RMSE | Realistic MAPE/WAPE/MAE/RMSE | MAPE gap |
|---|---|---|---|
| DNN | 8.81 / 7.47 / 85.1 / 119.6 | 9.48 / 7.93 / 90.4 / 126.9 | +0.67 pp |
| XGBoost | 8.53 / 7.20 / 82.0 / 117.2 | 8.89 / 7.45 / 84.9 / 124.0 | +0.36 pp |
| **Ensemble** | **8.46** / 7.14 / 81.3 / 115.8 | 9.00 / 7.52 / 85.7 / 123.4 | +0.54 pp |
| Linear | 8.64 / 7.36 / 83.8 / 117.5 | 8.92 / 7.56 / 86.2 / 122.6 | +0.28 pp |
| Naive | 11.59 (identical both scenarios — expected, no weather input) | | 0 |
| Weekly-Naive | 12.26 (identical both scenarios — expected, no weather input) | | 0 |

Sanity check: Naive and Weekly-Naive are numerically identical between scenarios in every run this
project has done with this experiment, since neither uses weather features — confirmed exact match
each time.

**What this means:** the honest cost of the perfect-foresight weather assumption is roughly
0.3–0.7 pp MAPE, model-dependent — not the dominant driver of accuracy (both scenarios comfortably
beat naive persistence by a wide margin). In the realistic scenario, XGBoost and Linear are close
enough to each other to be within this project's documented run-to-run noise band — read as a
near-tie for best individual model, not a settled ranking (see
`specs/completed/weather-hour-convention-fix.md` for the history of this specific finding moving
around across reruns).

## Verification

Full end-to-end rolling pipeline execution (not a syntax check) for every version of this
experiment's numbers cited above. `py_compile` clean. Sanity check on Naive/Weekly-Naive identity
between scenarios re-confirmed on every rerun.

## Revert

This is a standalone, additive file — deleting `experiments/realistic_weather_ablation.py` fully
reverts it with no effect on the main pipeline.

## How to run

`python3 experiments/realistic_weather_ablation.py` from the repo root (same requirements as the
main script; calls `input()` and the live Renewables.ninja API, same convention as
`MACHINE_LEARNING_RESIDUAL_DEMAND.py` — see `decisions.md` D-09 on why that flow isn't simplified).

## Source

`MACHINE LEARNING RD CONTEXT.md` §12.8, §12.11. See also `decisions.md` D-03, D-13.
