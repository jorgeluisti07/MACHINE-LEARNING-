# Weather hour-convention fix — closing the last open item from the external review (external review #11, weather half)

**Status:** shipped, current (2026-07-30). Closes the weather-side gap left open by
`specs/completed/leakage-and-alignment-fix.md`.

## What shipped

An adversarial re-check of every point in the external review, item by item against current code,
found one item marked "resolved" that was only half-resolved: #11 (hour-convention alignment). The
demand-vs-generation half was fixed earlier; the weather side had a code comment admitting it was
never checked. This closes that gap: `df_combined.index = df_combined.index + pd.Timedelta(hours=1)`,
applied right after the weather data is loaded/built, before export and before the join with real
generation, in both `MACHINE_LEARNING_RESIDUAL_DEMAND.py` and
`experiments/realistic_weather_ablation.py` (which duplicates the same data-loading logic).

## Why

See `mistakes.md` M-03 for the full incident record. In short: Renewables.ninja labels hourly
values by interval start (hour-beginning); ETESA demand/generation label by interval end
(hour-ending). Confirmed as a real bug via two independent checks: (1) Renewables.ninja's API
documentation states hour-beginning labeling explicitly; (2) empirically, the diurnal solar profile
showed a clean, consistent 1-hour offset across three independent landmarks (ramp start, peak, ramp
end) under the old unshifted join.

## Evidence — honest before/after (same 59-day test window, full rolling rerun, both scenarios)

| Model | Oracle MAPE% before → after | Oracle MAE MW before → after | Realistic MAPE% before → after | Realistic MAE MW before → after |
|---|---|---|---|---|
| DNN | 8.41 → 8.81 | 81.6 → 85.1 | 9.20 → 9.48 | 88.7 → 90.4 |
| XGBoost | 8.34 → 8.53 | 81.0 → 82.0 | 8.93 → 8.89 | 85.5 → 84.9 |
| **Ensemble** | 8.18 → 8.46 | 79.2 → 81.3 | 8.91 → 9.00 | 85.3 → 85.7 |
| Linear | 8.47 → 8.64 | 82.7 → 83.8 | 8.81 → 8.92 | 85.6 → 86.2 |
| Naive | unchanged | unchanged | unchanged | unchanged |
| Weekly-Naive | unchanged | unchanged | unchanged | unchanged |

Naive and Weekly-Naive exactly unchanged in both scenarios — sanity check confirming the fix
touched only weather-derived features. Every other cell moved in the expected worse-after-honest-
fix direction except XGBoost's realistic-scenario MAPE, which improved marginally (8.93→8.89%) —
within run-to-run noise, not a contradiction of the "fixing a leak/bug makes honest numbers worse"
pattern seen elsewhere in this project.

**Ranking consequence:** in the realistic scenario, XGBoost (8.89%) and Linear (8.92%) are now
close enough (0.03 pp) to be within this project's documented ±0.05 pp run-to-run noise band — no
longer a clean "Linear is best" story as an earlier run had suggested; call it a near-tie pending a
repeat run, not a settled ranking.

**This 8.46% (oracle) / 9.00% (realistic) Ensemble MAPE is the current honest baseline**, superseding
all earlier figures in this project's history.

## Robustness check

The hour-of-day / solar-calibration-ceiling finding from the diagnostic-chart review (see
`MACHINE LEARNING RD CONTEXT.md` §12.10) was re-checked against the corrected data: shape
correlation between oracle and realistic hourly-MAE curves is r=0.998 (was r=0.994 pre-fix) — if
anything a tighter match. Conclusion unchanged: the midday error ceiling is calibration-shape
driven, not a weather-timing artifact, and that conclusion was not itself an artifact of the
now-fixed misalignment bug — it holds on both the leaky and the corrected data.

## Verification

`py_compile` clean on both `.py` files; per-cell `ast.parse` clean on the notebook (67 cells, 0
errors). Full rolling rerun executed for both scenarios in the scratch environment (not just a
syntax check) to produce the honest before/after numbers above.

## Revert

The fix is a single `df_combined.index = df_combined.index + pd.Timedelta(hours=1)` line (plus the
corresponding comment update) in each of the two Python files — removing that line and reverting
the comment restores the pre-fix (misaligned) behavior. Not recommended; the misalignment is a
confirmed bug, not a modeling choice.

## Source

`MACHINE LEARNING RD CONTEXT.md` §12.11. See also `mistakes.md` M-03.
