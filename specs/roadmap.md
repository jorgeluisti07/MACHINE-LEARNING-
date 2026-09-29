# Roadmap — open items

Prioritized, evidence-based ideas not yet implemented. Extracted from `MACHINE LEARNING RD CONTEXT.md`
§10 (Optimization Roadmap) and §12.3 (External Review priority tiers). Shipped items have moved to
`specs/completed/` — see there for what's already done (the flat ensemble and the P5 XGBoost sweep,
in particular, are already shipped and are the "current best model," not open items).

**Holdout discipline applies to every item below** (`decisions.md` D-08): evaluate on the
**validation split only** while deciding whether to keep or discard an item. The test period is
touched exactly once more, at the very end, for the thesis number — do not re-run the test period
per candidate to pick a winner.

---

## P1 — Holiday-proximity features

**Targets:** the two biggest recorded single-day error events (Dec 23, Nov 26 in the original
analysis — 2 days before a major holiday, not on the holiday itself).

**Evidence:** 7 of the top-10 worst days in the original diagnostic sit within 0–2 days of a
national holiday; the existing `is_holiday` (same-day) feature structurally cannot represent
"holiday coming up" or "holiday just passed," and has near-zero XGBoost importance, consistent with
not capturing this pattern.

**Proposed action:** add `days_to_next_holiday` and `days_since_last_holiday` (clipped, e.g. ±3),
and/or `is_holiday_adjacent`. Leakage-free — the holiday calendar is known in advance.

**History — already tried once, reverted.** Implemented, unit-tested, adversarially reviewed, and
fully executed once already (30→32 features): target days' error dropped meaningfully (32%/17% on
the two worst days), but XGBoost improved while the DNN got worse across all four metrics, leaving
the flat-weight Ensemble mixed rather than a clean win. Reverted per user request. **If retried:**
compare against the current honest baseline (§12.11's 8.46% oracle Ensemble MAPE), not the older
pre-leakage-fix numbers this was originally measured against — XGBoost is now a different,
better-regularized model (post-P5), so the effect may differ. See
`MACHINE LEARNING RD CONTEXT.md` §8 row 15, §10 P1, §11 for exact commit hashes to re-apply the
original attempt if useful as a starting point.

**Expected impact:** may not move MAPE much (tail-day fixes don't move an average over ~1,400
hours much — see the diagnostic-chart review, `MACHINE LEARNING RD CONTEXT.md` §12.10, for the
arithmetic on why) but should cut the worst-day RMSE contribution.

---

## P2 — Prune near-zero-importance features

**Evidence:** several features consistently show < ~0.01 XGBoost gain each (`hidro_mw`,
`is_holiday`, `hidro_fraction_L24`, `hidro_anomaly_L24`, `residual_L48`, `temperature`,
`irradiance_diffuse`, `eolica_mw` in the original analysis — re-verify current importances before
acting, they may have shifted since P5's regularization change). `hour`/`hour_sin`/`hour_cos`
triple-encode the same signal for tree models specifically.

**Proposed action:** ablation run with the bottom ~6 features removed. Keep the calendar cyclic
encodings for the DNN even if trees don't need them — consider separate feature lists per model if
this is pursued.

**Note:** `hidro_mw` specifically was reconsidered and deliberately kept — see `decisions.md` D-06.
Any future P2 attempt should not re-litigate that specific decision without new evidence; it can
still prune other near-zero features.

**Expected impact:** likely small accuracy change either way, but a leaner model and a clean
"feature ablation" subsection for the thesis. If accuracy holds on the validation split, keep the
pruned set.

---

## P4 — Ramp/persistence-error features

**Targets:** the systematic lag visible in 7-day zoom plots — both models trail fast ramps, likely
from over-reliance on the current-hour `demanda_residual` feature as a persistence anchor.

**Proposed action:** add `residual_delta_3h = residual(t) − residual(t−3)` and a
yesterday's-persistence-error style feature (e.g. `naive_error_L24 = residual(t−24) −
residual(t−48)`) — both backward-looking, leakage-free by construction.

**Expected impact:** may help the model anticipate ramps instead of following them one step late.
Not yet evaluated against the current baseline.

---

## P6 — Quantile/pinball or Huber loss

**Targets:** robustness to the spiky error distribution (a handful of ~400–500 MW-error days
against a much smaller typical-day MAE) — squared-error training over-weights those days at the
expense of typical-day accuracy.

**Proposed action:** try `objective='reg:pseudohubererror'` in XGBoost and/or Huber loss in the
DNN. Alternatively, train P10/P50/P90 quantile models — an uncertainty band is a strong addition for
an operational-dispatch framing in the thesis.

**Expected impact:** more stable typical-day accuracy; quantiles add operational value beyond point
MAPE, independent of whether point-forecast accuracy itself improves.

**Note:** the biggest methodological lift of the remaining open items — budget accordingly.

---

## P7 — Pipeline code quality (no accuracy impact)

No accuracy impact expected — do whenever convenient, does not need validation-split evaluation
since it changes no model behavior:

- Replace the `globals()['model_dnn' + str(T_day)] = ...` per-iteration artifact stashing with a
  plain dict (`artifacts[T_day] = {...}`) — same inspectability, no namespace pollution, easier to
  serialize.
- Cache `rebuild_hidro_profile_features` per `T_day` if the rolling loop feels slow (it currently
  recomputes an O(N) map twice per day — once for DNN, once for XGBoost, same `T_day`).
- Factor the duplicated DNN/XGBoost plotting blocks into one `plot_model_results(name, preds, ...)`
  helper.

---

## Deferred by choice, not gaps

These are explicitly **not** open questions waiting on information — they're decisions the user
made to set the work aside. Listed here so they aren't mistaken for overlooked items; see
`decisions.md` for the full reasoning behind each.

### #13 — API token rotation + env-var migration

**Status:** deferred, user's action required (only the account owner can rotate the token). See
`decisions.md` D-10.

**When this becomes active work again:** once the user rotates the Renewables.ninja token, the
env-var change itself is small and low-risk — scoped tightly to the token-handling lines in both
`MACHINE_LEARNING_RESIDUAL_DEMAND.py` and `experiments/realistic_weather_ablation.py`, per the
standing rule that the download flow is user-owned (`decisions.md` D-09).

### #17 — Experimental/weaker model artifact

**Status:** deferred, user's choice. No LSTM or alternative-architecture attempt was ever actually
run in this project; adding one now would be new work, not recovery of something lost. See
`decisions.md` D-04.

**When this becomes active work again:** only if the user asks. If pursued, follow the
"Experimental models policy" (`decisions.md` D-11) — keep it in the repo, clearly labeled, even if
(especially if) it loses.

### Varying-horizon / single-daily-issue forecast redesign

**Status:** not on the roadmap as active work — the constant-horizon (hourly-reissue) design was
deliberately kept, see `decisions.md` D-05. Would only become worth prioritizing if
`backlog/etesa-submission-timing.md`'s open question resolves in a direction that makes it clearly
more representative of Panama's actual day-ahead process, and even then would likely be scoped as a
new standalone experiment (mirroring the realistic-weather-ablation pattern) rather than a
replacement for the current primary methodology.
