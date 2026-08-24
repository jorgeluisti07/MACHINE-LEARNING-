# Decisions Log

Locked methodological decisions for this project — choices that were deliberately made and
justified, not just bugs that got fixed (see `mistakes.md` for those). Each entry has a `D-` id.
Extracted from `MACHINE LEARNING RD CONTEXT.md`'s Fixes Log (§8) and External Review implementation
log (§12); full derivations and evidence live there — this file is the durable, citable summary.

**Convention:** a decision can be superseded by a later one. When that happens, mark the old entry
`SUPERSEDED by D-XX — kept for comparison, do not treat as current` rather than deleting it (the
same pattern already used in `MACHINE LEARNING RD CONTEXT.md` for superseded results numbers, e.g.
the pre-leakage-fix 6.78% MAPE figure).

---

### D-01 — Flat 0.5/0.5 ensemble, not a validation-weighted one

**Decision:** the shipped ensemble averages DNN and XGBoost with a fixed 0.5/0.5 weight.

**Why:** a validation-tuned weight search (minimizing WAPE on the first rolling window's
held-out validation slice, leakage-verified) picked w=0.30 (favor XGBoost), but on the actual
59-day test set the flat 0.5/0.5 still won on every metric (8.46% vs a worse weighted-ensemble
MAPE at the time). Root cause: the validation window sits in a different part of the year than
the test window (the wet→dry seasonal shift documented in §3), so relative model skill on
validation isn't a reliable guide to relative skill on test. This is a real, citable limitation of
naive validation-based ensemble weighting under distribution shift, not a bug in the search itself.

**Status:** weighted-ensemble code was removed from the pipeline per user request after the
negative result was recorded ("keep ensemble, remove weighted ensemble"). The finding stays
documented as a legitimate negative result.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §8 rows 16–18, §9.2, §10 (Weighted ensemble section).
**Revert:** `git revert 899160f` restores the weighted-ensemble code on top of the current state
(see CONTEXT.md §11 for full commit sequencing).

---

### D-02 — XGBoost hyperparameter sweep is validation-only, on the FIRST rolling window, done once

**Decision:** the sweep searches a small grid using only the first rolling window's (`T`, not
`T_last`) internal validation split, and the winning config is then fixed for every iteration of
the rolling loop — it is not re-tuned per rolling day.

**Why:** an early draft used `T_last`'s validation slice, and unit-testing caught (before any run)
that `T_last`'s training window overlaps 98.3% of the reported test period's rows — tuning against
it would leak actual test-period outcomes into hyperparameter selection. Using `T`'s validation
slice (rows entirely before the test window starts) guarantees zero date overlap with anything
reported. Re-tuning per rolling day was never on the table — it would multiply runtime by ~59x for
a benefit not expected to matter at this horizon, and would reopen the same overlap risk on every
later window unless each day's tuning slice were independently re-verified.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §8 rows 16–17, §9.2, §10 P5.

---

### D-03 — Perfect-foresight h+24 weather kept as a disclosed assumption, not dropped

**Decision:** the main (oracle) pipeline keeps the four `*_h24` weather features built from the
true future MERRA-2 value (`shift(-24)`), explicitly disclosed as an optimistic upper bound — it is
not replaced with a fabricated forecast, and it is not dropped from the primary reported model.

**Why:** no real operational NWP forecast data for Panama exists for this period, so inventing
forecast-error noise would be a worse methodological choice than being explicit about the
assumption. Instead, a second standalone experiment (`experiments/realistic_weather_ablation.py`)
drops the four `*_h24` features entirely and reports the honest cost: ≈0.3–0.9 pp MAPE across
models, model-dependent (largest for DNN, smallest for Linear/XGBoost) — see D-13.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §7.1, §12.2 point 4, §12.8, §12.11.

---

### D-04 — #17 (experimental/weaker model) deliberately deferred, not a gap

**Decision:** the external review's point #17 (add a deliberately-weaker model, e.g. an LSTM
attempt, to the repo per the "experimental models policy") is set aside — not scoped as active
work, not treated as an unresolved finding.

**Why:** no such experiment was ever actually run in this project; building one now would be new
work, not recovery of something lost. The user explicitly chose to defer this rather than treat it
as an open correctness or documentation gap.

**Status:** open in `specs/roadmap.md` as a deferred-by-choice item. Revisit only if the user asks.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.4 point 3, §12.3 Tier 4.

---

### D-05 — Constant-horizon (hourly-reissue) forecast framing kept over varying-horizon/single-issue

**Decision:** the pipeline evaluates a forecast conceptually re-generated every hour, always
looking exactly 24h ahead of that hour (constant h+24 horizon) — not a single daily forecast issued
once, covering the next day with varying horizons h+1...h+24.

**Why:** both framings are legitimate, standard choices in the STLF literature and in real
day-ahead market operations. Switching to the varying-horizon framing would be a substantial
architectural redesign (new multi-horizon target shape, different DNN/XGBoost output heads,
re-validated leakage-safety per horizon), not a quick fix, and the current constant-horizon design
is already what every feature (`*_h24`), the leakage-safe calibration/hydro rebuilds, and every
reported metric are built around. An independent code review confirmed the constant-horizon design
is not a leakage bug, just a different (and previously mis-described) definition — see M-06.

**Open sub-question (not blocking this decision):** whether ETESA/CND's actual day-ahead
submission process is a single daily deadline or a continuously-reissued operational forecast is
unconfirmed — see `backlog/etesa-submission-timing.md`. Answering it would inform whether a
supplementary varying-horizon experiment is worth adding later; it does not change this decision.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §2 (v26.22), external review conversation (2026-08),
independent code-review and ML-methodology subagent findings (not yet logged as a numbered §12.x
subsection — flagged in the summary at the end of this extraction).

---

### D-06 — `hidro_mw` kept as a raw current-hour model input

**Decision:** `hidro_mw` stays directly in `cols_order`, feeding both models raw, rather than being
dropped in favor of only the four derived hydro-state features.

**Why:** an earlier code comment claimed keeping it created "perfect collinearity" with
`demanda_residual` — checked and found inaccurate (true collinearity would require `termica_mw`,
which isn't in this dataset). `hidro_mw`'s actual low XGBoost importance (gain ≈ 0.005) reflects
redundancy with the already-present `demanda_residual` feature, not danger. An earlier feature-
pruning experiment (P2, tried and reverted) kept `hidro_mw` only as a structural dependency for
`rebuild_hidro_profile_features`, not as a direct model-input candidate — noted here since that
reasoning could inform a future revisit of P2.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 6, §12.5 #6.

---

### D-07 — Lag features switched to target-relative (144h/312h), not input-relative (168h/336h)

**Decision:** `residual_L144`/`residual_L312` give the residual demand at the same clock-hour,
exactly 1/2 weeks before the hour being **predicted** (target-relative), not before the input row
itself (input-relative, which the old `residual_L168`/`residual_L336` naming implied but didn't
deliver — those were actually 192h/360h before the target).

**Why:** measured training-only Pearson correlation on the leak-fixed, hour-aligned pipeline showed
a large gap favoring the target-relative framing: r=0.780 (144h, target-relative) vs r=0.667 (168h,
input-relative); r=0.764 (312h, target-relative) vs r=0.651 (336h, input-relative). The ≈0.11
Pearson-r gap was large enough to switch the design, not just document the old one as a status-quo
default. `LAG_1=24` (hydro state features) was left unchanged — those features describe "now," so
input-relative framing is correct there.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 9, §12.7.

---

### D-08 — Holdout discipline: correctness fixes may re-touch the test period once; modeling
variants may not

**Decision:** fixing a leakage/alignment bug and re-measuring on the same test window is legitimate
and does not "burn" the holdout, because the fix is justified by a data-generating-process argument
independent of whether the resulting score went up or down. Trying multiple modeling variants and
keeping whichever scores best on the test window is not legitimate. From the point this was signed
off, any *new* roadmap item (P4, P6, hydro-regime clustering, or anything else) must be evaluated on
the **validation split only**; the test period is touched exactly once more, at the very end, for
the number reported in the thesis.

**Why:** the Nov–Dec test window had already been used to judge multiple roadmap items (P1, P5,
weighted ensemble, P2) before this rule was formalized — repeated comparison against the same test
set is a known way to end up with an optimistic final number even with no code bug. The fact that
every correctness-driven re-evaluation this project has done made the honest numbers **worse**, not
better, is itself evidence the fixes weren't chosen to flatter the score.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 18, §12.4 ("Holdout discipline — signed off").

---

### D-09 — The Renewables.ninja download code is user-owned; do not modify without explicit instruction

**Decision:** the geocoding `input()` prompt, the API token handling, and the download/parse flow
in the data-download cells are off-limits for unprompted changes — including well-intentioned
reproducibility improvements.

**Why:** an earlier reproducibility pass (fixed coordinates instead of interactive geocoding,
env-var token, offline cache-load) was rolled back at the user's request — the user needs to enter
the location interactively, and this flow is theirs to manage. Reinforced by D-10 below (token
handling specifically).

**Source:** `MACHINE LEARNING RD CONTEXT.md` §8 row 10, standing rule at the top of the file.

---

### D-10 — API token stays hardcoded in-repo for now; rotation is the user's action, deferred by choice

**Decision:** the Renewables.ninja API token remains a plain-text literal in both
`MACHINE_LEARNING_RESIDUAL_DEMAND.py` and `experiments/realistic_weather_ablation.py`, not yet
moved to an environment variable.

**Why:** an env-var version was tried once and reverted per user request (D-09). Rotating the token
is something only the user can do (it requires their Renewables.ninja account access) — moving to
an env variable before rotation would just document the exposure pattern without fixing the actual
risk (the current token is already exposed in git history regardless of where it lives going
forward). This is a deliberately deferred item, not an overlooked one.

**Status:** open in `specs/roadmap.md`. Once the user rotates the token, the env-var change is a
small, low-risk edit scoped tightly to the token-handling lines.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §8 row 6, §12.2 point 13, §12.3 Tier 4.

---

### D-11 — Experimental/rejected models are kept in the repo, clearly labeled, never deleted

**Decision:** any non-selected or experimental model attempt (alternative architectures, alternative
feature sets) belongs in the repository — in a clearly-labeled "Experimental / not selected" section
or an `experiments/` folder — not deleted.

**Why:** a weaker result is still evidence for the thesis; explaining what was tried and why it
lost (e.g. "the NWP h+24 features made LSTM's temporal-persistence advantage redundant") is part of
the methodology story, not dead weight.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §10 ("Experimental models policy").

---

### D-12 — Weekly-naive and linear-regression baselines are comparison baselines, not candidate models

**Decision:** both new baselines added alongside DNN/XGBoost/Ensemble are deliberately minimal —
weekly-naive is exactly `residual_L144` read directly (same clock-hour, 1 week before target, by
construction, no new computation); linear regression uses the same leakage-safe walk-forward
feature pipeline as DNN/XGBoost but with no hyperparameter tuning, no plots, and no overfit-gap
tracking.

**Why:** these exist to show whether DNN/XGBoost add real value over something much simpler than
persistence, not to compete for the "best model" slot. Keeping them minimal is what makes them a
fair, legible baseline rather than a fifth candidate model muddying the comparison. (In practice
this produced a genuinely interesting finding: in the realistic-weather scenario, Linear regression
is close enough to XGBoost to be a near-tie within documented run-to-run noise — see D-13's source
section for the numbers, not a reason to promote Linear into a tuned candidate.)

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 19, §12.7 (#19b).

---

### D-13 — The realistic-weather ablation omits `*_h24` features entirely, rather than simulating forecast error

**Decision:** `experiments/realistic_weather_ablation.py` drops the four `*_h24` weather features
completely (26 features instead of 30) to answer "what does dropping perfect-foresight weather
cost," rather than trying to approximate real NWP forecast error by adding synthetic noise to the
`*_h24` values.

**Why:** there is no real operational NWP forecast dataset for Panama for this period to calibrate
synthetic noise against — inventing forecast-error numbers would produce a result that looks more
precise than it is. Omitting the features entirely is the honest choice, and it directly answers a
well-defined question (the cost of the oracle assumption) rather than a poorly-defined one (the
cost of an invented forecast-error model).

**Result (for context, not the decision itself):** honest cost of the perfect-foresight assumption
is ≈0.3–0.5 pp MAPE depending on model (current numbers, post weather-hour-convention fix, §12.11);
not the dominant driver of accuracy — both scenarios comfortably beat naive persistence.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 4, §12.8, §12.11.
