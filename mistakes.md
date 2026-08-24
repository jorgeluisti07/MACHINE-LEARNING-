# Mistakes Log

Genuine incidents — bugs found after the fact, not deliberate methodological choices (see
`decisions.md` for those). Each entry has an `M-` id and an **actionable rule**, not just
narration: read this file before touching leakage-sensitive code, timestamp joins, or comments in
this repo. Extracted from `MACHINE LEARNING RD CONTEXT.md`'s Fixes Log (§8) and External Review
implementation log (§12).

---

### M-01 — Solar calibration leaked target-hour information into a weather feature

**What happened:** the irradiance calibration factor was computed as a **full-year** per-hour ratio
of real metered solar to modeled ("ninja") solar, including test-period rows. Via the
`shift(-HORIZON)` that builds `irradiance_direct_h24`/`irradiance_diffuse_h24`, this made those
horizon-weather features proportional to `real_solar[t+24]` — one of the three terms that directly
defines the target (`demanda_residual_h24 = demanda_mw − solar_mw − eolica_mw` at t+24). This was a
feature mathematically entangled with the target, materially worse than the already-disclosed
perfect-foresight-weather assumption.

**Why it wasn't caught earlier:** it read as "just more weather calibration," similar in shape to
the already-fixed hydro-baseline leak (M-04), but was a separate code path never given the same
training-only-refit treatment. Caught by an external code review, not internally.

**Fix:** `rebuild_calibration_features(df_in, T)` refits a per-hour-of-day real/ninja factor from
training rows (`< T`) only, per rolling window, applied using only ninja inputs — never target-hour
real solar. Raw ninja columns are structurally excluded from the model's feature list (drawn
explicitly from `cols_order`, not passed through implicitly).

**Rule:** any statistic derived from the full dataset (a ratio, a mean, a calibration factor) that
feeds a feature used at or near the target hour must be refit **training-only, per rolling window**
— never computed once globally before the rolling loop starts. When adding a new calibration or
normalization step, ask explicitly: "does this use any row at or after the cutoff T?"

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 3, §12.6.

---

### M-02 — Demand file joined against generation using the wrong hour convention

**What happened:** `solar_eolica_hidro_horario_2025.csv` (generation) is hour-ending (the
00:00–01:00 interval is stamped 01:00); the demand loader mapped `H1` → `00:00` (hour-beginning),
one hour off. This placed demand one hour BEFORE generation, so
`demanda_residual = demanda − solar − eólica` silently mixed two different physical hours —
a corrupted target, for every row, not just an edge case.

**Why it wasn't caught earlier:** a quick check ("which labeled hour has peak solar output")
was only weakly diagnostic — it's consistent with either convention, so it looked resolved without
actually being resolved. It took a **structural** check (row-count signature: the generation file
spans exactly 8760 rows from `2025-01-01 01:00` to `2026-01-01 00:00`, which is only possible under
hour-ending labeling) to actually confirm the convention.

**Fix:** loader now maps `H_k → k:00` (was `(k-1):00`). Confirmed by a zero-NaN demand↔generation
join after the fix (any residual misalignment would show up as join gaps).

**Rule:** never assume a source file's hour-labeling convention from its column name alone (`H1`
looks like "hour 1 = 00:00" but isn't, here). Verify it structurally — row-count signature against
a known full-year length, or a documented source convention — before joining time series from
different sources on a datetime index.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 11, §12.6.

---

### M-03 — Weather timestamps had the same hour-convention bug, on the *other* side of the join, left unresolved for a full review cycle

**What happened:** Renewables.ninja labels hourly values by the **start** of the interval
(hour-beginning) — the opposite convention from the now-fixed demand/generation files
(hour-ending, M-02). This meant every weather feature (`irradiance_direct/diffuse`, `temperature`,
`wind_speed`, and their `_h24` versions) was silently paired with the wrong physical hour of
demand/generation, off by exactly one hour — the same class of bug as M-02, just on the weather
side, and it survived one full round of "fix the alignment issue" work because that round only
fixed the demand-vs-generation half and explicitly flagged (but didn't act on) the weather half.

**Why it wasn't caught earlier:** when M-02 was fixed, the code comment left an honest but
under-acted-on note — "Renewables.ninja weather timestamps may use hour-BEGINNING instead — a
separate alignment question, left unchanged here." That note sat in the code for a full review
cycle before being followed up. It was closed only by a later adversarial re-check that went
through every item in the external review, one by one, against current code, and noticed one
"resolved" item was only half-resolved.

**Fix:** confirmed via API documentation (hour-beginning stated explicitly) and empirically
(diurnal solar ramp/peak/decay all landed exactly 1 hour earlier under the old unshifted join —
ramp start hour 6 vs 7, peak 11 vs 12, ramp end 17 vs 18; cross-correlation also peaks at a +1h
shift). Fixed by shifting the weather data's index forward by 1 hour right after it's loaded/built,
before export and before the join with real generation, in both files that build weather features.

**Rule:** when a suspected alignment/leakage bug has two sides (two different data sources sharing
a timestamp join), fixing one side is not "the issue is handled" — explicitly re-check every other
source in the same join before closing the item. A code comment that says "left unchanged, separate
question" is a flag to revisit, not a closed loop; treat it as an open item in `specs/roadmap.md` or
`backlog/`, not as done.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 11, §12.11.

---

### M-04 — Hydro seasonal baseline computed once on the full year before the rolling loop started

**What happened:** `hidro_typical_h24`/`hidro_anomaly_L24` depend on a `(month, hour)` mean of
`hidro_mw`. Prior versions computed this mean once over the entire dataset before the rolling loop
began, so early rolling windows could see hydro averages that included months past their own
cutoff — look-ahead leakage.

**Fix:** `rebuild_hidro_profile_features(df_in, T)` refits the `(month, hour)` profile from
training-only rows `[:T]` inside `get_targets_features`, once per rolling window. Unseen
`(month, hour)` combinations fall back to the training-window mean. Verified with unit tests
(invariance to post-cutoff data, no NaN in any train/test slice, correct unseen-key fallback).

**Rule:** same as M-01 — this is the general pattern (a global full-dataset statistic feeding a
per-window feature), and it recurred twice in this project (hydro, then solar calibration) before
being treated as a reusable pattern. Any new "typical value" or "seasonal profile" feature must be
built with the same `rebuild_*_features(df_in, T)` per-window pattern from the start, not added ad
hoc and fixed later.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §7.2, §8 row 3.

---

### M-05 — Lag features were labeled relative to the wrong timestamp

**What happened:** `residual_L168`/`residual_L336` were named and reasoned about as "1 week / 2
weeks before the target," but `shift(168)`/`shift(336)` actually looks back from the **input** row,
not the target (which is input + 24h) — so they were really 192h/360h before the target, not a
clean week/fortnight. Not a leak (both directions only look backward), but a semantic-intent bug:
the feature didn't measure what its name and the surrounding reasoning claimed it measured.

**Fix:** switched to target-relative lags (`residual_L144`/`residual_L312`), verified by direct
arithmetic and confirmed with a training-only correlation check that the target-relative framing is
meaningfully stronger (see `decisions.md` D-07 for the numbers).

**Rule:** for any lag/lead feature, always state explicitly — in the variable name or the adjacent
comment — which timestamp it's relative to (input time or target time), and verify the claimed
distance by direct arithmetic on a concrete example, not by the shift amount alone. `shift(N)` on
an input-indexed column is never "N hours before the target" unless the target itself is at the
input row (horizon 0).

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 9, §12.7.

---

### M-06 — A documentation table described a different forecast design than the code actually implements

**What happened:** `MACHINE LEARNING RD CONTEXT.md` §2's "Output" row described the pipeline as
producing "one 24-hour block per iteration (h+1 through h+24)" — a varying-horizon, single-issue-
time design. The code has never done this; it always computes a constant h+24 horizon from each
row's own timestamp, reissued hourly. An external reviewer, working partly from this description,
raised a timing concern that two independent reviews (code-level arithmetic trace, and ML-
methodology) confirmed was not a leakage bug — the actual bug was that the documentation and the
code were describing two different designs.

**Fix:** corrected the table row and added an explicit "Timing definition" paragraph in §2 stating
the constant-horizon, hourly-reissue design plainly, and distinguishing it from the varying-horizon
alternative rather than leaving the distinction implicit.

**Rule:** after any structural decision about how forecasts are issued, indexed, or aggregated
(horizon definition, rolling-window cadence, block boundaries), explicitly re-read every doc table
or sentence that describes "what the output is" — a stale description doesn't just look untidy, it
can cause a reviewer to diagnose a bug that doesn't exist, or miss a real design tradeoff that does.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §2 (v26.22); see also `decisions.md` D-05.

---

### M-07 — A hardcoded result number in a comment went stale after the underlying data changed

**What happened:** a code comment citing "6.78% MAPE" as the current best result silently went
stale once the underlying leakage fixes changed the actual numbers — the comment kept asserting a
figure that no longer matched any real run's output.

**Fix:** results and forecasts are now written to `results_summary.csv`/`forecasts.csv` at the end
of every run, and treated as the only source of truth for citing a number — comments/docs point to
"the current run's CSV," not to a hardcoded figure.

**Rule:** never hardcode a result number (accuracy metric, feature-importance value, timing figure)
directly in a code comment or as unqualified prose in a doc without a pointer to the run/file that
produced it. If a number must appear in prose, cite the section/run it came from (this project's own
convention, e.g. "§12.6" or "results_summary.csv from this run") so a reader can verify it's still
current.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 16.

---

### M-08 — A validation-tuning step almost used a slice that overlapped 98.3% of the test period

**What happened:** the first draft of both the XGBoost hyperparameter sweep and the weighted-
ensemble weight search used the **last** rolling window's (`T_last`) internal validation split for
tuning. Because of the expanding-window design, `T_last`'s training window extends deep into
calendar dates that are later reported as test-period performance — its validation slice
`[7531, 8368)` overlapped 1368 of 1392 test rows (98.3%). Tuning against it would have let the
hyperparameter/weight choice be informed by actual outcomes on almost the entire test period before
that period was "graded" — a soft look-ahead leak into model selection, not into the forecasts
themselves, but real.

**Why it wasn't caught earlier:** it looks correct on a first read — "use the validation split" is
the right instinct — and the leak is specific to how an *expanding*-window design's later windows'
training data overlaps earlier windows' test data, which isn't obvious without tracing the actual
row ranges.

**Fix:** caught during unit-test-writing, before the buggy version was ever run or committed.
Redesigned to use `T` (the first, smallest window) instead — its validation slice `[6300, 7000)` is
provably entirely before the test period starts. A standalone unit test reproduces both the leak
(rejected design) and its absence (shipped design) side by side.

**Rule:** for any validation-only tuning step inside an expanding-window rolling design, use the
**first** window's validation slice, never the last — and explicitly verify (in a unit test, not by
inspection) zero date overlap between that validation slice and the reported test period before
trusting a "validation-only, leakage-free" claim.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §8 row 16, §9.2.

---

### M-09 — Row-count-based lag/shift features never explicitly verified the index was gap-free

**What happened:** every lag/shift feature in the pipeline assumes "N rows back = N hours back,"
which is only true if the hourly index has no gaps and no duplicate timestamps. This was never
explicitly checked anywhere — a single missing or duplicated hour would have silently made a
24-row shift not actually equal to 24 hours, with no error or warning.

**Fix:** added explicit `assert` checks (not just prints) for duplicate timestamps and for full
hourly coverage (via `pd.date_range` comparison), placed before any row-count-based shift is
computed. Passes on the real data (8,754 rows, gap-free).

**Rule:** any time series pipeline that uses positional shifts (`.shift(N)`, `.iloc[i-N]`) to mean
"N time units back" must assert index completeness and uniqueness before those shifts are computed
— don't trust row position without checking it corresponds to time position.

**Source:** `MACHINE LEARNING RD CONTEXT.md` §12.2 point 12, §12.6.

---

### M-10 — Internal tracking references leaked into notebook/code comments across multiple cleanup passes

**What happened:** comments accumulated review-tracking artifacts over time — "review #N" labels,
version tags (`v18`, `v22`, `v24`...), roadmap item references ("P3"), `.md` filename cross-
references, and narration of previous code versions' errors. These are meaningful in a running
methodology log (`MACHINE LEARNING RD CONTEXT.md`) but not in code that should explain only its own
current behavior to a reader who has no access to this project's history.

**Fix:** multiple cleanup passes across the notebook, `.py`, and `experiments/` script removed these
systematically; one pass caught a residual `(review #4)` tag that an earlier pass had missed because
it was scoped only to the main files, not the experiments script.

**Rule:** a code comment should explain only what the current code does and why, never the history
of how it got there, a tracking ID, or a pointer to a doc file. When doing a cleanup pass, check
every file that shares logic with the one being cleaned (e.g. `experiments/*.py` mirrors parts of
the main script) — a pass scoped to only the "main" files will miss duplicated artifacts elsewhere.

**Source:** this session's comment-cleanup requests and their follow-up adversarial-review findings
(not itself a numbered `MACHINE LEARNING RD CONTEXT.md` §12.x item — a recurring pattern noticed
across several cleanup passes rather than one logged incident).
