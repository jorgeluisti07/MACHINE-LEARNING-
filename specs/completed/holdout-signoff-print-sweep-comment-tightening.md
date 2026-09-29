# Holdout sign-off, print sweep, and comment tightening (external review #18, #20-remainder)

**Status:** shipped, current (2026-07-24).

## What shipped

1. **Holdout discipline formally signed off** — see `decisions.md` D-08 for the full rule. This
   record is the "when/why it became a locked decision" note; D-08 is the decision itself.
2. **Broader print-volume sweep (#20, remainder).** Surveyed every `print(...)` in the main script
   against the bar "clear data check or final result." Removed two genuinely low-value ones —
   `print(df.dtypes)` (bare informational dump, nothing validated) and a three-line "Data
   structures initialised" status block (pure narration, redundant with a real assertion a few
   cells later). Fixed the notebook-only equivalent (a trailing bare `df.dtypes` expression).
   Everything else surveyed (`describe()`, `isnull().sum()`, the stationarity table, the
   correlation printout, the XGBoost sweep table, all train/test/overfit/model-selection prints,
   the final results tables) was judged a genuine, bounded data check or result and left alone.
3. **Comment tightening.** Reviewed the largest comment blocks added for the leakage/alignment
   fixes (calibration header, lag-semantics constants, hour-ending convention, `origin_time`/
   `target_time`, the `hidro_mw` decision) and trimmed prose while keeping every load-bearing fact
   (the leak mechanism, the fix rationale, the empirical evidence, revert instructions). Net: 30
   lines removed from the `.py` with no loss of the reasoning a future reviewer would need.

## Why

Holdout discipline: see `decisions.md` D-08. Print sweep and comment tightening: excessive output
and comment volume were making it harder to spot the data checks and reasoning that actually
matter — the same underlying concern behind `mistakes.md` M-10 (internal tracking references
accumulating in comments over time), addressed here for print volume and prose length specifically.

## Verification

`py_compile`/per-cell `ast.parse` clean on both files. No rolling-loop rerun performed for this
batch — pure comment/print changes with no logic touched, and no variable removed that other code
depends on, so `py_compile`/`ast.parse` were judged sufficient verification for this specific
batch (contrast with `specs/completed/leakage-and-alignment-fix.md`, where a full rerun was
required because model behavior actually changed).

## Revert

Comment/print changes only — no functional revert needed beyond restoring the removed prints/
comments if a future reviewer specifically wants the removed verbosity back; no other code depends
on any of it.

## Source

`MACHINE LEARNING RD CONTEXT.md` §12.9. See also `decisions.md` D-08, `mistakes.md` M-10.
