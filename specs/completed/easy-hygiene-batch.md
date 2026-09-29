# Easy/trivial hygiene batch from the external review (external review #1, #6, #10, #14a/b/c, #15, #16, #19a, #20-partial)

**Status:** shipped, current (2026-07-23). 10 of the external review's 20 points, all pure
hygiene/documentation fixes with no effect on model behavior or reported metrics — verified by
re-running the full rolling pipeline and confirming results were unchanged.

## What shipped

| Item | Change |
|---|---|
| #20 (partial) | Removed global `warnings.filterwarnings('ignore')` |
| #14b | Removed redundant inline `pip install xgboost` (already pinned in `requirements.txt`) |
| #1 | Checked Parquet/Excel headings — already correct, no change needed |
| #15 | Checked MAE in results table — already present, no change needed |
| #14a | Pinned `requirements.txt` to exact versions for all 12 packages |
| #19a | Softened "the model can be used for forecasting" to "selected for further evaluation" |
| #10 | Added an explicit warning print for the 02/29/2025 unparseable-date row (later itself
  trimmed as unnecessary prose — see the comment-cleanup history in `mistakes.md` M-10) |
| #6 | `hidro_mw` in/out decision — see `decisions.md` D-06 |
| #16 | `results_summary.to_csv(...)` and `forecasts.csv` export added |
| #14c | Raw API JSON response saved for reproducibility |

## Why

The external review's easy/trivial tier — user explicitly selected these for a first batch,
deferring `#13` (token → env var, see `decisions.md` D-10) from the same tier.

## Verification

Full rolling pipeline rerun end-to-end in the scratch environment against the same data used for
the then-current shipped baseline: results unchanged (Ensemble 6.78% MAPE / 5.83% WAPE / 66.5 MAE /
94.3 RMSE — identical to the pre-change baseline, as expected since none of these items touch
feature construction or model training). `py_compile`/per-cell `ast.parse` clean on both files.
`README.md` "Expected output" section updated to list the new CSV outputs.

## Not done from this tier

`#13` (token → env var) — explicitly deferred, see `decisions.md` D-10. Print-volume trimming
(the rest of `#20` beyond the global-suppression removal) — done later, see
`specs/completed/holdout-signoff-print-sweep-comment-tightening.md`.

## Revert

These are 10 independent, low-risk hygiene fixes with no shared dependency — reverting any one
individually is safe (no other item in this batch depends on it). See
`MACHINE LEARNING RD CONTEXT.md` §11 for the general commit-log convention this project uses.

## Source

`MACHINE LEARNING RD CONTEXT.md` §12.5. See also `decisions.md` D-06, D-10.
