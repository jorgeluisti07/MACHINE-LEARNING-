# Origin/target-time labeling, lag-semantics fix, and two new baselines (external review #7, #8, #9, #19b, part of #20)

**Status:** shipped, current (2026-07-23).

## What shipped

1. **`origin_time`/`target_time` columns (#7/#8).** Forecasts, plots, and the `forecasts.csv`
   export were indexed by the row's own timestamp ("now," when the forecast is issued) even though
   the value held is for 24h later. Added explicit `origin_time`/`target_time` series
   (`target_time = origin_time + 24h`); every forecast-vs-actual plot and the CSV export now index
   by `target_time` — the hour actually being predicted.
2. **Lag semantics switched to target-relative (#9).** `residual_L168`/`residual_L336` renamed and
   redefined as `residual_L144`/`residual_L312`, based on a measured training-only correlation gap
   favoring the target-relative framing. See `decisions.md` D-07 and `mistakes.md` M-05.
3. **Weekly-naive and linear-regression baselines added (#19b).** See `decisions.md` D-12.
4. **One more excessive-print removal (#20, partial).** An unbounded ~8,700-row full-dataframe dump
   removed from both `.py` and `.ipynb`.

## Why

`origin_time`/`target_time`: indexing by the wrong timestamp is not computationally wrong (verified
the rolling loop's `T`-window logic was sound throughout) but is easy to misread and a real risk for
future bugs — worth fixing even without a demonstrated incident. Lag semantics: see
`specs/completed` sibling files and `mistakes.md` M-05. Baselines: beating one weak naive benchmark
isn't sufficient evidence of deployment readiness; two more baselines (one trivial, one simple-but-
tuned-nothing) give a fairer picture of how much value the DNN/XGBoost/Ensemble actually add.

## Evidence

Full results after this batch (59 rolling days — one more day than before, a side effect of the
lag-semantics change shrinking the required burn-in by 24 rows, not a bug):

| Model | MAPE% | WAPE% | MAE MW | RMSE MW |
|---|---|---|---|---|
| **Ensemble** | 8.18 | 6.95 | 79.2 | 112.7 |
| XGBoost | 8.34 | 7.10 | 81.0 | 115.2 |
| DNN | 8.41 | 7.16 | 81.6 | 114.9 |
| Linear | 8.47 | 7.26 | 82.7 | 116.0 |
| Naive | 11.59 | 10.06 | 114.7 | 163.0 |
| Weekly-Naive | 12.26 | 10.70 | 121.9 | 172.0 |

Two genuinely new findings from the added baselines: (1) plain linear regression on the same 30
features gets within ~0.3pp of the tuned nonlinear models — most of this pipeline's accuracy comes
from feature engineering, not model sophistication; (2) weekly-naive (12.26%) is *worse* than
day-to-day persistence (11.59%) despite a strong same-week-hour correlation (r=0.780) — correlation
with the target and being a good standalone point forecast are different properties.

(These numbers were later superseded by the weather hour-convention fix — see
`MACHINE LEARNING RD CONTEXT.md` §12.11 for current figures.)

## Verification

`py_compile`/per-cell `ast.parse` clean on both files. Full scratch rerun clean (index check passed,
Feb-29 warning fired as expected, no errors). Confirmed no behavior change from the `origin_time`/
`target_time` relabeling specifically: every downstream read of `forecasts[...]` uses `.iloc`
(positional), so relabeling the index alone was safe — confirmed by inspecting the scratch run's
`forecasts.csv` (correct 24h offset visible between `origin_time` and `target_time` columns).
`results_summary.csv` (6 rows) and `forecasts.csv` (1416 rows) written correctly.

## Revert

`git revert d106a7c` restores input-relative lags (168/336), origin-time-indexed forecasts, removes
the weekly-naive/linear baseline rows, and restores the full-dataframe print.

## Source

`MACHINE LEARNING RD CONTEXT.md` §12.7. See also `decisions.md` D-07, D-12; `mistakes.md` M-05.
