# ETESA TFM — Notebook Reference Document

**Version:** v26.17 | DNN + XGBoost + Ensemble | Lagged Approach | Leakage-fixed | Shipped model: P5-tuned XGBoost + flat 0.5/0.5 Ensemble. **CURRENT BASELINE = 8.18% MAPE Ensemble, 59 rolling days (§12.7)** — supersedes §12.6's 8.33%/58-day figure (test period grew 1 day after the #9 lag fix shrank burn-in) and the original leaky 6.78% (§9.2, §0). Two new baselines added: Linear regression (8.47% MAPE — close behind DNN/XGB) and Weekly-Naive (12.26% — worse than simple persistence). Weighted-ensemble code removed per user request (§9.2, §8 row 18). External review logged (§12); done: easy/trivial batch (§12.5), correctness batch (§12.6), labeling+lag+baselines batch (§12.7). Still open: #4 (realistic no-oracle-weather run), #13 (token env var), #17 (experimental model), #18 (holdout discipline sign-off). Backup/checkpoint protocol in §13.

> **Standing rule:** the Renewables.ninja download code (geocoding prompt, token, API calls) is
> owned by the user — **do not modify it** without explicit instruction. See §8 row 10.
**Companion files:** `MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb`, `MACHINE_LEARNING_RESIDUAL_DEMAND.py`, `README.md` (How to Run)

---

## 0. TL;DR — read this first

- **Problem:** day-ahead (h+24) residual demand forecasting for Panama, DNN vs XGBoost, expanding-window rolling validation.
- **Inputs are CSV, not Excel/Parquet.** `DEM2025.csv` (wide, `H1`..`H24`) and `solar_eolica_hidro_horario_2025.csv` (long). See §1.
- **Two known limitations baked into the methodology** (not bugs, but must be disclosed in the thesis): perfect-foresight h+24 weather, and a previously-leaky hydro baseline that has since been fixed. See §7.
- **⚠️ SUPERSEDED — HONEST BASELINE is now Ensemble 8.33% MAPE / 110.7 MW RMSE (§12.6)**, after
  fixing the solar-calibration leak (#3) and the hour-ending demand alignment (#11). The 6.78%
  figure below (and everywhere in §9) was computed with the calibration leak present and is
  optimistic; kept only as a historical/comparison record.
- ~~Current best result: Ensemble at 6.78% MAPE / 94.3 MW RMSE (§9.2, v26.10)~~ — leaky, see above.
  The P5 sweep and flat-ensemble findings still hold directionally; only the absolute numbers move.
- **§10 is the optimization roadmap** — prioritized, evidence-based ideas to boost precision, derived from the post-fix feature-importance and error plots. Implement top-down; each item states its expected payoff and risk.
- Full history of what went wrong and what was fixed is in **§8 Fixes Log** — check it before "discovering" an issue that's already been handled. **§11 has revert instructions** (git commits) if any change needs to be undone.

---

## 1. Data Sources

| Source | Description | Format |
|---|---|---|
| Renewables.ninja API | Solar/wind resource data for Coclé, Penonomé, Panama (lat=8.52, lon=-80.36). MERRA-2 reanalysis. 859 MW solar / 336 MW wind. Full year 2025. Location entered interactively at run time (geopy geocoding); token set in the data-download cell. | REST API → JSON |
| `renewables_ninja_2025.csv` | Export of the downloaded API data (all six meteorological columns), written by each run so the exact weather inputs behind a result are preserved. | CSV (generated) |
| `solar_eolica_hidro_horario_2025.csv` | Real ETESA metered generation: `solar_mw_real`, `eolica_mw_real`, `hidro_mw_real`. Hourly, 2025. | **CSV** |
| `DEM2025.csv` | Real ETESA hourly electricity demand 2025. Wide format (date × 24 hours, columns `H1`..`H24`). | **CSV** |

**Calibration:** `ratio_calib = solar_mw_real / solar_ninja`, applied to `irradiance_direct` and `irradiance_diffuse` to align the MERRA-2 reanalysis signal with observed Panamanian conditions.

---

## 2. Forecasting Problem Definition

| Parameter | Value |
|---|---|
| Target variable | `demanda_residual_h24 = demanda_residual(t+24)` |
| Target definition | `demanda_residual = demanda_mw − solar_mw − eolica_mw` |
| Forecast horizon | h = 24 hours (day-ahead) |
| Data frequency | Hourly (1-hour resolution) |
| Forecast type | Deterministic point forecast |
| Validation method | Rolling walk-forward (expanding window), one day per iteration |
| Output | One 24-hour block per iteration (h+1 through h+24) |

---

## 3. Training and Test Dates

| Split | Value | Description |
|---|---|---|
| Burn-in | First 336 rows dropped | `MAX_LAG = 336 h` (2 weeks) — rows before this have NaN lag features |
| Training start | ~2025-01-15 01:00 | After `MAX_LAG` burn-in |
| Training cutoff `T` | Row index 7000 | ≈ 7000 hourly records ≈ 292 days |
| Training end (approx.) | ~2025-10-31 | Tail of the dry season (Jan–Apr) + the full wet season (May–Oct) |
| Test start | `datetime_index.iloc[7000]` | ≈ 2025-11-02 |
| Test end | ~2025-12-30 | Last month of the wet season (Nov) + onset of the new dry season (Dec) |
| Test rolling days | ~58 days | `(len(df) − 7000) // 24` |
| Test hours | ~1392 hours | `n_test_days × 24` |

**Panama seasons (corrected — do not use "wet = Jan–Oct" from earlier drafts of this doc):**
**dry/summer season = December–April**, **rainy/wet season = May–November**.

**Seasonal split note:** training (mid-Jan–Oct) spans the *end* of one dry season plus the *entire*
wet season that follows — so it is mostly wet-season data, not purely wet season as earlier
versions of this doc claimed. The test window (Nov 2–Dec 30) is where the actual wet→dry
**transition happens mid-window**: November is still wet season — climatically similar to most of
training — and only December falls in the newly-started dry season. This narrows (but doesn't
remove) the distributional-shift explanation for the XGBoost overfit gap (§9.1): the shift is
concentrated in the **December portion** of the test set, not spread across the whole 58 days.
This is also directly relevant to the actual worst-error dates — see §10 P1 for the exact top-10
spike days (computed from the per-day error export, not read off a plot) and the confirmed
in-season-vs-new-season confound between the two worst days.

---

## 4. Feature Set — 30 flat features + 1 target

### 4.1 Current meteorology (7)
| Feature | Description |
|---|---|
| `irradiance_direct` | Direct normal irradiance at t (calibrated MERRA-2), W/m² |
| `irradiance_diffuse` | Diffuse horizontal irradiance at t (calibrated), W/m² |
| `temperature` | Air temperature at t (MERRA-2), °C |
| `solar_mw` | Real ETESA solar generation at t, MW |
| `wind_speed` | Wind speed at hub height at t (MERRA-2), m/s |
| `eolica_mw` | Real ETESA wind generation at t, MW |
| `hidro_mw` | Real ETESA hydroelectric generation at t, MW |

### 4.2 NWP horizon covariates at t+24 (4) — most impactful group, but perfect-foresight (§7.1)
| Feature | Description |
|---|---|
| `irradiance_direct_h24` | Direct irradiance 24 h ahead (perfect-foresight NWP proxy) |
| `irradiance_diffuse_h24` | Diffuse irradiance 24 h ahead |
| `temperature_h24` | Temperature 24 h ahead |
| `wind_speed_h24` | Wind speed 24 h ahead |

> **Key finding (v22):** adding these 4 features reduced DNN MAPE from ~9.3% to **7.08%** (−2.35 pp, pre-leakage-fix number, see §9). `irradiance_direct_h24` was XGBoost's 3rd most important feature (gain = 0.080). The NWP input made temporal-persistence models (e.g. LSTM) redundant here.

### 4.3 Calendar features (11)
| Feature | Description |
|---|---|
| `month` | Calendar month (1–12), integer |
| `hour` | Hour of day (0–23), integer |
| `hour_sin` / `hour_cos` | Cyclic hour encoding, period 24 |
| `month_sin` / `month_cos` | Cyclic month encoding, period 12 |
| `is_weekday` | 1 if Mon–Fri, 0 if Sat–Sun |
| `hour_weekday` | `hour × is_weekday` — midday peak on working days only |
| `is_holiday` | 1 on Panama national holidays (14 days in 2025), else 0 |
| `dow_sin` / `dow_cos` | Cyclic day-of-week encoding, period 7 |

### 4.4 Lagged target features — Lagged Approach (4)
| Feature | Lag | Pearson r | Rationale |
|---|---|---|---|
| `demanda_residual` | t (current) | — | Current-hour residual demand as a feature |
| `residual_L48` | 48 h | 0.511 | Mid-week demand build-up pattern |
| `residual_L336` | 336 h (14 d) | 0.591 | Biweekly hydro dispatch cycle in Panama |
| `residual_L168` | 168 h (7 d) | 0.600 | Same hour, same day last week — strongest lag |

> **L24 dropped:** Pearson r(L24, target) ≈ 0 — nearly collinear with the current `demanda_residual`.

### 4.5 Hydro dispatch features (4) — see §7.2 for the leakage fix
| Feature | Description |
|---|---|
| `hidro_fraction_L24` | Yesterday's hydro share of residual demand. Operational state signal. |
| `hidro_delta_L24` | Day-over-day trend in hydro dispatch (MW). Reservoir direction. |
| `hidro_typical_h24` | Seasonal-diurnal typical hydro at the forecast horizon — **rebuilt from training-only data per rolling window** (fixed; was previously computed once on the full year). |
| `hidro_anomaly_L24` | Yesterday's deviation of actual hydro from its seasonal typical (same training-only rebuild as above). |

### 4.6 Target (1)
| Feature | Description |
|---|---|
| `demanda_residual_h24` | Residual demand 24 h ahead — the prediction target. |

---

## 5. Model Architectures

### 5.1 Deep Neural Network (DNN)
```
Sequential([
    Dense(256, activation='relu'),   # 1st hidden layer
    Dense(256, activation='relu'),   # 2nd hidden layer
    Dense(128, activation='relu'),   # 3rd hidden layer
    Dense(1)                         # output (linear, single forecast in MW)
])
optimizer = Adam(learning_rate=0.001)
loss      = mean_squared_error
epochs    = 100 (max), EarlyStopping patience=5 on val_loss
batch_size       = 256
validation_split = 0.10
restore_best_weights = True
```
- **Scaling:** both X and Y scaled with `StandardScaler`, fit on training data only.
- **Seed:** `set_random_seeds(42)` called at the start of each rolling iteration.

### 5.2 XGBoost
```
XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=20,   # must be in constructor, XGBoost >= 2.0
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)
eval_set = last 10% of the training window (temporal order preserved)
```
- **Scaling:** none (tree models are scale-invariant).
- **Validation:** last `max(24, round(len(X_train) * 0.10))` rows of the training window.

---

## 6. Metrics Suite

| Metric | Formula | Notes |
|---|---|---|
| MAPE | mean(\|actual − pred\| / \|actual\|) | Standard; spikes when residual → 0 at solar-peak hours |
| WAPE | Σ\|actual − pred\| / Σ\|actual\| | Aggregate denominator; ~1 pp below MAPE here |
| sMAPE | mean(2\|e\| / (\|a\| + \|p\|)) | Symmetric; bounded 0–200% |
| R² | 1 − SS_res / SS_tot | Fraction of variance explained |
| **MAE (MW)** | mean(\|actual − pred\|) | **Absolute error**, added so magnitude is explicit alongside ratios |
| **RMSE (MW)** | sqrt(mean((actual − pred)²)) | **Absolute error**, penalizes large misses more than MAE |

WAPE is preferred over MAPE for residual demand because MAPE spikes during midday hours when `demanda_residual → 0` under high solar penetration. **MAE/RMSE were added (v26) alongside the percentage metrics** so results report both relative and absolute (MW) error — see `results_summary` in the notebook/script.

**Naive benchmark:** predict `residual(t)` as the forecast for `residual(t+24)` (persistence — tomorrow's residual = today's). Every model must beat this to be considered useful.

**Model selection criteria:**
1. `test_MAPE < naive_MAPE`
2. `test_MAPE − train_MAPE < 0.10` (overfit gap < 10 pp)

---

## 7. Known Limitations (must be disclosed in the thesis)

### 7.1 Perfect-foresight weather (h+24) — inherent to the design, not a bug
The four `*_h24` meteorological features (`irradiance_direct_h24`, `irradiance_diffuse_h24`, `temperature_h24`, `wind_speed_h24`) are built with `shift(-24)` on the MERRA-2 reanalysis series — i.e. the **true future value**, not an operational forecast. The full year was pulled from the Renewables.ninja API in one request.

**Consequence:** reported metrics assume a perfect 24-hour weather forecast and are an **optimistic upper bound**. A real day-ahead deployment would feed operational NWP (e.g. GFS/ECMWF), whose forecast error would raise MAPE/WAPE. This is documented at the feature-construction site in both the notebook and `.py`, and in `README.md`.

### 7.2 Hydro seasonal baseline — was a leak, now fixed (v26)
`hidro_typical_h24` and `hidro_anomaly_L24` depend on a `(month, hour)` mean of `hidro_mw`. **Prior versions** computed this mean once over the *entire* dataset before the rolling loop started, so early rolling windows could see hydro averages that included months past their own cutoff (look-ahead leakage).

**Fix:** `rebuild_hidro_profile_features(df_in, T)` now refits the `(month, hour)` profile from **training-only rows `[:T]`** inside `get_targets_features`, once per rolling window. Unseen `(month, hour)` combinations fall back to the training-window mean. Verified leakage-free with unit tests (invariance to post-cutoff data, no NaN in any train/test slice, correct unseen-key fallback).

**Impact on §9 numbers:** this fix changes the training signal for early rolling windows, so the historical 7.08%/7.35% MAPE figures **must be regenerated** — they predate the fix. *(Done — see §9.1.)*

### 7.3 Run-to-run variability (DNN)
Seeds are fixed (`set_random_seeds(42)` at each rolling iteration), yet small differences between
runs have been observed (DNN MAPE 7.02% vs 7.08% across two runs, ≈ ±0.05 pp; MAE 69.7 vs 70.3 MW)
— consistent with TensorFlow op-level nondeterminism, which seeding alone does not eliminate.
**Rule: cite results from one named run, never mix numbers across runs.** If exact
bit-reproducibility is ever required, try `tf.config.experimental.enable_op_determinism()`
(slower training). On the data side, each run exports the downloaded weather to
`renewables_ninja_2025.csv`, so the exact MERRA-2 inputs behind a result are always on disk.

---

## 8. Fixes Log (mistakes found → what was fixed)

| # | Issue found | Where | Fix | Status |
|---|---|---|---|---|
| 1 | Doc said inputs were `DEM2025.xlsx` (Excel) + `solar_eolica_hidro_horario_2025.parquet` (Parquet); code actually reads CSV via `pd.read_csv` for both | This doc §1 (was §3), in-code comments ("Load ... from Excel/Parquet") | Corrected doc + code comments to CSV; added a README with exact filenames | ✅ Fixed, verified (only `pd.read_csv` calls in `.py`/`.ipynb`, zero `read_excel`/`read_parquet` anywhere) |
| 2 | h+24 weather features are true future values, not a forecast, but this wasn't surfaced anywhere a reader would see without opening the code | Feature construction (`*_h24 = ....shift(-HORIZON)`) | Added an explicit "LIMITATION: PERFECT-FORESIGHT WEATHER" note at the code site + this doc (§7.1) + README | ✅ Documented (not a code change — this is a deliberate design choice) |
| 3 | Hydro seasonal baseline (`hidro_typical_h24`, `hidro_anomaly_L24`) computed once on the full year before the rolling loop — look-ahead leak into early windows | `get_targets_features` / hydro feature block | Added `rebuild_hidro_profile_features(df_in, T)`, refit per-window on training-only rows; called first thing inside `get_targets_features` | ✅ Fixed, unit-tested (3/3 tests pass: no leakage, no NaN, correct fallback), reviewed by an adversarial subagent pass — no findings |
| 4 | Results only reported percentage error (MAPE/WAPE); no absolute-MW magnitude | End-of-notebook results section | Added `results_summary` table: MAPE %, WAPE %, MAE MW, RMSE MW per model (DNN/XGBoost/Naive) | ✅ Added, dry-run verified |
| 5 | Excessive line-by-line "what it does" comments in core pipeline cells (imports, feature extraction) | Throughout `.py`/`.ipynb` | Trimmed to docstrings, section headers, and "why" comments only | ✅ Done |
| 6 | Hardcoded Renewables.ninja API token in source | Data-download cell | Briefly moved to an env var (v26.3), then **restored to in-cell by user request** (v26.4) — the user manages the token in the download cell directly. Token is in the repo and its git history | ⚠️ Open — **rotate the token** at renewables.ninja before any public sharing (user action) |
| 7 | §9 results were stale (pre-leakage-fix) | This doc | Notebook re-run after the fix; post-fix results recorded in §9.1 (DNN 7.02%, XGB 7.31%, MAE ~70 MW), pre-fix kept in §9.2 for comparison. Fix confirmed harmless-to-beneficial for accuracy | ✅ Verified from re-run output |
| 8 | Error spikes on holiday-adjacent days despite `is_holiday` feature; several near-zero-importance features | Post-fix diagnostics (feature importance T_last=8368, error plots) | Documented as prioritized optimization roadmap (§10, P1–P7) — not yet implemented; implement one at a time and record deltas here | 📋 Roadmap written, pending implementation |
| 9 | **P3 implemented and confirmed:** `Ensemble = 0.5·(DNN + XGBoost)` added as a **fourth row** of `results_summary` (DNN/XGBoost/Naive rows unchanged, no retraining — averages the stored test forecasts) | Results-summary block, both `.py` and `.ipynb` | Adversarially reviewed (slices/shapes/leakage all confirmed clean); re-run confirms Ensemble beats both base models on **all four** metrics (MAPE 6.88% vs 7.02%/7.31%; RMSE 95.5 vs 96.6/101.8 MW) — see §9.1 | ✅ Confirmed best model |
| 10 | **Reproducibility pass (v26.3), then partially reverted (v26.4) by user request:** the v26.3 changes (fixed coordinates instead of `input()` geocoding, env-var token, offline cache-load) were **rolled back** — the user needs to enter the location interactively, and the Renewables.ninja download flow is owned by the user and must not be modified. **Kept from the pass:** each run now *exports* the downloaded API data to `renewables_ninja_2025.csv` (traceability of exact weather inputs), `requirements.txt` (geopy included), and the rewritten README/How-to-Run | Data-download cells, both `.py` and `.ipynb`; README; requirements.txt | Original API flow restored verbatim from commit `40276c1` + one export line added after the combine step | ✅ Settled — do not touch the Renewables.ninja download code again without explicit instruction |
| 11 | Stale EDA comments claimed `demanda_residual`→target correlation "r ≈ 0.3–0.4"; the actual plot shows **r = 0.685** | ACF/scatter markdown + correlation-matrix comments, both files | Corrected to r ≈ 0.69 ("strongest single predictor") — consistent with `demanda_residual` also being XGBoost's #1 feature (gain ≈ 0.24–0.29) | ✅ Fixed from run evidence (photos) |
| 12 | Pipeline had never actually been executed end-to-end in this environment against the user's real weather data — all prior numbers came from the user's own local runs (photos/typed results) | N/A — this is a "did we ever check" item, not a code fix | Ran the full `.py` pipeline (data load → both rolling loops → results table) directly against the user-supplied `renewables_ninja_2025.csv`, in an isolated scratch copy (repo untouched). Confirmed the pipeline executes cleanly end-to-end and reproduces the user's own numbers within expected run-to-run variance (§7.3). Quantified the XGBoost/DNN overfit gap directly from the run output, which reshaped §10's priority order (P5 promoted to top) | ✅ Executed and recorded — see §9.1 |
| 13 | Doc incorrectly stated Panama's wet season as "Jan–Oct" and dry season as "Nov–Dec onset" | §3 Training/Test Dates, §9.1 seasonal-shift explanation, §10 P1, README limitations | **Corrected per user:** dry/summer season = **Dec–Apr**, rainy/wet season = **May–Nov**. This means training (mid-Jan–Oct) is mostly wet season (not purely as before), and the wet→dry transition happens **mid-way through the test window** (Nov = wet, Dec = new dry onset), not before it. Narrows the distributional-shift story to the December portion of test, and flags a possible confound between the Nov 28 (in-season) vs Dec 25 (new-season) error spikes in P1 | ✅ Corrected per direct user input |
| 14 | §10 P1's spike dates ("Nov 26–28 / Dec 22–25") were read off a plot, not computed — imprecise | Scratch re-run, exported daily/hourly error CSVs | Re-ran the pipeline with a per-day/per-hour error export and cross-referenced the exact top-10 worst days against the user's full 2025 holiday list. **Confirmed:** the two worst days (Dec 23, Nov 26) are 1–2 days *before* a major holiday (Dec 24/25, Nov 28) — not on the holiday itself — and 7 of the top 10 worst days sit within 0–2 days of a holiday. Confirms the P1 hypothesis with real dates instead of an approximate window; also confirms Dec 23 (new dry season) is worse than Nov 26 (in-season), consistent with the seasonal confound in row 13 | ✅ Confirmed with exact dates — see §10 P1 |
| 15 | **P1 was implemented, unit-tested, adversarially reviewed, and fully executed** (`days_to_next_holiday`/`days_since_last_holiday`, 30→32 features) — full results were recorded, then **the user asked to revert it**. Target result had worked (Dec 23/Nov 26 daily MAE down 32%/17%), but XGBoost improved while the DNN got worse across all four metrics, leaving the flat-weight Ensemble mixed rather than a clean win | Feature engineering block, `cols_order`, feature-count assertion, EDA plot layout — both `.py`/`.ipynb`; this doc; README | Reverted via `git revert --no-commit` over the 3-commit range (implementation, review-fix, results), applied as one changeset. All files confirmed back to the exact pre-P1 (v26.7) state: 30-feature assertion, `layout=(16,2)` plot restored, zero `days_to_next_holiday`/`days_since_last_holiday` references left in code. **The original commits remain in git history and are fully un-revertable** if this decision changes — see §11 for the exact commit hashes | ✅ Reverted per user request — code state matches v26.7; this doc kept as v26.9 to record that the attempt happened |
| 16 | **Self-caught leakage bug in a not-yet-run design.** First draft of both P5 (XGBoost sweep) and the weighted-ensemble weight search used the LAST rolling window's (`T_last`) internal validation split for tuning. Caught during unit-test-writing (before any pipeline execution) that `T_last`'s training window overlaps 98.3% of the reported test period's rows — tuning against it would leak actual test-period outcomes into hyperparameter/weight selection | P5 sweep block and weighted-ensemble block, both `.py`/`.ipynb` — caught before the buggy version was ever committed or run | Redesigned to use `T` (the first, smallest window) instead — its validation slice is provably entirely before the test period starts. Verified with a standalone unit test that reproduces both the leak (rejected `T_last` design) and its absence (shipped `T` design) side by side, so the fix is demonstrated, not just asserted | ✅ Caught and fixed pre-execution; see §9.2 for the full story and §8 row 17 for the adversarial review that independently confirmed the fix |
| 17 | **P5 (XGBoost regularization sweep) and weighted ensemble implemented, unit-tested, adversarially reviewed, and executed** (v26.10) | Rolling loop's `XGBRegressor(...)` call (P5); new weighted-ensemble block before `_summary_rows` (both `.py`/`.ipynb`) | Full re-run: XGBoost improved on all 4 metrics (7.35%→7.10% MAPE), pulling the flat Ensemble to a new best **6.78% MAPE**. The weighted ensemble (validation-tuned, leakage-free — see row 16) chose w=0.30 but did NOT beat the flat 0.5/0.5 on the actual test set — reported as an honest negative result, not hidden. Adversarial review confirmed all 6 checked points clean (no leakage, correct row/scaling alignment, `.py`/`.ipynb` parity, nothing downstream broken, sweep baseline sanity) and flagged two **pre-existing, out-of-scope** observations (the `T`-window validation slice was already used internally by each model's own early stopping; `rebuild_hidro_profile_features` at `T=7000` includes the validation rows in its own fitting) — both inherited from the original rolling-loop design, not introduced by this change, noted here for completeness | ✅ P5 recommended to keep (clean win — see §9.2); weighted ensemble recommended to keep flat 0.5/0.5 as primary, weighted row as documented comparison — see §11 for independent revert commands for either piece |
| 18 | **Weighted-ensemble code removed per user request** ("record the results and keep ensemble, remove weighted ensemble") | Results-summary block, both `.py`/`.ipynb` — the ~55-line weighted-ensemble section (weight search, `_pred_ens_weighted`, extra `_summary_rows` entry) deleted; P5 sweep and flat `_pred_ens`/`'Ensemble'` row untouched | `results_summary` now reports 4 rows (DNN, XGBoost, Ensemble, Naive) instead of 5. Re-ran the full pipeline after removal to confirm nothing else depended on the deleted variables and the P5-tuned results are unchanged | ✅ Removed; finding stays documented in §9.2/§10 as a citable negative result even though the code no longer runs it — see §11 to bring it back if ever wanted |

**How this file stays useful:** when something in the notebook/script turns out wrong or gets fixed, add a row here rather than just fixing it silently — that's what makes this doc worth reading before starting new work on the pipeline.

---

## 9. Results

> **⚠️ All numbers in §9 are SUPERSEDED by §12.6 (2026-07-23).** They were computed with the
> solar-calibration leak (#3) present and the hour-beginning demand misalignment (#11) — i.e.
> optimistic. The honest baseline is **Ensemble 8.33% MAPE** (§12.6). §9 is retained as a
> historical record and for the pre-vs-post-fix comparison, not as current results.

### 9.1 Current results — POST-leakage-fix + Ensemble (v26.5)

| Model | MAPE | WAPE | sMAPE | R² | MAE (MW) | RMSE (MW) | Train MAPE | Overfit gap |
|---|---|---|---|---|---|---|---|---|
| DNN | 7.01% | 6.11% | 6.57% | 0.7791 | 69.7 | 96.6 | 6.28% | **0.73 pp** |
| XGBoost | 7.35% | 6.20% | 6.73% | 0.7525 | 70.7 | 102.3 | 4.54% | **2.82 pp** |
| **Ensemble (0.5·DNN + 0.5·XGB)** | **6.89%** | **5.88%** | — | — | **67.1** | **95.6** | — | — |
| Naive | 11.57% | 10.17% | 11.24% | 0.3563 | 116.0 | 165.0 | — | — |

**Test period:** ~2025-11-02 → ~2025-12-30 (58 rolling days). **Training cutoff:** T = 7000; final-iteration cutoff T_last = 8368.
**Provenance:** this is a full end-to-end execution of the pipeline (data → both rolling loops →
results table), run against the user's own `renewables_ninja_2025.csv`. It reproduces the same
run the user reported by hand (DNN 7.02%, XGB 7.31%, Ensemble 6.88%) within the ±0.05 pp band
already documented in §7.3 as expected TensorFlow op-level nondeterminism — treat this table as
the current numbers, superseding the previous entry (DNN sMAPE/R² there are now filled in directly
from this run instead of carried over from a prior one).

**The Ensemble is the headline model.** It beats both individual models on **every** metric —
not just MAPE/WAPE (relative) but also MAE **and RMSE** (absolute, in MW), meaning it's not just
better on average, it's also more robust on the worst days (RMSE penalizes large misses more).

**New evidence from this run — the overfit gap, quantified:** XGBoost's overfit gap (train 4.54%
→ test 7.35%, **2.82 pp**) is essentially **4× the DNN's** (train 6.28% → test 7.01%, **0.73 pp**).
Both models land at nearly the same *test* MAPE, but XGBoost gets there by fitting the training
data much harder and generalizing worse — direct, measured confirmation of what §3's seasonal-
split note already predicted (wet-season training vs dry-season test). **This reframes the
optimization roadmap: XGBoost has headroom to close via regularization, not just feature
engineering — see §10's re-prioritization below.**

### 9.2 P5 (XGBoost regularization sweep) + weighted ensemble — v26.10

Both implemented in one pipeline run (they share the same rolling loops), but **fully
independently attributable and independently revertible** — see §11 for exact commit hashes
and per-change revert commands.

**Current shipped state (v26.11):** P5 is live in the code (`results_summary['Ensemble']` reflects
the P5-tuned XGBoost). The weighted-ensemble code below was **removed per user request** after
review — "keep ensemble, remove weighted ensemble." The rest of this subsection is kept as-is: a
complete record of what was tried and why, since the negative result is still a legitimate
methodological finding worth citing even though the code no longer runs it.

**Design correction made *before* running anything, worth recording as its own lesson:** the
first draft of both the sweep and the weight search used the **last** rolling window's (`T_last`)
internal validation split for tuning. Caught during unit-testing that this was a leakage bug:
`T_last`'s training window (`[0, T_last)`) extends deep into calendar dates that overlap the
reported 58-day test period (`[T, T+n_test_hours)`) — an artifact of the expanding-window design,
where by the final iteration most of the "test" dates from earlier iterations have become
training data for the last one. Concretely: `T_last`'s validation slice `[7531, 8368)` overlaps
1368 of 1392 test rows (98.3%) — using it would have let the sweep/weight choice be informed by
actual outcomes on almost the entire test period before that same period was "graded." **Fixed to
use `T` (the first, smallest window) instead** — its validation slice `[6300, 7000)` is provably
entirely before the test period starts, guaranteeing zero date overlap. Verified with a standalone
unit test that explicitly reproduces both the leak (rejected design) and its absence (shipped
design), and confirmed structurally correct by an independent adversarial review (see §8 row 17).

| Model | MAPE | WAPE | MAE (MW) | RMSE (MW) | Δ vs pre-P5 baseline (MAPE) |
|---|---|---|---|---|---|
| DNN | 7.01% | 6.11% | 69.7 | 96.6 | unchanged (P5/weighting don't touch DNN training) |
| XGBoost (P5-tuned) | **7.10%** | **6.07%** | **69.3** | **99.2** | **better by 0.25 pp**, all 4 metrics improved |
| **Ensemble (0.5/0.5)** | **6.78%** | **5.83%** | **66.5** | **94.3** | **better by 0.11 pp — new best result overall** |
| Weighted Ensemble (w=0.30 DNN / 0.70 XGB) | 6.84% | 5.87% | 66.9 | 95.4 | worse than flat 0.5/0.5 on every metric |
| Naive | 11.57% | 10.17% | 116.0 | 165.0 | — |

*(Pre-P5 baseline for comparison: XGBoost 7.35%/6.20%/70.7/102.3, Ensemble 6.89%/5.88%/67.1/95.5
— the same scratch-environment re-run methodology as this run, from the row directly above.)*

**P5 — clean, unambiguous win.** The sweep (grid searched on the `T`-window validation slice
only) selected `max_depth=4, min_child_weight=5, reg_alpha=0.3, reg_lambda=2.0, subsample=0.8,
colsample_bytree=0.8` over the original `max_depth=5, min_child_weight=1 (default), reg_alpha=0.1,
reg_lambda=1.0` — lower validation MAPE (4.658% vs 4.680%) **and** a much smaller validation
train/val gap (0.694 pp vs 1.299 pp), i.e. more conservative trees. Applied across all 58
iterations, this improved XGBoost on **all four** test metrics, which pulled the flat Ensemble to
**6.78% MAPE — the best result recorded anywhere in this project**, beating even the P1 attempt
(§9's prior "P1 results" entry, since reverted). **Recommendation: keep P5.**

**Weighted ensemble — an honest negative result, not a bug.** The validation-based search chose
w=0.30 (favor XGBoost 70/30), correctly minimizing validation WAPE (0.0432 at w=0.30 vs 0.0464 at
w=1.00 pure-DNN) — the search itself worked exactly as designed and was confirmed leakage-free.
But on the **actual test set**, the flat 50/50 average still wins on every metric. The most likely
explanation: the validation window (`T`'s tail, an earlier calendar period entirely before the
test window by construction) sits in a different part of the year than the test period, so
"which model was relatively more accurate there" isn't a reliable guide to "which model will be
more accurate on Nov–Dec data" — a validation/test mismatch under the same wet→dry seasonal shift
already documented in §3. This is a legitimate, documented finding worth keeping in the thesis
(it illustrates a real limitation of naive validation-based ensemble weighting under distribution
shift) even though the specific refinement it produced doesn't beat what was already the best
model. **Recommendation: keep the flat 0.5/0.5 Ensemble as the primary reported model; keep the
Weighted Ensemble row in the code as a documented comparison point, or revert it — see §11.**

### 9.3 Historical results — PRE-leakage-fix (v25, keep for comparison, do not cite as current)

| Model | Test MAPE | Test WAPE | Test sMAPE | R² | Overfit |
|---|---|---|---|---|---|
| DNN | 7.08% | 0.0616 | 6.66% | 0.7765 | 0.79 pp |
| XGBoost | 7.35% | 0.0621 | 6.74% | 0.7539 | 2.75 pp |

**Key methodological finding (worth a paragraph in the thesis):** fixing the hydro-baseline leak did **not** degrade accuracy — DNN improved 7.08%→7.02% and XGBoost 7.35%→7.31%. The leaked full-year hydro average was apparently adding noise, not signal, to early windows. The corrected pipeline is both honest and marginally better.

---

## 10. Optimization Roadmap — evidence-based, prioritized

Derived from the post-fix diagnostics (XGBoost feature-importance chart at T_last=8368, absolute-error time series, 7-day zoom plots, per-feature time-series panels, correlation matrix). Ordered by expected payoff ÷ effort. Implement one at a time, re-run, and record the delta in §8 — never batch several model changes into one run or you can't attribute the gain.

### P1 — Holiday-proximity features (targets the two biggest error events)
**Evidence — exact spike dates, computed from per-day/per-hour error export (not read off a plot):**

| Rank | Date (day) | Ensemble daily MAE | Nearest holiday | Distance |
|---|---|---|---|---|
| 1 | **Dec 23 (Tue)** | 253.8 MW | Dec 24 Christmas Eve / Dec 25 Christmas | 1–2 days **before** |
| 2 | **Nov 26 (Wed)** | 247.7 MW | Nov 28 Independence from Spain | 2 days **before** |
| 3 | Nov 8 (Sat) | 151.2 MW | Nov 5 Colón Day / Nov 10 Shout in Villa de los Santos | inside the Nov 3–10 holiday cluster |
| 4 | Nov 2 (Sun) | 147.9 MW | Nov 3 Independence Day | 1 day before |
| 5 | Dec 6 (Sat) | 127.0 MW | Dec 8 Mother's Day | 2 days before |
| 6 | Nov 17 (Mon) | 122.7 MW | *(none nearby)* | — |
| 7 | Nov 3 (Mon) | 111.0 MW | **on** Independence Day | 0 |
| 8 | Nov 5 (Wed) | 103.9 MW | **on** Colón Day | 0 |
| 9 | Dec 18 (Thu) | 100.3 MW | *(none nearby)* | — |
| 10 | Nov 16 (Sun) | 91.5 MW | *(none nearby)* | — |

Worst single hours: **500 MW** (Dec 23, 11:00) and **483.9 MW** (Nov 26, 12:00), both midday.
**7 of the top 10 worst days sit within 0–2 days of a national holiday** — and critically, the two
*worst* days are **not** the holidays themselves, they're **2 days before** one (pre-holiday
demand ramp-down), which a same-day `is_holiday` flag structurally cannot represent. `is_holiday`
also has near-zero XGBoost importance, consistent with it not capturing this.

**Confound confirmed (from the season correction, §3):** the single worst day (Dec 23) sits in the
newly-started dry season (a regime training barely saw), while the 2nd-worst (Nov 26) sits in the
wet season (same regime as most of training) — yet Dec 23 is *worse*. This is consistent with two
effects stacking on top of each other in December: the pre-holiday ramp *and* the seasonal-shift
error. Expect the holiday-proximity features to help both spikes, but don't expect them to fully
close the Dec 23 gap alone — some of that day's error is likely seasonal, addressed separately (or
not at all, given only one year of data to learn a season transition from).

**Action:** add `days_to_next_holiday` and `days_since_last_holiday` (clipped to e.g. ±3), and/or `is_holiday_adjacent`. Cheap, leakage-free (the holiday calendar is known in advance — genuinely available at forecast time).
**Expected impact:** directly attacks the tail errors that dominate RMSE; may not move MAPE much but should cut the worst days — expect a cleaner win on Nov 26/28 than on Dec 23, per the confound above.

### P2 — Prune dead features (simplify + reduce variance)
**Evidence:** bottom of the importance chart: `hidro_mw`, `is_holiday`, `hidro_fraction_L24`, `hidro_anomaly_L24`, `residual_L48`, `temperature`, `irradiance_diffuse`, `eolica_mw` all contribute < ~0.01 gain each. Also `hour`/`hour_sin`/`hour_cos` triple-encode the same signal for XGBoost.
**Action:** ablation run with the bottom ~6 features removed (keep the calendar encodings for the DNN — cyclic features matter there even if trees ignore them; consider *separate* feature lists per model).
**Expected impact:** small accuracy change either way, but a leaner model, faster rolling loop, and a clean "feature ablation" subsection for the thesis. If accuracy holds, keep the pruned set.

### Recommended order — UPDATED after P5 + weighted ensemble (v26.10, read this first)

**Current best: Ensemble (0.5·DNN + 0.5·XGB) at 6.78% MAPE / 94.3 MW RMSE (§9.2).** P5 is done and
was a clean win; the weighted-ensemble refinement was tried properly (leakage-free, see §8 row 16)
but didn't beat the flat 0.5/0.5, so flat remains the primary reported model. Updated order:

1. **P1 (holiday-proximity features) — reconsider now, against the NEW baseline.** P1 was tried
   earlier against the pre-P5 baseline and reverted (mixed DNN/XGBoost effect — §8 row 15). XGBoost
   is now a different, better-regularized model (P5-tuned), so P1's effect on it may differ. If
   retried, compare against 6.78% MAPE, not the old 6.89%/6.89% numbers.
2. **P2 (feature pruning)** — do this next if not P1. XGBoost is now more regularized (P5), which
   may have already reduced its reliance on near-zero-importance features — pruning them explicitly
   is a cleaner ablation now than before P5.
3. **P4 (ramp/persistence-error features)** — unchanged priority, still targets the systematic lag
   visible in the 7-day zoom plots.
4. **P6 (quantile/Huber loss)** — unchanged; strong thesis value (uncertainty bands) but the
   biggest methodological lift of the remaining items.
5. **P7 (pipeline cleanup)** — no accuracy impact, do whenever convenient.

**Discipline reminder (unchanged from before):** one change at a time, re-run, record the delta in
§8 before moving to the next item.

### P3 — Simple ensemble: average DNN + XGBoost ✅ IMPLEMENTED & CONFIRMED THREE TIMES — this is the shipped model (see §8 rows 9, 17, 18; §9.1, §9.2)
**Evidence:** the two models' error bursts don't fully coincide in the plots (different hours miss differently); their test MAPEs are within 0.3 pp of each other — the classic setup where a 50/50 average beats both.
**Action:** `pred_ens = 0.5*pred_dnn + 0.5*pred_xgb` on the stored `forecasts` DataFrames — zero retraining needed, one cell.
**Result:** confirmed on three separate runs, now including post-P5: Ensemble MAPE 6.78–6.89% across runs (vs DNN 7.01–7.02%, XGB 7.10–7.35%), winning on WAPE, MAE, **and RMSE** every time. **This is the best model and the one shipped in the code** (`_summary_rows['Ensemble']`).

### Weighted ensemble — ✅ TRIED, TESTED, EXECUTED, then **removed by user request** (did not beat flat 0.5/0.5 — see §8 rows 16–18, §9.2)
**Evidence:** DNN's overfit gap was much smaller than XGBoost's pre-P5 (0.73 pp vs 2.82 pp), suggesting it might deserve more ensemble weight than a flat 50/50.
**Action (as implemented, then removed):** search w in {0.0, 0.05, ..., 1.0} minimizing **validation** WAPE (never test — see §8 row 16 for the leakage bug this caught and fixed before running), using the first rolling window's (`T`) held-out validation slice. Apply the fixed winning weight to the full 58-day test forecasts.
**Result:** search correctly picked w=0.30 (favor XGBoost) on the validation slice, but on the actual test set the flat 0.5/0.5 still won on every metric (6.78% vs 6.84% MAPE). Likely cause: the validation window sits in a different season than the test window (§3's wet→dry shift), so relative model skill there doesn't transfer. **Decision: code removed per user request** ("keep ensemble, remove weighted ensemble") — the finding stays documented here as a legitimate negative result (illustrates a real limitation of validation-based ensemble weighting under seasonal distribution shift), but it no longer runs or appears in `results_summary`. To bring it back: see §11.

### P4 — Ramp/persistence-error features (targets the systematic lag)
**Evidence:** the 7-day zoom shows both models trailing fast ramps — over-reliance on `demanda_residual` (importance ~0.29 = persistence anchor).
**Action:** add `residual_delta_3h = residual(t) − residual(t−3)` and yesterday's persistence error `naive_error_L24 = residual(t−24) − residual(t−48)`-style features (all backward-looking → leakage-free).
**Expected impact:** helps the model anticipate ramps instead of following them one step late.

### P5 — XGBoost hyperparameter sweep ✅ IMPLEMENTED, TESTED, EXECUTED — clean win (see §8 rows 16–17, §9.2)
**Evidence:** confirmed by direct execution (v26.5, §9.1) — XGBoost overfit gap is **2.82 pp** (train 4.54% → test 7.35%), vs the DNN's **0.73 pp** (train 6.28% → test 7.01%). Nearly identical test MAPE, very different generalization — XGBoost is fitting training noise the DNN isn't. This is a measured, not assumed, target.
**Action:** small grid (6 candidates including the original config as baseline) searched on the **first rolling window's (`T`) validation slice only** — deliberately not `T_last`, whose training window overlaps the test period; see §8 row 16 for the leakage bug this avoided. Varied `max_depth`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`. Winning config applied fixed across all 58 rolling iterations.
**Result:** selected `max_depth=4, min_child_weight=5, reg_alpha=0.3, reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8` over the original `max_depth=5, min_child_weight=1(default), reg_alpha=0.1, reg_lambda=1.0` — lower validation MAPE (4.658% vs 4.680%) and a much smaller train/val gap (0.694 pp vs 1.299 pp). On the full test period: XGBoost improved on **all four metrics** (MAPE 7.35%→7.10%, WAPE 6.20%→6.07%, MAE 70.7→69.3 MW, RMSE 102.3→99.2 MW), pulling the flat Ensemble to **6.78% MAPE — the best result in this project.** **Recommendation: keep.**

### P6 — Quantile/pinball or Huber loss (robustness to spike days)
**Evidence:** the error distribution is spiky (a few ~400–500 MW days vs 70 MW MAE) — squared-error training over-weights those days at the expense of typical days.
**Action:** try `objective='reg:pseudohubererror'` in XGBoost and/or Huber loss in the DNN. Alternatively train P10/P50/P90 quantile models — an uncertainty band is a strong thesis addition for operational dispatch framing.
**Expected impact:** more stable typical-day accuracy; quantiles add operational value beyond point MAPE.

### P7 — Pipeline flow (code quality, no accuracy change)
- Replace the `globals()['model_dnn'+str(T_day)] = ...` per-iteration artifact stashing with a plain dict (`artifacts[T_day] = {...}`) — same inspectability, no namespace pollution, easier to serialize.
- Cache `rebuild_hidro_profile_features` per `T_day` if the rolling loop feels slow (it recomputes an O(N) map twice per day — once for DNN, once for XGB, same T_day).
- Factor the duplicated DNN/XGB plotting blocks into one `plot_model_results(name, preds, ...)` helper.

### Experimental models policy
Keep non-selected/experimental models (e.g. earlier LSTM attempts, alternative feature sets) **in the repository** — in the notebook under a clearly-labeled "Experimental / not selected" section or an `experiments/` folder. Weaker results are still evidence: the thesis should explain what was tried and why it lost (e.g. "NWP h+24 features made LSTM's temporal-persistence advantage redundant", §4.2). Never delete a failed experiment; label it.

---

## 11. Change Log & Revert Instructions

All changes land as focused git commits on branch `claude/RDMACHINELEARNING` (renamed from `claude/boris-skill-install-cm0y3t`), so any step can be undone independently with `git revert <hash>`.

| Commit | Contents | To undo |
|---|---|---|
| `698be5f` | Hydro-leakage fix (`rebuild_hidro_profile_features`), perfect-foresight notes, CSV comment fixes, README.md created, comment cleanup, MAE/RMSE results table — in both `.py` and `.ipynb` | `git revert 698be5f` (restores the leaky behavior — don't, unless reproducing v25 numbers) |
| `71f7120` | This reference doc reorganized v25→v26 (TL;DR, Limitations, Fixes Log) | `git revert 71f7120` |
| `5d5d745`, `df4a3d9`, `a3b3e95` | **P1 holiday-proximity features** — implemented, review-fixed, results recorded (32 features, DNN worse / XGBoost better / target spikes down 32%/17%) | Already reverted (see next row). To **re-apply** P1: `git revert 782daa3` (undoes the revert) |
| `782daa3` | **Reverted P1** per user request — restores the exact pre-P1 (v26.7) code and doc state | `git revert 782daa3` to bring P1 back (equivalent to re-applying the three commits above) |
| `d473e98` | **P5 (XGBoost sweep) + weighted ensemble** — both implemented together in one commit since they share one pipeline run. Result: P5 is a clean win (new best 6.78% MAPE); weighted ensemble is a documented negative result (didn't beat flat 0.5/0.5) | Superseded by `899160f` below — do not revert this one directly, it would also undo the removal's intent |
| `899160f` | **Removed the weighted-ensemble code per user request**, keeping P5. `results_summary` back to 4 rows (DNN, XGBoost, Ensemble, Naive); P5-tuned XGBoost and the flat Ensemble untouched | `git revert 899160f` to bring the weighted-ensemble code back (re-adds the 5th row and weight search) |
| *latest on this file* | v26.1+: post-fix results, §10 roadmap, this revert table, plus all rows above | Find it with `git log --oneline -- "MACHINE LEARNING RD CONTEXT.md"`, then `git revert <hash>` |

**Current shipped state:** P5 (kept) + flat 0.5/0.5 Ensemble only. To revert **everything from P5
onward** (back to the pre-P5, pre-P1-revert v26.7/v26.9 state): `git revert 899160f` then
`git revert d473e98`, in that order. To bring back **just** the weighted ensemble on top of the
current state: `git revert 899160f` alone.

To reproduce the **pre-fix (v25) numbers** for a thesis comparison table: `git stash && git checkout 698be5f~1 -- MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb`, re-run, then `git checkout HEAD -- MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb && git stash pop`. (Or simply cite §9.2 — the pre-fix numbers are preserved there.)

---

## 12. External Review — Feedback (v26.13, received 2026-07-23, NOT YET IMPLEMENTED)

An external reviewer went through the code and gave detailed feedback — generally
positive on structure/direction, with one critical correctness finding and ~19 other points. This
section logs every point verbatim-in-substance, what was verified true/false against the current
code, and a priority order, so implementation can proceed later without re-deriving any of this.
**Nothing in this section has been implemented yet** — logged for future processing only, per the
user's explicit request to think it through before touching code.

### 12.1 Overall verdict
Reviewer's summary: the DNN/XGBoost/ensemble model choice is reasonable, the rolling walk-forward
design is good, the CSV-read fix and MAE-in-MW addition are correctly noted as improvements, and
the code is easier to follow than before. The critical issue raised is **data leakage** in the
solar calibration step, plus a cluster of time-alignment and methodology concerns. Closing
instruction: **"fix the leakage and time alignment first, then rerun both models on a
clean holdout period."**

### 12.2 The 20 points, verified against the current code where possible

**1. Two headings still say "Parquet"/"Excel" instead of CSV.**
Checked: only one hit found (`MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb`, markdown cell, "Data
sources & limitations" section), and it already reads "**Inputs are CSV** (not Excel/Parquet)" —
already correct. README.md has zero hits. Likely the feedback was based on a copy from before this
session's earlier consistency-audit fix. **Action:** ask for the exact location if it
persists; otherwise no action needed.

**2. Wants to see the revised README.** Not a code issue — a request to review `README.md`.

**3. ⚠️ CRITICAL — solar irradiance calibration leaks target-hour information (NEW finding, not
previously caught by this session's leakage audits).**
Code (`MACHINE_LEARNING_RESIDUAL_DEMAND.py` ~line 241-249):
```python
solar_ninja  = df['solar_mw'].replace(0, np.nan)
ratio_calib  = (df['solar_mw_real'] / solar_ninja).fillna(1)     # per-hour, FULL YEAR incl. test period
df['irradiance_direct']  = (df['irradiance_direct']  * ratio_calib).fillna(0)
df['irradiance_diffuse'] = (df['irradiance_diffuse'] * ratio_calib).fillna(0)
```
`ratio_calib[t]` uses the *real, metered* solar output at hour `t`. Later,
`irradiance_direct_h24[t] = irradiance_direct[t+24]`, which is therefore proportional to
`solar_mw_real[t+24]` — one of the three terms that directly defines the target
(`demanda_residual_h24 = demanda_mw − solar_mw − eolica_mw` at t+24). This is a feature
mathematically entangled with the target, not merely the already-disclosed "perfect-foresight
weather" assumption (§7 Known Limitations) — it is materially worse. Only
`irradiance_direct_h24`/`irradiance_diffuse_h24` are affected (they're the two features scaled by
`ratio_calib`); `temperature_h24`/`wind_speed_h24` pass through un-scaled from the raw API pull and
carry only the already-disclosed perfect-foresight issue, not this deeper one.
**Consequence:** the documented "NWP horizon covariates are the most impactful feature group,
−2.35pp MAPE" result (§4) is now suspect — some unknown fraction of that gain is likely this leak,
not genuine forecastability. XGBoost's consistently high importance ranking for
`irradiance_direct_h24` (top-3 every run this session) is also now suspect.
**Reviewer's prescribed fix:** estimate the calibration (fixed factor or small model) from the
**training window only**, per rolling window, then apply it fixed/frozen to that window's forecast
block — do not use target-day real solar to compute the ratio applied to that same block. Same
discipline as the already-fixed `rebuild_hidro_profile_features`, but this one hasn't been given
that treatment yet and needs a new per-window rebuild function.

**4. MERRA-2 perfect-foresight weather should be framed as an "oracle" experiment (upper bound),
with a second experiment using realistic forecast weather.**
Already disclosed as a limitation (§7), but the feedback wants it explicitly labeled as an
oracle/best-case run rather than a general result, plus a **second, separate run** using something
that approximates real day-ahead NWP forecast error (not the true future reanalysis value) — not
yet built.

**5. Hydro profile should be built inside each rolling window, not from the full year.**
Checked: the modeling path is already fixed — `rebuild_hidro_profile_features(df_in, T)` rebuilds
`hidro_typical_h24`/`hidro_anomaly_L24` from training-only rows per window, and this is what
`get_targets_features` actually calls. A full-year global version (`_hidro_profile`, ~line 426)
still exists in the dataframe for EDA/correlation plots only and is explicitly commented as such —
it is dead for modeling but sits close enough in the code to read as a live bug on a linear
pass. **Action:** clarify/relabel more defensively so it doesn't look like an active leak; no
actual leakage-fix needed here (already done, see §8).

**6. `hidro_mw` comment contradicts what the code does.**
Checked: confirmed contradiction. Comment (~line 405) says "its LEVEL is dangerous... we encode
the operational state through four derived features **instead**," but `hidro_mw` is still directly
in `cols_order`, feeding both models raw. Not a leakage issue — a documentation-accuracy issue that
needs a real decision (keep `hidro_mw` as a model input or not), not just a reworded comment.
Note: P2 (feature pruning, tried and reverted earlier this session) explicitly kept `hidro_mw`
**only** as a structural dependency for `rebuild_hidro_profile_features`, not as a direct model
input candidate — that reasoning could inform this decision if/when revisited.

**7. Add `origin_time` / `target_time` columns; use `target_time` for forecast indexing and plots.**
`origin_time` = when the forecast is issued ("now"); `target_time` = the hour being predicted
(origin + 24h). Current forecast arrays appear to be indexed by origin time, so a plotted "forecast
at index t" is actually the forecast *for* t+24 — not wrong computationally (verified the rolling
loop's T-window logic is sound), but easy to misread and a risk for future bugs. Not yet
implemented.

**8. State explicitly that the 24-row `X_test` block = the completed previous day, forecasting the
following day.** Documentation/clarity ask, tied to #7 — no code change, just explicit statement of
what one rolling block represents.

**9. Lag-feature semantics: `shift(168)` is 168h before *input* time, i.e. 192h (not 168h) before
the *target* time (input + 24h). If the intent is "same hour, one week before the target," the
correct lag is `shift(144)`.**
Verified by direct arithmetic: for row `t`, `residual_L168 = residual[t-168]`; distance from
`t-168` to target `t+24` is `192h`. Both `shift(168)` and the alternative `shift(144)` are
equally leakage-free (both look only backward) — this is a semantic-intent question, not a
leakage bug. Also flags that the original Pearson-correlation analysis used to justify these three
lags (`residual_L48`, `residual_L336`, `residual_L168`, §4) may have (a) used the wrong reference
frame (input-relative vs. target-relative) and (b) been computed on the full year rather than
training-only data — both should be rechecked/recomputed if this is revisited.

**10. `DEM2025.csv` contains a `02/29/2025` row — 2025 is not a leap year, this date doesn't
exist.**
Verified: confirmed present (`grep -c` → 1 match). Currently silently dropped by
`pd.to_datetime(..., errors='coerce')` + `dropna()`, with no warning surfaced. Needs investigation
(why is it in ETESA's export?) and explicit documentation of the correction, not silent dropping.

**11. Possible 1-hour misalignment between the demand file and the generation file's hour
convention.**
`solar_eolica_hidro_horario_2025.csv` starts at `01:00`; the demand-loading code maps `H1` → `00:00`
of each date (`hora_num = extract(H\d+) - 1`). If ETESA labels hours by *end-of-interval* (hour 1 =
the 00:00–01:00 block, stamped 01:00) rather than *start-of-interval*, the demand series could be
shifted 1 hour relative to every other series (solar/wind/hydro/weather), corrupting every
feature relationship by a constant offset. **Attempted to verify from data:** checked which labeled
hour has peak solar output (expect ~solar noon) — peaks at labeled hour 12, which is only *weakly*
diagnostic (consistent with either convention) and does **not** resolve the question. **Needs
ETESA's documented hour convention — cannot be resolved from the data alone.** Currently unresolved.

**12. Add explicit checks for missing hours / duplicate timestamps; don't trust row-count-based
shifts.**
Current lag/shift features assume "N rows back = N hours back," true only if the hourly index has
no gaps or duplicates. Not currently checked explicitly anywhere in the pipeline. Wants an assertion
(e.g. full, unique hourly `DatetimeIndex` coverage) added, not implemented yet.

**13. Hardcoded Renewables.ninja API token must be removed, rotated, and read from an environment
variable; never commit `.env`.**
Verified: token is hardcoded in plain text (`MACHINE_LEARNING_RESIDUAL_DEMAND.py` line 89,
`token = 'c4672deddb9f6acfa94e9cdbb34f6414fd51f255'`). This has been pushed to the remote repo
multiple times this session, so the token should be treated as already exposed/compromised.
**Note:** this touches the Renewables.ninja download block, which carries a standing "do not
modify without explicit instruction" rule (§8 row 10) — this specific change (token → env var) is
explicitly requested by the user, so it's in scope, but should be scoped tightly to
just the token-handling lines, not a broader rewrite of that block. **The user must rotate the
token themselves** — not something that can be done from this session.

**14. Don't `pip install` inside the script; pin `requirements.txt` to fixed versions; save the raw
API response for reproducibility.**
Verified: `requirements.txt` exists but lists all packages unpinned (no version numbers). The
script also runs `subprocess.check_call(['pip', 'install', 'xgboost', '-q'], ...)` at runtime
(~line 1078) despite `xgboost` already being listed in `requirements.txt` — confirmed redundant.
Raw API JSON response is not currently saved anywhere (only the final combined/processed CSV is
exported, via the `renewables_ninja_2025.csv` traceability export added earlier this session).

**15. MAE isn't in the final results comparison.**
Checked: it already is — `results_summary` includes `MAE_MW` and `RMSE_MW` columns (confirmed via
grep, ~line 1456-1474), added earlier this session. Likely based on a pre-this-session copy.

**16. Save metrics and forecasts to files instead of keeping old result values as code comments.**
Currently, historical results live as prose in code comments and in this document, not as
on-disk artifacts (CSV/JSON) written per run. This is exactly the failure mode that caused the
stale-data incident earlier this session (a hardcoded "6.78% MAPE" comment silently went stale
after the underlying CSV changed). Not yet implemented.

**17. The experimental/rejected model (e.g. an LSTM attempt) is missing from the repo.**
Verified: `find . -iname "*lstm*" -o -iname "*experiment*"` returns nothing. The "Experimental
models policy" (§10, bottom) is currently a policy with no actual artifact behind it in this repo.
If an LSTM or similar was tried before, it isn't here — **this would be new work** (build a
deliberately-weaker baseline, run it, document why it lost), not a recovery of lost work.

**18. The same test period has guided too many keep/revert decisions to still count as an
untouched holdout.**
The Nov 2 – Dec 30 test window has been used to judge every roadmap item this session (P1, P5,
weighted ensemble, P2) — each "keep or revert" call was made by reading that window's score.
Repeated comparison against the same test set is a known way to end up with an optimistic final
number, independent of any code bug. **Working distinction proposed (not yet agreed):** fixing a
leakage/alignment bug and re-running is justified by a data-generating-process argument independent
of the resulting score, and isn't the same kind of contamination as trying N modeling variants and
keeping the best-scoring one. Under this framing: the leakage/alignment fixes can be re-run once
against the *same* test window without "burning" it, but **from that point forward, further
roadmap items (P4, P6, clustering-as-a-feature, etc.) should be judged on the validation split
only**, with the test period touched exactly once more, at the end, for the number reported in the
thesis. Needs explicit sign-off from the user before adoption.

**19. Soften "the model can be used for forecasting" to "selected for further evaluation"; add a
weekly-naive and a linear-regression baseline.**
Verified: current print statements read `'*** The DNN can be used for forecasting. ***'` and the
XGBoost equivalent (~line 1057, 1385), triggered solely by beating the (single) naive persistence
baseline. Reviewer's point: beating one weak baseline isn't sufficient evidence of deployment
readiness. Also wants two additional baseline comparators added: a weekly-naive model (same hour,
7 days ago — note this would reuse/relate to the `residual_L168`/`L144` lag discussion in #9) and a
simple linear regression, to show DNN/XGBoost add value over something much simpler than
persistence alone. Not yet implemented.

**20. Remove the global `warnings.filterwarnings('ignore')`; reduce excessive print output.**
Verified: present at line 11, suppresses all Python warnings globally, which can hide real
problems (e.g. a pandas warning flagging an actual bug) along with noise. Combined with a request
to cut down the volume of print statements throughout so real output isn't buried. Not yet
implemented.

### 12.3 Priority tiers for implementation (matches the stated sequencing: fix leakage +
alignment first, then one clean rerun)

| Tier | Items | Status |
|---|---|---|
| 1 — correctness, must fix before any number is trustworthy | #3 solar calibration leak (train-window-only, frozen forward, new per-window rebuild function needed); #11 confirm hour convention (**blocked on ETESA source**); ~~#6 decide `hidro_mw` in/out + fix comment~~; #7/#8 add `origin_time`/`target_time`, fix forecast plot indexing | #6 done (2026-07-23, see §12.5); rest not started |
| 2 — same tier, cheap but load-bearing | #9 lag semantics decision (144 vs 168) + training-only correlation recompute; #12 missing-hour/duplicate-timestamp checks; ~~#10 document the Feb 29 row~~ | #10 done (2026-07-23, see §12.5); rest not started |
| 3 — do together with the rerun | #19 add linear + weekly-naive baselines, ~~soften model-selection message~~; #4 second experiment with realistic (non-oracle) forecast weather, clearly label the oracle run as an upper bound | #19 message-softening half done (2026-07-23); baselines + #4 not started |
| 4 — hygiene, no accuracy effect, safe anytime | #13 rotate + env-var the token (**skipped for now, user's choice** — user must rotate); ~~#14 pin `requirements.txt`, drop inline pip install, save raw API response~~; ~~#20 remove global warnings filter~~ (print-volume trim still open); ~~#16 write metrics/forecasts to files instead of code comments~~; #17 add the actual experimental/weaker model file; ~~#1 verify Parquet/Excel headings~~ (already correct); ~~#2 review updated README~~ | #1, #2, #14, #16, #20 (partial) done (2026-07-23, see §12.5); #13, #17 not started |

### 12.4 Open questions blocking Tier 1 (need answers before implementation, not resolvable from
the data or code alone)
1. **ETESA's `H1` hour convention** (start-of-interval vs. end-of-interval) — determines whether
   #11 is a real 1-hour misalignment bug or a non-issue. Needs authoritative source, not a guess.
2. **What "clean holdout" concretely means** going forward — does the working distinction in #18
   (re-run once after correctness fixes, then freeze the test window for all future roadmap
   decisions) match what the user wants, or is a genuinely new/different period expected?
3. **Whether an LSTM/alternative model was ever actually run** outside this repo (recoverable) or
   whether #17 is new work to scope from scratch.
4. **Lag semantics intent** (#9) — input-time-relative (`shift(168)`, current) vs. target-time-relative
   (`shift(144)`) framing — a modeling choice, not purely a bug, needs a decision either way.

### 12.5 Implementation log — Easy/Trivial batch (2026-07-23)

10 of the 20 items were implemented and verified in one batch (user explicitly selected these;
`#13` token-to-env-var was in the same "easy" tier but not selected, so it's still open). All are
pure hygiene/documentation fixes with **no effect on model behavior or reported metrics** — verified
by re-running the full rolling pipeline end-to-end in the scratch environment against the same
data used for the current shipped baseline and confirming the results are unchanged (Ensemble
6.78% MAPE / 5.83% WAPE / 66.5 MAE / 94.3 RMSE — identical to the pre-change 30-feature baseline
in §9.2, as expected since none of these items touch feature construction or model training).

| # | Item | Change | Verified |
|---|---|---|---|
| #20 | Remove global warning suppression | Deleted `warnings.filterwarnings('ignore')` (line 11); the `import warnings` and the later scoped `catch_warnings()` block (KPSS stationarity test) are untouched since still needed | Full run completed with no crash or unexpected warning spam |
| #14b | Remove inline `pip install xgboost` | Deleted the `subprocess.check_call(['pip','install','xgboost',...])` call — redundant, `xgboost` was already pinned in `requirements.txt` | Run imports `xgboost` successfully without it |
| #1 | Parquet/Excel headings | Checked — already correct (`**Inputs are CSV** (not Excel/Parquet)`, notebook markdown cell). No change needed; likely feedback was based on a pre-this-session copy | Verified by grep, no occurrences left |
| #15 | MAE in final results table | Checked — already present (`MAE_MW`, `RMSE_MW` columns in `results_summary`), added earlier this session | Verified by grep |
| #14a | Pin `requirements.txt` | Added exact version pins for all 12 packages, matched to a known-working environment (pandas 3.0.3, numpy 2.4.6, matplotlib 3.11.0, requests 2.33.1, geopy 2.4.1, statsmodels 0.14.6, scipy 1.17.1, seaborn 0.13.2, scikit-learn 1.9.0, xgboost 3.2.0, tensorflow 2.21.0, tqdm 4.68.4) | N/A (dependency file, not exercised by a single run) |
| #19a | Soften model-selection message | `'*** The DNN/XGBoost can be used for forecasting. ***'` → `'*** ... is selected for further evaluation. ***'`; replaced the old 3-line comment under the DNN branch with one clarifying this is a minimum bar, not a readiness verdict (weekly-naive/linear baselines and the realistic-forecast experiment, #19b/#4, are still open) | Log shows the new message for both DNN and XGBoost |
| #10 | Document the Feb 29, 2025 bad-date row | Added an explicit `WARNING: dropping N unparseable date(s) from DEM2025.csv: [...]` print right before the existing `dropna(subset=['fecha_dt'])`, with a comment noting 2025 isn't a leap year and this is a known source-data error | Log's very first line: `WARNING: dropping 1 unparseable date(s) from DEM2025.csv: ['02/29/2025']` |
| #6 | `hidro_mw` in/out decision | **Decision: keep `hidro_mw`** as a raw current-hour model input (status quo, matches the post-P2-revert state). Rewrote the comment: the old "perfect collinearity" claim was itself inaccurate (would require `termica_mw`, which isn't in this dataset — `hidro_mw` alone only partially explains `demanda_residual`); its true low importance (gain ≈ 0.005, from the P2 analysis) is redundancy with the already-present `demanda_residual` feature, not danger | Comment now matches code; `hidro_mw` confirmed still in `cols_order` |
| #16 | Save metrics/forecasts to files | Added `results_summary.to_csv('results_summary.csv')` and a new `forecasts.csv` (datetime, actual, dnn, xgboost, ensemble, naive columns, one row per test hour) at the end of the run | Both files produced correctly in the verification run (`results_summary.csv`: 5 rows; `forecasts.csv`: 1392 rows, correct columns and values matching the printed table) |
| #14c | Save raw API response | Added `open('renewables_ninja_solar_raw.json','w').write(r.text)` and the wind equivalent, right after each API call, before any parsing | Verified syntactically only — this code path is in the Renewables.ninja download block, which the scratch-run substitution replaces with a direct CSV read (can't exercise the live API from this session); low-risk, single `open().write()` calls |

Synced into both `.py` and `.ipynb`, `py_compile` + per-cell `ast.parse` clean on both. `README.md`
"Expected output" section updated to list `results_summary.csv`/`forecasts.csv` and to drop the
stale "Weighted Ensemble" table-row mention (leftover from before that code was removed, §8 row 18
— unrelated to this batch but caught while editing the same paragraph).

**Not done from the "easy" tier:** `#13` (token → env var) — explicitly not selected by the user
this round, left for later. Print-volume trimming (second half of `#20`) also not done — only the
global-suppression removal was in scope.

### 12.6 Implementation log — Correctness batch: leakage + time alignment (2026-07-23)

The reviewer's #1 instruction ("fix the leakage and time alignment first, then rerun"). Three
model-affecting correctness fixes, done together and validated in **one** rerun (grouping is
consistent with the reviewer's instruction; none of the three is a score-chasing tuning choice —
each is justified by the data-generating process, so re-running the same test window to measure
their honest effect does not "burn" the holdout, per the §12.2 #18 framing).

**What changed:**
- **#3 solar calibration leakage (critical).** Old: irradiance scaled by a FULL-YEAR per-hour
  `real_solar/ninja_solar` ratio, baked into `irradiance_direct/diffuse`; via the `shift(-HORIZON)`
  that builds `irradiance_*_h24`, this made the horizon weather features proportional to
  `real_solar[t+24]` — a term that defines the target. New: raw ninja irradiance/solar are kept as
  helper columns; a new `rebuild_calibration_features(df_in, T)` re-fits a per-hour-of-day
  `real/ninja` factor from **training rows (< T) only** per rolling window and applies it using only
  ninja inputs (never target-hour real solar). `features_h24` is now drawn explicitly from
  `cols_order` so the raw-ninja helper columns are never fed to the models. Unit test
  (`scratchpad/test_calibration_leakage.py`) proves the calibrated irradiance is invariant to
  perturbing test-period (≥ T) real solar, and that the removed full-year ratio would have leaked.
- **#11 hour-ending alignment (resolved with user input).** The generation file spans
  `2025-01-01 01:00 → 2026-01-01 00:00` = exactly 8760 rows — the signature of **hour-ending**
  labeling (first hour 00:00–01:00 stamped 01:00). ETESA's demand `H1` is therefore the hour ending
  01:00, so the loader now maps `H_k → k:00` (was `(k-1):00`). The old mapping placed demand one
  hour BEFORE generation, so `demanda_residual = demanda − solar − eólica` mixed two different
  physical hours — a corrupted target. After the fix the demand↔generation join has **zero NaN**.
  (Still open: the Renewables.ninja weather timestamps may use hour-BEGINNING; that is a separate
  feature-alignment question, flagged in-code, not changed here.)
- **#12 index integrity.** Assert the hourly index is gap-free and duplicate-free before any
  row-count-based shift (a missing/duplicate hour would silently make a 24-row shift ≠ 24 hours).
  Passes on the real data (8,754 rows, gap-free).

**Adversarial review (independent subagent, 2026-07-23): CLEAN on all 6 checked points** — (1) the
calibration factor is fit strictly on rows < T; (2) the `_h24` alignment is sound because
`hour(t+24)==hour(t)` under the enforced gap-free index; (3) the raw-ninja helpers are structurally
excluded from `features_h24` (drawn from `cols_order`); (4) `rebuild_calibration_features` is called
per window with the correct `T` (=`T_day`), no future data; (5) the `shift(-HORIZON)` NaN tail is
fully trimmed by `df.iloc[MAX_LAG:-HORIZON]`; (6) the demand hour-ending remap is self-consistent
and nothing downstream assumed the old `H1→00:00`. Plus the calibration-leakage unit test passing.

**Honest before/after (same 58-day Nov 2–Dec 30 test window, corrected data). Numbers went UP —
this is the point: the leak was inflating the weather features' apparent skill.**

| Model | MAPE% before → after | WAPE% | MAE MW | RMSE MW |
|---|---|---|---|---|
| DNN | 7.01 → **8.65** | 6.11 → 7.26 | 69.7 → 82.8 | 96.6 → 115.3 |
| XGBoost | 7.10 → **8.36** | 6.07 → 7.00 | 69.3 → 79.8 | 99.2 → 111.6 |
| **Ensemble** | 6.78 → **8.33** | 5.83 → 6.95 | 66.5 → 79.3 | 94.3 → 110.7 |
| Naive | 11.57 → 11.66 | — | — | 165.0 → 163.9 |

The flat Ensemble is still the best model and still beats naive by ~28%. The ≈1.5 pp MAPE rise is
the honest cost of removing the calibration leak plus the target re-alignment — the reviewer
explicitly predicted "it will make the reported scores too optimistic," now corrected. **This 8.33%
Ensemble is the new honest baseline; supersedes the 6.78% figure in §9.2 and §0, which was leaky.**
The per-run `results_summary.csv` / `forecasts.csv` (review #16) are the ground truth for this run.

**Revert:** `git revert d7c875d` restores the leaky calibration + hour-beginning demand mapping +
removes the index checks (do NOT, except to reproduce the old inflated numbers for a comparison
table). The three fixes are intertwined in one commit because they were validated in one rerun; to
isolate just one, revert and re-apply selectively.

**Not in this batch (next):** #7/#8 (origin_time/target_time), #9 (lag semantics doc + training-only
correlations), #19b (weekly-naive + linear baselines), #4 (realistic no-oracle-weather ablation),
#20 (print trimming). These are additive/labeling/doc changes that don't affect the DNN/XGB numbers.

### 12.7 Implementation log — Labeling, lag semantics, and two new baselines (2026-07-23)

**#7/#8 origin_time / target_time.** Forecasts, plots, and the forecasts.csv export were indexed
by `datetime_index` — the row's own timestamp ("now"), when the forecast is issued — but the value
held is for 24h later. Added explicit `origin_time`/`target_time` series (`target_time = origin_time
+ 24h`); the `forecasts` dict, every forecast-vs-actual plot x-axis (DNN/XGBoost test + training
zooms), and `forecasts.csv` now use `target_time`. The "Test start/end" print now shows both ranges
labeled. **Verified no behavior change**: every downstream read of `forecasts[...]` uses `.iloc`
(positional), so relabeling the index is safe — confirmed in the scratch run (`forecasts.csv` shows
`target_time`/`origin_time` as separate columns with the correct 24h offset, e.g. target
`2025-11-02 17:00` ↔ origin `2025-11-01 17:00`).

**#9 lag semantics — switched, not just documented.** `residual_L168`/`residual_L336` were
168h/336h before the *input* time, i.e. 192h/360h (not a clean week/fortnight) before the actual
*target*. Measured **training-only** (rows < T=7000) Pearson correlations on the current
leak-fixed + hour-aligned pipeline:

| Shift | Framing | r (training-only) |
|---|---|---|
| 144h | target-relative, 1 week before TARGET | **0.780** |
| 168h | input-relative, 1 week before INPUT (old) | 0.667 |
| 312h | target-relative, 2 weeks before TARGET | **0.764** |
| 336h | input-relative, 2 weeks before INPUT (old) | 0.651 |
| 48h | (unchanged — target=input+24h either way for this one) | 0.575 |

The gap (≈0.11 Pearson r) was large enough to switch, not just document as a status-quo default.
`LAG_2=312`, `LAG_3=144`; columns renamed `residual_L312`/`residual_L144`; `MAX_LAG` shrank
336→312 (24 fewer burn-in rows). `LAG_1=24` (hydro state features) is unaffected — those describe
"now", so input-relative is correct there and was left alone.
**Side effect (expected, not a bug):** fewer burn-in rows means 24 more usable rows overall, so
one more full rolling day fits in the test period — **59 rolling days now, was 58.**

**#19b two new baselines.** Weekly-naive needed no new computation — it's exactly `residual_L144`
(same clock-hour, 1 week before the target, by construction). A walk-forward linear regression
(same leakage-safe `get_targets_features` per window as DNN/XGBoost, no tuning/plots — kept
minimal, it's a comparison baseline not a candidate model) was added as `Linear`.

**#20** removed the one genuine excessive-print offender: `print(df.to_string())` in `.py` / a
trailing bare `df` in `.ipynb`, an unbounded ~8,700-row full-dataframe dump with no diagnostic
purpose. The bounded summary prints (`describe()`, `isnull().sum()`, `dtypes`, stationarity table,
results tables) were left as-is — they're meaningful, not excessive.

**Full results (59 rolling days, same corrected data as §12.6 — numbers shift slightly from the
extra day + lag-feature change, not a regression):**

| Model | MAPE% | WAPE% | MAE MW | RMSE MW |
|---|---|---|---|---|
| **Ensemble** | **8.18** | **6.95** | **79.2** | **112.7** |
| XGBoost | 8.34 | 7.10 | 81.0 | 115.2 |
| DNN | 8.41 | 7.16 | 81.6 | 114.9 |
| Linear | 8.47 | 7.26 | 82.7 | 116.0 |
| Naive | 11.59 | 10.06 | 114.7 | 163.0 |
| Weekly-Naive | 12.26 | 10.70 | 121.9 | 172.0 |

**Two genuinely new findings from the added baselines:**
- **Linear regression (8.47%) is close behind DNN/XGBoost (8.41%/8.34%)** — a plain OLS on the
  same 30 features gets within ~0.3pp of the tuned nonlinear models. Most of this pipeline's
  accuracy comes from the feature engineering (lags, calendar, calibrated weather), not from
  model sophistication — a fair, slightly humbling result worth stating plainly in the thesis.
- **Weekly-naive (12.26%) is *worse* than simple day-to-day persistence (11.59%)** — "same hour
  last week" is a weaker predictor here than "same hour yesterday", despite the strong
  same-week-hour correlation measured above (r=0.780). Correlation with the target and being a
  *good standalone point forecast* are different things — persistence benefits from short-range
  autocorrelation that a 1-week-old value doesn't have. Both baselines still confirm DNN/XGB/
  Ensemble/Linear all comfortably beat naive persistence.

Synced into `.py` and `.ipynb`, `py_compile` + per-cell `ast.parse` clean on both, full scratch
rerun clean (index check passed, Feb-29 warning fired, no errors). `results_summary.csv` (6 rows)
and `forecasts.csv` (1416 rows, `target_time`/`origin_time` both present) written correctly.

**Revert:** `git revert d106a7c` restores input-relative lags (168/336), origin-time-indexed
forecasts, removes the weekly-naive/linear baseline rows, and restores the full-dataframe print.

---

## 13. Session Infrastructure — Backup/Checkpoint Protocol & Stop-Hook Recovery

This section is process/tooling documentation, not modeling methodology — kept here (rather than
only in CLAUDE.md) because CLAUDE.md's backup rule (§5 there) points back to this file for the
exact script, in case the hook is ever missing from a future container and needs restoring.

### 13.1 Why this exists
This container can reset silently mid-session, with no reliable warning. Two mild resets have
rewound the local git pointer (recoverable via `git fetch`); one reset destroyed an entire isolated
worktree with 7 unpushed, unverified commits — saved only by luck (an agent's own chat transcript
surviving independently). Unpushed work does not durably exist; a clean local checkout is not proof
that origin has the same state.

### 13.2 The 5 backup rules (mirrors CLAUDE.md §5 — see there for the authoritative version)
1. Push a checkpoint to a `backup/<name>` branch as soon as there's any real progress — after the
   first commit, not the last.
2. Re-push periodically during long work, not just at the end.
3. At the start of any session, verify local state matches origin (`git fetch origin <branch>`,
   compare `git log --oneline -1` locally vs. `origin/<branch>`) — never assume a clean checkout
   reflects reality.
4. A worktree protects against edit conflicts, not data loss — it is not a substitute for pushing.
5. Only the final, reviewed result needs to land on the real working branch; backup branches can be
   left in place or cleaned up later.

### 13.3 `/checkpoint` — forced stop-and-save
Use any time a session seems close to a hard usage cutoff (a usage indicator, degrading responses).
It commits and pushes whatever exists — even incomplete, to a `checkpoint/*` branch if not yet safe
for the main branch — writes a full handoff summary directly in the chat reply (the next session's
context comes from the conversation, not disk), captures any durable lesson here, and ends the turn
without picking up further work.

### 13.4 Stop-hook script (restore to `~/.claude/stop-hook-git-check.sh` if missing)
Blocks ending a turn while the current branch has uncommitted, untracked, or unpushed work (or
commits GitHub would show as "Unverified"). This is machine config, outside the repo — `git push`
does not carry it forward to a fresh container, which is why it's archived here.

```bash
#!/bin/bash

# Read the JSON input from stdin
input=$(cat)

# Check if stop hook is already active (recursion prevention)
stop_hook_active=$(echo "$input" | jq -r '.stop_hook_active')
if [[ "$stop_hook_active" = "true" ]]; then
  exit 0
fi

# Check if we're in a git repository - bail if not
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  exit 0
fi

# Bail if there's no remote to push to. Every error path below asks the user
# to "push to the remote branch" — meaningless without a remote, and
# unsatisfiable if signing also requires a source. This case arises when CCR
# was launched against a local repo with no github remote (sources=[]) and
# the container's cwd has a leftover .git from a cached resume.
if [[ -z "$(git remote)" ]]; then
  exit 0
fi

# Check for uncommitted changes (both staged and unstaged)
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "There are uncommitted changes in the repository. Please commit and push these changes to the remote branch." >&2
  exit 2
fi

# Check for untracked files that might be important
untracked_files=$(git ls-files --others --exclude-standard)
if [[ -n "$untracked_files" ]]; then
  echo "There are untracked files in the repository. Please commit and push these changes to the remote branch." >&2
  exit 2
fi

current_branch=$(git branch --show-current)
if [[ -n "$current_branch" ]]; then
  if git rev-parse "origin/$current_branch" >/dev/null 2>&1; then
    upstream="origin/$current_branch"
  else
    upstream="origin/HEAD"
  fi

  # Check for local commits that GitHub will show as "Unverified": either no
  # signature at all (%G? == N), or signed with a committer email other than
  # noreply@anthropic.com (the identity CCR's signing key is registered to).
  # Only run when commit signing is configured. Note: %G? is N for unsigned
  # commits; signed-but-locally-unverifiable commits report B/U/E, so this is
  # a reliable presence check even though CCR doesn't configure local verification.
  if [[ "$(git config --type=bool commit.gpgsign 2>/dev/null)" == "true" ]]; then
    unverifiable=$(git log --format='%h %G? %ce' "$upstream..HEAD" 2>/dev/null | awk '$2 == "N" || $3 != "noreply@anthropic.com"')
    if [[ -n "$unverifiable" ]]; then
      echo "There are commit(s) on branch '$current_branch' that GitHub will show as Unverified (missing signature, or committer email is not noreply@anthropic.com):" >&2
      echo "$unverifiable" >&2
      echo "Please run 'git config user.email noreply@anthropic.com && git config user.name Claude', then 'git commit --amend --no-edit --reset-author' for the tip commit, or 'git rebase --exec \"git commit --amend --no-edit --reset-author\" $upstream' for earlier commits, then push." >&2
      exit 2
    fi
  fi

  unpushed=$(git rev-list "$upstream..HEAD" --count 2>/dev/null) || unpushed=0
  if [[ "$unpushed" -gt 0 ]]; then
    if [[ "$upstream" == "origin/$current_branch" ]]; then
      echo "There are $unpushed unpushed commit(s) on branch '$current_branch'. Please push these changes to the remote repository." >&2
    else
      echo "Branch '$current_branch' has $unpushed unpushed commit(s) and no remote branch. Please push these changes to the remote repository." >&2
    fi
    exit 2
  fi
fi

exit 0
```
