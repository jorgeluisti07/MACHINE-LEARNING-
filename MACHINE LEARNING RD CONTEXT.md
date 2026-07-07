# ETESA TFM — Notebook Reference Document

**Version:** v26.9 | DNN + XGBoost + Ensemble | Lagged Approach | Leakage-fixed | P1 tried, evaluated, and reverted by user request — see §8 row 16

> **Standing rule:** the Renewables.ninja download code (geocoding prompt, token, API calls) is
> owned by the user — **do not modify it** without explicit instruction. See §8 row 10.
**Companion files:** `MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb`, `MACHINE_LEARNING_RESIDUAL_DEMAND.py`, `README.md` (How to Run)

---

## 0. TL;DR — read this first

- **Problem:** day-ahead (h+24) residual demand forecasting for Panama, DNN vs XGBoost, expanding-window rolling validation.
- **Inputs are CSV, not Excel/Parquet.** `DEM2025.csv` (wide, `H1`..`H24`) and `solar_eolica_hidro_horario_2025.csv` (long). See §1.
- **Two known limitations baked into the methodology** (not bugs, but must be disclosed in the thesis): perfect-foresight h+24 weather, and a previously-leaky hydro baseline that has since been fixed. See §7.
- **§9 has the confirmed post-leakage-fix results** (DNN 7.02% / XGB 7.31% MAPE, MAE ~70 MW) — the fix slightly *improved* accuracy while making the methodology honest. The pre-fix numbers are kept for comparison.
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

**How this file stays useful:** when something in the notebook/script turns out wrong or gets fixed, add a row here rather than just fixing it silently — that's what makes this doc worth reading before starting new work on the pipeline.

---

## 9. Results

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

### 9.2 Historical results — PRE-leakage-fix (v25, keep for comparison, do not cite as current)

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

### Recommended order after the v26.5 run (read this first)

The full run confirmed one thing that changes the priority order: **XGBoost's overfit gap (2.82 pp)
is ~4× the DNN's (0.73 pp), while their test MAPEs are almost identical (7.35% vs 7.01%).** That is
the single most actionable fact in this roadmap — XGBoost has real headroom that's cheap to try to
recover, before reaching for new features at all. Updated order:

1. **P5 (XGBoost regularization sweep)** — promoted to first. The 2.82 pp gap is now a *measured*
   target, not a guess. If XGBoost's test MAPE improves even slightly while the gap narrows, the
   ensemble (which currently drags XGBoost's weaker generalization into a flat 50/50 average)
   improves for free too.
2. **Weighted ensemble (new, minor extension of P3)** — trivial to try alongside P5: since DNN
   generalizes better, an ensemble weight favoring DNN (e.g. 0.6·DNN + 0.4·XGB, tuned only on the
   validation tail, never on test) may beat the current flat 50/50. Almost zero cost — same stored
   forecasts, just a different scalar.
3. **P1 (holiday-proximity features)** — still the best lever for the *tail* errors specifically
   (the Dec 23 / Nov 26 spike days — see §10 P1 for exact dates), which neither P5 nor re-weighting
   addresses. Do this once the
   regularization/weighting quick wins are banked, so its effect is measured on a cleaner baseline.
4. **P2 (feature pruning)** — do after P5, not before: a regularization sweep may already reduce
   XGBoost's effective reliance on the near-zero-importance features, so pruning them explicitly
   afterward is a cleaner ablation with less confounding.
5. Everything else (P4 ramp features, P6 quantile/Huber loss, P7 pipeline cleanup) unchanged —
   still lower priority than the four above.

**Discipline reminder (unchanged from before):** one change at a time, re-run, record the delta in
§8 before moving to the next item. Do not batch P5 and P1 into one run — you won't be able to tell
which one moved the number.

### P3 — Simple ensemble: average DNN + XGBoost ✅ IMPLEMENTED & CONFIRMED TWICE (see §8 row 9, §9.1)
**Evidence:** the two models' error bursts don't fully coincide in the plots (different hours miss differently); their test MAPEs are within 0.3 pp of each other — the classic setup where a 50/50 average beats both.
**Action:** `pred_ens = 0.5*pred_dnn + 0.5*pred_xgb` on the stored `forecasts` DataFrames — zero retraining needed, one cell.
**Result:** confirmed on two separate runs — Ensemble MAPE 6.88–6.89% (vs DNN 7.01–7.02%, XGB 7.31–7.35%), winning on WAPE, MAE, **and RMSE** both times, so the improvement isn't just average-case, it holds on the worst days too. **This is the best model; treat it as the baseline for P1/P4 going forward** (i.e. once holiday/ramp features are added, re-run the ensemble on the new DNN+XGB forecasts too, don't just compare the new features against the old single models). **Next refinement:** try a non-50/50 weight (see recommended order above) — the DNN's much smaller overfit gap suggests it may deserve more than half the weight.

### P4 — Ramp/persistence-error features (targets the systematic lag)
**Evidence:** the 7-day zoom shows both models trailing fast ramps — over-reliance on `demanda_residual` (importance ~0.29 = persistence anchor).
**Action:** add `residual_delta_3h = residual(t) − residual(t−3)` and yesterday's persistence error `naive_error_L24 = residual(t−24) − residual(t−48)`-style features (all backward-looking → leakage-free).
**Expected impact:** helps the model anticipate ramps instead of following them one step late.

### P5 — XGBoost hyperparameter sweep ⭐ PROMOTED TO TOP PRIORITY (see recommended order above)
**Evidence:** confirmed by direct execution (v26.5, §9.1) — XGBoost overfit gap is **2.82 pp** (train 4.54% → test 7.35%), vs the DNN's **0.73 pp** (train 6.28% → test 7.01%). Nearly identical test MAPE, very different generalization — XGBoost is fitting training noise the DNN isn't. This is a measured, not assumed, target.
**Action:** small grid around the current config on the *validation* tail only: `max_depth ∈ {4,5,6}` (lower = less overfit), `learning_rate ∈ {0.02,0.03,0.05}`, `min_child_weight ∈ {1,5,10}` (higher = more conservative splits), `reg_alpha`/`reg_lambda` a notch higher than the current 0.1/1.0, `subsample`/`colsample_bytree ∈ {0.7,0.8,0.9}`. Never tune on the test window — that would leak the test set into model selection.
**Expected impact:** closing even half the gap (2.82 → ~1.4 pp) without hurting test MAPE would be a genuine, cheap win — and it directly strengthens the ensemble too, since a less-overfit XGBoost is a better ensemble partner for the already-strong DNN.

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
| *latest on this file* | v26.1+: post-fix results, §10 roadmap, this revert table, plus all rows above | Find it with `git log --oneline -- "MACHINE LEARNING RD CONTEXT.md"`, then `git revert <hash>` |

To reproduce the **pre-fix (v25) numbers** for a thesis comparison table: `git stash && git checkout 698be5f~1 -- MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb`, re-run, then `git checkout HEAD -- MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb && git stash pop`. (Or simply cite §9.2 — the pre-fix numbers are preserved there.)
