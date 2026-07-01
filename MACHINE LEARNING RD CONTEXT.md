# ETESA TFM — Notebook Reference Document

**Version:** v26 | DNN + XGBoost | Lagged Approach | Leakage-fixed
**Companion files:** `MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb`, `MACHINE_LEARNING_RESIDUAL_DEMAND.py`, `README.md` (How to Run)

---

## 0. TL;DR — read this first

- **Problem:** day-ahead (h+24) residual demand forecasting for Panama, DNN vs XGBoost, expanding-window rolling validation.
- **Inputs are CSV, not Excel/Parquet.** `DEM2025.csv` (wide, `H1`..`H24`) and `solar_eolica_hidro_horario_2025.csv` (long). See §1.
- **Two known limitations baked into the methodology** (not bugs, but must be disclosed in the thesis): perfect-foresight h+24 weather, and a previously-leaky hydro baseline that has since been fixed. See §7.
- **§9 results (7.08% / 7.35% MAPE) predate the hydro-leakage fix** — they need to be regenerated after re-running the notebook. Treat them as historical/pre-fix until refreshed.
- Full history of what went wrong and what was fixed is in **§8 Fixes Log** — check it before "discovering" an issue that's already been handled.

---

## 1. Data Sources

| Source | Description | Format |
|---|---|---|
| Renewables.ninja API | Solar/wind resource data for Coclé, Penonomé, Panama (lat=8.52, lon=-80.36). MERRA-2 reanalysis. 859 MW solar / 336 MW wind. Full year 2025. | REST API → JSON |
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
| Training end (approx.) | ~2025-10-31 | Wet season in Panama (Jan–Oct) |
| Test start | `datetime_index.iloc[7000]` | ≈ 2025-11-02 |
| Test end | ~2025-12-30 | Dry season onset (Nov–Dec) |
| Test rolling days | ~58 days | `(len(df) − 7000) // 24` |
| Test hours | ~1392 hours | `n_test_days × 24` |

**Seasonal split note:** training (Jan–Oct) covers the wet season (peak hydro); test (Nov–Dec) covers the start of the dry season (declining hydro, rising thermal). This distributional shift is one reason the XGBoost overfit gap (2.75 pp, pre-fix) was larger than the DNN's (0.79 pp, pre-fix) — see §9 caveat.

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

**Impact on §9 numbers:** this fix changes the training signal for early rolling windows, so the historical 7.08%/7.35% MAPE figures **must be regenerated** — they predate the fix.

---

## 8. Fixes Log (mistakes found → what was fixed)

| # | Issue found | Where | Fix | Status |
|---|---|---|---|---|
| 1 | Doc said inputs were `DEM2025.xlsx` (Excel) + `solar_eolica_hidro_horario_2025.parquet` (Parquet); code actually reads CSV via `pd.read_csv` for both | This doc §1 (was §3), in-code comments ("Load ... from Excel/Parquet") | Corrected doc + code comments to CSV; added a README with exact filenames | ✅ Fixed, verified (only `pd.read_csv` calls in `.py`/`.ipynb`, zero `read_excel`/`read_parquet` anywhere) |
| 2 | h+24 weather features are true future values, not a forecast, but this wasn't surfaced anywhere a reader would see without opening the code | Feature construction (`*_h24 = ....shift(-HORIZON)`) | Added an explicit "LIMITATION: PERFECT-FORESIGHT WEATHER" note at the code site + this doc (§7.1) + README | ✅ Documented (not a code change — this is a deliberate design choice) |
| 3 | Hydro seasonal baseline (`hidro_typical_h24`, `hidro_anomaly_L24`) computed once on the full year before the rolling loop — look-ahead leak into early windows | `get_targets_features` / hydro feature block | Added `rebuild_hidro_profile_features(df_in, T)`, refit per-window on training-only rows; called first thing inside `get_targets_features` | ✅ Fixed, unit-tested (3/3 tests pass: no leakage, no NaN, correct fallback), reviewed by an adversarial subagent pass — no findings |
| 4 | Results only reported percentage error (MAPE/WAPE); no absolute-MW magnitude | End-of-notebook results section | Added `results_summary` table: MAPE %, WAPE %, MAE MW, RMSE MW per model (DNN/XGBoost/Naive) | ✅ Added, dry-run verified |
| 5 | Excessive line-by-line "what it does" comments in core pipeline cells (imports, feature extraction) | Throughout `.py`/`.ipynb` | Trimmed to docstrings, section headers, and "why" comments only | ✅ Done |
| 6 | Hardcoded Renewables.ninja API token in source | Data-download cell | **Not fixed** — flagged to the user, out of scope of the requested changes | ⚠️ Open — rotate the token and move to an env var before any public sharing |

**How this file stays useful:** when something in the notebook/script turns out wrong or gets fixed, add a row here rather than just fixing it silently — that's what makes this doc worth reading before starting new work on the pipeline.

---

## 9. Results (⚠️ PRE-LEAKAGE-FIX — regenerate before citing in the thesis)

| Model | Test MAPE | Test WAPE | Test sMAPE | R² | vs Naive | Overfit |
|---|---|---|---|---|---|---|
| **DNN** | **7.08%** | **0.0616** | **6.66%** | **0.7765** | **38.8%** | 0.79 pp |
| XGBoost | 7.35% | 0.0621 | 6.74% | 0.7539 | 36.4% | 2.75 pp |
| Naive | 11.57% | 0.1017 | 11.24% | 0.3563 | — | — |

**Test period:** ~2025-11-02 → ~2025-12-30 (58 rolling days). **Training cutoff:** T = 7000 hours.

**These numbers were produced before the §7.2/§8-#3 hydro-leakage fix.** Re-run the notebook (requires live Renewables.ninja API access + interactive geocoding input, see README "How to Run") and replace this table — including the new MAE/RMSE columns from `results_summary` — once refreshed.
