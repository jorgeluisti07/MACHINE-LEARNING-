# ETESA TFM — Notebook Reference Document

## Version: v25 | DNN + XGBoost | Lagged Approach

\---

## 

\---

## 2\. FORECASTING PROBLEM DEFINITION

|Parameter|Value|
|-|-|
|**Target variable**|`demanda\\\\\\\_residual\\\\\\\_h24 = demanda\\\\\\\_residual(t+24)`|
|**Target definition**|`demanda\\\\\\\_residual = demanda\\\\\\\_mw − solar\\\\\\\_mw − eolica\\\\\\\_mw`|
|**Forecast horizon**|h = 24 hours (day-ahead)|
|**Data frequency**|Hourly (1-hour resolution)|
|**Forecast type**|Deterministic point forecast|
|**Validation method**|Rolling walk-forward (expanding window), one day per iteration|
|**Output**|One 24-hour block per iteration (h+1 through h+24)|

\---

## 3\. DATA SOURCES

|Source|Description|Format|
|-|-|-|
|Renewables.ninja API|Solar and wind resource data for Coclé, Penonomé, Panama (lat=8.52, lon=-80.36). MERRA-2 reanalysis. 859 MW solar / 336 MW wind. Full year 2025.|REST API → JSON|
|`solar\\\\\\\_eolica\\\\\\\_hidro\\\\\\\_horario\\\\\\\_2025.parquet`|Real ETESA metered generation: `solar\\\\\\\_mw\\\\\\\_real`, `eolica\\\\\\\_mw\\\\\\\_real`, `hidro\\\\\\\_mw\\\\\\\_real`. Hourly, 2025.|Parquet|
|`DEM2025.xlsx`|Real ETESA hourly electricity demand 2025. Wide format (date × 24 hours).|Excel|

**Calibration:** `ratio\\\\\\\_calib = solar\\\\\\\_mw\\\\\\\_real / solar\\\\\\\_ninja` applied to `irradiance\\\\\\\_direct` and
`irradiance\\\\\\\_diffuse` to align the MERRA-2 reanalysis signal with observed Panamanian conditions.

\---

## 4\. TRAINING AND TEST DATES

|Split|Value|Description|
|-|-|-|
|**Burn-in**|First 336 rows dropped|MAX\_LAG = 336 h (2 weeks) — rows before this have NaN lag features|
|**Training start**|\~2025-01-15 01:00|After MAX\_LAG burn-in|
|**Training cutoff T**|Row index 7000|≈ 7000 hourly records = \~292 days|
|**Training end (approx.)**|\~2025-10-31|Wet season in Panama (Jan–Oct)|
|**Test start**|`datetime\\\\\\\_index.iloc\\\\\\\[7000]`|≈ 2025-11-02|
|**Test end**|\~2025-12-30|Dry season onset (Nov–Dec)|
|**Test rolling days**|\~58 days|`(len(df) − 7000) // 24`|
|**Test hours**|\~1392 hours|`n\\\\\\\_test\\\\\\\_days × 24`|

**Note on seasonal split:** The train period (Jan–Oct) covers the wet season (peak hydro production).
The test period (Nov–Dec) covers the beginning of the dry season (declining hydro, increasing thermal).
This distributional shift is one reason why the XGBoost overfit gap (2.75 pp) is larger than the DNN (0.79 pp).

\---

## 5\. FEATURE SET — 30 FLAT FEATURES + 1 TARGET

### 5.1 Current meteorology (7 features)

|Feature|Description|
|-|-|
|`irradiance\\\\\\\_direct`|Direct normal irradiance at t (calibrated MERRA-2), W/m²|
|`irradiance\\\\\\\_diffuse`|Diffuse horizontal irradiance at t (calibrated), W/m²|
|`temperature`|Air temperature at t (MERRA-2), °C|
|`solar\\\\\\\_mw`|Real ETESA solar generation at t, MW|
|`wind\\\\\\\_speed`|Wind speed at hub height at t (MERRA-2), m/s|
|`eolica\\\\\\\_mw`|Real ETESA wind generation at t, MW|
|`hidro\\\\\\\_mw`|Real ETESA hydroelectric generation at t, MW|

### 5.2 NWP horizon covariates at t+24 (4 features) — *Most impactful group*

|Feature|Description|
|-|-|
|`irradiance\\\\\\\_direct\\\\\\\_h24`|Direct irradiance 24 hours ahead (perfect-foresight NWP proxy)|
|`irradiance\\\\\\\_diffuse\\\\\\\_h24`|Diffuse irradiance 24 hours ahead|
|`temperature\\\\\\\_h24`|Temperature 24 hours ahead|
|`wind\\\\\\\_speed\\\\\\\_h24`|Wind speed 24 hours ahead|

> \\\\\\\*\\\\\\\*Key finding (v22):\\\\\\\*\\\\\\\* Adding these 4 NWP horizon features reduced DNN MAPE from \\\\\\\~9.3 % to \\\\\\\*\\\\\\\*7.08 %\\\\\\\*\\\\\\\* (−2.35 pp).
> `irradiance\\\\\\\_direct\\\\\\\_h24` is the \\\\\\\*\\\\\\\*3rd most important feature\\\\\\\*\\\\\\\* in XGBoost (gain = 0.080).
> The NWP input made the temporal-persistence advantage of LSTM models redundant.

### 5.3 Calendar features (11 features)

|Feature|Description|
|-|-|
|`month`|Calendar month (1–12), integer|
|`hour`|Hour of day (0–23), integer|
|`hour\\\\\\\_sin`|sin(2π × hour / 24) — cyclic hour encoding|
|`hour\\\\\\\_cos`|cos(2π × hour / 24) — cyclic hour encoding|
|`month\\\\\\\_sin`|sin(2π × month / 12) — cyclic month encoding|
|`month\\\\\\\_cos`|cos(2π × month / 12) — cyclic month encoding|
|`is\\\\\\\_weekday`|1 if Mon–Fri, 0 if Sat–Sun|
|`hour\\\\\\\_weekday`|hour × is\_weekday — interaction: midday peak on working days only|
|`is\\\\\\\_holiday`|1 on Panama national holidays (14 days in 2025), 0 otherwise|
|`dow\\\\\\\_sin`|sin(2π × dayofweek / 7) — cyclic day-of-week encoding|
|`dow\\\\\\\_cos`|cos(2π × dayofweek / 7) — cyclic day-of-week encoding|

### 5.4 Lagged target features — Lagged Approach (4 features)

|Feature|Lag|Pearson r|Rationale|
|-|-|-|-|
|`demanda\\\\\\\_residual`|t (current)|—|Current-hour residual demand as a feature|
|`residual\\\\\\\_L48`|48 h|0.511|Mid-week demand build-up pattern|
|`residual\\\\\\\_L336`|336 h = 14 days|0.591|Biweekly hydroelectric dispatch cycle in Panama|
|`residual\\\\\\\_L168`|168 h = 7 days|0.600|Same hour, same day last week — strongest lag|

> \\\\\\\*\\\\\\\*L24 was dropped:\\\\\\\*\\\\\\\* Pearson r(L24, target) ≈ 0 — nearly collinear with the current `demanda\\\\\\\_residual` feature.

### 5.5 Hydro dispatch features (4 features)

|Feature|Description|
|-|-|
|`hidro\\\\\\\_fraction\\\\\\\_L24`|Yesterday's hydro share of residual demand. Operational state signal.|
|`hidro\\\\\\\_delta\\\\\\\_L24`|Day-over-day trend in hydro dispatch (MW). Reservoir level direction.|
|`hidro\\\\\\\_typical\\\\\\\_h24`|Seasonal-diurnal typical hydro at the forecast horizon (groupby month×hour mean, shifted -24).|
|`hidro\\\\\\\_anomaly\\\\\\\_L24`|Yesterday's deviation of actual hydro from its seasonal typical. Drought/surplus signal.|

### 5.6 Target (1)

|Feature|Description|
|-|-|
|`demanda\\\\\\\_residual\\\\\\\_h24`|Residual demand 24 hours ahead. The value to be predicted.|

\---

## 6\. MODEL ARCHITECTURES

### 6.1 Deep Neural Network (DNN)

```
Sequential(\\\\\\\[
    Dense(256, activation='relu'),   # 1st hidden layer
    Dense(256, activation='relu'),   # 2nd hidden layer
    Dense(128, activation='relu'),   # 3rd hidden layer
    Dense(1)                         # output (linear, single forecast in MW)
])
optimizer = Adam(learning\\\\\\\_rate=0.001)
loss      = mean\\\\\\\_squared\\\\\\\_error
epochs    = 100 (max), EarlyStopping patience=5 on val\\\\\\\_loss
batch\\\\\\\_size      = 256
validation\\\\\\\_split = 0.10
restore\\\\\\\_best\\\\\\\_weights = True
```

* **Scaling:** Both X and Y scaled with `StandardScaler` (fit on training data only).
* **Seed:** `set\\\\\\\_random\\\\\\\_seeds(42)` called at the start of each rolling iteration.

### 6.2 XGBoost

```
XGBRegressor(
    n\\\\\\\_estimators=500,
    learning\\\\\\\_rate=0.03,
    max\\\\\\\_depth=5,
    subsample=0.8,
    colsample\\\\\\\_bytree=0.8,
    reg\\\\\\\_alpha=0.1,
    reg\\\\\\\_lambda=1.0,
    early\\\\\\\_stopping\\\\\\\_rounds=20,   # MUST be in constructor (XGBoost ≥ 2.0)
    random\\\\\\\_state=42,
    verbosity=0,
    n\\\\\\\_jobs=-1
)
eval\\\\\\\_set = last 10 % of training window (temporal order preserved)
```

* **Scaling:** None (tree models are scale-invariant).
* **Validation:** last `max(24, round(len(X\\\\\\\_train) × 0.10))` rows of training window.

\---

## 7\. METRICS SUITE

|Metric|Formula|Notes|
|-|-|-|
|MAPE|mean(\|actual − pred\| / \|actual\|)|Standard; sensitive when residual → 0 at solar peak hours|
|WAPE|Σ\|actual − pred\| / Σ\|actual\||Aggregate denominator; consistently \~1 pp below MAPE here|
|sMAPE|mean(2\|e\| / (\|a\| + \|p\|))|Symmetric; bounded 0–200 %|
|R²|1 − SS\_res / SS\_tot|Fraction of variance explained|

WAPE is preferred for residual demand because MAPE can disrupt during midday hours when
`demanda\\\\\\\_residual → 0` under high solar penetration.

\---

## 8\. NAIVE BENCHMARK

```
Naive prediction at time t: predict residual(t) as the forecast for residual(t+24)
(persistence model: tomorrow's residual = today's residual)
```

Every model must beat the naive benchmark (lower test MAPE) to be considered useful.
The snn\_forec model selection criteria are:

1. `test\\\\\\\_MAPE < naive\\\\\\\_MAPE`
2. `test\\\\\\\_MAPE − train\\\\\\\_MAPE < 0.10` (overfit gap < 10 percentage points)

\---

## 9\. BEST RESULTS

|Model|Test MAPE|Test WAPE|Test sMAPE|R²|vs Naive|Overfit|
|-|-|-|-|-|-|-|
|**DNN**|**7.08 %**|**0.0616**|**6.66 %**|**0.7765**|**38.8 %**|0.79 pp|
|XGBoost|7.35 %|0.0621|6.74 %|0.7539|36.4 %|2.75 pp|
|Naive|11.57 %|0.1017|11.24 %|0.3563|—|—|

**Test period:** \~2025-11-02 → \~2025-12-30 (58 rolling days)
**Training cutoff:** T = 7000 hours

\---

## 

