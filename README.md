# Residual Demand Forecasting — ETESA Panama 2025

Day-ahead (h+24) hourly **residual demand** forecasting for the Panamanian grid, comparing a
**Deep Neural Network (DNN)** and **XGBoost** (plus their **ensemble**, the best model), evaluated
with expanding-window (walk-forward) rolling validation against a naive persistence benchmark.

Residual demand is defined as:

```
demanda_residual = demanda_mw − solar_mw − eolica_mw
```

i.e. the portion of demand that hydro + thermal must supply once variable renewables are removed.

## Repository contents

| File | Description |
|---|---|
| `MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb` | Main notebook (intended entry point) |
| `MACHINE_LEARNING_RESIDUAL_DEMAND.py` | Script mirror of the notebook (runs non-interactively) |
| `MACHINE LEARNING RD CONTEXT.md` | Methodology reference: assumptions, fixes log, results, optimization roadmap |
| `README.md` | This file |
| `requirements.txt` | Python dependencies |
| `DEM2025.csv` | ETESA hourly demand (input) |
| `solar_eolica_hidro_horario_2025.csv` | Real ETESA generation (input) |
| `renewables_ninja_2025.csv` | Weather cache, created on first run (not committed by default — commit it if you want fully offline reproducibility for others) |

## How to Run

### 1. Required input files

Place these next to the notebook, with these exact names:

| File | Format | Columns |
|---|---|---|
| `DEM2025.csv` | wide CSV | first column = date (`Unnamed: 0`), then `H1`…`H24` = hourly demand in MW |
| `solar_eolica_hidro_horario_2025.csv` | long CSV | `datetime`, `solar_mw_real`, `eolica_mw_real`, `hidro_mw_real` |

### 2. Weather data (Renewables.ninja API, cached)

The meteorological features (`irradiance_direct`, `irradiance_diffuse`, `temperature`,
`wind_speed`) come from the [Renewables.ninja](https://www.renewables.ninja) API (MERRA-2
reanalysis) and are **cached locally** for reproducibility:

- **First run:** set your API token in the environment —
  ```bash
  export RENEWABLES_NINJA_TOKEN=<your token>   # from renewables.ninja/profile
  ```
  The response is saved to `renewables_ninja_2025.csv`.
- **Later runs:** the cache is loaded automatically — fully offline, no token needed.
  Delete the file to force a re-download.

Site coordinates (Penonomé, Coclé, Panama: lat 8.52, lon −80.36) are fixed constants in the
code. There is **no interactive input** anywhere — both the notebook and the `.py` run
top-to-bottom unattended.

### 3. Environment (Python 3.11)

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
jupyter notebook MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb   # run cells top to bottom
# or, non-interactively:
python MACHINE_LEARNING_RESIDUAL_DEMAND.py
```

### 5. Expected output

- Rolling day-ahead forecast plots (actual vs forecast, 7-day zoom, absolute error) for DNN and XGBoost.
- XGBoost feature-importance chart (gain).
- Naive persistence benchmark and per-model selection (must beat naive; overfit gap < 10 pp).
- A final **results table** reporting, per model (DNN, XGBoost, **Ensemble**, Naive), both relative
  error (**MAPE %**, **WAPE %**) and absolute error in MW (**MAE**, **RMSE**).

## Assumptions & limitations

- **Perfect-foresight weather (h+24).** The four `*_h24` meteorological features use
  `shift(-24)` — the *true* future MERRA-2 reanalysis value, not an operational forecast (the full
  year is downloaded in one request). Reported metrics therefore assume a **perfect 24-hour weather
  forecast** and are an optimistic upper bound; a real deployment would feed NWP (e.g. GFS/ECMWF)
  whose forecast error would raise MAPE/WAPE.
- **Leakage-safe hydro baseline.** The seasonal-diurnal hydro features (`hidro_typical_h24`,
  `hidro_anomaly_L24`) are rebuilt inside each rolling window from **training-only data**
  (`rebuild_hidro_profile_features`), so no window sees a `(month, hour)` hydro average that
  includes months past its own cutoff.
- **Seasonal train/test shift.** Training covers Jan–Oct (wet season, peak hydro); the test window
  Nov–Dec is the dry-season onset. Part of the test error reflects this distribution shift, which
  is realistic for deployment but worth keeping in mind when reading the metrics.
- **Run-to-run variability.** Seeds are fixed (`set_random_seeds(42)` per rolling iteration), but
  TensorFlow op-level nondeterminism can still move DNN metrics by ≈ ±0.05 pp MAPE between runs.
  The data side is fully deterministic once the weather cache exists.

See `MACHINE LEARNING RD CONTEXT.md` for the full methodology reference, results history,
fixes log, and optimization roadmap.
