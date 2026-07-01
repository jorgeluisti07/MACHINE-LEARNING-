# Residual Demand Forecasting — ETESA Panama 2025

Day-ahead (h+24) hourly **residual demand** forecasting for the Panamanian grid, comparing a
**Deep Neural Network (DNN)** and **XGBoost**, evaluated with expanding-window (walk-forward)
rolling validation against a naive persistence benchmark.

Residual demand is defined as:

```
demanda_residual = demanda_mw − solar_mw − eolica_mw
```

i.e. the portion of demand that hydro + thermal must supply once variable renewables are removed.

## Repository contents

| File | Description |
|---|---|
| `MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb` | Main notebook (intended entry point) |
| `MACHINE_LEARNING_RESIDUAL_DEMAND.py` | Script mirror of the notebook |
| `MACHINE LEARNING RD CONTEXT.md` | Methodology reference document |
| `DEM2025.csv` | ETESA hourly demand (input) |
| `solar_eolica_hidro_horario_2025.csv` | Real ETESA generation (input) |

## How to Run

### Required input files

Place these next to the notebook, with these exact names:

| File | Format | Columns |
|---|---|---|
| `DEM2025.csv` | wide CSV | first column = date (`Unnamed: 0`), then `H1`…`H24` = hourly demand in MW |
| `solar_eolica_hidro_horario_2025.csv` | long CSV | `datetime`, `solar_mw_real`, `eolica_mw_real`, `hidro_mw_real` |

The meteorological features (`irradiance_direct`, `irradiance_diffuse`, `temperature`,
`wind_speed`) are **not** in the CSVs — they are pulled live from the
[Renewables.ninja](https://www.renewables.ninja) API (MERRA-2 dataset). Running the pipeline
therefore requires:

- internet access and a valid Renewables.ninja API token (set in the *data download* cell), and
- a location for geocoding — the notebook prompts with `input()`; enter `Penonomé, Coclé, Panama`.

### Environment (Python 3.11)

```bash
pip install pandas numpy matplotlib requests geopy statsmodels scipy seaborn \
            scikit-learn xgboost tensorflow tqdm
```

### Run

```bash
jupyter notebook MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb   # then run cells top to bottom
```

The `.py` mirrors the notebook, but it calls `input()` and the live API on import, so the notebook
is the intended entry point.

### Expected output

- Rolling day-ahead forecast plots (actual vs forecast, 7-day zoom, absolute error) for DNN and XGBoost.
- XGBoost feature-importance chart (gain).
- Naive persistence benchmark and per-model selection (must beat naive; overfit gap < 10 pp).
- A final **results table** reporting, per model (DNN, XGBoost, Naive), both relative error
  (**MAPE %**, **WAPE %**) and absolute error in MW (**MAE**, **RMSE**), so the operational
  magnitude of the error is explicit alongside the ratios.

## Limitations

- **Perfect-foresight weather (h+24).** The four `*_h24` meteorological features use
  `shift(-24)` — the *true* future MERRA-2 reanalysis value, not an operational forecast (the full
  year is downloaded in one request). Reported metrics therefore assume a **perfect 24-hour weather
  forecast** and are an optimistic upper bound; a real deployment would feed NWP (e.g. GFS/ECMWF)
  whose forecast error would raise MAPE/WAPE.
- **Leakage-safe hydro baseline.** The seasonal-diurnal hydro features (`hidro_typical_h24`,
  `hidro_anomaly_L24`) are rebuilt inside each rolling window from **training-only data**
  (`rebuild_hidro_profile_features`), so no window sees a `(month, hour)` hydro average that
  includes months past its own cutoff.
