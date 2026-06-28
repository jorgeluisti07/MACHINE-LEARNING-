#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual Demand Forecasting — ETESA Panama 2025
Models: Deep Neural Network (DNN) + XGBoost | Horizon h=24 | Lagged Approach
Converted from MACHINE_LEARNING_RESIDUAL_DEMAND.ipynb
"""


import warnings
warnings.filterwarnings('ignore')
# Suppress all warnings so the notebook output stays clean.

# ────────────────────────────────────────────────────────────────────────────
# Horizon **h = 24** (day-ahead, hourly resolution).
# Two models: **Deep Neural Network (DNN)** and **XGBoost**.
# Feature approach: **Lagged values** (`use_polynomials = False`),
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Data Preprocessing
# Data Preprocessing: clean / organise the input data into the format required by the models. <br>
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# setting the constants
# ────────────────────────────────────────────────────────────────────────────

use_polynomials = False
# False means we use the Lagged Values approach
# True would mean Polynomials Transform

HORIZON = 24
# We forecast 24 hours ahead (day-ahead, h = 24).
# This is the standard operational horizon for electricity market clearing in Panama.

LAG_1 = 24
# 24-hour lag: same hour yesterday. Used in hydro features.

LAG_2 = 336
# 336-hour lag (= 14 days): same hour two weeks ago.
# Pearson correlation with target: r = 0.591 (biweekly hydro dispatch cycle).

LAG_3 = 168
# 168-hour lag (= 7 days): same hour last week.
# Pearson correlation with target: r = 0.600 — the strongest single lag.

MAX_LAG = LAG_2
# We use LAG_2 = 336 hours as the burn-in period.
# The first MAX_LAG rows are dropped after feature construction to avoid NaN values.

# ────────────────────────────────────────────────────────────────────────────
# LAG_2 (r = 0.591) and LAG_3 (r = 0.600) were chosen over LAG_1 (r ≈ 0) by Pearson correlation analysis.
# LAG_1 (24 h) was dropped because it is nearly collinear with the current value of `demanda_residual`,
# adding no information. The 48-hour lag (r = 0.511) is included as `residual_L48` in the feature list below.
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# data download
# ────────────────────────────────────────────────────────────────────────────

import requests, json as _json
# requests: used to call the Renewables.ninja REST API and download solar/wind resource data.
# json (aliased _json): parse the API response from JSON format into a Python dictionary.

import pandas as pd
# pandas: the main library for tabular data manipulation throughout the notebook.

import numpy as np
# numpy: numerical array operations, used for math (sin, cos, exp) and metric computations.

import matplotlib.pyplot as plt
# matplotlib: all plots (time-series, error bars, histograms, feature importance bar charts).

pd.set_option("display.max_rows",    None)
pd.set_option("display.max_columns", None)
# Show the full DataFrame when we call df in a cell, without truncation.
# Useful during development to inspect all 31 columns.

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent='etesa_tfm')
# Create a geolocator object using the OpenStreetMap Nominatim service.
# user_agent is a required label so the service knows who is calling it.

place = input('Enter location (e.g. Penonomé, Coclé, Panama): ')
# Ask the user to type the name of the location for the API query.

information = geolocator.geocode(place)
# Convert the place name into geographic coordinates (geocoding).

lat = information[1][0]
lon = information[1][1]
# Extract latitude and longitude from the geocoder result tuple.

print(lat, lon)
# Display the coordinates to verify they correspond to the intended location.

token    = 'c4672deddb9f6acfa94e9cdbb34f6414fd51f255'
# Personal access token for the Renewables.ninja API — replace with your own if expired.

api_base   = 'https://www.renewables.ninja/api/'
url_solar  = api_base + 'data/pv'
url_wind   = api_base + 'data/wind'
# Two endpoint URLs: one for photovoltaic (solar) and one for wind power output.

args_solar = {
    'lat': lat, 'lon': lon,
    'date_from': '2025-01-01', 'date_to': '2025-12-31',
    # Full calendar year 2025 — matches the ETESA real-generation data period.
    'dataset':  'merra2',
    # MERRA-2 is NASA's reanalysis product: global, hourly, 0.5° × 0.625° grid.
    'capacity':     859000,   # installed solar capacity in kW (859 MW in Panama 2025)
    'system_loss':  0.1,      # 10 % system losses (wiring, inverter, soiling)
    'tracking':     1,        # single-axis tracking (tilts panel to follow the sun)
    'tilt':         5,        # 5° fixed tilt (small tilt helps rain cleaning)
    'azim':         180,      # south-facing azimuth (optimal for Northern Hemisphere)
    'format':       'json',
    'local_time':   'true',   # return timestamps in Panama local time (UTC-5)
    'raw':          'true',   # include raw meteorological variables (irradiance, temp)
}

args_wind = {
    'lat': lat, 'lon': lon,
    'date_from': '2025-01-01', 'date_to': '2025-12-31',
    'capacity':  336000,                      # installed wind capacity in kW (336 MW)
    'height':    150,                          # hub height in metres
    'turbine':   'Vestas V80 2000',            # turbine model for power curve lookup
    'format':    'json',
    'local_time': 'true',
    'raw':        'true',
}
# The raw flag returns irradiance_direct, irradiance_diffuse, temperature, wind_speed —
# which become our meteorological features and, shifted by -24, the NWP horizon covariates.

s = requests.session()
s.headers = {'Authorization': 'Token ' + token}
# Create an HTTP session and attach the API token to every request header.

# ── Solar ────────────────────────────────────────────────────────────────────────
r = s.get(url_solar, params=args_solar)
print('Solar status:', r.status_code)
# Status 200 means the request succeeded. Any other code indicates an error.

parsed_solar = _json.loads(r.text)
# Parse the raw JSON text into a Python dictionary.

data = pd.read_json(_json.dumps(parsed_solar['data']), orient='index')
# The 'data' key holds the time-indexed results. orient='index' tells pandas that
# the outer dictionary keys are the row indices (timestamps).

data['electricity'] = data['electricity'] / 1000
# Convert from kW (the API default) to MW (our working unit for ETESA data).

data['local_time'] = pd.to_datetime(data['local_time']).dt.tz_localize(None)
# Convert the timestamp string to a pandas datetime and strip timezone info
# so it aligns cleanly with the ETESA real-generation data (which is tz-naive).

data = data.set_index('local_time')
# Use the timestamp as the row index for time-series alignment.

# ── Wind ─────────────────────────────────────────────────────────────────────────
r_wind = s.get(url_wind, params=args_wind)
parsed_wind = _json.loads(r_wind.text)
wind_data = pd.read_json(_json.dumps(parsed_wind['data']), orient='index')
wind_data['local_time'] = pd.to_datetime(wind_data['local_time']).dt.tz_localize(None)
wind_data = wind_data.set_index('local_time')
wind_data['electricity'] = wind_data['electricity'] / 1000
# Same pipeline as solar — convert kW→MW and set the timestamp index.

# ── Combine API outputs ──────────────────────────────────────────────────────────
df_combined = pd.concat([
    data[['irradiance_direct','irradiance_diffuse','temperature','electricity']].rename(
        columns={'electricity': 'solar_mw'}),
    wind_data[['wind_speed','electricity']].rename(
        columns={'electricity': 'wind_mw'})
], axis=1).dropna()
# Concatenate the solar columns and the wind columns side by side (axis=1).
# dropna() removes any hour where at least one API value is missing.

# ── Load ETESA real-generation data from Parquet ─────────────────────────────────
real = pd.read_csv('solar_eolica_hidro_horario_2025.csv', index_col=0, parse_dates=True)
real.index = pd.to_datetime(real.index)
# Read the CSV file. index_col=0 restores the datetime index, parse_dates ensures
# the index is parsed as datetime64.
# Columns: solar_mw_real, eolica_mw_real, hidro_mw_real — real ETESA metered values.

cols_to_drop = [c for c in real.columns if c in df_combined.columns]
df_combined  = df_combined.drop(columns=cols_to_drop).join(real, how='left')
# Drop any duplicate columns before joining, then left-join to keep every ninja row
# and attach the real ETESA generation columns.

# ── Load ETESA real demand from Excel ────────────────────────────────────────────
demanda_raw = pd.read_csv('DEM2025.csv')
# CSV already has clean headers: Unnamed:0 (dates), H1..H24 (hourly demand).

demanda_raw = demanda_raw.rename(columns={demanda_raw.columns[0]: 'fecha'})
# Rename the first column to 'fecha' for clarity.

demanda_long = demanda_raw.melt(id_vars=['fecha'], var_name='hora_str', value_name='demanda_mw')
# Convert the wide format (one column per hour) to long format (one row per hour),
# producing columns: fecha, hora_str (e.g. 'H1'), demanda_mw.

demanda_long  = demanda_long.dropna(subset=['fecha'])
demanda_long['fecha_dt']  = pd.to_datetime(demanda_long['fecha'], errors='coerce')
demanda_long  = demanda_long.dropna(subset=['fecha_dt'])
# Parse dates; drop any rows where the date cannot be converted.

demanda_long['hora_num'] = (
    demanda_long['hora_str'].str.extract(r'(\d+)').astype(float).fillna(0).astype(int) - 1
)
# Extract the numeric part from 'H1', 'H2', … 'H24' and subtract 1 so hours run 0–23.

demanda_long['timestamp'] = (
    demanda_long['fecha_dt'] + pd.to_timedelta(demanda_long['hora_num'], unit='h')
)
# Build a proper datetime timestamp by adding the hour offset to the date.

demanda_long = demanda_long.set_index('timestamp')[['demanda_mw']].sort_index()
demanda_long['demanda_mw'] = pd.to_numeric(demanda_long['demanda_mw'], errors='coerce')
# Set the timestamp as the index, keep only the demand column, sort chronologically.

if 'demanda_mw' in df_combined.columns:
    df_combined = df_combined.drop(columns=['demanda_mw'])
# Guard against duplicate column names before the join.

df_combined = df_combined.join(demanda_long, how='left')
# Join the demand series onto the meteorological data using the datetime index.

df_combined = df_combined[df_combined.index >= '2025-01-01 01:00:00']
# Drop the incomplete first hour (00:00 row often has missing demand data from ETESA).

df = df_combined
print('Columns:', df.columns.tolist())

# ────────────────────────────────────────────────────────────────────────────
# calibration
# The Renewables.ninja API provides MERRA-2 reanalysis data scaled to the installed capacity.
# However, the real irradiance measured at the Panama solar plants differs from the MERRA-2 signal.
# We calibrate the irradiance columns by computing the ratio between real ETESA solar generation
# and the Ninja solar output and applying that ratio to all irradiance features.
# This ensures that `irradiance_direct` and `irradiance_diffuse` reflect observed conditions
# rather than a global reanalysis average.
# ────────────────────────────────────────────────────────────────────────────

solar_ninja  = df['solar_mw'].replace(0, np.nan)
# The raw Ninja solar column. We replace 0 with NaN to avoid division by zero at night.

ratio_calib  = (df['solar_mw_real'] / solar_ninja).fillna(1)
# Compute the calibration ratio: real / ninja. Where ninja is NaN (night hours),
# the ratio defaults to 1 (no correction — irradiance is zero anyway).

df['irradiance_direct']  = (df['irradiance_direct']  * ratio_calib).fillna(0)
df['irradiance_diffuse'] = (df['irradiance_diffuse'] * ratio_calib).fillna(0)
# Scale both irradiance components by the calibration ratio.
# At night (solar_ninja = 0) both irradiance values are zero

df['solar_mw']  = df['solar_mw_real']
df['eolica_mw'] = df['eolica_mw_real']
df['hidro_mw']  = df['hidro_mw_real']
# Replace the Ninja electricity outputs with the real ETESA metered values.
# This is critical: we must forecast against real observed generation, not modelled output.

cols_order = [
    'irradiance_direct','irradiance_diffuse','temperature',
    'solar_mw','wind_speed','eolica_mw','hidro_mw','demanda_mw'
]
df = df[cols_order]
# Retain only the base columns we need. All other API columns (e.g. wind_mw from Ninja)
# are replaced by the calibrated real-data versions.

print(f'Base: {df.shape[1]} cols, {df.shape[0]} records.')
print(f'hidro_mw — mean={df.hidro_mw.mean():.0f} MW  min={df.hidro_mw.min():.0f}  max={df.hidro_mw.max():.0f}')
# Print a quick sanity check: typical hydro range for Panama is 600–1400 MW.

# ────────────────────────────────────────────────────────────────────────────
# check for missing values
# ────────────────────────────────────────────────────────────────────────────

print(df.isnull().sum())
# Count the number of NaN values in each column.
# A non-zero count means we have missing hours that must be handled before modelling.

# ────────────────────────────────────────────────────────────────────────────
# correct datatypes
# ────────────────────────────────────────────────────────────────────────────

df.reset_index(inplace=True)
# Move the datetime index back into a regular column called 'local_time'.
# This allows us to use .dt accessor methods to extract hour, month, day-of-week.

print(df.dtypes)
# Inspect the column data types.

# ────────────────────────────────────────────────────────────────────────────
# feature engineering — calendar and cyclic features
# The ETESA load follows strong intra-day and intra-week patterns. We encode these using:
# - **Integer features** (`month`, `hour`, `is_weekday`) for tree models.
# - **Cyclic features** (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos`)
# so that the DNN recognises that hour 23 is adjacent to hour 0, and December is adjacent to January.
# - **Holiday indicator** (`is_holiday`): binary flag for the 14 official Panamanian public holidays in 2025.
# - **Interaction feature** (`hour_weekday = hour × is_weekday`): captures the midday peak that only
# occurs on working days — v18 finding, small but consistent gain for tree models.
# ────────────────────────────────────────────────────────────────────────────

df['month'] = df['local_time'].dt.month
# Extract the calendar month (1–12) as an integer feature.

df['hour']  = df['local_time'].dt.hour
# Extract the hour of day (0–23) as an integer feature.

df['is_weekday'] = (df['local_time'].dt.dayofweek < 5).astype(int)
# 1 if Monday–Friday, 0 if Saturday–Sunday.
# dayofweek returns 0=Monday … 6=Sunday.

df['hour_weekday'] = df['hour'] * df['is_weekday']
# Interaction feature: equals the hour value on weekdays, zero on weekends.
# Captures the different diurnal shapes between working and non-working days.

# National holidays
_panama_holidays_2025 = [
    '2025-01-01', '2025-01-09', '2025-03-04', '2025-03-05', '2025-04-18',
    '2025-05-01', '2025-08-15', '2025-11-03', '2025-11-04', '2025-11-05',
    '2025-11-10', '2025-11-28', '2025-12-08', '2025-12-25',
]
_holiday_set = {pd.Timestamp(d).date() for d in _panama_holidays_2025}
# Build a set of date objects for O(1) lookup.

df['is_holiday'] = df['local_time'].dt.date.map(lambda d: int(d in _holiday_set))
# Map each row's date to 1 (holiday) or 0 (not a holiday).
# Note: is_holiday is a SPARSE binary feature (only 14 days per year = ~1 of the data).
# It is suitable for flat feature spaces (XGBoost handles sparse splits well),


# ── Day-of-week cyclic encoding ───────────────────────────────────────────────────
df['dow_sin'] = np.sin(2 * np.pi * df['local_time'].dt.dayofweek / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['local_time'].dt.dayofweek / 7)
# Cyclic encoding ensures that Sunday (6) and Monday (0) are numerically close.
# sin and cos together uniquely encode each day (neither alone does so).

datetime_index = df['local_time'].copy()
df = df.drop(columns=['local_time'])
# Save the datetime column separately so we can use it for plot axes later,
# then remove it from the feature DataFrame.

print(f'Holidays 2025: {len(_panama_holidays_2025)} days → {df.is_holiday.sum()} hours marked')
print(df.to_string())

# ────────────────────────────────────────────────────────────────────────────
# target variable and feature construction
# The **residual demand** is defined as:
# > `demanda_residual = demanda_mw − solar_mw − eolica_mw`
# This is the portion of demand that must be met by hydroelectric, thermal, and imports
# not by variable renewables. Forecasting residual demand is the core operational problem
# under high renewable penetration, because residual demand drives hydroelectric dispatch
# scheduling and thermal unit commitment decisions.
# The **lag features** are the pre-computed shifted values of `demanda_residual`.
# The NWP horizon features are the meteorological values shifted 24 hours forward (future NWP at the target time).
# The hydro dispatch features capture the operational state of the hydroelectric fleet.
# ────────────────────────────────────────────────────────────────────────────

# Target: residual demand
df['demanda_residual'] = df['demanda_mw'] - df['solar_mw'] - df['eolica_mw']
# Residual demand = total demand minus variable renewables (solar + wind).
# This is what hydro + thermal must supply. Our forecasting target.

# Cyclic encoding
df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)
# Encode the hour of day cyclically. Period = 24 hours.

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
# Encode the month of year cyclically. Period = 12 months.

# ── Lagged target features (Lagged Approach)
# Pearson correlations with target: r(L168)=0.600 > r(L336)=0.591 > r(L48)=0.511
# r(L24) ≈ 0  →  dropped (nearly collinear with the current-hour 'demanda_residual').

df['residual_L336'] = df['demanda_residual'].shift(LAG_2)
# 336-hour lag (2 weeks): captures the biweekly hydroelectric dispatch cycle
# that is characteristic of Panama's grid management.

df['residual_L168'] = df['demanda_residual'].shift(LAG_3)
# 168-hour lag (1 week): same hour, same day of the week, last week.
# Strongest single lag predictor (r = 0.600).

df['residual_L48']  = df['demanda_residual'].shift(48)
# 48-hour lag (2 days): captures the mid-week demand build-up pattern.

# NWP Horizon features at t+24
# Meteorological forecast at the prediction horizon.
# Standard in day-ahead operational forecasting.
# Here we use MERRA-2 reanalysis as a perfect-foresight proxy.
# irradiance_direct_h24 was feature importance #3 in XGBoost (gain = 0.080).
# Adding these 4 features reduced DNN MAPE from ~9.3 % to 7.08 % (−2.35 pp).
df['irradiance_direct_h24']  = df['irradiance_direct'].shift(-HORIZON)
df['irradiance_diffuse_h24'] = df['irradiance_diffuse'].shift(-HORIZON)
df['temperature_h24']        = df['temperature'].shift(-HORIZON)
df['wind_speed_h24']         = df['wind_speed'].shift(-HORIZON)
# shift(-24) places the value that occurs 24 hours in the future alongside the current row,
# making the h=24 NWP forecast available as a feature at training and prediction time.

# Hydro dispatch features
# Note: hidro_mw itself is already in the feature set, but its LEVEL is dangerous
# (hidro + termica = residual_demand by definition → perfect collinearity if both are used).
# We encode the operational state through four derived features instead.

df['hidro_fraction_L24'] = (
    df['hidro_mw'].shift(LAG_1) /
    df['demanda_residual'].shift(LAG_1).replace(0, np.nan)
).clip(0, 1.5).fillna(0.5)
# Yesterday's hydro share of residual demand (0–1.5, clipped for outliers).
# Captures whether the system was in a hydro-heavy or hydro-light operational state.

df['hidro_delta_L24'] = (
    df['hidro_mw'].shift(LAG_1) - df['hidro_mw'].shift(LAG_1 * 2)
).fillna(0)
# Day-over-day trend in hydro dispatch. Positive = reservoir building, negative = drawing down.

_hidro_profile = df.groupby(['month', 'hour'])['hidro_mw'].transform('mean')
# Compute the typical hydro dispatch for each (month, hour) combination across the full year.
# This is the seasonal-diurnal hydro baseline.

df['hidro_typical_h24'] = _hidro_profile.shift(-HORIZON)
# The expected hydro level at the forecast horizon (h+24), based on the seasonal profile.

df['hidro_anomaly_L24'] = (df['hidro_mw'] - _hidro_profile).shift(LAG_1)
# Yesterday's deviation of actual hydro from its seasonal baseline.
# Positive = surplus reservoir conditions; Negative = drought / low-water stress signal.

# Target at h+24
df['demanda_residual_h24'] = df['demanda_residual'].shift(-HORIZON)
# The value we want to predict: residual demand 24 hours ahead.
# shift(-24) aligns each row's features with the target 24 hours later.

#Final feature order: 30 flat features + 1 target
cols_order = [
    # Current meteorology (7)
    'irradiance_direct', 'irradiance_diffuse', 'temperature',
    'solar_mw', 'wind_speed', 'eolica_mw', 'hidro_mw',
    # NWP horizon covariates at t+24 (4) ← most impactful group (v22)
    'irradiance_direct_h24', 'irradiance_diffuse_h24',
    'temperature_h24', 'wind_speed_h24',
    # Calendar features (11)
    'month', 'hour', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_weekday', 'hour_weekday', 'is_holiday', 'dow_sin', 'dow_cos',
    # Lagged target — Lagged Approach, snn_forec (4)
    'demanda_residual',
    'residual_L48', 'residual_L336', 'residual_L168',
    # Hydro dispatch state (4)
    'hidro_fraction_L24', 'hidro_delta_L24', 'hidro_typical_h24', 'hidro_anomaly_L24',
    # Target (1)
    'demanda_residual_h24',
]
df             = df[cols_order]
# Trim: drop first MAX_LAG rows (NaN from lag features) and last HORIZON rows (NaN from future shift).
df             = df.iloc[MAX_LAG:-HORIZON].reset_index(drop=True)
datetime_index = datetime_index.iloc[MAX_LAG:-HORIZON].reset_index(drop=True)
# Also trim datetime_index to match the trimmed df row-for-row.

print(f'Records:  {len(df):,} hourly ({len(df)/24:.0f} days)')
print(f'Datetime: {datetime_index.iloc[0]} → {datetime_index.iloc[-1]}')
print(f'Features: {len(df.columns)-1} flat + 1 target = {len(df.columns)} columns')
print(df.describe().round(2))

# ────────────────────────────────────────────────────────────────────────────
# Exploratory Data Analysis
# EDA: inspect the data through plots, autocorrelation, and correlation analysis.
# This section plot the data → check stationarity → correlation matrix.
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# plot the data
# ────────────────────────────────────────────────────────────────────────────

df.plot(subplots=True, figsize=(14, 42), layout=(16, 2))
# Plot each of the 31 columns in its own sub-panel.
# layout=(16, 2) gives 32 slots — enough for 31 columns. layout=(9, 2) would fail (only 18 slots).
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# autocorrelation analysis and target scatter
# The ACF reveals the dominant periodic structure: peaks at lags 24 (daily), 168 (weekly),
# and 336 (biweekly). These directly motivate the choice of lag features.
# The scatter plot confirms that the current residual demand is a useful but imperfect predictor
# of the 24-hour-ahead target (r ≈ 0.3–0.4), justifying the addition of lag and NWP features.
# ────────────────────────────────────────────────────────────────────────────

from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats.stats import pearsonr
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

plot_acf(df['demanda_residual'], lags=350, ax=axes[0], alpha=0.05)
# Autocorrelation Function up to lag 200. Blue shading = 95 % confidence band.
# Bars outside the band are statistically significant autocorrelations.

for lag, color, label in [(24, 'red', 'L24'), (168, 'green', 'L168'), (336, 'blue', 'L336')]:
    axes[0].axvline(x=lag, color=color, linestyle='--', linewidth=1.5, label=label)
# Vertical lines highlight the lag values used as features.

axes[0].set_title('ACF demanda_residual — 200 lags')
axes[0].legend()

axes[1].scatter(df['demanda_residual'].values, df['demanda_residual_h24'].values,
                alpha=0.08, s=2, color='steelblue')
# Scatter: current residual demand (x) vs. target 24 h ahead (y).
# Low alpha (0.08) avoids overplotting the ~7 000 training points.

corr_r = df[['demanda_residual', 'demanda_residual_h24']].corr().iloc[0, 1]
axes[1].set_title(f'residual(t) vs target h=24   r={corr_r:.3f}')
axes[1].set_xlabel('demanda_residual(t)')
axes[1].set_ylabel('target h=24')

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# correlation matrix
# If some time series are non-stationary, the Pearson correlation matrix can be misleading
# (spurious correlations). We use the KPSS test to identify non-stationary columns.
# A KPSS p-value ≤ 1 % means the series is NOT stationary (null hypothesis = stationarity is rejected).
# ────────────────────────────────────────────────────────────────────────────

from statsmodels.tsa.stattools import kpss

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    stationarity = pd.DataFrame(
        index=df.columns,
        data=[kpss(df[c])[1] for c in df.columns],
        columns=['KPSS p-value']
    )
# Run KPSS stationarity test for every column.
# If p-value > 0.10 (shown as 0.10) → stationary. We do NOT reject H0.
# If p-value = 0.01 (the minimum reported) → NOT stationary.
print(stationarity)

r = df.corr()
# Compute the Pearson correlation matrix for all 31 columns.

print('Correlations with target (demanda_residual_h24):')
print(r['demanda_residual_h24'].drop('demanda_residual_h24').sort_values(ascending=False).round(3))
# Print the correlation of each feature with the target, sorted highest-to-lowest.
# Key expected findings:
#   demanda_residual (current):   r ≈ 0.3–0.4   (direct but weak — NWP needed)
#   residual_L168:                r ≈ 0.600      (strongest lag)
#   residual_L336:                r ≈ 0.591
#   irradiance_direct_h24:        r ≈ -0.4       (high solar → low residual demand)

plt.figure(figsize=(10, 8))
sns.heatmap(r, vmin=-1, vmax=1, annot=False, cmap='Spectral')
plt.title('Pearson Correlation Matrix — 31 features + target')
plt.tight_layout()
plt.show()
# Dark red = strong positive correlation;

# ────────────────────────────────────────────────────────────────────────────
# Data Structures
# - `predictions` — **dictionary of lists**: each list holds the training-set prediction DataFrame for every rolling iteration.
# - `forecasts`   — **dictionary of DataFrames**: each DataFrame holds the test-set (day-ahead) forecasts, indexed by datetime.
# - `errors`      — test-set MAPE per model (primary metric).
# - `training_errors` — training-set MAPE per model.
# - Additional DataFrames for WAPE, sMAPE, and R² (both test and training).
# Two models only: **`deep_network`** (DNN) and **`xgboost`**.
# ────────────────────────────────────────────────────────────────────────────

from sklearn.metrics import (mean_absolute_percentage_error, mean_absolute_error,
                              mean_squared_error, r2_score)

def compute_metrics(actual, predicted):
    """
    Compute four error metrics: MAPE, WAPE, sMAPE, R² — v24 metric suite.

    MAPE  (Shringi et al. 2025): standard, but sensitive when residual demand approaches zero.
    WAPE  (Feng et al. 2026)   : uses the sum of actuals as denominator — robust near zero.
    sMAPE                      : symmetric, bounded 0–200 %, avoids MAPE asymmetry.
    R²                         : fraction of variance explained; 1.0 = perfect forecast.

    WAPE is consistently ~1 pp lower than MAPE here because during high-solar midday hours
    the residual demand shrinks toward zero, causing MAPE to spike in individual rows.
    The aggregate WAPE denominator dilutes those hours.
    """
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)

    mape  = mean_absolute_percentage_error(actual, predicted)
    # sklearn's MAPE: mean(|actual - predicted| / |actual|).

    wape  = np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual)) + 1e-8)
    # Weighted APE: total absolute error / total absolute actual.
    # The 1e-8 guard prevents division by zero on an all-zero actual (edge case).

    smape = np.mean(2 * np.abs(actual - predicted) /
                    (np.abs(actual) + np.abs(predicted) + 1e-8))
    # Symmetric MAPE: 2|e| / (|a| + |p|), averaged across rows.

    r2    = r2_score(actual, predicted)
    # Coefficient of determination: 1 - SS_res / SS_tot.

    return (mape, wape, smape, r2)

predictions = {
    'deep_network': [],
    'xgboost':      [],
}
# Dictionary of lists.
# Each key is a model name. Each list grows by one entry per rolling iteration:
# entry i is a DataFrame of training-set predictions for the model fitted up to T_day_i.

target_h24   = 'demanda_residual_h24'
# The column name of our prediction target.

features_h24 = [c for c in df.columns if c != target_h24]
# List of all feature column names — every column except the target.
# Should be exactly 30 features.

forecasts = {
    k: pd.DataFrame(data=np.nan, columns=[target_h24], index=datetime_index)
    for k in predictions
}
# Each DataFrame has one column (target_h24) and one row per hour in the full dataset.
# Rows in the training window stay NaN; the rolling loop fills in the test window.

_cols = list(predictions.keys())
# ['deep_network', 'xgboost'] — used to construct the error DataFrames below.

errors          = pd.DataFrame(index=[target_h24], columns=_cols)
training_errors = pd.DataFrame(index=[target_h24], columns=_cols)
# MAPE DataFrames: rows = targets (only one: demanda_residual_h24), columns = models.

errors_wape    = pd.DataFrame(index=[target_h24], columns=_cols)
errors_smape   = pd.DataFrame(index=[target_h24], columns=_cols)
errors_r2      = pd.DataFrame(index=[target_h24], columns=_cols)
training_wape  = pd.DataFrame(index=[target_h24], columns=_cols)
training_smape = pd.DataFrame(index=[target_h24], columns=_cols)
training_r2    = pd.DataFrame(index=[target_h24], columns=_cols)


print('Data structures initialised — 2 models × 4 metrics.')
print(f'Features: {len(features_h24)} flat (lagged approach)')
print(f'Models:   {_cols}')

# ────────────────────────────────────────────────────────────────────────────
# Feature Extraction via Lagged Approach
# this is done by the function `get_targets_features`, which returns `Y_train`, `X_train`, `X_test`
# based on the **lagged values approach** (the parameter `use_polynomials = False`).
# all lag features (`residual_L48`, `residual_L168`, `residual_L336`) and NWP horizon
# features (`irradiance_direct_h24`) have already been pre-computed as columns of `df` in Section 1.
# The function's job here is simply to **slice** the DataFrame at cutoff `T` to produce training and test splits.
# Below we import the `StandardScaler` which is required when `scale=True` (for the DNN).
# For XGBoost we use `scale=False` because tree models are scale-invariant.
# ────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import StandardScaler
# StandardScaler fits a mean and standard deviation to the training data,
# then transforms it to zero mean and unit variance.
# It is applied to both X and Y for the DNN (for neural networks).
# It is NOT applied for XGBoost because gradient-boosted trees do not need normalised inputs.

def get_targets_features(df_in, T, scale=False):
    """
    Feature extraction — Lagged Approach (snn_forec pattern).

    Parameters
    ----------
    df_in  : DataFrame — the full dataset with all 30 features and the target column.
    T      : int       — training cutoff (number of rows used for training).
    scale  : bool      — if True, apply StandardScaler to X and Y (for DNN).
                         if False, return raw arrays (for XGBoost).

    Returns
    -------
    scale=False → (Y_train, X_train, X_test)
    scale=True  → (Y_train, X_train, X_test, sx, sy)
                   where sx = feature scaler, sy = target scaler.

    The test window covers the next HORIZON=24 rows starting at row T.
    """
    Y_train = df_in[[target_h24]].iloc[:T].copy()
    # Target column for rows 0…T-1. Shape: (T, 1).

    X_train = df_in[features_h24].iloc[:T].copy()
    # Feature matrix for rows 0…T-1. Shape: (T, 30).

    X_test  = df_in[features_h24].iloc[T:T + HORIZON].copy()
    # Feature matrix for the next 24 rows (one day-ahead forecast window). Shape: (24, 30).

    if scale:
        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        # Fit scalers on training data only — never on test data (would cause data leakage).

        X_train = pd.DataFrame(sx.transform(X_train), columns=X_train.columns)
        X_test  = pd.DataFrame(sx.transform(X_test),  columns=X_test.columns)
        Y_train = pd.DataFrame(sy.transform(Y_train), columns=Y_train.columns)
        # Apply the same fitted scaler to transform training and test features.
        # X_test uses the scaler fitted on X_train (same mean and std).

        return Y_train, X_train, X_test, sx, sy

    return Y_train, X_train, X_test

# ────────────────────────────────────────────────────────────────────────────
# Below we verify the function by calling it once at the fixed training cutoff T = 7000.
# We check that the shapes are correct and that the feature count equals exactly 30.
# ────────────────────────────────────────────────────────────────────────────

T_test = 7000
Y_v, X_v, Xt_v = get_targets_features(df_in=df, T=T_test)

print(f'X_train: {X_v.shape}   X_test: {Xt_v.shape}   (should be (7000, 30) and (24, 30))')
print(f'Y_train: {Y_v.shape}   (should be (7000, 1))')

assert X_v.shape[1] == 30, f'Feature count error: {X_v.shape[1]}'
# Hard assertion: the number of features must be exactly 30 as documented in the context.

print(f'\n Shapes correct')
print(f'\nFeature list ({len(features_h24)} features):')
print(features_h24)
# Print the feature list so it is documented in the notebook output.

# ────────────────────────────────────────────────────────────────────────────
# Deep Neural Network (DNN) — Rolling h=24
# > `Dense(256, relu) → Dense(256, relu) → Dense(128, relu) → Dense(1)`
# - Trained with **Adam optimiser** (learning rate 0.001) and **MSE loss**.
# - **EarlyStopping** with patience=5 on validation loss; `restore_best_weights=True`.
# - Validation split = 10 % (last 10 % of the training window, preserving temporal order).
# - Both X and Y are scaled with StandardScaler (for neural networks).
# - Seed 42 is fixed at each iteration so results are reproducible.
# **Best result: 7.08 % MAPE, WAPE=0.0616, R²=0.7765, overfit=0.79 pp.**
# Key finding: adding NWP horizon features reduced DNN MAPE from ~9.3 % to 7.08 % (−2.35 pp).
# meteorological forecasts outperforms sequential models without them.
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# 5.1_ Using Rolling Predictions to train the model
# The rolling loop below implements **walk-forward (expanding window) validation** — the standard
# evaluation protocol for time-series forecasting
# For each iteration:
# 1. The training window expands by one day (`T_day = T + day_idx × HORIZON`).
# 2. The model is re-fitted on all data up to `T_day`.
# 3. The model forecasts the next 24 hours (one full day).
# 4. The forecast is stored in `forecasts['deep_network']` and the training predictions in `predictions['deep_network']`.
# 5. The model object and all intermediate DataFrames are saved in `globals()` under keys like
# `model_dnn7000`, `X_train_dnn7000`
# ────────────────────────────────────────────────────────────────────────────

import os, random, tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

# tqdm wraps the for-loop and displays a progress bar
# EarlyStopping stops training when val_loss has not improved for 'patience' epochs.

def set_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Fix the Python hash seed for determinism across runs.
    tf.random.set_seed(seed)
    # Fix TensorFlow's random seed (affects weight initialisation).
    np.random.seed(seed)
    # Fix NumPy's random seed (affects dropout, shuffling, etc.).
    random.seed(seed)
    # Fix Python's built-in random seed.
# This function ensures that every DNN model in the rolling loop starts with
# the same weight initialisation, making results reproducible.

T               = 7000
# Training cutoff: first T=7000 hours are used for initial training.
# Approximately January 1 – late October 2025 (given ~8,400 valid rows after burn-in).

period_selected = len(df) - T
# Number of hours in the test period.

n_test_days  = period_selected // HORIZON
# Number of complete 24-hour days in the test period.
# Each day = one rolling iteration.

n_test_hours = n_test_days * HORIZON
# Total test hours (rounded down to complete days).

print(f'Train:         {T:,} hours ({T/24:.0f} days)')
print(f'Test:          {period_selected:,} hours ({period_selected/24:.0f} days)')
print(f'Rolling days:  {n_test_days} iterations ({n_test_hours:,} hours in test)')
print(f'Test start:    {datetime_index.iloc[T]}')
print(f'Test end:      {datetime_index.iloc[T + n_test_hours - 1]}')

# ── Naive benchmark ────────────────────────────────────────────────────────────────
naive_preds  = df['demanda_residual'].iloc[T:T + n_test_hours].values
naive_actual = df[target_h24].iloc[T:T + n_test_hours].values
# Naive model = persistence: predict tomorrow's residual demand = today's residual demand.
# This is the minimum acceptable benchmark: every model must beat it.

naive_mape, naive_wape, naive_smape, naive_r2 = compute_metrics(naive_actual, naive_preds)
print(f'\nNaive: MAPE={naive_mape:.2%}  WAPE={naive_wape:.4f}  sMAPE={naive_smape:.2%}  R²={naive_r2:.4f}')

for day_idx in tqdm(range(n_test_days), desc='DNN rolling h=24'):

    set_random_seeds(42)
    # Reset seeds at the start of each iteration so every DNN model is trained with
    # the same weight initialisation strategy. Reproducibility requires this.

    T_day = T + day_idx * HORIZON
    # The training cutoff for this iteration: T + day 0 → T_day = T, T + day 1 → T+24, etc.
    # The training window expands by 24 hours with each iteration (expanding window).

    Y_train, X_train, X_test, scaler_x, scaler_y =         get_targets_features(df_in=df, T=T_day, scale=True)
    # Extract scaled training and test arrays.
    # scale=True: X_train, X_test, Y_train are all standardised (mean=0, std=1).
    # scaler_x and scaler_y are the fitted scalers — needed for inverse_transform below.

    model = Sequential([
        Dense(256, activation='relu'),
        # First hidden layer: 256 neurons, ReLU activation.
        # ReLU (Rectified Linear Unit) = max(0, x). Avoids the vanishing gradient problem.
        Dense(256, activation='relu'),
        # Second hidden layer: 256 neurons, ReLU. Two layers of 256 give the DNN enough
        # capacity to capture the nonlinear interactions between irradiance, hydro, and demand.
        Dense(128, activation='relu'),
        # Third hidden layer: 128 neurons. Narrows before the output.
        Dense(1)
        # Output layer: 1 neuron (single forecast — demanda_residual_h24 in MW).
        # No activation = linear output, appropriate for regression.
    ])
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    # Adam with lr=0.001: the standard adaptive optimiser. MSE loss for regression.

    model.fit(
        X_train, Y_train,
        epochs=100,          # maximum 100 training epochs
        batch_size=256,      # 256 samples per gradient update (fast for ~7000 rows)
        validation_split=0.1,# last 10 % of training set used for validation
        callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                  restore_best_weights=True)],
        # Stop early if validation loss does not improve for 5 epochs.
        # restore_best_weights=True reloads the weights from the best epoch.
        verbose=0            # suppress per-epoch output (progress bar from tqdm is enough)
    )

    df_Y_pred1 = pd.DataFrame(
        scaler_y.inverse_transform(model.predict(X_train, verbose=0)),
        columns=[target_h24],
        index=datetime_index[:T_day]
    )
    # Training-set predictions: inverse-transform from the scaled space back to MW.
    # index = the actual datetime labels of the training window.

    df_Y_pred2 = pd.DataFrame(
        scaler_y.inverse_transform(model.predict(X_test, verbose=0)),
        columns=[target_h24],
        index=datetime_index[T_day:T_day + HORIZON]
    )
    # Test-set predictions (24-hour forecast): inverse-transform to MW.
    # index = the actual datetime labels of the 24-hour forecast window.

    forecasts['deep_network'].iloc[T_day:T_day + HORIZON] =         df_Y_pred2[target_h24].values.reshape(-1, 1)
    # Store the 24-hour forecast in the forecasts DataFrame at the correct rows.
    # reshape(-1, 1) ensures the array is 2-D for DataFrame assignment.

    predictions['deep_network'].append(df_Y_pred1)
    # Append this iteration's training predictions to the list.
    # The last element predictions['deep_network'][-1] = training predictions for the
    # largest training window (used for training MAPE calculation)

    # store all per-iteration artifacts in globals()
    globals()['model_dnn'        + str(T_day)] = model
    globals()['scaler_x_dnn'     + str(T_day)] = scaler_x
    globals()['scaler_y_dnn'     + str(T_day)] = scaler_y
    globals()['X_train_dnn'      + str(T_day)] = X_train.copy()
    globals()['X_test_dnn'       + str(T_day)] = X_test.copy()
    globals()['df_Y_pred1_dnn'   + str(T_day)] = df_Y_pred1.copy()
    globals()['df_Y_pred2_dnn'   + str(T_day)] = df_Y_pred2.copy()
    globals()['predictions_dnn_' + str(T_day)] = predictions['deep_network'].copy()
    globals()['forecasts_dnn'    + str(T_day)] = forecasts['deep_network'].copy()
    # Each key is the variable name plus T_day (an integer).
    # This allows inspection of any specific rolling-window model after the loop finishes.

# ────────────────────────────────────────────────────────────────────────────
# 5.2_ Plotting the test set predictions
# Three panels: (1) full test period, (2) first 7 days zoom, (3) absolute error.
# ────────────────────────────────────────────────────────────────────────────

T_last       = T + (n_test_days - 1) * HORIZON
# T_last is the training cutoff of the final rolling iteration.

actual_test  = df[target_h24].iloc[T:T + n_test_hours].values
# Actual residual demand values over the test period.

pred_test    = forecasts['deep_network'][target_h24].iloc[T:T + n_test_hours].values.astype(float)
# DNN forecasts for the test period, retrieved from the forecasts DataFrame.

dates_test   = datetime_index.iloc[T:T + n_test_hours]
# Datetime labels for the test period (used as x-axis in plots).

test_mape_dnn, test_wape_dnn, test_smape_dnn, test_r2_dnn =     compute_metrics(actual_test, pred_test)
# Compute all four metrics for the test period.

_mae_dnn = mean_absolute_error(actual_test, pred_test)
# Mean Absolute Error in MW — useful for operational interpretation
# (average forecast error expressed in the same unit as demand).

zoom = 7 * HORIZON
# Number of hours to show in the zoom panel: 7 days × 24 hours.

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

axes[0].plot(dates_test, actual_test, label='Actual', color='steelblue', linewidth=0.8)
axes[0].plot(dates_test, pred_test,   label='DNN Forecast', color='orange', linewidth=0.8)
axes[0].set_title(
    f'DNN h=24 — Test period\n'
    f'MAPE={test_mape_dnn:.2%} | WAPE={test_wape_dnn:.4f} | '
    f'sMAPE={test_smape_dnn:.2%} | R²={test_r2_dnn:.4f} | Naive={naive_mape:.2%}', fontsize=10)
axes[0].set_ylabel('MW')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(dates_test[:zoom], actual_test[:zoom], color='steelblue', linewidth=1.2, label='Actual')
axes[1].plot(dates_test[:zoom], pred_test[:zoom],   color='orange',    linewidth=1.2, label='DNN h=24')
axes[1].set_title('DNN h=24 — First 7 days (zoom)', fontsize=11)
axes[1].set_ylabel('MW')
axes[1].legend()
axes[1].grid(alpha=0.3)

abs_error = np.abs(actual_test - pred_test)
axes[2].fill_between(dates_test, abs_error, alpha=0.5, color='tomato', label='|Error| MW')
axes[2].axhline(_mae_dnn, color='red', linestyle='--', linewidth=1, label=f'MAE={_mae_dnn:.1f} MW')
axes[2].set_title('DNN h=24 — Absolute error', fontsize=11)
axes[2].set_ylabel('MW')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# 5.3_ Plotting the training set predictions
# We use `predictions['deep_network'][-1]` — the training predictions of the last rolling iteration
# (the largest training window).
# ────────────────────────────────────────────────────────────────────────────

actual_train = df[target_h24].iloc[:T_last].values
# Actual residual demand values over the training period.

pred_train   = predictions['deep_network'][-1][target_h24].values
# Training-set predictions from the last rolling iteration (largest window).

dates_train  = datetime_index.iloc[:T_last]
# Datetime labels for the training period.

train_mape_dnn, train_wape_dnn, train_smape_dnn, train_r2_dnn =     compute_metrics(actual_train, pred_train)
# Training-set metrics — used in the overfitting analysis below.

zoom_tr = 7 * HORIZON
# First 7 days for the training zoom panel.

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

axes[0].plot(dates_train, actual_train, label='Actual', color='steelblue', linewidth=0.6, alpha=0.8)
axes[0].plot(dates_train, pred_train,   label='DNN Training', color='darkorange', linewidth=0.6, alpha=0.8)
axes[0].set_title(
    f'DNN h=24 — Training (T={T_last})\n'
    f'MAPE={train_mape_dnn:.2%} | WAPE={train_wape_dnn:.4f} | R²={train_r2_dnn:.4f}', fontsize=11)
axes[0].set_ylabel('MW')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(dates_train[:zoom_tr], actual_train[:zoom_tr], color='steelblue', linewidth=1.2, label='Actual')
axes[1].plot(dates_train[:zoom_tr], pred_train[:zoom_tr],   color='darkorange', linewidth=1.2, label='DNN Training (zoom)')
axes[1].set_title('DNN h=24 — Training first 7 days (zoom)', fontsize=11)
axes[1].set_ylabel('MW')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# 5.4_ Training set MAPE
# 5.5_ Test set MAPE
# 5.6_ Overfitting analysis
# 5.7_ Naive model benchmark
# 5.8_ Model selection
# All four metrics (MAPE, WAPE, sMAPE, R²) for both training and test are stored into the
# snn_forec error DataFrames. Then we check overfitting (test MAPE − train MAPE)
# and run the model selection criteria: beat naive AND overfit < 10 pp.
# ────────────────────────────────────────────────────────────────────────────

# ── Store all metrics in the error DataFrames
training_errors.loc[target_h24, 'deep_network'] = train_mape_dnn
training_wape.loc[target_h24,   'deep_network'] = train_wape_dnn
training_smape.loc[target_h24,  'deep_network'] = train_smape_dnn
training_r2.loc[target_h24,     'deep_network'] = train_r2_dnn
# Training-set metrics: fills in the 'deep_network' column of each error DataFrame.

errors.loc[target_h24,       'deep_network'] = test_mape_dnn
errors_wape.loc[target_h24,  'deep_network'] = test_wape_dnn
errors_smape.loc[target_h24, 'deep_network'] = test_smape_dnn
errors_r2.loc[target_h24,    'deep_network'] = test_r2_dnn
# Test-set metrics: fills in the 'deep_network' column.

does_it_overfit_dnn = float(test_mape_dnn - train_mape_dnn)
# Overfitting gap = test MAPE − train MAPE.
# A positive value means the model is generalising imperfectly.
# The snn_forec threshold is 0.10 (10 pp). DNN target: < 1 pp.

improvement_dnn = (naive_mape - test_mape_dnn) / naive_mape * 100
# Percentage improvement over the naive benchmark.

print(f'DNN Train : MAPE={train_mape_dnn:.2%}  WAPE={train_wape_dnn:.4f}  R²={train_r2_dnn:.4f}')
print(f'DNN Test  : MAPE={test_mape_dnn:.2%}  WAPE={test_wape_dnn:.4f}  sMAPE={test_smape_dnn:.2%}  R²={test_r2_dnn:.4f}')
print(f'DNN Overfit: {does_it_overfit_dnn:.2%}  |  Improvement over naive: {improvement_dnn:.1f} %')
print()

#5.7 Naive model benchmark
print(f'Naive benchmark: MAPE={naive_mape:.2%}  WAPE={naive_wape:.4f}  R²={naive_r2:.4f}')
print()

#5.8 Model selection
print('====== DNN h=24 — MODEL SELECTION ')
if naive_mape <= test_mape_dnn:
    print('The DNN does NOT beat the naive model. Not recommended for forecasts.')
elif does_it_overfit_dnn >= 0.1:
    print(f'The DNN overfits ({does_it_overfit_dnn:.2%} gap). Not recommended for forecasts.')
else:
    print(
        f'*** The DNN can be used for forecasting. ***\n'
        f'    MAPE={test_mape_dnn:.2%} | R²={test_r2_dnn:.4f} | '
        f'Improvement over naive: {improvement_dnn:.1f} %'
    )
# Ie a model can be used for forecasts only if:
#it beats the naive benchmark, AND
#it does not overfit (test MAPE - train MAPE < 10 pp).

# ────────────────────────────────────────────────────────────────────────────
# XGBoost — Rolling h=24
# Tree models are scale-invariant, so no StandardScaler is applied.
# The validation set is the last 10 % of the training window, preserving temporal order.
# **Best result : 7.35 % MAPE, WAPE=0.0621, R²=0.7539, overfit=2.75 pp.**
# Feature importance confirms that `irradiance_direct_h24` is the 3rd most important feature (gain=0.080),
# validating the NWP horizon hypothesis.
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# 6.1_ Using Rolling Predictions to train the model
# ────────────────────────────────────────────────────────────────────────────

import subprocess
subprocess.check_call(['pip', 'install', 'xgboost', '-q'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
from xgboost import XGBRegressor
# XGBRegressor: the scikit-learn compatible XGBoost regression interface.

for day_idx in tqdm(range(n_test_days), desc='XGBoost rolling h=24'):

    T_day = T + day_idx * HORIZON
    # Training cutoff for this iteration (same expanding-window logic as DNN).

    Y_train_xgb, X_train_xgb, X_test_xgb =         get_targets_features(df_in=df, T=T_day, scale=False)
    # scale=False: no StandardScaler for XGBoost — tree models are scale-invariant.
    # Returns raw DataFrames (not numpy arrays), which XGBoost handles natively.

    # Temporal validation set: last 10 % of training window
    val_size  = max(24, round(len(X_train_xgb) * 0.10))
    # At least 24 hours for the validation set. round() then max() ensures
    # we always have a meaningful validation set even at the start of the rolling loop.

    X_val_xgb = X_train_xgb.iloc[-val_size:]
    y_val_xgb = Y_train_xgb.values.ravel()[-val_size:]
    # Validation set: the LAST val_size rows of the training window (most recent data).
    # Using the last 10 % (not a random split) preserves temporal order — the validation
    # set is more similar in distribution to the test set (same season / time of year).

    X_tr_xgb = X_train_xgb.iloc[:-val_size]
    y_tr_xgb = Y_train_xgb.values.ravel()[:-val_size]
    # The actual training subset: all rows up to the start of the validation window.
    # .ravel() converts the (N, 1) DataFrame to a 1-D array as required by XGBoost.

    model_xgb = XGBRegressor(
        n_estimators=500,       # maximum number of boosting rounds (trees)
        learning_rate=0.03,     # shrinkage per round — smaller = more robust
        max_depth=5,            # maximum tree depth — controls model complexity
        subsample=0.8,          # fraction of training rows per tree (row subsampling)
        colsample_bytree=0.8,   # fraction of features per tree (column subsampling)
        reg_alpha=0.1,          # L1 regularisation on leaf weights
        reg_lambda=1.0,         # L2 regularisation on leaf weights
        early_stopping_rounds=20,  # CONSTRUCTOR (XGBoost ≥ 2.0 requirement)
        # Stop if validation metric has not improved for 20 consecutive rounds.
        random_state=42,        # random seed for reproducibility
        verbosity=0,            # suppress all XGBoost console output
        n_jobs=-1               # use all available CPU cores
    )
    model_xgb.fit(
        X_tr_xgb, y_tr_xgb,
        eval_set=[(X_val_xgb, y_val_xgb)],
        # eval_set provides the validation data for early stopping.
        verbose=False
        # verbose=False suppresses per-round eval output (separate from verbosity=0 above).
    )

    df_Y_pred1_xgb = pd.DataFrame(
        model_xgb.predict(X_train_xgb),
        columns=[target_h24],
        index=datetime_index[:T_day]
    )
    # Training predictions in MW (no inverse_transform needed — XGBoost is unscaled).

    df_Y_pred2_xgb = pd.DataFrame(
        model_xgb.predict(X_test_xgb),
        columns=[target_h24],
        index=datetime_index[T_day:T_day + HORIZON]
    )
    # 24-hour test forecast in MW.

    forecasts['xgboost'].iloc[T_day:T_day + HORIZON] =         df_Y_pred2_xgb[target_h24].values.reshape(-1, 1)
    # Store in the forecasts DataFrame (same pattern as DNN section 5.1).

    predictions['xgboost'].append(df_Y_pred1_xgb)
    # Append training predictions to the list.


    globals()['model_xgb'        + str(T_day)] = model_xgb
    globals()['X_train_xgb'      + str(T_day)] = X_train_xgb.copy()
    globals()['X_test_xgb'       + str(T_day)] = X_test_xgb.copy()
    globals()['df_Y_pred1_xgb'   + str(T_day)] = df_Y_pred1_xgb.copy()
    globals()['df_Y_pred2_xgb'   + str(T_day)] = df_Y_pred2_xgb.copy()
    globals()['predictions_xgb_' + str(T_day)] = predictions['xgboost'].copy()
    globals()['forecasts_xgb'    + str(T_day)] = forecasts['xgboost'].copy()

# ────────────────────────────────────────────────────────────────────────────
# 6.2_ Plotting the test set predictions
# ────────────────────────────────────────────────────────────────────────────

actual_test_xgb = df[target_h24].iloc[T:T + n_test_hours].values
# Actual residual demand values over the test period (same as for DNN).

pred_test_xgb   = forecasts['xgboost'][target_h24].iloc[T:T + n_test_hours].values.astype(float)
# XGBoost day-ahead forecasts for the test period.

dates_test_xgb  = datetime_index.iloc[T:T + n_test_hours]
# Datetime labels for the test period.

test_mape_xgb, test_wape_xgb, test_smape_xgb, test_r2_xgb =     compute_metrics(actual_test_xgb, pred_test_xgb)
# All four metrics for the test period.

_mae_xgb = mean_absolute_error(actual_test_xgb, pred_test_xgb)
# Mean absolute error in MW.

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

axes[0].plot(dates_test_xgb, actual_test_xgb, label='Actual',  color='steelblue',  linewidth=0.8)
axes[0].plot(dates_test_xgb, pred_test_xgb,   label='XGBoost', color='darkorange', linewidth=0.8)
axes[0].set_title(
    f'XGBoost h=24 — Test period\n'
    f'MAPE={test_mape_xgb:.2%} | WAPE={test_wape_xgb:.4f} | '
    f'sMAPE={test_smape_xgb:.2%} | R²={test_r2_xgb:.4f} | Naive={naive_mape:.2%}', fontsize=10)
axes[0].set_ylabel('MW')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(dates_test_xgb[:zoom], actual_test_xgb[:zoom], color='steelblue',  linewidth=1.2, label='Actual')
axes[1].plot(dates_test_xgb[:zoom], pred_test_xgb[:zoom],   color='darkorange', linewidth=1.2, label='XGBoost h=24')
axes[1].set_title('XGBoost h=24 — First 7 days (zoom)', fontsize=11)
axes[1].set_ylabel('MW')
axes[1].legend()
axes[1].grid(alpha=0.3)

abs_error_xgb = np.abs(actual_test_xgb - pred_test_xgb)
axes[2].fill_between(dates_test_xgb, abs_error_xgb, alpha=0.5, color='darkorange', label='|Error| MW')
axes[2].axhline(_mae_xgb, color='saddlebrown', linestyle='--', linewidth=1, label=f'MAE={_mae_xgb:.1f} MW')
axes[2].set_title('XGBoost h=24 — Absolute error', fontsize=11)
axes[2].set_ylabel('MW')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# 6.3_ Plotting the training set predictions
# ────────────────────────────────────────────────────────────────────────────

actual_train_xgb = df[target_h24].iloc[:T_last].values
# Actual residual demand over the training period.

pred_train_xgb   = predictions['xgboost'][-1][target_h24].values
# Training predictions from the last rolling iteration.

train_mape_xgb, train_wape_xgb, train_smape_xgb, train_r2_xgb =     compute_metrics(actual_train_xgb, pred_train_xgb)
# Training-set metrics.

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

axes[0].plot(datetime_index.iloc[:T_last], actual_train_xgb,
             label='Actual', color='steelblue', linewidth=0.6, alpha=0.8)
axes[0].plot(datetime_index.iloc[:T_last], pred_train_xgb,
             label='XGBoost Training', color='darkorange', linewidth=0.6, alpha=0.8)
axes[0].set_title(
    f'XGBoost h=24 — Training\n'
    f'MAPE={train_mape_xgb:.2%} | WAPE={train_wape_xgb:.4f} | R²={train_r2_xgb:.4f}', fontsize=11)
axes[0].set_ylabel('MW')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(datetime_index.iloc[:zoom_tr], actual_train_xgb[:zoom_tr],
             color='steelblue',  linewidth=1.2, label='Actual')
axes[1].plot(datetime_index.iloc[:zoom_tr], pred_train_xgb[:zoom_tr],
             color='darkorange', linewidth=1.2, label='XGBoost (zoom)')
axes[1].set_title('XGBoost h=24 — Training first 7 days (zoom)', fontsize=11)
axes[1].set_ylabel('MW')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# 6.4_ Feature Importance
# XGBoost computes feature importance as the mean **gain** contributed by a feature across all tree splits.
# Higher gain = the feature creates larger reductions in the loss function.
# Expected top features:
# 1. `demanda_residual` (0.244) — the current residual demand is the single best predictor.
# 2. `hour` (0.175) — strong diurnal cycle.
# 3. `irradiance_direct_h24` (0.080) — NWP horizon, confirms the h24 meteorological hypothesis.
# 4. `hour_sin` (0.059)
# 5. `residual_L336` (0.058)
# ────────────────────────────────────────────────────────────────────────────

model_xgb_last   = globals()['model_xgb'   + str(T_last)]
# Retrieve the XGBoost model fitted on the largest training window.

X_train_xgb_last = globals()['X_train_xgb' + str(T_last)]
# Retrieve the corresponding feature matrix (needed for column names).

importance_xgb = pd.Series(
    model_xgb_last.feature_importances_,
    index=X_train_xgb_last.columns
).sort_values(ascending=True)
# feature_importances_ returns an array of gain values, one per feature.
# We wrap it in a Series with column names as the index, then sort for a readable bar chart.

plt.figure(figsize=(9, 7))
importance_xgb.plot(kind='barh', color='darkorange')
# Horizontal bar chart: each bar = one feature, length = importance gain.
plt.title(f'XGBoost h=24 — Feature Importance v25 (T={T_last})')
plt.xlabel('Importance (gain)')
plt.tight_layout()
plt.show()

print('Top 10 features:')
print(importance_xgb.sort_values(ascending=False).head(10).round(4).to_dict())
# Print the top 10 features and their importance scores.

print(f'\nBest n_estimators (early stopping): {model_xgb_last.best_iteration}')
# best_iteration: the tree count at which early stopping triggered.
# A value well below 500 means the model converged before the maximum.

# ────────────────────────────────────────────────────────────────────────────
# 6.5_ Training set MAPE
# 6.6_ Test set MAPE
# 6.7_ Overfitting analysis
# 6.8_ Model selection
# ────────────────────────────────────────────────────────────────────────────

# Store all metrics in the DataFrames
training_errors.loc[target_h24, 'xgboost'] = train_mape_xgb
training_wape.loc[target_h24,   'xgboost'] = train_wape_xgb
training_smape.loc[target_h24,  'xgboost'] = train_smape_xgb
training_r2.loc[target_h24,     'xgboost'] = train_r2_xgb

errors.loc[target_h24,       'xgboost'] = test_mape_xgb
errors_wape.loc[target_h24,  'xgboost'] = test_wape_xgb
errors_smape.loc[target_h24, 'xgboost'] = test_smape_xgb
errors_r2.loc[target_h24,    'xgboost'] = test_r2_xgb
# Same pattern as DNN — fill in the 'xgboost' column of each error DataFrame.

does_it_overfit_xgb = float(test_mape_xgb - train_mape_xgb)
# Overfitting gap for XGBoost.

improvement_xgb = (naive_mape - test_mape_xgb) / naive_mape * 100
# Percentage improvement over the naive benchmark.

print(f'XGB Train : MAPE={train_mape_xgb:.2%}  WAPE={train_wape_xgb:.4f}  R²={train_r2_xgb:.4f}')
print(f'XGB Test  : MAPE={test_mape_xgb:.2%}  WAPE={test_wape_xgb:.4f}  sMAPE={test_smape_xgb:.2%}  R²={test_r2_xgb:.4f}')
print(f'XGB Overfit: {does_it_overfit_xgb:.2%}  |  Improvement over naive: {improvement_xgb:.1f} %')
print()

#Model selection
print('====== XGBoost h=24 — MODEL SELECTION')
if naive_mape <= test_mape_xgb:
    print('XGBoost does NOT beat the naive model. Not recommended for forecasts.')
elif does_it_overfit_xgb >= 0.1:
    print(f'XGBoost overfits ({does_it_overfit_xgb:.2%} gap). Not recommended for forecasts.')
else:
    print(
        f'*** XGBoost can be used for forecasting. ***\n'
        f'    MAPE={test_mape_xgb:.2%} | R²={test_r2_xgb:.4f} | '
        f'Improvement over naive: {improvement_xgb:.1f} %'
    )

# ────────────────────────────────────────────────────────────────────────────
# Model Comparison
# DNN + XGBoost only**
# The comparison table below aggregates all metrics and includes:
# - The naive benchmark for context.
# - Model selection result (beat naive + no overfit).
# ────────────────────────────────────────────────────────────────────────────

# ── Side-by-side plot: 2 models, first 7 test days
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
# One panel per model, side by side for direct visual comparison.

dates_plot  = datetime_index.iloc[T:T + n_test_hours]
actual_plot = df[target_h24].iloc[T:T + n_test_hours].values

all_preds = {
    'DNN':     forecasts['deep_network'][target_h24].iloc[T:T + n_test_hours].values.astype(float),
    'XGBoost': forecasts['xgboost'][target_h24].iloc[T:T + n_test_hours].values.astype(float),
}

for ax, (name, preds), color in zip(axes, all_preds.items(), ['orange', 'darkorange']):
    m, wv, sv, r2v = compute_metrics(actual_plot, preds)
    ax.plot(dates_plot[:zoom], actual_plot[:zoom],
            color='steelblue', linewidth=1.0, label='Actual', alpha=0.8)
    ax.plot(dates_plot[:zoom], preds[:zoom],
            color=color,       linewidth=1.0, label=name)
    ax.set_title(f'{name}\nMAPE={m:.2%} | WAPE={wv:.4f} | R²={r2v:.4f}', fontsize=10)
    ax.set_ylabel('MW')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # Each panel shows the first 7 test days (zoom = 168 hours).
    # Showing the full test period (1392 hours) on a 6-inch wide panel would make
    # individual forecast cycles unreadable.

plt.suptitle(
    'v25 — DNN vs XGBoost | Lagged Approach | 30 Features | First 7 Test Days',
    fontsize=12, y=1.02)
plt.tight_layout()
plt.show()
