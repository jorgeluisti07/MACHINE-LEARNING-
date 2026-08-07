#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic (non-oracle) weather experiment — ETESA Panama 2025 residual demand forecasting

The main pipeline's h+24 weather features use shift(-24) on the TRUE MERRA-2 value, not an
operational forecast — a disclosed "perfect-foresight" upper bound. This script drops those
four *_h24 features entirely (no fabricated forecast; no real NWP data available for this
period), so the perfect-foresight number has something honest to compare against. Everything
else — data loading, leakage-safe calibration/hydro rebuilds, hour-ending alignment, index
checks, lag features, DNN/XGBoost/Ensemble, weekly-naive/linear baselines — is identical to
MACHINE_LEARNING_RESIDUAL_DEMAND.py.

Kept leaner than the main script: no EDA plots, correlation heatmap, or diagnostic plots —
this only answers what dropping perfect-foresight weather costs.

Run: same requirements as the main script (see ../requirements.txt). Calls input() and the
live Renewables.ninja API — run from the repo root so it can find/write DEM2025.csv,
solar_eolica_hidro_horario_2025.csv, renewables_ninja_2025.csv (reused if present).

Output: results_summary_realistic.csv, forecasts_realistic.csv (distinct filenames so
running both scripts doesn't clobber either's output).
"""

import warnings

HORIZON = 24
LAG_1 = 24
LAG_2 = 312
LAG_3 = 144
MAX_LAG = LAG_2

import requests, json as _json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_rows",    None)
pd.set_option("display.max_columns", None)

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent='etesa_tfm')
place = input('Enter location (e.g. Penonomé, Coclé, Panama): ')
information = geolocator.geocode(place)
lat = information[1][0]
lon = information[1][1]
print(lat, lon)

token    = 'c4672deddb9f6acfa94e9cdbb34f6414fd51f255'
api_base   = 'https://www.renewables.ninja/api/'
url_solar  = api_base + 'data/pv'
url_wind   = api_base + 'data/wind'

args_solar = {
    'lat': lat, 'lon': lon,
    'date_from': '2025-01-01', 'date_to': '2025-12-31',
    'dataset':  'merra2',
    'capacity':     859000,
    'system_loss':  0.1,
    'tracking':     1,
    'tilt':         5,
    'azim':         180,
    'format':       'json',
    'local_time':   'true',
    'raw':          'true',
}

args_wind = {
    'lat': lat, 'lon': lon,
    'date_from': '2025-01-01', 'date_to': '2025-12-31',
    'capacity':  336000,
    'height':    150,
    'turbine':   'Vestas V80 2000',
    'format':    'json',
    'local_time': 'true',
    'raw':        'true',
}

s = requests.session()
s.headers = {'Authorization': 'Token ' + token}

r = s.get(url_solar, params=args_solar)
print('Solar status:', r.status_code)
with open('renewables_ninja_solar_raw.json', 'w') as _f:
    _f.write(r.text)
parsed_solar = _json.loads(r.text)
data = pd.read_json(_json.dumps(parsed_solar['data']), orient='index')
data['electricity'] = data['electricity'] / 1000
data['local_time'] = pd.to_datetime(data['local_time']).dt.tz_localize(None)
data = data.set_index('local_time')

r_wind = s.get(url_wind, params=args_wind)
with open('renewables_ninja_wind_raw.json', 'w') as _f:
    _f.write(r_wind.text)
parsed_wind = _json.loads(r_wind.text)
wind_data = pd.read_json(_json.dumps(parsed_wind['data']), orient='index')
wind_data['local_time'] = pd.to_datetime(wind_data['local_time']).dt.tz_localize(None)
wind_data = wind_data.set_index('local_time')
wind_data['electricity'] = wind_data['electricity'] / 1000

df_combined = pd.concat([
    data[['irradiance_direct','irradiance_diffuse','temperature','electricity']].rename(
        columns={'electricity': 'solar_mw'}),
    wind_data[['wind_speed','electricity']].rename(
        columns={'electricity': 'wind_mw'})
], axis=1).dropna()

df_combined.index = df_combined.index + pd.Timedelta(hours=1)
# Renewables.ninja labels each hourly value by the start of the interval (hour-beginning),
# while ETESA demand/generation label by the end (hour-ending). Shifted here to match, same
# fix and evidence as the main script (see MACHINE_LEARNING_RESIDUAL_DEMAND.py).

df_combined.to_csv('renewables_ninja_2025.csv')

# ── Load ETESA real-generation data from CSV ─────────────────────────────────────
real = pd.read_csv('solar_eolica_hidro_horario_2025.csv', index_col=0, parse_dates=True)
real.index = pd.to_datetime(real.index)
cols_to_drop = [c for c in real.columns if c in df_combined.columns]
df_combined  = df_combined.drop(columns=cols_to_drop).join(real, how='left')

# ── Load ETESA real demand from CSV ──────────────────────────────────────────────
demanda_raw = pd.read_csv('DEM2025.csv')
demanda_raw = demanda_raw.rename(columns={demanda_raw.columns[0]: 'fecha'})
demanda_long = demanda_raw.melt(id_vars=['fecha'], var_name='hora_str', value_name='demanda_mw')
demanda_long  = demanda_long.dropna(subset=['fecha'])
demanda_long['fecha_dt']  = pd.to_datetime(demanda_long['fecha'], errors='coerce')

_bad_dates = sorted(demanda_long.loc[demanda_long['fecha_dt'].isna(), 'fecha'].unique())
if _bad_dates:
    print(f'WARNING: dropping {len(_bad_dates)} unparseable date(s) from DEM2025.csv: {_bad_dates}')

demanda_long  = demanda_long.dropna(subset=['fecha_dt'])
demanda_long['hora_num'] = (
    demanda_long['hora_str'].str.extract(r'(\d+)').astype(float).fillna(0).astype(int)
)
# HOUR-ENDING convention (same fix as the main script — see its comments for the full
# derivation): H1 -> 01:00, ..., H24 -> next-day 00:00.
demanda_long['timestamp'] = (
    demanda_long['fecha_dt'] + pd.to_timedelta(demanda_long['hora_num'], unit='h')
)
demanda_long = demanda_long.set_index('timestamp')[['demanda_mw']].sort_index()
demanda_long['demanda_mw'] = pd.to_numeric(demanda_long['demanda_mw'], errors='coerce')

if 'demanda_mw' in df_combined.columns:
    df_combined = df_combined.drop(columns=['demanda_mw'])
df_combined = df_combined.join(demanda_long, how='left')
df_combined = df_combined[df_combined.index >= '2025-01-01 01:00:00']

df = df_combined
print('Columns:', df.columns.tolist())

# Index integrity (same check as the main script).
_idx = pd.DatetimeIndex(df.index)
_dupes = _idx[_idx.duplicated()]
assert _dupes.empty, f'Duplicate timestamps in the hourly index: {list(_dupes[:5])}'
_full = pd.date_range(_idx.min(), _idx.max(), freq='h')
_missing = _full.difference(_idx)
assert _missing.empty, f'{len(_missing)} missing hour(s) in the index.'
print(f'Index OK: {len(_idx):,} unique, gap-free hourly rows ({_idx.min()} → {_idx.max()}).')

# ── Leakage-safe calibration setup (same as main script — see its comments for the full
# leakage derivation this avoids). Raw ninja columns preserved; the actual calibration
# factor is fit per rolling window, training-only, in rebuild_calibration_features below.
df['irr_direct_ninja']  = df['irradiance_direct']
df['irr_diffuse_ninja'] = df['irradiance_diffuse']
df['solar_ninja']       = df['solar_mw']

df['solar_mw']  = df['solar_mw_real']
df['eolica_mw'] = df['eolica_mw_real']
df['hidro_mw']  = df['hidro_mw_real']

BASE_COLS = [
    'irradiance_direct','irradiance_diffuse','temperature',
    'solar_mw','wind_speed','eolica_mw','hidro_mw','demanda_mw',
    'irr_direct_ninja','irr_diffuse_ninja','solar_ninja',
]
df = df[BASE_COLS]
print(f'Base: {df.shape[1]} cols, {df.shape[0]} records.')

df.reset_index(inplace=True)

df['month'] = df['local_time'].dt.month
df['hour']  = df['local_time'].dt.hour
df['is_weekday'] = (df['local_time'].dt.dayofweek < 5).astype(int)
df['hour_weekday'] = df['hour'] * df['is_weekday']

_panama_holidays_2025 = [
    '2025-01-01', '2025-01-09', '2025-03-04', '2025-03-05', '2025-04-18',
    '2025-05-01', '2025-08-15', '2025-11-03', '2025-11-04', '2025-11-05',
    '2025-11-10', '2025-11-28', '2025-12-08', '2025-12-25',
]
_holiday_set = {pd.Timestamp(d).date() for d in _panama_holidays_2025}
df['is_holiday'] = df['local_time'].dt.date.map(lambda d: int(d in _holiday_set))

df['dow_sin'] = np.sin(2 * np.pi * df['local_time'].dt.dayofweek / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['local_time'].dt.dayofweek / 7)

datetime_index = df['local_time'].copy()
df = df.drop(columns=['local_time'])

print(f'Holidays 2025: {len(_panama_holidays_2025)} days → {df.is_holiday.sum()} hours marked')

# Target: residual demand
df['demanda_residual'] = df['demanda_mw'] - df['solar_mw'] - df['eolica_mw']

df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['residual_L312'] = df['demanda_residual'].shift(LAG_2)
df['residual_L144'] = df['demanda_residual'].shift(LAG_3)
df['residual_L48']  = df['demanda_residual'].shift(48)

# NO *_h24 weather features here — the entire point of this experiment. Raw ninja h24 shifts
# are kept below only so rebuild_calibration_features (shared with the main script) has a
# consistent signature — they're never added to cols_order.
df['irr_direct_ninja_h24']   = df['irr_direct_ninja'].shift(-HORIZON)
df['irr_diffuse_ninja_h24']  = df['irr_diffuse_ninja'].shift(-HORIZON)

# Hydro dispatch features (same as main script).
df['hidro_fraction_L24'] = (
    df['hidro_mw'].shift(LAG_1) /
    df['demanda_residual'].shift(LAG_1).replace(0, np.nan)
).clip(0, 1.5).fillna(0.5)

df['hidro_delta_L24'] = (
    df['hidro_mw'].shift(LAG_1) - df['hidro_mw'].shift(LAG_1 * 2)
).fillna(0)

_hidro_profile = df.groupby(['month', 'hour'])['hidro_mw'].transform('mean')
df['hidro_typical_h24'] = _hidro_profile.shift(-HORIZON)
df['hidro_anomaly_L24'] = (df['hidro_mw'] - _hidro_profile).shift(LAG_1)

df['demanda_residual_h24'] = df['demanda_residual'].shift(-HORIZON)

# 26 flat features + 1 target (30 in the main script, minus the 4 *_h24 weather features).
# Current (non-shifted) irradiance/temperature/wind_speed are kept — already-observed "now"
# conditions, legitimately available at forecast time.
cols_order = [
    'irradiance_direct', 'irradiance_diffuse', 'temperature',
    'solar_mw', 'wind_speed', 'eolica_mw', 'hidro_mw',
    'month', 'hour', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_weekday', 'hour_weekday', 'is_holiday', 'dow_sin', 'dow_cos',
    'demanda_residual',
    'residual_L48', 'residual_L312', 'residual_L144',
    'hidro_fraction_L24', 'hidro_delta_L24', 'hidro_typical_h24', 'hidro_anomaly_L24',
    'demanda_residual_h24',
]
NINJA_HELPERS = ['irr_direct_ninja', 'irr_diffuse_ninja', 'solar_ninja',
                  'irr_direct_ninja_h24', 'irr_diffuse_ninja_h24']
df             = df[cols_order + NINJA_HELPERS]
df             = df.iloc[MAX_LAG:-HORIZON].reset_index(drop=True)
datetime_index = datetime_index.iloc[MAX_LAG:-HORIZON].reset_index(drop=True)

origin_time = datetime_index
target_time = datetime_index + pd.Timedelta(hours=HORIZON)

print(f'Records:  {len(df):,} hourly ({len(df)/24:.0f} days)')
print(f'Features: {len(df.columns)-1} flat + 1 target = {len(df.columns)} columns '
      f'(26 model features + 5 ninja-helper plumbing columns)')

# Feature extraction (identical logic to the main script)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_percentage_error, mean_absolute_error,
                              mean_squared_error, r2_score)

def compute_metrics(actual, predicted):
    mape = mean_absolute_percentage_error(actual, predicted)
    wape = np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual)) + 1e-8)
    smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-8))
    r2 = r2_score(actual, predicted)
    return (mape, wape, smape, r2)

def rmse_mw(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))

def rebuild_hidro_profile_features(df_in, T):
    """Leakage-safe hydro profile rebuild — identical to the main script's function."""
    train_profile = df_in.iloc[:T].groupby(['month', 'hour'])['hidro_mw'].mean()
    train_mean    = df_in['hidro_mw'].iloc[:T].mean()
    keys          = zip(df_in['month'].to_numpy(), df_in['hour'].to_numpy())
    base          = pd.Series([train_profile.get(k, train_mean) for k in keys],
                              index=df_in.index, dtype=float)
    out = df_in.copy()
    out['hidro_typical_h24'] = base.shift(-HORIZON).fillna(base)
    out['hidro_anomaly_L24'] = (out['hidro_mw'] - base).shift(LAG_1).fillna(0.0)
    return out

def rebuild_calibration_features(df_in, T):
    """Leakage-safe solar calibration rebuild — identical to the main script's function.
    Only irradiance_direct/irradiance_diffuse (current, non-h24) actually matter here since
    this experiment doesn't use the _h24 weather features, but kept faithful to the main
    script's implementation for consistency."""
    train = df_in.iloc[:T]
    day   = train['solar_ninja'] > 0
    ratio = train.loc[day, 'solar_mw'] / train.loc[day, 'solar_ninja']
    k_by_hour = ratio.groupby(train.loc[day, 'hour']).mean()
    k_global  = float(ratio.mean()) if len(ratio) else 1.0
    k = df_in['hour'].map(k_by_hour).fillna(k_global).to_numpy()
    out = df_in.copy()
    out['irradiance_direct']  = (out['irr_direct_ninja'].to_numpy()  * k)
    out['irradiance_diffuse'] = (out['irr_diffuse_ninja'].to_numpy() * k)
    return out

target_h24   = 'demanda_residual_h24'
features_h24 = [c for c in cols_order if c != target_h24]
print(f'Model features: {len(features_h24)} (should be 26)')
assert len(features_h24) == 26, f'Feature count error: {len(features_h24)}'

def get_targets_features(df_in, T, scale=False):
    df_in = rebuild_calibration_features(df_in, T)
    df_in = rebuild_hidro_profile_features(df_in, T)

    Y_train = df_in[[target_h24]].iloc[:T].copy()
    X_train = df_in[features_h24].iloc[:T].copy()
    X_test  = df_in[features_h24].iloc[T:T + HORIZON].copy()

    if scale:
        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        X_train = pd.DataFrame(sx.transform(X_train), columns=X_train.columns)
        X_test  = pd.DataFrame(sx.transform(X_test),  columns=X_test.columns)
        Y_train = pd.DataFrame(sy.transform(Y_train), columns=Y_train.columns)
        return Y_train, X_train, X_test, sx, sy

    return Y_train, X_train, X_test

T = 7000
period_selected = len(df) - T
n_test_days  = period_selected // HORIZON
n_test_hours = n_test_days * HORIZON

print(f'Train:         {T:,} hours ({T/24:.0f} days)')
print(f'Test:          {period_selected:,} hours ({period_selected/24:.0f} days)')
print(f'Rolling days:  {n_test_days} iterations ({n_test_hours:,} hours in test)')
print(f'Test origin range (forecasts issued):  {origin_time.iloc[T]} → {origin_time.iloc[T + n_test_hours - 1]}')
print(f'Test target range (days being forecast): {target_time.iloc[T]} → {target_time.iloc[T + n_test_hours - 1]}')

naive_preds  = df['demanda_residual'].iloc[T:T + n_test_hours].values
naive_actual = df[target_h24].iloc[T:T + n_test_hours].values
naive_mape, naive_wape, naive_smape, naive_r2 = compute_metrics(naive_actual, naive_preds)
print(f'\nNaive: MAPE={naive_mape:.2%}  WAPE={naive_wape:.4f}  sMAPE={naive_smape:.2%}  R²={naive_r2:.4f}')

weekly_naive_preds = df['residual_L144'].iloc[T:T + n_test_hours].values
weekly_naive_mape, weekly_naive_wape, weekly_naive_smape, weekly_naive_r2 = \
    compute_metrics(naive_actual, weekly_naive_preds)
print(f'Weekly-naive: MAPE={weekly_naive_mape:.2%}  WAPE={weekly_naive_wape:.4f}  '
      f'sMAPE={weekly_naive_smape:.2%}  R²={weekly_naive_r2:.4f}')

# DNN — Rolling h=24 (same architecture/training regime as the main script)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

def set_random_seeds(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

forecasts = {
    'deep_network': pd.DataFrame(data=np.nan, columns=[target_h24], index=target_time),
    'xgboost':      pd.DataFrame(data=np.nan, columns=[target_h24], index=target_time),
}
predictions = {'deep_network': [], 'xgboost': []}

for day_idx in tqdm(range(n_test_days), desc='DNN rolling h=24'):
    set_random_seeds(42)
    T_day = T + day_idx * HORIZON
    Y_train, X_train, X_test, scaler_x, scaler_y = get_targets_features(df_in=df, T=T_day, scale=True)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    model.fit(
        X_train, Y_train, epochs=100, batch_size=256, validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0,
    )

    df_Y_pred1 = pd.DataFrame(
        scaler_y.inverse_transform(model.predict(X_train, verbose=0)),
        columns=[target_h24], index=target_time[:T_day])
    df_Y_pred2 = pd.DataFrame(
        scaler_y.inverse_transform(model.predict(X_test, verbose=0)),
        columns=[target_h24], index=target_time[T_day:T_day + HORIZON])

    forecasts['deep_network'].iloc[T_day:T_day + HORIZON] = df_Y_pred2[target_h24].values.reshape(-1, 1)
    predictions['deep_network'].append(df_Y_pred1)

actual_test = df[target_h24].iloc[T:T + n_test_hours].values
pred_test_dnn = forecasts['deep_network'][target_h24].iloc[T:T + n_test_hours].values.astype(float)
test_mape_dnn, test_wape_dnn, test_smape_dnn, test_r2_dnn = compute_metrics(actual_test, pred_test_dnn)
print(f'\nDNN Test: MAPE={test_mape_dnn:.2%}  WAPE={test_wape_dnn:.4f}  '
      f'sMAPE={test_smape_dnn:.2%}  R²={test_r2_dnn:.4f}')

# XGBoost — Rolling h=24 (same P5 sweep protocol as the main script: validation-only,
# T window, before the test period)

from xgboost import XGBRegressor

_Y_sweep, _X_sweep, _ = get_targets_features(df_in=df, T=T, scale=False)
_val_size_sweep = max(24, round(len(_X_sweep) * 0.10))
_X_val_sweep = _X_sweep.iloc[-_val_size_sweep:]
_y_val_sweep = _Y_sweep.values.ravel()[-_val_size_sweep:]
_X_tr_sweep  = _X_sweep.iloc[:-_val_size_sweep]
_y_tr_sweep  = _Y_sweep.values.ravel()[:-_val_size_sweep]

XGB_PARAM_GRID = [
    (5, 1,  0.1, 1.0, 0.8, 0.8),
    (4, 1,  0.1, 1.0, 0.8, 0.8),
    (4, 5,  0.3, 2.0, 0.8, 0.8),
    (4, 5,  0.3, 2.0, 0.7, 0.7),
    (3, 5,  0.3, 2.0, 0.7, 0.7),
    (4, 10, 0.5, 3.0, 0.7, 0.7),
]
_sweep_rows = []
for _md, _mcw, _ra, _rl, _ss, _cs in XGB_PARAM_GRID:
    _m = XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=_md, min_child_weight=_mcw,
        subsample=_ss, colsample_bytree=_cs, reg_alpha=_ra, reg_lambda=_rl,
        early_stopping_rounds=20, random_state=42, verbosity=0, n_jobs=-1,
    )
    _m.fit(_X_tr_sweep, _y_tr_sweep, eval_set=[(_X_val_sweep, _y_val_sweep)], verbose=False)
    _val_mape   = mean_absolute_percentage_error(_y_val_sweep, _m.predict(_X_val_sweep)) * 100
    _train_mape = mean_absolute_percentage_error(_y_tr_sweep, _m.predict(_X_tr_sweep)) * 100
    _sweep_rows.append({
        'max_depth': _md, 'min_child_weight': _mcw, 'reg_alpha': _ra, 'reg_lambda': _rl,
        'subsample': _ss, 'colsample_bytree': _cs,
        'val_mape': _val_mape, 'train_mape': _train_mape, 'gap': _val_mape - _train_mape,
    })

_sweep_df = pd.DataFrame(_sweep_rows).sort_values('val_mape').reset_index(drop=True)
print('\nXGBoost hyperparameter sweep (validation-only, T window):')
print(_sweep_df.round(3).to_string())

_best_row = _sweep_df.iloc[0]
XGB_PARAMS = dict(
    max_depth=int(_best_row['max_depth']), min_child_weight=int(_best_row['min_child_weight']),
    reg_alpha=float(_best_row['reg_alpha']), reg_lambda=float(_best_row['reg_lambda']),
    subsample=float(_best_row['subsample']), colsample_bytree=float(_best_row['colsample_bytree']),
)
print(f'Selected XGBoost config: {XGB_PARAMS}')

for day_idx in tqdm(range(n_test_days), desc='XGBoost rolling h=24'):
    T_day = T + day_idx * HORIZON
    Y_train_xgb, X_train_xgb, X_test_xgb = get_targets_features(df_in=df, T=T_day, scale=False)

    _val_size = max(24, round(len(X_train_xgb) * 0.10))
    model_xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.03, early_stopping_rounds=20,
        random_state=42, verbosity=0, n_jobs=-1, **XGB_PARAMS,
    )
    model_xgb.fit(
        X_train_xgb.iloc[:-_val_size], Y_train_xgb.values.ravel()[:-_val_size],
        eval_set=[(X_train_xgb.iloc[-_val_size:], Y_train_xgb.values.ravel()[-_val_size:])],
        verbose=False,
    )

    df_Y_pred1_xgb = pd.DataFrame(
        model_xgb.predict(X_train_xgb), columns=[target_h24], index=target_time[:T_day])
    df_Y_pred2_xgb = pd.DataFrame(
        model_xgb.predict(X_test_xgb), columns=[target_h24], index=target_time[T_day:T_day + HORIZON])

    forecasts['xgboost'].iloc[T_day:T_day + HORIZON] = df_Y_pred2_xgb[target_h24].values.reshape(-1, 1)
    predictions['xgboost'].append(df_Y_pred1_xgb)

pred_test_xgb = forecasts['xgboost'][target_h24].iloc[T:T + n_test_hours].values.astype(float)
test_mape_xgb, test_wape_xgb, test_smape_xgb, test_r2_xgb = compute_metrics(actual_test, pred_test_xgb)
print(f'\nXGBoost Test: MAPE={test_mape_xgb:.2%}  WAPE={test_wape_xgb:.4f}  '
      f'sMAPE={test_smape_xgb:.2%}  R²={test_r2_xgb:.4f}')

importances = dict(zip(features_h24, model_xgb.feature_importances_))
top10 = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])
print(f'\nTop 10 features (no h+24 weather available):\n{top10}')

# Linear regression baseline

from sklearn.linear_model import LinearRegression

linear_preds = np.full(n_test_hours, np.nan)
for day_idx in tqdm(range(n_test_days), desc='Linear rolling h=24'):
    T_day = T + day_idx * HORIZON
    Y_train_lin, X_train_lin, X_test_lin = get_targets_features(df_in=df, T=T_day, scale=False)
    lin_model = LinearRegression().fit(X_train_lin, Y_train_lin)
    linear_preds[day_idx * HORIZON:(day_idx + 1) * HORIZON] = \
        lin_model.predict(X_test_lin).reshape(-1)

linear_mape, linear_wape, linear_smape, linear_r2 = compute_metrics(naive_actual, linear_preds)
print(f'Linear regression: MAPE={linear_mape:.2%}  WAPE={linear_wape:.4f}  '
      f'sMAPE={linear_smape:.2%}  R²={linear_r2:.4f}')

# Results summary

pred_ens = 0.5 * (pred_test_dnn + pred_test_xgb)

_summary_rows = {
    'DNN':          pred_test_dnn,
    'XGBoost':      pred_test_xgb,
    'Ensemble':     pred_ens,
    'Linear':       linear_preds.astype(float),
    'Naive':        naive_preds.astype(float),
    'Weekly-Naive': weekly_naive_preds.astype(float),
}

results_summary = pd.DataFrame(
    [
        {
            'MAPE_%':  compute_metrics(actual_test, _p)[0] * 100,
            'WAPE_%':  compute_metrics(actual_test, _p)[1] * 100,
            'MAE_MW':  mean_absolute_error(actual_test, _p),
            'RMSE_MW': rmse_mw(actual_test, _p),
        }
        for _p in _summary_rows.values()
    ],
    index=list(_summary_rows.keys()),
).round({'MAPE_%': 2, 'WAPE_%': 2, 'MAE_MW': 1, 'RMSE_MW': 1})

print(f'\n=== REALISTIC (non-oracle weather) results, {n_test_days} rolling days ===')
print('No irradiance_direct_h24 / irradiance_diffuse_h24 / temperature_h24 / wind_speed_h24 —')
print('compare against results_summary.csv from the main (oracle) pipeline for the same period.')
print(results_summary.to_string())

results_summary.to_csv('results_summary_realistic.csv')

_test_origin_dates = origin_time.iloc[T:T + n_test_hours].reset_index(drop=True)
_test_target_dates = target_time.iloc[T:T + n_test_hours].reset_index(drop=True)
forecasts_out = pd.DataFrame({
    'origin_time': _test_origin_dates,
    'target_time': _test_target_dates,
    'actual':   actual_test,
    'dnn':      pred_test_dnn,
    'xgboost':  pred_test_xgb,
    'ensemble': pred_ens,
    'naive':    naive_preds.astype(float),
}).set_index('target_time')
forecasts_out.to_csv('forecasts_realistic.csv')
print(f"\nSaved 'results_summary_realistic.csv' and 'forecasts_realistic.csv' "
      f"({len(forecasts_out)} rows) for this run.")
