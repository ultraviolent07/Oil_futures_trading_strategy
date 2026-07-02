import os, sys, warnings, requests, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import numba
from numba import jit, prange
import yfinance as yf
from fredapi import Fred

START_DATE   = "2019-01-01"
END_DATE     = datetime.today().strftime("%Y-%m-%d")
FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"
CP_HORMUZ    = "chokepoint6"
CP_BAB       = "chokepoint2"
CP_SUEZ      = "chokepoint1"
CP_CAPE      = "chokepoint14"

print("Loading PortWatch CSV...")
df_pw = pd.read_csv('Dataset/portwatch_chokepoints.csv')
if 'date' not in df_pw.columns and 'Date' in df_pw.columns:
    df_pw.rename(columns={'Date': 'date'}, inplace=True)
df_pw['date'] = pd.to_datetime(df_pw['date']).dt.tz_localize(None)

chokepoints = [CP_HORMUZ, CP_BAB, CP_SUEZ, CP_CAPE]
dfs = []
for cp in chokepoints:
    d = df_pw[df_pw['portid'] == cp].copy()
    d.set_index('date', inplace=True)
    d = d.sort_index()
    series = d['n_tanker']
    series.name = cp
    dfs.append(series)

df_ships = pd.concat(dfs, axis=1)
df_ships = df_ships.resample('D').sum().fillna(0)

# 7-day rolling mean
df_ships_7d = df_ships.rolling(window=7, min_periods=1).mean()

for cp in chokepoints:
    mean_7d = df_ships_7d[cp].expanding().mean()
    std_7d = df_ships_7d[cp].expanding().std()
    df_ships[f'{cp}_z'] = (df_ships_7d[cp] - mean_7d) / (std_7d + 1e-9)

df_ships = df_ships.dropna()

print("Downloading Brent Crude Data...")
oil = yf.download("BZ=F", start=START_DATE, end=END_DATE)
df_oil = pd.DataFrame()
df_oil['brent_close'] = oil['Close']
df_oil['brent_high'] = oil['High']
df_oil['brent_low'] = oil['Low']
df_oil.index = pd.to_datetime(df_oil.index).tz_localize(None)
df_oil = df_oil.dropna()

print("Downloading VIX Data...")
fred = Fred(api_key=FRED_API_KEY)
vix = fred.get_series('VIXCLS', start_date=START_DATE)
df_vix = pd.DataFrame({'vix': vix})
df_vix.index = pd.to_datetime(df_vix.index).tz_localize(None)
df_vix = df_vix.dropna()

df_master = df_oil.join(df_vix, how='outer')
df_master['vix'] = df_master['vix'].ffill()
df_master = df_master.dropna()

df_master = df_master.join(df_ships[[f'{c}_z' for c in chokepoints]], how='left')
df_master.ffill(inplace=True)
df_master.dropna(inplace=True)

df_master['sma_50'] = df_master['brent_close'].rolling(50).mean()

delta = df_master['brent_close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-9)
df_master['rsi'] = 100 - (100 / (1 + rs))

high_low = df_master['brent_high'] - df_master['brent_low']
high_close = np.abs(df_master['brent_high'] - df_master['brent_close'].shift())
low_close = np.abs(df_master['brent_low'] - df_master['brent_close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df_master['atr'] = tr.rolling(14).mean()

df_master.dropna(inplace=True)

df_master['vix_median_60'] = df_master['vix'].rolling(60).median()
df_master['vix_regime'] = (df_master['vix'] > df_master['vix_median_60']).astype(int)
df_master.dropna(inplace=True)

@jit(nopython=True)
def compute_cdi_numba(hormuz_z, bab_z, rerouting, persistence, vix_regime, w1=0.4, w2=0.3, w3=0.2, w4=0.1):
    n = len(hormuz_z)
    cdi = np.zeros(n)
    
    for i in range(n):
        h = min(max(-hormuz_z[i] / 4, -1.0), 1.0)
        b = min(max(-bab_z[i] / 4, -1.0), 1.0)
        r = rerouting[i]
        p = min(persistence[i] / 7.0, 1.0)
        raw = w1*h + w2*b + w3*r + w4*p
        cdi[i] = raw * vix_regime[i]
            
    return cdi

rerouting_proxy = np.where((df_master[f'{CP_BAB}_z'] < -1.5) & (df_master[f'{CP_CAPE}_z'] > 1.0), 1.0, 0.0)
persistence_proxy = np.where((df_master[f'{CP_HORMUZ}_z'].rolling(10).mean() < -1.0), 1.0, 0.0)

df_master['cdi'] = compute_cdi_numba(
    df_master[f'{CP_HORMUZ}_z'].values,
    df_master[f'{CP_BAB}_z'].values,
    rerouting_proxy,
    persistence_proxy,
    df_master['vix_regime'].fillna(0).values
)

df_master.fillna(0, inplace=True)
df_master.to_csv('antigravit-quant-prof/df_master_cached.csv')
print("Successfully generated and cached df_master_cached.csv!")
