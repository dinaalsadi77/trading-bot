import pandas as pd
import numpy as np
import os

INPUT_FILE  = "forex_labeled/USDJPY_M15_labeled.csv"
OUTPUT_FILE = "forex_labeled/USDJPY_M15_labeled.csv"  # overwrite in place

df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
c = df["close"]
h = df["high"]
l = df["low"]
o = df["open"]

# RSI 14
delta = c.diff()
gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
df["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# MACD
ema12 = c.ewm(span=12, adjust=False).mean()
ema26 = c.ewm(span=26, adjust=False).mean()
df["macd"]        = ema12 - ema26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"]   = df["macd"] - df["macd_signal"]

# EMA cross (ema9 > ema21)
ema9  = c.ewm(span=9,  adjust=False).mean()
ema21 = c.ewm(span=21, adjust=False).mean()
df["ema_cross"] = (ema9 > ema21).astype(int)

# ATR 14
tr = pd.concat([
    h - l,
    (h - c.shift()).abs(),
    (l - c.shift()).abs()
], axis=1).max(axis=1)
df["atr_14"] = tr.ewm(span=14, adjust=False).mean()

# Bollinger Bands
sma20  = c.rolling(20).mean()
std20  = c.rolling(20).std()
bb_up  = sma20 + 2 * std20
bb_lo  = sma20 - 2 * std20
df["bb_width"] = (bb_up - bb_lo) / sma20
df["bb_pct"]   = (c - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)

# Returns
df["return_1"]  = c.pct_change(1)
df["return_5"]  = c.pct_change(5)
df["return_10"] = c.pct_change(10)

# Rolling volatility
df["volatility_10"] = df["return_1"].rolling(10).std()
df["volatility_20"] = df["return_1"].rolling(20).std()

# Stochastic K/D
low14  = l.rolling(14).min()
high14 = h.rolling(14).max()
df["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
df["stoch_d"] = df["stoch_k"].rolling(3).mean()

df.to_csv(OUTPUT_FILE, index=False)
print(f"Done. Saved {len(df)} rows with indicators to {OUTPUT_FILE}")
print(f"Columns: {list(df.columns)}")