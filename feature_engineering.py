import os
import pandas as pd
import numpy as np

# ── pip install pandas numpy ta ──────────────────────────────
try:
    import ta
except ImportError:
    print("Installing 'ta' library...")
    os.system("pip install ta")
    import ta

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR  = "forex_data"      # folder with raw CSVs from Phase 1
OUTPUT_DIR = "forex_features"  # folder for feature-enriched CSVs
PAIRS      = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================

def add_features(df):
    """
    Add all technical indicator features to a OHLCV DataFrame.
    Returns the enriched DataFrame with NaN rows dropped.
    """

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── 1. RSI (Relative Strength Index) ────────────────────
    # Measures overbought/oversold momentum (0–100)
    # > 70 = overbought, < 30 = oversold
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # ── 2. MACD (Moving Average Convergence Divergence) ─────
    # Trend direction and momentum
    macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_indicator.macd()          # MACD line
    df["macd_signal"] = macd_indicator.macd_signal()   # Signal line
    df["macd_hist"]   = macd_indicator.macd_diff()     # Histogram (MACD - Signal)

    # ── 3. EMA Crossover ─────────────────────────────────────
    # Difference between fast and slow EMA
    # Positive = uptrend, Negative = downtrend
    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema9"]       = ema9
    df["ema21"]      = ema21
    df["ema50"]      = ema50
    df["ema_cross"]  = ema9 - ema21    # key crossover signal

    # ── 4. ATR (Average True Range) ──────────────────────────
    # Measures volatility — used for stop-loss sizing
    df["atr_14"] = ta.volatility.AverageTrueRange(
        high, low, close, window=14
    ).average_true_range()

    # ── 5. Bollinger Bands ───────────────────────────────────
    # Volatility bands around price
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_width"]  = bb.bollinger_wband()   # band width = volatility measure
    df["bb_pct"]    = bb.bollinger_pband()   # % position within bands (0–1)

    # ── 6. Price Returns ─────────────────────────────────────
    # Percentage change over N periods
    df["return_1"]  = close.pct_change(1)    # 1-period return
    df["return_5"]  = close.pct_change(5)    # 5-period return
    df["return_10"] = close.pct_change(10)   # 10-period return

    # ── 7. Volatility (Rolling Std of Returns) ───────────────
    # Rolling standard deviation of 1-period returns
    df["volatility_10"] = df["return_1"].rolling(window=10).std()
    df["volatility_20"] = df["return_1"].rolling(window=20).std()

    # ── 8. Stochastic Oscillator ─────────────────────────────
    # Momentum indicator comparing close to recent range
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14)
    df["stoch_k"] = stoch.stoch()           # %K line
    df["stoch_d"] = stoch.stoch_signal()    # %D line (signal)

    # ── Drop NaN rows (from indicator warmup period) ─────────
    df = df.dropna().reset_index(drop=True)

    return df


# ============================================================
# PROCESS ALL PAIRS
# ============================================================

print("=" * 55)
print(" Feature Engineering — Phase 2")
print("=" * 55)

feature_cols = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "ema9", "ema21", "ema50", "ema_cross",
    "atr_14", "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "return_1", "return_5", "return_10",
    "volatility_10", "volatility_20",
    "stoch_k", "stoch_d"
]

for pair in PAIRS:
    input_path  = os.path.join(INPUT_DIR,  f"{pair}_H1.csv")
    output_path = os.path.join(OUTPUT_DIR, f"{pair}_H1_features.csv")

    print(f"\nProcessing {pair}...")

    # Load raw data
    if not os.path.exists(input_path):
        print(f"  SKIPPED — file not found: {input_path}")
        continue

    df = pd.read_csv(input_path, parse_dates=["datetime"])
    print(f"  Loaded   : {len(df):,} raw candles")

    # Add features
    df = add_features(df)
    print(f"  Features : {len(feature_cols)} indicators added")
    print(f"  Remaining: {len(df):,} candles (after dropping NaN warmup rows)")

    # Save
    df.to_csv(output_path, index=False)
    print(f"  Saved    : {output_path}")

# ============================================================
# PREVIEW
# ============================================================

print("\n" + "=" * 55)
print(" Preview — EURUSD first 3 rows")
print("=" * 55)

preview_path = os.path.join(OUTPUT_DIR, "EURUSD_H1_features.csv")
if os.path.exists(preview_path):
    df_preview = pd.read_csv(preview_path)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(df_preview[["datetime", "close", "rsi_14", "macd", "ema_cross", "atr_14", "bb_width", "return_1"]].head(3).to_string(index=False))
    print(f"\nTotal columns : {len(df_preview.columns)}")
    print(f"Total rows    : {len(df_preview):,}")

print("\n" + "=" * 55)
print(" Done! Check the forex_features folder.")
print(" Next: Phase 3 — Labeling (BUY / SELL / HOLD)")
print("=" * 55)