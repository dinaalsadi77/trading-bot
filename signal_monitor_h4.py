"""
================================================================
 Step 12 — Live Signal Monitor (Paper Trading)  ·  H4 Edition
================================================================
 Fires on every H4 bar close: 00:00, 04:00, 08:00, 12:00,
 16:00, 20:00 UTC.  Waits for the exact close + 5 s buffer
 so the bar is fully formed on the broker side.

 Usage:
   - MetaTrader 5 must be open and logged in
   - Run: python signal_monitor_h4.py
   - Keep the terminal window open
   - Take every signal on your MT5 demo account manually

 Pairs monitored : EURUSD, GBPUSD, AUDUSD, USDJPY
 Risk per trade  : 0.5% of account
 R:R target      : 1:2.5
 Stop loss       : 2.5× ATR14
 Timeframe       : H4  (bar close every 4 hours)

 Log file: paper_trading_log.csv
================================================================
"""

import os, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── auto-install missing packages ───────────────────────────────
def install(pkg):
    os.system(f"pip install {pkg} -q --break-system-packages")

try:
    import MetaTrader5 as mt5
except ImportError:
    install("MetaTrader5"); import MetaTrader5 as mt5

try:
    from xgboost import XGBClassifier
except ImportError:
    install("xgboost"); from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except ImportError:
    install("lightgbm"); import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    install("catboost")
    try:
        from catboost import CatBoostClassifier
        CATBOOST_AVAILABLE = True
    except Exception:
        CATBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
except ImportError:
    install("imbalanced-learn")
    from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ================================================================
# CONFIG
# ================================================================

PAIRS              = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
LOG_FILE           = "paper_trading_log.csv"
TIMEFRAME          = None          # set in main() after mt5 import
TIMEFRAME_MINUTES  = 240           # H4 = 240 minutes
BAR_CLOSE_BUFFER_S = 5             # seconds after bar close before checking
LOOKBACK_BARS      = 600           # H4 bars to fetch  (~100 days)
TRAIN_BARS         = 450           # bars for model training

# Trade settings
RISK_PCT   = 0.005    # 0.5%
RR_RATIO   = 2.5
ATR_MULT   = 2.5

# v10 model settings
ATR_MULTIPLIER      = 1.8
FORWARD_BARS        = 10
TOP_N_FEATURES      = 20
BUY_FLOOR           = 0.62
SELL_CEIL           = 0.38
ADX_MODERATE        = 25
ADX_STRONG          = 28
DE_TREND_THRESHOLD  = 0.35
DE_WINDOW           = 20
MIN_RANGING_TRAIN   = 80
RECENCY_HALF_LIFE   = 200

BASE_FEATURES = [
    "rsi_14","macd","macd_signal","macd_hist",
    "ema_cross","atr_14","bb_width","bb_pct",
    "return_1","return_5","return_10",
    "volatility_10","volatility_20","stoch_k","stoch_d",
]
SESSION_FEATURES = [
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "is_london","is_ny","is_overlap",
]
V2_FEATURES = [
    "adx_14","cci_20","williams_r","price_vs_ema200",
    "candle_body_pct","hl_ratio","swing_dist_high","swing_dist_low",
]
JPY_FEATURES = [
    "jpy_momentum_diff","volatility_regime","price_vs_ema50","trend_alignment",
]
V5_FEATURES = ["directional_efficiency"]
FEATURE_COLS = BASE_FEATURES + SESSION_FEATURES + V2_FEATURES + JPY_FEATURES + V5_FEATURES

# ================================================================
# TIMING  — wait for next H4 bar close
# ================================================================

def seconds_until_next_bar_close(tf_minutes, buffer_s=BAR_CLOSE_BUFFER_S):
    """
    Returns how many seconds to sleep so we wake up exactly
    `buffer_s` seconds after the next H4 bar closes.
    H4 bars close at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.
    We use local time here — if your broker uses UTC offset, adjust below.
    """
    now            = datetime.now()
    total_minutes  = now.hour * 60 + now.minute
    minutes_into   = total_minutes % tf_minutes          # minutes elapsed in current bar
    minutes_left   = tf_minutes - minutes_into           # minutes until next close
    seconds_left   = minutes_left * 60 - now.second      # fine-tune with seconds
    return max(0, seconds_left) + buffer_s


def wait_for_bar_close():
    sleep_s    = seconds_until_next_bar_close(TIMEFRAME_MINUTES)
    wake_time  = datetime.now() + timedelta(seconds=sleep_s)
    hours_left = sleep_s / 3600
    print(f"\n  ⏳ Next H4 bar closes at {wake_time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"({hours_left:.2f} h from now)")
    print(f"  (Ctrl+C to stop)\n")
    time.sleep(sleep_s)

# ================================================================
# FEATURE ENGINEERING
# ================================================================

def build_features(df):
    df = df.copy()
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]

    # RSI 14
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["rsi_14"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # EMA cross (9/21)
    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    df["ema_cross"] = (ema9 > ema21).astype(int)

    # ATR 14
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(span=14, adjust=False).mean()

    # Bollinger Bands (20)
    sma   = c.rolling(20).mean()
    std   = c.rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    df["bb_width"] = ((upper - lower) / sma.replace(0, np.nan)).fillna(0)
    df["bb_pct"]   = ((c - lower) / (upper - lower).replace(0, np.nan)).fillna(0.5)

    # Returns
    df["return_1"]  = c.pct_change(1).fillna(0)
    df["return_5"]  = c.pct_change(5).fillna(0)
    df["return_10"] = c.pct_change(10).fillna(0)
    ret = c.pct_change()
    df["volatility_10"] = ret.rolling(10).std().fillna(0)
    df["volatility_20"] = ret.rolling(20).std().fillna(0)

    # Stochastic (14,3)
    ll14 = l.rolling(14).min()
    hh14 = h.rolling(14).max()
    k    = ((c - ll14) / (hh14 - ll14).replace(0, np.nan) * 100).fillna(50)
    df["stoch_k"] = k
    df["stoch_d"] = k.rolling(3).mean().fillna(50)

    # Session features (hour / day-of-week)
    dt = pd.to_datetime(df["datetime"])
    hr = dt.dt.hour
    dw = dt.dt.dayofweek
    df["hour_sin"]   = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hr / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * dw / 5)
    df["dow_cos"]    = np.cos(2 * np.pi * dw / 5)
    df["is_london"]  = ((hr >= 8)  & (hr < 17)).astype(int)
    df["is_ny"]      = ((hr >= 13) & (hr < 22)).astype(int)
    df["is_overlap"] = ((hr >= 13) & (hr < 17)).astype(int)

    # ADX 14
    n    = len(df)
    hi   = h.values; lo = l.values; cl = c.values
    tr_a = np.zeros(n); pdm = np.zeros(n); ndm = np.zeros(n)
    for i in range(1, n):
        tr_a[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
        up       = hi[i] - hi[i-1]
        dn       = lo[i-1] - lo[i]
        pdm[i]   = max(up, 0) if up > dn else 0
        ndm[i]   = max(dn, 0) if dn > up else 0
    str_v = pd.Series(tr_a).ewm(span=14, adjust=False).mean().values
    spdm  = pd.Series(pdm).ewm(span=14, adjust=False).mean().values
    sndm  = pd.Series(ndm).ewm(span=14, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(str_v > 0, 100 * spdm / str_v, 0)
        ndi = np.where(str_v > 0, 100 * sndm / str_v, 0)
        dx  = np.where((pdi + ndi) > 0,
                       100 * np.abs(pdi - ndi) / (pdi + ndi), 0)
    df["adx_14"] = pd.Series(dx).ewm(span=14, adjust=False).mean().values

    # CCI 20
    tp     = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad    = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci_20"] = np.where(mad > 0, (tp - sma_tp) / (0.015 * mad), 0)

    # Williams %R 14
    hh14w = h.rolling(14).max()
    ll14w = l.rolling(14).min()
    df["williams_r"] = np.where(
        (hh14w - ll14w) > 0, -100 * (hh14w - c) / (hh14w - ll14w), -50)

    # EMA 200 distance
    ema200 = c.ewm(span=200, adjust=False).mean()
    df["price_vs_ema200"] = (c - ema200) / ema200 * 100

    # Candle body / HL ratio
    df["candle_body_pct"] = ((c - o).abs() /
                             (h - l).replace(0, np.nan)).fillna(0.5)
    df["hl_ratio"]        = ((h - l) /
                             df["atr_14"].replace(0, np.nan)).fillna(1.0)

    # Swing distances (20-bar rolling high/low)
    rh = c.rolling(20).max()
    rl = c.rolling(20).min()
    df["swing_dist_high"] = (rh - c) / c * 100
    df["swing_dist_low"]  = (c - rl) / c * 100

    # JPY-style momentum diff (RSI14 vs RSI50)
    d50  = c.diff()
    g50  = d50.clip(lower=0).ewm(span=50, adjust=False).mean()
    l50  = (-d50.clip(upper=0)).ewm(span=50, adjust=False).mean()
    rsi50 = (100 - 100 / (1 + g50 / l50.replace(0, np.nan))).fillna(50)
    df["jpy_momentum_diff"] = df["rsi_14"] - rsi50

    # Volatility regime (vol5 / vol20)
    vol5  = ret.rolling(5).std()
    vol20 = ret.rolling(20).std()
    df["volatility_regime"] = (vol5 / vol20.replace(0, np.nan)).fillna(1.0).clip(0, 5)

    # EMA 50 distance & trend alignment (20/50/200)
    ema50 = c.ewm(span=50,  adjust=False).mean()
    ema20 = c.ewm(span=20,  adjust=False).mean()
    df["price_vs_ema50"]  = (c - ema50) / ema50 * 100
    df["trend_alignment"] = ((ema20 > ema50).astype(int) +
                              (ema50 > ema200).astype(int) - 1).astype(float)

    # Directional efficiency
    net  = ret.rolling(DE_WINDOW).sum().abs()
    path = ret.abs().rolling(DE_WINDOW).sum()
    df["directional_efficiency"] = (
        net / path.replace(0, np.nan)).fillna(0.5).clip(0, 1)

    # ── cast ALL feature columns to float64 (prevents isnan TypeError) ──
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df

# ================================================================
# REGIME + SETUP
# ================================================================

def _f(row, key, default=np.nan):
    """Safe float extraction from dict or Series row."""
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def classify_regime(row):
    de  = _f(row, "directional_efficiency")
    adx = _f(row, "adx_14")
    ta  = _f(row, "trend_alignment")

    has_de  = not np.isnan(de)
    has_adx = not np.isnan(adx)
    has_ta  = not np.isnan(ta)

    if has_de and has_adx and has_ta:
        rule1 = de > DE_TREND_THRESHOLD and adx > ADX_STRONG
        rule2 = ta != 0 and adx > ADX_MODERATE
        return "trending" if (rule1 or rule2) else "ranging"
    elif has_adx:
        return "trending" if adx > ADX_STRONG else "ranging"
    return "ranging"


def is_setup(row, regime):
    if regime == "trending":
        near_ema50   = abs(_f(row, "price_vs_ema50",  999)) < 0.5
        strong_trend = _f(row, "adx_14",              0)   > ADX_MODERATE
        aligned_emas = _f(row, "trend_alignment",     0)   != 0
        vol_ok       = _f(row, "volatility_regime",   999) < 2.0
        return near_ema50 and strong_trend and aligned_emas and vol_ok
    else:
        rsi    = _f(row, "rsi_14",           50)
        bb     = _f(row, "bb_pct",            0.5)
        vol_ok = _f(row, "volatility_regime", 999) < 2.0
        return (rsi < 25 or rsi > 75 or bb < 0.03 or bb > 0.97) and vol_ok

# ================================================================
# MODEL
# ================================================================

def _is_constant(arr, tol=1e-6):
    return float(np.std(arr)) < tol


def recency_weights(n):
    pos = np.arange(n)
    w   = np.power(2.0, (pos - (n - 1)) / RECENCY_HALF_LIFE)
    return w / w.sum() * n


def safe_balance(X, y):
    counts   = pd.Series(y).value_counts()
    minority = int(counts.min())
    if minority >= 10:
        try:
            sm = SMOTE(random_state=42, k_neighbors=min(5, minority - 1))
            return sm.fit_resample(X, y)
        except Exception:
            pass
    if minority >= 3:
        try:
            return RandomOverSampler(random_state=42).fit_resample(X, y)
        except Exception:
            pass
    return X, y


def select_features(X, y, feature_cols, top_n):
    top_n = min(top_n, len(feature_cols), max(1, len(X) // 10))
    try:
        rf = RandomForestClassifier(
            n_estimators=80, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=feature_cols)
        return imp.nlargest(top_n).index.tolist()
    except Exception:
        return feature_cols[:top_n]


def train_and_predict(X_train, y_train, X_pred, feature_cols, regime):
    sel = select_features(X_train, y_train, feature_cols, TOP_N_FEATURES)
    fi  = [feature_cols.index(f) for f in sel]

    sc  = StandardScaler()
    Xts = sc.fit_transform(X_train[:, fi])
    Xps = sc.transform(X_pred[:, fi])

    Xf, yf = safe_balance(Xts, y_train)
    sw      = recency_weights(len(Xf))

    models = {
        "XGB": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.75, colsample_bytree=0.75, min_child_weight=5,
            scale_pos_weight=0.7, eval_metric="logloss",
            random_state=42, verbosity=0),
        "LGB": LGBMClassifier(
            n_estimators=300, num_leaves=31, max_depth=5,
            learning_rate=0.05, subsample=0.75, colsample_bytree=0.75,
            min_child_samples=10,
            class_weight="balanced" if regime == "ranging" else None,
            random_state=42, verbose=-1),
    }
    if CATBOOST_AVAILABLE:
        models["CAT"] = CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.05,
            subsample=0.75, random_seed=42, verbose=0)

    probas = []
    for name, m in models.items():
        try:
            m.fit(Xf, yf, sample_weight=sw)
            p = m.predict_proba(Xps)[:, 1]
            if not _is_constant(p):
                probas.append(p)
        except Exception:
            pass

    if not probas:
        return None
    return float(np.mean([p[0] for p in probas]))

# ================================================================
# LABELS FOR TRAINING
# ================================================================

def make_labels_for_training(df):
    close  = df["close"].values
    atr    = df["atr_14"].values
    labels = []
    for i in range(len(df)):
        if i + FORWARD_BARS >= len(df):
            labels.append(np.nan); continue
        barrier = ATR_MULTIPLIER * atr[i]
        upper   = close[i] + barrier
        lower   = close[i] - barrier
        window  = close[i+1:i+1+FORWARD_BARS]
        hu = int(np.argmax(window >= upper)) if np.any(window >= upper) else None
        hl = int(np.argmax(window <= lower)) if np.any(window <= lower) else None
        if hu is not None and hl is not None:
            labels.append(1 if hu <= hl else 0)
        elif hu is not None:
            labels.append(1)
        elif hl is not None:
            labels.append(0)
        else:
            labels.append(np.nan)
    df = df.copy()
    df["label"] = labels
    return df

# ================================================================
# POSITION SIZING
# ================================================================

def compute_position_size(account_balance, atr_value, symbol):
    risk_amount = account_balance * RISK_PCT
    stop_pips   = atr_value * ATR_MULT * 100
    if "JPY" in symbol:
        pip_value_per_lot = 1000
    else:
        pip_value_per_lot = 10000
    lot_size = risk_amount / (stop_pips * (pip_value_per_lot / 10000))
    lot_size = max(0.01, round(lot_size, 2))
    tp_pips  = stop_pips * RR_RATIO
    return lot_size, round(stop_pips, 1), round(tp_pips, 1)

# ================================================================
# LOGGING
# ================================================================

def init_log():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=[
            "timestamp","pair","direction","regime","confidence",
            "entry_price","stop_pips","tp_pips","lot_size",
            "atr_14","adx_14","rsi_14",
            "result","actual_pnl","notes",
        ]).to_csv(LOG_FILE, index=False)
        print(f"  Log created: {LOG_FILE}")


def log_signal(row_dict):
    df  = pd.read_csv(LOG_FILE) if os.path.exists(LOG_FILE) else pd.DataFrame()
    df  = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)


def already_signaled_this_bar(pair, direction):
    """Block duplicate signals within the same 4-hour bar."""
    if not os.path.exists(LOG_FILE):
        return False
    try:
        df = pd.read_csv(LOG_FILE)
        if len(df) == 0:
            return False
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(hours=TIMEFRAME_MINUTES / 60)
        recent = df[(df["timestamp"] > cutoff) &
                    (df["pair"] == pair) &
                    (df["direction"] == direction)]
        return len(recent) > 0
    except Exception:
        return False

# ================================================================
# SIGNAL CHECK — ONE PAIR
# ================================================================

def check_pair(symbol, account_balance):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, LOOKBACK_BARS)
    if rates is None or len(rates) < 100:
        return None

    df = pd.DataFrame(rates)
    df["time"]     = pd.to_datetime(df["time"], unit="s")
    df             = df.rename(columns={"time": "datetime", "tick_volume": "volume"})
    df             = df[df["datetime"].dt.dayofweek < 5].reset_index(drop=True)

    df = build_features(df)
    df = df.dropna(subset=["atr_14", "rsi_14", "adx_14"]).reset_index(drop=True)

    if len(df) < TRAIN_BARS + 20:
        return None

    # Latest closed bar (index -1 is the bar that just closed)
    current      = df.iloc[-1]
    current_dict = current.to_dict()

    regime = classify_regime(current_dict)

    if not is_setup(current_dict, regime):
        return None

    # Training slice: exclude last FORWARD_BARS (unresolved)
    df_train = df.iloc[-(TRAIN_BARS + FORWARD_BARS):-FORWARD_BARS].copy()
    df_train = make_labels_for_training(df_train)
    df_train = df_train.dropna(subset=["label"]).copy()

    df_train["regime_live"] = df_train.apply(
        lambda r: classify_regime(r.to_dict()), axis=1)
    df_regime = df_train[df_train["regime_live"] == regime].copy()

    min_train = 50 if regime == "trending" else MIN_RANGING_TRAIN
    if len(df_regime) < min_train:
        return None

    available = [f for f in FEATURE_COLS if f in df_regime.columns]
    if len(available) < 5:
        return None

    X_train  = df_regime[available].values.astype(float)
    y_train  = df_regime["label"].values.astype(int)
    X_pred   = current[available].values.reshape(1, -1).astype(float)

    # Clean NaNs
    nan_rows = np.any(np.isnan(X_train), axis=1)
    X_train  = X_train[~nan_rows]
    y_train  = y_train[~nan_rows]
    if np.any(np.isnan(X_pred)):
        X_pred = np.nan_to_num(X_pred, nan=0.0)

    if len(X_train) < min_train:
        return None

    proba = train_and_predict(X_train, y_train, X_pred, available, regime)
    if proba is None:
        return None

    if proba >= BUY_FLOOR:
        direction  = "BUY"
        confidence = proba
    elif proba <= SELL_CEIL:
        direction  = "SELL"
        confidence = 1 - proba
    else:
        return None

    atr_val                  = float(current["atr_14"])
    lot_size, stop_pips, tp_pips = compute_position_size(
        account_balance, atr_val, symbol)
    entry_price = float(current["close"])

    if direction == "BUY":
        sl_price = round(entry_price - atr_val * ATR_MULT, 5)
        tp_price = round(entry_price + atr_val * ATR_MULT * RR_RATIO, 5)
    else:
        sl_price = round(entry_price + atr_val * ATR_MULT, 5)
        tp_price = round(entry_price - atr_val * ATR_MULT * RR_RATIO, 5)

    return {
        "symbol"     : symbol,
        "direction"  : direction,
        "regime"     : regime,
        "confidence" : round(confidence * 100, 1),
        "entry_price": entry_price,
        "sl_price"   : sl_price,
        "tp_price"   : tp_price,
        "stop_pips"  : stop_pips,
        "tp_pips"    : tp_pips,
        "lot_size"   : lot_size,
        "atr_14"     : round(atr_val, 5),
        "adx_14"     : round(float(current.get("adx_14", 0)), 1),
        "rsi_14"     : round(float(current.get("rsi_14", 50)), 1),
        "bar_time"   : str(current["datetime"]),
    }

# ================================================================
# DISPLAY
# ================================================================

def print_signal(sig, account_balance):
    risk_amount   = account_balance * RISK_PCT
    reward_amount = risk_amount * RR_RATIO
    now           = datetime.now().strftime("%Y-%m-%d %H:%M")
    arrow         = "↑" if sig["direction"] == "BUY" else "↓"

    print(f"\n{'█'*66}")
    print(f"  ★ SIGNAL  {now}  [H4]")
    print(f"{'█'*66}")
    print(f"  Pair      : {sig['symbol']}")
    print(f"  Direction : {sig['direction']}  {arrow}")
    print(f"  Regime    : {sig['regime'].upper()}")
    print(f"  Confidence: {sig['confidence']}%")
    print(f"{'─'*66}")
    print(f"  Entry     : {sig['entry_price']}")
    print(f"  Stop Loss : {sig['sl_price']}  ({sig['stop_pips']} pips)")
    print(f"  Take Profit: {sig['tp_price']}  ({sig['tp_pips']} pips)")
    print(f"  R:R       : 1:{RR_RATIO}")
    print(f"{'─'*66}")
    print(f"  Lot size  : {sig['lot_size']} lots")
    print(f"  Risk      : ${risk_amount:.2f}  ({RISK_PCT*100:.1f}% of ${account_balance:,.0f})")
    print(f"  Reward    : ${reward_amount:.2f}")
    print(f"{'─'*66}")
    print(f"  ATR14     : {sig['atr_14']}  |  ADX: {sig['adx_14']}  |  RSI: {sig['rsi_14']}")
    print(f"  Bar close : {sig['bar_time']}")
    print(f"{'█'*66}")
    print(f"\n  ► ACTION: Open {sig['direction']} {sig['lot_size']} lots {sig['symbol']}")
    print(f"            SL = {sig['sl_price']}    TP = {sig['tp_price']}")
    print(f"\n  Fill in {LOG_FILE} when trade closes:")
    print(f"    result (WIN/LOSS) · actual_pnl · notes")


def print_scan_summary(pair_results, account_balance):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n  [{now}] H4 bar scan — account ${account_balance:,.0f}")
    for pair, status in pair_results.items():
        print(f"    {pair:<8} {status}")

# ================================================================
# MAIN
# ================================================================

def main():
    global TIMEFRAME

    print("=" * 66)
    print(" Step 12 — Live Signal Monitor  ·  H4 Edition")
    print(" Paper Trading Mode")
    print("=" * 66)
    print(f" Pairs     : {', '.join(PAIRS)}")
    print(f" Timeframe : H4  (fires every 4 hours on bar close)")
    print(f" Risk      : {RISK_PCT*100:.1f}% per trade")
    print(f" R:R       : 1:{RR_RATIO}")
    print(f" Stop SL   : {ATR_MULT}× ATR14")
    print(f" Log file  : {LOG_FILE}")
    print("=" * 66)

    if not mt5.initialize():
        print(f"\n  ✗ MT5 failed to connect: {mt5.last_error()}")
        print("  Make sure MetaTrader 5 is open and logged in.")
        return

    TIMEFRAME = mt5.TIMEFRAME_H4

    print(f"\n  MT5 connected : {mt5.version()}")
    info = mt5.account_info()
    if info:
        balance = float(info.balance)
        print(f"  Account       : {info.login} ({info.server})")
        print(f"  Balance       : ${balance:,.2f}")
        print(f"  Account type  : {'Demo' if info.trade_mode == 0 else 'Live'}")
    else:
        balance = 10_000.0

    init_log()

    print(f"\n  Monitor running.  Waits for each H4 bar to close before scanning.")

    signals_session = 0

    try:
        while True:
            # ── wait for the next H4 bar close ──────────────────
            wait_for_bar_close()

            # ── refresh balance ──────────────────────────────────
            info = mt5.account_info()
            balance = float(info.balance) if info else balance

            pair_results = {}

            for symbol in PAIRS:
                try:
                    sig = check_pair(symbol, balance)
                    if sig is not None:
                        direction = sig["direction"]
                        if not already_signaled_this_bar(symbol, direction):
                            print_signal(sig, balance)
                            signals_session += 1
                            log_signal({
                                "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "pair"       : symbol,
                                "direction"  : direction,
                                "regime"     : sig["regime"],
                                "confidence" : sig["confidence"],
                                "entry_price": sig["entry_price"],
                                "stop_pips"  : sig["stop_pips"],
                                "tp_pips"    : sig["tp_pips"],
                                "lot_size"   : sig["lot_size"],
                                "atr_14"     : sig["atr_14"],
                                "adx_14"     : sig["adx_14"],
                                "rsi_14"     : sig["rsi_14"],
                                "result"     : "",
                                "actual_pnl" : "",
                                "notes"      : "",
                            })
                            pair_results[symbol] = f"★ {direction} signal fired!"
                        else:
                            pair_results[symbol] = "signal (already logged this bar)"
                    else:
                        pair_results[symbol] = "no signal"
                except Exception as e:
                    pair_results[symbol] = f"error: {str(e)[:60]}"

            print_scan_summary(pair_results, balance)

    except KeyboardInterrupt:
        print(f"\n\n  Monitor stopped.")
        print(f"  Signals fired this session : {signals_session}")
        print(f"  Log : {LOG_FILE}")
        print(f"\n  Fill in result / actual_pnl / notes for each closed trade.")
        mt5.shutdown()


if __name__ == "__main__":
    main()