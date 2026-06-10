"""
model_training.py
=================
Walk-forward training and evaluation pipeline for the USDJPY ensemble model.

Trains two regime-separated models (trending / ranging) using XGBoost,
LightGBM, CatBoost, and RandomForest on M15 OHLCV data.  Outputs holdout
metrics, walk-forward tables, signal logs, and LaTeX rows for the paper.

Evaluation threshold : BUY >= 0.64 / SELL <= 0.36
Production threshold : BUY >= 0.67  (auto_trader.py CONFIDENCE_MAP default)
Results are reported at the evaluation threshold; the higher production floor
reduces live overtrading frequency.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional dependency loader — installs missing packages at runtime.
# In a managed environment (Docker / CI) prefer pinning these in requirements.
# ---------------------------------------------------------------------------

def _install(pkg):
    os.system(f"pip install {pkg} -q --break-system-packages")


try:
    from xgboost import XGBClassifier
except ImportError:
    _install("xgboost")
    from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except ImportError:
    _install("lightgbm")
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    _install("catboost")
    try:
        from catboost import CatBoostClassifier
        CATBOOST_AVAILABLE = True
    except Exception:
        CATBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
except ImportError:
    _install("imbalanced-learn")
    from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR  = "forex_labeled"

MODEL_VERSION = "v12"
TIMEFRAME_TAG = "M15"
OUTPUT_DIR = f"forex_results_{PAIR.lower()}_{MODEL_VERSION}_{TIMEFRAME_TAG}"

PAIR           = "USDJPYm"
ATR_MULTIPLIER = 1.2
N_WINDOWS      = 10
MIN_TRAIN_BARS = 200
FORWARD_BARS   = 20
TOP_N_FEATURES = 25

SIGNAL_PCT = 0.20
# FIX 5: these are the EVALUATION thresholds used for all paper numbers.
# Production auto_trader.py uses 0.67 for forex — document this clearly.
BUY_FLOOR  = 0.64
SELL_CEIL  = 0.36
IMBALANCE_THRESHOLD = 0.35
MIN_SIGNALS = 15

DE_TREND_THRESHOLD = 0.35
ADX_STRONG         = 28
ADX_MODERATE       = 26
DE_WINDOW          = 20

# FIX 4: single authoritative constant — used in both walk-forward AND
#         the final model block so coverage is identical.
MIN_RANGING_TRAIN        = 80
# FIX 6: suppress results when ranging test set is too small to be reliable
MIN_RANGING_TEST_SAMPLES = 15
RECENCY_HALF_LIFE        = 800

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FEATURE LISTS
# ============================================================

BASE_FEATURES = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "ema_cross", "atr_14", "bb_width", "bb_pct",
    "return_1", "return_3", "return_5", "return_10", "return_20", "volatility_10", "volatility_20",
    "stoch_k", "stoch_d", "volume_ratio",
]
SESSION_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_london", "is_ny", "is_overlap",
]
V2_FEATURES = [
    "adx_14", "cci_20", "williams_r",
    "price_vs_ema200", "candle_body_pct", "hl_ratio",
    "swing_dist_high", "swing_dist_low", "adx_slope",
]
JPY_FEATURES = [
    "momentum_diff", "volatility_regime",
    "price_vs_ema50", "trend_alignment",
]
V5_FEATURES = ["directional_efficiency"]

FEATURE_COLS = (BASE_FEATURES + SESSION_FEATURES +
                V2_FEATURES + JPY_FEATURES + V5_FEATURES)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_session_features(df):
    dt   = pd.to_datetime(df["datetime"])
    hour = dt.dt.hour
    dow  = dt.dt.dayofweek
    df = df.copy()
    df["hour_sin"]   = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * dow  / 5)
    df["dow_cos"]    = np.cos(2 * np.pi * dow  / 5)
    df["is_london"]  = ((hour >= 8)  & (hour < 17)).astype(int)
    df["is_ny"]      = ((hour >= 13) & (hour < 22)).astype(int)
    df["is_overlap"] = ((hour >= 13) & (hour < 17)).astype(int)
    return df


def compute_adx(df, period=14):
    high  = df["high"].values  if "high"  in df.columns else None
    low   = df["low"].values   if "low"   in df.columns else None
    close = df["close"].values if "close" in df.columns else None
    if high is None or low is None or close is None:
        return np.zeros(len(df))
    n = len(close)
    tr = np.zeros(n); pdm = np.zeros(n); ndm = np.zeros(n)
    for i in range(1, n):
        tr[i]  = max(high[i]-low[i],
                     abs(high[i]-close[i-1]),
                     abs(low[i]-close[i-1]))
        pdm[i] = (max(high[i]-high[i-1], 0)
                  if (high[i]-high[i-1]) > (low[i-1]-low[i]) else 0)
        ndm[i] = (max(low[i-1]-low[i], 0)
                  if (low[i-1]-low[i]) > (high[i]-high[i-1]) else 0)
    str_v = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    spdm  = pd.Series(pdm).ewm(span=period, adjust=False).mean().values
    sndm  = pd.Series(ndm).ewm(span=period, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(str_v > 0, 100*spdm/str_v, 0)
        ndi = np.where(str_v > 0, 100*sndm/str_v, 0)
        dx  = np.where((pdi+ndi) > 0, 100*np.abs(pdi-ndi)/(pdi+ndi), 0)
    return pd.Series(dx).ewm(span=period, adjust=False).mean().values


def compute_cci(df, period=20):
    if not all(c in df.columns for c in ["high", "low", "close"]):
        return np.zeros(len(df))
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = np.where(mad > 0, (tp - sma) / (0.015 * mad), 0)
    return cci


def add_v2_features(df):
    df = df.copy()
    df["adx_14"] = compute_adx(df, 14)
    df["cci_20"] = compute_cci(df, 20)
    if all(c in df.columns for c in ["high", "low", "close"]):
        hh = df["high"].rolling(14).max()
        ll = df["low"].rolling(14).min()
        df["williams_r"] = np.where(
            (hh-ll) > 0, -100*(hh-df["close"])/(hh-ll), -50)
    else:
        df["williams_r"] = 0.0
    if "close" in df.columns:
        ema200 = df["close"].ewm(span=200, adjust=False).mean()
        df["price_vs_ema200"] = (df["close"] - ema200) / ema200 * 100
    else:
        df["price_vs_ema200"] = 0.0
    if all(c in df.columns for c in ["open", "close", "high", "low"]):
        body  = (df["close"] - df["open"]).abs()
        total = (df["high"] - df["low"]).replace(0, np.nan)
        df["candle_body_pct"] = (body / total).fillna(0.5)
    else:
        df["candle_body_pct"] = 0.5
    if all(c in df.columns for c in ["high", "low", "atr_14"]):
        df["hl_ratio"] = (
            (df["high"]-df["low"]) / df["atr_14"].replace(0, np.nan)
        ).fillna(1.0)
    else:
        df["hl_ratio"] = 1.0
    if "close" in df.columns:
        rh = df["close"].rolling(20).max()
        rl = df["close"].rolling(20).min()
        df["swing_dist_high"] = (rh - df["close"]) / df["close"] * 100
        df["swing_dist_low"]  = (df["close"] - rl) / df["close"] * 100
    else:
        df["swing_dist_high"] = 0.0
        df["swing_dist_low"]  = 0.0
    
    if "adx_slope" not in df.columns and "adx_14" in df.columns:
        df["adx_slope"] = df["adx_14"].diff(3).fillna(0.0)
    
    return df


def add_base_features(df):
    """Compute base technical indicators. ATR and RSI use Wilder EWMA (span=14)."""
    df = df.copy()
    close = df["close"]
    high  = df["high"]  if "high"  in df.columns else close
    low   = df["low"]   if "low"   in df.columns else close

    if "rsi_14" not in df.columns:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    if "macd" not in df.columns:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

    if "ema_cross" not in df.columns:
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        df["ema_cross"] = (ema9 - ema21) / (close + 1e-10) * 100

    if "atr_14" not in df.columns:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        # Wilder EWMA for ATR (matches live pipeline in auto_trader.py)
        df["atr_14"] = tr.ewm(span=14, adjust=False).mean()

    if "bb_width" not in df.columns:
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_up  = bb_mid + 2 * bb_std
        bb_lo  = bb_mid - 2 * bb_std
        df["bb_width"] = (bb_up - bb_lo) / (bb_mid + 1e-10)
        df["bb_pct"]   = (close - bb_lo) / (bb_up - bb_lo + 1e-10)

    if "return_1" not in df.columns:
        df["return_1"]  = close.pct_change(1)  * 100
        df["return_5"]  = close.pct_change(5)  * 100
        df["return_10"] = close.pct_change(10) * 100

    if "volatility_10" not in df.columns:
        ret = close.pct_change()
        df["volatility_10"] = ret.rolling(10).std() * 100
        df["volatility_20"] = ret.rolling(20).std() * 100

    if "stoch_k" not in df.columns:
        ll14 = low.rolling(14).min()
        hh14 = high.rolling(14).max()
        df["stoch_k"] = 100 * (close - ll14) / (hh14 - ll14 + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    
    if "return_20" not in df.columns:
        df["return_20"] = close.pct_change(20) * 100
    
    if "return_3" not in df.columns:
        df["return_3"] = close.pct_change(3) * 100
    
    if "volume_ratio" not in df.columns:
        if "volume" in df.columns:
            df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
            df["volume_ratio"] = df["volume_ratio"].fillna(1.0).clip(0, 10)
        else:
            df["volume_ratio"] = 1.0
    
    if "adx_slope" not in df.columns:
        if "adx_14" in df.columns:
            df["adx_slope"] = df["adx_14"].diff(3)
        else:
            df["adx_slope"] = 0.0
    
    return df


def add_jpy_features(df):
    df = df.copy()
    if "close" not in df.columns:
        for f in JPY_FEATURES:
            df[f] = 0.0
        return df
    close = df["close"]
    if "rsi_14" in df.columns:
        delta50 = close.diff()
        gain50  = delta50.clip(lower=0).ewm(span=50, adjust=False).mean()
        loss50  = (-delta50.clip(upper=0)).ewm(span=50, adjust=False).mean()
        rs50    = gain50 / loss50.replace(0, np.nan)
        rsi50   = 100 - (100 / (1 + rs50))
        df["momentum_diff"] = df["rsi_14"] - rsi50.fillna(50)
    else:
        df["momentum_diff"] = 0.0
    returns = close.pct_change()
    vol5  = returns.rolling(5).std()
    vol20 = returns.rolling(20).std()
    df["volatility_regime"] = (
        vol5 / vol20.replace(0, np.nan)
    ).fillna(1.0).clip(0, 5)
    ema50  = close.ewm(span=50, adjust=False).mean()
    ema20  = close.ewm(span=20, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    df["price_vs_ema50"]  = (close - ema50) / ema50 * 100
    bull20_50  = (ema20  > ema50).astype(int)
    bull50_200 = (ema50  > ema200).astype(int)
    df["trend_alignment"] = (bull20_50 + bull50_200 - 1).astype(float)
    return df


def add_v5_features(df):
    df = df.copy()
    if "close" not in df.columns:
        df["directional_efficiency"] = 0.5
        return df
    returns    = df["close"].pct_change()
    net_move   = returns.rolling(DE_WINDOW).sum().abs()
    total_path = returns.abs().rolling(DE_WINDOW).sum()
    de = (net_move / total_path.replace(0, np.nan)).fillna(0.5).clip(0, 1)
    df["directional_efficiency"] = de
    return df

# ============================================================
# COMPOSITE REGIME CLASSIFIER
# ============================================================

def classify_regime(df, pair=""):
    adx_strong, adx_moderate = get_adx_thresholds(pair)
    regime = pd.Series("ranging", index=df.index)
    has_de  = "directional_efficiency" in df.columns
    has_adx = "adx_14" in df.columns
    has_ta  = "trend_alignment" in df.columns
    if has_de and has_adx and has_ta:
        rule1 = (
            (df["directional_efficiency"] > DE_TREND_THRESHOLD) &
            (df["adx_14"] > adx_strong)
        )
        # Silver needs looser rule2 due to its choppier nature
        if pair == "XAGUSDm":
            rule2 = (
                (df["trend_alignment"] != 0) &
                (df["adx_14"] > adx_strong)
            )
        else:
            rule2 = (
                (df["trend_alignment"] == 1) &
                (df["adx_14"] > adx_strong)
            )
        regime[rule1 | rule2] = "trending"
    elif has_adx:
        regime[df["adx_14"] > adx_strong] = "trending"
    return regime


def regime_counts(regime_series):
    counts = regime_series.value_counts()
    total  = len(regime_series)
    return {r: (counts.get(r, 0), counts.get(r, 0) / total * 100)
            for r in ["trending", "ranging"]}

# ============================================================
# REGIME-SEPARATED SETUP FILTERS
# ============================================================

def filter_to_setups(df):
    has_regime = "regime" in df.columns
    has_vr     = "volatility_regime" in df.columns
    has_ema50  = "price_vs_ema50" in df.columns
    has_adx    = "adx_14" in df.columns
    has_ta     = "trend_alignment" in df.columns

    rsi_extreme = (df["rsi_14"] < 25) | (df["rsi_14"] > 75)
    bb_extreme  = (df["bb_pct"] < 0.03) | (df["bb_pct"] > 0.97)
    ranging_filter = rsi_extreme | bb_extreme

    if has_ema50 and has_adx and has_ta:
        near_ema50      = df["price_vs_ema50"].abs() < 0.3
        strong_trend    = df["adx_14"] > ADX_MODERATE
        aligned_emas    = df["trend_alignment"] != 0
        trending_filter = near_ema50 & strong_trend & aligned_emas
    else:
        trending_filter = ranging_filter

    if has_regime:
        tr_mask  = df["regime"] == "trending"
        ra_mask  = df["regime"] == "ranging"
        combined = (tr_mask & trending_filter) | (ra_mask & ranging_filter)
    else:
        combined = ranging_filter | trending_filter

    if has_vr:
        combined = combined & (df["volatility_regime"] < 2.0)

    out = df[combined].copy()

    if has_regime:
        n_tr = int((df["regime"] == "trending").sum())
        n_ra = int((df["regime"] == "ranging").sum())
        n_tr_pass = int((tr_mask & trending_filter).sum())
        n_ra_pass = int((ra_mask & ranging_filter).sum())
        if has_vr:
            calm = df["volatility_regime"] < 2.0
            n_tr_pass = int((tr_mask & trending_filter & calm).sum())
            n_ra_pass = int((ra_mask & ranging_filter & calm).sum())
        print(f"   Setup filter: trending {n_tr_pass}/{n_tr} "
              f"({n_tr_pass/max(n_tr,1)*100:.1f}%)  |  "
              f"ranging {n_ra_pass}/{n_ra} "
              f"({n_ra_pass/max(n_ra,1)*100:.1f}%)")

    return out

# ============================================================
# LABELING
# ============================================================

def make_binary_labels_full(df_full, df_filtered,
                             forward_bars=FORWARD_BARS,
                             max_pos=None):
    """
    Parameters
    ----------
    max_pos : int or None
        If set, rows whose barrier window extends beyond this position in
        df_full are dropped, preventing label leakage at walk-forward
        window boundaries.  Pass train_end_pos - 1 from the WF loop.
    """
    close_full = df_full["close"].values
    pos_lookup = {idx: pos
                  for pos, idx in enumerate(df_full.index.tolist())}
    labels = []; keep_mask = []
    for row_idx, row in df_filtered.iterrows():
        pos = pos_lookup.get(row_idx)
        if pos is None or pos + forward_bars >= len(close_full):
            keep_mask.append(False); labels.append(-1); continue

        # drop bars whose barrier window crosses the training boundary
        if max_pos is not None and pos + forward_bars > max_pos:
            keep_mask.append(False); labels.append(-1); continue

        barrier   = ATR_MULTIPLIER * row["atr_14"]
        upper     = close_full[pos] + barrier
        lower     = close_full[pos] - barrier
        window    = close_full[pos+1 : pos+1+forward_bars]
        hit_upper = np.argmax(window >= upper) if np.any(window >= upper) else None
        hit_lower = np.argmax(window <= lower) if np.any(window <= lower) else None
        if hit_upper is not None and hit_lower is not None:
            label = 1 if hit_upper <= hit_lower else 0
            keep_mask.append(True)
        elif hit_upper is not None:
            label = 1; keep_mask.append(True)
        elif hit_lower is not None:
            label = 0; keep_mask.append(True)
        else:
            label = -1; keep_mask.append(False)
        labels.append(label)
    df_out = df_filtered.copy()
    df_out["binary_label"] = labels
    return df_out[keep_mask].copy()

# ============================================================
# WALK-FORWARD SPLITTER
# ============================================================

def walk_forward_splits(df_full, df_setups, n_windows, min_train_bars):
    """
    Returns list of (train_setups, test_setups, train_end_pos) tuples.
    train_end_pos is the last positional index in df_full that belongs
    to the training window — pass it as max_pos to make_binary_labels_full
    to prevent label leakage at window boundaries.
    """
    n         = len(df_full)
    test_size = max((n - min_train_bars) // (n_windows + 1), 50)
    splits    = []

    full_pos_lookup = {idx: pos for pos, idx in enumerate(df_full.index.tolist())}

    for i in range(n_windows):
        train_end_pos = min_train_bars + i * test_size
        test_end_pos  = train_end_pos + test_size
        if test_end_pos > n:
            break

        train_end_idx = df_full.index[train_end_pos - 1]
        test_end_idx  = df_full.index[test_end_pos  - 1]

        train_setups = df_setups[df_setups.index <= train_end_idx].copy()
        test_setups  = df_setups[
            (df_setups.index > train_end_idx) &
            (df_setups.index <= test_end_idx)
        ].copy()

        # Sanity check: at least min_train_bars worth of raw setups
        if len(train_setups) >= min_train_bars // 2 and len(test_setups) >= 30:
            splits.append((train_setups, test_setups, train_end_pos - 1))

    return splits

# ============================================================
# HYBRID THRESHOLD
# ============================================================

def hybrid_threshold(probas,
                     signal_pct=SIGNAL_PCT,
                     buy_floor=BUY_FLOOR,
                     sell_ceil=SELL_CEIL):
    buy_pct  = float(np.percentile(probas, (1 - signal_pct) * 100))
    sell_pct = float(np.percentile(probas, signal_pct * 100))
    buy_thr  = max(buy_pct,  buy_floor)
    sell_thr = min(sell_pct, sell_ceil)
    if buy_thr <= sell_thr:
        mid      = (buy_thr + sell_thr) / 2
        buy_thr  = mid + 1e-4
        sell_thr = mid - 1e-4
    return buy_thr, sell_thr


def apply_threshold(probas, buy_thr, sell_thr):
    pred = np.full(len(probas), -1, dtype=int)
    pred[probas >= buy_thr]  = 1
    pred[probas <= sell_thr] = 0
    return pred

# ============================================================
# BALANCING
# ============================================================

def safe_balance(X, y, random_state=42):
    counts   = pd.Series(y).value_counts()
    minority = int(counts.min())
    if minority >= 10:
        try:
            k = min(5, minority - 1)
            sm = SMOTE(random_state=random_state, k_neighbors=k)
            return sm.fit_resample(X, y)
        except Exception:
            pass
    if minority >= 3:
        try:
            ros = RandomOverSampler(random_state=random_state)
            return ros.fit_resample(X, y)
        except Exception:
            pass
    return X, y


def smart_balance(X, y, random_state=42,
                  imbalance_threshold=IMBALANCE_THRESHOLD):
    counts       = pd.Series(y).value_counts(normalize=True)
    minority_pct = float(counts.min())
    if minority_pct < imbalance_threshold:
        return safe_balance(X, y, random_state=random_state), True
    return (X, y), False

# ============================================================
# RECENCY WEIGHTING
# ============================================================

def recency_weights(n, half_life=RECENCY_HALF_LIFE):
    positions = np.arange(n)
    weights   = np.power(2.0, (positions - (n - 1)) / half_life)
    return weights / weights.sum() * n

# ============================================================
# PER-FOLD FEATURE SELECTION
# ============================================================

def select_top_features_fold(X, y, feature_cols, top_n):
    top_n = min(top_n, len(feature_cols), max(1, len(X) // 10))
    try:
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6,
            random_state=42, n_jobs=-1
        )
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=feature_cols)
        return imp.nlargest(top_n).index.tolist()
    except Exception:
        return feature_cols[:top_n]

# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    sig_mask = y_pred >= 0
    n_sig = int(sig_mask.sum())
    if n_sig < MIN_SIGNALS:
        return None
    yt = y_true[sig_mask]
    yp = y_pred[sig_mask]
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    return {
        "n_signals":  n_sig,
        "signal_pct": n_sig / len(y_true) * 100,
        "f1":   f1_score(yt, yp, average="binary", zero_division=0),
        "prec": precision_score(yt, yp, average="binary", zero_division=0),
        "rec":  recall_score(yt, yp, average="binary", zero_division=0),
        "acc":  accuracy_score(yt, yp),
        "cm":   cm,
        "y_true_sig": yt,
        "y_pred_sig": yp,
        "sig_mask":   sig_mask,
    }


def print_metrics(res, label=""):
    if res is None:
        return
    cm = res["cm"]
    f1 = res["f1"]
    if f1 >= 0.70:   bkt = "★★ excellent"
    elif f1 >= 0.65: bkt = "★ strong"
    elif f1 >= 0.55: bkt = "good"
    elif f1 >= 0.45: bkt = "acceptable"
    else:             bkt = "weak"
    tag = f"  [{label}]" if label else ""
    print(f"{tag}  signals={res['n_signals']} ({res['signal_pct']:.0f}%)")
    print(f"  F1={f1*100:.1f}%  Prec={res['prec']*100:.1f}%  "
          f"Rec={res['rec']*100:.1f}%  Acc={res['acc']*100:.1f}%  → {bkt}")
    print(f"  CM  TP={cm[1,1]}  TN={cm[0,0]}  "
          f"FP={cm[0,1]}  FN={cm[1,0]}")
    gap = res["prec"] - res["rec"]
    if abs(gap) > 10:
        direction = "precision-heavy" if gap > 0 else "recall-heavy"
        print(f"  [diag] Prec-Rec gap = {gap*100:+.1f}pp  ({direction})")

# ============================================================
# UTILITY
# ============================================================

def _is_constant(arr, tol=1e-6):
    return float(np.std(arr)) < tol

# ============================================================
# SIGNAL SAVING HELPER
# ============================================================

def collect_signals(df_test, regime_mask, pred, proba, y_true_aligned,
                    sig_mask, regime_name, window_num, source, all_signals):
    sig_indices = np.where(sig_mask)[0]
    sig_rows    = df_test[regime_mask].iloc[sig_indices]

    for i, (row_idx, row) in enumerate(sig_rows.iterrows()):
        direction = "BUY" if pred[sig_mask][i] == 1 else "SELL"
        correct   = bool(pred[sig_mask][i] == y_true_aligned[i])
        all_signals.append({
            "datetime"  : row["datetime"],
            "regime"    : regime_name,
            "direction" : direction,
            "proba"     : round(float(proba[sig_mask][i]), 5),
            "atr_14"    : round(float(row["atr_14"]), 5),
            "entry"     : round(float(row["close"]), 5),
            "true_label": int(y_true_aligned[i]),
            "correct"   : correct,
            "window"    : window_num,
            "source"    : source,
        })

# ============================================================
# TRENDING MODEL
# ============================================================

def get_trending_base_models():
    m = {
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.75,
            min_child_weight=5, eval_metric="logloss",
            scale_pos_weight=0.80,
            random_state=42, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500, num_leaves=63, max_depth=6,
            learning_rate=0.03, subsample=0.75,
            colsample_bytree=0.75, min_child_samples=15,
            class_weight={0: 1.0, 1: 0.90},
            random_state=7, verbose=-1,
        ),
    }
    if CATBOOST_AVAILABLE:
        m["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.04,
            subsample=0.75, colsample_bylevel=0.75,
            class_weights=[1.0, 0.90],
            random_seed=21, verbose=0,
        )
    else:
        m["LightGBM_B"] = LGBMClassifier(
            n_estimators=400, num_leaves=31, max_depth=5,
            learning_rate=0.05, boosting_type="dart",
            subsample=0.8, colsample_bytree=0.8,
            class_weight={0: 1.0, 1: 0.8},
            random_state=17, verbose=-1,
        )
    return m


def get_meta_learner():
    return LGBMClassifier(
        n_estimators=200, num_leaves=15, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        random_state=13, verbose=-1,
    )


def train_trending_model(X_train, y_train, X_test,
                          feature_cols, verbose=True):
    models    = get_trending_base_models()
    n_m       = len(models)
    n         = len(X_train)
    n_folds   = 3
    fold_sz   = max(n // n_folds, 1)
    oof       = np.full((n, n_m), 0.5)
    oof_valid = np.zeros(n_m, dtype=bool)
    feat_imp  = defaultdict(float)

    counts_full       = pd.Series(y_train).value_counts(normalize=True)
    minority_pct_full = float(counts_full.min())
    if verbose:
        print(f"    [INFO] Trending label split: "
              f"BUY={counts_full.get(1,0)*100:.1f}%  "
              f"SELL={counts_full.get(0,0)*100:.1f}%  "
              f"→ {'balancing' if minority_pct_full < IMBALANCE_THRESHOLD else 'raw'}")

    for fi in range(n_folds):
        vs  = fi * fold_sz
        ve  = vs + fold_sz if fi < n_folds - 1 else n
        tr  = list(range(0, vs)) + list(range(ve, n))
        vl  = list(range(vs, ve))
        if not tr or not vl:
            continue
        (Xtr_fold, ytr_fold), _ = smart_balance(X_train[tr], y_train[tr])
        sw_fold = recency_weights(len(Xtr_fold))
        Xvl = X_train[vl]
        for mi, (name, model) in enumerate(models.items()):
            try:
                model.fit(Xtr_fold, ytr_fold, sample_weight=sw_fold)
                fold_p = model.predict_proba(Xvl)[:, 1]
                if not _is_constant(fold_p):
                    oof[vl, mi] = fold_p
                    oof_valid[mi] = True
                else:
                    if verbose:
                        print(f"    [WARN] {name} OOF fold {fi}: constant output")
            except Exception as e:
                if verbose:
                    print(f"    [WARN] {name} OOF fold {fi} failed: {e}")

    oof_var    = oof.var(axis=0)
    valid_cols = np.where(oof_var > 1e-4)[0]
    use_meta   = len(valid_cols) >= 2

    meta = None
    if use_meta:
        meta = get_meta_learner()
        try:
            meta.fit(oof[:, valid_cols], y_train)
        except Exception as e:
            if verbose:
                print(f"    [WARN] Meta fit failed: {e}")
            meta = None
    else:
        if verbose:
            print(f"    [WARN] OOF diversity low — skipping meta")

    (Xf, yf), balanced = smart_balance(X_train, y_train)
    sw_fit = recency_weights(len(Xf))

    test_p     = np.full((len(X_test), n_m), 0.5)
    test_valid = np.zeros(n_m, dtype=bool)

    for mi, (name, model) in enumerate(models.items()):
        try:
            model.fit(Xf, yf, sample_weight=sw_fit)
            proba = model.predict_proba(X_test)[:, 1]
            if not _is_constant(proba):
                test_p[:, mi] = proba
                test_valid[mi] = True
                if hasattr(model, "feature_importances_") and feature_cols:
                    for fn, imp in zip(feature_cols, model.feature_importances_):
                        feat_imp[fn] += imp / n_m
            else:
                if verbose:
                    print(f"    [WARN] {name} test constant — skipped")
        except Exception as e:
            if verbose:
                print(f"    [WARN] {name} retrain failed: {e}")

    n_valid = int(test_valid.sum())
    if n_valid == 0:
        if verbose:
            print(f"    [ERROR] All base models failed")
        return None, {}

    valid_test_cols = np.where(test_valid)[0]

    if meta is not None and len(valid_cols) >= 2:
        shared_cols = np.intersect1d(valid_cols, valid_test_cols)
        if len(shared_cols) >= 2:
            try:
                test_meta_in = np.full((len(X_test), len(valid_cols)), 0.5)
                for c in shared_cols:
                    v_pos = np.where(valid_cols == c)[0]
                    if len(v_pos):
                        test_meta_in[:, v_pos[0]] = test_p[:, c]
                final_proba = meta.predict_proba(test_meta_in)[:, 1]
                if _is_constant(final_proba):
                    if verbose:
                        print(f"    [WARN] Meta output constant — using mean")
                    final_proba = test_p[:, valid_test_cols].mean(axis=1)
            except Exception as e:
                if verbose:
                    print(f"    [WARN] Meta predict failed: {e}")
                final_proba = test_p[:, valid_test_cols].mean(axis=1)
        else:
            final_proba = test_p[:, valid_test_cols].mean(axis=1)
    else:
        final_proba = test_p[:, valid_test_cols].mean(axis=1)

    if _is_constant(final_proba):
        if verbose:
            print(f"    [ERROR] Final proba constant after all fallbacks")
        return None, {}

    return final_proba, dict(feat_imp)

# ============================================================
# RANGING MODEL
# ============================================================

def train_ranging_model(X_train, y_train, X_test,
                         feature_cols, verbose=True):
    # FIX 4: use MIN_RANGING_TRAIN (80) consistently
    if len(X_train) < MIN_RANGING_TRAIN:
        return None, {}

    n_leaves  = min(15, max(4, len(X_train) // 20))
    min_child = max(3, len(X_train) // 30)

    model = LGBMClassifier(
        n_estimators=200, num_leaves=n_leaves, max_depth=4,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=min_child,
        class_weight="balanced",
        random_state=42, verbose=-1,
    )

    Xf, yf = safe_balance(X_train, y_train)
    sw_fit  = recency_weights(len(Xf))

    try:
        model.fit(Xf, yf, sample_weight=sw_fit)
        if not hasattr(model, "booster_"):
            if verbose:
                print(f"    [ERROR] Ranging: booster_ missing")
            return None, {}
        proba = model.predict_proba(X_test)[:, 1]
        if _is_constant(proba):
            if verbose:
                print(f"    [WARN] Ranging: constant proba")
            return None, {}
        feat_imp = {}
        if hasattr(model, "feature_importances_") and feature_cols:
            feat_imp = dict(zip(feature_cols, model.feature_importances_))
        return proba, feat_imp
    except Exception as e:
        if verbose:
            print(f"    [ERROR] Ranging fit failed: {e}")
        return None, {}

# ============================================================
# MAIN
# ============================================================

print("=" * 66)
print(" Forex Phase 6 v12 — USDJPY: Class-Weight Precision Fix")
print(" Fixes: 1-label-leakage 2-ATR-EWMA 4-ranging-threshold")
print("        5-confidence-note 6-small-sample-suppress")
print("=" * 66)
print(f" Pair        : {PAIR}")
print(f" Features    : {len(FEATURE_COLS)} total")
print(f" Regime      : Composite DE+ADX+alignment")
print(f" Signal pct  : {SIGNAL_PCT*100:.0f}% per side")
# FIX 5: be explicit that these are EVALUATION thresholds
print(f" Conf floor  : BUY>={BUY_FLOOR}  SELL<={SELL_CEIL}  [EVALUATION thresholds]")
print(f"               NOTE: production auto_trader.py uses 0.67 for forex.")
print(f"               Paper reports evaluation thresholds; see Section 4 footnote.")
print(f" ADX         : strong={ADX_STRONG}  moderate={ADX_MODERATE}")
print(f" Min signals : {MIN_SIGNALS} per window")
print(f" Min ranging train : {MIN_RANGING_TRAIN}")
print(f" Min ranging test  : {MIN_RANGING_TEST_SAMPLES} (suppress small-sample results)")
print("=" * 66)

input_path = os.path.join(INPUT_DIR, f"{PAIR}_M15_labeled.csv")
if not os.path.exists(input_path):
    print(f"\n[ERROR] File not found: {input_path}")
    print(f"  Expected: {os.path.abspath(input_path)}")
    exit(1)

print(f"\n Loading {PAIR} data...")
df_full = pd.read_csv(input_path, parse_dates=["datetime"])

# filter weekends
if "datetime" in df_full.columns:
    df_full = df_full[pd.to_datetime(df_full["datetime"]).dt.dayofweek < 5].copy()

df_full = add_base_features(df_full)
df_full = add_session_features(df_full)
df_full = add_v2_features(df_full)
df_full = add_jpy_features(df_full)
df_full = add_v5_features(df_full)
df_full["regime"] = classify_regime(df_full)

available_features = [f for f in FEATURE_COLS if f in df_full.columns]
missing = [f for f in FEATURE_COLS if f not in df_full.columns]
if missing:
    print(f" [NOTE] Features missing from CSV: {missing}")

df_full = df_full.dropna(subset=available_features).reset_index(drop=True)

full_reg = regime_counts(df_full["regime"])
print(f"\n Full data regime (composite classifier):")
for r, (cnt, pct) in full_reg.items():
    bar = "█" * int(pct / 2)
    print(f"   {r:<10}: {cnt:>6,} bars ({pct:4.1f}%)  {bar}")

print(f"\n Applying regime-separated setup filters:")
df_setups = filter_to_setups(df_full)

# Full-dataset labels — used for hold-out and final model
df_labeled = make_binary_labels_full(df_full, df_setups)
buy_pct    = (df_labeled["binary_label"] == 1).mean() * 100
sell_pct   = (df_labeled["binary_label"] == 0).mean() * 100

lab_reg = regime_counts(df_labeled["regime"])
print(f"\n Labeled setup regime distribution:")
for r, (cnt, pct) in lab_reg.items():
    print(f"   {r:<10}: {cnt:>5,} ({pct:.1f}%)")
print(f" Total labeled : {len(df_labeled):,}  "
      f"BUY={buy_pct:.1f}%  SELL={sell_pct:.1f}%")

if len(df_labeled) < MIN_TRAIN_BARS + 100:
    print("[ERROR] Not enough labeled setups"); exit(1)

holdout_n  = max(int(len(df_labeled) * 0.10), 50)
df_wf      = df_labeled.iloc[:-holdout_n].copy()
df_holdout = df_labeled.iloc[-holdout_n:].copy()

# Align df_setups with walk-forward split: only keep setups in WF range
wf_max_idx     = df_wf.index[-1]
df_setups_wf   = df_setups[df_setups.index <= wf_max_idx].copy()

print(f"\n Walk-forward set : {len(df_wf):,} bars")
print(f" Hold-out set     : {len(df_holdout):,} bars "
      f"({df_holdout['datetime'].min().date()} → "
      f"{df_holdout['datetime'].max().date()})")

# FIX 1: pass df_full + df_setups_wf so each window re-labels with max_pos
splits = walk_forward_splits(df_full, df_setups_wf, N_WINDOWS, MIN_TRAIN_BARS)
print(f" WF windows       : {len(splits)}")

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

all_results   = []
all_signals   = []
grand_tr_true = []; grand_tr_pred = []
grand_ra_true = []; grand_ra_pred = []
all_trend_imp = defaultdict(list)
all_range_imp = defaultdict(list)

print(f"\n{'─'*66}")
print(f" Walk-forward validation")
print(f"{'─'*66}\n")

# FIX 1: splits now yields (train_setups, test_setups, train_end_pos)
for w_idx, (train_setups_raw, test_setups_raw, train_end_pos) in enumerate(splits):

    # FIX 1: re-label with boundary guard
    train_df = make_binary_labels_full(
        df_full, train_setups_raw,
        forward_bars=FORWARD_BARS,
        max_pos=train_end_pos           # ← prevents leakage
    )
    test_df = make_binary_labels_full(
        df_full, test_setups_raw,
        forward_bars=FORWARD_BARS,
        max_pos=None                    # test labels can look forward freely
    )

    if len(train_df) < MIN_TRAIN_BARS // 2 or len(test_df) < 10:
        print(f"  Window {w_idx+1}: too few labeled samples after boundary guard — skip")
        continue

    y_train       = train_df["binary_label"].values.astype(int)
    y_test        = test_df["binary_label"].values.astype(int)
    test_regimes  = test_df["regime"].values
    tr_train_mask = train_df["regime"].values == "trending"
    tr_test_mask  = test_regimes == "trending"
    ra_train_mask = train_df["regime"].values == "ranging"
    ra_test_mask  = test_regimes == "ranging"

    n_tr_train = tr_train_mask.sum()
    n_tr_test  = tr_test_mask.sum()
    n_ra_train = ra_train_mask.sum()
    n_ra_test  = ra_test_mask.sum()

    print(f"  Window {w_idx+1}  "
          f"({test_df['datetime'].min().date()} → "
          f"{test_df['datetime'].max().date()})")
    print(f"  TRAIN: T={n_tr_train}  R={n_ra_train}  "
          f"| TEST: T={n_tr_test}  R={n_ra_test}")

    win_res = {
        "window":     w_idx + 1,
        "test_start": str(test_df["datetime"].min().date()),
        "test_end":   str(test_df["datetime"].max().date()),
        "n_tr_train": n_tr_train, "n_tr_test": n_tr_test,
        "n_ra_train": n_ra_train, "n_ra_test": n_ra_test,
    }

    # ── TRENDING ──────────────────────────────────────────────
    if n_tr_train >= MIN_TRAIN_BARS // 3 and n_tr_test >= MIN_SIGNALS:
        Xtr = train_df[available_features].values[tr_train_mask]
        ytr = y_train[tr_train_mask]
        Xte = test_df[available_features].values[tr_test_mask]
        yte = y_test[tr_test_mask]

        sel = select_top_features_fold(Xtr, ytr, available_features, TOP_N_FEATURES)
        fi  = [available_features.index(f) for f in sel]

        sc  = StandardScaler()
        Xs  = sc.fit_transform(Xtr[:, fi])
        Xts = sc.transform(Xte[:, fi])

        proba, imp = train_trending_model(Xs, ytr, Xts, sel, verbose=True)
        if imp:
            for fn, v in imp.items():
                all_trend_imp[fn].append(v)

        if proba is not None:
            print(f"  [TRENDING] proba=[{proba.min():.3f},{proba.max():.3f}]"
                  f"  mean={proba.mean():.3f}  std={proba.std():.3f}")
            buy_thr, sell_thr = hybrid_threshold(proba)
            pred = apply_threshold(proba, buy_thr, sell_thr)
            res  = compute_metrics(yte, pred)
            print(f"  [TRENDING] thr={buy_thr:.3f}/{sell_thr:.3f}")
            print_metrics(res, "TRENDING")
            if res:
                grand_tr_true.extend(res["y_true_sig"].tolist())
                grand_tr_pred.extend(res["y_pred_sig"].tolist())
                win_res.update({"tr_f1": res["f1"], "tr_prec": res["prec"],
                                "tr_rec": res["rec"], "tr_acc": res["acc"],
                                "tr_signals": res["n_signals"]})
                collect_signals(
                    df_test=test_df, regime_mask=tr_test_mask,
                    pred=pred, proba=proba,
                    y_true_aligned=res["y_true_sig"],
                    sig_mask=res["sig_mask"],
                    regime_name="trending", window_num=w_idx + 1,
                    source="walkforward", all_signals=all_signals,
                )
            else:
                print(f"  [TRENDING] < {MIN_SIGNALS} signals after threshold")
                win_res.update({"tr_f1": np.nan, "tr_signals": 0})
        else:
            print(f"  [TRENDING] all base models failed")
            win_res.update({"tr_f1": np.nan, "tr_signals": 0})
    else:
        reason = (f"train={n_tr_train}<{MIN_TRAIN_BARS//3}"
                  if n_tr_train < MIN_TRAIN_BARS // 3
                  else f"test={n_tr_test}<{MIN_SIGNALS}")
        print(f"  [TRENDING] skip ({reason})")
        win_res.update({"tr_f1": np.nan, "tr_signals": 0})

    # ── RANGING ───────────────────────────────────────────────
    # FIX 6: suppress results when n_ra_test < MIN_RANGING_TEST_SAMPLES
    if n_ra_test < 10:
        print(f"  [RANGING]  {n_ra_test} test bars — skip")
        win_res.update({"ra_f1": np.nan, "ra_signals": 0,
                        "ra_n": n_ra_test, "ra_status": "skip_test"})
    elif n_ra_test < MIN_RANGING_TEST_SAMPLES:
        # FIX 6: small sample — train if possible but suppress the result
        print(f"  [RANGING]  {n_ra_test} test bars < {MIN_RANGING_TEST_SAMPLES} "
              f"— result suppressed for paper")
        win_res.update({"ra_f1": np.nan, "ra_signals": 0,
                        "ra_n": n_ra_test,
                        "ra_status": f"suppressed(n={n_ra_test})"})
    elif n_ra_train < MIN_RANGING_TRAIN:  # FIX 4: was 40, now 80
        print(f"  [RANGING]  {n_ra_train} train bars "
              f"(need {MIN_RANGING_TRAIN}) — abstain")
        win_res.update({"ra_f1": np.nan, "ra_signals": 0,
                        "ra_n": n_ra_test, "ra_status": "abstain"})
    else:
        Xra = train_df[available_features].values[ra_train_mask]
        yra = y_train[ra_train_mask]
        Xrt = test_df[available_features].values[ra_test_mask]
        yrt = y_test[ra_test_mask]

        max_feat = min(TOP_N_FEATURES, max(5, len(Xra) // 10))
        sel_r = select_top_features_fold(Xra, yra, available_features, max_feat)
        fir   = [available_features.index(f) for f in sel_r]

        sc_r  = StandardScaler()
        Xrs   = sc_r.fit_transform(Xra[:, fir])
        Xrts  = sc_r.transform(Xrt[:, fir])

        ra_proba, ra_imp = train_ranging_model(Xrs, yra, Xrts, sel_r, verbose=True)
        if ra_imp:
            for fn, v in ra_imp.items():
                all_range_imp[fn].append(v)

        if ra_proba is not None:
            print(f"  [RANGING]  proba=[{ra_proba.min():.3f},{ra_proba.max():.3f}]"
                  f"  mean={ra_proba.mean():.3f}  std={ra_proba.std():.3f}")
            buy_r, sell_r = hybrid_threshold(ra_proba)
            pred_r = apply_threshold(ra_proba, buy_r, sell_r)
            res_r  = compute_metrics(yrt, pred_r)
            print(f"  [RANGING]  thr={buy_r:.3f}/{sell_r:.3f}")
            print_metrics(res_r, "RANGING")
            if res_r:
                grand_ra_true.extend(res_r["y_true_sig"].tolist())
                grand_ra_pred.extend(res_r["y_pred_sig"].tolist())
                win_res.update({"ra_f1": res_r["f1"],
                                "ra_prec": res_r["prec"],
                                "ra_rec":  res_r["rec"],
                                "ra_acc":  res_r["acc"],
                                "ra_signals": res_r["n_signals"],
                                "ra_n": n_ra_test,
                                "ra_status": "ok"})
                collect_signals(
                    df_test=test_df, regime_mask=ra_test_mask,
                    pred=pred_r, proba=ra_proba,
                    y_true_aligned=res_r["y_true_sig"],
                    sig_mask=res_r["sig_mask"],
                    regime_name="ranging", window_num=w_idx + 1,
                    source="walkforward", all_signals=all_signals,
                )
            else:
                print(f"  [RANGING]  < {MIN_SIGNALS} signals")
                win_res.update({"ra_f1": np.nan, "ra_signals": 0,
                                "ra_n": n_ra_test, "ra_status": "few_signals"})
        else:
            print(f"  [RANGING]  model abstained")
            win_res.update({"ra_f1": np.nan, "ra_signals": 0,
                            "ra_n": n_ra_test, "ra_status": "abstain"})

    all_results.append(win_res)
    print()

# ============================================================
# HOLD-OUT TEST
# ============================================================

print(f"{'─'*66}")
print(f" Final hold-out test (unseen — used ONCE)")
print(f"{'─'*66}\n")

y_wf       = df_wf["binary_label"].values.astype(int)
y_ho       = df_holdout["binary_label"].values.astype(int)
ho_regimes = df_holdout["regime"].values
ho_reg     = regime_counts(pd.Series(ho_regimes))

print(f"  Period  : {df_holdout['datetime'].min().date()} → "
      f"{df_holdout['datetime'].max().date()}")
print(f"  Setups  : {len(y_ho)}")
for r, (cnt, pct) in ho_reg.items():
    print(f"  {r:<10}: {cnt} ({pct:.1f}%)")
print()

ho_tr_res = None
ho_ra_res = None

# ── Trending hold-out ─────────────────────────────────────
tr_ho_mask = ho_regimes == "trending"
if tr_ho_mask.sum() >= MIN_SIGNALS:
    Xtr_wf = df_wf[available_features].values[df_wf["regime"].values == "trending"]
    ytr_wf = y_wf[df_wf["regime"].values == "trending"]
    sel_ho = select_top_features_fold(Xtr_wf, ytr_wf, available_features, TOP_N_FEATURES)
    fi_ho  = [available_features.index(f) for f in sel_ho]
    sc_ho  = StandardScaler()
    Xtr_s  = sc_ho.fit_transform(Xtr_wf[:, fi_ho])
    Xho_s  = sc_ho.transform(df_holdout[available_features].values[tr_ho_mask][:, fi_ho])
    yho_tr = y_ho[tr_ho_mask]

    ho_tr_proba, _ = train_trending_model(Xtr_s, ytr_wf, Xho_s, sel_ho, verbose=True)
    if ho_tr_proba is not None:
        print(f"  [TRENDING hold-out]  proba=[{ho_tr_proba.min():.3f},"
              f"{ho_tr_proba.max():.3f}]  mean={ho_tr_proba.mean():.3f}  "
              f"std={ho_tr_proba.std():.3f}")
        b_thr, s_thr = hybrid_threshold(ho_tr_proba)
        ho_tr_pred   = apply_threshold(ho_tr_proba, b_thr, s_thr)
        ho_tr_res    = compute_metrics(yho_tr, ho_tr_pred)
        print(f"  [TRENDING hold-out]  thr={b_thr:.3f}/{s_thr:.3f}")
        print_metrics(ho_tr_res, "TRENDING hold-out")
        if ho_tr_res:
            collect_signals(
                df_test=df_holdout, regime_mask=tr_ho_mask,
                pred=ho_tr_pred, proba=ho_tr_proba,
                y_true_aligned=ho_tr_res["y_true_sig"],
                sig_mask=ho_tr_res["sig_mask"],
                regime_name="trending", window_num=0,
                source="holdout", all_signals=all_signals,
            )
    else:
        print(f"  [TRENDING hold-out] all base models failed")
else:
    print(f"  [TRENDING] only {tr_ho_mask.sum()} test bars — skip")

print()

# ── Ranging hold-out ──────────────────────────────────────
ra_ho_mask = ho_regimes == "ranging"
Xra_wf     = df_wf[available_features].values[df_wf["regime"].values == "ranging"]
yra_wf     = y_wf[df_wf["regime"].values == "ranging"]

# FIX 4: MIN_RANGING_TRAIN (80) consistent here too
if ra_ho_mask.sum() >= 10 and len(Xra_wf) >= MIN_RANGING_TRAIN:
    max_f  = min(TOP_N_FEATURES, max(5, len(Xra_wf) // 10))
    sel_ra = select_top_features_fold(Xra_wf, yra_wf, available_features, max_f)
    fi_ra  = [available_features.index(f) for f in sel_ra]
    sc_ra  = StandardScaler()
    Xra_s  = sc_ra.fit_transform(Xra_wf[:, fi_ra])
    Xho_ra = sc_ra.transform(df_holdout[available_features].values[ra_ho_mask][:, fi_ra])
    yho_ra = y_ho[ra_ho_mask]

    ho_ra_proba, _ = train_ranging_model(Xra_s, yra_wf, Xho_ra, sel_ra, verbose=True)
    if ho_ra_proba is not None:
        br, sr = hybrid_threshold(ho_ra_proba)
        ho_ra_pred = apply_threshold(ho_ra_proba, br, sr)
        ho_ra_res  = compute_metrics(yho_ra, ho_ra_pred)
        print(f"  [RANGING  hold-out]  thr={br:.3f}/{sr:.3f}")
        print_metrics(ho_ra_res, "RANGING hold-out")
        if ho_ra_res:
            collect_signals(
                df_test=df_holdout, regime_mask=ra_ho_mask,
                pred=ho_ra_pred, proba=ho_ra_proba,
                y_true_aligned=ho_ra_res["y_true_sig"],
                sig_mask=ho_ra_res["sig_mask"],
                regime_name="ranging", window_num=0,
                source="holdout", all_signals=all_signals,
            )
    else:
        print(f"  [RANGING hold-out] model abstained")
else:
    reason = (f"test={ra_ho_mask.sum()}<10"
              if ra_ho_mask.sum() < 10
              else f"train={len(Xra_wf)}<{MIN_RANGING_TRAIN}")
    print(f"  [RANGING] hold-out abstain ({reason})")

# ============================================================
# HOLDOUT RESULTS SUMMARY
# ============================================================

print(f"\n{'='*66}")
print(f" HOLDOUT RESULTS — copy these numbers into your paper")
# FIX 5: document threshold distinction clearly in output
print(f" NOTE: evaluation used BUY>={BUY_FLOOR} / SELL<={SELL_CEIL}")
print(f"       production auto_trader.py uses 0.67 for forex pairs.")
print(f"       Paper should state: 'Results reported at evaluation threshold")
print(f"       (0.64); production system uses a higher floor (0.67) to")
print(f"       reduce overtrading frequency.'")
print(f"{'='*66}")


def _fmt(res, key):
    if res and key in res:
        return round(res[key] * 100, 1)
    return "--"


ho_tr_f1   = _fmt(ho_tr_res, "f1")
ho_tr_prec = _fmt(ho_tr_res, "prec")
ho_tr_rec  = _fmt(ho_tr_res, "rec")
ho_tr_acc  = _fmt(ho_tr_res, "acc")
ho_tr_n    = ho_tr_res["n_signals"] if ho_tr_res else 0

ho_ra_f1   = _fmt(ho_ra_res, "f1")
ho_ra_prec = _fmt(ho_ra_res, "prec")
ho_ra_rec  = _fmt(ho_ra_res, "rec")
ho_ra_acc  = _fmt(ho_ra_res, "acc")
ho_ra_n    = ho_ra_res["n_signals"] if ho_ra_res else 0

print(f"\n  Trending | F1={ho_tr_f1}%  P={ho_tr_prec}%  "
      f"R={ho_tr_rec}%  Acc={ho_tr_acc}%  signals={ho_tr_n}")
print(f"  Ranging  | F1={ho_ra_f1}%  P={ho_ra_prec}%  "
      f"R={ho_ra_rec}%  Acc={ho_ra_acc}%  signals={ho_ra_n}")

print(f"\n  LaTeX row (paste directly into your paper table):")
print(f"  {PAIR} & "
      f"{ho_tr_f1}\\% & {ho_tr_prec}\\% & {ho_tr_rec}\\% & {ho_tr_acc}\\% & "
      f"{ho_ra_f1}\\% & {ho_ra_prec}\\% & {ho_ra_rec}\\% & {ho_ra_acc}\\% \\\\")

# ── Walk-forward paper table with FIX 6 annotations ───────────
print(f"\n  Walk-forward table (n = ranging test-set size):")
print(f"  '--' = suppressed (n < {MIN_RANGING_TEST_SAMPLES})")
for r in all_results:
    tr_f1_val = r.get('tr_f1', np.nan)
    tr_f1_str = f"{tr_f1_val*100:.1f}" if isinstance(tr_f1_val, float) and not np.isnan(tr_f1_val) else "--"

    ra_f1_val = r.get('ra_f1', np.nan)
    ra_f1_str = f"{ra_f1_val*100:.1f}" if isinstance(ra_f1_val, float) and not np.isnan(ra_f1_val) else "--"

    ra_prec_val = r.get('ra_prec', np.nan)
    ra_prec_str = f"{ra_prec_val*100:.1f}" if isinstance(ra_prec_val, float) and not np.isnan(ra_prec_val) else "--"

    status = r.get("ra_status", "")
    n_str  = str(r.get("ra_n", "--"))
    note   = f"  % {status}" if status and status != "ok" else ""
    print(
        f"W{r['window']} & "
        f"{tr_f1_str}\\% & "
        f"{ra_f1_str}\\% & $n={n_str}$ \\\\ {note}"
    )

if ho_tr_res:
    gap = ho_tr_res["prec"] - ho_tr_res["rec"]
    direction = "precision-heavy" if gap > 0 else "recall-heavy"
    print(f"\n  [diag] Trending Prec-Rec gap = {gap*100:+.1f}pp  ({direction})")
    if ho_tr_res["f1"] >= 0.65:
        print(f"  [diag] v12 target MET  (F1 >= 65%)")
    else:
        print(f"  [diag] v12 target MISSED  (F1 < 65%)  "
              f"— consider further BUY-weight tuning")

holdout_dict = {
    "pair": PAIR,
    "evaluation_thresholds": {"buy_floor": BUY_FLOOR, "sell_ceil": SELL_CEIL},
    "production_thresholds": {"buy_floor": 0.67, "sell_ceil": 0.33,
                               "note": "auto_trader.py CONFIDENCE_MAP default"},
    "holdout_period": {
        "start": str(df_holdout["datetime"].min().date()),
        "end":   str(df_holdout["datetime"].max().date()),
        "n_setups": int(len(df_holdout)),
    },
    "trending": {
        "f1":      ho_tr_f1, "prec":    ho_tr_prec,
        "rec":     ho_tr_rec, "acc":     ho_tr_acc,
        "signals": ho_tr_n,
    },
    "ranging": {
        "f1":      ho_ra_f1, "prec":    ho_ra_prec,
        "rec":     ho_ra_rec, "acc":     ho_ra_acc,
        "signals": ho_ra_n,
    },
}
holdout_json_path = os.path.join(OUTPUT_DIR, "holdout_results.json")
with open(holdout_json_path, "w") as f:
    json.dump(holdout_dict, f, indent=2)
print(f"\n  Saved → {holdout_json_path}")
print(f"{'='*66}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def print_importance(imp_dict, label, top_n=10):
    avg = {k: np.mean(v) for k, v in imp_dict.items() if v}
    if not avg: return
    print(f"\n{'─'*66}")
    print(f" Top {top_n} features — {label}")
    print(f"{'─'*66}")
    s  = sorted(avg.items(), key=lambda x: x[1], reverse=True)
    mx = s[0][1]
    for rank, (fn, imp) in enumerate(s[:top_n], 1):
        bar = "█" * int(30 * imp / mx)
        tag = (" ← JPY" if fn in JPY_FEATURES else
               " ← DE"  if fn in V5_FEATURES  else "")
        print(f"  {rank:2}. {fn:<26} {bar:<30} {imp:.4f}{tag}")

print_importance(all_trend_imp, "TRENDING model")
print_importance(all_range_imp, "RANGING model")

# ============================================================
# GRAND SUMMARY
# ============================================================

print(f"\n{'='*66}")
print(f" Phase 6 v12 — {PAIR} Summary")
print(f"{'='*66}")

if all_results:
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{PAIR.lower()}_v12_results.csv"),
        index=False
    )

    tr_rows = results_df.dropna(subset=["tr_f1"])
    if len(tr_rows):
        print(f"\n  Trending model ({len(tr_rows)} windows with results):")
        print(f"  Avg F1={tr_rows['tr_f1'].mean()*100:.1f}%  "
              f"Prec={tr_rows['tr_prec'].mean()*100:.1f}%  "
              f"Rec={tr_rows['tr_rec'].mean()*100:.1f}%")

    ra_rows = results_df.dropna(subset=["ra_f1"])
    if len(ra_rows):
        print(f"\n  Ranging model ({len(ra_rows)} windows with results):")
        print(f"  Avg F1={ra_rows['ra_f1'].mean()*100:.1f}%  "
              f"Prec={ra_rows['ra_prec'].mean()*100:.1f}%")

    if grand_tr_true:
        ov = {
            "f1":   f1_score(grand_tr_true, grand_tr_pred, average="binary", zero_division=0),
            "prec": precision_score(grand_tr_true, grand_tr_pred, average="binary", zero_division=0),
            "rec":  recall_score(grand_tr_true, grand_tr_pred, average="binary", zero_division=0),
        }
        print(f"\n  Pooled TRENDING ({len(grand_tr_true):,} signals):")
        print(f"  F1={ov['f1']*100:.1f}%  "
              f"Prec={ov['prec']*100:.1f}%  "
              f"Rec={ov['rec']*100:.1f}%")
        print(f"  vs random : +{(ov['f1']-0.5)*100:.1f}pp")

    if grand_ra_true:
        ov_r = {
            "f1":   f1_score(grand_ra_true, grand_ra_pred, average="binary", zero_division=0),
            "prec": precision_score(grand_ra_true, grand_ra_pred, average="binary", zero_division=0),
        }
        print(f"\n  Pooled RANGING ({len(grand_ra_true):,} signals):")
        print(f"  F1={ov_r['f1']*100:.1f}%  Prec={ov_r['prec']*100:.1f}%")

if all_signals:
    signals_df   = pd.DataFrame(all_signals)
    signals_path = os.path.join(OUTPUT_DIR, "signals.csv")
    signals_df.to_csv(signals_path, index=False)
    print(f"\n{'='*66}")
    print(f" Signal log saved → {signals_path}")
    print(f"  Total signals : {len(signals_df)}")
    wf_n  = (signals_df["source"] == "walkforward").sum()
    ho_n  = (signals_df["source"] == "holdout").sum()
    tr_n  = (signals_df["regime"] == "trending").sum()
    ra_n  = (signals_df["regime"] == "ranging").sum()
    win_n = (signals_df["correct"] == True).sum()
    acc   = win_n / len(signals_df) * 100
    print(f"  Walk-forward  : {wf_n}  |  Hold-out : {ho_n}")
    print(f"  Trending      : {tr_n}  |  Ranging  : {ra_n}")
    print(f"  Correct       : {win_n}/{len(signals_df)} ({acc:.1f}%)")
    print(f"{'='*66}")
else:
    print("\n  [WARN] No signals collected — check MIN_SIGNALS threshold")

print(f"\n Results saved to: {OUTPUT_DIR}/")
print(f"{'='*66}")