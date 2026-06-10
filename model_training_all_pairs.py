"""
model_training_all_pairs.py
============================
Walk-forward training and evaluation pipeline for all 7 trading pairs.

Iterates over EURUSDm, USDJPYm, GBPUSDm, AUDUSDm, XAUUSDm, XAGUSDm,
BTCUSDm — applying per-symbol ATR multipliers, forward-bar horizons, and
ADX thresholds.  Outputs per-pair results and a combined summary CSV.

Evaluation threshold : BUY >= 0.64 / SELL <= 0.36
Production threshold : BUY >= 0.67  (auto_trader.py CONFIDENCE_MAP default)
"""

import os
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional dependency loader — installs missing packages at runtime.
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
OUTPUT_DIR = "forex_results_usdjpy_v12_M15"

PAIR           = "USDJPY"
ATR_MULTIPLIER = 1.2
N_WINDOWS      = 8
MIN_TRAIN_BARS = 300
FORWARD_BARS   = 20
TOP_N_FEATURES = 25

SIGNAL_PCT = 0.20           # back to v10 — enough signals per window
BUY_FLOOR  = 0.64           # slight raise from v10's 0.62
SELL_CEIL  = 0.36           # slight lower from v10's 0.38
IMBALANCE_THRESHOLD = 0.35
MIN_SIGNALS = 15            # back to v10

DE_TREND_THRESHOLD = 0.35
ADX_STRONG         = 28
ADX_MODERATE       = 26     # slight raise from v10's 25
DE_WINDOW          = 20

MIN_RANGING_TRAIN = 80
RECENCY_HALF_LIFE = 800

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PER-SYMBOL CONFIGURATION
# ============================================================

ALL_PAIRS     = ["USDJPYm", "EURUSDm", "GBPUSDm", "AUDUSDm",
                 "XAUUSDm", "XAGUSDm", "BTCUSDm"]

METAL_SYMBOLS = {"XAUUSDm", "XAGUSDm", "BTCUSDm"}

ATR_MULTIPLIER_MAP = {
    "XAUUSDm": 2.0,
    "XAGUSDm": 1.8,
    "BTCUSDm": 3.0,
    "default": 1.2,
}

FORWARD_BARS_MAP = {
    "XAUUSDm": 30,
    "XAGUSDm": 25,
    "BTCUSDm": 35,
    "default": 20,
}

ADX_STRONG_MAP = {
    "XAUUSDm": 24,
    "XAGUSDm": 24,
    "BTCUSDm": 22,
    "default": 28,
}

ADX_MODERATE_MAP = {
    "XAUUSDm": 20,
    "XAGUSDm": 20,
    "BTCUSDm": 18,
    "default": 26,
}

# ============================================================
# FEATURE LISTS
# ============================================================

BASE_FEATURES = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "ema_cross", "atr_14", "bb_width", "bb_pct",
    "return_1", "return_5", "return_10",
    "volatility_10", "volatility_20",
    "stoch_k", "stoch_d",
]
SESSION_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_london", "is_ny", "is_overlap",
]
V2_FEATURES = [
    "adx_14", "cci_20", "williams_r",
    "price_vs_ema200", "candle_body_pct", "hl_ratio",
    "swing_dist_high", "swing_dist_low",
]
JPY_FEATURES = [
    "momentum_diff", "volatility_regime",
    "price_vs_ema50",    "trend_alignment",
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

def classify_regime(df):
    regime = pd.Series("ranging", index=df.index)
    has_de  = "directional_efficiency" in df.columns
    has_adx = "adx_14" in df.columns
    has_ta  = "trend_alignment" in df.columns
    if has_de and has_adx and has_ta:
        rule1 = ((df["directional_efficiency"] > DE_TREND_THRESHOLD) &
                 (df["adx_14"] > ADX_STRONG))
        rule2 = ((df["trend_alignment"] != 0) &
                 (df["adx_14"] > ADX_MODERATE))
        regime[rule1 | rule2] = "trending"
    elif has_adx:
        regime[df["adx_14"] > ADX_STRONG] = "trending"
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
        near_ema50      = df["price_vs_ema50"].abs() < 0.3   # v11 tighter filter kept
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
                             forward_bars=FORWARD_BARS):
    close_full = df_full["close"].values
    pos_lookup = {idx: pos
                  for pos, idx in enumerate(df_full.index.tolist())}
    labels = []; keep_mask = []
    for row_idx, row in df_filtered.iterrows():
        pos = pos_lookup.get(row_idx)
        if pos is None or pos + forward_bars >= len(close_full):
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

def walk_forward_splits(df, n_windows, min_train_bars):
    n         = len(df)
    test_size = max((n - min_train_bars) // (n_windows + 1), 50)
    splits    = []
    for i in range(n_windows):
        train_end = min_train_bars + i * test_size
        test_end  = train_end + test_size
        if test_end > n: break
        train_df = df.iloc[:train_end].copy()
        test_df  = df.iloc[train_end:test_end].copy()
        if len(train_df) >= min_train_bars and len(test_df) >= 30:
            splits.append((train_df, test_df))
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
# TRENDING MODEL — v12: class weights on ALL three base models
# ============================================================

def get_trending_base_models():
    # All base models penalize BUY (class 1) with weight < 1.0 to reduce
    # overtrading on false positives.
    # XGB  : scale_pos_weight < 1 penalises the positive class
    # LGB  : class_weight dict with BUY (1) weighted lower than SELL (0)
    # CatBoost: class_weights list [SELL_weight, BUY_weight]
    m = {
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.75,
            min_child_weight=5, eval_metric="logloss",
            scale_pos_weight=0.65,
            random_state=42, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500, num_leaves=63, max_depth=6,
            learning_rate=0.03, subsample=0.75,
            colsample_bytree=0.75, min_child_samples=15,
            class_weight={0: 1.0, 1: 0.8},
            random_state=7, verbose=-1,
        ),
    }
    if CATBOOST_AVAILABLE:
        m["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.04,
            subsample=0.75, colsample_bylevel=0.75,
            class_weights=[1.0, 0.8],
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
            print(f"    [WARN] OOF diversity low (max_var={oof_var.max():.2e}) — skipping meta")

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
# MAIN LOOP — runs for all 7 pairs
# ============================================================

all_pair_summaries = []

for PAIR in ALL_PAIRS:

    ATR_MULTIPLIER = ATR_MULTIPLIER_MAP.get(PAIR, ATR_MULTIPLIER_MAP["default"])
    FORWARD_BARS   = FORWARD_BARS_MAP.get(PAIR,   FORWARD_BARS_MAP["default"])
    ADX_STRONG     = ADX_STRONG_MAP.get(PAIR,     ADX_STRONG_MAP["default"])
    ADX_MODERATE   = ADX_MODERATE_MAP.get(PAIR,   ADX_MODERATE_MAP["default"])

    PAIR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, PAIR)
    os.makedirs(PAIR_OUTPUT_DIR, exist_ok=True)

    pair_is_metal = PAIR in METAL_SYMBOLS

    print("\n" + "=" * 66)
    print(f" Forex Phase 6 v12 — {PAIR} "
          f"[{'METAL' if pair_is_metal else 'FOREX'}]")
    print("=" * 66)

    print(f" ATR mult    : {ATR_MULTIPLIER}x"
          f"  |  Forward bars : {FORWARD_BARS}")

    print(f" ADX strong  : {ADX_STRONG}"
          f"       |  ADX moderate : {ADX_MODERATE}")

    print(f" Signal pct  : {SIGNAL_PCT*100:.0f}%"
          f"  |  BUY>={BUY_FLOOR}"
          f"  SELL<={SELL_CEIL}")

    print("=" * 66)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    input_csv = os.path.join(
        INPUT_DIR,
        f"{PAIR}_M15_labeled.csv"
    )

    if not os.path.exists(input_csv):
        input_csv = os.path.join(
            INPUT_DIR,
            f"{PAIR}_H1_labeled.csv"
        )

    if not os.path.exists(input_csv):
        print(f"[SKIP] No labeled file found for {PAIR}")
        continue

    print(f"\n Loading {PAIR} data from {input_csv}...")

    df_full = pd.read_csv(
        input_csv,
        parse_dates=["datetime"]
    )

    # ------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------
    df_full = add_session_features(df_full)
    df_full = add_v2_features(df_full)
    df_full = add_jpy_features(df_full)
    df_full = add_v5_features(df_full)

    df_full["regime"] = classify_regime(df_full)

    available_features = [
        f for f in FEATURE_COLS
        if f in df_full.columns
    ]

    df_full = df_full.dropna(
        subset=available_features
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Regime summary
    # ------------------------------------------------------------
    full_reg = regime_counts(df_full["regime"])

    print("\n Full data regime:")

    for r, (cnt, pct) in full_reg.items():
        bar = "█" * int(pct / 2)

        print(f"   {r:<10}: {cnt:>6,} bars "
              f"({pct:4.1f}%)  {bar}")

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------
    print("\n Applying setup filters...")

    df_setups = filter_to_setups(df_full)

    df_labeled = make_binary_labels_full(
        df_full,
        df_setups
    )

    buy_pct = (
        (df_labeled["binary_label"] == 1).mean() * 100
    )

    sell_pct = (
        (df_labeled["binary_label"] == 0).mean() * 100
    )

    print(f" Total labeled: {len(df_labeled):,}"
          f"  BUY={buy_pct:.1f}%"
          f"  SELL={sell_pct:.1f}%")

    if len(df_labeled) < MIN_TRAIN_BARS + 100:
        print(f"[SKIP] Not enough labeled setups for {PAIR}")
        continue

    # ------------------------------------------------------------
    # Train / Holdout split
    # ------------------------------------------------------------
    holdout_n = max(
        int(len(df_labeled) * 0.10),
        50
    )

    df_wf = df_labeled.iloc[:-holdout_n].copy()
    df_holdout = df_labeled.iloc[-holdout_n:].copy()

    print(f" Walk-forward : {len(df_wf):,} bars")

    print(f" Hold-out     : {len(df_holdout):,} bars")

    splits = walk_forward_splits(
        df_wf,
        N_WINDOWS,
        MIN_TRAIN_BARS
    )

    print(f" WF windows   : {len(splits)}")

    # ------------------------------------------------------------
    # Result containers
    # ------------------------------------------------------------
    all_results = []
    all_signals = []

    grand_tr_true = []
    grand_tr_pred = []

    # ------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------
    print(f"\n{'─'*66}")
    print(" Walk-forward validation")
    print(f"{'─'*66}\n")

    for w_idx, (train_df, test_df) in enumerate(splits):

        print(f"  Window {w_idx+1}")

        # --------------------------------------------------------
        # Your existing window logic continues here...
        # (same as your original script)
        # --------------------------------------------------------
        # Feature prep
        # --------------------------------------------------------
        available_features = [f for f in FEATURE_COLS if f in train_df.columns]

        X_train = train_df[available_features].values
        y_train = train_df["binary_label"].values
        X_test  = test_df[available_features].values
        y_test  = test_df["binary_label"].values

        tr_mask = train_df["regime"] == "trending"
        ra_mask = train_df["regime"] == "ranging"
        tr_mask_test = test_df["regime"] == "trending"
        ra_mask_test = test_df["regime"] == "ranging"

        # --------------------------------------------------------
        # Trending window
        # --------------------------------------------------------
        X_tr = X_train[tr_mask.values]
        y_tr = y_train[tr_mask.values]
        X_tr_test = X_test[tr_mask_test.values]
        y_tr_test = y_test[tr_mask_test.values]

        tr_res = None
        if len(X_tr) >= MIN_TRAIN_BARS // 2 and len(X_tr_test) >= 10:
            top_feats = select_top_features_fold(
                X_tr, y_tr, available_features, TOP_N_FEATURES)
            feat_idx  = [available_features.index(f) for f in top_feats]
            tr_proba, _ = train_trending_model(
                X_tr[:, feat_idx], y_tr,
                X_tr_test[:, feat_idx],
                top_feats, verbose=False,
            )
            if tr_proba is not None:
                buy_thr, sell_thr = hybrid_threshold(tr_proba)
                tr_pred = apply_threshold(tr_proba, buy_thr, sell_thr)
                tr_res  = compute_metrics(y_tr_test, tr_pred)
                print_metrics(tr_res, label=f"W{w_idx+1} TRENDING")
                if tr_res:
                    collect_signals(
                        test_df, tr_mask_test, tr_pred, tr_proba,
                        tr_res["y_true_sig"], tr_res["sig_mask"],
                        "trending", w_idx + 1, "walkforward", all_signals,
                    )

        # --------------------------------------------------------
        # Ranging window
        # --------------------------------------------------------
        X_ra = X_train[ra_mask.values]
        y_ra = y_train[ra_mask.values]
        X_ra_test = X_test[ra_mask_test.values]
        y_ra_test = y_test[ra_mask_test.values]

        ra_res = None
        if len(X_ra) >= MIN_RANGING_TRAIN and len(X_ra_test) >= 10:
            top_feats_ra = select_top_features_fold(
                X_ra, y_ra, available_features, TOP_N_FEATURES)
            feat_idx_ra  = [available_features.index(f) for f in top_feats_ra]
            ra_proba, _ = train_ranging_model(
                X_ra[:, feat_idx_ra], y_ra,
                X_ra_test[:, feat_idx_ra],
                top_feats_ra, verbose=False,
            )
            if ra_proba is not None:
                buy_thr_r, sell_thr_r = hybrid_threshold(ra_proba)
                ra_pred = apply_threshold(ra_proba, buy_thr_r, sell_thr_r)
                ra_res  = compute_metrics(y_ra_test, ra_pred)
                print_metrics(ra_res, label=f"W{w_idx+1} RANGING")
                if ra_res:
                    collect_signals(
                        test_df, ra_mask_test, ra_pred, ra_proba,
                        ra_res["y_true_sig"], ra_res["sig_mask"],
                        "ranging", w_idx + 1, "walkforward", all_signals,
                    )

        all_results.append({
            "window":   w_idx + 1,
            "tr_f1":    tr_res["f1"]   if tr_res else np.nan,
            "tr_prec":  tr_res["prec"] if tr_res else np.nan,
            "tr_rec":   tr_res["rec"]  if tr_res else np.nan,
            "ra_f1":    ra_res["f1"]   if ra_res else np.nan,
            "ra_prec":  ra_res["prec"] if ra_res else np.nan,
        })
        grand_tr_true.extend(
            tr_res["y_true_sig"].tolist() if tr_res else [])

    # ----------------------------------------------------------------
    # Pair summary
    # ----------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(PAIR_OUTPUT_DIR, f"{PAIR.lower()}_results.csv"),
        index=False,
    )

    tr_rows = results_df.dropna(subset=["tr_f1"])
    ra_rows = results_df.dropna(subset=["ra_f1"])

    pair_summary = {
        "pair":        PAIR,
        "type":        "METAL" if pair_is_metal else "FOREX",
        "tr_avg_f1":   round(tr_rows["tr_f1"].mean() * 100, 1) if len(tr_rows) else None,
        "tr_avg_prec": round(tr_rows["tr_prec"].mean() * 100, 1) if len(tr_rows) else None,
        "ra_avg_f1":   round(ra_rows["ra_f1"].mean() * 100, 1) if len(ra_rows) else None,
        "ra_avg_prec": round(ra_rows["ra_prec"].mean() * 100, 1) if len(ra_rows) else None,
        "n_signals":   len(all_signals),
    }
    all_pair_summaries.append(pair_summary)

    print(f"\n  {PAIR} done | "
          f"TR avg F1={pair_summary['tr_avg_f1']}%  "
          f"RA avg F1={pair_summary['ra_avg_f1']}%  "
          f"signals={pair_summary['n_signals']}")

# ================================================================
# GRAND SUMMARY
# ================================================================
print("\n" + "="*66)
print(" GRAND SUMMARY — All 7 Pairs")
print("="*66)

if all_pair_summaries:

    summary_df = pd.DataFrame(all_pair_summaries)

    print(summary_df.to_string(index=False))

    summary_path = os.path.join(
        OUTPUT_DIR,
        "all_pairs_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved: {summary_path}")

print("="*66)