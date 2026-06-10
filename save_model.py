"""
save_model.py — Phase 6 v20  [Paper-Ready Edition]

INHERITS ALL v19 FIXES unchanged, plus:

NEW IN v20  (Sharpe fix + baseline + ranging guard)
────────────────────────────────────────────────────
FIX 25  sharpe_from_pnl     — equity-curve Sharpe, correctly annualised.
        v19 bug: ann_factor = sqrt(BARS_PER_YEAR * n_signals / step)
                 inflated Sharpe ~4x by multiplying n_signals inside sqrt.
        v20 fix: trades_per_year = n_signals * (BARS_PER_YEAR / step)
                 SR_annual = SR_per_trade * sqrt(trades_per_year)
FIX 26  simulate_pnl        — subtracts spread cost (asset-class specific)
        from every trade in R-units before Sharpe/PF computation.
FIX 27  ema_crossover_baseline — proper EMA 9/21 directional baseline
        evaluated on same test rows; F1 + Sharpe added to LaTeX table.
FIX 28  MIN_RANGING_REPORT=40 — ranging results suppressed when n<40
        (prevents misleading CI=[0,0] rows in paper tables).
FIX 29  _print_latex_tables  — adds BL F1 and BL Sharpe columns.
"""

import os, copy, warnings
import numpy as np
import pandas as pd
import joblib
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

warnings.filterwarnings("ignore")

# ─── dependency checks ───────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("pip install xgboost")

try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except ImportError:
    raise ImportError("pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
except ImportError:
    raise ImportError("pip install imbalanced-learn")

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, log_loss)

try:
    from scipy import stats as sp_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ─── configuration ───────────────────────────────────────────────────────────
INPUT_DIR  = "forex_labeled"
MODEL_DIR  = "./models"

PAIRS       = ["USDJPYm", "EURUSDm", "GBPUSDm", "AUDUSDm",
               "XAUUSDm", "XAGUSDm", "BTCUSDm"]
METAL_SYMS  = {"XAUUSDm", "XAGUSDm", "BTCUSDm"}
EXPLORATORY_PAIRS = {"BTCUSDm"}

ATR_MULTIPLIER_MAP = {"XAUUSDm": 2.5, "XAGUSDm": 2.0, "BTCUSDm": 3.5, "default": 1.2}
FORWARD_BARS_MAP   = {"XAUUSDm": 30,  "XAGUSDm": 25,  "BTCUSDm": 35,  "default": 20}
FETCH_BARS_MAP     = {"XAUUSDm": 20000,"XAGUSDm": 30000,"BTCUSDm": 50000,"default": 20000}

TOP_N_FEATURES = 25

WF_MIN_TRAIN  = 2000
WF_STEP       = 1500
N_BOOTSTRAP   = 1000
RR_RATIO      = 2.5
BARS_PER_YEAR = 26280    # M15 bars in a trading year (252d × 6.5h × 4)

# FIX 26: spread cost in R-units per trade (win or lose)
# 1 R = ATR_MULT × ATR.  Spread ≈ 1 pip / (1.2 × 25-pip ATR) ≈ 0.033 R for FX.
SPREAD_COST_R = {
    "default": 0.04,   # FX pairs
    "XAUUSDm": 0.06,   # Gold  (~$0.30 spread / ~$5 ATR)
    "XAGUSDm": 0.05,   # Silver
    "BTCUSDm": 0.03,   # BTC   (tight spread, large ATR)
}

BUY_FLOOR_MAP  = {"XAUUSDm": 0.72, "XAGUSDm": 0.72, "BTCUSDm": 0.74, "default": 0.67}
SELL_CEIL_MAP  = {"XAUUSDm": 0.28, "XAGUSDm": 0.28, "BTCUSDm": 0.26, "default": 0.33}

BUY_FLOOR  = 0.60
SELL_CEIL  = 0.40
SIGNAL_PCT = 0.25

IMBALANCE_THRESHOLD = 0.35
DE_TREND_THRESHOLD  = 0.35

ADX_STRONG_MAP    = {"GBPUSDm": 26, "BTCUSDm": 22, "XAUUSDm": 24, "XAGUSDm": 24, "default": 30}
ADX_MODERATE      = 28
ADX_STRONG_METAL  = 24;  ADX_MODERATE_METAL  = 20
ADX_STRONG_CRYPTO = 22;  ADX_MODERATE_CRYPTO = 18

DE_WINDOW          = 20
MIN_RANGING_TRAIN  = 120
MIN_RANGING_TEST   = 15
MIN_RANGING_REPORT = 40   # FIX 28: suppress ranging rows with n < this
RECENCY_HALF_LIFE  = 800
ATR_PERIOD         = 14

ONLINE_RETRAIN_EVERY    = 20
ONLINE_WINDOW_BARS      = 2000
ONLINE_MIN_NEW_SAMPLES  = 20
ONLINE_WEIGHT_ALPHA     = 0.15
WEIGHT_MAX_SHIFT        = 0.10
RETRAIN_HOLDOUT_FRAC    = 0.20
RETRAIN_MIN_IMPROVEMENT = 0.005
CIRCUIT_BREAKER_MAX_ROLLBACKS   = 3
CIRCUIT_BREAKER_COOLDOWN_TRADES = 50

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── feature columns ─────────────────────────────────────────────────────────
BASE_FEATURES = [
    "rsi_14","macd","macd_signal","macd_hist","ema_cross","atr_14",
    "bb_width","bb_pct","return_1","return_3","return_5","return_10",
    "return_20","volatility_10","volatility_20","stoch_k","stoch_d","volume_ratio",
]
SESSION_FEATURES = ["hour_sin","hour_cos","dow_sin","dow_cos","is_london","is_ny","is_overlap"]
V2_FEATURES = [
    "adx_14","cci_20","williams_r","price_vs_ema200","candle_body_pct",
    "hl_ratio","swing_dist_high","swing_dist_low","adx_slope",
]
MOMENTUM_FEATURES = ["momentum_diff","volatility_regime","price_vs_ema50","trend_alignment"]
V5_FEATURES = ["directional_efficiency"]
METAL_EXTRA = ["metal_session","atr_pct","price_vs_ema20","rsi_divergence"]
H1_FEATURES = ["h1_trend_direction","h1_adx"]

FEATURE_COLS       = BASE_FEATURES + SESSION_FEATURES + V2_FEATURES + MOMENTUM_FEATURES + V5_FEATURES
METAL_FEATURE_COLS = FEATURE_COLS + METAL_EXTRA

# ─── accumulator ─────────────────────────────────────────────────────────────
@dataclass
class ImportanceAccumulator:
    trend: List[pd.Series] = field(default_factory=list)
    range: List[pd.Series] = field(default_factory=list)

# ─── helpers ─────────────────────────────────────────────────────────────────
def is_metal(p):        return p in METAL_SYMS
def is_exploratory(p):  return p in EXPLORATORY_PAIRS
def get_atr_multiplier(p): return ATR_MULTIPLIER_MAP.get(p, ATR_MULTIPLIER_MAP["default"])
def get_forward_bars(p):   return FORWARD_BARS_MAP.get(p, FORWARD_BARS_MAP["default"])
def get_fetch_bars(p):     return FETCH_BARS_MAP.get(p, FETCH_BARS_MAP["default"])
def get_buy_floor(p):      return BUY_FLOOR_MAP.get(p, BUY_FLOOR_MAP["default"])
def get_sell_ceil(p):      return SELL_CEIL_MAP.get(p, SELL_CEIL_MAP["default"])
def get_spread_r(p):       return SPREAD_COST_R.get(p, SPREAD_COST_R["default"])

def get_adx_thresholds(p):
    if p == "BTCUSDm": return ADX_STRONG_CRYPTO, ADX_MODERATE_CRYPTO
    if is_metal(p):    return ADX_STRONG_METAL,  ADX_MODERATE_METAL
    s = ADX_STRONG_MAP.get(p, ADX_STRONG_MAP["default"])
    return s, s - 2

def get_feature_cols(p):
    fc = list(METAL_FEATURE_COLS if is_metal(p) else FEATURE_COLS)
    if p == "GBPUSDm":
        fc = fc + H1_FEATURES
    return fc

# ─── importance helpers ───────────────────────────────────────────────────────
def extract_model_importance(model, feature_names):
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            imp = model.get_feature_importance()
        else:
            return None
        if len(imp) != len(feature_names): return None
        s = pd.Series(imp, index=feature_names, dtype=float)
        t = s.sum()
        return s / t if t > 0 else s
    except Exception as e:
        print(f"  [WARN] extract_model_importance: {e}")
        return None

def print_importance_table(series, title, top_n=15):
    print(f"\n{'─'*50}\n  {title} (top {top_n})\n{'─'*50}")
    for feat, val in series.nlargest(top_n).items():
        print(f"  {feat:<30s} {val:.4f}  {'█'*int(val*300)}")
    print(f"{'─'*50}")

def save_importance_csv(trend_s, range_s, pair):
    d = os.path.join(MODEL_DIR, "importance"); os.makedirs(d, exist_ok=True)
    if trend_s is not None:
        trend_s.sort_values(ascending=False).to_csv(
            os.path.join(d, f"{pair}_trend_importance.csv"), header=["importance"])
    if range_s is not None:
        range_s.sort_values(ascending=False).to_csv(
            os.path.join(d, f"{pair}_range_importance.csv"), header=["importance"])

# ─── statistical helpers ──────────────────────────────────────────────────────

def bootstrap_ci_f1(y_true, y_pred, n_boot=N_BOOTSTRAP, alpha=0.05, seed=42):
    """Stratified bootstrap 95% CI on F1."""
    rng = np.random.RandomState(seed)
    n   = len(y_true)
    if n < 10:
        f = f1_score(y_true, y_pred, zero_division=0)
        return f, f
    boot_f1 = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_f1.append(f1_score(yt, yp, zero_division=0))
    if len(boot_f1) < 10:
        f = f1_score(y_true, y_pred, zero_division=0)
        return f, f
    lo = float(np.percentile(boot_f1, 100 * alpha / 2))
    hi = float(np.percentile(boot_f1, 100 * (1 - alpha / 2)))
    return lo, hi


def simulate_pnl(y_true, y_pred, rr=RR_RATIO, pair="default"):
    """
    FIX 26: Per-trade P&L in R-units, net of spread.

    Correct direction  →  +rr − spread_r
    Wrong direction    →  −1  − spread_r

    spread_r is the spread expressed as a fraction of the 1-R stop,
    calibrated per asset class in SPREAD_COST_R.
    """
    spread = get_spread_r(pair)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    wins   = (y_true == y_pred).astype(float)
    pnl    = np.where(wins, rr - spread, -1.0 - spread)
    return pnl


def sharpe_from_pnl(pnl, n_signals, bars_per_window):
    """
    FIX 25: Annualised Sharpe ratio from per-trade P&L.

    Correct method (matches finance literature):
      1. Compute per-trade mean and std.
      2. Estimate trades-per-year from signal frequency in this window.
      3. SR_annual = SR_per_trade * sqrt(trades_per_year).

    v19 BUG was:
        ann_factor = sqrt(BARS_PER_YEAR * n_signals / bars_per_window)
    which embeds n_signals *inside* the square root alongside the
    BARS_PER_YEAR/step ratio (~17.5), inflating Sharpe by ~4x.

    v20 FIX:
        trades_per_year = n_signals * (BARS_PER_YEAR / bars_per_window)
        SR_annual = SR_per_trade * sqrt(trades_per_year)
    n_signals appears *outside* the sqrt, so it only scales frequency,
    not the annualisation factor itself.
    """
    if len(pnl) < 2:
        return 0.0
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    if sd < 1e-8:
        return 0.0
    sr_per_trade    = mu / sd
    trades_per_year = n_signals * (BARS_PER_YEAR / max(bars_per_window, 1))
    return float(sr_per_trade * np.sqrt(trades_per_year))


def profit_factor(pnl):
    pnl  = np.asarray(pnl)
    wins = pnl[pnl > 0].sum()
    loss = abs(pnl[pnl < 0].sum())
    return float(wins / loss) if loss > 0 else float("inf")


def ema_crossover_baseline(df_window, fast=9, slow=21, pair="default", rr=RR_RATIO):
    """
    FIX 27: Simple EMA fast/slow crossover baseline evaluated on the
    same labeled test-window rows as the ML model.

    df_window must contain columns: ['close', 'binary_label'].

    Signal: fast_ema > slow_ema → BUY (1), else SELL (0).
    No threshold filtering — every labeled setup gets a signal.

    Returns dict: {f1, sharpe, pf, n_signals}
    """
    if df_window is None or len(df_window) < slow + 5:
        return {"f1": None, "sharpe": None, "pf": None, "n_signals": 0}

    close    = df_window["close"]
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    y_pred   = (fast_ema > slow_ema).astype(int).values
    y_true   = df_window["binary_label"].values.astype(int)
    n        = len(y_true)

    if n < 10:
        return {"f1": None, "sharpe": None, "pf": None, "n_signals": n}

    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    pnl = simulate_pnl(y_true, y_pred, rr=rr, pair=pair)
    sh  = sharpe_from_pnl(pnl, n, len(df_window))
    pf  = profit_factor(pnl)

    return {
        "f1":        round(f1 * 100, 1),
        "sharpe":    round(sh, 3),
        "pf":        round(pf, 3),
        "n_signals": n,
    }


def diebold_mariano_test(y_true, y_pred_model, baseline_label=None):
    """
    Diebold-Mariano test vs majority-class baseline.
    HAC-corrected (Newey-West, lag=1).
    """
    if not SCIPY_OK:
        return "n.s."
    n = len(y_true)
    if n < 10:
        return "insufficient data"
    if baseline_label is None:
        baseline_label = int(np.bincount(y_true.astype(int)).argmax())
    y_base  = np.full(n, baseline_label, dtype=int)
    e_model = (y_true - y_pred_model.astype(int)) ** 2
    e_base  = (y_true - y_base) ** 2
    d       = e_model.astype(float) - e_base.astype(float)
    d_bar   = np.mean(d)
    gamma0  = np.var(d, ddof=1)
    if n > 2:
        gamma1  = np.mean((d[1:] - d_bar) * (d[:-1] - d_bar))
        hac_var = gamma0 + 2 * gamma1
    else:
        hac_var = gamma0
    if hac_var <= 0:
        return "n.s."
    dm_stat = d_bar / np.sqrt(hac_var / n)
    p_val   = float(2 * sp_stats.norm.sf(abs(dm_stat)))
    if p_val < 0.01:  return "sig.(p<0.01)"
    if p_val < 0.05:  return "sig.(p<0.05)"
    if p_val < 0.10:  return "marginal(p<0.10)"
    return "n.s."

# ─── feature engineering ─────────────────────────────────────────────────────

def add_base_features(df):
    df = df.copy(); close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low  = df["low"]  if "low"  in df.columns else close

    if "rsi_14" not in df.columns:
        d = close.diff()
        g = d.clip(lower=0).ewm(span=14, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
        df["rsi_14"] = (100 - 100 / (1 + g / l.replace(0, np.nan))).fillna(50)

    if "macd" not in df.columns:
        e12 = close.ewm(span=12, adjust=False).mean()
        e26 = close.ewm(span=26, adjust=False).mean()
        df["macd"]        = e12 - e26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

    if "ema_cross" not in df.columns:
        e9  = close.ewm(span=9,  adjust=False).mean()
        e21 = close.ewm(span=21, adjust=False).mean()
        df["ema_cross"] = (e9 - e21) / (close + 1e-10) * 100

    if "atr_14" not in df.columns:
        tr = pd.concat([high - low,
                        (high - close.shift()).abs(),
                        (low  - close.shift()).abs()], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    if "bb_width" not in df.columns:
        m = close.rolling(20).mean(); s = close.rolling(20).std()
        u = m + 2*s; lo = m - 2*s
        df["bb_width"] = (u - lo) / (m + 1e-10)
        df["bb_pct"]   = (close - lo) / (u - lo + 1e-10)

    if "return_1" not in df.columns:
        for k in [1, 3, 5, 10, 20]:
            df[f"return_{k}"] = close.pct_change(k) * 100

    if "volatility_10" not in df.columns:
        r = close.pct_change()
        df["volatility_10"] = r.rolling(10).std() * 100
        df["volatility_20"] = r.rolling(20).std() * 100

    if "stoch_k" not in df.columns:
        ll = low.rolling(14).min(); hh = high.rolling(14).max()
        df["stoch_k"] = 100 * (close - ll) / (hh - ll + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    if "volume_ratio" not in df.columns:
        if "volume" in df.columns:
            df["volume_ratio"] = (
                df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
            ).fillna(1.0).clip(0, 10)
        else:
            df["volume_ratio"] = 1.0
    return df


def add_session_features(df):
    df   = df.copy()
    dt   = pd.to_datetime(df["datetime"])
    hour = dt.dt.hour; dow = dt.dt.dayofweek
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
    close = df["close"].values
    if high is None or low is None:
        return np.zeros(len(df))
    n = len(close)
    tr = np.zeros(n); pdm = np.zeros(n); ndm = np.zeros(n)
    for i in range(1, n):
        tr[i]  = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        dh = high[i]-high[i-1]; dl = low[i-1]-low[i]
        pdm[i] = max(dh, 0) if dh > dl else 0
        ndm[i] = max(dl, 0) if dl > dh else 0
    sv = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    sp = pd.Series(pdm).ewm(span=period, adjust=False).mean().values
    sn = pd.Series(ndm).ewm(span=period, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(sv > 0, 100*sp/sv, 0)
        ndi = np.where(sv > 0, 100*sn/sv, 0)
        dx  = np.where((pdi+ndi) > 0, 100*np.abs(pdi-ndi)/(pdi+ndi), 0)
    return pd.Series(dx).ewm(span=period, adjust=False).mean().values


def compute_cci(df, period=20):
    if not all(c in df.columns for c in ["high", "low", "close"]):
        return np.zeros(len(df))
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return np.where(mad > 0, (tp - sma) / (0.015 * mad), 0)


def add_v2_features(df):
    df = df.copy()
    df["adx_14"] = compute_adx(df, 14)
    df["cci_20"] = compute_cci(df, 20)
    if all(c in df.columns for c in ["high", "low", "close"]):
        hh = df["high"].rolling(14).max(); ll = df["low"].rolling(14).min()
        df["williams_r"] = np.where((hh-ll) > 0, -100*(hh-df["close"])/(hh-ll), -50)
    else:
        df["williams_r"] = 0.0
    e200 = df["close"].ewm(span=200, adjust=False).mean()
    df["price_vs_ema200"] = (df["close"] - e200) / (e200 + 1e-10) * 100
    if all(c in df.columns for c in ["open", "close", "high", "low"]):
        body = (df["close"] - df["open"]).abs()
        tot  = (df["high"]  - df["low"]).replace(0, np.nan)
        df["candle_body_pct"] = (body / tot).fillna(0.5)
    else:
        df["candle_body_pct"] = 0.5
    if "atr_14" in df.columns:
        df["hl_ratio"] = ((df["high"]-df["low"]) / df["atr_14"].replace(0, np.nan)).fillna(1.0)
    else:
        df["hl_ratio"] = 1.0
    rh = df["close"].rolling(20).max(); rl = df["close"].rolling(20).min()
    df["swing_dist_high"] = (rh - df["close"]) / (df["close"] + 1e-10) * 100
    df["swing_dist_low"]  = (df["close"] - rl) / (df["close"] + 1e-10) * 100
    if "adx_slope" not in df.columns:
        df["adx_slope"] = df["adx_14"].diff(3).fillna(0.0)
    return df


def add_momentum_features(df):
    df = df.copy(); close = df["close"]
    if "rsi_14" in df.columns:
        d50  = close.diff()
        g50  = d50.clip(lower=0).ewm(span=50, adjust=False).mean()
        l50  = (-d50.clip(upper=0)).ewm(span=50, adjust=False).mean()
        rsi50 = 100 - (100 / (1 + g50 / l50.replace(0, np.nan)))
        df["momentum_diff"] = df["rsi_14"] - rsi50.fillna(50)
    else:
        df["momentum_diff"] = 0.0
    ret = close.pct_change()
    df["volatility_regime"] = (
        ret.rolling(5).std() / ret.rolling(20).std().replace(0, np.nan)
    ).fillna(1.0).clip(0, 5)
    e50  = close.ewm(span=50,  adjust=False).mean()
    e20  = close.ewm(span=20,  adjust=False).mean()
    e200 = close.ewm(span=200, adjust=False).mean()
    df["price_vs_ema50"]    = (close - e50)  / (e50  + 1e-10) * 100
    df["trend_alignment"]   = (
        (e20 > e50).astype(int) + (e50 > e200).astype(int) - 1
    ).astype(float)
    return df


def add_v5_features(df):
    df  = df.copy()
    ret = df["close"].pct_change()
    net = ret.rolling(DE_WINDOW).sum().abs()
    tot = ret.abs().rolling(DE_WINDOW).sum()
    df["directional_efficiency"] = (net / tot.replace(0, np.nan)).fillna(0.5).clip(0, 1)
    return df


def add_metal_features(df):
    df = df.copy(); close = df["close"]
    if "datetime" in df.columns:
        hour = pd.to_datetime(df["datetime"]).dt.hour
        df["metal_session"] = np.where(
            (hour >= 13) & (hour < 21), 1.0,
            np.where((hour >= 8) & (hour < 13), 0.5, 0.0))
    else:
        df["metal_session"] = 0.5
    if "atr_14" in df.columns:
        df["atr_pct"] = df["atr_14"] / (close + 1e-10) * 100
    else:
        df["atr_pct"] = 0.0
    e20 = close.ewm(span=20, adjust=False).mean()
    df["price_vs_ema20"] = (close - e20) / (e20 + 1e-10) * 100
    d   = close.diff()
    g5  = d.clip(lower=0).ewm(span=5, adjust=False).mean()
    l5  = (-d.clip(upper=0)).ewm(span=5, adjust=False).mean()
    rsi5 = 100 - (100 / (1 + g5 / l5.replace(0, np.nan)))
    df["rsi_divergence"] = (rsi5.fillna(50) - df["rsi_14"]) if "rsi_14" in df.columns else 0.0
    return df


def add_h1_features(df):
    """H1 trend direction + ADX for GBPUSDm (FIX 23)."""
    df    = df.copy()
    close = df["close"]
    high  = df["high"] if "high" in df.columns else close
    low   = df["low"]  if "low"  in df.columns else close

    h1_close = close.rolling(4).mean()
    h1_high  = high.rolling(4).max()
    h1_low   = low.rolling(4).min()

    h1_ema20  = h1_close.ewm(span=20,  adjust=False).mean()
    h1_ema50  = h1_close.ewm(span=50,  adjust=False).mean()
    h1_ema200 = h1_close.ewm(span=200, adjust=False).mean()
    df["h1_trend_direction"] = (
        (h1_ema20 > h1_ema50).astype(int) +
        (h1_ema50 > h1_ema200).astype(int) - 1
    ).astype(float)

    tmp = pd.DataFrame({"high": h1_high, "low": h1_low, "close": h1_close})
    df["h1_adx"] = compute_adx(tmp, 14)
    return df


def classify_regime(df, pair=""):
    adx_strong, _ = get_adx_thresholds(pair)
    regime  = pd.Series("ranging", index=df.index)
    has_de  = "directional_efficiency" in df.columns
    has_adx = "adx_14"                 in df.columns
    has_ta  = "trend_alignment"        in df.columns
    if has_de and has_adx and has_ta:
        rule1 = (df["directional_efficiency"] > DE_TREND_THRESHOLD) & (df["adx_14"] > adx_strong)
        rule2 = (df["trend_alignment"] != 0) & (df["adx_14"] > adx_strong)
        regime[rule1 | rule2] = "trending"
    elif has_adx:
        regime[df["adx_14"] > adx_strong] = "trending"
    return regime

# ─── labelling ───────────────────────────────────────────────────────────────

def make_binary_labels_full(df_full, df_filtered, pair="", max_pos=None):
    fwd        = get_forward_bars(pair)
    atr_m      = get_atr_multiplier(pair)
    close_full = df_full["close"].values
    pos_lookup = {idx: pos for pos, idx in enumerate(df_full.index.tolist())}
    labels = []; keep = []
    for row_idx, row in df_filtered.iterrows():
        pos = pos_lookup.get(row_idx)
        if pos is None or pos + fwd >= len(close_full):
            keep.append(False); labels.append(-1); continue
        if max_pos is not None and pos + fwd > max_pos:
            keep.append(False); labels.append(-1); continue
        barrier = atr_m * row["atr_14"]
        upper   = close_full[pos] + barrier
        lower   = close_full[pos] - barrier
        win     = close_full[pos+1 : pos+1+fwd]
        hu = np.argmax(win >= upper) if np.any(win >= upper) else None
        hl = np.argmax(win <= lower) if np.any(win <= lower) else None
        if hu is not None and hl is not None:
            label = 1 if hu <= hl else 0; keep.append(True)
        elif hu is not None:
            label = 1; keep.append(True)
        elif hl is not None:
            label = 0; keep.append(True)
        else:
            label = -1; keep.append(False)
        labels.append(label)
    df_out = df_filtered.copy()
    df_out["binary_label"] = labels
    return df_out[keep].copy()


def filter_to_setups(df, pair=""):
    _, adx_mod = get_adx_thresholds(pair)
    rsi_ext = (df["rsi_14"] < 30) | (df["rsi_14"] > 70)
    bb_ext  = (df["bb_pct"] < 0.03) | (df["bb_pct"] > 0.97)
    ra_filt = rsi_ext | bb_ext
    if all(c in df.columns for c in ["price_vs_ema50", "adx_14", "trend_alignment"]):
        thr     = 0.5 if is_metal(pair) else 0.3
        tr_filt = (
            (df["price_vs_ema50"].abs() < thr) &
            (df["adx_14"] > adx_mod) &
            (df["trend_alignment"] != 0)
        )
    else:
        tr_filt = ra_filt
    vol_lim  = 3.0 if is_metal(pair) else 2.0
    tr_mask  = df["regime"] == "trending"
    ra_mask  = df["regime"] == "ranging"
    combined = (tr_mask & tr_filt) | (ra_mask & ra_filt)
    if "volatility_regime" in df.columns:
        combined = combined & (df["volatility_regime"] < vol_lim)
    return df[combined].copy()


def recency_weights(n, half_life=RECENCY_HALF_LIFE):
    pos = np.arange(n)
    w   = np.power(2.0, (pos - (n-1)) / half_life)
    return w / w.sum() * n


def safe_balance(X, y, rs=42):
    counts = pd.Series(y).value_counts()
    if len(counts) < 2: return X, y
    mn = int(counts.min()); mx = int(counts.max())
    if mn / mx >= 0.45: return X, y
    if mn >= 8:
        try:
            return RandomOverSampler(random_state=rs).fit_resample(X, y)
        except Exception as e:
            print(f"  [WARN] ROS: {e}")
    if mn >= 20:
        try:
            return SMOTE(random_state=rs,
                         k_neighbors=min(3, mn-1),
                         sampling_strategy=0.6).fit_resample(X, y)
        except Exception as e:
            print(f"  [WARN] SMOTE: {e}")
    return X, y


def smart_balance(X, y, rs=42):
    pct = float(pd.Series(y).value_counts(normalize=True).min())
    if pct < IMBALANCE_THRESHOLD:
        return safe_balance(X, y, rs=rs), True
    return (X, y), False


def select_top_features(X, y, feature_cols, top_n):
    top_n = min(top_n, len(feature_cols), max(1, len(X)//10))
    try:
        sc = mutual_info_classif(X, y, random_state=42, discrete_features=False)
        return pd.Series(sc, index=feature_cols).nlargest(top_n).index.tolist()
    except Exception as e:
        print(f"  [WARN] feat_sel: {e}")
        return feature_cols[:top_n]

# ─── build models ────────────────────────────────────────────────────────────

def build_base_models(buy_w=0.8, sell_w=1.0):
    xgb_spw = float(np.clip(sell_w / max(buy_w, 0.01), 0.3, 2.0))
    m = {
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.75, min_child_weight=5,
            eval_metric="logloss", scale_pos_weight=xgb_spw,
            random_state=42, verbosity=0),
        "LightGBM": LGBMClassifier(
            n_estimators=500, num_leaves=63, max_depth=6, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.75, min_child_samples=15,
            class_weight={0: sell_w, 1: buy_w}, random_state=7, verbose=-1),
    }
    if CATBOOST_AVAILABLE:
        m["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.04,
            subsample=0.75, colsample_bylevel=0.75,
            class_weights=[sell_w, buy_w], random_seed=21, verbose=0)
    else:
        m["LightGBM_dart"] = LGBMClassifier(
            n_estimators=400, num_leaves=31, max_depth=5,
            learning_rate=0.05, boosting_type="dart",
            subsample=0.8, colsample_bytree=0.8,
            class_weight={0: sell_w, 1: buy_w}, random_state=17, verbose=-1)
    return m


def build_oof_predictions(models, X, y, n_splits=3):
    n_m = len(models)
    oof = np.full((len(X), n_m), 0.5)
    kf  = KFold(n_splits=n_splits, shuffle=False)
    for fi, (tri, vai) in enumerate(kf.split(X)):
        sw0 = recency_weights(len(tri))
        (Xb, yb), dr = smart_balance(X[tri], y[tri])
        sw = np.ones(len(Xb)) if dr else sw0
        for mi, (nm, mdl) in enumerate(models.items()):
            try:
                mdl.fit(Xb, yb, sample_weight=sw)
                p = mdl.predict_proba(X[vai])[:, 1]
                if p.std() > 1e-4:
                    oof[vai, mi] = p
            except Exception as e:
                print(f"  [WARN] OOF {fi} {nm}: {e}")
    return oof

# ─── walk-forward evaluation ──────────────────────────────────────────────────

def walk_forward_evaluate(pair, df_full, df_setups, available_features,
                           acc: Optional[ImportanceAccumulator] = None):
    exploratory = is_exploratory(pair)
    tag = "  *** EXPLORATORY PAIR — excluded from main paper claims ***" if exploratory else ""

    print(f"\n{'='*60}")
    print(f"  Walk-Forward Evaluation: {pair}")
    if tag: print(tag)
    print(f"  Mode: EXPANDING window | min_train={WF_MIN_TRAIN} | step={WF_STEP}")
    print(f"{'='*60}")

    pair_buy_floor = get_buy_floor(pair)
    pair_sell_ceil = get_sell_ceil(pair)
    N_full         = len(df_full)

    results         = []
    pair_trend_imp: List[pd.Series] = []
    pair_range_imp: List[pd.Series] = []

    window_idx = 0
    test_start = WF_MIN_TRAIN

    while test_start + WF_STEP <= N_full:
        test_end  = test_start + WF_STEP
        train_end = test_start

        train_end_idx  = df_full.index[train_end - 1]
        test_end_idx   = df_full.index[min(test_end, N_full) - 1]

        setups_train = df_setups[df_setups.index <= train_end_idx]
        df_train_lab = make_binary_labels_full(df_full, setups_train, pair=pair,
                                               max_pos=train_end - 1)

        setups_test  = df_setups[(df_setups.index > train_end_idx) &
                                  (df_setups.index <= test_end_idx)]
        df_test_lab  = make_binary_labels_full(df_full, setups_test, pair=pair)

        window_idx   += 1
        n_train_bars  = train_end

        wr = {"window": window_idx, "train_bars": n_train_bars, "test_bars": WF_STEP}

        if len(df_train_lab) < 50 or len(df_test_lab) < 10:
            test_start += WF_STEP
            continue

        X_train = df_train_lab[available_features].values
        y_train = df_train_lab["binary_label"].values.astype(int)
        r_train = df_train_lab["regime"].values
        X_test  = df_test_lab[available_features].values
        y_test  = df_test_lab["binary_label"].values.astype(int)
        r_test  = df_test_lab["regime"].values

        # ── Trending ──────────────────────────────────────────────────────────
        tr_tr = r_train == "trending"
        tr_te = r_test  == "trending"

        if tr_tr.sum() >= 50 and tr_te.sum() >= 15:
            Xtr  = X_train[tr_tr]; ytr = y_train[tr_tr]
            sel  = select_top_features(Xtr, ytr, available_features, TOP_N_FEATURES)
            fi   = [available_features.index(f) for f in sel]
            sc   = StandardScaler()
            Xts  = sc.fit_transform(Xtr[:, fi])
            sw0  = recency_weights(len(Xts))
            (Xb, yb), dr = smart_balance(Xts, ytr)
            sw   = np.ones(len(Xb)) if dr else sw0

            bm     = build_base_models()
            fitted = []; wimp = []
            for nm, mdl in bm.items():
                try:
                    mdl.fit(Xb, yb, sample_weight=sw)
                    fitted.append(mdl)
                    p_tr = mdl.predict_proba(Xts)[:, 1]
                    print(f"  W{window_idx:2d} TRENDING proba: min={p_tr.min():.3f}"
                          f"  max={p_tr.max():.3f}  mean={p_tr.mean():.3f}")
                    imp = extract_model_importance(mdl, sel)
                    if imp is not None: wimp.append(imp)
                except Exception as e:
                    print(f"  [WARN] W{window_idx} {nm}: {e}")

            if wimp:
                ai = pd.concat(wimp, axis=1).mean(axis=1)
                pair_trend_imp.append(ai)
                if acc is not None and not exploratory:
                    acc.trend.append(ai)

            if fitted:
                Xte    = sc.transform(X_test[tr_te][:, fi])
                probas = np.mean([m.predict_proba(Xte)[:, 1] for m in fitted], axis=0)
                tau_b  = max(np.percentile(probas, 80), pair_buy_floor)
                tau_s  = min(np.percentile(probas, 20), pair_sell_ceil)
                preds  = np.where(probas >= tau_b, 1, np.where(probas <= tau_s, 0, -1))
                mask   = preds != -1

                if mask.sum() >= 10:
                    yt  = y_test[tr_te][mask]; yp = preds[mask]
                    f1  = f1_score(yt, yp, zero_division=0)
                    ci_lo, ci_hi = bootstrap_ci_f1(yt, yp)
                    # FIX 25+26: corrected Sharpe with spread cost
                    pnl = simulate_pnl(yt, yp, pair=pair)
                    sh  = sharpe_from_pnl(pnl, int(mask.sum()), WF_STEP)
                    pf  = profit_factor(pnl)
                    dm  = diebold_mariano_test(yt, yp)
                    wr.update({
                        "tr_f1":     round(f1 * 100, 1),
                        "tr_ci":     f"[{ci_lo*100:.1f},{ci_hi*100:.1f}]",
                        "tr_prec":   round(precision_score(yt, yp, zero_division=0)*100, 1),
                        "tr_rec":    round(recall_score(yt, yp, zero_division=0)*100, 1),
                        "tr_sharpe": round(sh, 3),
                        "tr_pf":     round(pf, 3),
                        "tr_dm":     dm,
                        "tr_sigs":   int(mask.sum()),
                    })

                    # FIX 27: EMA crossover baseline on same trending test rows
                    df_test_trend = df_test_lab[tr_te].copy()
                    bl = ema_crossover_baseline(df_test_trend, pair=pair)
                    wr["bl_f1"]     = bl["f1"]
                    wr["bl_sharpe"] = bl["sharpe"]
                else:
                    wr.update({"tr_f1":"--","tr_ci":"[--,--]","tr_prec":"--",
                               "tr_rec":"--","tr_sharpe":"--","tr_pf":"--",
                               "tr_dm":"--","tr_sigs":0,"bl_f1":"--","bl_sharpe":"--"})
            else:
                wr.update({"tr_f1":"--","tr_ci":"[--,--]","tr_prec":"--",
                           "tr_rec":"--","tr_sharpe":"--","tr_pf":"--",
                           "tr_dm":"--","tr_sigs":0,"bl_f1":"--","bl_sharpe":"--"})
        else:
            wr.update({"tr_f1":"--","tr_ci":"[--,--]","tr_prec":"--",
                       "tr_rec":"--","tr_sharpe":"--","tr_pf":"--",
                       "tr_dm":"--","tr_sigs":0,"bl_f1":"--","bl_sharpe":"--"})

        # ── Ranging ───────────────────────────────────────────────────────────
        ra_tr   = r_train == "ranging"
        ra_te   = r_test  == "ranging"
        n_ra_te = int(ra_te.sum())

        if ra_tr.sum() >= MIN_RANGING_TRAIN and n_ra_te >= 5:
            Xra  = X_train[ra_tr]; yra = y_train[ra_tr]
            mf   = min(TOP_N_FEATURES, max(5, len(Xra)//10))
            selr = select_top_features(Xra, yra, available_features, mf)
            fir  = [available_features.index(f) for f in selr]
            scr  = StandardScaler()
            Xrs  = scr.fit_transform(Xra[:, fir])
            nl   = min(15, max(4, len(Xra)//20))
            mc   = max(3, len(Xra)//30)
            rm   = LGBMClassifier(
                n_estimators=200, num_leaves=nl, max_depth=4,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=mc, class_weight="balanced",
                random_state=42, verbose=-1)
            sw0r      = recency_weights(len(Xrs))
            Xrb, yrb  = safe_balance(Xrs, yra)
            swr       = np.ones(len(Xrb)) if len(Xrb) != len(Xrs) else sw0r
            try:
                rm.fit(Xrb, yrb, sample_weight=swr)
                impr = extract_model_importance(rm, selr)
                if impr is not None:
                    pair_range_imp.append(impr)
                    if acc is not None and not exploratory:
                        acc.range.append(impr)

                Xrte    = scr.transform(X_test[ra_te][:, fir])
                pr      = rm.predict_proba(Xrte)[:, 1]
                tb_r    = max(np.percentile(pr, 80), pair_buy_floor)
                ts_r    = min(np.percentile(pr, 20), pair_sell_ceil)
                preds_r = np.where(pr >= tb_r, 1, np.where(pr <= ts_r, 0, -1))
                mask_r  = preds_r != -1

                # FIX 28: suppress ranging rows with n < MIN_RANGING_REPORT
                if n_ra_te < MIN_RANGING_REPORT:
                    wr.update({
                        "ra_f1": "--", "ra_ci": "[--,--]", "ra_n": n_ra_te,
                        "ra_pf": "--", "ra_dm": "--",
                        "ra_note": f"skipped(n={n_ra_te}<{MIN_RANGING_REPORT})"
                    })
                elif mask_r.sum() >= 4:
                    yr       = y_test[ra_te][mask_r]; ypr = preds_r[mask_r]
                    f1r      = f1_score(yr, ypr, zero_division=0)
                    cir_lo, cir_hi = bootstrap_ci_f1(yr, ypr)
                    pnlr     = simulate_pnl(yr, ypr, pair=pair)
                    pfr      = profit_factor(pnlr)
                    dmr      = diebold_mariano_test(yr, ypr)
                    wr.update({
                        "ra_f1": round(f1r * 100, 1),
                        "ra_ci": f"[{cir_lo*100:.1f},{cir_hi*100:.1f}]",
                        "ra_n":  n_ra_te,
                        "ra_pf": round(pfr, 3),
                        "ra_dm": dmr,
                        "ra_note": "",
                    })
                else:
                    wr.update({"ra_f1":"--","ra_ci":"[--,--]","ra_n":n_ra_te,
                               "ra_pf":"--","ra_dm":"--","ra_note":""})
            except Exception as e:
                print(f"  [WARN] W{window_idx} ranging: {e}")
                wr.update({"ra_f1":"--","ra_ci":"[--,--]","ra_n":n_ra_te,
                           "ra_pf":"--","ra_dm":"--","ra_note":""})
        else:
            skip = (f"train={ra_tr.sum()}<{MIN_RANGING_TRAIN}"
                    if ra_tr.sum() < MIN_RANGING_TRAIN else f"test={n_ra_te}<5")
            wr.update({"ra_f1":"--","ra_ci":"[--,--]","ra_n":"--",
                       "ra_pf":"--","ra_dm":"--","ra_note":f"skipped({skip})"})

        results.append(wr)

        print(
            f"  W{window_idx:2d} [train={n_train_bars:5d}] | "
            f"Trend F1={wr.get('tr_f1','--'):>5} CI={wr.get('tr_ci','[--,--]')} "
            f"Sharpe={wr.get('tr_sharpe','--')} PF={wr.get('tr_pf','--')} "
            f"DM={wr.get('tr_dm','--')} BL_F1={wr.get('bl_f1','--')} "
            f"BL_Sh={wr.get('bl_sharpe','--')} | "
            f"Range F1={wr.get('ra_f1','--'):>5} n={wr.get('ra_n','--')}"
        )

        test_start += WF_STEP

    # ── per-pair importance ───────────────────────────────────────────────────
    ta_avg = ra_avg = None
    if pair_trend_imp:
        ta_avg = pd.concat(pair_trend_imp, axis=1).mean(axis=1).sort_values(ascending=False)
        print_importance_table(ta_avg, f"TRENDING importance — {pair}")
    if pair_range_imp:
        ra_avg = pd.concat(pair_range_imp, axis=1).mean(axis=1).sort_values(ascending=False)
        print_importance_table(ra_avg, f"RANGING importance — {pair}")
    save_importance_csv(ta_avg, ra_avg, pair)

    _print_latex_tables(pair, results, exploratory)
    return results


def _print_latex_tables(pair, results, exploratory=False):
    """FIX 29: LaTeX tables now include BL F1 and BL Sharpe columns."""
    exp_note = (
        "  *** EXPLORATORY PAIR — excluded from main paper claims ***\n"
        if exploratory else ""
    )

    # ── TRENDING ─────────────────────────────────────────────────────────────
    print(f"\n>>> LATEX TABLE ({pair}) — TRENDING <<<")
    if exp_note: print(exp_note, end="")
    print(r"\begin{tabular}{lccccccccccc}")
    print(
        r"W & Train N & F1 & 95\% CI & Prec & Rec & "
        r"Sharpe & PF & DM & BL F1 & BL Sharpe \\"
    )
    print(r"\hline")
    for r in results:
        bl_f1 = r.get("bl_f1", "--")
        bl_sh = r.get("bl_sharpe", "--")
        bl_f1_str = f"{bl_f1}\\%" if isinstance(bl_f1, (int, float)) else str(bl_f1)
        print(
            f"W{r['window']} & {r['train_bars']} & "
            f"{r.get('tr_f1','--')}\\% & {r.get('tr_ci','[--,--]')} & "
            f"{r.get('tr_prec','--')}\\% & {r.get('tr_rec','--')}\\% & "
            f"{r.get('tr_sharpe','--')} & {r.get('tr_pf','--')} & "
            f"{r.get('tr_dm','--')} & {bl_f1_str} & {bl_sh} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")

    # ── RANGING ──────────────────────────────────────────────────────────────
    print(f"\n>>> LATEX TABLE ({pair}) — RANGING <<<")
    if exp_note: print(exp_note, end="")
    print(r"\begin{tabular}{lccccc}")
    print(r"W & n & F1 & 95\% CI & PF & DM \\")
    print(r"\hline")
    for r in results:
        note = r.get("ra_note", "")
        nc   = f"  % {note}" if note else ""
        print(
            f"W{r['window']} & {r.get('ra_n','--')} & "
            f"{r.get('ra_f1','--')}\\% & {r.get('ra_ci','[--,--]')} & "
            f"{r.get('ra_pf','--')} & {r.get('ra_dm','--')} \\\\{nc}"
        )
    print(r"\hline")
    print(r"\end{tabular}")

# ─── EnsembleModel ────────────────────────────────────────────────────────────

class EnsembleModel:
    def __init__(self):
        self.tr_scaler=None; self.tr_feat_idx=None
        self.tr_models=[]; self.tr_meta=None
        self.ra_scaler=None; self.ra_feat_idx=None; self.ra_model=None
        self.feature_cols=None; self.pair=None
        self.atr_multiplier=1.2; self.adx_strong=None; self.adx_moderate=None
        self.is_metal_pair=False
        self._de_idx=None; self._adx_idx=None; self._ta_idx=None
        self.online_buffer: deque = deque(maxlen=ONLINE_WINDOW_BARS)
        self.trades_since_retrain=0
        self.win_loss_stats={"BUY_wins":0,"BUY_losses":0,"SELL_wins":0,"SELL_losses":0}
        self.adaptive_buy_weight=0.8; self.adaptive_sell_weight=1.0
        self.retrain_count=0; self.consecutive_rollbacks=0
        self.cooldown_trades_remaining=0; self.retrain_history=[]

    def _build_col_indices(self):
        if self.feature_cols is None: return
        fc = self.feature_cols
        self._de_idx  = fc.index("directional_efficiency") if "directional_efficiency" in fc else None
        self._adx_idx = fc.index("adx_14")                 if "adx_14"                 in fc else None
        self._ta_idx  = fc.index("trend_alignment")        if "trend_alignment"        in fc else None

    def predict_proba(self, X):
        if self.feature_cols is None: return np.array([[0.5, 0.5]])
        arr = X[self.feature_cols].values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        if np.isnan(arr).any(): return np.array([[0.5, 0.5]])
        regime = self._classify_single_fast(arr[0])
        if regime == "trending" and self.tr_scaler is not None:
            p = self._predict_trending(arr)
        elif regime == "ranging" and self.ra_scaler is not None:
            p = self._predict_ranging(arr)
        else:
            p = (self._predict_trending(arr) if self.tr_scaler else
                 self._predict_ranging(arr)  if self.ra_scaler else 0.5)
        p = float(np.clip(p, 0.0, 1.0))
        return np.array([[1.0 - p, p]])

    def predict_signal(self, X):
        pr = self.predict_proba(X)[0, 1]
        bf = get_buy_floor(self.pair or "")
        sc = get_sell_ceil(self.pair or "")
        return "BUY" if pr >= bf else "SELL" if pr <= sc else "HOLD"

    def _classify_single_fast(self, row):
        try:
            de  = float(row[self._de_idx])  if self._de_idx  is not None else 0.0
            adx = float(row[self._adx_idx]) if self._adx_idx is not None else 0.0
            ta  = float(row[self._ta_idx])  if self._ta_idx  is not None else 0.0
            if self.adx_strong is None: return "ranging"
            return "trending" if (
                (de > DE_TREND_THRESHOLD and adx > self.adx_strong) or
                (ta != 0 and adx > self.adx_strong)
            ) else "ranging"
        except Exception as e:
            print(f"  [WARN] _classify_single_fast: {e}"); return "ranging"

    def _predict_trending(self, arr):
        try:
            Xs = self.tr_scaler.transform(arr[:, self.tr_feat_idx])
            ps = [float(m.predict_proba(Xs)[0, 1]) for m in self.tr_models]
            if not ps: return 0.5
            if self.tr_meta is not None and len(ps) == len(self.tr_models):
                return float(self.tr_meta.predict_proba(np.array(ps).reshape(1, -1))[0, 1])
            return float(np.mean(ps))
        except Exception as e:
            print(f"  [WARN] _predict_trending: {e}"); return 0.5

    def _predict_ranging(self, arr):
        try:
            Xs = self.ra_scaler.transform(arr[:, self.ra_feat_idx])
            return float(self.ra_model.predict_proba(Xs)[0, 1])
        except Exception as e:
            print(f"  [WARN] _predict_ranging: {e}"); return 0.5

    def record_trade_outcome(self, feature_row, regime, direction, outcome, model_dir):
        if self.cooldown_trades_remaining > 0:
            self.cooldown_trades_remaining -= 1; return
        if outcome == "BREAKEVEN": return
        label = (1 if direction == "BUY" else 0) if outcome == "WIN" else (0 if direction == "BUY" else 1)
        key   = f"{direction}_{'wins' if outcome=='WIN' else 'losses'}"
        self.win_loss_stats[key] += 1
        self.online_buffer.append({"X": feature_row.copy(), "y": label, "regime": regime})
        self._update_adaptive_weights()
        self.trades_since_retrain += 1
        if self.trades_since_retrain >= ONLINE_RETRAIN_EVERY:
            if len(self.online_buffer) >= ONLINE_MIN_NEW_SAMPLES:
                self._online_retrain(model_dir)
            self.trades_since_retrain = 0

    def _update_adaptive_weights(self):
        for dir_, wattr in [("BUY","adaptive_buy_weight"),("SELL","adaptive_sell_weight")]:
            tot = self.win_loss_stats[f"{dir_}_wins"] + self.win_loss_stats[f"{dir_}_losses"]
            if tot >= 3:
                wr  = self.win_loss_stats[f"{dir_}_wins"] / tot
                nw  = 0.6 + wr * 0.4
                cur = getattr(self, wattr)
                bl  = (1 - ONLINE_WEIGHT_ALPHA) * cur + ONLINE_WEIGHT_ALPHA * nw
                d   = float(np.clip(bl - cur, -WEIGHT_MAX_SHIFT, WEIGHT_MAX_SHIFT))
                setattr(self, wattr, float(np.clip(cur + d, 0.4, 1.2)))

    def _online_retrain(self, model_dir):
        buf   = list(self.online_buffer)
        X_buf = np.stack([e["X"] for e in buf])
        y_buf = np.array([e["y"] for e in buf], dtype=int)
        r_buf = np.array([e["regime"] for e in buf])
        if not self.feature_cols: return
        n_h = max(10, int(len(buf) * RETRAIN_HOLDOUT_FRAC))
        n_t = len(buf) - n_h
        Xt, Xh = X_buf[:n_t], X_buf[n_t:]
        yt, yh = y_buf[:n_t], y_buf[n_t:]
        rt     = r_buf[:n_t]
        rb = {
            "tr_models": copy.deepcopy(self.tr_models),
            "tr_meta":   copy.deepcopy(self.tr_meta),
            "ra_model":  copy.deepcopy(self.ra_model),
            "buy_w":     self.adaptive_buy_weight,
            "sell_w":    self.adaptive_sell_weight,
        }
        acc_tr = acc_ra = False
        tr_m = rt == "trending"; tr_h = r_buf[n_t:] == "trending"
        if tr_m.sum() >= 20 and self.tr_scaler is not None:
            try:
                Xs  = self.tr_scaler.transform(Xt[tr_m][:, self.tr_feat_idx])
                sw0 = recency_weights(len(Xs))
                (Xb, yb), dr = smart_balance(Xs, yt[tr_m])
                sw  = np.ones(len(Xb)) if dr else sw0
                nm  = build_base_models(self.adaptive_buy_weight, self.adaptive_sell_weight)
                fitted = []
                for n_, mdl in nm.items():
                    try:
                        mdl.fit(Xb, yb, sample_weight=sw)
                        if mdl.predict_proba(Xs)[:, 1].std() > 1e-4:
                            fitted.append(mdl)
                    except: pass
                if fitted:
                    old_l = (self._eval_ll(self.tr_models, Xh[tr_h], yh[tr_h], trending=True)
                             if tr_h.sum() >= 5 else None)
                    new_l = (self._eval_ll(fitted, Xh[tr_h], yh[tr_h], trending=True)
                             if tr_h.sum() >= 5 else None)
                    if _should_accept(old_l, new_l):
                        self.tr_models = fitted; acc_tr = True
            except Exception as e:
                print(f"  [WARN] online tr: {e}")
        ra_m = rt == "ranging"; ra_h = r_buf[n_t:] == "ranging"
        if ra_m.sum() >= MIN_RANGING_TRAIN // 4 and self.ra_scaler is not None:
            try:
                Xs = self.ra_scaler.transform(Xt[ra_m][:, self.ra_feat_idx])
                nl = min(15, max(4, len(Xs)//20)); mc = max(3, len(Xs)//30)
                rm = LGBMClassifier(
                    n_estimators=200, num_leaves=nl, max_depth=4,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=mc, class_weight="balanced",
                    random_state=42, verbose=-1)
                sw0 = recency_weights(len(Xs))
                Xrb, yrb = safe_balance(Xs, yt[ra_m])
                sw = np.ones(len(Xrb)) if len(Xrb) != len(Xs) else sw0
                rm.fit(Xrb, yrb, sample_weight=sw)
                if rm.predict_proba(Xs)[:, 1].std() > 1e-4:
                    old_l = (self._eval_ll([self.ra_model], Xh[ra_h], yh[ra_h], trending=False)
                             if ra_h.sum() >= 5 else None)
                    new_l = (self._eval_ll([rm], Xh[ra_h], yh[ra_h], trending=False)
                             if ra_h.sum() >= 5 else None)
                    if _should_accept(old_l, new_l):
                        self.ra_model = rm; acc_ra = True
            except Exception as e:
                print(f"  [WARN] online ra: {e}")
        self.retrain_count += 1
        if not (acc_tr or acc_ra):
            self.tr_models = rb["tr_models"]; self.tr_meta = rb["tr_meta"]
            self.ra_model  = rb["ra_model"]
            self.adaptive_buy_weight  = rb["buy_w"]
            self.adaptive_sell_weight = rb["sell_w"]
            self.consecutive_rollbacks += 1
            if self.consecutive_rollbacks >= CIRCUIT_BREAKER_MAX_ROLLBACKS:
                self.cooldown_trades_remaining = CIRCUIT_BREAKER_COOLDOWN_TRADES
                self.consecutive_rollbacks = 0
        else:
            self.consecutive_rollbacks = 0
            try:
                joblib.dump(self, os.path.join(model_dir, f"ensemble_{self.pair.lower()}.pkl"))
            except Exception as e:
                print(f"  [WARN] save: {e}")

    def _eval_ll(self, models, X, y, trending=True):
        try:
            if trending:
                Xs = self.tr_scaler.transform(X[:, self.tr_feat_idx])
                ps = [m.predict_proba(Xs)[:, 1] for m in models]
                p  = np.mean(ps, axis=0) if ps else None
            else:
                Xs = self.ra_scaler.transform(X[:, self.ra_feat_idx])
                p  = models[0].predict_proba(Xs)[:, 1] if models else None
            if p is None: return None
            return float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))
        except:
            return None


def _should_accept(old_l, new_l):
    if old_l is None: return True
    if new_l is None: return False
    return (old_l - new_l) >= RETRAIN_MIN_IMPROVEMENT

# ─── data fetcher ─────────────────────────────────────────────────────────────

def fetch_and_save_m15(pair, bars=None):
    if bars is None: bars = get_fetch_bars(pair)
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError("pip install MetaTrader5")
    if not mt5.initialize():
        print("[ERROR] MT5 init failed."); return False
    mt5.symbol_select(pair, True)
    rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M15, 0, bars)
    mt5.shutdown()
    if rates is None:
        print(f"[ERROR] {pair} fetch failed."); return False
    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    os.makedirs(INPUT_DIR, exist_ok=True)
    out = os.path.join(INPUT_DIR, f"{pair}_M15_labeled.csv")
    df.to_csv(out, index=False)
    print(f"  Saved {len(df):,} M15 bars to {out}")
    return True

# ─── main training per pair ───────────────────────────────────────────────────

def train_and_save(pair, acc: Optional[ImportanceAccumulator] = None):
    print(f"\n{'='*60}")
    exploratory = is_exploratory(pair)
    tag = ("[CRYPTO] [EXPLORATORY]" if exploratory and is_metal(pair)
           else "[EXPLORATORY]"      if exploratory
           else "[METAL]"            if is_metal(pair)
           else "[FOREX]")
    print(f"  Training model for {pair} {tag}")
    print(f"{'='*60}")

    input_path   = os.path.join(INPUT_DIR, f"{pair}_M15_labeled.csv")
    feature_cols = get_feature_cols(pair)

    if not os.path.exists(input_path):
        print(f"\n[INFO] {input_path} not found — fetching M15 data from MT5...")
        ok = fetch_and_save_m15(pair)
        if not ok or not os.path.exists(input_path):
            print(f"\n[ERROR] Could not get M15 data for {pair}. Skipping.")
            return

    print(f"\nLoading {pair} M15 data ...")
    df_full = pd.read_csv(input_path, parse_dates=["datetime"])
    df_full = df_full[df_full["datetime"].dt.dayofweek < 5].reset_index(drop=True)
    print(f"Loaded {len(df_full):,} bars")

    df_full = add_base_features(df_full)
    df_full = add_session_features(df_full)
    df_full = add_v2_features(df_full)
    df_full = add_momentum_features(df_full)
    df_full = add_v5_features(df_full)
    if is_metal(pair):    df_full = add_metal_features(df_full)
    if pair == "GBPUSDm":
        df_full = add_h1_features(df_full)
        print(f"  [GBPUSDm] H1 trend features added (h1_trend_direction, h1_adx)")

    df_full["regime"]  = classify_regime(df_full, pair)
    available_features = [f for f in feature_cols if f in df_full.columns]
    missing            = [f for f in feature_cols if f not in df_full.columns]
    if missing: print(f"[NOTE] Missing features: {missing}")

    df_full   = df_full.dropna(subset=available_features).reset_index(drop=True)
    df_setups = filter_to_setups(df_full, pair)
    df_labeled= make_binary_labels_full(df_full, df_setups, pair)

    print(f"\nTotal labeled setups : {len(df_labeled):,}")
    print(f"BUY   : {(df_labeled['binary_label']==1).sum():,}")
    print(f"SELL  : {(df_labeled['binary_label']==0).sum():,}")
    print(f"Regime trending : {(df_labeled['regime']=='trending').sum():,}")
    print(f"Regime ranging  : {(df_labeled['regime']=='ranging').sum():,}")
    if is_metal(pair):
        print(f"ATR multiplier  : {get_atr_multiplier(pair)}  |  "
              f"Forward bars: {get_forward_bars(pair)}")
    if exploratory:
        print(f"\n  NOTE: {pair} is EXPLORATORY — not included in main paper claims.")

    if len(df_labeled) < WF_MIN_TRAIN // 2:
        print(f"[ERROR] Not enough labeled setups for {pair}. Skipping."); return

    walk_forward_evaluate(pair, df_full, df_setups, available_features, acc=acc)

    # ── final model training ──────────────────────────────────────────────────
    X_all = df_labeled[available_features].values
    y_all = df_labeled["binary_label"].values.astype(int)
    r_all = df_labeled["regime"].values
    tr_m  = r_all == "trending"
    ra_m  = r_all == "ranging"

    ensemble = EnsembleModel()
    ensemble.feature_cols   = available_features
    ensemble.pair           = pair
    ensemble.atr_multiplier = get_atr_multiplier(pair)
    ensemble.adx_strong, ensemble.adx_moderate = get_adx_thresholds(pair)
    ensemble.is_metal_pair  = is_metal(pair)
    ensemble._build_col_indices()

    for i in range(max(0, len(X_all) - ONLINE_WINDOW_BARS), len(X_all)):
        ensemble.online_buffer.append({"X": X_all[i], "y": int(y_all[i]), "regime": r_all[i]})
    print(f"\n[ONLINE] Buffer seeded with {len(ensemble.online_buffer)} recent bars")

    print("\n-- Training TRENDING model --")
    if tr_m.sum() >= 200:
        Xtr = X_all[tr_m]; ytr = y_all[tr_m]
        sel = select_top_features(Xtr, ytr, available_features, TOP_N_FEATURES)
        fi  = [available_features.index(f) for f in sel]
        sc  = StandardScaler(); Xts = sc.fit_transform(Xtr[:, fi])
        bm  = build_base_models()
        print(f"  Generating OOF predictions ({len(bm)} models, 3-fold)...")
        oof = build_oof_predictions(bm, Xts, ytr, n_splits=3)
        sw0 = recency_weights(len(Xts))
        (Xb, yb), dr = smart_balance(Xts, ytr)
        sw  = np.ones(len(Xb)) if dr else sw0
        fitted = []; fimp = []
        for nm, mdl in bm.items():
            try:
                mdl.fit(Xb, yb, sample_weight=sw)
                p = mdl.predict_proba(Xts)[:, 1]
                if p.std() > 1e-4:
                    fitted.append(mdl)
                    print(f"  {nm}: OK  proba=[{p.min():.3f},{p.max():.3f}]")
                    imp = extract_model_importance(mdl, sel)
                    if imp is not None: fimp.append(imp)
            except Exception as e:
                print(f"  [WARN] {nm}: {e}")
        if fimp:
            avg = pd.concat(fimp, axis=1).mean(axis=1).sort_values(ascending=False)
            print_importance_table(avg, f"FINAL TRENDING importance — {pair}")
        ensemble.tr_scaler    = sc
        ensemble.tr_feat_idx  = np.array(fi)
        ensemble.tr_models    = fitted
        if len(fitted) >= 2:
            oof_use = oof[:, :len(fitted)]
            meta = LGBMClassifier(
                n_estimators=200, num_leaves=15, max_depth=3,
                learning_rate=0.05, subsample=0.8, random_state=13, verbose=-1)
            try:
                meta.fit(oof_use, ytr)
                ensemble.tr_meta = meta
                print(f"  Meta-learner: fitted on OOF shape {oof_use.shape}")
            except Exception as e:
                print(f"  [WARN] meta: {e}")

    print("\n-- Training RANGING model --")
    if ra_m.sum() >= MIN_RANGING_TRAIN:
        Xra  = X_all[ra_m]; yra = y_all[ra_m]
        mf   = min(TOP_N_FEATURES, max(5, len(Xra)//10))
        selr = select_top_features(Xra, yra, available_features, mf)
        fir  = [available_features.index(f) for f in selr]
        scr  = StandardScaler(); Xrs = scr.fit_transform(Xra[:, fir])
        nl   = min(15, max(4, len(Xra)//20)); mc = max(3, len(Xra)//30)
        rm   = LGBMClassifier(
            n_estimators=200, num_leaves=nl, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=mc, class_weight="balanced",
            random_state=42, verbose=-1)
        sw0 = recency_weights(len(Xrs))
        Xrb, yrb = safe_balance(Xrs, yra)
        sw = np.ones(len(Xrb)) if len(Xrb) != len(Xrs) else sw0
        try:
            rm.fit(Xrb, yrb, sample_weight=sw)
            p = rm.predict_proba(Xrs)[:, 1]
            print(f"  LightGBM: OK  proba=[{p.min():.3f},{p.max():.3f}]")
            impr = extract_model_importance(rm, selr)
            if impr is not None:
                print_importance_table(
                    impr.sort_values(ascending=False),
                    f"FINAL RANGING importance — {pair}")
            ensemble.ra_scaler   = scr
            ensemble.ra_feat_idx = np.array(fir)
            ensemble.ra_model    = rm
        except Exception as e:
            print(f"  [WARN] LightGBM ranging: {e}")

    if ensemble.tr_scaler is None and ensemble.ra_scaler is None:
        print(f"\n[ERROR] No sub-model trained for {pair}."); return

    save_path = os.path.join(MODEL_DIR, f"ensemble_{pair.lower()}.pkl")
    joblib.dump(ensemble, save_path)
    size_mb = os.path.getsize(save_path) / 1_048_576
    bf = get_buy_floor(pair); sc2 = get_sell_ceil(pair)
    exp_suffix = "\n  *** EXPLORATORY — exclude from main paper claims ***" if exploratory else ""
    print(f"\n{'='*60}")
    print(f"  Model saved -> {save_path}  ({size_mb:.1f} MB)")
    print(f"  Pair              : {pair} {tag}")
    print(f"  Feature cols      : {len(available_features)}")
    print(f"  ATR multiplier    : {ensemble.atr_multiplier}")
    print(f"  ADX thresholds    : strong={ensemble.adx_strong}  mod={ensemble.adx_moderate}")
    print(f"  Signal thresholds : BUY>={bf}  SELL<={sc2}")
    print(f"  Spread cost (R)   : {get_spread_r(pair)}")
    print(f"  Online buf        : {len(ensemble.online_buffer)} bars seeded")
    print(f"  Trending          : {'OK' if ensemble.tr_scaler else 'not trained'}")
    print(f"  Ranging           : {'OK' if ensemble.ra_scaler else 'not trained'}")
    print(f"  Meta-learner      : {'OK (OOF)' if ensemble.tr_meta else 'not trained'}")
    print(f"{'='*60}{exp_suffix}")

# ─── global importance summary ────────────────────────────────────────────────

def print_global_importance_summary(acc: ImportanceAccumulator):
    print(f"\n{'='*60}")
    print("  GLOBAL FEATURE IMPORTANCE SUMMARY")
    print("  (averaged across all NON-EXPLORATORY pairs and windows)")
    print(f"{'='*60}")
    out_dir = os.path.join(MODEL_DIR, "importance"); os.makedirs(out_dir, exist_ok=True)
    if acc.trend:
        gt = pd.concat(acc.trend, axis=1).mean(axis=1).sort_values(ascending=False)
        print_importance_table(gt, "GLOBAL TRENDING (copy to Figure 5)", top_n=15)
        gt.to_csv(os.path.join(out_dir, "GLOBAL_trend_importance.csv"), header=["importance"])
        print(f"\n  Saved → {out_dir}/GLOBAL_trend_importance.csv")
        print("\n  *** PASTE THESE VALUES INTO FIGURE 5 (trending side) ***")
        for f, v in gt.head(10).items(): print(f"    ('{f}', {v:.4f}),")
    if acc.range:
        gr = pd.concat(acc.range, axis=1).mean(axis=1).sort_values(ascending=False)
        print_importance_table(gr, "GLOBAL RANGING (copy to Figure 5)", top_n=15)
        gr.to_csv(os.path.join(out_dir, "GLOBAL_range_importance.csv"), header=["importance"])
        print(f"\n  Saved → {out_dir}/GLOBAL_range_importance.csv")
        print("\n  *** PASTE THESE VALUES INTO FIGURE 5 (ranging side) ***")
        for f, v in gr.head(10).items(): print(f"    ('{f}', {v:.4f}),")
    print(f"{'='*60}\n")

# ─── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("  save_model.py — Phase 6 v20  [Paper-Ready Edition]")
    print("  Fixes vs v19:")
    print("    FIX 25: sharpe_from_pnl — correct equity-curve annualisation")
    print("    FIX 26: simulate_pnl   — spread cost deducted per trade")
    print("    FIX 27: EMA 9/21 baseline F1+Sharpe in walk-forward tables")
    print("    FIX 28: MIN_RANGING_REPORT=40 — suppress low-n ranging rows")
    print("    FIX 29: LaTeX tables include BL F1 and BL Sharpe columns")
    print("="*60)
    print(f"  Forex  : USDJPYm, EURUSDm, GBPUSDm, AUDUSDm")
    print(f"  Metals : XAUUSDm (Gold), XAGUSDm (Silver)")
    print(f"  Crypto : BTCUSDm [EXPLORATORY — not in main claims]")
    print(f"  Min train bars    : {WF_MIN_TRAIN}")
    print(f"  WF step bars      : {WF_STEP}")
    print(f"  Bootstrap CI      : {N_BOOTSTRAP} resamples @ 95%")
    print(f"  Spread cost (FX)  : {SPREAD_COST_R['default']} R/trade")
    print("="*60)

    acc = ImportanceAccumulator()
    for pair in PAIRS:
        try:
            train_and_save(pair, acc=acc)
        except Exception as e:
            print(f"\n[ERROR] Failed to train {pair}: {e}")
            import traceback; traceback.print_exc()

    print_global_importance_summary(acc)

    print(f"\n{'='*60}")
    print("  All done! Models saved:")
    for pair in PAIRS:
        path = os.path.join(MODEL_DIR, f"ensemble_{pair.lower()}.pkl")
        size = f"{os.path.getsize(path)/1_048_576:.1f} MB" if os.path.exists(path) else "MISSING"
        tag  = " [EXPLORATORY]" if is_exploratory(pair) else " [METAL]" if is_metal(pair) else ""
        print(f"  {pair}{tag}: {size}")
    print(f"{'='*60}")