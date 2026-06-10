"""
================================================================
 Step 12 — Live Signal Monitor (Paper Trading)
================================================================
 Checks all 4 pairs every 60 minutes via MT5.
 Prints full trade instructions when a signal fires.
 Logs every signal to paper_trading_log.csv.

 Usage:
   python signal_monitor.py
   (Keep MetaTrader 5 open and logged in)
================================================================
"""

import os, time, warnings, traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

def _install(pkg):
    os.system(f"pip install {pkg} -q --break-system-packages")

try:    import MetaTrader5 as mt5
except: _install("MetaTrader5"); import MetaTrader5 as mt5

try:    from xgboost import XGBClassifier
except: _install("xgboost"); from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except:
    _install("lightgbm"); import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier

try:
    from catboost import CatBoostClassifier
    CATBOOST = True
except:
    _install("catboost")
    try:    from catboost import CatBoostClassifier; CATBOOST = True
    except: CATBOOST = False

try:    from imblearn.over_sampling import SMOTE, RandomOverSampler
except: _install("imbalanced-learn"); from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ── CONFIG ───────────────────────────────────────────────────────
PAIRS           = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
LOG_FILE        = "paper_trading_log.csv"
CHECK_INTERVAL  = 60        # minutes
LOOKBACK_BARS   = 600       # bars fetched from MT5
TRAIN_BARS      = 400       # bars used for model training
FORWARD_BARS    = 10        # labelling look-forward

RISK_PCT        = 0.005     # 0.5% risk per trade
RR_RATIO        = 2.5
ATR_MULT        = 2.5       # stop = ATR * 2.5
ATR_LABEL_MULT  = 1.8       # barrier for labelling

BUY_FLOOR       = 0.62
SELL_CEIL       = 0.38
SIGNAL_PCT      = 0.20
MIN_TRAIN       = 60        # minimum training rows per regime
ADX_MOD         = 25
ADX_STR         = 28
DE_THRESH       = 0.35
DE_WIN          = 20
RECENCY_HL      = 200
TOP_FEAT        = 18

FEATURE_COLS = [
    "rsi_14","macd","macd_signal","macd_hist","ema_cross",
    "atr_14","bb_width","bb_pct","return_1","return_5","return_10",
    "volatility_10","volatility_20","stoch_k","stoch_d",
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "is_london","is_ny","is_overlap",
    "adx_14","cci_20","williams_r","price_vs_ema200",
    "candle_body_pct","hl_ratio","swing_dist_high","swing_dist_low",
    "jpy_momentum_diff","volatility_regime","price_vs_ema50",
    "trend_alignment","directional_efficiency",
]

# ── FEATURES ─────────────────────────────────────────────────────
def build_features(df):
    df = df.copy().reset_index(drop=True)
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)

    # RSI 14
    d  = c.diff()
    g  = d.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["rsi_14"] = (100 - 100/(1 + g/ls.replace(0, np.nan))).fillna(50)

    # MACD
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # EMA cross
    e9  = c.ewm(span=9,  adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    df["ema_cross"] = (e9 > e21).astype(float)

    # ATR 14
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(span=14, adjust=False).mean()

    # Bollinger
    sma = c.rolling(20).mean()
    std = c.rolling(20).std()
    ub  = sma + 2*std; lb = sma - 2*std
    df["bb_width"] = ((ub-lb)/sma.replace(0,np.nan)).fillna(0)
    df["bb_pct"]   = ((c-lb)/(ub-lb).replace(0,np.nan)).fillna(0.5)

    # Returns & vol
    r = c.pct_change()
    df["return_1"]      = c.pct_change(1).fillna(0)
    df["return_5"]      = c.pct_change(5).fillna(0)
    df["return_10"]     = c.pct_change(10).fillna(0)
    df["volatility_10"] = r.rolling(10).std().fillna(0)
    df["volatility_20"] = r.rolling(20).std().fillna(0)

    # Stochastic
    ll = l.rolling(14).min(); hh = h.rolling(14).max()
    k  = ((c-ll)/(hh-ll).replace(0,np.nan)*100).fillna(50)
    df["stoch_k"] = k
    df["stoch_d"] = k.rolling(3).mean().fillna(50)

    # Session
    dt = pd.to_datetime(df["datetime"])
    hr = dt.dt.hour.astype(float)
    dw = dt.dt.dayofweek.astype(float)
    df["hour_sin"]   = np.sin(2*np.pi*hr/24)
    df["hour_cos"]   = np.cos(2*np.pi*hr/24)
    df["dow_sin"]    = np.sin(2*np.pi*dw/5)
    df["dow_cos"]    = np.cos(2*np.pi*dw/5)
    df["is_london"]  = ((hr>=8)  & (hr<17)).astype(float)
    df["is_ny"]      = ((hr>=13) & (hr<22)).astype(float)
    df["is_overlap"] = ((hr>=13) & (hr<17)).astype(float)

    # ADX 14 (pure numpy)
    hi = h.values; lo = l.values; cl = c.values; n = len(cl)
    tr_ = np.zeros(n); pdm = np.zeros(n); ndm = np.zeros(n)
    for i in range(1, n):
        tr_[i]  = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
        up       = hi[i]-hi[i-1]; dn = lo[i-1]-lo[i]
        pdm[i]   = max(up,0) if up > dn else 0
        ndm[i]   = max(dn,0) if dn > up else 0
    atr_ = pd.Series(tr_).ewm(span=14, adjust=False).mean().values
    pdi_ = pd.Series(pdm).ewm(span=14, adjust=False).mean().values
    ndi_ = pd.Series(ndm).ewm(span=14, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(atr_>0, 100*pdi_/atr_, 0)
        ndi = np.where(atr_>0, 100*ndi_/atr_, 0)
        dx  = np.where((pdi+ndi)>0, 100*np.abs(pdi-ndi)/(pdi+ndi), 0)
    df["adx_14"] = pd.Series(dx).ewm(span=14, adjust=False).mean().values

    # CCI
    tp     = (h+l+c)/3
    smatp  = tp.rolling(20).mean()
    mad    = tp.rolling(20).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
    df["cci_20"] = np.where(mad.values>0, (tp-smatp)/(0.015*mad), 0)

    # Williams %R
    hh14 = h.rolling(14).max(); ll14 = l.rolling(14).min()
    df["williams_r"] = np.where((hh14-ll14).values>0,
                                 -100*(hh14-c).values/(hh14-ll14).values, -50)

    # EMA200 features
    e200 = c.ewm(span=200, adjust=False).mean()
    df["price_vs_ema200"] = ((c-e200)/e200*100).values

    # Candle
    df["candle_body_pct"] = ((c-o).abs()/(h-l).replace(0,np.nan)).fillna(0.5)
    df["hl_ratio"]        = ((h-l)/df["atr_14"].replace(0,np.nan)).fillna(1.0)

    # Swing
    rh = c.rolling(20).max(); rl = c.rolling(20).min()
    df["swing_dist_high"] = ((rh-c)/c*100).values
    df["swing_dist_low"]  = ((c-rl)/c*100).values

    # Momentum diff
    d50 = c.diff()
    g50 = d50.clip(lower=0).ewm(span=50, adjust=False).mean()
    l50 = (-d50.clip(upper=0)).ewm(span=50, adjust=False).mean()
    r50 = 100 - (100/(1+g50/l50.replace(0,np.nan)))
    df["jpy_momentum_diff"] = (df["rsi_14"] - r50.fillna(50)).values

    # Volatility regime
    v5  = r.rolling(5).std()
    v20 = r.rolling(20).std()
    df["volatility_regime"] = (v5/v20.replace(0,np.nan)).fillna(1.0).clip(0,5)

    # Price vs EMA50 & trend alignment
    e50 = c.ewm(span=50, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    df["price_vs_ema50"]  = ((c-e50)/e50*100).values
    df["trend_alignment"] = ((e20>e50).astype(float) + (e50>e200).astype(float) - 1).values

    # Directional efficiency
    net  = r.rolling(DE_WIN).sum().abs()
    path = r.abs().rolling(DE_WIN).sum()
    df["directional_efficiency"] = (net/path.replace(0,np.nan)).fillna(0.5).clip(0,1)

    return df


# ── REGIME ───────────────────────────────────────────────────────
def regime_series(df):
    """Return a Series of regime strings for each row. All numpy — no isnan."""
    de  = pd.to_numeric(df.get("directional_efficiency", 0), errors="coerce").fillna(0).values
    adx = pd.to_numeric(df.get("adx_14",  0), errors="coerce").fillna(0).values
    ta  = pd.to_numeric(df.get("trend_alignment", 0), errors="coerce").fillna(0).values
    rule1   = (de > DE_THRESH) & (adx > ADX_STR)
    rule2   = (ta != 0)        & (adx > ADX_MOD)
    is_trend = rule1 | rule2
    return pd.Series(
        np.where(is_trend, "trending", "ranging"),
        index=df.index
    )


def regime_for_row(de, adx, ta):
    """Classify a single bar — all plain floats."""
    de  = float(de  if de  == de  else 0)
    adx = float(adx if adx == adx else 0)
    ta  = float(ta  if ta  == ta  else 0)
    if (de > DE_THRESH and adx > ADX_STR) or (ta != 0 and adx > ADX_MOD):
        return "trending"
    return "ranging"


def is_setup(de, adx, ta, rsi, bbp, vr, ema50_dist, regime):
    de  = float(de  if de  == de  else 0)
    adx = float(adx if adx == adx else 0)
    ta  = float(ta  if ta  == ta  else 0)
    rsi = float(rsi if rsi == rsi else 50)
    bbp = float(bbp if bbp == bbp else 0.5)
    vr  = float(vr  if vr  == vr  else 1)
    e50 = float(ema50_dist if ema50_dist == ema50_dist else 999)
    if regime == "trending":
        return (abs(e50) < 0.5) and (adx > ADX_MOD) and (ta != 0) and (vr < 2.0)
    else:
        return ((rsi < 25 or rsi > 75) or (bbp < 0.03 or bbp > 0.97)) and (vr < 2.0)


# ── LABELLING ────────────────────────────────────────────────────
def make_labels(df):
    cl  = df["close"].values.astype(float)
    atr = df["atr_14"].values.astype(float)
    out = np.full(len(df), np.nan)
    for i in range(len(df) - FORWARD_BARS):
        bar = ATR_LABEL_MULT * atr[i]
        up  = cl[i] + bar; dn = cl[i] - bar
        w   = cl[i+1:i+1+FORWARD_BARS]
        hu  = int(np.argmax(w >= up)) if np.any(w >= up) else None
        hl  = int(np.argmax(w <= dn)) if np.any(w <= dn) else None
        if   hu is not None and hl is not None: out[i] = 1 if hu <= hl else 0
        elif hu is not None:                     out[i] = 1
        elif hl is not None:                     out[i] = 0
    df = df.copy()
    df["label"] = out
    return df


# ── MODEL ────────────────────────────────────────────────────────
def recency_w(n):
    pos = np.arange(n, dtype=float)
    w   = np.power(2.0, (pos-(n-1))/RECENCY_HL)
    return w / w.sum() * n

def safe_bal(X, y):
    mn = int(pd.Series(y).value_counts().min())
    if mn >= 10:
        try:
            return SMOTE(random_state=42, k_neighbors=min(5,mn-1)).fit_resample(X, y)
        except Exception: pass
    if mn >= 3:
        try: return RandomOverSampler(random_state=42).fit_resample(X, y)
        except Exception: pass
    return X, y

def top_features(X, y, cols, n):
    n = min(n, len(cols), max(1, len(X)//10))
    try:
        rf = RandomForestClassifier(n_estimators=60, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        return pd.Series(rf.feature_importances_, index=cols).nlargest(n).index.tolist()
    except Exception:
        return cols[:n]

def predict_proba(X_tr, y_tr, X_pr, cols, regime):
    sel = top_features(X_tr, y_tr, cols, TOP_FEAT)
    fi  = [cols.index(f) for f in sel]
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X_tr[:, fi])
    Xp  = sc.transform(X_pr[:, fi])
    Xf, yf = safe_bal(Xs, y_tr)
    sw     = recency_w(len(Xf))

    models = [
        XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                      subsample=0.75, colsample_bytree=0.75,
                      scale_pos_weight=0.7, eval_metric="logloss",
                      random_state=42, verbosity=0),
        LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=5,
                       learning_rate=0.05, subsample=0.75,
                       class_weight="balanced" if regime=="ranging" else None,
                       random_state=42, verbose=-1),
    ]
    if CATBOOST:
        models.append(CatBoostClassifier(iterations=200, depth=5,
                      learning_rate=0.05, random_seed=42, verbose=0))
    probas = []
    for m in models:
        try:
            m.fit(Xf, yf, sample_weight=sw)
            p = float(m.predict_proba(Xp)[0, 1])
            if 0 < p < 1:
                probas.append(p)
        except Exception:
            pass
    return float(np.mean(probas)) if probas else None


# ── LOGGING ──────────────────────────────────────────────────────
def init_log():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=[
            "timestamp","pair","direction","regime","confidence",
            "entry_price","sl_price","tp_price","stop_pips","tp_pips",
            "lot_size","atr_14","adx_14","rsi_14",
            "result","actual_pnl","notes"
        ]).to_csv(LOG_FILE, index=False)
        print(f"  Log created: {LOG_FILE}")

def log_signal(d):
    df = pd.read_csv(LOG_FILE) if os.path.exists(LOG_FILE) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

def already_logged(pair, direction):
    try:
        df = pd.read_csv(LOG_FILE)
        if len(df) == 0: return False
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(hours=1)
        return len(df[(df["timestamp"]>cutoff)&(df["pair"]==pair)&(df["direction"]==direction)]) > 0
    except Exception:
        return False


# ── POSITION SIZE ─────────────────────────────────────────────────
def position_size(balance, atr, symbol):
    risk    = balance * RISK_PCT
    sl_pips = atr * ATR_MULT * (100 if "JPY" in symbol else 10000)
    pip_val = 0.01 if "JPY" in symbol else 1.0  # per micro-lot per pip (approx)
    lots    = risk / (sl_pips * pip_val * 100)
    lots    = max(0.01, round(lots, 2))
    tp_pips = sl_pips * RR_RATIO
    return lots, round(sl_pips, 1), round(tp_pips, 1)


# ── CHECK ONE PAIR ────────────────────────────────────────────────
def check_pair(symbol, balance):
    # 1. Fetch bars
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, LOOKBACK_BARS)
    if rates is None or len(rates) < 150:
        return None, "not enough bars"

    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume":"volume"})
    df = df[df["datetime"].dt.dayofweek < 5].reset_index(drop=True)

    # 2. Build features
    df = build_features(df)
    req = ["atr_14","rsi_14","adx_14","directional_efficiency","trend_alignment"]
    df  = df.dropna(subset=req).reset_index(drop=True)
    if len(df) < TRAIN_BARS + 20:
        return None, "not enough clean bars"

    # 3. Regime + setup filter on latest bar
    last = df.iloc[-1]
    de   = float(last["directional_efficiency"])
    adx  = float(last["adx_14"])
    ta   = float(last["trend_alignment"])
    rsi  = float(last["rsi_14"])
    bbp  = float(last["bb_pct"])
    vr   = float(last["volatility_regime"])
    e50  = float(last["price_vs_ema50"])
    regime = regime_for_row(de, adx, ta)

    if not is_setup(de, adx, ta, rsi, bbp, vr, e50, regime):
        return None, f"no setup ({regime})"

    # 4. Build training set
    df["regime"] = regime_series(df)
    df_tr = df.iloc[-(TRAIN_BARS+FORWARD_BARS):-FORWARD_BARS].copy()
    df_tr = make_labels(df_tr).dropna(subset=["label"])
    df_re = df_tr[df_tr["regime"] == regime]

    if len(df_re) < MIN_TRAIN:
        return None, f"too few {regime} train rows ({len(df_re)})"

    avail  = [f for f in FEATURE_COLS if f in df_re.columns]
    X_tr   = df_re[avail].values.astype(float)
    y_tr   = df_re["label"].values.astype(int)
    X_pred = last[avail].values.astype(float).reshape(1, -1)

    # Replace any NaN in prediction row
    X_pred = np.nan_to_num(X_pred, nan=0.0)

    # 5. Predict
    proba = predict_proba(X_tr, y_tr, X_pred, avail, regime)
    if proba is None:
        return None, "model returned None"

    # 6. Threshold
    if proba >= BUY_FLOOR:
        direction = "BUY";  conf = proba
    elif proba <= SELL_CEIL:
        direction = "SELL"; conf = 1 - proba
    else:
        return None, f"uncertain (p={proba:.3f})"

    # 7. Position sizing
    atr_val          = float(last["atr_14"])
    lots, sl_p, tp_p = position_size(balance, atr_val, symbol)
    entry            = float(last["close"])
    if direction == "BUY":
        sl = round(entry - atr_val*ATR_MULT, 5)
        tp = round(entry + atr_val*ATR_MULT*RR_RATIO, 5)
    else:
        sl = round(entry + atr_val*ATR_MULT, 5)
        tp = round(entry - atr_val*ATR_MULT*RR_RATIO, 5)

    return {
        "symbol":     symbol,
        "direction":  direction,
        "regime":     regime,
        "confidence": round(conf*100, 1),
        "entry":      entry,
        "sl":         sl,
        "tp":         tp,
        "sl_pips":    sl_p,
        "tp_pips":    tp_p,
        "lots":       lots,
        "atr":        round(atr_val, 5),
        "adx":        round(adx, 1),
        "rsi":        round(rsi, 1),
        "bar_time":   str(last["datetime"]),
    }, "signal"


# ── PRINT SIGNAL ─────────────────────────────────────────────────
def print_signal(s, balance):
    risk = balance * RISK_PCT
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'█'*66}")
    print(f"  ★ SIGNAL  {now}")
    print(f"{'█'*66}")
    print(f"  Pair      : {s['symbol']}")
    print(f"  Direction : {s['direction']}  {'↑' if s['direction']=='BUY' else '↓'}")
    print(f"  Regime    : {s['regime'].upper()}")
    print(f"  Confidence: {s['confidence']}%")
    print(f"  {'─'*64}")
    print(f"  Entry     : {s['entry']}")
    print(f"  Stop Loss : {s['sl']}  ({s['sl_pips']} pips)")
    print(f"  Take Profit:{s['tp']}  ({s['tp_pips']} pips)")
    print(f"  R:R       : 1:{RR_RATIO}")
    print(f"  {'─'*64}")
    print(f"  Lot size  : {s['lots']} lots")
    print(f"  Risk $    : ${risk:.2f}  ({RISK_PCT*100:.1f}% of ${balance:,.0f})")
    print(f"  Reward $  : ${risk*RR_RATIO:.2f}")
    print(f"  {'─'*64}")
    print(f"  ATR14     : {s['atr']}  |  ADX: {s['adx']}  |  RSI: {s['rsi']}")
    print(f"  Bar time  : {s['bar_time']}")
    print(f"{'█'*66}")
    print(f"  ► ACTION: {s['direction']} {s['lots']} lots {s['symbol']}")
    print(f"            SL={s['sl']}  TP={s['tp']}")
    print(f"  Fill in result/pnl in {LOG_FILE} after trade closes.")

# ── MAIN LOOP ────────────────────────────────────────────────────
def main():
    print("="*66)
    print(" Step 12 — Live Signal Monitor | Paper Trading Mode")
    print("="*66)
    print(f" Pairs: {', '.join(PAIRS)}  |  Risk: {RISK_PCT*100:.1f}%  |  R:R: 1:{RR_RATIO}")
    print(f" Check every {CHECK_INTERVAL} min  |  Log: {LOG_FILE}")
    print("="*66)

    if not mt5.initialize():
        print(f"  ✗ MT5 failed: {mt5.last_error()}"); return

    info = mt5.account_info()
    print(f"  MT5: {mt5.version()}")
    if info:
        print(f"  Account: {info.login} ({info.server})")
        print(f"  Balance: ${info.balance:,.2f}  [{'Demo' if info.trade_mode==0 else 'Live'}]")

    init_log()
    print(f"\n  Monitor running. Ctrl+C to stop.\n")

    session_signals = 0

    try:
        while True:
            balance = float(mt5.account_info().balance) if mt5.account_info() else 100000.0
            status  = {}

            for sym in PAIRS:
                try:
                    sig, reason = check_pair(sym, balance)
                    if sig and not already_logged(sym, sig["direction"]):
                        print_signal(sig, balance)
                        session_signals += 1
                        log_signal({
                            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "pair":        sym,
                            "direction":   sig["direction"],
                            "regime":      sig["regime"],
                            "confidence":  sig["confidence"],
                            "entry_price": sig["entry"],
                            "sl_price":    sig["sl"],
                            "tp_price":    sig["tp"],
                            "stop_pips":   sig["sl_pips"],
                            "tp_pips":     sig["tp_pips"],
                            "lot_size":    sig["lots"],
                            "atr_14":      sig["atr"],
                            "adx_14":      sig["adx"],
                            "rsi_14":      sig["rsi"],
                            "result":      "",
                            "actual_pnl":  "",
                            "notes":       "",
                        })
                        status[sym] = f"★ {sig['direction']} fired!"
                    elif sig:
                        status[sym] = f"signal (already logged)"
                    else:
                        status[sym] = reason
                except Exception as e:
                    status[sym] = f"error: {e}"
                    print(f"  [{sym}] {traceback.format_exc()}")

            nxt = datetime.now() + timedelta(minutes=CHECK_INTERVAL)
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Scan — ${balance:,.0f}")
            for p, s in status.items():
                print(f"    {p:<8} {s}")
            print(f"  Next: {nxt.strftime('%H:%M:%S')}  (Ctrl+C to stop)")

            time.sleep(CHECK_INTERVAL * 60)

    except KeyboardInterrupt:
        mt5.shutdown()
        print(f"\n  Stopped. Signals this session: {session_signals}")
        print(f"  Log: {LOG_FILE}")

if __name__ == "__main__":
    main()