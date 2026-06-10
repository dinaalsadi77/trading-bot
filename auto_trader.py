"""
auto_trader.py
==============
Live MetaTrader 5 trading bot.

Loads ensemble ML models (one per symbol), scans for signals on every new
M15 candle, and places/manages orders with ATR-based sizing, spread filters,
session filters, a daily loss circuit breaker, and a pullback close guard.

Symbols  : EURUSDm, USDJPYm, GBPUSDm, AUDUSDm, XAUUSDm, XAGUSDm, BTCUSDm
Timeframe: M15
Models   : ./models/ensemble_<symbol>.pkl  (trained by model_training_all_pairs.py)
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import os
import csv
import joblib
from datetime import datetime, date

from save_model import (
    EnsembleModel,
    FEATURE_COLS, METAL_FEATURE_COLS,
    DE_TREND_THRESHOLD,
    ADX_STRONG_MAP, ADX_STRONG_METAL, ADX_MODERATE_METAL,
    ADX_STRONG_CRYPTO, ADX_MODERATE_CRYPTO,
    add_session_features, add_v2_features,
    add_momentum_features, add_v5_features,
    add_base_features, add_metal_features, add_h1_features,
    is_metal, is_exploratory, get_feature_cols, get_adx_thresholds,
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

SYMBOLS = ["EURUSDm", "USDJPYm", "GBPUSDm", "AUDUSDm", "XAUUSDm", "XAGUSDm", "BTCUSDm"]

MODEL_DIR = "./models"

MODEL_FILES = {
    "USDJPYm": "ensemble_usdjpym.pkl",
    "EURUSDm": "ensemble_eurusdm.pkl",
    "GBPUSDm": "ensemble_gbpusdm.pkl",
    "AUDUSDm": "ensemble_audusdm.pkl",
    "XAUUSDm": "ensemble_xauusdm.pkl",
    "XAGUSDm": "ensemble_xagusdm.pkl",
    "BTCUSDm": "ensemble_btcusdm.pkl",
}

# Per-symbol confidence thresholds
CONFIDENCE_MAP = {
    "XAUUSDm": 0.72,
    "XAGUSDm": 0.72,
    "BTCUSDm": 0.74,
    "default":  0.67,
}

RISK_REWARD_RATIO = 2.5

MAX_SPREAD_MAP = {
    "EURUSDm": 0.00025,
    "USDJPYm": 0.00025,
    "GBPUSDm": 0.00030,
    "AUDUSDm": 0.00030,
    "XAUUSDm": 0.60,
    "XAGUSDm": 0.04,
    "BTCUSDm": 40.0,
}

ATR_SL_MULTIPLIER_MAP = {
    "XAUUSDm": 2.2,
    "XAGUSDm": 2.0,
    "BTCUSDm": 3.0,
    "default":  2.0,
}

RISK_PCT_MAP = {
    "XAUUSDm": 0.01,
    "XAGUSDm": 0.01,
    "BTCUSDm": 0.005,
    "default":  0.01,
}

MIN_SL_MAP = {
    "BTCUSDm": 200.0,
    "XAUUSDm": 10.0,
    "XAGUSDm": 0.30,
}

MIN_SL_FOREX = 0.0010   # 10 pips minimum for all forex pairs

PULLBACK_CLOSE_RATIO  = 0.50
MIN_PEAK_PROFIT_RATIO = 0.0003

MIN_LOT  = 0.01
MAX_LOT  = 0.50
LOT_STEP = 0.01

TIMEFRAME      = mt5.TIMEFRAME_M15
CANDLE_SECONDS = 900

MAX_OPEN_PER_CATEGORY = 1

MAGIC_NUMBER = 202401

SCAN_INTERVAL = 5

DEAL_HISTORY_FROM = datetime(2000, 1, 1)   # earliest date for history_deals_get

ORDER_COMMENT_PREFIX = "ML"        # prefix for trade comments: "ML BUY 0.72"
FILL_PROBE_COMMENT   = "fill_probe"

DAILY_LOSS_LIMIT_PCT  = 0.03
DAILY_LOSS_LIMIT_FILE = "session_balance.txt"

LOSS_COOLDOWN_CANDLES = 4

BLOCKED_HOURS_UTC = set(range(21, 23)) | {0, 1}

LOG_FILE   = "auto_trader.log"
TRADE_FILE = "trade_log.csv"

# -------------------------------------------------
# LOGGING
# -------------------------------------------------

_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_fmt)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_fmt)
_stream_handler.stream = open(1, "w", encoding="utf-8", closefd=False, buffering=1)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_file_handler)
logging.root.addHandler(_stream_handler)
log = logging.getLogger("AutoTrader")

# -------------------------------------------------
# TRADE LOG
# -------------------------------------------------

FIELDS = [
    "timestamp", "symbol", "direction", "confidence",
    "lot_size", "entry_price", "sl_price", "tp_price",
    "ticket", "status", "close_price", "pnl", "result", "category"
]

def init_trade_log():
    if not os.path.exists(TRADE_FILE):
        with open(TRADE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)


def log_trade(row):
    with open(TRADE_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


def update_trade_log(ticket, close_price, pnl, result):
    if not os.path.exists(TRADE_FILE):
        return
    rows = []
    with open(TRADE_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("ticket", "")) == str(ticket):
                row["close_price"] = round(close_price, 5)
                row["pnl"]         = round(pnl, 2)
                row["status"]      = "closed"
                row["result"]      = result
            rows.append(row)
    with open(TRADE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

# -------------------------------------------------
# MT5 CONNECTION
# -------------------------------------------------

def connect_mt5():
    if not mt5.initialize():
        log.error("MT5 init failed")
        return False
    acc = mt5.account_info()
    if acc is None:
        log.error("No account info")
        return False
    for s in SYMBOLS:
        mt5.symbol_select(s, True)
    log.info(f"Connected: {acc.login} | Balance {acc.balance:.2f}")
    return True


def ensure_connected():
    info = mt5.terminal_info()
    if info is None or not info.connected:
        log.warning("MT5 disconnected - reconnecting...")
        mt5.shutdown()
        time.sleep(3)
        if not connect_mt5():
            log.error("Reconnect failed - skipping this cycle")
            return False
    return True

# -------------------------------------------------
# SESSION SAFETY FILTER
# -------------------------------------------------

def is_safe_session() -> bool:
    hour = datetime.utcnow().hour
    if hour in BLOCKED_HOURS_UTC:
        log.info(f"Session filter: blocking new trades at UTC hour={hour}")
        return False
    return True

# -------------------------------------------------
# DAILY LOSS CIRCUIT BREAKER
# -------------------------------------------------

_session_start_balance = None


def _balance_file_for_today() -> str:
    today = date.today().isoformat()
    return f"session_balance_{today}.txt"


def load_session_balance():
    path = _balance_file_for_today()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                val = float(f.read().strip())
                log.info(f"Loaded session start balance from {path}: {val:.2f}")
                return val
        except (ValueError, IOError):
            pass
    return None


def save_session_balance(balance: float):
    path = _balance_file_for_today()
    try:
        with open(path, "w") as f:
            f.write(str(balance))
    except IOError as e:
        log.error(f"Could not save session balance: {e}")


def check_daily_loss_limit() -> bool:
    global _session_start_balance
    acc = mt5.account_info()
    if acc is None:
        return False

    if _session_start_balance is None:
        saved = load_session_balance()
        if saved is not None:
            _session_start_balance = saved
        else:
            _session_start_balance = acc.balance
            save_session_balance(_session_start_balance)
            log.info(f"Session start balance set: {_session_start_balance:.2f}")
        return False

    loss_pct = (_session_start_balance - acc.balance) / max(_session_start_balance, 1.0)
    if loss_pct >= DAILY_LOSS_LIMIT_PCT:
        log.warning(
            f"DAILY LOSS LIMIT HIT: "
            f"start={_session_start_balance:.2f}  "
            f"now={acc.balance:.2f}  "
            f"loss={loss_pct*100:.1f}% >= {DAILY_LOSS_LIMIT_PCT*100:.0f}%"
        )
        return True
    return False

# -------------------------------------------------
# LOSS COOLDOWN TRACKER
# -------------------------------------------------

_last_loss_bar: dict = {}


def record_loss_bar(symbol: str, bar_time: int):
    _last_loss_bar[symbol] = bar_time
    log.info(f"{symbol}: loss recorded — cooldown for {LOSS_COOLDOWN_CANDLES} candles")


def in_loss_cooldown(symbol: str, current_bar: int) -> bool:
    last = _last_loss_bar.get(symbol)
    if last is None:
        return False
    bars_elapsed = (current_bar - last) // CANDLE_SECONDS
    if bars_elapsed < LOSS_COOLDOWN_CANDLES:
        log.info(
            f"{symbol}: loss cooldown active — "
            f"{LOSS_COOLDOWN_CANDLES - bars_elapsed} candle(s) remaining"
        )
        return True
    return False

# -------------------------------------------------
# TRADE OUTCOME TRACKING
# -------------------------------------------------

_open_tickets: dict = {}

def register_ticket(ticket, symbol, direction, conf, lot,
                    entry, sl, tp, feature_row: np.ndarray, regime: str):
    _open_tickets[ticket] = {
        "symbol":      symbol,
        "direction":   direction,
        "confidence":  conf,
        "lot_size":    lot,
        "entry_price": entry,
        "sl_price":    sl,
        "tp_price":    tp,
        "feature_row": feature_row,
        "regime":      regime,
    }
    log.info(
        f"Registered ticket {ticket} for outcome tracking "
        f"({symbol} {direction} regime={regime})"
    )


def check_closed_trades(models: dict):
    if not _open_tickets:
        return
    try:
        to_date = datetime.utcnow()
        deals   = mt5.history_deals_get(DEAL_HISTORY_FROM, to_date)

        if deals is None:
            return

        closed_tickets     = set()
        closed_trade_rows  = []

        for deal in deals:
            if deal.magic != MAGIC_NUMBER:
                continue
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue
            if deal.order not in _open_tickets:
                continue

            info  = _open_tickets[deal.order]
            pnl   = round(deal.profit, 2)
            close = round(deal.price, 5)

            if pnl > 0:
                result = "WIN"
            elif pnl < 0:
                result = "LOSS"
            else:
                result = "BREAKEVEN"

            log.info(
                f"TRADE CLOSED | {info['symbol']} {info['direction']} "
                f"| entry={info['entry_price']:.5f}  close={close:.5f} "
                f"| PnL={pnl:.2f} | {result} | ticket={deal.order}"
            )

            update_trade_log(deal.order, close, pnl, result)
            closed_tickets.add(deal.order)
            closed_trade_rows.append({
                "symbol": info["symbol"],
                "result": result,
                "pnl":    pnl,
            })

            if result == "LOSS":
                bar_time = int(deal.time // CANDLE_SECONDS) * CANDLE_SECONDS
                record_loss_bar(info["symbol"], bar_time)

            symbol = info["symbol"]
            if symbol in models and info.get("feature_row") is not None:
                try:
                    models[symbol].record_trade_outcome(
                        feature_row=info["feature_row"],
                        regime=info["regime"],
                        direction=info["direction"],
                        outcome=result,
                        model_dir=MODEL_DIR,
                    )
                except Exception as e:
                    log.error(f"record_trade_outcome failed for {symbol}: {e}")

        for t in closed_tickets:
            del _open_tickets[t]
            _peak_profit.pop(t, None)

        if closed_tickets:
            print_session_summary(closed_trade_rows)

    except Exception as e:
        log.error(f"check_closed_trades error: {e}")


def print_session_summary(closed_rows: list):
    """Print a win/loss summary from a list of closed trade dicts."""
    try:
        if not closed_rows:
            return

        wins      = [r for r in closed_rows if r["result"] == "WIN"]
        losses    = [r for r in closed_rows if r["result"] == "LOSS"]
        total_pnl = sum(float(r["pnl"]) for r in closed_rows if r["pnl"])
        win_rate  = len(wins) / len(closed_rows) * 100

        log.info(
            f"SESSION SUMMARY | "
            f"Trades={len(closed_rows)}  "
            f"Wins={len(wins)}  "
            f"Losses={len(losses)}  "
            f"WinRate={win_rate:.1f}%  "
            f"TotalPnL={total_pnl:.2f}"
        )

        symbols_seen = set(r["symbol"] for r in closed_rows)
        for sym in sorted(symbols_seen):
            sym_trades = [r for r in closed_rows if r["symbol"] == sym]
            sym_wins   = [r for r in sym_trades if r["result"] == "WIN"]
            sym_pnl    = sum(float(r["pnl"]) for r in sym_trades if r["pnl"])
            sym_wr     = len(sym_wins) / len(sym_trades) * 100 if sym_trades else 0
            tag        = " [METAL]" if is_metal(sym) else ""
            log.info(
                f"  {sym}{tag}: {len(sym_trades)} trades | "
                f"WinRate={sym_wr:.1f}% | PnL={sym_pnl:.2f}"
            )

    except Exception as e:
        log.error(f"print_session_summary error: {e}")

# -------------------------------------------------
# DATA
# -------------------------------------------------

def fetch_ohlc(symbol, bars=600):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, bars)
    if rates is None:
        log.warning(f"{symbol}: fetch_ohlc returned None | {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    return df

# -------------------------------------------------
# FEATURES
# -------------------------------------------------

_feature_cache: dict = {}
_FEATURE_CACHE_MAX = len(MODEL_FILES)   # one entry per symbol is enough


def compute_indicators(df, symbol=""):
    if df is None or len(df) < 202:
        return None

    last_closed_time = int(df["time"].iloc[-2])
    cached = _feature_cache.get(symbol)
    if cached and cached[0] == last_closed_time:
        return cached[1]

    # evict oldest entry when cache is full
    if len(_feature_cache) >= _FEATURE_CACHE_MAX:
        _feature_cache.pop(next(iter(_feature_cache)))

    try:
        df_closed = df.iloc[:-1].copy()

        df_closed = add_base_features(df_closed)
        df_closed = add_session_features(df_closed)
        df_closed = add_v2_features(df_closed)
        df_closed = add_momentum_features(df_closed)
        df_closed = add_v5_features(df_closed)

        if is_metal(symbol):
            df_closed = add_metal_features(df_closed)

        adx_s, adx_m = get_adx_thresholds(symbol)

        # GBPUSDm requires H1 features to match training pipeline
        if symbol == "GBPUSDm":
            df_closed = add_h1_features(df_closed)

        has_de  = "directional_efficiency" in df_closed.columns
        has_adx = "adx_14"                in df_closed.columns
        has_ta  = "trend_alignment"        in df_closed.columns

        if has_de and has_adx and has_ta:
            rule1 = (
                (df_closed["directional_efficiency"] > DE_TREND_THRESHOLD) &
                (df_closed["adx_14"] > adx_s)
            )
            rule2 = (
                (df_closed["trend_alignment"] != 0) &
                (df_closed["adx_14"] > adx_s)
            )
            df_closed["regime"] = np.where(rule1 | rule2, "trending", "ranging")
        elif has_adx:
            df_closed["regime"] = np.where(
                df_closed["adx_14"] > adx_s, "trending", "ranging"
            )
        else:
            df_closed["regime"] = "ranging"

        df_closed = df_closed.dropna()
        if len(df_closed) < 50:
            return None

        _feature_cache[symbol] = (last_closed_time, df_closed)
        return df_closed

    except Exception as e:
        log.error(f"{symbol} feature error: {e}")
        return None

# -------------------------------------------------
# FEATURE CONSISTENCY GUARD
# -------------------------------------------------

def validate_model_features(symbol: str, model: EnsembleModel) -> bool:
    if model.feature_cols is None:
        log.error(f"{symbol}: model.feature_cols is None")
        return False

    expected_cols = get_feature_cols(symbol)
    model_set     = set(model.feature_cols)
    expected_set  = set(expected_cols)

    missing_from_pipeline = model_set - expected_set
    missing_from_model    = expected_set - model_set

    ok = True
    if missing_from_pipeline:
        log.error(
            f"{symbol}: model expects features the live pipeline does NOT produce: "
            f"{sorted(missing_from_pipeline)}. Re-train with save_model.py."
        )
        ok = False

    if missing_from_model:
        log.warning(
            f"{symbol}: pipeline produces features model ignores: "
            f"{sorted(missing_from_model)}."
        )

    if ok:
        metal_tag = " [METAL]" if is_metal(symbol) else ""
        log.info(
            f"{symbol}{metal_tag}: feature names consistent ✓ "
            f"({len(model.feature_cols)} cols)"
        )

    return ok

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------

def load_models():
    models = {}
    for symbol, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            log.warning(f"{symbol}: model file not found at {path} - skipping")
            continue
        try:
            model = joblib.load(path)
            if validate_model_features(symbol, model):
                models[symbol] = model
                metal_tag = " [METAL]" if is_metal(symbol) else ""
                exp_tag   = " [EXPLORATORY]" if is_exploratory(symbol) else ""
                log.info(
                    f"{symbol}{metal_tag}{exp_tag}: model loaded | "
                    f"Features: {len(model.feature_cols)} | "
                    f"ATR mult: {model.atr_multiplier}"
                )
            else:
                log.error(f"{symbol}: model SKIPPED due to feature mismatch")
        except Exception as e:
            log.error(f"{symbol}: failed to load model - {e}")
    return models

# -------------------------------------------------
# SPREAD FILTER
# -------------------------------------------------

def spread_ok(symbol):
    tick  = mt5.symbol_info_tick(symbol)
    if not tick:
        return False
    limit  = MAX_SPREAD_MAP.get(symbol, 0.00025)
    spread = tick.ask - tick.bid
    if spread > limit:
        log.info(f"{symbol}: spread {spread:.5f} > limit {limit:.5f} — skip")
        return False
    return True

# -------------------------------------------------
# POSITION LIMITS
# -------------------------------------------------

def open_positions_by_category():
    pos = mt5.positions_get(magic=MAGIC_NUMBER)
    if pos is None:
        return 0, 0
    forex_open = sum(1 for p in pos if not is_metal(p.symbol))
    metal_open = sum(1 for p in pos if is_metal(p.symbol))
    return forex_open, metal_open


def in_symbol(symbol):
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        return False
    return any(p.magic == MAGIC_NUMBER for p in pos)

# -------------------------------------------------
# SIGNAL
# -------------------------------------------------

def get_signal(model, df_closed, symbol):
    try:
        last_row  = df_closed.iloc[-1]
        regime    = last_row.get("regime", "ranging")

        # guard against model expecting columns not yet present in df_closed
        missing = [c for c in model.feature_cols if c not in df_closed.columns]
        if missing:
            log.warning(f"{symbol}: missing columns in df_closed: {missing} — skipping")
            return None, 0, None, None

        row_feats = df_closed.iloc[-1:][model.feature_cols]

        if row_feats.isnull().any().any():
            log.warning(f"{symbol}: NaN in features - skipping")
            return None, 0, None, None

        proba  = model.predict_proba(row_feats)[0]
        p_sell = float(proba[0])
        p_buy  = float(proba[1])

        threshold = CONFIDENCE_MAP.get(symbol, CONFIDENCE_MAP["default"])

        metal_tag = " [METAL]" if is_metal(symbol) else ""
        log.info(
            f"{symbol}{metal_tag} | regime={regime} | "
            f"BUY={p_buy:.3f}  SELL={p_sell:.3f} | threshold={threshold:.2f}"
        )

        feature_row = df_closed.iloc[-1][model.feature_cols].values.astype(float)

        if p_buy >= threshold:
            log.info(f"{symbol} >> BUY signal  conf={p_buy:.3f}")
            return "BUY", p_buy, feature_row, regime

        if p_sell >= threshold:
            log.info(f"{symbol} >> SELL signal  conf={p_sell:.3f}")
            return "SELL", p_sell, feature_row, regime

        log.info(
            f"{symbol}: no signal "
            f"(best conf={max(p_buy, p_sell):.3f} < {threshold:.2f})"
        )
        return None, max(p_buy, p_sell), None, None

    except Exception as e:
        log.error(f"{symbol} signal error: {e}")
        return None, 0, None, None

# -------------------------------------------------
# LOT SIZING
# -------------------------------------------------

def compute_lot(symbol, sl_dist):
    acc = mt5.account_info()
    if acc is None:
        return MIN_LOT

    info = mt5.symbol_info(symbol)
    if info is None or sl_dist <= 0:
        return MIN_LOT

    risk_pct    = RISK_PCT_MAP.get(symbol, RISK_PCT_MAP["default"])
    risk_amount = acc.balance * risk_pct
    pip_value   = info.trade_tick_value / info.trade_tick_size
    sl_pips     = sl_dist / info.point

    if pip_value <= 0 or sl_pips <= 0:
        return MIN_LOT

    risk_per_lot = sl_pips * pip_value

    if risk_per_lot <= 0:
        return MIN_LOT

    raw_lot = risk_amount / risk_per_lot
    lot     = round(round(raw_lot / LOT_STEP) * LOT_STEP, 2)
    lot     = max(MIN_LOT, min(MAX_LOT, lot))

    metal_tag = " [METAL]" if is_metal(symbol) else ""
    log.info(
        f"{symbol}{metal_tag} | balance={acc.balance:.2f}  "
        f"risk_pct={risk_pct*100:.2f}%  risk_amt={risk_amount:.2f}  "
        f"sl_pips={sl_pips:.1f}  lot={lot}"
    )
    return lot

# -------------------------------------------------
# FILLING MODE
# -------------------------------------------------

_filling_cache: dict = {}

def _probe_filling(symbol, order_type, price, sl, tp):
    for mode in [mt5.ORDER_FILLING_IOC,
                 mt5.ORDER_FILLING_FOK,
                 mt5.ORDER_FILLING_RETURN]:
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       MIN_LOT,
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,
            "magic":        MAGIC_NUMBER,
            "comment":      FILL_PROBE_COMMENT,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mode,
        }
        result = mt5.order_check(req)
        if result is not None and result.retcode != 10030:
            log.info(f"{symbol}: filling mode probed OK -> {mode}")
            return mode
    log.warning(f"{symbol}: all filling modes rejected - defaulting to RETURN")
    return mt5.ORDER_FILLING_RETURN


def get_filling_mode(symbol, order_type=None, price=0.0, sl=0.0, tp=0.0):
    if symbol in _filling_cache:
        return _filling_cache[symbol]
    if order_type is not None:
        mode = _probe_filling(symbol, order_type, price, sl, tp)
    else:
        info  = mt5.symbol_info(symbol)
        flags = info.filling_mode if info else 0
        if   flags & 2: mode = mt5.ORDER_FILLING_IOC
        elif flags & 4: mode = mt5.ORDER_FILLING_RETURN
        elif flags & 1: mode = mt5.ORDER_FILLING_FOK
        else:           mode = mt5.ORDER_FILLING_RETURN
    _filling_cache[symbol] = mode
    return mode

# -------------------------------------------------
# PLACE ORDER
# -------------------------------------------------

def place_order(symbol, direction, conf, atr, feature_row, regime):
    if np.isnan(atr) or atr <= 0:
        log.warning(f"{symbol}: invalid ATR={atr} — skipping trade")
        return None

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        log.error(f"{symbol}: no tick data")
        return None

    sl_mult = ATR_SL_MULTIPLIER_MAP.get(symbol, ATR_SL_MULTIPLIER_MAP["default"])

    if is_metal(symbol):
        min_sl = MIN_SL_MAP.get(symbol, 0.0002)
    else:
        min_sl = max(MIN_SL_MAP.get(symbol, 0.0002), MIN_SL_FOREX)

    sl_dist = max(atr * sl_mult, min_sl)
    tp_dist = sl_dist * RISK_REWARD_RATIO
    lot     = compute_lot(symbol, sl_dist)

    if direction == "BUY":
        price      = tick.ask
        sl         = price - sl_dist
        tp         = price + tp_dist
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price      = tick.bid
        sl         = price + sl_dist
        tp         = price - tp_dist
        order_type = mt5.ORDER_TYPE_SELL

    sym_info = mt5.symbol_info(symbol)
    if sym_info:
        sl_pips = sl_dist / sym_info.point
        log.info(f"{symbol}: SL dist={sl_dist:.5f} ({sl_pips:.1f} pts/pips)  TP dist={tp_dist:.5f}")

    filling = get_filling_mode(symbol, order_type, price, sl, tp)

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    20,
        "magic":        MAGIC_NUMBER,
        "comment":      f"{ORDER_COMMENT_PREFIX} {direction} {conf:.2f}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    result = mt5.order_send(request)

    if result is not None and result.retcode == 10030:
        log.warning(f"{symbol}: retcode 10030 - cycling fallbacks")
        _filling_cache.pop(symbol, None)
        for fallback in [mt5.ORDER_FILLING_IOC,
                         mt5.ORDER_FILLING_FOK,
                         mt5.ORDER_FILLING_RETURN]:
            if fallback == filling:
                continue
            request["type_filling"] = fallback
            result = mt5.order_send(request)
            if result is not None and result.retcode != 10030:
                _filling_cache[symbol] = fallback
                break

    if result is None:
        log.error(f"{symbol}: order_send returned None | {mt5.last_error()}")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(
            f"{symbol}: ORDER FAILED | retcode={result.retcode} | "
            f"comment={result.comment}"
        )
        return None

    metal_tag = " [METAL]" if is_metal(symbol) else ""
    log.info(
        f"ORDER OK {symbol}{metal_tag} {direction} "
        f"lot={lot} entry={price:.5f} SL={sl:.5f} TP={tp:.5f} "
        f"sl_dist={sl_dist:.5f} ticket={result.order}"
    )

    category = "METAL" if is_metal(symbol) else "FOREX"
    log_trade({
        "timestamp":   datetime.utcnow().isoformat(),
        "symbol":      symbol,
        "direction":   direction,
        "confidence":  round(conf, 4),
        "lot_size":    lot,
        "entry_price": price,
        "sl_price":    sl,
        "tp_price":    tp,
        "ticket":      result.order,
        "status":      "open",
        "close_price": "",
        "pnl":         "",
        "result":      "",
        "category":    category,
    })

    register_ticket(
        result.order, symbol, direction, conf, lot,
        price, sl, tp, feature_row, regime
    )

    return result.order

# -------------------------------------------------
# ACTIVE TRADE MANAGEMENT
# -------------------------------------------------

_peak_profit: dict = {}


def _close_position(p, reason: str = "active_close") -> bool:
    symbol = p.symbol
    ticket = p.ticket

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        log.error(f"{symbol} ticket={ticket}: no tick data for close")
        return False

    if p.type == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask

    filling = get_filling_mode(symbol, close_type, price, 0.0, 0.0)

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       p.volume,
        "type":         close_type,
        "position":     ticket,
        "price":        price,
        "deviation":    30,
        "magic":        MAGIC_NUMBER,
        "comment":      reason[:31],
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    result = mt5.order_send(request)

    if result is not None and result.retcode == 10030:
        log.warning(f"{symbol} ticket={ticket}: close retcode 10030 — cycling fallbacks")
        _filling_cache.pop(symbol, None)
        for fallback in [mt5.ORDER_FILLING_IOC,
                         mt5.ORDER_FILLING_FOK,
                         mt5.ORDER_FILLING_RETURN]:
            if fallback == filling:
                continue
            request["type_filling"] = fallback
            result = mt5.order_send(request)
            if result is not None and result.retcode != 10030:
                _filling_cache[symbol] = fallback
                break

    if result is None:
        log.error(f"{symbol} ticket={ticket}: close order_send returned None | {mt5.last_error()}")
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"{symbol} ticket={ticket}: CLOSE FAILED | retcode={result.retcode} | comment={result.comment}")
        return False

    log.info(f"CLOSED OK | {symbol} ticket={ticket} | price={price:.5f} | reason={reason}")
    return True


def manage_open_trades():
    pos = mt5.positions_get(magic=MAGIC_NUMBER)
    if not pos:
        return

    acc = mt5.account_info()
    min_meaningful = (acc.balance * MIN_PEAK_PROFIT_RATIO) if acc else 0.30

    for p in pos:
        symbol = p.symbol
        ticket = p.ticket
        profit = p.profit
        entry  = p.price_open
        tp     = p.tp

        if not tp:
            log.warning(f"{symbol} ticket={ticket}: TP not set — skipping active management")
            continue

        tp_dist = abs(entry - tp)
        if tp_dist <= 0:
            continue

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue

        if p.type == mt5.ORDER_TYPE_BUY:
            current_price = tick.bid
            price_dist    = current_price - entry
        else:
            current_price = tick.ask
            price_dist    = entry - current_price

        prev_peak            = _peak_profit.get(ticket, profit)
        _peak_profit[ticket] = max(prev_peak, profit)
        peak                 = _peak_profit[ticket]

        metal_tag = " [METAL]" if is_metal(symbol) else ""
        log.info(
            f"MANAGE {symbol}{metal_tag} ticket={ticket} | "
            f"profit={profit:.2f} peak={peak:.2f} | "
            f"price_dist={price_dist:.5f} tp_dist={tp_dist:.5f}"
        )

        # Rule 1: TP reached
        if price_dist >= tp_dist * 0.98:
            log.info(f"TP ACTIVE CLOSE: {symbol} ticket={ticket} | profit={profit:.2f}")
            if _close_position(p, reason="TP_active"):
                _peak_profit.pop(ticket, None)
            continue

        # Rule 2: Pullback guard
        if peak >= min_meaningful and profit < peak * PULLBACK_CLOSE_RATIO:
            log.info(
                f"PULLBACK CLOSE: {symbol} ticket={ticket} | "
                f"profit={profit:.2f} < {PULLBACK_CLOSE_RATIO*100:.0f}% of peak={peak:.2f}"
            )
            if _close_position(p, reason="pullback_guard"):
                _peak_profit.pop(ticket, None)

# -------------------------------------------------
# EQUITY LOGGER
# -------------------------------------------------

def log_equity(cycle):
    if cycle % 20 == 0:
        acc = mt5.account_info()
        if acc:
            forex_open, metal_open = open_positions_by_category()
            log.info(
                f"ACCOUNT | balance={acc.balance:.2f}  "
                f"equity={acc.equity:.2f}  "
                f"free_margin={acc.margin_free:.2f}  "
                f"forex_open={forex_open}  metal_open={metal_open}"
            )

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------

_last_bar_time: dict = {}


def run():
    init_trade_log()

    if not connect_mt5():
        return

    models = load_models()
    if not models:
        log.error("No models loaded - exiting. Run save_model.py first.")
        return

    forex_models = [s for s in models if not is_metal(s)]
    metal_models = [s for s in models if is_metal(s)]
    log.info(f"Forex models : {forex_models}")
    log.info(f"Metal models : {metal_models}")
    log.info(
        f"Auto-trader started | "
        f"max trades: {MAX_OPEN_PER_CATEGORY} forex + {MAX_OPEN_PER_CATEGORY} metals | "
        f"risk/trade: {RISK_PCT_MAP['default']*100:.1f}% | "
        f"daily limit: {DAILY_LOSS_LIMIT_PCT*100:.0f}%"
    )
    log.info(f"Blocked UTC hours: {sorted(BLOCKED_HOURS_UTC)}")
    log.info(f"Loss cooldown: {LOSS_COOLDOWN_CANDLES} candles after each loss")
    log.info(f"Min SL forex: {MIN_SL_FOREX} | ATR mult default: {ATR_SL_MULTIPLIER_MAP['default']}")

    cycle = 0

    while True:
        cycle += 1
        log.info(f"--- Cycle {cycle} ---")

        if not ensure_connected():
            log.warning("Sleeping 10s before retry")
            time.sleep(10)
            continue

        check_closed_trades(models)
        manage_open_trades()

        if check_daily_loss_limit():
            log.warning("Daily loss limit reached — bot paused. Will resume tomorrow.")
            time.sleep(60)
            continue

        log_equity(cycle)

        forex_open, metal_open = open_positions_by_category()
        log.info(f"Open positions: forex={forex_open}  metals={metal_open}")

        session_safe = is_safe_session()
        if not session_safe:
            log.info("Session filter active — skipping new entries this cycle")

        for s in SYMBOLS:

            if s not in models:
                log.info(f"{s}: no model loaded - skip")
                continue

            if in_symbol(s):
                log.info(f"{s}: already in position - skip")
                continue

            if is_metal(s):
                if metal_open >= MAX_OPEN_PER_CATEGORY:
                    log.info(f"{s} [METAL]: metal limit reached - skip")
                    continue
            else:
                if forex_open >= MAX_OPEN_PER_CATEGORY:
                    log.info(f"{s} [FOREX]: forex limit reached - skip")
                    continue

            df_raw = fetch_ohlc(s)
            if df_raw is None:
                continue

            current_bar = int(df_raw["time"].iloc[-2])

            if _last_bar_time.get(s) == current_bar:
                log.info(f"{s}: same candle — no new signal")
                continue

            _last_bar_time[s] = current_bar

            if in_loss_cooldown(s, current_bar):
                continue

            metal_tag = " [METAL]" if is_metal(s) else ""
            log.info(f"{s}{metal_tag}: NEW candle @ {current_bar} - running model")

            if not spread_ok(s):
                continue

            df_closed = compute_indicators(df_raw, symbol=s)
            if df_closed is None:
                continue

            sig, conf, feature_row, regime = get_signal(models[s], df_closed, s)

            if "atr_14" not in df_closed.columns:
                log.warning(f"{s}: atr_14 column missing — skipping")
                continue

            atr = float(df_closed["atr_14"].iloc[-1])
            if np.isnan(atr) or atr <= 0:
                log.warning(f"{s}: ATR={atr} invalid — skipping")
                continue

            if sig:
                if not session_safe:
                    log.info(
                        f"{s}: signal={sig} suppressed by session filter "
                        f"(UTC hour={datetime.utcnow().hour})"
                    )
                    continue

                log.info(f"{s} FINAL SIGNAL {sig}  conf={conf:.3f}  atr={atr:.5f}")
                place_order(s, sig, conf, atr, feature_row, regime)

                if is_metal(s):
                    metal_open += 1
                else:
                    forex_open += 1

        log.info(f"Cycle {cycle} done. Sleeping {SCAN_INTERVAL}s ...")
        time.sleep(SCAN_INTERVAL)


# -------------------------------------------------

if __name__ == "__main__":
    run()