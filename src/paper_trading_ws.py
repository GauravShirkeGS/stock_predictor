# src/paper_trade_ws.py
"""
Real-time paper trading bot using Alpaca market-data websocket + REST orders.
- Builds rolling OHLCV history per symbol
- Computes technical indicators consistent with training
- Uses saved XGBoost model + feature_columns to predict Buy/Hold/Sell
- Buys using budget fraction of account cash (multiple shares) and sells full position
- Tracks local entry price for SL/TP checks
"""

import os
import json
import time
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone
import requests
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import websocket

# ------------------ CONFIG ------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PK2D3Q5ROEM4ZAXJ99WQ")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "ginRvISIbqB81QLvvRvi7qx2OXRUtKdktLa7pZoN")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_WS = os.getenv("ALPACA_DATA_WS", "wss://stream.data.alpaca.markets/v2/iex")
MODEL_PATH = os.getenv("MODEL_PATH", "model/xgb_trading_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "model/feature_columns.pkl")

SYMBOL = os.getenv("PAPER_SYMBOL", "AAPL")    # Single-symbol version; can be extended to list
TRADE_SIZE_FRACTION = float(os.getenv("TRADE_SIZE_FRACTION", 0.10))  # fraction of cash per trade
STOP_LOSS = float(os.getenv("STOP_LOSS", 0.02))   # 2%
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", 0.04))  # 4%
HISTORY_LEN = int(os.getenv("HISTORY_LEN", 300))  # bars to keep

# ------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# In-memory market history per symbol
hist = defaultdict(lambda: {
    "ts": deque(maxlen=HISTORY_LEN),
    "open": deque(maxlen=HISTORY_LEN),
    "high": deque(maxlen=HISTORY_LEN),
    "low": deque(maxlen=HISTORY_LEN),
    "close": deque(maxlen=HISTORY_LEN),
    "volume": deque(maxlen=HISTORY_LEN),
})

# bookkeeping
model = None
feature_columns = None
local_positions = {}   # symbol -> {qty, entry_price, entry_time}

HEADERS = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}


def load_model_and_features():
    global model, feature_columns
    if model is None or feature_columns is None:
        logging.info("Loading model and feature list...")
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURES_PATH)
        logging.info("Loaded model (%s) and %d features.", MODEL_PATH, len(feature_columns))
    return model, feature_columns


# ----------------- REST helpers -----------------
def place_market_order(symbol: str, qty: int, side: str):
    url = f"{ALPACA_BASE_URL}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side.lower(),
        "type": "market",
        "time_in_force": "gtc"
    }
    r = requests.post(url, json=payload, headers=HEADERS, timeout=10)
    if r.status_code in (200, 201):
        logging.info("✅ Order placed: %s %d %s", side.upper(), qty, symbol)
        return r.json()
    else:
        logging.error("Order failed: %s %s", r.status_code, r.text)
        raise RuntimeError(f"Order failed: {r.status_code} {r.text}")


def get_account_cash():
    url = f"{ALPACA_BASE_URL}/v2/account"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    j = r.json()
    return float(j.get("cash", 0.0))


def get_position_qty(symbol):
    url = f"{ALPACA_BASE_URL}/v2/positions/{symbol}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code == 200:
        j = r.json()
        return int(float(j.get("qty", 0)))
    return 0


# --------------- Indicator builder ----------------
def compute_indicators_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the common technical indicators used in training.
    Keeps naming consistent with training feature columns.
    """
    if df.empty:
        return df
    # make sure float types
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # SMAs
    df["sma_10"] = ta.sma(df["close"], length=10)
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["sma_50"] = ta.sma(df["close"], length=50)

    # RSI
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    # MACD
    macd = ta.macd(df["close"])
    df["macd"] = macd.get("MACD_12_26_9")
    df["macd_signal"] = macd.get("MACDs_12_26_9")
    df["macd_diff"] = macd.get("MACDh_12_26_9")

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20)
    df["bb_upper"] = bb.get("BBU_20_2.0")
    df["bb_middle"] = bb.get("BBM_20_2.0")
    df["bb_lower"] = bb.get("BBL_20_2.0")

    # Volatility and derived
    df["volatility"] = df["close"].rolling(window=10).std()
    df["vol_pct_change"] = df["volatility"].pct_change()
    df["rsi_cross_50"] = (df["rsi_14"] > 50).astype(int)

    return df


def build_feature_vector_for_last_row(symbol: str):
    """
    Build numpy vector matching feature_columns order using the most recent row.
    If feature missing fill 0.0 (safe fallback).
    """
    load_model_and_features()
    D = hist[symbol]
    if len(D["close"]) < 20:
        return None  # not enough history

    df = pd.DataFrame(
        {
            "open": list(D["open"]),
            "high": list(D["high"]),
            "low": list(D["low"]),
            "close": list(D["close"]),
            "volume": list(D["volume"]),
        },
        index=list(D["ts"])
    )
    df = compute_indicators_for_df(df)
    last = df.iloc[-1:]
    # build features in same order as saved; skip training-only columns gracefully
    vec = []
    for col in feature_columns:
        # skip label-like columns if accidentally saved
        if col.lower() in ("label", "label_raw", "future_close"):
            continue
        # attempt multiple casings to be robust
        if col in last.columns:
            v = last.iloc[0][col]
        elif col.lower() in last.columns:
            v = last.iloc[0][col.lower()]
        elif col.upper() in last.columns:
            v = last.iloc[0][col.upper()]
        else:
            v = 0.0
        if pd.isna(v):
            v = 0.0
        try:
            vec.append(float(v))
        except Exception:
            vec.append(0.0)
    return np.array(vec).reshape(1, -1), df


# ---------------- WebSocket handlers ----------------
def on_open(ws):
    logging.info("WS opened. Authenticating and subscribing to bars for %s", SYMBOL)
    auth_msg = {"action": "auth", "key": ALPACA_API_KEY, "secret": ALPACA_SECRET_KEY}
    ws.send(json.dumps(auth_msg))
    # subscribe to bars for the symbol
    sub_msg = {"action": "subscribe", "bars": [SYMBOL]}
    ws.send(json.dumps(sub_msg))


def on_message(ws, message):
    try:
        data = json.loads(message)
    except Exception:
        logging.debug("raw message: %s", message)
        return

    # Alpaca sends lists; unify handling
    items = data if isinstance(data, list) else [data]
    for item in items:
        # control messages
        if item.get("T") in ("success", "subscription", "error"):
            logging.debug("Market Data: %s", item)
            continue

        # bar messages type: 'b' (bar)
        if item.get("ev") == "b" or item.get("T") == "b":
            # Alpaca bar schema: 'ev' 'sym' 't' 'v' 'o' 'h' 'l' 'c' 'n' 'vw'
            sym = item.get("sym") or item.get("S")
            close = float(item.get("c") or item.get("close") or 0.0)
            o = float(item.get("o") or item.get("open") or close)
            h = float(item.get("h") or item.get("high") or close)
            l = float(item.get("l") or item.get("low") or close)
            v = float(item.get("v") or item.get("v") or 0.0)
            t_raw = item.get("t") or item.get("s") or item.get("timestamp")
            # parse timestamp robustly
            try:
                ts = pd.to_datetime(t_raw).to_pydatetime().replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
            # update history
            D = hist[sym]
            D["ts"].append(ts)
            D["open"].append(o)
            D["high"].append(h)
            D["low"].append(l)
            D["close"].append(close)
            D["volume"].append(v)

            logging.info("📊 Received bar for %s - Close Price: %.4f", sym, close)

            # If we have enough history, compute features and predict
            res = build_feature_vector_for_last_row(sym)
            print("after res data :")
            if res is None:
                print(" res is none data :")
                logging.debug("Not enough history for %s yet", sym)
                return
            print("after res not none data :")
            features, df_full = res
            print("start predict data :")
            # predict
            try:
                print("before load_model_and_features data :")
                load_model_and_features()
                print("after load_model_and_features data :")
                pred = int(model.predict(features)[0])
                print("prediction val: %s", pred)
            except Exception as e:
                logging.exception("Prediction failed: %s", e)
                return

            # action: 0=Hold,1=Buy,2=Sell (match training mapping)
            logging.info("Model predicted %s for %s", {0: "HOLD", 1: "BUY", 2: "SELL"}.get(pred, pred), sym)
            try:
                execute_trade_logic(sym, pred, close)
            except Exception as e:
                logging.exception("Trade execution failed: %s", e)


def on_error(ws, error):
    logging.error("Websocket error: %s", error)


def on_close(ws, code, reason):
    logging.warning("Websocket closed: %s %s", code, reason)


# ---------------- Trading logic ----------------
def execute_trade_logic(symbol: str, pred_label: int, price: float):
    """
    pred_label: 0=Hold,1=Buy,2=Sell
    price: latest close price used for decision (float)
    """
    # refresh on-account and position
    try:
        acct_cash = get_account_cash()
    except Exception:
        acct_cash = 0.0
    try:
        live_qty = get_position_qty(symbol)
    except Exception:
        live_qty = 0

    print("pred_label :"+ pred_label)
    local = local_positions.get(symbol, {"qty": live_qty, "entry_price": None, "entry_time": None})
    # Sync local qty with live if mismatch
    if local["qty"] != live_qty:
        local["qty"] = live_qty

    # BUY logic: only if zero position
    if pred_label == 1 and local["qty"] == 0:
        budget = max(1.0, acct_cash * TRADE_SIZE_FRACTION)
        qty = int(budget // price)
        if qty <= 0:
            logging.info("Budget insufficient to buy any shares (budget=%.2f, price=%.2f)", budget, price)
            return
        # place order
        try:
            place_market_order(symbol, qty, "buy")
        except Exception as e:
            logging.error("Order failed: %s", e)
            return
        # track local
        local_positions[symbol] = {"qty": qty, "entry_price": price, "entry_time": datetime.now(timezone.utc)}
        logging.info("Bought %d %s @ %.4f", qty, symbol, price)
        return

    # SELL logic: if we have position, sell all
    if pred_label == 2 and local["qty"] > 0:
        qty = local["qty"]
        try:
            place_market_order(symbol, qty, "sell")
        except Exception as e:
            logging.error("Order failed: %s", e)
            return
        logging.info("Sold %d %s @ %.4f", qty, symbol, price)
        # reset local
        local_positions[symbol] = {"qty": 0, "entry_price": None, "entry_time": None}
        return

    # SL/TP check (if holding)
    print("entry price :"+ local)
    if local["qty"] > 0 and local["entry_price"] is not None:
        change = (price - local["entry_price"]) / local["entry_price"]
        print("change : "+ change)
        if change <= -STOP_LOSS or change >= TAKE_PROFIT:
            qty = local["qty"]
            try:
                place_market_order(symbol, qty, "sell")
            except Exception as e:
                logging.error("SL/TP sell failed: %s", e)
                return
            logging.info("SL/TP: sold %d %s @ %.4f (change %.4f)", qty, symbol, price, change)
            local_positions[symbol] = {"qty": 0, "entry_price": None, "entry_time": None}


# ----------------- Runner -----------------
def run_ws():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logging.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        return

    load_model_and_features()
    ws = websocket.WebSocketApp(
        ALPACA_DATA_WS,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    while True:
        try:
            logging.info("Connecting to data websocket...")
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except KeyboardInterrupt:
            logging.info("Interrupted by user, closing.")
            break
        except Exception as e:
            logging.exception("WS loop exception: %s. Reconnecting in 5s...", e)
            time.sleep(5)


if __name__ == "__main__":
    logging.info("Starting paper trading websocket bot...")
    run_ws()
