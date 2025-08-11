import os
import json
import time
import logging
import asyncio
import websockets
import requests
import joblib
import numpy as np
from datetime import datetime
from pytz import timezone

# ================== CONFIG ==================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PK2D3Q5ROEM4ZAXJ99WQ")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "ginRvISIbqB81QLvvRvi7qx2OXRUtKdktLa7pZoN")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
MARKET_DATA_WSS = "wss://stream.data.alpaca.markets/v2/sip"  # Full US market feed
TRADING_WSS = "wss://paper-api.alpaca.markets/stream"

MODEL_PATH = "model/xgb_trading_model.pkl"
FEATURES_PATH = "model/feature_columns.pkl"

SYMBOLS = ["AAPL", "AMD", "NVDA", "MSFT"]  # Stocks under $200 for paper testing
TRADE_QTY = 1
# ============================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

model = None
feature_columns = None

def load_model():
    global model, feature_columns
    logging.info("Loading model...")
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    logging.info("Model and features loaded.")

def market_open():
    """Check if US market is open (9:30–16:00 ET)."""
    now = datetime.now(timezone("US/Eastern"))
    return now.hour >= 9 and now.minute >= 30 and now.hour < 16

def predict_action(features):
    """Predict Buy/Hold/Sell."""
    features = np.array(features).reshape(1, -1)
    pred = model.predict(features)[0]
    return {0: "Hold", 1: "Buy", 2: "Sell"}.get(pred, "Hold")

def place_order(symbol, side, qty):
    """Send order to Alpaca."""
    url = f"{BASE_URL}/v2/orders"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    order = {
        "symbol": symbol,
        "qty": qty,
        "side": side.lower(),
        "type": "market",
        "time_in_force": "gtc"
    }
    r = requests.post(url, headers=headers, json=order)
    if r.status_code == 200:
        logging.info(f"✅ Order placed: {side} {qty} {symbol}")
    else:
        logging.error(f"Order failed: {r.text}")

async def handle_market_data():
    """Stream market data from Alpaca."""
    async for ws in websockets.connect(MARKET_DATA_WSS, ping_interval=20, ping_timeout=20):
        try:
            # Authenticate
            await ws.send(json.dumps({
                "action": "auth",
                "key": ALPACA_API_KEY,
                "secret": ALPACA_SECRET_KEY
            }))
            logging.info("Connected to Market Data WebSocket.")

            # Subscribe to trades
            await ws.send(json.dumps({"action": "subscribe", "trades": SYMBOLS}))

            async for msg in ws:
                data = json.loads(msg)
                logging.info(f"Market Data: {data}")

                if not market_open():
                    continue

                # Here: fetch latest features → predict → trade
                # (Placeholder example)
                action = predict_action(np.random.rand(len(feature_columns)))
                print("=======================================")
                print("Action name : "+action)
                if action == "Buy":
                    place_order(SYMBOLS[0], "buy", TRADE_QTY)
                elif action == "Sell":
                    place_order(SYMBOLS[0], "sell", TRADE_QTY)

        except websockets.ConnectionClosed:
            logging.warning("Market data socket closed. Reconnecting...")
            await asyncio.sleep(5)
            continue

async def handle_trading_updates():
    """Stream account and order updates."""
    async for ws in websockets.connect(TRADING_WSS, ping_interval=20, ping_timeout=20):
        try:
            # Authenticate
            await ws.send(json.dumps({
                "action": "authenticate",
                "data": {"key_id": ALPACA_API_KEY, "secret_key": ALPACA_SECRET_KEY}
            }))
            logging.info("Connected to Trading Updates WebSocket.")

            # Listen for updates
            async for msg in ws:
                data = json.loads(msg)
                logging.info(f"Trading Update: {data}")

        except websockets.ConnectionClosed:
            logging.warning("Trading update socket closed. Reconnecting...")
            await asyncio.sleep(5)
            continue

async def main():
    load_model()
    await asyncio.gather(
        handle_market_data(),
        handle_trading_updates()
    )

if __name__ == "__main__":
    logging.info("🚀 Starting Paper Trading with WebSockets...")
    asyncio.run(main())
