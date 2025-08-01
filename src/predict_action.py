# src/predict_action.py

import joblib
import numpy as np
from data_fetcher import fetch_features

MODEL_PATH = "model/buy_sell_classifier.pkl"

def predict_trading_action(symbol: str, interval: str, datetime: str) -> str:
    """
    Predict trading action (Buy/Sell/Hold) for a given symbol, interval, and datetime.
    Uses previous row's technical indicators to predict the action.
    """
    print(f"🔍 Predicting for {symbol} at {datetime} using {interval} interval...")

    df = fetch_features(symbol, interval)
    df = df.dropna()

    if datetime not in df.index:
        raise ValueError(f"❌ Datetime '{datetime}' not found in the available data")

    idx = df.index.get_loc(datetime)
    if idx < 1:
        raise ValueError("❌ Not enough historical data to predict for this timestamp")

    row = df.iloc[idx - 1]  # Use previous candle's indicators

    features = [
        row["Open"], row["High"], row["Low"], row["Close"], row["Volume"],
        row["sma_10"], row["sma_20"], row["rsi_14"],
        row["macd"], row["macd_signal"],
        row["bb_upper"], row["bb_lower"],
        row["volatility"]
    ]

    model = joblib.load(MODEL_PATH)
    pred = model.predict([features])[0]

    label_map = {1: "Buy", 0: "Hold", -1: "Sell"}
    action = label_map.get(pred, "Unknown")

    print(f"✅ Predicted Action: {action}")
    return action
