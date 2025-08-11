# src/predict_action.py

import joblib
import numpy as np
from src.model_handler import load_model_and_features
from src.data_fetcher import fetch_features
import pandas as pd


MODEL_PATH = "model/buy_sell_classifier.pkl"

# def predict_trading_action(symbol: str, interval: str, datetime: str) -> str:
#     """
#     Predict trading action (Buy/Sell/Hold) for a given symbol, interval, and datetime.
#     Uses previous row's technical indicators to predict the action.
#     """
#     print(f"🔍 Predicting for {symbol} at {datetime} using {interval} interval...")
#
#     df = fetch_features(symbol, interval)
#     df = df.dropna()
#
#     if datetime not in df.index:
#         raise ValueError(f"❌ Datetime '{datetime}' not found in the available data")
#
#     idx = df.index.get_loc(datetime)
#     if idx < 1:
#         raise ValueError("❌ Not enough historical data to predict for this timestamp")
#
#     row = df.iloc[idx - 1]  # Use previous candle's indicators
#
#     features = [
#         row["Open"], row["High"], row["Low"], row["Close"], row["Volume"],
#         row["sma_10"], row["sma_20"], row["rsi_14"],
#         row["macd"], row["macd_signal"],
#         row["bb_upper"], row["bb_lower"],
#         row["volatility"]
#     ]
#
#     model = joblib.load(MODEL_PATH)
#     pred = model.predict([features])[0]
#
#     label_map = {1: "Buy", 0: "Hold", -1: "Sell"}
#     action = label_map.get(pred, "Unknown")
#
#     print(f"✅ Predicted Action: {action}")
#     return action

def predict_trading_action(symbol, interval, datetime_str):
    df = fetch_features(symbol, interval)
    df = df.dropna()
    import pandas as pd

    # Convert both sides to datetime objects (safe and robust)
    df["date"] = pd.to_datetime(df["date"])
    datetime_obj = pd.to_datetime(datetime_str)

    # Filter row safely
    row = df[df["date"] == datetime_obj]

    if row.empty:
        raise ValueError(f"❌ Datetime '{datetime_str}' not found in the available data")

    # Load model & features
    model, feature_columns = load_model_and_features()
    X = row[feature_columns]
    prediction = model.predict(X)[0]

    label_map = {0: "Sell", 1: "Hold", 2: "Buy"}
    return label_map.get(prediction, "Unknown")
