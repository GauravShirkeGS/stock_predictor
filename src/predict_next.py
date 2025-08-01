# src/predict_next.py

from src.data_fetcher import fetch_features
from src.model_handler import predict_next_candle
import pandas as pd
from src.constants import FEATURES_USED


def prepare_features(df):
    """
    Prepare features from technical indicator-enhanced dataframe.
    """
    # Select only the latest row for prediction
    latest = df.iloc[-1]

    # Choose relevant features
    features = [
        latest["Open"], latest["High"], latest["Low"], latest["Close"], latest["Volume"],
        latest["sma_10"], latest["sma_20"], latest["rsi_14"],
        latest["macd"], latest["macd_signal"],
        latest["bb_upper"], latest["bb_lower"],
        latest["volatility"]
    ]

    return features

def get_prediction(symbol="AAPL", interval="15min"):
    """
    Get real-time features and predict the next OHLC candle.
    """
    df = fetch_features(symbol=symbol, interval=interval)

    if df is None or len(df) < 30:
        return {"error": "Not enough data to predict"}

    features = prepare_features(df)
    prediction = predict_next_candle(features)

    # Estimate next candle time
    last_time = pd.to_datetime(df["Date"].iloc[-1])
    interval_minutes = int(interval.replace("min", "")) if "min" in interval else 1
    next_time = last_time + pd.Timedelta(minutes=interval_minutes)

    prediction["predicted_candle_time"] = next_time.strftime("%Y-%m-%d %H:%M")
    return prediction
