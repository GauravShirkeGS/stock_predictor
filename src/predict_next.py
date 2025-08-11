import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_fetcher import fetch_features
from src.model_handler import predict_trade_action, feature_columns
import pandas as pd



def prepare_features(df):
    """
    Prepare latest row using saved feature columns.
    """
    latest = df.iloc[-1]
    return latest[feature_columns].values.reshape(1, -1)


def get_prediction(symbol="AAPL", interval="15min"):
    """
    Fetch latest features and predict Buy/Sell/Hold action.
    """
    df = fetch_features(symbol=symbol, interval=interval)

    if df is None or len(df) < 30:
        return {"error": "Not enough data to predict"}

    features = prepare_features(df)
    action = predict_trade_action(df)

    last_time = pd.to_datetime(df["Date"].iloc[-1])
    interval_minutes = int(interval.replace("min", "")) if "min" in interval else 1
    next_time = last_time + pd.Timedelta(minutes=interval_minutes)

    return {
        "predicted_action": action,
        "predicted_candle_time": next_time.strftime("%Y-%m-%d %H:%M")
    }
