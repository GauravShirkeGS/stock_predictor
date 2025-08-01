# src/api.py

from fastapi import FastAPI, HTTPException, Query
from predict_action import predict_trading_action
from data_fetcher import fetch_features
from datetime import datetime as dt

app = FastAPI(title="Stock Trading Signal API")


@app.get("/predict")
def predict_action(
    symbol: str = Query(..., example="AAPL"),
    interval: str = Query(..., example="15min"),
    datetime_str: str = Query(..., example="2024-06-20 15:30:00")
):
    """
    Predict Buy/Sell/Hold action for a given datetime.
    """
    try:
        action = predict_trading_action(symbol, interval, datetime_str)
        return {"symbol": symbol, "interval": interval, "datetime": datetime_str, "action": action}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict/latest")
def latest_prediction(
    symbol: str = Query(..., example="AAPL"),
    interval: str = Query(..., example="15min")
):
    """
    Predict Buy/Sell/Hold action for the latest available data.
    """
    try:
        df = fetch_features(symbol, interval)
        df = df.dropna()
        if len(df) < 2:
            raise ValueError("Not enough data to make a prediction")

        latest_index = df.index[-1]
        latest_datetime_str = latest_index.strftime("%Y-%m-%d %H:%M:%S")

        action = predict_trading_action(symbol, interval, latest_datetime_str)
        return {
            "symbol": symbol,
            "interval": interval,
            "latest_datetime": latest_datetime_str,
            "action": action
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
