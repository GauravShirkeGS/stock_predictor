# src/api.py

from fastapi import FastAPI, HTTPException, Query
from src.predict_action import predict_trading_action
from src.data_fetcher import fetch_features
import  numpy as np
import pandas as pd
from datetime import datetime as dt

app = FastAPI(title="Stock Trading Signal API")


@app.get("/predict")
def predict_action(
    symbol: str = Query(..., description="Stock symbol, e.g., AAPL"),
    interval: str = Query(..., description="Candle interval, e.g., 15min"),
    datetime_str: str = Query(..., description="Datetime in format YYYY-MM-DD HH:MM:SS")
):
    """
    Predict Buy/Sell/Hold action for a specific timestamp.
    """
    try:
        action = predict_trading_action(symbol, interval, datetime_str)
        return {
            "symbol": symbol,
            "interval": interval,
            "datetime": datetime_str,
            "action": action
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/predict/latest")
def latest_prediction(
    symbol: str = Query(..., description="Stock symbol, e.g., AAPL"),
    interval: str = Query(..., description="Candle interval, e.g., 15min")
):
    """
    Predict Buy/Sell/Hold action for the latest available data.
    """
    try:
        df = fetch_features(symbol, interval)
        df = df.dropna()

        if df is None or df.empty or len(df) < 2:
            raise ValueError("Not enough valid data to make a prediction.")

        # Convert integer timestamp to datetime (if it's not already)
        latest_datetime_str = df.iloc[-1]["date"]

        print("step-1")
        action = predict_trading_action(symbol, interval, latest_datetime_str)
        print("step-2")
        return {
            "symbol": symbol,
            "interval": interval,
            "latest_datetime": latest_datetime_str,
            "action": action
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
