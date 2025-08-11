import os
import requests
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

import os
import requests
import pandas as pd

load_dotenv()

API_KEY = os.getenv("TWELVE_DATA_API_KEY")
BASE_URL = "https://api.twelvedata.com/time_series"


TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")


def fetch_stock_data(symbol: str, interval: str, outputsize: int = 1000) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given stock and interval using Twelve Data API.
    Returns a pandas DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": TWELVE_DATA_API_KEY,
        "outputsize": outputsize,
        "format": "JSON"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "values" not in data:
        print("Error fetching data:", data)
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])
    df.rename(columns={
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def fetch_ohlc_data(symbol: str, interval: str, outputsize=1000):
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": outputsize,
        "format": "JSON"
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "values" not in data:
        raise ValueError(f"Error fetching data: {data.get('message', 'Unknown error')}")

    df = pd.DataFrame(data["values"])
    df.rename(columns={
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)

    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    return df


import pandas as pd
import requests
import os

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "your_api_key_here")


def fetch_features(symbol, interval="15min", lookback=300):
    df = fetch_stock_data(symbol, interval, lookback)

    # Basic technical indicators
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_diff"] = df["macd"] - df["macd_signal"]

    bb = ta.bbands(df["close"], length=20)
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]
    df["bb_middle"] = (df["bb_upper"] + df["bb_lower"]) / 2

    df["volatility"] = df["close"].rolling(10).std()
    df["vol_pct_change"] = df["volume"].pct_change()

    # Derived features
    df["rsi_cross_50"] = (df["rsi"] > 50).astype(int)

    df.dropna(inplace=True)
    return df
