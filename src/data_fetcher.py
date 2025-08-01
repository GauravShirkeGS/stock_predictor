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
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
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


def fetch_features(symbol: str, interval: str, lookback: int = 1000) -> pd.DataFrame:
    df = fetch_ohlc_data(symbol, interval, lookback)

    # ✅ Rename Date to datetime and convert to datetime type
    if "Date" in df.columns:
        df.rename(columns={"Date": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Add TA indicators
    df["sma_10"] = ta.sma(df["Close"], length=10)
    df["sma_20"] = ta.sma(df["Close"], length=20)
    df["rsi_14"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]

    bb = ta.bbands(df["Close"], length=20)
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]

    df["volatility"] = df["Close"].rolling(window=10).std()

    df.dropna(inplace=True)

    return df
