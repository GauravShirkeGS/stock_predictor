# src/features.py

import pandas as pd
import pandas_ta as ta


def add_technical_indicators(df):
    df = df.copy()

    # Basic Indicators
    df["sma_10"] = ta.sma(df["Close"], length=10)
    df["sma_20"] = ta.sma(df["Close"], length=20)
    df["rsi_14"] = ta.rsi(df["Close"], length=14)

    # MACD
    macd = ta.macd(df["Close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]

    # Bollinger Bands
    bb = ta.bbands(df["Close"], length=20)
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]

    # Volatility (standard deviation)
    df["volatility"] = df["Close"].rolling(window=10).std()

    # Custom engineered features
    df["macd_diff"] = df["macd"] - df["macd_signal"]
    df["rsi_cross_50"] = (df["rsi_14"] > 50).astype(int)
    df["vol_pct_change"] = df["Volume"].pct_change().fillna(0)

    bb = ta.bbands(df["Close"], length=20, std=2)
    df["bb_middle"] = bb["BBM_20_2.0"]

    # Drop rows with any NaNs created by indicators
    df = df.dropna()

    return df
