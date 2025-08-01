# src/backtester.py

import pandas as pd
import joblib
from src.data_fetcher import fetch_features
from src.constants import FEATURES_USED

INITIAL_CAPITAL = 100.0  # Start with ₹100
symbol = "AAPL"
interval = "1h"

def backtest():
    print("🔄 Fetching data for backtesting...")
    df = fetch_features(symbol, interval)

    if df.isnull().values.any():
        df = df.dropna()

    model = joblib.load("model/buy_sell_classifier.pkl")

    df["prediction"] = model.predict(df[FEATURES_USED])

    capital = INITIAL_CAPITAL
    holding = 0.0  # Number of shares held
    buy_price = 0.0

    history = []

    for i, row in df.iterrows():
        action = row["prediction"]
        close_price = row["Close"]

        # Buy
        if action == 1 and capital > 0:
            holding = capital / close_price
            buy_price = close_price
            capital = 0
            history.append((row.name, "BUY", close_price))

        # Sell
        elif action == -1 and holding > 0:
            capital = holding * close_price
            holding = 0
            history.append((row.name, "SELL", close_price))

        # Hold
        else:
            history.append((row.name, "HOLD", close_price))

    # Final valuation
    final_value = capital + (holding * df.iloc[-1]["Close"])
    profit = final_value - INITIAL_CAPITAL
    profit_pct = (profit / INITIAL_CAPITAL) * 100

    print(f"\n📈 Final value: ₹{final_value:.2f}")
    print(f"💰 Profit/Loss: ₹{profit:.2f} ({profit_pct:.2f}%)")

    # Optionally print history
    for entry in history[-10:]:
        print(f"{entry[0]} → {entry[1]} at ₹{entry[2]:.2f}")

if __name__ == "__main__":
    backtest()
