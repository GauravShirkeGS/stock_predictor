from src.data_fetcher import fetch_features
import pandas as pd

if __name__ == "__main__":
    symbol = "AAPL"  # 🔁 Replace with your stock symbol
    interval = "15min"  # ✅ 15-minute candle

    print(f"Fetching {interval} data for {symbol}...")
    df = fetch_features(symbol, interval)

    if df.empty:
        print("No data found.")
    else:
        filename = f"{symbol}_{interval}_data.csv"
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
