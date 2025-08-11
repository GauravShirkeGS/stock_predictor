import pandas as pd
import joblib
from data_fetcher import fetch_stock_data
from feature_engineering import add_technical_indicators, add_labels
from backtester import backtest_strategy
from config import MODEL_PATH, STOCKS, INTERVAL

# Load trained model
model = joblib.load(MODEL_PATH)

# Load the exact feature columns used during training
with open("model/feature_columns.pkl", "rb") as f:
    FEATURE_COLUMNS = joblib.load(f)

for stock in STOCKS:
    try:
        print(f"\n🔁 Backtesting on {stock}")

        # Step 1: Fetch data
        df = fetch_stock_data(stock, interval=INTERVAL)
        print(f"\n📄 Raw fetched data for {stock}:")
        print(df.tail(5))  # show recent rows only
        print(f"📑 Columns: {df.columns.tolist()}")

        # Step 2: Normalize column names
        df.columns = df.columns.str.lower()

        # Step 3: Rename datetime to 'date'
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)

        # Step 4: Validate required OHLCV columns
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")

        # Step 5: Feature engineering
        df = add_technical_indicators(df)
        df = add_labels(df)

        df.dropna(inplace=True)

        # Step 6: Check missing model features
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing feature columns for {stock}: {missing_cols}")
            continue

        # Step 7: Predict
        X = df[FEATURE_COLUMNS]
        df["Predicted_Label"] = model.predict(X)

        # Step 8: Backtest
        df_trades, metrics = backtest_strategy(df)


        # Step 9: Results
        print(f"\n📊 {stock} Backtest Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}")

        print("\n📄 Trade Log (last 5):")
        if df_trades.empty:
            print("❗ No trades were executed.")
        else:
            print(df_trades.tail(5))

    except Exception as e:
        print(f"❌ Error during backtest on {stock}: {e}")
