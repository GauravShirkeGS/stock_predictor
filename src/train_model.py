from data_fetcher import fetch_features
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

MODEL_PATH = "model/buy_sell_classifier.pkl"

def generate_labels(df, threshold=0.002):  # ← Add this function
    """
    Labels each row as:
    - 1 for Buy
    - -1 for Sell
    - 0 for Hold
    """
    labels = []
    for i in range(len(df) - 1):
        current_close = df.iloc[i]["Close"]
        next_close = df.iloc[i + 1]["Close"]
        change_pct = (next_close - current_close) / current_close

        if change_pct > threshold:
            labels.append(1)   # Buy
        elif change_pct < -threshold:
            labels.append(-1)  # Sell
        else:
            labels.append(0)   # Hold

    return labels

def prepare_classification_data(df):  # ← Add this too
    X = []
    for i in range(len(df) - 1):
        current = df.iloc[i]
        features = [
            current["Open"], current["High"], current["Low"], current["Close"], current["Volume"],
            current["sma_10"], current["sma_20"], current["rsi_14"],
            current["macd"], current["macd_signal"],
            current["bb_upper"], current["bb_lower"],
            current["volatility"]
        ]
        X.append(features)

    y = generate_labels(df)
    return np.array(X), np.array(y)

def train_classifier(symbol="AAPL", interval="15min"):  # ← New training function
    print("Fetching data...")
    df = fetch_features(symbol=symbol, interval=interval)
    df.dropna(inplace=True)

    print("Preparing data...")
    X, y = prepare_classification_data(df)

    print("Training classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_classifier()
