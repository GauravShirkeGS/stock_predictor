import pandas as pd
import joblib
from xgboost import XGBClassifier
from data_fetcher import fetch_stock_data
from feature_engineering import add_technical_indicators, add_labels
from config import MODEL_PATH
import os

def train_model():
    print("🔄 Fetching and preparing data...")

    # 1. Fetch raw data
    df = fetch_stock_data("AAPL", interval="1h")  # Train on AAPL
    df.columns = [col.lower() for col in df.columns]  # Make all columns lowercase

    # 2. Add indicators and labels
    df = add_technical_indicators(df)
    df = add_labels(df, future_window=5, threshold_buy=0.02, threshold_sell=-0.02)
    df.dropna(inplace=True)

    # 3. Features & Labels
    feature_cols = [col for col in df.columns if col not in ['label','Label', 'datetime', 'date', 'symbol']]
    X = df[feature_cols]
    y = df["Label"]
    print(df['Label'].value_counts(normalize=True) * 100)

    print("📊 Training data shape:", X.shape)

    # 4. Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)

    # 5. Save model and features
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, "model/feature_columns.pkl")

    print("✅ Model trained and saved.")
    print("📁 Features used:", feature_cols)

if __name__ == "__main__":
    train_model()
