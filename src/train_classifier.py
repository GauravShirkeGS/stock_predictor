# src/train_classifier.py

import pandas as pd
import numpy as np
import os
import pickle
import joblib
from collections import Counter
from data_fetcher import fetch_features
from src.features import add_technical_indicators
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from src.constants import FEATURE_COLUMNS

MODEL_PATH = "model/buy_sell_classifier.pkl"


def generate_labels(df, threshold=0.001):
    labels = []
    for i in range(len(df) - 1):
        current_close = df.iloc[i]["Close"]
        next_close = df.iloc[i + 1]["Close"]
        change_pct = (next_close - current_close) / current_close
        if change_pct > threshold:
            labels.append(1)   # Buy
        elif change_pct < -threshold:
            labels.append(2)  # Sell
        else:
            labels.append(0)   # Hold
    print("✅ Label Distribution:", Counter(labels))
    return labels


def prepare_training_data(df):
    X = df.iloc[:-1][FEATURE_COLUMNS]
    y = pd.Series(generate_labels(df))
    print("Label distribution:\n", y.value_counts(normalize=True))
    return X, y


def train_classifier(symbol="AAPL", interval="15min"):
    print("📥 Fetching data...")
    df = fetch_features(symbol, interval, lookback=3000)
    df = add_technical_indicators(df)
    df = df[:-1]  # Align for label shift

    if len(df) < 60:
        raise ValueError("Not enough data to train")

    print("⚙️ Preparing features and labels...")
    X, y = prepare_training_data(df)
    print(f"✅ Label Distribution: {Counter(y)}")

    print("🧠 Training XGBoost classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    clf.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ XGBoost model saved to {MODEL_PATH}")

    # Save feature list
    with open("model/features_used.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    y_pred = clf.predict(X_test)
    labels_present = sorted(unique_labels(y_test, y_pred))
    target_names = [ {2: "Sell", 0: "Hold", 1: "Buy"}[label] for label in labels_present ]
    print("📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, labels=labels_present, target_names=target_names))


if __name__ == "__main__":
    train_classifier()
