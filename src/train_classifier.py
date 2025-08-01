# src/train_classifier.py

from data_fetcher import fetch_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from collections import Counter

MODEL_PATH = "model/buy_sell_classifier.pkl"


def generate_labels(df, threshold=0.001):
    """
    Generate Buy (1), Sell (-1), Hold (0) labels based on % price movement.
    """
    labels = []
    for i in range(len(df) - 1):
        current_close = df.iloc[i]["Close"]
        next_close = df.iloc[i + 1]["Close"]
        change_pct = (next_close - current_close) / current_close

        if change_pct > threshold:
            labels.append(1)  # Buy
        elif change_pct < -threshold:
            labels.append(-1)  # Sell
        else:
            labels.append(0)  # Hold

    print("✅ Label Distribution:", Counter(labels))
    return labels


def prepare_training_data(df):

    X = df.iloc[:-1][[
        "Open", "High", "Low", "Close", "Volume",
        "sma_10", "sma_20", "rsi_14", "macd", "macd_signal",
        "bb_upper", "bb_lower", "volatility"
    ]]
    y = generate_labels(df)
    return X, y


def train_classifier(symbol="AAPL", interval="15min"):
    print("📥 Fetching data...")
    df = fetch_features(symbol, interval, lookback=3000)

    # Drop last row to align features with labels
    df = df[:-1]

    if len(df) < 60:
        raise ValueError("Not enough data to train")

    print("⚙️ Preparing features and labels...")
    X, y = prepare_training_data(df)
    from collections import Counter
    print(f"✅ Label Distribution: {Counter(y)}")

    print("🧠 Training classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight={-1: 1.5, 0: 0.5, 1: 1.5},
        random_state=42
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    from sklearn.utils.multiclass import unique_labels

    print("📊 Classification Report:\n")

    # Map label values to names
    label_names = {-1: "Sell", 0: "Hold", 1: "Buy"}

    # Get only the labels that are actually present
    labels_present = sorted(unique_labels(y_test, y_pred))

    # Generate readable names for them
    target_names = [label_names[label] for label in labels_present]

    # Print classification report safely
    print(classification_report(y_test, y_pred, labels=labels_present, target_names=target_names))

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Classifier saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_classifier()
