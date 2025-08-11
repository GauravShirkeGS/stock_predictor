import joblib
import numpy as np
import os

MODEL_PATH = os.path.join("model", "xgb_trading_model.pkl")
FEATURES_PATH = os.path.join("model", "feature_columns.pkl")

model = None
feature_columns = None

def load_model_and_features():
    """
    Loads the XGBoost classification model and feature column list.
    """
    global model, feature_columns

    if model is None:
        model = joblib.load(MODEL_PATH)

    if feature_columns is None:
        feature_columns = joblib.load(FEATURES_PATH)

    return model, feature_columns

def predict_trade_action(features: list):
    """
    Predicts Buy/Hold/Sell from feature list using XGBoost classification model.
    """
    model, feature_columns = load_model_and_features()

    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]

    label_map = { 2: "Sell", 0: "Hold", 1: "Buy" }

    return label_map.get(prediction, "Unknown")

