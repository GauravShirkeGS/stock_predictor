import joblib
import numpy as np
import os

# Load the model once (avoid reloading each time)
MODEL_PATH = os.path.join("model", "rf_candle_predictor.pkl")
model = joblib.load(MODEL_PATH)

def predict_next_candle(features):
    """
    Takes a 1D numpy array of features and returns the predicted OHLC values.
    """
    if not isinstance(features, np.ndarray):
        features = np.array(features).reshape(1, -1)
    elif len(features.shape) == 1:
        features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    return {
        "predicted_open": round(prediction[0], 2),
        "predicted_high": round(prediction[1], 2),
        "predicted_low": round(prediction[2], 2),
        "predicted_close": round(prediction[3], 2),
    }

def predict_trade_action(features):
    if not isinstance(features, np.ndarray):
        features = np.array(features).reshape(1, -1)
    elif len(features.shape) == 1:
        features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    return {
        -1: "Sell",
         0: "Hold",
         1: "Buy"
    }[prediction]
