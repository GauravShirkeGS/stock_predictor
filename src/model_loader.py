import joblib

def load_model(model_path="model/rf_candle_predictor.pkl"):
    return joblib.load(model_path)
