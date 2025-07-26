from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), "..", "model", "rf_candle_predictor.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

import os
data_path = os.path.join(os.path.dirname(__file__), "../data/data.csv")
data = pd.read_csv(os.path.abspath(data_path))
data['time'] = pd.to_datetime(data['time'])

# FastAPI instance
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to OHLC Predictor API"}

@app.get("/predict")
def predict_next_candle(input_datetime: str = Query(..., description="Format: YYYY-MM-DD HH:MM")):
    try:
        # Convert input
        dt = datetime.strptime(input_datetime, "%Y-%m-%d %H:%M")

        # Filter up to that datetime
        df_filtered = data[data['time'] <= dt].sort_values('time')

        if len(df_filtered) < 2:
            return {"error": "Not enough data before the given time."}

        prev1 = df_filtered.iloc[-1]
        prev2 = df_filtered.iloc[-2]

        features = pd.DataFrame([{
            'open_prev1': prev1['open'],
            'high_prev1': prev1['high'],
            'low_prev1': prev1['low'],
            'close_prev1': prev1['close'],
            'open_prev2': prev2['open'],
            'high_prev2': prev2['high'],
            'low_prev2': prev2['low'],
            'close_prev2': prev2['close'],
        }])

        prediction = model.predict(features)[0]
        next_time = prev1['time'] + timedelta(minutes=15)

        return {
            "predicted_candle_time": next_time.strftime("%Y-%m-%d %H:%M"),
            "predicted_open": round(prediction[0], 2),
            "predicted_high": round(prediction[1], 2),
            "predicted_low": round(prediction[2], 2),
            "predicted_close": round(prediction[3], 2),
        }

    except Exception as e:
        return {"error": str(e)}
