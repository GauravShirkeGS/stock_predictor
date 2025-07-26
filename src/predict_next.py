import pandas as pd
import joblib

# Load model
model = joblib.load('../model/rf_candle_predictor.pkl')

# Load data
df = pd.read_csv('../data/data.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

# Get last two candles
last_two = df.tail(2)

# Prepare input features
latest_data = {
    'open_prev1': last_two.iloc[-1]['open'],
    'high_prev1': last_two.iloc[-1]['high'],
    'low_prev1':  last_two.iloc[-1]['low'],
    'close_prev1': last_two.iloc[-1]['close'],
    'open_prev2': last_two.iloc[-2]['open'],
    'high_prev2': last_two.iloc[-2]['high'],
    'low_prev2':  last_two.iloc[-2]['low'],
    'close_prev2': last_two.iloc[-2]['close'],
}

# Predict
X_input = pd.DataFrame([latest_data])
predicted = model.predict(X_input)[0]

# Calculate next candle's timestamp
last_time = df['time'].iloc[-1]
time_delta = last_time - df['time'].iloc[-2]
predicted_time = last_time + time_delta

# Show result
print(f"\n📅 Predicted Candle Time: {predicted_time}")
print("📈 Predicted Next Candle:")
print(f"Open:  {predicted[0]:.2f}")
print(f"High:  {predicted[1]:.2f}")
print(f"Low:   {predicted[2]:.2f}")
print(f"Close: {predicted[3]:.2f}")
