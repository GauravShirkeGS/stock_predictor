import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Resolve correct path
base_dir = os.path.dirname(os.path.dirname(__file__))  # Goes one level up from /src
data_path = os.path.join(base_dir, "data", "data.csv")

# Load data
df = pd.read_csv(data_path)
df['time'] = pd.to_datetime(df['time'])
df.sort_values('time', inplace=True)

X = []
y = []

for i in range(2, len(df)):
    prev2 = df.iloc[i-2]
    prev1 = df.iloc[i-1]
    curr = df.iloc[i]

    X.append([
        prev2['open'], prev2['high'], prev2['low'], prev2['close'],
        prev1['open'], prev1['high'], prev1['low'], prev1['close'],
    ])
    y.append([curr['open'], curr['high'], curr['low'], curr['close']])

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
model_path = os.path.join(base_dir, "model", "rf_candle_predictor.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved at:", model_path)
