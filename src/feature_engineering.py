import pandas_ta as ta
import pandas as pd
import numpy as np

def add_technical_indicators(df):
    # ✅ 1. Basic Cleanup
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # ✅ 2. Add SMA indicators
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # ✅ 3. Add RSI (same period as used in training)
    df['rsi'] = ta.rsi(df['close'], length=14)

    # ✅ 4. MACD
    macd_df = ta.macd(df['close'])
    df['macd'] = macd_df['MACD_12_26_9']
    df['macd_signal'] = macd_df['MACDs_12_26_9']
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # ✅ 5. Bollinger Bands
    bb_df = ta.bbands(df['close'], length=20)
    df['bb_upper'] = bb_df['BBU_20_2.0']
    df['bb_lower'] = bb_df['BBL_20_2.0']
    df['bb_middle'] = (df['bb_upper'] + df['bb_lower']) / 2

    # ✅ 6. Volatility Features
    df['vol_pct_change'] = df['volume'].pct_change()
    df['volatility'] = df['close'].rolling(window=10).std()

    # ✅ 7. RSI Cross Feature (bool/int)
    df['rsi_cross_50'] = (df['rsi'] > 50).astype(int)

    # ✅ 8. Drop rows with any NaN (from rolling indicators)
    df.dropna(inplace=True)
    return df


def add_labels(df, future_window=5, threshold_buy=0.02, threshold_sell=-0.02):
    """
    Adds labels to the dataframe: 1 = Buy, -1 = Sell, 0 = Hold
    Based on forward return percentage over the next `future_window` candles.
    """
    df = df.copy()

    # Calculate future return percentage
    df['Future_Close'] = df['close'].shift(-future_window)
    df['Return_%'] = (df['Future_Close'] - df['close']) / df['close']

    # Apply thresholds
    conditions = [
        df['Return_%'] > threshold_buy,
        df['Return_%'] < threshold_sell
    ]
    choices = [1, 2]  # Buy, Sell

    df['Label'] = np.select(conditions, choices, default=0)  # Hold = 0

    df.drop(columns=['Future_Close', 'Return_%'], inplace=True)
    return df


