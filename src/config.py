# config.py

# Model path
MODEL_PATH = "model/xgb_trading_model.pkl"

# Stocks to train/test on
STOCKS = ["AAPL", "TRP", "QQQ"]

# Candle interval (e.g., '15min', '1h', etc.)
INTERVAL = "15min"

# Feature columns to use for training
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_20", "RSI", "MACD", "MACD_signal",
    "BB_upper", "BB_lower", "Volatility"
]

# Capital and trading strategy settings for backtesting
INITIAL_CAPITAL = 100000  # Example: ₹1,00,000
POSITION_SIZE = 0.1       # 10% of capital per trade
STOP_LOSS_PCT = 0.02      # 2% stop-loss
TAKE_PROFIT_PCT = 0.04    # 4% take-profit
