import pandas as pd
import numpy as np


def backtest_strategy(
    df,
    initial_capital=1000,
    stop_loss=0.02,
    take_profit=0.04,
    brokerage=0.001,
    delayed_signal=True  # Use previous candle's signal for realism
):
    """
    Backtest Buy/Sell/Hold predictions with capital, brokerage, stop-loss/take-profit logic.

    Parameters:
        - df: DataFrame with at least ['close', 'Predicted_Label']
        - initial_capital: Starting amount in ₹
        - stop_loss: Stop loss % (e.g. 0.02 for 2%)
        - take_profit: Take profit % (e.g. 0.04 for 4%)
        - brokerage: % fee per trade (e.g. 0.001 = 0.1%)
        - delayed_signal: if True, acts on previous candle's prediction

    Returns:
        - trades: DataFrame of trade logs
        - metrics: Dictionary of performance stats
        - capital_curve: Series of capital over time
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_time = None
    trade_log = []
    capital_curve = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        signal = prev_row["Predicted_Label"] if delayed_signal else row["Predicted_Label"]
        price = row["close"]
        timestamp = row.get('datetime', row.name)

        # Record current capital for curve
        capital_curve.append({
            'datetime': timestamp,
            'capital': capital + (position * price if position > 0 else 0)
        })

        # Execute Buy
        if signal == 1 and position == 0:
            quantity = capital // (price * (1 + brokerage))
            if quantity > 0:
                entry_price = price
                entry_time = timestamp
                position = quantity
                capital -= quantity * price * (1 + brokerage)

        # Execute Sell
        elif signal == 2 and position > 0:
            exit_price = price
            gross = position * exit_price
            net = gross * (1 - brokerage)
            pnl = net - (position * entry_price * (1 + brokerage))
            capital += net
            trade_log.append({
                'Entry': entry_time, 'Exit': timestamp,
                'Entry Price': entry_price, 'Exit Price': exit_price,
                'Quantity': position, 'PnL ₹': round(pnl, 2),
                'Capital': round(capital, 2),
                'Duration': str(pd.to_datetime(timestamp) - pd.to_datetime(entry_time))
            })
            position = 0

        # Check SL/TP even if signal is Hold
        elif position > 0:
            change = (price - entry_price) / entry_price
            if change <= -stop_loss or change >= take_profit:
                exit_price = price
                gross = position * exit_price
                net = gross * (1 - brokerage)
                pnl = net - (position * entry_price * (1 + brokerage))
                capital += net
                trade_log.append({
                    'Entry': entry_time, 'Exit': timestamp,
                    'Entry Price': entry_price, 'Exit Price': exit_price,
                    'Quantity': position, 'PnL ₹': round(pnl, 2),
                    'Capital': round(capital, 2),
                    'Duration': str(pd.to_datetime(timestamp) - pd.to_datetime(entry_time))
                })
                position = 0

    capital_series = pd.DataFrame(capital_curve).set_index('datetime')['capital']
    df_trades = pd.DataFrame(trade_log)
    metrics = compute_backtest_metrics(df_trades, initial_capital, capital, capital_series)

    return df_trades, metrics, capital_series


def compute_backtest_metrics(df_trades, initial_capital, final_capital, capital_series):
    returns = df_trades["PnL ₹"] / initial_capital
    total_return = (final_capital - initial_capital) / initial_capital * 100
    win_trades = df_trades[df_trades["PnL ₹"] > 0]
    loss_trades = df_trades[df_trades["PnL ₹"] < 0]

    win_rate = len(win_trades) / len(df_trades) * 100 if not df_trades.empty else 0
    sharpe_ratio = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if not returns.empty else 0

    drawdowns = capital_series / capital_series.cummax() - 1
    max_drawdown = drawdowns.min() * 100 if not drawdowns.empty else 0

    profit_factor = win_trades["PnL ₹"].sum() / abs(loss_trades["PnL ₹"].sum()) if not loss_trades.empty else np.inf
    expectancy = (df_trades["PnL ₹"].mean()) if not df_trades.empty else 0
    avg_duration = pd.to_timedelta(df_trades["Duration"]).mean() if not df_trades.empty else pd.Timedelta(0)

    return {
        "Total Return (%)": round(total_return, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Final Capital (₹)": round(final_capital, 2),
        "Number of Trades": len(df_trades),
        "Profit Factor": round(profit_factor, 2),
        "Expectancy (₹)": round(expectancy, 2),
        "Avg. Trade Duration": str(avg_duration)
    }
