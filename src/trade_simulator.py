# src/trade_simulator.py

def simulate_trades(df, predictions, initial_capital=100000, stop_loss=0.02, take_profit=0.04):
    capital = initial_capital
    position = None
    entry_price = 0
    trades = []

    for i in range(len(predictions)):
        action = predictions[i]
        row = df.iloc[i]
        close = row["close"]

        if action == 1 and position is None:
            # Buy
            position = {
                "entry_price": close,
                "entry_index": i
            }

        elif action == -1 and position:
            # Sell
            exit_price = close
            return_pct = ((exit_price - position["entry_price"]) / position["entry_price"]) * 100
            capital *= (1 + return_pct / 100)

            trades.append({
                "Buy @": round(position["entry_price"], 2),
                "Sell @": round(exit_price, 2),
                "Return%": round(return_pct, 2)
            })

            position = None  # Reset

        elif position:
            # Check SL/TP if holding
            high = row["high"]
            low = row["low"]

            sl_price = position["entry_price"] * (1 - stop_loss)
            tp_price = position["entry_price"] * (1 + take_profit)

            if low <= sl_price:
                # Stop-loss hit
                return_pct = ((sl_price - position["entry_price"]) / position["entry_price"]) * 100
                capital *= (1 + return_pct / 100)
                trades.append({
                    "Buy @": round(position["entry_price"], 2),
                    "Sell @": round(sl_price, 2),
                    "Return%": round(return_pct, 2),
                    "Reason": "Stop Loss"
                })
                position = None

            elif high >= tp_price:
                # Take-profit hit
                return_pct = ((tp_price - position["entry_price"]) / position["entry_price"]) * 100
                capital *= (1 + return_pct / 100)
                trades.append({
                    "Buy @": round(position["entry_price"], 2),
                    "Sell @": round(tp_price, 2),
                    "Return%": round(return_pct, 2),
                    "Reason": "Take Profit"
                })
                position = None

    return capital, trades
