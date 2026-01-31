import numpy as np
import pandas as pd

# Simulate price data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')
prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.01))
df = pd.DataFrame({'price': prices, 'date': dates})

# Moving averages
df['sma_50'] = df['price'].rolling(50).mean()
df['sma_200'] = df['price'].rolling(200).mean()

# Signal generation: 1 = long, 0 = flat, -1 = short
df['signal'] = 0
df.loc[df['sma_50'] > df['sma_200'], 'signal'] = 1
df.loc[df['sma_50'] <= df['sma_200'], 'signal'] = 0

# Position changes (detect crossovers)
df['position'] = df['signal'].diff()  # 1 = buy, -1 = sell, 0 = hold

# Execution: fill at next day's price with slippage
slippage_bps = 5  # 5 basis points
commission = 10  # $10 per trade
df['fill_price'] = df['price'].shift(-1)  # Next day's open (simplified)

# Calculate transaction costs
df['slippage_cost'] = 0.0
df.loc[df['position'] != 0, 'slippage_cost'] = (
    df.loc[df['position'] != 0, 'fill_price'] * (slippage_bps / 10000)
)
df['commission_cost'] = 0.0
df.loc[df['position'] != 0, 'commission_cost'] = commission

# Portfolio value (assume 1 share per trade for simplicity)
df['holdings'] = df['signal']  # 1 when long, 0 when flat
df['holdings'] = df['holdings'].fillna(method='ffill').fillna(0)

# Daily PnL
df['daily_pnl'] = df['holdings'].shift(1) * df['price'].diff()
df.loc[df['position'] != 0, 'daily_pnl'] -= (
    df.loc[df['position'] != 0, 'slippage_cost'] + 
    df.loc[df['position'] != 0, 'commission_cost']
)

# Cumulative equity
initial_capital = 10000
df['equity'] = initial_capital + df['daily_pnl'].fillna(0).cumsum()

# Performance metrics
total_return = (df['equity'].iloc[-1] - initial_capital) / initial_capital
sharpe_ratio = df['daily_pnl'].mean() / df['daily_pnl'].std() * np.sqrt(252)
max_drawdown = (df['equity'].cummax() - df['equity']).max() / df['equity'].cummax().max()
num_trades = df['position'].abs().sum()

print(f"Total Return: {total_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Number of Trades: {int(num_trades)}")
print(f"Final Equity: ${df['equity'].iloc[-1]:,.2f}")