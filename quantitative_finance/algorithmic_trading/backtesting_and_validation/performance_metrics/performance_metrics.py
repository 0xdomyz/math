import numpy as np
import pandas as pd

# Simulate equity curve
np.random.seed(99)
n_days = 500
daily_returns = np.random.randn(n_days) * 0.015 + 0.0005  # Mean 0.05%, vol 1.5%
equity_curve = 100000 * (1 + daily_returns).cumprod()
df = pd.DataFrame({'equity': equity_curve, 'returns': daily_returns})

# Cumulative return
cumulative_return = (df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0]

# CAGR
years = n_days / 252
cagr = (df['equity'].iloc[-1] / df['equity'].iloc[0]) ** (1 / years) - 1

# Sharpe ratio (assume 3% risk-free rate)
excess_returns = df['returns'] - 0.03 / 252
sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

# Sortino ratio (downside risk only)
downside_returns = df['returns'][df['returns'] < 0]
downside_std = downside_returns.std()
sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252)

# Maximum drawdown
running_max = df['equity'].cummax()
drawdown = (df['equity'] - running_max) / running_max
max_drawdown = drawdown.min()

# Calmar ratio
calmar_ratio = cagr / abs(max_drawdown)

# Simulate individual trades for win/loss stats
np.random.seed(55)
n_trades = 100
trade_pnl = np.random.randn(n_trades) * 500 + 50  # Avg $50, vol $500
winning_trades = trade_pnl[trade_pnl > 0]
losing_trades = trade_pnl[trade_pnl < 0]

hit_rate = len(winning_trades) / n_trades
avg_win = winning_trades.mean()
avg_loss = abs(losing_trades.mean())
profit_factor = winning_trades.sum() / abs(losing_trades.sum())
expectancy = (hit_rate * avg_win) - ((1 - hit_rate) * avg_loss)

# Print performance report
print("=" * 50)
print("PERFORMANCE METRICS REPORT")
print("=" * 50)
print(f"Cumulative Return:        {cumulative_return:>10.2%}")
print(f"CAGR:                     {cagr:>10.2%}")
print(f"Sharpe Ratio:             {sharpe_ratio:>10.2f}")
print(f"Sortino Ratio:            {sortino_ratio:>10.2f}")
print(f"Max Drawdown:             {max_drawdown:>10.2%}")
print(f"Calmar Ratio:             {calmar_ratio:>10.2f}")
print("-" * 50)
print(f"Number of Trades:         {n_trades:>10}")
print(f"Hit Rate:                 {hit_rate:>10.2%}")
print(f"Average Win:              ${avg_win:>9,.2f}")
print(f"Average Loss:             ${avg_loss:>9,.2f}")
print(f"Win/Loss Ratio:           {avg_win/avg_loss:>10.2f}")
print(f"Profit Factor:            {profit_factor:>10.2f}")
print(f"Expectancy per Trade:     ${expectancy:>9,.2f}")
print("=" * 50)