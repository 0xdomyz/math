import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic price data with mean reversion
np.random.seed(777)
n_days = 1000
prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
prices = prices - np.linspace(0, 50, n_days)  # Inject mean reversion trend
df = pd.DataFrame({'price': prices})

# Split: In-sample (70%), Out-of-sample (30%)
split_idx = int(0.7 * n_days)
is_data = df.iloc[:split_idx]
oos_data = df.iloc[split_idx:]

# Parameter grid: lookback period for z-score calculation
lookback_range = range(10, 201, 10)
is_sharpes = []
oos_sharpes = []

for lookback in lookback_range:
    # In-sample optimization
    is_copy = is_data.copy()
    is_copy['rolling_mean'] = is_copy['price'].rolling(lookback).mean()
    is_copy['rolling_std'] = is_copy['price'].rolling(lookback).std()
    is_copy['z_score'] = (is_copy['price'] - is_copy['rolling_mean']) / is_copy['rolling_std']
    is_copy['signal'] = 0
    is_copy.loc[is_copy['z_score'] < -2, 'signal'] = 1  # Buy oversold
    is_copy.loc[is_copy['z_score'] > 2, 'signal'] = -1  # Sell overbought
    is_copy['returns'] = is_copy['price'].pct_change()
    is_copy['strategy_returns'] = is_copy['signal'].shift(1) * is_copy['returns']
    is_sharpe = is_copy['strategy_returns'].mean() / is_copy['strategy_returns'].std() * np.sqrt(252)
    is_sharpes.append(is_sharpe if not np.isnan(is_sharpe) else 0)
    
    # Out-of-sample test
    oos_copy = oos_data.copy()
    oos_copy['rolling_mean'] = oos_copy['price'].rolling(lookback).mean()
    oos_copy['rolling_std'] = oos_copy['price'].rolling(lookback).std()
    oos_copy['z_score'] = (oos_copy['price'] - oos_copy['rolling_mean']) / oos_copy['rolling_std']
    oos_copy['signal'] = 0
    oos_copy.loc[oos_copy['z_score'] < -2, 'signal'] = 1
    oos_copy.loc[oos_copy['z_score'] > 2, 'signal'] = -1
    oos_copy['returns'] = oos_copy['price'].pct_change()
    oos_copy['strategy_returns'] = oos_copy['signal'].shift(1) * oos_copy['returns']
    oos_sharpe = oos_copy['strategy_returns'].mean() / oos_copy['strategy_returns'].std() * np.sqrt(252)
    oos_sharpes.append(oos_sharpe if not np.isnan(oos_sharpe) else 0)

# Find optimal parameter
optimal_idx = np.argmax(is_sharpes)
optimal_lookback = list(lookback_range)[optimal_idx]
optimal_is_sharpe = is_sharpes[optimal_idx]
optimal_oos_sharpe = oos_sharpes[optimal_idx]

# Calculate degradation ratio
degradation_ratio = optimal_oos_sharpe / optimal_is_sharpe if optimal_is_sharpe != 0 else 0

print(f"Optimal Lookback: {optimal_lookback} days")
print(f"In-Sample Sharpe: {optimal_is_sharpe:.2f}")
print(f"Out-of-Sample Sharpe: {optimal_oos_sharpe:.2f}")
print(f"Degradation Ratio: {degradation_ratio:.2f}")

if degradation_ratio < 0.4:
    print("⚠️  WARNING: Severe overfitting detected!")
elif degradation_ratio < 0.7:
    print("⚠️  Moderate overfitting; review parameter stability")
else:
    print("✓ Acceptable OOS performance")

# Plot parameter sensitivity
plt.figure(figsize=(10, 5))
plt.plot(lookback_range, is_sharpes, label='In-Sample Sharpe', marker='o')
plt.plot(lookback_range, oos_sharpes, label='Out-of-Sample Sharpe', marker='s')
plt.axvline(optimal_lookback, color='red', linestyle='--', label=f'Optimal ({optimal_lookback})')
plt.xlabel('Lookback Period (days)')
plt.ylabel('Sharpe Ratio')
plt.title('Parameter Sensitivity: In-Sample vs Out-of-Sample')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()