import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Simulate order book with bid-ask spread
n_periods = 1000
tick_size = 0.01

# True value follows random walk
true_value = 100 + np.cumsum(np.random.normal(0, 0.1, n_periods))

# Spread components
order_processing_cost = 0.02  # Fixed cost per trade
inventory_risk_factor = 0.015  # Risk premium
adverse_selection_prob = 0.3  # 30% of trades are informed

# Generate quoted spreads
base_spread = 2 * (order_processing_cost + inventory_risk_factor)
volatility = np.abs(np.diff(true_value, prepend=true_value[0]))
volatility_smoothed = np.convolve(volatility, np.ones(20)/20, mode='same')

# Spread widens with volatility
quoted_spread = base_spread + 2 * volatility_smoothed
quoted_spread = np.maximum(quoted_spread, tick_size)  # Minimum tick size

# Generate bid and ask prices
half_spread = quoted_spread / 2
bid_prices = true_value - half_spread
ask_prices = true_value + half_spread

# Simulate trades
n_trades = 500
trade_times = np.sort(np.random.choice(n_periods, n_trades, replace=False))
trade_directions = np.random.choice([-1, 1], n_trades)  # -1=sell, 1=buy

# Determine if trade is informed
is_informed = np.random.random(n_trades) < adverse_selection_prob

execution_prices = np.zeros(n_trades)
midpoints = np.zeros(n_trades)
effective_spreads = np.zeros(n_trades)
realized_spreads = np.zeros(n_trades)

for i, t in enumerate(trade_times):
    midpoint = (bid_prices[t] + ask_prices[t]) / 2
    midpoints[i] = midpoint
    
    # Buy at ask, sell at bid (assuming market orders)
    if trade_directions[i] == 1:  # Buy
        execution_prices[i] = ask_prices[t]
        # Informed traders move price permanently
        if is_informed[i]:
            if t < n_periods - 1:
                true_value[t+1:] += 0.03  # Permanent impact
    else:  # Sell
        execution_prices[i] = bid_prices[t]
        if is_informed[i]:
            if t < n_periods - 1:
                true_value[t+1:] -= 0.03
    
    # Effective spread: 2 * |execution - midpoint|
    effective_spreads[i] = 2 * np.abs(execution_prices[i] - midpoint)
    
    # Realized spread: effective spread minus price change
    if t < n_periods - 5:
        future_midpoint = (bid_prices[t+5] + ask_prices[t+5]) / 2
        price_change = (future_midpoint - midpoint) * np.sign(trade_directions[i])
        realized_spreads[i] = effective_spreads[i] - 2 * price_change
    else:
        realized_spreads[i] = np.nan

# Calculate percentage spreads
percentage_spreads = (quoted_spread / true_value) * 100

# Roll estimator for effective spread (using trade price series)
def roll_estimator(prices):
    """Estimate spread from serial covariance of price changes (Roll 1984)"""
    price_changes = np.diff(prices)
    if len(price_changes) < 2:
        return np.nan
    cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
    if cov >= 0:
        return 0  # No negative covariance
    return 2 * np.sqrt(-cov)

roll_estimate = roll_estimator(execution_prices)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Bid-Ask spread over time with trades
sample_period = slice(100, 300)
axes[0, 0].plot(bid_prices[sample_period], 'b-', label='Bid', linewidth=1.5)
axes[0, 0].plot(ask_prices[sample_period], 'r-', label='Ask', linewidth=1.5)
axes[0, 0].plot(true_value[sample_period], 'g--', label='True Value', linewidth=1, alpha=0.7)

# Plot trades in this period
trades_in_period = (trade_times >= 100) & (trade_times < 300)
buy_trades = trades_in_period & (trade_directions == 1)
sell_trades = trades_in_period & (trade_directions == -1)

axes[0, 0].scatter(trade_times[buy_trades] - 100, execution_prices[buy_trades], 
                   c='green', marker='^', s=50, alpha=0.6, label='Buy')
axes[0, 0].scatter(trade_times[sell_trades] - 100, execution_prices[sell_trades], 
                   c='orange', marker='v', s=50, alpha=0.6, label='Sell')

axes[0, 0].set_title('Bid-Ask Spread Dynamics')
axes[0, 0].set_xlabel('Time Period')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Spread measures comparison
axes[0, 1].hist(quoted_spread[trade_times], bins=30, alpha=0.5, 
               label=f'Quoted (mean=${np.mean(quoted_spread[trade_times]):.4f})', density=True)
axes[0, 1].hist(effective_spreads, bins=30, alpha=0.5, 
               label=f'Effective (mean=${np.mean(effective_spreads):.4f})', density=True)
axes[0, 1].hist(realized_spreads[~np.isnan(realized_spreads)], bins=30, alpha=0.5, 
               label=f'Realized (mean=${np.nanmean(realized_spreads):.4f})', density=True)
axes[0, 1].axvline(roll_estimate, color='black', linestyle='--', 
                  label=f'Roll Estimate=${roll_estimate:.4f}')
axes[0, 1].set_title('Spread Measures Distribution')
axes[0, 1].set_xlabel('Spread ($)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

print("Spread Statistics:")
print(f"Mean Quoted Spread: ${np.mean(quoted_spread):.4f}")
print(f"Mean Effective Spread: ${np.mean(effective_spreads):.4f}")
print(f"Mean Realized Spread: ${np.nanmean(realized_spreads):.4f}")
print(f"Roll Estimator: ${roll_estimate:.4f}")
print(f"Adverse Selection Cost: ${np.mean(effective_spreads) - np.nanmean(realized_spreads):.4f}")

# Plot 3: Spread vs volatility relationship
window_size = 50
rolling_volatility = np.array([np.std(true_value[max(0, i-window_size):i+1]) 
                               for i in range(n_periods)])

axes[1, 0].scatter(rolling_volatility[trade_times], effective_spreads, 
                   alpha=0.3, s=20)
# Fit linear relationship
mask = ~np.isnan(effective_spreads)
slope, intercept, r_value, _, _ = stats.linregress(
    rolling_volatility[trade_times[mask]], effective_spreads[mask]
)
x_fit = np.linspace(rolling_volatility.min(), rolling_volatility.max(), 100)
axes[1, 0].plot(x_fit, slope * x_fit + intercept, 'r-', 
               label=f'R²={r_value**2:.3f}', linewidth=2)
axes[1, 0].set_title('Spread-Volatility Relationship')
axes[1, 0].set_xlabel('Rolling Volatility (50-period)')
axes[1, 0].set_ylabel('Effective Spread ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Percentage spread over time
axes[1, 1].plot(percentage_spreads, linewidth=1)
axes[1, 1].axhline(percentage_spreads.mean(), color='r', linestyle='--', 
                  label=f'Mean: {percentage_spreads.mean():.3f}%')
axes[1, 1].fill_between(range(n_periods), 
                        percentage_spreads.mean() - percentage_spreads.std(),
                        percentage_spreads.mean() + percentage_spreads.std(),
                        alpha=0.2, label='±1 SD')
axes[1, 1].set_title('Percentage Spread Time Series')
axes[1, 1].set_xlabel('Time Period')
axes[1, 1].set_ylabel('Spread (%)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate bid-ask bounce (Roll 1984)
# Transaction prices bounce between bid and ask creating negative autocorrelation
transaction_returns = np.diff(execution_prices) / execution_prices[:-1]
acf_lag1 = np.corrcoef(transaction_returns[:-1], transaction_returns[1:])[0, 1]

print(f"\nBid-Ask Bounce Analysis:")
print(f"Transaction return autocorrelation (lag 1): {acf_lag1:.4f}")
print(f"Expected negative autocorrelation from spread: {-(quoted_spread.mean()/(4*true_value.mean()))**2:.4f}")

# Decompose spread components
informed_trades_mask = is_informed
uninformed_trades_mask = ~is_informed

effective_informed = effective_spreads[informed_trades_mask].mean()
effective_uninformed = effective_spreads[uninformed_trades_mask].mean()
realized_informed = np.nanmean(realized_spreads[informed_trades_mask])
realized_uninformed = np.nanmean(realized_spreads[uninformed_trades_mask])

print(f"\nSpread Decomposition:")
print(f"Effective spread (informed trades): ${effective_informed:.4f}")
print(f"Effective spread (uninformed trades): ${effective_uninformed:.4f}")
print(f"Realized spread (informed trades): ${realized_informed:.4f}")
print(f"Realized spread (uninformed trades): ${realized_uninformed:.4f}")
print(f"Adverse selection component (informed): ${effective_informed - realized_informed:.4f}")
