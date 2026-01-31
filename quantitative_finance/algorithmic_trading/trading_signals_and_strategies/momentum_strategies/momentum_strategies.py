import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Simulated price data (would use real OHLC in practice)
np.random.seed(42)
n_days = 2000
dates = pd.date_range('2015-01-01', periods=n_days, freq='D')

# 5 stocks with different momentum profiles
price_data = {}
for stock in ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']:
    # Generate price with trending behavior
    drift = np.random.uniform(0.0002, 0.0008)  # Positive drift (trending)
    volatility = np.random.uniform(0.015, 0.025)
    returns = np.random.normal(drift, volatility, n_days)
    price = 100 * np.exp(np.cumsum(returns))
    price_data[stock] = pd.Series(price, index=dates)

prices_df = pd.DataFrame(price_data)

print("="*100)
print("MOMENTUM STRATEGY BACKTEST: Multi-Scale Signals")
print("="*100)
print(f"\nPrice Data (first 10 days):")
print(prices_df.head(10))

# Step 1: Calculate momentum signals (multiple lookback windows)
print(f"\n" + "="*100)
print("STEP 1: Momentum Signal Calculation")
print("="*100)

lookback_windows = {1: 20, 3: 60, 6: 126, 12: 252}  # months: days
momentum_signals = {}

for window_name, window_days in lookback_windows.items():
    momentum = prices_df.pct_change(window_days)
    momentum_signals[f'{window_name}M'] = momentum
    print(f"\n{window_name}-Month Momentum (lookback {window_days} days):")
    print(f"  Mean: {momentum.mean().mean():.4f}, Std: {momentum.std().mean():.4f}")

# Combined momentum signal (average of normalized scores)
def normalize_signal(signal):
    """Z-score normalization per stock."""
    return signal.rolling(252).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0)

combined_signal = pd.DataFrame(index=dates)
for window_name in momentum_signals.keys():
    combined_signal[f'signal_{window_name}'] = normalize_signal(momentum_signals[window_name])

# Final combined signal (average across windows)
combined_signal['combined'] = combined_signal.mean(axis=1)

print(f"\nCombined Signal (average of normalized scores):")
print(f"  Min: {combined_signal['combined'].min():.2f}")
print(f"  Max: {combined_signal['combined'].max():.2f}")
print(f"  Mean: {combined_signal['combined'].mean():.4f}")

# Step 2: Generate trading signals
print(f"\n" + "="*100)
print("STEP 2: Trading Signal Generation")
print("="*100)

threshold = 0.5  # Z-score threshold for long/short
position = combined_signal['combined'].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))

print(f"\nSignal Distribution:")
print(f"  Long (position = 1): {(position == 1).sum()} days ({(position == 1).sum() / len(position) * 100:.1f}%)")
print(f"  Neutral (position = 0): {(position == 0).sum()} days ({(position == 0).sum() / len(position) * 100:.1f}%)")
print(f"  Short (position = -1): {(position == -1).sum()} days ({(position == -1).sum() / len(position) * 100:.1f}%)")

# Step 3: Portfolio construction (long-short, equal-weighted)
print(f"\n" + "="*100)
print("STEP 3: Portfolio Construction (Long-Short)")
print("="*100)

# For simplicity: assume equal allocation to top/bottom momentum stocks each day
portfolio_value = []
daily_returns = []

for day in range(1, len(prices_df)):
    # Get current momentum signals
    current_momentum = combined_signal['combined'].iloc[day]
    
    if pd.isna(current_momentum):
        portfolio_value.append(100 if day == 1 else portfolio_value[-1])
        daily_returns.append(0)
        continue
    
    # Get price returns for all stocks
    price_returns = prices_df.pct_change().iloc[day]
    
    # Long top momentum stock, short bottom (simplified: top 1, bottom 1)
    stocks_ranked = current_momentum.nlargest(2).index.tolist()
    
    if len(stocks_ranked) >= 2:
        long_stock = stocks_ranked[0]
        short_stock = stocks_ranked[-1]
        
        # Portfolio: +1 long, -1 short (market-neutral)
        portfolio_return = 0.5 * price_returns[long_stock] - 0.5 * price_returns[short_stock]
        daily_returns.append(portfolio_return)
    else:
        daily_returns.append(0)
    
    # Accumulate portfolio value
    if day == 1:
        portfolio_value.append(100 * (1 + daily_returns[-1]))
    else:
        portfolio_value.append(portfolio_value[-1] * (1 + daily_returns[-1]))

portfolio_series = pd.Series(portfolio_value, index=dates[1:])
daily_returns_series = pd.Series(daily_returns, index=dates[1:])

# Step 4: Performance metrics
print(f"\nStrategy Performance:")
print(f"  Starting Value: $100")
print(f"  Ending Value: ${portfolio_value[-1]:.2f}")
print(f"  Total Return: {(portfolio_value[-1] / 100 - 1) * 100:.2f}%")
print(f"  Annualized Return: {(np.mean(daily_returns) * 252) * 100:.2f}%")
print(f"  Annualized Volatility: {np.std(daily_returns) * np.sqrt(252) * 100:.2f}%")
print(f"  Sharpe Ratio: {(np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252)):.2f}")

# Maximum drawdown
cumulative = np.cumprod(1 + daily_returns)
running_max = np.maximum.accumulate(cumulative)
drawdown = (cumulative - running_max) / running_max
max_drawdown = np.min(drawdown)
print(f"  Maximum Drawdown: {max_drawdown * 100:.2f}%")

# Win rate
win_rate = (np.array(daily_returns) > 0).sum() / len(daily_returns)
print(f"  Win Rate: {win_rate * 100:.1f}%")

# Comparison to buy-and-hold (equal-weight all stocks)
buy_hold_returns = prices_df.pct_change().mean(axis=1)
buy_hold_value = 100 * np.cumprod(1 + buy_hold_returns)

print(f"\nBuy-and-Hold (Equal-Weight Comparison):")
print(f"  Ending Value: ${buy_hold_value.iloc[-1]:.2f}")
print(f"  Total Return: {(buy_hold_value.iloc[-1] / 100 - 1) * 100:.2f}%")
print(f"  Annualized Return: {(np.mean(buy_hold_returns) * 252) * 100:.2f}%")

# Step 5: Momentum decay analysis
print(f"\n" + "="*100)
print("STEP 5: Momentum Decay Analysis")
print("="*100)

# Measure returns in periods following signal generation
decay_analysis = []
for signal_window in ['1M', '3M', '6M', '12M']:
    signal_col = f'signal_{signal_window}'
    
    # Calculate returns in next N periods after high/low momentum signal
    holding_periods = []
    for day in range(100, len(prices_df) - 20):
        if pd.notna(combined_signal[signal_col].iloc[day]):
            signal_val = combined_signal[signal_col].iloc[day]
            if signal_val > 1.0:  # High momentum
                # Calculate next 20-day return
                return_20d = (prices_df.iloc[day + 20] / prices_df.iloc[day] - 1).mean()
                holding_periods.append({'signal': 'strong_long', 'return': return_20d})
            elif signal_val < -1.0:  # Low momentum
                return_20d = (prices_df.iloc[day + 20] / prices_df.iloc[day] - 1).mean()
                holding_periods.append({'signal': 'strong_short', 'return': return_20d})
    
    if holding_periods:
        df_hold = pd.DataFrame(holding_periods)
        long_return = df_hold[df_hold['signal'] == 'strong_long']['return'].mean()
        short_return = df_hold[df_hold['signal'] == 'strong_short']['return'].mean()
        decay_analysis.append({'window': signal_window, 'long_20d_ret': long_return, 'short_20d_ret': short_return})

decay_df = pd.DataFrame(decay_analysis)
print(f"\nMomentum Decay (20-day forward returns after signal):")
print(decay_df.to_string(index=False))

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Strategy vs Buy-and-Hold
ax = axes[0, 0]
ax.plot(dates[1:], portfolio_series, label='Momentum Strategy', linewidth=2)
ax.plot(dates[1:], buy_hold_value.iloc[1:], label='Buy-and-Hold (EW)', linewidth=2, alpha=0.7)
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('Momentum Strategy vs Buy-and-Hold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_yscale('log')

# Plot 2: Daily Returns Distribution
ax = axes[0, 1]
ax.hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(daily_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(daily_returns)*100:.3f}%')
ax.set_xlabel('Daily Return')
ax.set_ylabel('Frequency')
ax.set_title('Daily Returns Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Cumulative Drawdown
ax = axes[1, 0]
cumulative_ret = np.cumprod(1 + daily_returns) - 1
running_max_ret = np.maximum.accumulate(cumulative_ret)
drawdown_series = (cumulative_ret - running_max_ret) / (1 + running_max_ret)
ax.fill_between(dates[1:], 0, drawdown_series * 100, alpha=0.5, color='red')
ax.set_ylabel('Drawdown (%)')
ax.set_title('Drawdown Over Time')
ax.grid(alpha=0.3)

# Plot 4: Momentum Signal Evolution
ax = axes[1, 1]
ax.plot(dates, combined_signal['combined'], label='Combined Signal', linewidth=1)
ax.axhline(y=threshold, color='green', linestyle='--', alpha=0.5, label=f'Long Threshold ({threshold})')
ax.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5, label=f'Short Threshold (-{threshold})')
ax.fill_between(dates, threshold, 5, alpha=0.2, color='green')
ax.fill_between(dates, -5, -threshold, alpha=0.2, color='red')
ax.set_ylabel('Signal (Z-score)')
ax.set_title('Momentum Signal Over Time (Combined Multi-Scale)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Statistics table
print(f"\n" + "="*100)
print("PERFORMANCE SUMMARY TABLE")
print("="*100)

summary_df = pd.DataFrame({
    'Metric': ['Total Return', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
    'Momentum': [
        f"{(portfolio_value[-1]/100 - 1)*100:.2f}%",
        f"{np.mean(daily_returns)*252*100:.2f}%",
        f"{np.std(daily_returns)*np.sqrt(252)*100:.2f}%",
        f"{(np.mean(daily_returns)*252)/(np.std(daily_returns)*np.sqrt(252)):.2f}",
        f"{max_drawdown*100:.2f}%",
        f"{win_rate*100:.1f}%"
    ],
    'Buy-and-Hold': [
        f"{(buy_hold_value.iloc[-1]/100 - 1)*100:.2f}%",
        f"{np.mean(buy_hold_returns)*252*100:.2f}%",
        f"{np.std(buy_hold_returns)*np.sqrt(252)*100:.2f}%",
        f"{(np.mean(buy_hold_returns)*252)/(np.std(buy_hold_returns)*np.sqrt(252)):.2f}",
        "N/A",
        "N/A"
    ]
})

print(summary_df.to_string(index=False))