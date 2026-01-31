import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Market Efficiency Testing
def generate_returns(n_periods=1000, model='random_walk', momentum_strength=0):
    """
    Generate return series under different efficiency assumptions
    
    model: 'random_walk', 'momentum', 'mean_reversion'
    """
    returns = np.zeros(n_periods)
    
    if model == 'random_walk':
        # Efficient market: i.i.d. returns
        returns = np.random.normal(0.0005, 0.02, n_periods)
    
    elif model == 'momentum':
        # Momentum: Positive autocorrelation
        returns[0] = np.random.normal(0.0005, 0.02)
        for t in range(1, n_periods):
            returns[t] = momentum_strength * returns[t-1] + \
                        np.random.normal(0.0005, 0.02)
    
    elif model == 'mean_reversion':
        # Mean reversion: Negative autocorrelation
        returns[0] = np.random.normal(0.0005, 0.02)
        for t in range(1, n_periods):
            returns[t] = -momentum_strength * returns[t-1] + \
                        np.random.normal(0.0005, 0.02)
    
    return returns

def variance_ratio_test(returns, k):
    """
    Variance ratio test: VR(k) = Var(k-period return) / (k * Var(1-period))
    Efficient market: VR = 1
    """
    n = len(returns)
    
    # k-period returns (non-overlapping)
    k_returns = []
    for i in range(0, n - k + 1, k):
        k_ret = np.sum(returns[i:i+k])
        k_returns.append(k_ret)
    
    if len(k_returns) < 2:
        return np.nan
    
    var_k = np.var(k_returns, ddof=1)
    var_1 = np.var(returns, ddof=1)
    
    if var_1 > 0:
        vr = var_k / (k * var_1)
    else:
        vr = np.nan
    
    return vr

def autocorrelation_test(returns, max_lag=20):
    """Calculate autocorrelations"""
    n = len(returns)
    mean_r = np.mean(returns)
    var_r = np.var(returns)
    
    autocorrs = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        
        cov = np.mean((returns[lag:] - mean_r) * (returns[:-lag] - mean_r))
        
        if var_r > 0:
            autocorrs.append(cov / var_r)
        else:
            autocorrs.append(0)
    
    return autocorrs

# Simulate different market types
print("Market Efficiency Tests")
print("=" * 70)

n_periods = 1000

# Three scenarios
scenarios = {
    'Efficient (Random Walk)': generate_returns(n_periods, 'random_walk', 0),
    'Momentum (Inefficient)': generate_returns(n_periods, 'momentum', 0.15),
    'Mean Reversion': generate_returns(n_periods, 'mean_reversion', 0.15)
}

# Test each scenario
results = {}

for name, returns in scenarios.items():
    print(f"\n{name}:")
    print("-" * 70)
    
    # Summary statistics
    print(f"Mean Return: {returns.mean()*252:.2f}% annual")
    print(f"Volatility: {returns.std()*np.sqrt(252):.2f}% annual")
    print(f"Sharpe Ratio: {(returns.mean() / returns.std()) * np.sqrt(252):.2f}")
    
    # Autocorrelation
    autocorrs = autocorrelation_test(returns, max_lag=20)
    print(f"\nAutocorrelation lag-1: {autocorrs[0]:.4f}")
    
    # Test significance
    if abs(autocorrs[0]) > 1.96 / np.sqrt(n_periods):
        print(f"  → Significantly different from 0 (inefficient)")
    else:
        print(f"  → Not significant (consistent with efficiency)")
    
    # Variance ratio tests
    print(f"\nVariance Ratio Tests:")
    for k in [2, 5, 10]:
        vr = variance_ratio_test(returns, k)
        print(f"  VR({k}) = {vr:.3f}", end="")
        
        if abs(vr - 1.0) < 0.1:
            print(" [Random Walk]")
        elif vr > 1.0:
            print(" [Momentum/Slow Adjustment]")
        else:
            print(" [Mean Reversion]")
    
    results[name] = {
        'returns': returns,
        'autocorrs': autocorrs
    }

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Cumulative returns
for name, data in results.items():
    returns = data['returns']
    cum_returns = np.cumprod(1 + returns) - 1
    axes[0, 0].plot(cum_returns * 100, label=name, linewidth=2, alpha=0.7)

axes[0, 0].set_xlabel('Trading Days')
axes[0, 0].set_ylabel('Cumulative Return (%)')
axes[0, 0].set_title('Price Paths Under Different Efficiency Assumptions')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)

# Plot 2: Autocorrelation functions
lags = range(1, 21)
for name, data in results.items():
    autocorrs = data['autocorrs']
    axes[0, 1].plot(lags, autocorrs, marker='o', label=name, linewidth=2, alpha=0.7)

# 95% confidence bands
axes[0, 1].axhline(1.96 / np.sqrt(n_periods), color='red', linestyle='--', 
                  linewidth=1, alpha=0.5, label='95% CI')
axes[0, 1].axhline(-1.96 / np.sqrt(n_periods), color='red', linestyle='--', 
                  linewidth=1, alpha=0.5)
axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[0, 1].set_xlabel('Lag (days)')
axes[0, 1].set_ylabel('Autocorrelation')
axes[0, 1].set_title('Autocorrelation Function (ACF)')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Return distributions
for name, data in results.items():
    returns = data['returns']
    axes[1, 0].hist(returns * 100, bins=50, alpha=0.5, label=name, edgecolor='black')

axes[1, 0].set_xlabel('Daily Return (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Return Distributions')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Variance ratios
vr_periods = [2, 3, 5, 10, 20]
for name, data in results.items():
    returns = data['returns']
    vrs = [variance_ratio_test(returns, k) for k in vr_periods]
    axes[1, 1].plot(vr_periods, vrs, marker='o', label=name, linewidth=2, alpha=0.7)

axes[1, 1].axhline(1.0, color='red', linestyle='--', linewidth=2, 
                  label='Random Walk (VR=1)')
axes[1, 1].set_xlabel('Period Length (k)')
axes[1, 1].set_ylabel('Variance Ratio VR(k)')
axes[1, 1].set_title('Variance Ratio Test')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Momentum trading strategy test
print(f"\n\nMomentum Trading Strategy (6-month)")
print("=" * 70)

# Test on momentum scenario
returns_mom = scenarios['Momentum (Inefficient)']
lookback = 120  # 6 months

strategy_returns = []
positions = []

for t in range(lookback, len(returns_mom)):
    # Calculate past 6-month return
    past_return = np.sum(returns_mom[t-lookback:t])
    
    # Signal: 1 if past positive, -1 if negative
    if past_return > 0:
        position = 1  # Long
    else:
        position = -1  # Short
    
    # Strategy return
    strategy_return = position * returns_mom[t]
    
    strategy_returns.append(strategy_return)
    positions.append(position)

strategy_returns = np.array(strategy_returns)
market_returns = returns_mom[lookback:]

print(f"Market Buy-and-Hold:")
print(f"  Mean Return: {market_returns.mean()*252:.2f}% annual")
print(f"  Volatility: {market_returns.std()*np.sqrt(252):.2f}%")
print(f"  Sharpe: {(market_returns.mean()/market_returns.std())*np.sqrt(252):.2f}")

print(f"\nMomentum Strategy:")
print(f"  Mean Return: {strategy_returns.mean()*252:.2f}% annual")
print(f"  Volatility: {strategy_returns.std()*np.sqrt(252):.2f}%")
print(f"  Sharpe: {(strategy_returns.mean()/strategy_returns.std())*np.sqrt(252):.2f}")

# Statistical test
excess_returns = strategy_returns - market_returns
t_stat, p_value = stats.ttest_1samp(excess_returns, 0)

print(f"\nExcess Return Test:")
print(f"  Mean Excess: {excess_returns.mean()*252:.2f}% annual")
print(f"  t-statistic: {t_stat:.2f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  → Strategy significantly outperforms (market inefficient)")
else:
    print("  → No significant outperformance (consistent with efficiency)")

# Cumulative performance
cum_market = np.cumprod(1 + market_returns) - 1
cum_strategy = np.cumprod(1 + strategy_returns) - 1

print(f"\nCumulative Performance:")
print(f"  Market: {cum_market[-1]*100:.1f}%")
print(f"  Strategy: {cum_strategy[-1]*100:.1f}%")
print(f"  Outperformance: {(cum_strategy[-1] - cum_market[-1])*100:.1f}%")
