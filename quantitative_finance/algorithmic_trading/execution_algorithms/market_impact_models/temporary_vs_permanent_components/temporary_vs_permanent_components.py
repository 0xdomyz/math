import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Generate synthetic trade execution data
np.random.seed(42)
n_trades = 500

# Trade parameters
trade_size_pct = np.random.uniform(0.05, 2.0, n_trades)  # % of daily volume
trade_direction = np.random.choice([-1, 1], n_trades)  # -1 = sell, +1 = buy
market_vol = np.random.uniform(0.8, 1.5, n_trades)  # Vol multiplier (normal=1)

# Price movements (bid-ask adjusted)
# Temporary: Spread + inventory cost (decays quickly)
# Permanent: Information (doesn't decay)

# Model: I_total = temp_component + permanent_component
# temp_component = 2 + 0.5 * size + 0.1 * vol (bid-ask + inventory)
# perm_component = 0.1 * size + 0.05 * vol (info-based)

temp_baseline = 2.0  # Bid-ask spread (bps)
temp_slope = 0.5  # Inventory cost per % volume
temp_vol_sensitivity = 0.1

perm_baseline = 0.1
perm_slope = 0.1  # Information content per % volume
perm_vol_sensitivity = 0.05

temp_impact_true = temp_baseline + temp_slope * trade_size_pct + temp_vol_sensitivity * market_vol
perm_impact_true = perm_baseline * trade_size_pct + perm_vol_sensitivity * market_vol

# Add noise
temp_noise = np.random.normal(0, 0.3, n_trades)
perm_noise = np.random.normal(0, 0.1, n_trades)

temp_impact_obs = np.maximum(temp_impact_true + temp_noise, 0.5)
perm_impact_obs = np.maximum(perm_impact_true + perm_noise, 0)

total_impact_obs = temp_impact_obs + perm_impact_obs

# Post-trade price data (simulated)
# Price reverts partially from temp; doesn't revert from perm
reversion_windows = [1, 5, 30]  # minutes
reversion_data = []

for window in reversion_windows:
    # Reversion factor: How much of temporary impact reverts in this window
    # Assume exponential decay: revert_pct = 1 - exp(-t / tau), tau = 10 min
    tau = 10  # Minutes for half-life-ish decay
    revert_pct = 1 - np.exp(-window / tau)
    
    # Observed price change (accounting for partial reversion)
    remaining_temp = temp_impact_obs * (1 - revert_pct)
    observed_price_change = remaining_temp + perm_impact_obs  # Temp partially reverted; perm stays
    
    reversion_data.append({
        'window_min': window,
        'revert_pct': revert_pct,
        'price_change': observed_price_change,
    })

# Create DataFrame
df = pd.DataFrame({
    'trade_size_pct': trade_size_pct,
    'market_vol': market_vol,
    'total_impact': total_impact_obs,
})

# Add post-trade price data
for rv in reversion_data:
    df[f'price_change_{rv["window_min"]}min'] = rv['price_change']

print(f"Trade data: {len(df)} trades\n")
print(f"Impact statistics:\n{df[['trade_size_pct', 'total_impact']].describe()}\n")

# Estimation Method 1: Linear regression on total impact
# Total = temp_base + temp_slope*size + perm_slope*size + vol_effect
# Try to disentangle by looking at reversion patterns

# Method 1: Use 1-min vs 30-min price changes
# Assumption: 1-min captures more temporary; 30-min is mostly permanent
price_1min = df['price_change_1min']
price_30min = df['price_change_30min']

# Regression: price_1min = α + β*size
slope_1min, intercept_1min, r2_1min, _, _ = linregress(df['trade_size_pct'], price_1min)

# Regression: price_30min = γ + δ*size
slope_30min, intercept_30min, r2_30min, _, _ = linregress(df['trade_size_pct'], price_30min)

print("ESTIMATION: Temporary vs Permanent (Regression on Size):")
print(f"1-min price change: Intercept={intercept_1min:.3f}, Slope={slope_1min:.3f}, R²={r2_1min:.3f}")
print(f"30-min price change: Intercept={intercept_30min:.3f}, Slope={slope_30min:.3f}, R²={r2_30min:.3f}")

# Estimate components (rough approximation)
est_temp_baseline = intercept_1min - intercept_30min  # What reverts in 1-30 min
est_temp_slope = slope_1min - slope_30min
est_perm_baseline = intercept_30min
est_perm_slope = slope_30min

print(f"\nEstimated Temporary Component:")
print(f"  Baseline: {est_temp_baseline:.3f} bps, Slope: {est_temp_slope:.3f} bps per % vol")
print(f"Estimated Permanent Component:")
print(f"  Baseline: {est_perm_baseline:.3f} bps, Slope: {est_perm_slope:.3f} bps per % vol")

print(f"\nTrue (Simulated) Parameters:")
print(f"  Temporary: {temp_baseline:.3f} + {temp_slope:.3f}*size")
print(f"  Permanent: {perm_baseline:.3f}*size + {perm_vol_sensitivity:.3f}*vol")

# Decomposition analysis
df['est_temp'] = est_temp_baseline + est_temp_slope * df['trade_size_pct']
df['est_perm'] = est_perm_baseline + est_perm_slope * df['trade_size_pct']
df['est_total'] = df['est_temp'] + df['est_perm']
df['perm_ratio'] = df['est_perm'] / df['est_total']

print(f"\nDecomposition of Total Impact:")
print(f"  Average temporary: {df['est_temp'].mean():.2f} bps ({df['est_temp'].mean()/df['est_total'].mean()*100:.0f}%)")
print(f"  Average permanent: {df['est_perm'].mean():.2f} bps ({df['est_perm'].mean()/df['est_total'].mean()*100:.0f}%)")
print(f"  Average total: {df['est_total'].mean():.2f} bps")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Total impact vs trade size with reversion windows
ax = axes[0, 0]
ax.scatter(df['trade_size_pct'], df['total_impact'], alpha=0.5, s=30, label='Total (t=0)', color='black')
ax.scatter(df['trade_size_pct'], df['price_change_1min'], alpha=0.5, s=30, label='1-min later', color='blue')
ax.scatter(df['trade_size_pct'], df['price_change_30min'], alpha=0.5, s=30, label='30-min later', color='green')
# Add trend lines
size_range = np.linspace(df['trade_size_pct'].min(), df['trade_size_pct'].max(), 100)
ax.plot(size_range, intercept_1min + slope_1min * size_range, '--', linewidth=2, color='blue')
ax.plot(size_range, intercept_30min + slope_30min * size_range, '--', linewidth=2, color='green')
ax.set_xlabel('Trade Size (% of daily volume)')
ax.set_ylabel('Price Change (bps)')
ax.set_title('Post-Trade Price Reversion Over Time')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Estimated temporary vs permanent by size
ax = axes[0, 1]
ax.scatter(df['trade_size_pct'], df['est_temp'], alpha=0.6, s=30, label='Temporary', color='orange')
ax.scatter(df['trade_size_pct'], df['est_perm'], alpha=0.6, s=30, label='Permanent', color='red')
ax.scatter(df['trade_size_pct'], df['est_total'], alpha=0.6, s=30, label='Total', color='black')
ax.set_xlabel('Trade Size (% of daily volume)')
ax.set_ylabel('Impact (bps)')
ax.set_title('Decomposed Impact Components')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Proportion permanent vs trade size
ax = axes[1, 0]
ax.scatter(df['trade_size_pct'], df['perm_ratio'] * 100, alpha=0.6, s=30, color='purple')
ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Trade Size (% of daily volume)')
ax.set_ylabel('Permanent Impact (% of total)')
ax.set_title('Permanent Impact Ratio vs Trade Size')
ax.grid(alpha=0.3)

# Plot 4: Reversion trajectory (average)
ax = axes[1, 1]
windows = [0, 1, 5, 30]
avg_prices = [df['total_impact'].mean()] + [df[f'price_change_{w}min'].mean() for w in [1, 5, 30]]
ax.plot(windows, avg_prices, 'o-', linewidth=2, markersize=8, color='steelblue')
# Add permanent component line (no further reversion after ~30 min)
perm_level = df['est_perm'].mean()
ax.axhline(perm_level, color='red', linestyle='--', linewidth=2, label=f'Estimated Permanent (~{perm_level:.2f} bps)')
ax.set_xlabel('Time Since Trade (minutes)')
ax.set_ylabel('Observed Price Impact (bps)')
ax.set_title('Price Reversion Trajectory (Average Trade)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n\nSUMMARY: How Much Impact is Recoverable?")
temp_fraction = df['est_temp'].mean() / df['est_total'].mean()
perm_fraction = 1 - temp_fraction
print(f"Temporary (recoverable via patience): {temp_fraction*100:.0f}%")
print(f"Permanent (unavoidable): {perm_fraction*100:.0f}%")
print(f"⟹ Optimal strategy: Reduce temporary via patient execution; accept permanent as cost")