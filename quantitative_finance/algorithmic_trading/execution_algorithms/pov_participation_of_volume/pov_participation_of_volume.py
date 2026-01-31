import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate intraday volume with random variations
np.random.seed(42)
minutes = np.arange(1, 391)
hours = 9.5 + minutes / 60

# Base intraday pattern (U-shape)
morning_ramp = 1 + 0.4 * np.sin(np.pi * minutes / 100)
lunch_dip = 0.7
afternoon_ramp = 1 + 0.3 * np.sin(np.pi * (minutes - 210) / 180)

vol_multiplier = np.ones(390)
vol_multiplier[:90] = morning_ramp[:90]
vol_multiplier[90:210] = lunch_dip
vol_multiplier[210:] = afternoon_ramp[210:]

# Add random fluctuations
base_vol = 1000
volume_daily = (np.random.poisson(base_vol, 390) * vol_multiplier).astype(int)
total_daily_vol = volume_daily.sum()

# Prices: random walk with drift
price_base = 100
price_changes = np.random.normal(0.0001, 0.05, 390)
prices = price_base + np.cumsum(price_changes)

# Create DataFrame
market_data = pd.DataFrame({
    'minute': minutes,
    'hour': hours,
    'volume': volume_daily,
    'price': prices,
    'cum_volume': volume_daily.cumsum(),
})

# Calculate VWAP benchmark
market_data['cum_dollar_vol'] = (market_data['price'] * market_data['volume']).cumsum()
market_data['vwap'] = market_data['cum_dollar_vol'] / market_data['cum_volume']

# POV Execution with Different Participation Rates
target_order = 60000

# Test different POV percentages
pov_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
execution_results = {}

for pov_rate in pov_rates:
    # Allocate based on POV rate
    target_qty_per_min = (market_data['volume'] * pov_rate).values
    target_qty_per_min = target_qty_per_min.astype(int)
    
    # Adjust last bar to hit total target
    cumsum = np.cumsum(target_qty_per_min)
    if cumsum[-1] < target_order:
        target_qty_per_min[-1] += (target_order - cumsum[-1])
    elif cumsum[-1] > target_order:
        # Proportionally reduce all
        target_qty_per_min = (target_qty_per_min / cumsum[-1] * target_order).astype(int)
        target_qty_per_min[-1] += (target_order - target_qty_per_min.sum())
    
    # Simulate execution
    filled_qty = target_qty_per_min.copy()
    filled_price = prices.copy()
    
    execution_cost = (filled_qty * filled_price).sum()
    avg_exec_price = execution_cost / filled_qty.sum()
    
    execution_results[f'POV_{int(pov_rate*100)}%'] = {
        'avg_exec_price': avg_exec_price,
        'cost_vs_vwap': avg_exec_price - market_data['vwap'].iloc[-1],
        'cost_bps': (avg_exec_price - market_data['vwap'].iloc[-1]) / market_data['vwap'].iloc[-1] * 10000,
        'filled_qty': filled_qty.sum(),
        'cumsum_allocation': np.cumsum(filled_qty)
    }

# Comparison table
print("="*70)
print("POV EXECUTION COMPARISON")
print("="*70)
print(f"{'POV Rate':<12} {'Avg Price':<12} {'vs VWAP':<12} {'Cost (bps)':<12}")
print("-"*70)

for pov_label, results in execution_results.items():
    pov_pct = pov_label.replace('POV_', '').replace('%', '')
    print(f"{pov_label:<12} ${results['avg_exec_price']:<11.4f} "
          f"${results['cost_vs_vwap']:<11.4f} {results['cost_bps']:<11.2f}")

print(f"\nBenchmark VWAP: ${market_data['vwap'].iloc[-1]:.4f}")
print(f"Total Daily Volume: {total_daily_vol:,}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Volume Profile
ax = axes[0, 0]
ax.bar(hours, volume_daily, width=0.015, alpha=0.6, color='steelblue')
ax.set_title('Market Volume Profile')
ax.set_xlabel('Time')
ax.set_ylabel('Volume (shares)')
ax.grid(alpha=0.3, axis='y')

# Plot 2: Price and VWAP
ax = axes[0, 1]
ax.plot(hours, prices, 'b-', linewidth=1.5, label='Price')
ax.plot(hours, market_data['vwap'], 'r--', linewidth=2, label='VWAP Benchmark')
ax.set_title('Intraday Price vs VWAP')
ax.set_xlabel('Time')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: POV Allocations (cumulative)
ax = axes[1, 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(pov_rates)))
for i, (pov_label, results) in enumerate(execution_results.items()):
    ax.plot(hours, results['cumsum_allocation'], linewidth=2, 
           label=pov_label, color=colors[i])
ax.set_title('Cumulative Execution Progress by POV Rate')
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Shares Filled')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Execution Cost Comparison
ax = axes[1, 1]
pov_labels = list(execution_results.keys())
costs_bps = [execution_results[label]['cost_bps'] for label in pov_labels]
colors_bar = ['green' if c < 0 else 'red' for c in costs_bps]
ax.bar(range(len(pov_labels)), costs_bps, color=colors_bar, alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xticks(range(len(pov_labels)))
ax.set_xticklabels(pov_labels, rotation=45)
ax.set_ylabel('Execution Cost vs VWAP (bps)')
ax.set_title('Execution Performance by POV Rate')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\nKey Insights:")
print("- Lower POV%: Slower execution, lower market impact, longer time at risk")
print("- Higher POV%: Faster execution, higher market impact, shorter time window")
print("- Optimal POV depends on: Volatility, urgency, liquidity, information risk")