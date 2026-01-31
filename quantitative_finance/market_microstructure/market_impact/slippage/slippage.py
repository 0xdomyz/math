import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# Simulate realistic order execution data with multiple benchmarks
np.random.seed(42)
n_trades = 500

# Generate trade characteristics
trades = pd.DataFrame({
    'trade_id': range(n_trades),
    'stock': np.random.choice(['AAPL', 'MSFT', 'GOOG', 'AMZN'], n_trades),
    'order_size_pct_vol': np.random.lognormal(-1, 1.5, n_trades),  # % of daily volume
    'order_type': np.random.choice(['market', 'limit', 'algo_vwap', 'algo_twap'], n_trades),
    'time_of_day': np.random.choice(['open', 'midday', 'close'], n_trades),
})

# Generate price data (simulated)
trades['arrival_price'] = 100 + np.random.normal(0, 5, n_trades)
trades['vwap'] = trades['arrival_price'] + np.random.normal(0, 0.5, n_trades)
trades['twap'] = trades['arrival_price'] + np.random.normal(0, 0.7, n_trades)

# Generate execution prices (depends on order type and market condition)
spread_by_stock = {'AAPL': 0.02, 'MSFT': 0.03, 'GOOG': 0.04, 'AMZN': 0.025}
impact_by_size = trades['order_size_pct_vol'].apply(lambda x: 0.0002 * np.sqrt(x))

execution_prices = []
for idx, row in trades.iterrows():
    spread = spread_by_stock[row['stock']]
    impact = impact_by_size.iloc[idx]
    
    # Order type effect
    if row['order_type'] == 'market':
        # Market orders: Pay spread + impact
        exec_price = row['arrival_price'] + 0.5 * spread + impact
    elif row['order_type'] == 'limit':
        # Limit orders: May miss (50% fill rate assumed), but get better price
        exec_price = row['arrival_price'] - 0.5 * spread + 0.3 * impact
    elif row['order_type'] == 'algo_vwap':
        # VWAP algo: Tracks VWAP, half spread cost
        exec_price = row['vwap'] + 0.5 * 0.5 * spread + 0.5 * impact
    else:  # algo_twap
        # TWAP algo: Simple time-weighted, pays spread
        exec_price = row['twap'] + 0.7 * 0.5 * spread + 0.7 * impact
    
    execution_prices.append(exec_price)

trades['execution_price'] = execution_prices

# Calculate slippage vs different benchmarks
trades['slippage_vs_arrival'] = (trades['execution_price'] - trades['arrival_price']) / trades['arrival_price']
trades['slippage_vs_vwap'] = (trades['execution_price'] - trades['vwap']) / trades['vwap']
trades['slippage_vs_twap'] = (trades['execution_price'] - trades['twap']) / trades['twap']

# Convert to basis points
trades['slippage_ap_bps'] = trades['slippage_vs_arrival'] * 10000
trades['slippage_vwap_bps'] = trades['slippage_vs_vwap'] * 10000
trades['slippage_twap_bps'] = trades['slippage_vs_twap'] * 10000

# Estimate cost components
trades['spread_cost'] = spread_by_stock[trades['stock'].iloc[0]] / (2 * trades['arrival_price'])
trades['impact_cost'] = impact_by_size
trades['commission_cost'] = 0.0005  # 0.5 bps

print("="*100)
print("SLIPPAGE ANALYSIS")
print("="*100)

print(f"\nStep 1: Summary Statistics")
print(f"-" * 50)
print(f"Total trades: {n_trades}")
print(f"By order type:")
print(trades['order_type'].value_counts())
print(f"\nBy stock:")
print(trades['stock'].value_counts())

print(f"\nStep 2: Slippage Statistics (Basis Points)")
print(f"-" * 50)

slippage_stats = pd.DataFrame({
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
    'vs Arrival Price': [
        trades['slippage_ap_bps'].mean(),
        trades['slippage_ap_bps'].median(),
        trades['slippage_ap_bps'].std(),
        trades['slippage_ap_bps'].min(),
        trades['slippage_ap_bps'].max(),
        trades['slippage_ap_bps'].quantile(0.25),
        trades['slippage_ap_bps'].quantile(0.75),
    ],
    'vs VWAP': [
        trades['slippage_vwap_bps'].mean(),
        trades['slippage_vwap_bps'].median(),
        trades['slippage_vwap_bps'].std(),
        trades['slippage_vwap_bps'].min(),
        trades['slippage_vwap_bps'].max(),
        trades['slippage_vwap_bps'].quantile(0.25),
        trades['slippage_vwap_bps'].quantile(0.75),
    ],
    'vs TWAP': [
        trades['slippage_twap_bps'].mean(),
        trades['slippage_twap_bps'].median(),
        trades['slippage_twap_bps'].std(),
        trades['slippage_twap_bps'].min(),
        trades['slippage_twap_bps'].max(),
        trades['slippage_twap_bps'].quantile(0.25),
        trades['slippage_twap_bps'].quantile(0.75),
    ],
})

print(slippage_stats.to_string(index=False))

print(f"\nStep 3: Slippage by Order Type")
print(f"-" * 50)

order_type_slippage = trades.groupby('order_type')[['slippage_ap_bps', 'slippage_vwap_bps']].agg(['mean', 'std', 'count'])
print(order_type_slippage.round(2))

print(f"\nStep 4: Slippage by Stock")
print(f"-" * 50)

stock_slippage = trades.groupby('stock')[['slippage_ap_bps', 'order_size_pct_vol']].agg(['mean', 'std'])
print(stock_slippage.round(2))

print(f"\nStep 5: Slippage by Order Size (Correlation)")
print(f"-" * 50)

corr_size_slippage_ap = trades['order_size_pct_vol'].corr(trades['slippage_ap_bps'])
corr_size_slippage_vwap = trades['order_size_pct_vol'].corr(trades['slippage_vwap_bps'])
print(f"Correlation (order size vs slippage vs AP): {corr_size_slippage_ap:.3f}")
print(f"Correlation (order size vs slippage vs VWAP): {corr_size_slippage_vwap:.3f}")
print(f"Interpretation: {'Strong' if abs(corr_size_slippage_ap) > 0.5 else 'Moderate' if abs(corr_size_slippage_ap) > 0.3 else 'Weak'} relationship")

print(f"\nStep 6: Cost Component Breakdown")
print(f"-" * 50)

avg_spread_cost = trades['spread_cost'].mean() * 10000
avg_impact_cost = trades['impact_cost'].mean() * 10000
avg_commission = 0.5

print(f"Average spread cost: {avg_spread_cost:.2f} bps")
print(f"Average market impact: {avg_impact_cost:.2f} bps")
print(f"Average commission: {avg_commission:.2f} bps")
print(f"Total component estimate: {avg_spread_cost + avg_impact_cost + avg_commission:.2f} bps")
print(f"Actual average slippage (vs AP): {trades['slippage_ap_bps'].mean():.2f} bps")

print(f"\nStep 7: Slippage by Time of Day")
print(f"-" * 50)

tod_slippage = trades.groupby('time_of_day')[['slippage_ap_bps']].agg(['mean', 'std', 'count'])
print(tod_slippage.round(2))

# VISUALIZATION
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Distribution of slippage
ax = axes[0, 0]
ax.hist(trades['slippage_ap_bps'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(trades['slippage_ap_bps'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {trades['slippage_ap_bps'].mean():.1f} bps")
ax.set_xlabel('Slippage vs Arrival Price (bps)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Slippage')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Slippage by order type
ax = axes[0, 1]
order_type_means = trades.groupby('order_type')['slippage_ap_bps'].mean()
order_type_means.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_ylabel('Average Slippage (bps)')
ax.set_title('Slippage by Order Type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(alpha=0.3, axis='y')

# Plot 3: Slippage vs order size (scatter)
ax = axes[0, 2]
scatter = ax.scatter(trades['order_size_pct_vol'], trades['slippage_ap_bps'], 
                     c=trades['order_type'].astype('category').cat.codes, cmap='tab10', alpha=0.6, s=30)
ax.set_xlabel('Order Size (% of daily volume)')
ax.set_ylabel('Slippage (bps)')
ax.set_title('Slippage vs Order Size')
ax.grid(alpha=0.3)
ax.set_xscale('log')

# Plot 4: Slippage by stock
ax = axes[1, 0]
stock_means = trades.groupby('stock')['slippage_ap_bps'].mean()
stock_means.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_ylabel('Average Slippage (bps)')
ax.set_title('Slippage by Stock')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(alpha=0.3, axis='y')

# Plot 5: Box plot of slippage by order type
ax = axes[1, 1]
slippage_by_type = [trades[trades['order_type'] == ot]['slippage_ap_bps'].values 
                    for ot in trades['order_type'].unique()]
bp = ax.boxplot(slippage_by_type, labels=trades['order_type'].unique(), patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Slippage (bps)')
ax.set_title('Slippage Distribution by Order Type')
ax.grid(alpha=0.3, axis='y')

# Plot 6: Slippage by time of day
ax = axes[1, 2]
tod_means = trades.groupby('time_of_day')['slippage_ap_bps'].mean()
tod_means.plot(kind='bar', ax=ax, color=['green', 'blue', 'red'], edgecolor='black', alpha=0.7)
ax.set_ylabel('Average Slippage (bps)')
ax.set_title('Slippage by Time of Day')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("KEY INSIGHTS")
print(f"="*100)
print(f"- Market orders: Highest slippage ({trades[trades['order_type']=='market']['slippage_ap_bps'].mean():.2f} bps) - pay spread + impact")
print(f"- VWAP/TWAP algos: Lower slippage ({trades[trades['order_type']=='algo_vwap']['slippage_ap_bps'].mean():.2f} bps) - track benchmark")
print(f"- Limit orders: Variable (some miss execution, some get better prices)")
print(f"- Larger orders: Higher slippage (sqrt-law market impact)")
print(f"- Best execution: VWAP benchmark typically optimal")
