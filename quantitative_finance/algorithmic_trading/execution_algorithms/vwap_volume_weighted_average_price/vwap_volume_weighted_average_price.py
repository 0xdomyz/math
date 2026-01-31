import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate intraday volume profile
np.random.seed(42)

# Trading day: 9:30am - 4:00pm (390 minutes)
minutes = np.arange(1, 391)
hours = 9.5 + minutes / 60

# Typical intraday volume pattern (U-shaped)
base_vol = 1000  # base volume per minute
morning_ramp = 1 + 0.5 * np.sin(np.pi * minutes / 100)  # 9:30-11:00
lunch_dip = 0.6  # 11:00-1:00 (around minute 90-210)
afternoon_ramp = 1 + 0.3 * np.sin(np.pi * (minutes - 210) / 180)  # 1:00-4:00

vol_multiplier = np.ones(390)
vol_multiplier[:90] = morning_ramp[:90]
vol_multiplier[90:210] = lunch_dip
vol_multiplier[210:] = afternoon_ramp[210:]

# Simulate intraday prices (random walk)
price_base = 100
price_changes = np.random.normal(0, 0.05, 390)  # ~5 bps volatility
prices = price_base + np.cumsum(price_changes)

# Daily volume
daily_volume = np.random.poisson(base_vol, 390) * vol_multiplier
daily_volume = daily_volume.astype(int)

total_daily_vol = daily_volume.sum()
print(f"Total Daily Volume: {total_daily_vol:,} shares")

# Create DataFrame
intraday_data = pd.DataFrame({
    'minute': minutes,
    'hour': hours,
    'price': prices,
    'volume': daily_volume,
    'cum_volume': daily_volume.cumsum(),
    'cum_dollar_volume': (prices * daily_volume).cumsum(),
})

# Calculate VWAP
intraday_data['vwap'] = intraday_data['cum_dollar_volume'] / intraday_data['cum_volume']

print("\nIntraday VWAP Profile (Sample):")
print(intraday_data[::60][['hour', 'price', 'volume', 'vwap']].to_string(index=False))

# VWAP Execution Strategy
target_order = 50000  # 50k shares to execute
volume_pct = intraday_data['volume'] / total_daily_vol

# Allocation: proportional to volume
target_qty = (volume_pct * target_order).values
target_qty = target_qty.astype(int)

# Simulate fills: assume we get filled at market price
filled_qty = target_qty.copy()
filled_qty[-1] += (target_order - target_qty.sum())  # Adjust last bar for rounding

filled_price = prices.copy()
execution_cost = (filled_qty * filled_price).sum()
avg_execution_price = execution_cost / target_order
final_vwap = intraday_data['vwap'].iloc[-1]

print(f"\n{'='*60}")
print("VWAP EXECUTION ANALYSIS")
print(f"{'='*60}")
print(f"Order Size: {target_order:,} shares")
print(f"Final VWAP: ${final_vwap:.4f}")
print(f"Avg Execution Price: ${avg_execution_price:.4f}")
print(f"Slippage: ${avg_execution_price - final_vwap:.4f}")
print(f"Slippage (bps): {(avg_execution_price - final_vwap) / final_vwap * 10000:.2f} bps")
print(f"Total Cost: ${execution_cost:,.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Price and VWAP
ax = axes[0, 0]
ax.plot(hours, prices, 'b-', linewidth=1.5, label='Price')
ax.plot(hours, intraday_data['vwap'], 'r--', linewidth=2, label='VWAP')
ax.fill_between(hours, prices, intraday_data['vwap'], alpha=0.2)
ax.set_title('Price vs VWAP')
ax.set_xlabel('Time')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Intraday Volume Profile
ax = axes[0, 1]
ax.bar(hours, daily_volume, width=0.015, alpha=0.7, color='green')
ax.set_title('Intraday Volume Profile')
ax.set_xlabel('Time')
ax.set_ylabel('Volume (shares)')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Allocation vs Actual Fill
ax = axes[1, 0]
ax.plot(hours, target_qty, 'b-', linewidth=2, marker='o', markersize=3, 
       label='Target Allocation')
ax.set_title('VWAP Allocation Schedule')
ax.set_xlabel('Time')
ax.set_ylabel('Shares per Minute')
ax.grid(alpha=0.3)
ax.legend()

# Plot 4: Cumulative execution vs volume
ax = axes[1, 1]
ax.plot(hours, intraday_data['cum_volume'].values, 'g-', linewidth=2, 
       label='Cumulative Market Volume')
ax.plot(hours, np.cumsum(filled_qty), 'b--', linewidth=2, 
       label='Cumulative Filled Qty')
ax.set_title('Execution Progress vs Market Volume')
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Volume')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Performance Summary
print(f"\nExecution Summary:")
print(f"Filled Qty: {filled_qty.sum():,} shares")
print(f"Avg Fill Price: ${avg_execution_price:.4f}")
print(f"VWAP Benchmark: ${final_vwap:.4f}")
print(f"Outperformance: ${(final_vwap - avg_execution_price) * target_order:,.2f}")