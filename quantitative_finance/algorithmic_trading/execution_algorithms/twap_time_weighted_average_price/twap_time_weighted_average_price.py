import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate intraday market
minutes = np.arange(1, 391)
hours = 9.5 + minutes / 60

# Generate prices with trend
base_price = 100
trend = np.linspace(0, 2, 390)  # Slight uptrend
prices = base_price + trend + np.random.normal(0, 0.3, 390)

# Generate volume (U-shaped)
base_vol = 1000
vol_morning = base_vol * 1.2
vol_lunch = base_vol * 0.6
vol_afternoon = base_vol * 1.3

volumes = np.concatenate([
    np.random.poisson(vol_morning, 90),
    np.random.poisson(vol_lunch, 120),
    np.random.poisson(vol_afternoon, 180)
])

# Create market data
market_data = pd.DataFrame({
    'minute': minutes,
    'hour': hours,
    'price': prices,
    'volume': volumes,
})

# Calculate benchmarks
market_data['cum_volume'] = market_data['volume'].cumsum()
market_data['dollar_volume'] = market_data['price'] * market_data['volume']
market_data['cum_dollar_volume'] = market_data['dollar_volume'].cumsum()
market_data['vwap'] = market_data['cum_dollar_volume'] / market_data['cum_volume']
market_data['twap'] = market_data['price'].expanding().mean()

# Execute order using TWAP and VWAP
target_order = 50000

# TWAP: Equal allocation per minute
twap_qty_per_min = np.full(390, target_order / 390)
twap_filled = (twap_qty_per_min * market_data['price'].values).sum()
twap_avg_price = twap_filled / target_order

# VWAP: Proportional to volume
vwap_qty_per_min = market_data['volume'].values * (target_order / market_data['volume'].sum())
vwap_filled = (vwap_qty_per_min * market_data['price'].values).sum()
vwap_avg_price = vwap_filled / target_order

final_vwap = market_data['vwap'].iloc[-1]
final_twap = market_data['twap'].iloc[-1]

print("="*60)
print("TWAP vs VWAP EXECUTION COMPARISON")
print("="*60)
print(f"\nBenchmarks:")
print(f"  Final VWAP:  ${final_vwap:.4f}")
print(f"  Final TWAP:  ${final_twap:.4f}")
print(f"\nExecution Results (50k shares):")
print(f"  TWAP Execution Avg: ${twap_avg_price:.4f}")
print(f"    vs TWAP benchmark: ${twap_avg_price - final_twap:.4f} ({(twap_avg_price - final_twap)/final_twap*10000:.1f} bps)")
print(f"\n  VWAP Execution Avg: ${vwap_avg_price:.4f}")
print(f"    vs VWAP benchmark: ${vwap_avg_price - final_vwap:.4f} ({(vwap_avg_price - final_vwap)/final_vwap*10000:.1f} bps)")
print(f"\nComparison:")
print(f"  VWAP Execution better by: ${twap_avg_price - vwap_avg_price:.4f} "
      f"({(twap_avg_price - vwap_avg_price)/vwap_avg_price*10000:.1f} bps)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Price trajectory
ax = axes[0, 0]
ax.plot(hours, prices, 'b-', linewidth=1.5, label='Market Price')
ax.axhline(y=twap_avg_price, color='g', linestyle='--', linewidth=2, label=f'TWAP Exec (${twap_avg_price:.2f})')
ax.axhline(y=vwap_avg_price, color='r', linestyle='--', linewidth=2, label=f'VWAP Exec (${vwap_avg_price:.2f})')
ax.set_title('Price Movement & Execution Prices')
ax.set_xlabel('Time')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Volume profile
ax = axes[0, 1]
colors = ['red' if v < np.mean(volumes) else 'green' for v in volumes]
ax.bar(hours, volumes, width=0.015, alpha=0.6, color=colors)
ax.axhline(y=np.mean(volumes), color='black', linestyle='--', linewidth=1, label='Avg Volume')
ax.set_title('Intraday Volume Profile')
ax.set_xlabel('Time')
ax.set_ylabel('Volume (shares)')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Cumulative execution
ax = axes[1, 0]
twap_cumsum = np.cumsum(twap_qty_per_min)
vwap_cumsum = np.cumsum(vwap_qty_per_min)
ax.plot(hours, twap_cumsum, 'g-', linewidth=2, label='TWAP Cumulative')
ax.plot(hours, vwap_cumsum, 'r-', linewidth=2, label='VWAP Cumulative')
ax.axhline(y=target_order, color='black', linestyle=':', linewidth=1, label='Target Order')
ax.set_title('Cumulative Execution Progress')
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Shares')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Per-minute slicing comparison
ax = axes[1, 1]
x_pos = np.arange(0, 390, 30)
ax.scatter(hours[x_pos], twap_qty_per_min[x_pos], s=50, alpha=0.6, label='TWAP Qty/Min', color='green')
ax.scatter(hours[x_pos], vwap_qty_per_min[x_pos], s=50, alpha=0.6, label='VWAP Qty/Min', color='red')
ax.set_title('Allocation per Minute (Sample)')
ax.set_xlabel('Time')
ax.set_ylabel('Shares per Minute')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()