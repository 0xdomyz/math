import numpy as np
import pandas as pd

# Simulate order execution
np.random.seed(123)
order_size = 100_000  # shares to buy
arrival_price = 50.00  # decision price at 10:00 AM
intraday_prices = 50.00 + np.cumsum(np.random.randn(100) * 0.02)  # Random walk

# Simulate fills using VWAP algo (executes over time)
fill_times = np.linspace(0, 99, 20).astype(int)  # 20 fills evenly spaced
fill_shares = order_size / len(fill_times)  # Equal-sized fills
fill_prices = intraday_prices[fill_times]

# Assume 5% of order unfilled
filled_shares = 0.95 * order_size
unfilled_shares = order_size - filled_shares
closing_price = intraday_prices[-1]

# Calculate implementation shortfall components
# 1. Executed cost (filled shares)
executed_value = fill_shares * fill_prices.sum()
arrival_value = arrival_price * filled_shares
execution_cost = executed_value - arrival_value
execution_cost_bps = (execution_cost / arrival_value) * 10000

# 2. Opportunity cost (unfilled shares)
opportunity_cost = unfilled_shares * (closing_price - arrival_price)
opportunity_cost_bps = (opportunity_cost / (arrival_price * order_size)) * 10000

# 3. Total implementation shortfall
total_cost = execution_cost + opportunity_cost
total_cost_bps = (total_cost / (arrival_price * order_size)) * 10000

# Average fill price
avg_fill_price = (fill_shares * fill_prices.sum()) / filled_shares

# Slippage vs VWAP
intraday_vwap = intraday_prices[:len(fill_times)].mean()  # Simplified VWAP
vwap_slippage_bps = ((avg_fill_price - intraday_vwap) / arrival_price) * 10000

print("=" * 70)
print("TRANSACTION COST ANALYTICS (Implementation Shortfall)")
print("=" * 70)
print(f"Order Details:")
print(f"  Total Order Size:              {order_size:>12,} shares")
print(f"  Arrival Price (Decision):      ${arrival_price:>11.2f}")
print(f"  Closing Price:                 ${closing_price:>11.2f}")
print(f"  Intraday Price Move:           {(closing_price - arrival_price):>11.2f} ({(closing_price/arrival_price - 1)*100:+.2f}%)")
print()
print(f"Execution Summary:")
print(f"  Filled Shares:                 {filled_shares:>12,.0f} ({filled_shares/order_size:.0%})")
print(f"  Unfilled Shares:               {unfilled_shares:>12,.0f} ({unfilled_shares/order_size:.0%})")
print(f"  Average Fill Price:            ${avg_fill_price:>11.2f}")
print(f"  Number of Fills:               {len(fill_times):>12}")
print()
print(f"Cost Breakdown:")
print(f"  Execution Cost (Filled):       ${execution_cost:>11,.2f}  ({execution_cost_bps:>6.1f} bps)")
print(f"  Opportunity Cost (Unfilled):   ${opportunity_cost:>11,.2f}  ({opportunity_cost_bps:>6.1f} bps)")
print(f"  Total Implementation Shortfall:${total_cost:>11,.2f}  ({total_cost_bps:>6.1f} bps)")
print()
print(f"Benchmark Comparison:")
print(f"  Slippage vs. Arrival Price:    {(avg_fill_price - arrival_price):>11.2f}  ({(avg_fill_price/arrival_price - 1)*10000:>6.1f} bps)")
print(f"  Slippage vs. Intraday VWAP:    {(avg_fill_price - intraday_vwap):>11.2f}  ({vwap_slippage_bps:>6.1f} bps)")
print()

# Interpretation
if total_cost_bps < 10:
    assessment = "✓ Excellent execution (< 10 bps)"
elif total_cost_bps < 25:
    assessment = "✓ Good execution (10-25 bps)"
elif total_cost_bps < 50:
    assessment = "⚠️  Acceptable execution (25-50 bps)"
else:
    assessment = "⚠️  Poor execution (> 50 bps)"

print(f"Assessment: {assessment}")
print("=" * 70)

# Visualize fills over time
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(intraday_prices, label='Intraday Price', linewidth=2)
plt.scatter(fill_times, fill_prices, color='red', s=100, zorder=5, label='Fill Prices')
plt.axhline(arrival_price, color='green', linestyle='--', linewidth=2, label=f'Arrival Price (${arrival_price:.2f})')
plt.axhline(intraday_vwap, color='orange', linestyle=':', linewidth=2, label=f'Intraday VWAP (${intraday_vwap:.2f})')
plt.xlabel('Time (Intraday Intervals)')
plt.ylabel('Price ($)')
plt.title('Execution TCA: Fill Prices vs. Benchmarks')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()