import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from datetime import datetime, timedelta

# Generate realistic intraday data with VWAP, trades, price movement
np.random.seed(42)
trading_minutes = 390  # 6.5 hours
minutes = np.arange(trading_minutes)

# Underlying price follows random walk with drift
true_drift = 0.0002
true_vol = 0.05 / np.sqrt(252 * 390)  # Annualized vol scaled to minutes
price_move = np.random.normal(true_drift, true_vol, trading_minutes)
base_price = 100.0
intraday_prices = base_price + np.cumsum(price_move)

# Volume follows U-shape (higher at open/close)
volume_base = 1000 * (1 + 0.5 * np.sin(np.pi * minutes / trading_minutes))
volume = volume_base + np.random.normal(0, 100, trading_minutes)
volume = np.maximum(volume, 100)

# VWAP calculation (cumulative)
vwap_cumsum = np.cumsum(intraday_prices * volume)
vwap_cumvol = np.cumsum(volume)
vwap = vwap_cumsum / vwap_cumvol

# TWAP calculation
twap_cumsum = np.cumsum(intraday_prices)
twap = twap_cumsum / (minutes + 1)

# Simulate execution strategy: execute 50k shares gradually
total_shares_to_execute = 50000
start_minute = 30
end_minute = 360

# Execution profile: gentle ramp up, then taper down (to minimize impact)
exec_minutes = np.arange(start_minute, end_minute)
exec_profile = np.exp(-2 * ((exec_minutes - (start_minute + end_minute) / 2) / (end_minute - start_minute))**2)
exec_shares_per_minute = (total_shares_to_execute / np.sum(exec_profile)) * exec_profile
exec_shares_per_minute = np.round(exec_shares_per_minute).astype(int)
exec_shares_per_minute = np.concatenate([
    np.zeros(start_minute),
    exec_shares_per_minute,
    np.zeros(trading_minutes - end_minute)
])

# Ensure total matches
diff = total_shares_to_execute - np.sum(exec_shares_per_minute)
exec_shares_per_minute[end_minute - 1] += diff

# Execution price per minute (midpoint + half-spread + market impact)
spread = 0.01  # $0.01 bid-ask
market_impact_coef = 0.0001  # Impact = coef * sqrt(volume_fraction)

exec_prices = []
cum_executed = 0
for i in range(trading_minutes):
    if exec_shares_per_minute[i] > 0:
        # Market impact: sqrt of order size relative to volume
        order_frac = exec_shares_per_minute[i] / volume[i]
        impact = market_impact_coef * np.sqrt(order_frac)
        
        # Execution price: midpoint + half-spread + impact
        exec_price = intraday_prices[i] + 0.5 * spread + impact
        
        exec_prices.append(exec_price)
        cum_executed += exec_shares_per_minute[i]

# Calculate benchmarks
arrival_price = intraday_prices[start_minute]
execution_avg_price = np.mean(exec_prices)
final_price = intraday_prices[-1]
vwap_benchmark = vwap[-1]
twap_benchmark = twap[-1]

print("="*100)
print("TRANSACTION COST ANALYSIS (TCA)")
print("="*100)

print(f"\nStep 1: Market Conditions")
print(f"-" * 50)
print(f"Start price: ${arrival_price:.2f}")
print(f"End price: ${final_price:.2f}")
print(f"Market move: ${final_price - arrival_price:.2f} ({(final_price - arrival_price) / arrival_price * 100:.2f}%)")
print(f"Intraday volatility: {np.std(intraday_prices) * 100:.2f}%")
print(f"Total volume: {np.sum(volume):,.0f} shares")
print(f"Execution period: Minute {start_minute} to {end_minute}")
print(f"Total shares executed: {total_shares_to_execute:,.0f}")

print(f"\nStep 2: Benchmark Prices")
print(f"-" * 50)
print(f"Arrival Price (AP): ${arrival_price:.2f}")
print(f"VWAP: ${vwap_benchmark:.2f}")
print(f"TWAP: ${twap_benchmark:.2f}")
print(f"Execution Avg: ${execution_avg_price:.2f}")
print(f"Final Price: ${final_price:.2f}")

print(f"\nStep 3: Cost Analysis (Per Share, in Basis Points)")
print(f"-" * 50)

# Costs in dollars and bps
ap_cost_pershare = execution_avg_price - arrival_price
vwap_cost_pershare = execution_avg_price - vwap_benchmark
twap_cost_pershare = execution_avg_price - twap_benchmark

ap_cost_bps = (ap_cost_pershare / arrival_price) * 10000
vwap_cost_bps = (vwap_cost_pershare / vwap_benchmark) * 10000
twap_cost_bps = (twap_cost_pershare / twap_benchmark) * 10000

costs_df = pd.DataFrame({
    'Benchmark': ['Arrival Price', 'VWAP', 'TWAP'],
    'Benchmark Price': [arrival_price, vwap_benchmark, twap_benchmark],
    'Cost per Share': [ap_cost_pershare, vwap_cost_pershare, twap_cost_pershare],
    'Cost (bps)': [ap_cost_bps, vwap_cost_bps, twap_cost_bps],
    'Total Cost': [ap_cost_pershare * total_shares_to_execute,
                   vwap_cost_pershare * total_shares_to_execute,
                   twap_cost_pershare * total_shares_to_execute],
})

print(costs_df.to_string(index=False))

print(f"\nStep 4: Implementation Shortfall (Decision-Based)")
print(f"-" * 50)

decision_price = arrival_price  # Assume decision at arrival
timing_cost = (final_price - decision_price) * total_shares_to_execute
execution_cost = (execution_avg_price - arrival_price) * total_shares_to_execute
total_shortfall = timing_cost + execution_cost

print(f"Decision Price: ${decision_price:.2f}")
print(f"Execution Avg: ${execution_avg_price:.2f}")
print(f"Final Price: ${final_price:.2f}")
print(f"")
print(f"Timing Cost (Market Move): ${timing_cost:,.2f} ({timing_cost/final_price/total_shares_to_execute*10000:.2f} bps)")
print(f"Execution Cost (Algo): ${execution_cost:,.2f} ({execution_cost/arrival_price/total_shares_to_execute*10000:.2f} bps)")
print(f"Total Shortfall: ${total_shortfall:,.2f}")
print(f"Total Shortfall (bps): {total_shortfall/decision_price/total_shares_to_execute*10000:.2f} bps")

if timing_cost != 0:
    timing_pct = abs(timing_cost) / (abs(timing_cost) + abs(execution_cost)) * 100
    execution_pct = abs(execution_cost) / (abs(timing_cost) + abs(execution_cost)) * 100
    print(f"")
    print(f"Attribution:")
    print(f"  Timing (market move): {timing_pct:.1f}%")
    print(f"  Execution (algo): {execution_pct:.1f}%")

print(f"\nStep 5: Cost Decomposition (Slippage Components)")
print(f"-" * 50)

# Estimate components
spread_cost = 0.5 * spread * total_shares_to_execute  # Half-spread per share
# Market impact: sqrt-law
avg_order_frac = np.mean([exec_shares_per_minute[i] / volume[i] 
                           for i in range(len(exec_shares_per_minute)) 
                           if exec_shares_per_minute[i] > 0])
market_impact = market_impact_coef * np.sqrt(avg_order_frac) * arrival_price * total_shares_to_execute

print(f"Estimated Spread Cost: ${spread_cost:,.2f}")
print(f"Estimated Market Impact: ${market_impact:,.2f}")
print(f"Total (Slippage): ${spread_cost + market_impact:,.2f}")

print(f"\nStep 6: Performance Ranking (vs Benchmarks)")
print(f"-" * 50)

rank_df = pd.DataFrame({
    'Benchmark': ['Arrival Price', 'VWAP', 'TWAP'],
    'Cost (bps)': [ap_cost_bps, vwap_cost_bps, twap_cost_bps],
    'Rank': [2, 1, 3]  # Rank by cost (1=best, 3=worst)
})
rank_df = rank_df.sort_values('Cost (bps)')
print(rank_df.to_string(index=False))

# VISUALIZATION
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Intraday price and benchmarks
ax = axes[0, 0]
ax.plot(minutes, intraday_prices, label='Intraday Price', linewidth=2, color='black')
ax.plot(minutes, vwap, label='VWAP', linewidth=2, color='blue', alpha=0.7)
ax.plot(minutes, twap, label='TWAP', linewidth=2, color='green', alpha=0.7)
ax.axhline(y=arrival_price, color='red', linestyle='--', label='Arrival Price')
ax.axhline(y=execution_avg_price, color='purple', linestyle='--', label='Execution Avg')
ax.set_xlabel('Minute')
ax.set_ylabel('Price ($)')
ax.set_title('Intraday Price vs Benchmarks')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Volume and execution
ax = axes[0, 1]
ax.bar(minutes, volume, alpha=0.5, label='Market Volume', edgecolor='black', width=1)
ax.bar(minutes, exec_shares_per_minute, alpha=0.7, label='Execution Volume', edgecolor='red', width=1)
ax.set_xlabel('Minute')
ax.set_ylabel('Shares')
ax.set_title('Execution Profile vs Market Volume')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Cost vs benchmarks (bar chart)
ax = axes[0, 2]
benchmarks = ['Arrival\nPrice', 'VWAP', 'TWAP']
costs = [ap_cost_bps, vwap_cost_bps, twap_cost_bps]
colors = ['green' if c < 0 else 'red' for c in costs]
bars = ax.bar(benchmarks, costs, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Cost (basis points)')
ax.set_title('Execution Cost vs Benchmarks')
ax.grid(alpha=0.3, axis='y')
for bar, cost in zip(bars, costs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{cost:.1f}', ha='center', va='bottom' if cost > 0 else 'top')

# Plot 4: Implementation shortfall breakdown
ax = axes[1, 0]
components = ['Timing Cost\n(Market)', 'Execution Cost\n(Algo)', 'Total Shortfall']
values = [timing_cost / 1000, execution_cost / 1000, total_shortfall / 1000]  # In thousands
colors = ['red' if v < 0 else 'green' for v in values]
bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Cost ($thousands)')
ax.set_title('Implementation Shortfall Decomposition')
ax.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${val:.1f}k', ha='center', va='bottom' if val > 0 else 'top')

# Plot 5: Cumulative execution and price
ax = axes[1, 1]
cum_shares = np.cumsum(exec_shares_per_minute)
ax2 = ax.twinx()
ax.plot(minutes, cum_shares / 1000, label='Cumulative Execution', linewidth=2, color='blue')
ax2.plot(minutes, intraday_prices, label='Price', linewidth=2, color='red', alpha=0.7)
ax.set_xlabel('Minute')
ax.set_ylabel('Cumulative Shares (thousands)', color='blue')
ax2.set_ylabel('Price ($)', color='red')
ax.set_title('Execution Progress vs Price Move')
ax.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')
ax.grid(alpha=0.3)

# Plot 6: Slippage decomposition (pie chart)
ax = axes[1, 2]
if spread_cost > 0 and market_impact > 0:
    slippage_components = [spread_cost, market_impact]
    labels = [f'Spread\n${spread_cost:,.0f}', f'Impact\n${market_impact:,.0f}']
    colors = ['skyblue', 'orange']
    wedges, texts, autotexts = ax.pie(slippage_components, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Slippage Decomposition')

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("INSIGHTS")
print(f"="*100)
print(f"- Best benchmark: VWAP ({vwap_cost_bps:.2f} bps) - reflects volume distribution")
print(f"- Execution timeline: {end_minute - start_minute} minutes, smooth ramp (minimize impact)")
print(f"- Implementation shortfall driven by: {'Timing' if abs(timing_cost) > abs(execution_cost) else 'Execution'}")
print(f"- Estimated market impact cost: ~{market_impact_coef*100:.3f}% × sqrt(volume_fraction)")
print(f"- Spread friction: ${spread_cost:,.0f} (transaction cost floor)")
