import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# Simulate high-frequency market making
np.random.seed(42)

# Parameters
n_minutes = 60  # Trading session (1 hour)
timestamps = pd.date_range('2024-01-01 09:30', periods=n_minutes*60, freq='1S')
price_path = 100 + np.cumsum(np.random.normal(0, 0.05, n_minute*60))  # Brownian motion

# Order flow simulation
buy_orders = np.random.poisson(5, n_minutes*60)    # Expected 5 buy orders/sec
sell_orders = np.random.poisson(5, n_minutes*60)   # Expected 5 sell orders/sec

# Add informed order pattern (minority)
informed_start = int(n_minutes*60 * 0.3)  # Informed traders active at 30% mark
buy_orders[informed_start:informed_start+600] += np.random.poisson(3, 600)

# Initialize state
portfolio_state = {
    'timestamp': [],
    'mid_price': [],
    'mm_bid': [],
    'mm_ask': [],
    'inventory': [],
    'cumul_pnl': [],
    'vpin': [],
    'effective_spread': [],
    'trades': []
}

inventory = 0
cumul_pnl = 0
spread_history = deque(maxlen=60)  # Rolling 60-second window for VPIN
price_momentum = deque(maxlen=10)
order_imbalance = deque(maxlen=30)

print("="*100)
print("HIGH-FREQUENCY TRADING: MARKET MAKER SIMULATION")
print("="*100)

# Simulation loop
for i in range(1, len(timestamps)):
    mid = price_path[i]
    
    # Compute VPIN: Order flow toxicity indicator
    buy_vol = buy_orders[i]
    sell_vol = sell_orders[i]
    total_vol = buy_vol + sell_vol
    
    if total_vol > 0:
        buy_ratio = buy_vol / total_vol
        # Simple VPIN: Probability buy orders are informed (based on price correlation)
        price_change = price_path[i] - price_path[i-1]
        vpin = min(100, max(0, 50 + (price_change / 0.01) * 20))  # Simplified
    else:
        vpin = 50
    
    # Inventory adjustment: Quote tighter if neutral, wider if large position
    base_spread = 0.01
    inventory_cost = abs(inventory) * 0.001  # Penalty proportional to position
    
    # Toxicity adjustment: Widen spreads if high VPIN
    toxicity_cost = max(0, (vpin - 50) / 50) * 0.02  # Up to 2 cents extra
    
    # Total spread
    total_spread = base_spread + inventory_cost + toxicity_cost
    
    # Quote placement
    bid = mid - total_spread / 2
    ask = mid + total_spread / 2
    
    # Simulated fills based on order flow
    bid_fills = np.random.binomial(n=int(buy_orders[i]), p=0.3, size=1)[0]
    ask_fills = np.random.binomial(n=int(sell_orders[i]), p=0.3, size=1)[0]
    
    # P&L calculation
    pnl_from_trades = bid_fills * (mid - bid) - ask_fills * (ask - mid)  # Spread profit
    pnl_from_inventory = inventory * (mid - price_path[i-1])  # Inventory change
    total_pnl = pnl_from_trades + pnl_from_inventory
    cumul_pnl += total_pnl
    
    # Update inventory
    inventory += ask_fills - bid_fills
    
    # Record state
    portfolio_state['timestamp'].append(timestamps[i])
    portfolio_state['mid_price'].append(mid)
    portfolio_state['mm_bid'].append(bid)
    portfolio_state['mm_ask'].append(ask)
    portfolio_state['inventory'].append(inventory)
    portfolio_state['cumul_pnl'].append(cumul_pnl)
    portfolio_state['vpin'].append(vpin)
    portfolio_state['effective_spread'].append(total_spread)
    
    if i % 600 == 0:  # Print every 10 minutes
        print(f"Time: {timestamps[i]} | Inventory: {inventory:+.0f} | Cumul P&L: ${cumul_pnl:.2f} | VPIN: {vpin:.1f} | Spread: {total_spread*100:.2f}¢")

df_mm = pd.DataFrame(portfolio_state)

print(f"\n" + "="*100)
print("MARKET MAKER PERFORMANCE SUMMARY")
print("="*100)

print(f"\nTotal P&L: ${cumul_pnl:.2f}")
print(f"Winning seconds: {(df_mm['cumul_pnl'].diff() > 0).sum()}")
print(f"Losing seconds: {(df_mm['cumul_pnl'].diff() < 0).sum()}")
print(f"Win rate: {(df_mm['cumul_pnl'].diff() > 0).sum() / len(df_mm) * 100:.1f}%")
print(f"\nInventory stats:")
print(f"  Average inventory: {df_mm['inventory'].mean():.1f} shares")
print(f"  Max long: {df_mm['inventory'].max():.1f} shares")
print(f"  Max short: {df_mm['inventory'].min():.1f} shares")
print(f"\nSpread stats:")
print(f"  Average spread: {df_mm['effective_spread'].mean()*100:.2f}¢")
print(f"  Min spread: {df_mm['effective_spread'].min()*100:.2f}¢")
print(f"  Max spread: {df_mm['effective_spread'].max()*100:.2f}¢")
print(f"\nToxicity (VPIN):")
print(f"  Mean VPIN: {df_mm['vpin'].mean():.1f}")
print(f"  High toxicity periods (VPIN > 70): {(df_mm['vpin'] > 70).sum()} seconds")

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Price with MM quotes
ax = axes[0, 0]
ax.plot(df_mm['timestamp'], df_mm['mid_price'], label='Mid Price', linewidth=1.5, color='black')
ax.plot(df_mm['timestamp'], df_mm['mm_bid'], label='MM Bid', linewidth=0.8, alpha=0.7, color='red')
ax.plot(df_mm['timestamp'], df_mm['mm_ask'], label='MM Ask', linewidth=0.8, alpha=0.7, color='green')
ax.fill_between(df_mm['timestamp'], df_mm['mm_bid'], df_mm['mm_ask'], alpha=0.2, color='blue')
ax.set_title('Price & Market Maker Quotes')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Inventory position over time
ax = axes[0, 1]
ax.bar(df_mm['timestamp'], df_mm['inventory'], width=0.01, alpha=0.7, color=['red' if x < 0 else 'green' for x in df_mm['inventory']])
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_title('Inventory Position (Market Maker)')
ax.set_ylabel('Shares')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Cumulative P&L and VPIN
ax1 = axes[1, 0]
ax1.plot(df_mm['timestamp'], df_mm['cumul_pnl'], linewidth=2, label='Cumul P&L', color='blue')
ax1.set_ylabel('P&L ($)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(df_mm['timestamp'], df_mm['vpin'], linewidth=1.5, label='VPIN (Toxicity)', color='red', alpha=0.7)
ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='High Toxicity')
ax2.set_ylabel('VPIN', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax1.set_title('P&L vs Order Flow Toxicity')
ax1.grid(alpha=0.3)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Plot 4: Spread adjustment
ax = axes[1, 1]
ax.plot(df_mm['timestamp'], df_mm['effective_spread']*100, linewidth=1.5, label='Total Spread', color='purple')
ax.fill_between(df_mm['timestamp'], 0, df_mm['effective_spread']*100, alpha=0.3, color='purple')
ax.set_title('Dynamic Spread Over Time')
ax.set_ylabel('Spread (cents)')
ax.set_xlabel('Time')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("INSIGHTS")
print("="*100)
print(f"- Market making: Spread income (0.5-2 bps) vs inventory losses (large moves)")
print(f"- Toxicity detection: VPIN spikes = informed trading, widen spreads defensively")
print(f"- Inventory management: Control position size to limit downside")
print(f"- Dynamic pricing: Continuous quote adjustment critical (vs static quotes)")
print(f"- Profitability: Highly leveraged (small margins on each trade, but 1000s/day)")