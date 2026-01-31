import numpy as np
import matplotlib.pyplot as plt
from collections import deque

np.random.seed(42)

# ============================================================================
# LATENCY SIMULATION
# ============================================================================

print("="*70)
print("LATENCY & SPEED IN ALGORITHMIC TRADING")
print("="*70)

# Market parameters
n_events = 1000  # Number of arbitrage opportunities
opportunity_duration_us = 500  # Opportunity exists for 500 microseconds
profit_per_trade = 0.50  # $0.50 per successful arbitrage

# Trader latency profiles (microseconds)
traders = {
    'HFT Colocation': 50,      # Ultra-fast
    'Low-Latency': 200,        # Fast
    'Standard Algo': 5000,     # Slow (5ms)
    'Remote Server': 20000     # Very slow (20ms)
}

print(f"\nMarket Simulation:")
print(f"   Arbitrage Opportunities: {n_events}")
print(f"   Opportunity Duration: {opportunity_duration_us}μs")
print(f"   Profit per Trade: ${profit_per_trade:.2f}")

print(f"\nTrader Latencies:")
for trader, latency in traders.items():
    print(f"   {trader}: {latency}μs ({latency/1000:.1f}ms)")

# ============================================================================
# ARBITRAGE RACE SIMULATION
# ============================================================================

def simulate_arbitrage_race(traders, n_events, opportunity_duration_us, profit_per_trade):
    """
    Simulate arbitrage opportunities where fastest trader wins.
    Each opportunity appears at random time, lasts for opportunity_duration_us.
    """
    results = {trader: {'wins': 0, 'profit': 0, 'attempts': 0} 
              for trader in traders.keys()}
    
    for event in range(n_events):
        # All traders detect opportunity simultaneously (t=0)
        # But execute at different times based on latency
        
        trader_arrival_times = {trader: latency 
                               for trader, latency in traders.items()}
        
        # Find fastest trader
        fastest_trader = min(trader_arrival_times, key=trader_arrival_times.get)
        fastest_time = trader_arrival_times[fastest_trader]
        
        # Check if fastest trader arrives before opportunity expires
        if fastest_time <= opportunity_duration_us:
            results[fastest_trader]['wins'] += 1
            results[fastest_trader]['profit'] += profit_per_trade
        
        # Count attempts for all traders
        for trader in traders.keys():
            results[trader]['attempts'] += 1
    
    return results

results = simulate_arbitrage_race(traders, n_events, opportunity_duration_us, profit_per_trade)

print("\n" + "="*70)
print("ARBITRAGE RACE RESULTS")
print("="*70)

for trader, stats in results.items():
    win_rate = (stats['wins'] / stats['attempts']) * 100
    print(f"\n{trader}:")
    print(f"   Wins: {stats['wins']} / {stats['attempts']} ({win_rate:.1f}%)")
    print(f"   Total Profit: ${stats['profit']:,.2f}")

# ============================================================================
# LATENCY SENSITIVITY ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("LATENCY SENSITIVITY: How Much Does Speed Matter?")
print("="*70)

latency_range_us = np.array([10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
win_rates = []

for latency in latency_range_us:
    single_trader = {'Test Trader': latency}
    test_results = simulate_arbitrage_race(single_trader, n_events, 
                                          opportunity_duration_us, profit_per_trade)
    win_rate = (test_results['Test Trader']['wins'] / n_events) * 100
    win_rates.append(win_rate)
    
    print(f"\nLatency {latency}μs: Win Rate {win_rate:.1f}%, Profit ${test_results['Test Trader']['profit']:,.0f}")

# ============================================================================
# NETWORK LATENCY MODEL (Distance vs Technology)
# ============================================================================

print("\n" + "="*70)
print("NETWORK TECHNOLOGY COMPARISON")
print("="*70)

distances_km = np.array([1, 10, 50, 100, 500, 1000, 2000])

# Latency models (microseconds per km)
fiber_latency_per_km = 5.0  # ~2/3 speed of light
microwave_latency_per_km = 3.3  # ~speed of light (straight line)

fiber_latencies_ms = (distances_km * fiber_latency_per_km) / 1000
microwave_latencies_ms = (distances_km * microwave_latency_per_km) / 1000
improvement_pct = ((fiber_latencies_ms - microwave_latencies_ms) / fiber_latencies_ms) * 100

print("\nRound-Trip Latency by Distance:")
print(f"{'Distance (km)':<15} {'Fiber (ms)':<12} {'Microwave (ms)':<15} {'Improvement':<12}")
print("-" * 60)
for i, dist in enumerate(distances_km):
    print(f"{dist:<15} {fiber_latencies_ms[i]:<12.2f} {microwave_latencies_ms[i]:<15.2f} {improvement_pct[i]:<12.1f}%")

# ============================================================================
# CO-LOCATION ADVANTAGE CALCULATION
# ============================================================================

print("\n" + "="*70)
print("CO-LOCATION ROI ANALYSIS")
print("="*70)

# Assumptions
colocation_cost_per_month = 50_000  # $50K/month
trades_per_day = 10_000
trading_days_per_year = 252
profit_per_trade_with_advantage = 0.10  # $0.10 per trade (tighter)

# Without colocation
remote_latency_ms = 5
remote_win_rate = 0.20  # 20% win rate (slower)

# With colocation
colo_latency_us = 50
colo_win_rate = 0.90  # 90% win rate (much faster)

# Calculate annual P&L
remote_annual_profit = (
    trades_per_day * trading_days_per_year * 
    remote_win_rate * profit_per_trade_with_advantage
)

colo_annual_profit = (
    trades_per_day * trading_days_per_year * 
    colo_win_rate * profit_per_trade_with_advantage
)

colo_annual_cost = colocation_cost_per_month * 12
net_benefit = colo_annual_profit - remote_annual_profit - colo_annual_cost

print(f"\nAssumptions:")
print(f"   Trades per Day: {trades_per_day:,}")
print(f"   Profit per Trade: ${profit_per_trade_with_advantage:.2f}")
print(f"   Co-location Cost: ${colocation_cost_per_month:,}/month")

print(f"\nWithout Co-location:")
print(f"   Latency: {remote_latency_ms}ms")
print(f"   Win Rate: {remote_win_rate*100:.0f}%")
print(f"   Annual Profit: ${remote_annual_profit:,.0f}")

print(f"\nWith Co-location:")
print(f"   Latency: {colo_latency_us}μs")
print(f"   Win Rate: {colo_win_rate*100:.0f}%")
print(f"   Annual Profit: ${colo_annual_profit:,.0f}")
print(f"   Annual Cost: ${colo_annual_cost:,.0f}")

print(f"\n** NET BENEFIT: ${net_benefit:,.0f}/year **")
print(f"   ROI: {(net_benefit / colo_annual_cost)*100:.0f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Win rates by trader type
ax1 = axes[0, 0]
trader_names = list(results.keys())
win_percentages = [(results[t]['wins'] / results[t]['attempts']) * 100 
                  for t in trader_names]
colors = ['green', 'yellow', 'orange', 'red']
bars = ax1.bar(trader_names, win_percentages, color=colors, edgecolor='black', alpha=0.7)
ax1.set_ylabel('Win Rate (%)')
ax1.set_title('Arbitrage Win Rate by Latency Profile')
ax1.set_xticklabels(trader_names, rotation=15, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Add latency labels
for i, (bar, trader) in enumerate(zip(bars, trader_names)):
    height = bar.get_height()
    latency = traders[trader]
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{latency}μs', ha='center', va='bottom', fontsize=9)

# Plot 2: Latency sensitivity curve
ax2 = axes[0, 1]
ax2.plot(latency_range_us / 1000, win_rates, 'o-', linewidth=3, markersize=8, color='blue')
ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% (always win)')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='0% (never win)')
ax2.axvline(x=opportunity_duration_us / 1000, color='orange', linestyle='--', 
           linewidth=2, label=f'Opportunity Duration ({opportunity_duration_us}μs)')
ax2.set_xlabel('Latency (ms)')
ax2.set_ylabel('Win Rate (%)')
ax2.set_title('Latency Sensitivity: Win Rate vs Speed')
ax2.set_xscale('log')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Network technology comparison
ax3 = axes[1, 0]
ax3.plot(distances_km, fiber_latencies_ms, 'o-', linewidth=2, markersize=8, 
        label='Fiber Optic', color='blue')
ax3.plot(distances_km, microwave_latencies_ms, 's-', linewidth=2, markersize=8, 
        label='Microwave', color='red')
ax3.fill_between(distances_km, microwave_latencies_ms, fiber_latencies_ms, 
                alpha=0.2, color='green', label='Microwave Advantage')
ax3.set_xlabel('Distance (km)')
ax3.set_ylabel('Round-Trip Latency (ms)')
ax3.set_title('Network Technology: Fiber vs Microwave')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Co-location ROI
ax4 = axes[1, 1]
scenarios = ['Remote\nServer', 'Co-location\n(Gross)', 'Co-location\n(Net)']
profits = [remote_annual_profit / 1000, colo_annual_profit / 1000, 
          (colo_annual_profit - colo_annual_cost) / 1000]
colors_roi = ['red', 'lightgreen', 'darkgreen']
bars_roi = ax4.bar(scenarios, profits, color=colors_roi, edgecolor='black', alpha=0.7)
ax4.set_ylabel('Annual Profit ($thousands)')
ax4.set_title('Co-location ROI Analysis')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, profit in zip(bars_roi, profits):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'${profit:.0f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add cost line
ax4.axhline(y=colo_annual_cost / 1000, color='red', linestyle='--', 
           linewidth=2, label=f'Colo Cost: ${colo_annual_cost/1000:.0f}K')
ax4.legend(fontsize=9)

plt.tight_layout()
plt.savefig('latency_speed_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: latency_speed_analysis.png")
plt.show()