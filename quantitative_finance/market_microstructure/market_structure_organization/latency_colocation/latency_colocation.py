import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class LatencyArbitrageSimulator:
    def __init__(self, num_venues=2):
        self.num_venues = num_venues
        self.prices = [100.0] * num_venues  # Initial prices
        self.latencies = []  # ms for each participant
        self.arbitrage_opportunities = []
        self.profits_captured = []
        
    def generate_price_move(self):
        """Random price move on venue 0 (leading)"""
        move = np.random.normal(0, 0.1)  # Random walk
        return move
    
    def simulate_opportunity(self, trader_latency_ms):
        """Check if trader can arbitrage based on latency"""
        # Price moves on Venue 0
        move = self.generate_price_move()
        self.prices[0] += move
        
        # Time for news to reach other venues
        news_propagation_time = 0.5  # 0.5ms news reaches other venues
        
        # Does trader capture the arbitrage before it closes?
        spread = abs(self.prices[0] - self.prices[1])
        
        # Trader needs to:
        # 1. See price move (trader_latency_ms)
        # 2. Decide and transmit order (0.1ms)
        # 3. Exchange processes order (0.1ms)
        total_trader_time = trader_latency_ms + 0.2
        
        # If trader faster than spread closes, captures profit
        if total_trader_time < news_propagation_time + 0.1:
            profit = spread / 2  # Gets half the spread
            return profit, spread
        else:
            return 0, spread
    
    def arms_race_simulation(self, years=5):
        """Simulate arms race where each firm improves latency"""
        initial_latency = 10.0  # 10ms (starting point)
        latency_improvement_per_year = 0.5  # 50% improvement per year
        cost_per_year = 5.0  # $5M per year
        profit_per_millisecond = 0.5  # $500K per millisecond advantage
        
        firms = [{'latency': initial_latency, 'profit': 0, 'cost': 0} 
                for _ in range(5)]  # 5 competing firms
        
        year_data = {'latency': [], 'profits': [], 'costs': []}
        
        for year in range(years):
            # Each firm improves to capture more
            for firm in firms:
                # Improvement: lower latency by factor
                firm['latency'] *= latency_improvement_per_year
                
                # Profit based on speed advantage
                # When all equal speed, profit decreases
                avg_latency = np.mean([f['latency'] for f in firms])
                speed_advantage = (avg_latency - firm['latency']) / avg_latency
                firm['profit'] = max(0, speed_advantage * 10)  # $ millions
                
                # Cost of technology
                firm['cost'] = cost_per_year
            
            year_data['latency'].append(np.mean([f['latency'] for f in firms]))
            year_data['profits'].append(np.mean([f['profit'] for f in firms]))
            year_data['costs'].append(np.mean([f['cost'] for f in firms]))
        
        return firms, year_data

# Scenario 1: Single latency advantage
print("Scenario 1: Latency Advantage Persistence")
print("=" * 80)

sim = LatencyArbitrageSimulator(num_venues=2)
latencies = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]  # milliseconds
profits = []

for latency in latencies:
    total_profit = 0
    opportunities = 0
    
    for _ in range(1000):
        profit, spread = sim.simulate_opportunity(latency)
        total_profit += profit
        if profit > 0:
            opportunities += 1
    
    avg_profit_per_trade = total_profit / 1000 if total_profit > 0 else 0
    profits.append(total_profit)
    
    print(f"Latency: {latency:>6.2f} ms | Total Profit: ${total_profit:>10,.0f} | Opportunities: {opportunities:>3} ({opportunities/10:.1f}%)")

# Scenario 2: Colocation vs Non-collocated
print(f"\n\nScenario 2: Colocation Value Over Time")
print("=" * 80)

# Collocated trader (0.5ms latency)
collocated_latency = 0.5
non_collocated_latency = 5.0
colocation_cost_monthly = 2.0  # $2,000/month = $24K/year

collocated_pnl = []
non_collocated_pnl = []
net_benefit = []

sim = LatencyArbitrageSimulator(num_venues=2)

for day in range(252):  # Trading year
    daily_profit_collocated = 0
    daily_profit_non_collocated = 0
    
    for _ in range(1000):  # 1000 opportunities per day
        profit_col, _ = sim.simulate_opportunity(collocated_latency)
        profit_non_col, _ = sim.simulate_opportunity(non_collocated_latency)
        
        daily_profit_collocated += profit_col
        daily_profit_non_collocated += profit_non_col
    
    collocated_pnl.append(daily_profit_collocated)
    non_collocated_pnl.append(daily_profit_non_collocated)
    net_benefit.append(daily_profit_collocated - daily_profit_non_collocated - colocation_cost_monthly)

cumulative_collocated = np.cumsum(collocated_pnl)
cumulative_non_collocated = np.cumsum(non_collocated_pnl)
cumulative_net_benefit = np.cumsum(net_benefit)

print(f"Annual Collocated PnL:      ${cumulative_collocated[-1]:>12,.0f}")
print(f"Annual Non-Collocated PnL:  ${cumulative_non_collocated[-1]:>12,.0f}")
print(f"Advantage (before cost):    ${cumulative_collocated[-1] - cumulative_non_collocated[-1]:>12,.0f}")
print(f"Colocation Cost (annual):   ${colocation_cost_monthly * 12:>12,.0f}")
print(f"Net Colocation Benefit:     ${cumulative_net_benefit[-1]:>12,.0f}")

# Scenario 3: Arms race dynamics
print(f"\n\nScenario 3: Latency Arms Race (5-Year Horizon)")
print("=" * 80)

sim = LatencyArbitrageSimulator()
firms, year_data = sim.arms_race_simulation(years=5)

for year in range(5):
    print(f"Year {year+1}:")
    print(f"  Avg Latency: {year_data['latency'][year]:.3f} ms")
    print(f"  Avg Profit:  ${year_data['profits'][year]:.2f}M")
    print(f"  Avg Cost:    ${year_data['costs'][year]:.2f}M")
    print(f"  Net Profit:  ${year_data['profits'][year] - year_data['costs'][year]:.2f}M")
    print()

# Scenario 4: Market impact of latency divergence
print(f"\n\nScenario 4: Spread Evolution with Latency Divergence")
print("=" * 80)

traders = [
    {'name': 'HFT Collocated', 'latency': 0.1, 'count': 5},
    {'name': 'Fast Traders', 'latency': 1.0, 'count': 20},
    {'name': 'Normal Traders', 'latency': 10.0, 'count': 100},
    {'name': 'Retail (Internet)', 'latency': 100.0, 'count': 1000},
]

sim = LatencyArbitrageSimulator(num_venues=2)
execution_times = []
for trader in traders:
    for _ in range(trader['count']):
        # Random order within trader class latency
        execution_time = trader['latency'] * np.random.uniform(0.5, 1.5)
        execution_times.append(execution_time)

print(f"Execution Time Distribution:")
print(f"  Min: {np.min(execution_times):.2f} ms (fastest HFT)")
print(f"  25%: {np.percentile(execution_times, 25):.2f} ms")
print(f"  Median: {np.median(execution_times):.2f} ms")
print(f"  75%: {np.percentile(execution_times, 75):.2f} ms")
print(f"  Max: {np.max(execution_times):.2f} ms (slowest retail)")
print(f"  Range: {np.max(execution_times) - np.min(execution_times):.2f} ms")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Profit vs Latency
axes[0, 0].plot(latencies, profits, 'o-', linewidth=2, markersize=8, color='blue')
axes[0, 0].set_xlabel('Latency (milliseconds)')
axes[0, 0].set_ylabel('Annual Profit ($)')
axes[0, 0].set_title('Scenario 1: Latency Advantage Value')
axes[0, 0].set_xscale('log')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Colocation benefit over time
days = np.arange(len(cumulative_collocated))
axes[0, 1].plot(days, cumulative_collocated / 1000, label='Collocated', linewidth=2)
axes[0, 1].plot(days, cumulative_non_collocated / 1000, label='Non-collocated', linewidth=2)
axes[0, 1].plot(days, cumulative_net_benefit / 1000, label='Net Benefit (after cost)', linewidth=2, linestyle='--')
axes[0, 1].set_xlabel('Trading Days')
axes[0, 1].set_ylabel('Cumulative Profit ($1000s)')
axes[0, 1].set_title('Scenario 2: Colocation Value Over Year')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Arms race dynamics
years = np.arange(1, 6)
axes[1, 0].plot(years, year_data['latency'], 'o-', linewidth=2, markersize=8, label='Avg Latency (ms)')
ax2 = axes[1, 0].twinx()
ax2.plot(years, year_data['profits'], 's-', linewidth=2, markersize=8, color='orange', label='Avg Profit ($M)')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Latency (ms)', color='blue')
ax2.set_ylabel('Profit ($M)', color='orange')
axes[1, 0].set_title('Scenario 3: Arms Race Dynamics')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Execution time distribution
trader_names = [t['name'] for t in traders for _ in range(t['count'])]
trader_latencies = [t['latency'] for t in traders for _ in range(t['count'])]

axes[1, 1].scatter(trader_latencies, execution_times, alpha=0.3, s=50)
axes[1, 1].set_xlabel('Trader Class Latency (ms)')
axes[1, 1].set_ylabel('Actual Execution Time (ms)')
axes[1, 1].set_title('Scenario 4: Execution Time Distribution')
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(alpha=0.3)

# Add trader class annotations
for trader in traders:
    avg_latency = np.mean([t for t, name in zip(execution_times, trader_names) if name == trader['name']])
    axes[1, 1].annotate(trader['name'], xy=(trader['latency'], avg_latency), 
                       xytext=(10, 10), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Latency advantage value: {(profits[-1] - profits[0]) / profits[0] * 100:.0f}% improvement (0.5ms vs 10ms)")
print(f"Colocation ROI: {cumulative_net_benefit[-1] / (colocation_cost_monthly * 12) * 100:.0f}%")
print(f"Arms race outcome: Latency fell {year_data['latency'][0] / year_data['latency'][-1]:.1f}x, profits fell {year_data['profits'][0] / year_data['profits'][-1]:.1f}x")
print(f"Fastest traders advantage: {(np.max(execution_times) - np.min(execution_times)):.2f}ms")
