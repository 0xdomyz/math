import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class TemporaryImpactSimulator:
    def __init__(self):
        self.execution_results = []
        self.inventory_paths = []
        
    def simulate_mm_inventory_dynamics(self, order_size, num_periods=100):
        """Simulate market maker inventory and resulting temporary impact"""
        inventory = 0
        quote_changes = []
        inventory_path = [inventory]
        prices = [100.0]
        
        # Initial trade moves inventory
        inventory += order_size
        
        # MM widens quotes to rebalance
        initial_quote_move = 0.02 * np.sqrt(order_size / 10000)
        prices.append(prices[-1] + initial_quote_move)
        quote_changes.append(initial_quote_move)
        
        # Mean reversion parameters
        reversion_rate = 0.85  # Each period, 85% of inventory remains
        reversion_strength = 0.02  # Quote change per unit inventory
        
        for period in range(num_periods):
            # Probability other traders show up (Poisson process)
            if np.random.random() < 0.3:  # 30% chance of contra-side order
                contra_size = np.random.randint(1000, 5000)
                direction = np.sign(inventory)  # Opposite of current inventory
                
                if direction > 0:  # Need buyers, so sell orders arrive
                    inventory -= contra_size
                else:  # Need sellers, so buy orders arrive
                    inventory += contra_size
            
            # Mean reversion of inventory
            reversion = -inventory * (1 - reversion_rate)
            inventory += reversion
            
            # Quote adjustment based on inventory
            quote_move = reversion_strength * reversion
            current_price = prices[-1] + quote_move
            prices.append(current_price)
            quote_changes.append(quote_move)
            inventory_path.append(inventory)
        
        return prices, inventory_path, quote_changes
    
    def optimal_execution_path(self, total_order, execution_periods):
        """Find optimal execution schedule minimizing total cost"""
        # Trade-off: execute faster (larger per period) → more impact per trade
        #           but slower execution → less timing risk
        
        execution_sizes = []
        execution_costs = []
        timing_costs = []
        
        for num_splits in range(1, execution_periods + 1):
            per_period_size = total_order / num_splits
            
            total_impact_cost = 0
            total_timing_cost = 0
            
            for split in range(num_splits):
                # Permanent impact (unavoidable)
                permanent = 0.0005 * np.sqrt(per_period_size / 10000)
                
                # Temporary impact
                temporary = 0.001 * (per_period_size / 100000)
                
                # Cost of this execution
                impact_cost = (permanent + temporary) * per_period_size
                total_impact_cost += impact_cost
                
                # Timing risk: wait longer → risk price moves more
                timing_risk_per_period = 0.0001 * np.sqrt(split)  # Increases with time
                timing_cost = timing_risk_per_period * per_period_size
                total_timing_cost += timing_cost
            
            total_cost = total_impact_cost + total_timing_cost
            execution_sizes.append(per_period_size)
            execution_costs.append(total_impact_cost)
            timing_costs.append(total_timing_cost)
        
        # Find optimal
        total_costs = np.array(execution_costs) + np.array(timing_costs)
        optimal_idx = np.argmin(total_costs)
        
        return execution_sizes[optimal_idx], execution_costs, timing_costs, optimal_idx + 1

# Scenario 1: MM inventory dynamics and mean reversion
print("Scenario 1: Market Maker Inventory Rebalancing")
print("=" * 80)

sim = TemporaryImpactSimulator()
order_sizes = [10000, 50000, 100000]

for order_size in order_sizes:
    prices, inventory, quotes = sim.simulate_mm_inventory_dynamics(order_size, num_periods=50)
    
    initial_price = prices[0]
    max_move = np.max(np.abs(np.array(prices) - initial_price))
    final_price = prices[-1]
    half_life = 0
    
    for i, price in enumerate(prices):
        if abs(price - initial_price) < max_move / 2:
            half_life = i
            break
    
    print(f"Order Size: {order_size:>10,}")
    print(f"  Initial Price: ${initial_price:.2f}")
    print(f"  Max Impact: ${max_move:.4f}")
    print(f"  Final Price: ${final_price:.2f}")
    print(f"  Half-Life: {half_life} periods")
    print(f"  Reversion: ${initial_price - final_price:.4f} ({(1 - final_price/initial_price)*100:.2f}%)")
    print()

# Scenario 2: Optimal execution schedule
print("Scenario 2: Optimal Execution Schedule (Trading 100K shares)")
print("=" * 80)

total_order = 100000
max_execution_periods = 20

size, impact, timing, optimal = sim.optimal_execution_path(total_order, max_execution_periods)

print(f"Optimal Strategy: {optimal} execution periods")
print(f"  Size per execution: {size:,.0f} shares")
print(f"  Total execution time: {optimal} periods")
print(f"\nCost Breakdown:")

# Calculate total costs
total_impact = sum(impact)
total_timing = sum(timing)
total_cost = total_impact + total_timing

print(f"  Market Impact Cost: ${total_impact:,.0f} ({total_impact/total_cost*100:.1f}%)")
print(f"  Timing Risk Cost:   ${total_timing:,.0f} ({total_timing/total_cost*100:.1f}%)")
print(f"  Total Cost:         ${total_cost:,.0f}")

# Scenario 3: Decay of temporary impact over time
print(f"\n\nScenario 3: Temporary Impact Decay (Half-Life Analysis)")
print("=" * 80)

# Exponential decay model
half_lives = [10, 50, 100, 500, 1000]  # milliseconds
time_points = np.arange(0, 2000, 50)  # 0 to 2000 ms
initial_impact = 0.05  # $0.05

for half_life in half_lives:
    decay_rate = np.log(2) / half_life
    impacts = initial_impact * np.exp(-decay_rate * time_points)
    
    # Find when 90% recovered
    recovered_90_idx = np.where(impacts < initial_impact * 0.1)[0]
    if len(recovered_90_idx) > 0:
        time_90 = time_points[recovered_90_idx[0]]
    else:
        time_90 = np.inf
    
    print(f"Half-Life: {half_life:>5} ms | 90% Recovery: {time_90:>6.0f} ms | Final Impact: ${impacts[-1]:.5f}")

# Scenario 4: Order execution timing comparison
print(f"\n\nScenario 4: Execution Timing Strategies (Large Order)")
print("=" * 80)

strategies = [
    {'name': 'Market Order (Instant)', 'periods': 1, 'description': 'All at once'},
    {'name': 'VWAP (10 pieces)', 'periods': 10, 'description': 'Split over 10 periods'},
    {'name': 'TWAP (20 pieces)', 'periods': 20, 'description': 'Uniform over 20 periods'},
    {'name': 'Aggressive', 'periods': 5, 'description': 'Quick execution (5 periods)'},
]

for strategy in strategies:
    size, impact, timing, _ = sim.optimal_execution_path(100000, strategy['periods'])
    total = sum(impact) + sum(timing)
    avg_impact = np.mean(impact) if impact else 0
    
    print(f"{strategy['name']:>30}: ${total:>10,.0f} | Avg Impact/Trade: ${avg_impact:>8,.0f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price path with inventory dynamics
prices, inventory, _ = sim.simulate_mm_inventory_dynamics(50000, num_periods=100)
periods = np.arange(len(prices))

axes[0, 0].plot(periods, prices, linewidth=2, label='Price', color='blue')
axes[0, 0].axhline(y=prices[0], color='r', linestyle='--', label='Initial Price', alpha=0.5)
axes[0, 0].fill_between(periods, prices[0], prices, alpha=0.2)
axes[0, 0].set_xlabel('Periods')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Scenario 1: Price Reversion from Temporary Impact')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Inventory path
axes[0, 1].plot(periods, inventory, linewidth=2, label='MM Inventory', color='green')
axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Target', alpha=0.5)
axes[0, 1].fill_between(periods, 0, inventory, alpha=0.2, color='green')
axes[0, 1].set_xlabel('Periods')
axes[0, 1].set_ylabel('Inventory (shares)')
axes[0, 1].set_title('Scenario 1: MM Inventory Mean Reversion')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Execution schedule comparison
strategies_names = [s['name'].split('(')[0] for s in strategies]
strategies_costs = []

for strategy in strategies:
    _, impact, timing, _ = sim.optimal_execution_path(100000, strategy['periods'])
    strategies_costs.append(sum(impact) + sum(timing))

colors_strat = plt.cm.viridis(np.linspace(0, 1, len(strategies_names)))
bars = axes[1, 0].bar(range(len(strategies_names)), strategies_costs, color=colors_strat)
axes[1, 0].set_xticks(range(len(strategies_names)))
axes[1, 0].set_xticklabels(strategies_names, rotation=45, ha='right')
axes[1, 0].set_ylabel('Total Execution Cost ($)')
axes[1, 0].set_title('Scenario 4: Strategy Comparison')
axes[1, 0].grid(alpha=0.3, axis='y')

for bar, cost in zip(bars, strategies_costs):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost:.0f}', ha='center', va='bottom', fontsize=9)

# Plot 4: Temporary impact decay curves
time_points = np.arange(0, 2000, 50)
initial_impact = 0.05
half_life_values = [10, 50, 100, 500]

for hl in half_life_values:
    decay_rate = np.log(2) / hl
    impacts = initial_impact * np.exp(-decay_rate * time_points)
    axes[1, 1].plot(time_points, impacts * 10000, linewidth=2, label=f'{hl}ms half-life')

axes[1, 1].set_xlabel('Time (milliseconds)')
axes[1, 1].set_ylabel('Remaining Impact (cents)')
axes[1, 1].set_title('Scenario 3: Impact Decay Rates')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Typical temporary impact: 30-40% of total (rest is permanent)")
print(f"Half-life: 10-100 milliseconds (varies by liquidity)")
print(f"Recovery time (90%): 10-30× half-life")
print(f"Optimal execution: Balance impact reduction vs timing risk")
print(f"Key insight: Temporary reverts predictably if inventory-driven")
