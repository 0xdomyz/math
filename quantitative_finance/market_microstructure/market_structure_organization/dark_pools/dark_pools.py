import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class MarketImpactSimulator:
    def __init__(self):
        self.trades = []
        self.prices_public = []
        self.prices_dark = []
        self.impact_public = []
        self.impact_dark = []
        
    def execute_public_exchange(self, order_size, nbbo=100.0):
        """Execute large order on public exchange (visible, causes impact)"""
        initial_price = nbbo
        
        # Market impact model: ΔP = λ * sqrt(Q/V)
        # Larger orders cause larger price moves
        price_impact = 0.001 * np.sqrt(order_size / 100000)
        
        # Price moves against order
        execution_price = initial_price + price_impact
        
        # Permanent impact: Price moves and stays
        final_price = initial_price + price_impact * 0.8  # 80% permanent
        
        # Temporary impact: Price rebounds slightly
        rebound_price = initial_price + price_impact * 0.3
        
        self.prices_public.append(execution_price)
        self.impact_public.append(execution_price - initial_price)
        
        return {
            'execution': execution_price,
            'final': final_price,
            'rebound': rebound_price,
            'total_impact': execution_price - initial_price,
            'permanent_impact': final_price - initial_price
        }
    
    def execute_dark_pool(self, order_size, nbbo=100.0):
        """Execute through dark pool (hidden, minimal impact)"""
        # Dark pool trades at NBBO or slight improvement
        # No market impact because hidden
        
        execution_price = nbbo - 0.001  # Slight improvement from NBBO
        final_price = nbbo  # No permanent impact
        
        # Small chance order doesn't fill (illiquid pool)
        fill_probability = max(0.3, 1 - order_size / 500000)
        
        if np.random.random() > fill_probability:
            # Partial fill or rejection
            filled_size = order_size * fill_probability
            # Must execute remainder on public exchange (worse)
            remainder_impact = self.execute_public_exchange(order_size - filled_size, nbbo)
            execution_price = (execution_price * filled_size + remainder_impact['execution'] * (order_size - filled_size)) / order_size
        
        self.prices_dark.append(execution_price)
        self.impact_dark.append(execution_price - nbbo)
        
        return {
            'execution': execution_price,
            'final': final_price,
            'rebound': final_price,
            'total_impact': execution_price - nbbo,
            'permanent_impact': final_price - nbbo
        }

# Scenario 1: Varying order sizes
print("Scenario 1: Impact of Order Size")
print("=" * 80)

sim = MarketImpactSimulator()
order_sizes = [10000, 50000, 100000, 250000, 500000, 1000000]

public_impacts = []
dark_impacts = []

for size in order_sizes:
    public = sim.execute_public_exchange(size, nbbo=100.0)
    dark = sim.execute_dark_pool(size, nbbo=100.0)
    
    public_impacts.append(abs(public['total_impact']))
    dark_impacts.append(abs(dark['total_impact']))
    
    print(f"Order Size: {size:>10,} shares")
    print(f"  Public Exchange Impact: {public['total_impact']:>8.4f} ({public['total_impact']*100/100:.2f}%)")
    print(f"  Dark Pool Impact:       {dark['total_impact']:>8.4f} ({dark['total_impact']*100/100:.2f}%)")
    print(f"  Savings:                ${abs(public['total_impact'] - dark['total_impact']) * size:>15,.0f}")
    print()

# Scenario 2: Slicing strategy
print("Scenario 2: Slicing Large Order (100K shares over 5 periods)")
print("=" * 80)

nbbo = 100.0
total_order = 100000
num_pieces = 5

print("Strategy 1: Execute entire order immediately (public exchange)")
immediate_result = sim.execute_public_exchange(total_order, nbbo)
immediate_cost = immediate_result['total_impact'] * total_order
print(f"  Execution Price: ${immediate_result['execution']:.4f}")
print(f"  Total Cost: ${immediate_cost:>15,.0f}")

print("\nStrategy 2: Slice into 5 pieces, execute through dark pool")
slice_size = total_order // num_pieces
slice_prices = []
total_dark_cost = 0

for i in range(num_pieces):
    result = sim.execute_dark_pool(slice_size, nbbo)
    slice_prices.append(result['execution'])
    total_dark_cost += result['total_impact'] * slice_size
    nbbo = result['final']  # Update reference price

avg_slice_price = np.mean(slice_prices)
print(f"  Average Execution Price: ${avg_slice_price:.4f}")
print(f"  Total Cost: ${total_dark_cost:>15,.0f}")
print(f"  Savings vs Immediate: ${immediate_cost - total_dark_cost:>15,.0f}")

# Scenario 3: Market impact recovery
print(f"\n\nScenario 3: Price Recovery After Large Order")
print("=" * 80)

nbbo = 100.0
large_order = 500000

public = sim.execute_public_exchange(large_order, nbbo)
print(f"Initial NBBO: ${nbbo:.4f}")
print(f"Execution Price: ${public['execution']:.4f}")
print(f"Immediate Impact: ${public['total_impact']:.4f} ({public['total_impact']*100:.2f}%)")
print(f"Price After Recovery: ${public['rebound']:.4f}")
print(f"Permanent Impact: ${public['permanent_impact']:.4f} ({public['permanent_impact']*100:.2f}%)")
print(f"Temporary Impact: ${public['total_impact'] - public['permanent_impact']:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Market impact vs order size
axes[0, 0].plot(order_sizes, [p*10000 for p in public_impacts], 'o-', linewidth=2, markersize=8, label='Public Exchange')
axes[0, 0].plot(order_sizes, [d*10000 for d in dark_impacts], 's-', linewidth=2, markersize=8, label='Dark Pool')
axes[0, 0].set_xlabel('Order Size (shares)')
axes[0, 0].set_ylabel('Price Impact (cents)')
axes[0, 0].set_title('Scenario 1: Market Impact Comparison')
axes[0, 0].set_xscale('log')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Execution cost comparison
cost_savings = [(p - d) * s for p, d, s in zip(public_impacts, dark_impacts, order_sizes)]
colors_cost = ['green' if s > 0 else 'red' for s in cost_savings]

axes[0, 1].bar(range(len(order_sizes)), [s/1000 for s in cost_savings], color=colors_cost, alpha=0.7)
axes[0, 1].set_xticks(range(len(order_sizes)))
axes[0, 1].set_xticklabels([f'{s/1000:.0f}K' for s in order_sizes], rotation=45)
axes[0, 1].set_ylabel('Cost Savings ($1000s)')
axes[0, 1].set_title('Scenario 1: Dark Pool Savings by Order Size')
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Slicing effectiveness
strategies = ['Immediate\nPublic', 'Sliced\nDark Pool']
costs = [immediate_cost/1000, total_dark_cost/1000]
colors_strat = ['red', 'green']

bars = axes[1, 0].bar(strategies, costs, color=colors_strat, alpha=0.7)
axes[1, 0].set_ylabel('Execution Cost ($1000s)')
axes[1, 0].set_title('Scenario 2: Slicing Strategy Comparison')
axes[1, 0].grid(alpha=0.3, axis='y')

for bar, cost in zip(bars, costs):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost:.1f}K', ha='center', va='bottom', fontweight='bold')

# Plot 4: Impact decomposition (permanent vs temporary)
impact_types = ['Immediate', 'Permanent', 'Temporary']
impact_values = [public['total_impact']*100, public['permanent_impact']*100, 
                (public['total_impact'] - public['permanent_impact'])*100]
colors_impact = ['blue', 'red', 'orange']

bars = axes[1, 1].bar(impact_types, impact_values, color=colors_impact, alpha=0.7)
axes[1, 1].set_ylabel('Price Impact (basis points)')
axes[1, 1].set_title('Scenario 3: Impact Decomposition (500K share order)')
axes[1, 1].grid(alpha=0.3, axis='y')

for bar, impact in zip(bars, impact_values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{impact:.1f}bps', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Average Impact (Public): {np.mean(public_impacts)*10000:.2f} cents per $10K traded")
print(f"Average Impact (Dark):   {np.mean(dark_impacts)*10000:.2f} cents per $10K traded")
print(f"Avg Savings:             {(np.mean(public_impacts) - np.mean(dark_impacts))*10000:.2f} cents per $10K traded")
print(f"Total Savings (100K order): ${(immediate_cost - total_dark_cost):,.0f}")
