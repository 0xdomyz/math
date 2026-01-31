import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

np.random.seed(42)

# ============================================================================
# MARKET SIMULATOR
# ============================================================================

@dataclass

class POVExecution(ExecutionStrategy):
    """Percent of Volume: target participation rate."""
    
    def __init__(self, target_rate=0.1, duration=10):
        self.target_rate = target_rate
        self.duration = duration
    
    def execute(self, market: MarketSimulator, total_qty: int, side: str) -> dict:
        arrival_state = market.get_market_state()
        arrival_price = arrival_state.mid_price
        
        execution_prices = []
        quantities = []
        remaining = total_qty
        
        for _ in range(self.duration):
            state = market.get_market_state()
            if state is None or remaining <= 0:
                break
            
            # Trade target % of volume
            slice_qty = int(min(self.target_rate * state.volume, remaining))
            
            if slice_qty > 0:
                exec_price, _ = market.execute_order(slice_qty, side)
                execution_prices.append(exec_price)
                quantities.append(slice_qty)
                remaining -= slice_qty
            
            market.advance()
        
        if len(execution_prices) > 0:
            avg_price = np.average(execution_prices, weights=quantities)
            slippage_bps = (avg_price - arrival_price) / arrival_price * 10000
            if side == 'sell':
                slippage_bps = -slippage_bps
        else:
            avg_price = arrival_price
            slippage_bps = 0
        
        return {
            'strategy': f'POV ({self.target_rate*100:.0f}%)',
            'avg_price': avg_price,
            'arrival_price': arrival_price,
            'slippage_bps': slippage_bps,
            'num_periods': len(execution_prices),
            'completion': sum(quantities) / total_qty if total_qty > 0 else 0
        }

# ============================================================================
# SIMULATION & COMPARISON
# ============================================================================

print("="*70)
print("ALGORITHMIC TRADING STRATEGY COMPARISON")
print("="*70)

total_quantity = 50000  # shares
side = 'buy'
n_simulations = 100

strategies = [
    AggressiveExecution(),
    TWAPExecution(n_slices=10),
    VWAPExecution(duration=10),
    POVExecution(target_rate=0.1, duration=10),
    POVExecution(target_rate=0.2, duration=10)
]

# Run Monte Carlo simulation
results = {s.__class__.__name__ + str(i): [] for i, s in enumerate(strategies)}

for sim in range(n_simulations):
    for i, strategy in enumerate(strategies):
        market = MarketSimulator(n_periods=100)
        result = strategy.execute(market, total_quantity, side)
        key = strategy.__class__.__name__ + str(i)
        results[key].append(result)

# Aggregate statistics
print(f"\nOrder: {side.upper()} {total_quantity:,} shares")
print(f"Simulations: {n_simulations}")
print("\n" + "="*70)

strategy_names = []
avg_slippages = []
std_slippages = []

for i, strategy in enumerate(strategies):
    key = strategy.__class__.__name__ + str(i)
    slippages = [r['slippage_bps'] for r in results[key]]
    completions = [r['completion'] for r in results[key]]
    
    avg_slip = np.mean(slippages)
    std_slip = np.std(slippages)
    avg_completion = np.mean(completions)
    
    strategy_name = results[key][0]['strategy']
    strategy_names.append(strategy_name)
    avg_slippages.append(avg_slip)
    std_slippages.append(std_slip)
    
    print(f"\n{strategy_name}:")
    print(f"   Avg Slippage: {avg_slip:.2f} ± {std_slip:.2f} bps")
    print(f"   Completion:   {avg_completion*100:.1f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Slippage comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(strategy_names))
bars = ax1.bar(x_pos, avg_slippages, yerr=std_slippages, 
              capsize=5, alpha=0.7, edgecolor='black')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(strategy_names, rotation=15, ha='right')
ax1.set_ylabel('Slippage (bps)')
ax1.set_title('Average Execution Slippage by Strategy')
ax1.grid(True, alpha=0.3, axis='y')

# Color bars
colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
for bar, color in zip(bars, colors):
    bar.set_color(color)

# Plot 2: Slippage distribution (box plot)
ax2 = axes[0, 1]
slippage_data = []
for i in range(len(strategies)):
    key = strategies[i].__class__.__name__ + str(i)
    slippage_data.append([r['slippage_bps'] for r in results[key]])

bp = ax2.boxplot(slippage_data, labels=[s[:10] for s in strategy_names], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Slippage (bps)')
ax2.set_title('Slippage Distribution')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Sample execution path
ax3 = axes[1, 0]
market_sample = MarketSimulator(n_periods=50)
time_axis = np.arange(50)
ax3.plot(time_axis, market_sample.prices, 'k-', linewidth=2, label='Market Price', alpha=0.7)

# Show TWAP execution points
market_twap = MarketSimulator(n_periods=50)
twap_strat = TWAPExecution(n_slices=10)
twap_result = twap_strat.execute(market_twap, total_quantity, 'buy')

exec_times = np.linspace(0, 9, 10)
exec_prices = []
for t in exec_times:
    idx = int(t)
    exec_prices.append(market_sample.prices[idx])

ax3.scatter(exec_times, exec_prices, c='blue', s=100, zorder=5, label='TWAP Executions')
ax3.axhline(y=twap_result['avg_price'], color='blue', linestyle='--', 
           linewidth=2, label=f'Avg: ${twap_result["avg_price"]:.2f}')
ax3.set_xlabel('Time Period')
ax3.set_ylabel('Price ($)')
ax3.set_title('Sample TWAP Execution Path')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Market volume pattern
ax4 = axes[1, 1]
sample_market = MarketSimulator(n_periods=100)
ax4.fill_between(time_axis[:100], 0, sample_market.volumes[:100], 
                alpha=0.3, color='gray', label='Market Volume')
ax4.plot(time_axis[:100], sample_market.volumes[:100], 'k-', linewidth=2)
ax4.set_xlabel('Time Period')
ax4.set_ylabel('Volume (shares)')
ax4.set_title('Intraday Volume Pattern (U-Shaped)')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9)

plt.tight_layout()
plt.savefig('algorithmic_trading_overview.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: algorithmic_trading_overview.png")
plt.show()