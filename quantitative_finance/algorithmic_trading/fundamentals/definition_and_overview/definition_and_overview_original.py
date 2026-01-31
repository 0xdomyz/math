import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

np.random.seed(42)

# ============================================================================
# MARKET SIMULATOR
# ============================================================================

@dataclass
class MarketState:
    """Snapshot of market at specific time."""
    time: int
    mid_price: float
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    volume: int  # Recent period volume

class MarketSimulator:
    """Realistic market with spread, impact, volume patterns."""
    
    def __init__(self, initial_price=100.0, n_periods=100):
        self.initial_price = initial_price
        self.n_periods = n_periods
        self.current_period = 0
        
        # Generate price path (GBM)
        returns = np.random.normal(0, 0.001, n_periods)
        self.prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate spread (mean-reverting, widens with volatility)
        self.spreads = 0.05 + 0.02 * np.abs(np.random.normal(0, 1, n_periods))
        
        # Generate volume (U-shaped intraday pattern)
        time_factor = np.linspace(0, 2*np.pi, n_periods)
        base_volume = 1000 * (1 + 0.5*np.cos(time_factor - np.pi))
        self.volumes = base_volume * (1 + 0.3*np.random.randn(n_periods))
        self.volumes = np.maximum(self.volumes, 100)
    
    def get_market_state(self) -> MarketState:
        """Return current market snapshot."""
        if self.current_period >= self.n_periods:
            return None
        
        mid = self.prices[self.current_period]
        spread = self.spreads[self.current_period]
        
        return MarketState(
            time=self.current_period,
            mid_price=mid,
            bid_price=mid - spread/2,
            ask_price=mid + spread/2,
            bid_size=int(self.volumes[self.current_period] * 0.1),
            ask_size=int(self.volumes[self.current_period] * 0.1),
            volume=int(self.volumes[self.current_period])
        )
    
    def execute_order(self, quantity: int, side: str) -> Tuple[float, float]:
        """
        Execute market order, return (avg_price, market_impact_bps).
        side: 'buy' or 'sell'
        """
        state = self.get_market_state()
        
        # Base price (cross spread)
        if side == 'buy':
            base_price = state.ask_price
        else:
            base_price = state.bid_price
        
        # Market impact: proportional to (quantity / volume)^0.5
        participation = quantity / state.volume if state.volume > 0 else 0
        impact_bps = 50 * np.sqrt(participation)  # Square root impact
        
        if side == 'buy':
            execution_price = base_price * (1 + impact_bps / 10000)
        else:
            execution_price = base_price * (1 - impact_bps / 10000)
        
        return execution_price, impact_bps
    
    def advance(self):
        """Move to next time period."""
        self.current_period += 1

# ============================================================================
# EXECUTION STRATEGIES
# ============================================================================

class ExecutionStrategy:
    """Base class for execution algorithms."""
    
    def execute(self, market: MarketSimulator, total_qty: int, side: str) -> dict:
        raise NotImplementedError

class AggressiveExecution(ExecutionStrategy):
    """Execute entire order immediately (market order)."""
    
    def execute(self, market: MarketSimulator, total_qty: int, side: str) -> dict:
        arrival_state = market.get_market_state()
        arrival_price = arrival_state.mid_price
        
        exec_price, impact = market.execute_order(total_qty, side)
        market.advance()
        
        slippage_bps = (exec_price - arrival_price) / arrival_price * 10000
        if side == 'sell':
            slippage_bps = -slippage_bps
        
        return {
            'strategy': 'Aggressive',
            'avg_price': exec_price,
            'arrival_price': arrival_price,
            'slippage_bps': slippage_bps,
            'num_periods': 1,
            'completion': 1.0
        }

class TWAPExecution(ExecutionStrategy):
    """Time-Weighted Average Price: uniform slicing."""
    
    def __init__(self, n_slices=10):
        self.n_slices = n_slices
    
    def execute(self, market: MarketSimulator, total_qty: int, side: str) -> dict:
        arrival_state = market.get_market_state()
        arrival_price = arrival_state.mid_price
        
        slice_size = total_qty / self.n_slices
        execution_prices = []
        
        for _ in range(self.n_slices):
            if market.get_market_state() is None:
                break
            
            exec_price, _ = market.execute_order(int(slice_size), side)
            execution_prices.append(exec_price)
            market.advance()
        
        avg_price = np.mean(execution_prices)
        slippage_bps = (avg_price - arrival_price) / arrival_price * 10000
        if side == 'sell':
            slippage_bps = -slippage_bps
        
        return {
            'strategy': f'TWAP ({self.n_slices} slices)',
            'avg_price': avg_price,
            'arrival_price': arrival_price,
            'slippage_bps': slippage_bps,
            'num_periods': self.n_slices,
            'completion': len(execution_prices) / self.n_slices
        }

class VWAPExecution(ExecutionStrategy):
    """Volume-Weighted Average Price: trade proportional to volume."""
    
    def __init__(self, duration=10):
        self.duration = duration
    
    def execute(self, market: MarketSimulator, total_qty: int, side: str) -> dict:
        arrival_state = market.get_market_state()
        arrival_price = arrival_state.mid_price
        
        # Forecast volume for scheduling
        volume_forecast = []
        start_period = market.current_period
        for i in range(self.duration):
            idx = start_period + i
            if idx < market.n_periods:
                volume_forecast.append(market.volumes[idx])
            else:
                volume_forecast.append(market.volumes[-1])
        
        total_forecast = sum(volume_forecast)
        
        # Execute proportional to volume
        execution_prices = []
        quantities = []
        
        for vol_forecast in volume_forecast:
            if market.get_market_state() is None:
                break
            
            slice_qty = int(total_qty * (vol_forecast / total_forecast))
            if slice_qty > 0:
                exec_price, _ = market.execute_order(slice_qty, side)
                execution_prices.append(exec_price)
                quantities.append(slice_qty)
            
            market.advance()
        
        avg_price = np.average(execution_prices, weights=quantities)
        slippage_bps = (avg_price - arrival_price) / arrival_price * 10000
        if side == 'sell':
            slippage_bps = -slippage_bps
        
        return {
            'strategy': 'VWAP',
            'avg_price': avg_price,
            'arrival_price': arrival_price,
            'slippage_bps': slippage_bps,
            'num_periods': len(execution_prices),
            'completion': sum(quantities) / total_qty
        }

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