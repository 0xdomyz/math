# TWAP Algorithms: Time-Weighted Average Price Execution

## 1. Concept Skeleton
**Definition:** Time-weighted execution strategy splitting orders into equal slices distributed uniformly across trading period  
**Purpose:** Minimize market impact through temporal diversification; benchmark for execution quality  
**Prerequisites:** Order execution basics, market impact models, volume profiles

## 2. Comparative Framing
| Strategy | TWAP | VWAP | POV | Optimal (Almgren-Chriss) |
|----------|------|------|-----|--------------------------|
| **Time Allocation** | Uniform | Volume-weighted | Dynamic with volume | Risk-optimal |
| **Execution Logic** | Equal slices over time | Match intraday volume curve | Fixed % of volume | Minimize cost variance |
| **Market Impact** | Medium (predictable) | Lower (follows volume) | Variable (spikes with volume) | Lowest (theoretically) |
| **Complexity** | Low | Medium | Medium | High |

## 3. Examples + Counterexamples

**Simple Example:**  
Execute 100,000 shares over 60 minutes: TWAP submits 1,667 shares every minute (uniform slicing)

**Failure Case:**  
Market closes at 4:00 PM but TWAP scheduled through 4:30 PM → unfilled orders, missed execution

**Edge Case:**  
High volatility spike at 10:30 AM causes adverse price movement; TWAP unresponsive → accumulates slippage

## 4. Layer Breakdown
```
TWAP Execution Framework:
├─ Order Parameters:
│   ├─ Total Quantity: Q (shares to execute)
│   ├─ Duration: T (execution horizon in minutes)
│   ├─ Slice Interval: Δt (time between child orders)
│   └─ Child Order Size: q = Q / (T/Δt)
├─ Execution Loop:
│   ├─ Schedule: t₀, t₀+Δt, t₀+2Δt, ..., t₀+T
│   ├─ At each time tᵢ:
│   │   ├─ Submit child order of size q
│   │   ├─ Order type: Market or limit with tolerance
│   │   ├─ Fill monitoring: Track execution
│   │   └─ Adjust remaining: If partial fill, carry forward
│   └─ Final Adjustment: Submit remaining balance at T
├─ Performance Metrics:
│   ├─ Arrival Price: P₀ (benchmark at start)
│   ├─ Average Execution Price: P̄ = Σ(Pᵢ·qᵢ) / Q
│   ├─ TWAP Slippage: P̄ - P₀ (for buys; P₀ - P̄ for sells)
│   └─ Implementation Shortfall: (P̄ - P₀) / P₀
└─ Risk Factors:
    ├─ Timing Risk: Price moves against during execution
    ├─ Market Impact: Predictable slicing allows front-running
    ├─ Fill Risk: Limits may not execute in fast markets
    └─ Opportunity Cost: Delayed execution misses favorable moves
```

**Interaction:** Schedule slices → Execute uniformly → Measure slippage → Compare to benchmark

## 5. Mini-Project
Simulate TWAP execution with market impact and compare to other strategies:
```python
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate synthetic intraday price path
np.random.seed(42)

n_minutes = 60  # 1-hour execution window
initial_price = 100.0
volatility = 0.02  # 2% hourly vol

# Realistic intraday volume curve (U-shaped)
time_grid = np.arange(n_minutes)
volume_curve = 100 * (1 + 0.5*np.cos(2*np.pi*time_grid/n_minutes - np.pi))  # U-shape
volume_curve = volume_curve / volume_curve.sum()  # Normalize

# Price path with drift and noise
price_drift = 0.001  # 0.1% drift
price_changes = np.random.normal(price_drift, volatility/np.sqrt(n_minutes), n_minutes)
prices = initial_price * np.exp(np.cumsum(price_changes))

# Order parameters
total_quantity = 100000  # shares to execute
slice_interval = 1  # 1 minute between slices

# ============================================================================
# TWAP IMPLEMENTATION
# ============================================================================

def execute_twap(prices, total_qty, duration, slice_interval=1, market_impact_coef=0.00001):
    """
    Execute TWAP strategy: uniform slices over time.
    
    market_impact_coef: Price impact per share (temporary component)
    """
    n_slices = int(duration / slice_interval)
    slice_size = total_qty / n_slices
    
    execution_prices = []
    execution_times = []
    executed_qty = []
    remaining = total_qty
    
    for t in range(0, duration, slice_interval):
        if remaining <= 0:
            break
        
        # Market impact: temporary price increase for buying
        impact = market_impact_coef * slice_size * prices[t]
        execution_price = prices[t] * (1 + impact)
        
        # Execute slice
        qty_executed = min(slice_size, remaining)
        execution_prices.append(execution_price)
        execution_times.append(t)
        executed_qty.append(qty_executed)
        remaining -= qty_executed
    
    execution_prices = np.array(execution_prices)
    executed_qty = np.array(executed_qty)
    
    # Average execution price
    avg_price = np.sum(execution_prices * executed_qty) / total_qty
    
    # Slippage
    arrival_price = prices[0]
    slippage = avg_price - arrival_price
    slippage_bps = (slippage / arrival_price) * 10000
    
    return {
        'execution_prices': execution_prices,
        'execution_times': execution_times,
        'executed_qty': executed_qty,
        'avg_price': avg_price,
        'arrival_price': arrival_price,
        'slippage': slippage,
        'slippage_bps': slippage_bps,
        'n_slices': len(execution_prices)
    }

# ============================================================================
# VWAP IMPLEMENTATION (for comparison)
# ============================================================================

def execute_vwap(prices, volume_curve, total_qty, duration, market_impact_coef=0.00001):
    """
    Execute VWAP strategy: follow volume curve.
    """
    execution_prices = []
    execution_times = []
    executed_qty = []
    
    for t in range(duration):
        # Slice size proportional to volume
        slice_size = total_qty * volume_curve[t]
        
        # Market impact
        impact = market_impact_coef * slice_size * prices[t]
        execution_price = prices[t] * (1 + impact)
        
        execution_prices.append(execution_price)
        execution_times.append(t)
        executed_qty.append(slice_size)
    
    execution_prices = np.array(execution_prices)
    executed_qty = np.array(executed_qty)
    
    avg_price = np.sum(execution_prices * executed_qty) / total_qty
    arrival_price = prices[0]
    slippage = avg_price - arrival_price
    slippage_bps = (slippage / arrival_price) * 10000
    
    return {
        'execution_prices': execution_prices,
        'execution_times': execution_times,
        'executed_qty': executed_qty,
        'avg_price': avg_price,
        'arrival_price': arrival_price,
        'slippage': slippage,
        'slippage_bps': slippage_bps
    }

# ============================================================================
# EXECUTION SIMULATIONS
# ============================================================================

# Run TWAP
twap_result = execute_twap(prices, total_quantity, n_minutes, 
                          slice_interval=1, market_impact_coef=0.00001)

# Run VWAP
vwap_result = execute_vwap(prices, volume_curve, total_quantity, n_minutes,
                          market_impact_coef=0.00001)

# Aggressive execution (all at once)
aggressive_impact = 0.0001 * total_quantity * prices[0]
aggressive_price = prices[0] * (1 + aggressive_impact)
aggressive_slippage_bps = (aggressive_price - prices[0]) / prices[0] * 10000

# Passive execution (all at end)
passive_price = prices[-1]
passive_slippage_bps = (passive_price - prices[0]) / prices[0] * 10000

print("="*70)
print("TWAP EXECUTION ANALYSIS")
print("="*70)

print(f"\n1. TWAP Strategy Results:")
print(f"   Arrival Price: ${twap_result['arrival_price']:.2f}")
print(f"   Average Execution Price: ${twap_result['avg_price']:.2f}")
print(f"   Slippage: ${twap_result['slippage']:.4f} ({twap_result['slippage_bps']:.2f} bps)")
print(f"   Number of Slices: {twap_result['n_slices']}")

print(f"\n2. VWAP Strategy Results:")
print(f"   Average Execution Price: ${vwap_result['avg_price']:.2f}")
print(f"   Slippage: ${vwap_result['slippage']:.4f} ({vwap_result['slippage_bps']:.2f} bps)")

print(f"\n3. Aggressive Execution (Immediate):")
print(f"   Execution Price: ${aggressive_price:.2f}")
print(f"   Slippage: {aggressive_slippage_bps:.2f} bps (high market impact)")

print(f"\n4. Passive Execution (All at end):")
print(f"   Execution Price: ${passive_price:.2f}")
print(f"   Slippage: {passive_slippage_bps:.2f} bps (timing risk)")

print(f"\n5. Performance Ranking:")
strategies = [
    ('TWAP', twap_result['slippage_bps']),
    ('VWAP', vwap_result['slippage_bps']),
    ('Aggressive', aggressive_slippage_bps),
    ('Passive', passive_slippage_bps)
]
strategies_sorted = sorted(strategies, key=lambda x: abs(x[1]))
for i, (name, slip) in enumerate(strategies_sorted, 1):
    print(f"   {i}. {name:12} {slip:8.2f} bps")

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("SENSITIVITY ANALYSIS: TWAP Slice Frequency")
print("="*70)

slice_intervals = [1, 2, 5, 10, 15, 30]
sensitivity_results = []

for interval in slice_intervals:
    result = execute_twap(prices, total_quantity, n_minutes, 
                         slice_interval=interval, market_impact_coef=0.00001)
    sensitivity_results.append({
        'interval': interval,
        'n_slices': result['n_slices'],
        'slippage_bps': result['slippage_bps']
    })
    print(f"\nSlice every {interval} min: {result['n_slices']:2d} slices, "
          f"slippage {result['slippage_bps']:7.2f} bps")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Price path with TWAP executions
ax1 = axes[0, 0]
ax1.plot(time_grid, prices, 'k-', linewidth=2, label='Market Price', alpha=0.7)
ax1.scatter(twap_result['execution_times'], twap_result['execution_prices'], 
           c='blue', s=80, alpha=0.6, marker='o', label='TWAP Executions')
ax1.axhline(y=twap_result['arrival_price'], color='green', linestyle='--', 
           linewidth=2, label=f'Arrival: ${twap_result["arrival_price"]:.2f}')
ax1.axhline(y=twap_result['avg_price'], color='red', linestyle='--', 
           linewidth=2, label=f'TWAP Avg: ${twap_result["avg_price"]:.2f}')
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Price ($)')
ax1.set_title('TWAP Execution: Price Path & Executions')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative quantity executed
ax2 = axes[0, 1]
twap_cumulative = np.cumsum(twap_result['executed_qty'])
vwap_cumulative = np.cumsum(vwap_result['executed_qty'])

ax2.plot(twap_result['execution_times'], twap_cumulative, 
        'o-', linewidth=2, label='TWAP (Linear)', color='blue')
ax2.plot(vwap_result['execution_times'], vwap_cumulative, 
        's-', linewidth=2, label='VWAP (Volume-Weighted)', color='orange')
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('Cumulative Quantity')
ax2.set_title('Execution Progress: TWAP vs VWAP')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Volume curve overlay
ax3 = axes[1, 0]
ax3_twin = ax3.twinx()

# Volume curve
ax3.bar(time_grid, volume_curve * total_quantity, alpha=0.3, color='gray', 
       label='Market Volume')
# TWAP slices
ax3_twin.bar(twap_result['execution_times'], twap_result['executed_qty'], 
            alpha=0.7, color='blue', width=0.8, label='TWAP Slices')

ax3.set_xlabel('Time (minutes)')
ax3.set_ylabel('Market Volume', color='gray')
ax3_twin.set_ylabel('TWAP Slice Size', color='blue')
ax3.set_title('TWAP vs Intraday Volume Profile')
ax3.tick_params(axis='y', labelcolor='gray')
ax3_twin.tick_params(axis='y', labelcolor='blue')
ax3.grid(True, alpha=0.3)

# Plot 4: Sensitivity analysis
ax4 = axes[1, 1]
intervals = [r['interval'] for r in sensitivity_results]
slippages = [r['slippage_bps'] for r in sensitivity_results]
n_slices_list = [r['n_slices'] for r in sensitivity_results]

ax4.plot(intervals, slippages, 'o-', linewidth=2, markersize=10, color='red')
ax4.set_xlabel('Slice Interval (minutes)')
ax4.set_ylabel('Slippage (bps)', color='red')
ax4.set_title('TWAP Sensitivity: Slice Frequency vs Slippage')
ax4.tick_params(axis='y', labelcolor='red')
ax4.grid(True, alpha=0.3)

# Add number of slices on secondary axis
ax4_twin = ax4.twinx()
ax4_twin.plot(intervals, n_slices_list, 's--', linewidth=2, 
             markersize=8, color='blue', alpha=0.7)
ax4_twin.set_ylabel('Number of Slices', color='blue')
ax4_twin.tick_params(axis='y', labelcolor='blue')

plt.tight_layout()
plt.savefig('twap_execution_analysis.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved: twap_execution_analysis.png")
plt.show()

# ============================================================================
# MONTE CARLO: TWAP ROBUSTNESS
# ============================================================================

print("\n" + "="*70)
print("MONTE CARLO SIMULATION: TWAP Performance Distribution")
print("="*70)

n_simulations = 1000
twap_slippages = []
vwap_slippages = []

for sim in range(n_simulations):
    # Generate random price path
    sim_prices = initial_price * np.exp(np.cumsum(
        np.random.normal(price_drift, volatility/np.sqrt(n_minutes), n_minutes)))
    
    # TWAP
    twap_sim = execute_twap(sim_prices, total_quantity, n_minutes, 
                           slice_interval=1, market_impact_coef=0.00001)
    twap_slippages.append(twap_sim['slippage_bps'])
    
    # VWAP
    vwap_sim = execute_vwap(sim_prices, volume_curve, total_quantity, n_minutes,
                           market_impact_coef=0.00001)
    vwap_slippages.append(vwap_sim['slippage_bps'])

twap_slippages = np.array(twap_slippages)
vwap_slippages = np.array(vwap_slippages)

print(f"\nTWAP Statistics (n={n_simulations}):")
print(f"   Mean Slippage: {twap_slippages.mean():.2f} bps")
print(f"   Std Dev: {twap_slippages.std():.2f} bps")
print(f"   95% VaR (worst 5%): {np.percentile(twap_slippages, 95):.2f} bps")
print(f"   Max Slippage: {twap_slippages.max():.2f} bps")

print(f"\nVWAP Statistics (n={n_simulations}):")
print(f"   Mean Slippage: {vwap_slippages.mean():.2f} bps")
print(f"   Std Dev: {vwap_slippages.std():.2f} bps")
print(f"   95% VaR: {np.percentile(vwap_slippages, 95):.2f} bps")
print(f"   Max Slippage: {vwap_slippages.max():.2f} bps")

# Create distribution plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(twap_slippages, bins=50, alpha=0.6, label='TWAP', density=True)
ax.hist(vwap_slippages, bins=50, alpha=0.6, label='VWAP', density=True)
ax.axvline(twap_slippages.mean(), color='blue', linestyle='--', 
          linewidth=2, label=f'TWAP Mean: {twap_slippages.mean():.1f} bps')
ax.axvline(vwap_slippages.mean(), color='orange', linestyle='--', 
          linewidth=2, label=f'VWAP Mean: {vwap_slippages.mean():.1f} bps')
ax.set_xlabel('Slippage (bps)')
ax.set_ylabel('Density')
ax.set_title(f'Monte Carlo: TWAP vs VWAP Slippage Distribution (n={n_simulations})')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('twap_monte_carlo.png', dpi=100, bbox_inches='tight')
print("✓ Visualization saved: twap_monte_carlo.png")
plt.show()
```

## 6. Challenge Round
When is TWAP suboptimal?
- Strong intraday patterns: Misses low-volume windows with better liquidity
- Trending markets: Uniform slicing accumulates adverse price movement
- Low-liquidity stocks: Predictable timing invites predatory trading
- Large orders: Market impact overwhelms timing diversification benefits
- Volatile periods: Fixed schedule can't adapt to sudden price swings

## 7. Key References
- [Kissell & Glantz, Optimal Trading Strategies](https://www.amazon.com/Optimal-Trading-Strategies-Quantitative-Approaches/dp/0814407242)
- [VWAP vs TWAP Comparison (Bloomberg)](https://www.bloomberg.com/professional/product/execution-management/)
- [CME Execution Strategies Guide](https://www.cmegroup.com/education/courses/execution-strategies.html)

---
**Status:** Standard execution benchmark | **Complements:** VWAP, Implementation Shortfall, Almgren-Chriss