# Trade Scheduling: Intraday Volume Patterns & Optimal Timing

## 1. Concept Skeleton
**Definition:** Strategically timing trades to align with predicted intraday volume and liquidity patterns; exploiting U-shaped volume curves  
**Purpose:** Minimize market impact by trading during high-liquidity periods; avoid moving prices during thin markets  
**Prerequisites:** Intraday volume forecasting, liquidity measurement, transaction cost modeling

## 2. Comparative Framing
| Approach | Trade Scheduling | VWAP | TWAP | Almgren-Chriss |
|----------|-----------------|------|------|----------------|
| **Timing Logic** | Liquidity-optimal windows | Volume forecast matching | Uniform time | Risk-cost tradeoff |
| **Information Used** | Historical volume curves | Same-day volume | Clock only | Volatility + risk aversion |
| **Adaptability** | Pre-determined with updates | Static forecast | Fixed | Static optimization |
| **Complexity** | Medium (forecasting) | Medium | Low | High (theory) |

## 3. Examples + Counterexamples

**Simple Example:**  
Predict U-shaped volume: Trade 40% at open (9:30-10:00), 20% mid-day, 40% at close (3:30-4:00) → Lower impact

**Failure Case:**  
Schedule heavy trading at close → News breaks 15 minutes before → Surge volume, volatility spike → Catastrophic slippage

**Edge Case:**  
Predict Friday volume = Thursday volume → Actually 50% lower (holiday weekend) → Schedule too aggressive, dominates tape

## 4. Layer Breakdown
```
Trade Scheduling Framework:
├─ Volume Forecasting:
│   ├─ Historical Patterns:
│   │   ├─ Intraday Shape: U-curve (high open, low midday, high close)
│   │   ├─ Day-of-Week: Monday high, Friday low
│   │   └─ Calendar Effects: Month-end, opex, FOMC
│   ├─ Predictive Models:
│   │   ├─ Seasonal Decomposition: Trend + Daily Pattern + Noise
│   │   ├─ Autoregressive: ARIMA on volume time series
│   │   └─ Machine Learning: XGBoost with feature engineering
│   └─ Real-Time Updates:
│       ├─ Morning Volume → Update afternoon forecast
│       ├─ News Events → Adjust participation
│       └─ Market Stress → Reduce aggressiveness
├─ Schedule Optimization:
│   ├─ Objective: Minimize market impact function
│   │   Impact(t) = λ × (quantity_t / volume_t)^α
│   │   Total Cost = Σ_t Impact(t)
│   ├─ Constraints:
│   │   ├─ Total quantity: Σ_t q_t = Q
│   │   ├─ Completion time: T_max
│   │   ├─ Max participation: q_t / V_t ≤ r_max
│   │   └─ Minimum slice: q_t ≥ q_min (if q_t > 0)
│   └─ Solution Methods:
│       ├─ Convex Optimization: Quadratic programming
│       ├─ Dynamic Programming: Time-varying costs
│       └─ Heuristic: Proportional to volume^β forecast
├─ Intraday Volume Patterns:
│   ├─ Opening Session (9:30-10:30):
│   │   ├─ Volume: 20-30% of daily
│   │   ├─ Drivers: Overnight info, index rebalancing
│   │   └─ Liquidity: High but volatile
│   ├─ Midday Lull (11:00-14:00):
│   │   ├─ Volume: 15-25% of daily
│   │   ├─ Drivers: European close (11-12), lunch
│   │   └─ Liquidity: Lowest spreads but thin depth
│   ├─ Closing Session (15:00-16:00):
│   │   ├─ Volume: 30-40% of daily
│   │   ├─ Drivers: Benchmarking (MOC), day traders exit
│   │   └─ Liquidity: High but crowded (algo traffic)
│   └─ Special Periods:
│       ├─ 9:30-9:35: Opening auction (extreme volume)
│       ├─ 15:50-16:00: Closing auction (MOC imbalance)
│       └─ 10:00, 14:00: Economic releases (volume spikes)
├─ Implementation:
│   ├─ Pre-Trade Planning:
│   │   ├─ Forecast volume curve: V̂(t)
│   │   ├─ Optimize schedule: q*(t)
│   │   └─ Set participation limits: r_max(t)
│   ├─ Intraday Execution:
│   │   ├─ Monitor actual vs forecast: V_actual(t) / V̂(t)
│   │   ├─ Adjust schedule: If V low → reduce q, if V high → increase q
│   │   └─ Track completion: Shortfall triggers acceleration
│   └─ Post-Trade Analysis:
│       ├─ Realized vs benchmark: VWAP_exec - VWAP_market
│       ├─ Impact attribution: Timing vs sizing vs market move
│       └─ Forecast accuracy: MAPE on volume predictions
└─ Advanced Topics:
    ├─ Multi-Asset: Correlate schedules (pairs trading)
    ├─ Cross-Venue: Dark pool vs lit timing
    ├─ Adversarial: Avoid predictable patterns (randomization)
    └─ Event-Driven: FOMC, earnings → regime-specific schedules
```

**Interaction:** Forecast volume → Optimize timing → Execute during liquidity peaks → Adapt to realized volume

## 5. Mini-Project
Simulate trade scheduling using U-shaped volume curve and compare to naive strategies:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

# ============================================================================
# INTRADAY VOLUME SIMULATION
# ============================================================================

def generate_intraday_volume(n_minutes=390, noise_level=0.3):
    """
    Generate realistic U-shaped intraday volume curve.
    n_minutes: Trading day length (390 = 6.5 hours)
    """
    time_grid = np.linspace(0, 1, n_minutes)
    
    # U-shaped base pattern (high at open/close, low midday)
    # Using combination of exponentials to create realistic shape
    opening_surge = 2.5 * np.exp(-10 * time_grid)  # Decay from open
    closing_surge = 2.5 * np.exp(-10 * (1 - time_grid))  # Ramp to close
    midday_base = 0.5  # Minimum midday volume
    
    base_curve = opening_surge + closing_surge + midday_base
    
    # Add realistic noise (autocorrelated)
    noise = np.random.normal(0, noise_level, n_minutes)
    smoothed_noise = np.convolve(noise, np.ones(5)/5, mode='same')
    
    volume_curve = base_curve * (1 + smoothed_noise)
    volume_curve = np.maximum(volume_curve, 0.1)  # Floor at 10% of base
    
    # Normalize to total daily volume = 10 million shares
    total_daily_volume = 10_000_000
    volume_curve = volume_curve / volume_curve.sum() * total_daily_volume
    
    return volume_curve

# Generate market volume
n_minutes = 390
market_volume = generate_intraday_volume(n_minutes)

# Generate price path (correlated with volume)
initial_price = 100.0
price_volatility = 0.0002  # per minute
price_changes = np.random.normal(0, price_volatility, n_minutes)
# Add microstructure: higher volume → higher volatility
price_changes *= (1 + 0.3 * market_volume / market_volume.mean())
prices = initial_price * np.exp(np.cumsum(price_changes))

# ============================================================================
# MARKET IMPACT MODEL
# ============================================================================

def market_impact_cost(slice_size, market_volume_t, price_t, alpha=0.6):
    """
    Nonlinear market impact: cost ∝ (quantity / volume)^alpha
    alpha < 1: Concave (economies of scale in liquidity)
    """
    if market_volume_t == 0 or slice_size == 0:
        return 0
    
    # Participation rate
    participation = slice_size / market_volume_t
    
    # Temporary impact (bps)
    impact_bps = 10 * (participation ** alpha)
    
    # Total cost ($)
    cost = slice_size * price_t * (impact_bps / 10000)
    
    return cost

# ============================================================================
# TRADE SCHEDULING STRATEGIES
# ============================================================================

total_quantity = 500_000  # shares to execute

# Strategy 1: UNIFORM (naive TWAP-like)
uniform_schedule = np.ones(n_minutes) * (total_quantity / n_minutes)

# Strategy 2: VOLUME-WEIGHTED (like VWAP)
volume_weighted_schedule = (market_volume / market_volume.sum()) * total_quantity

# Strategy 3: OPTIMIZED (minimize impact)
def total_impact_objective(schedule):
    """Objective: total market impact cost."""
    total_cost = 0
    for t in range(n_minutes):
        cost_t = market_impact_cost(schedule[t], market_volume[t], prices[t])
        total_cost += cost_t
    return total_cost

# Constraints: sum to total quantity, non-negative
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - total_quantity}
]
bounds = [(0, total_quantity * 0.1)] * n_minutes  # Max 10% in any minute

# Initial guess: volume-weighted
x0 = volume_weighted_schedule

print("="*70)
print("TRADE SCHEDULING OPTIMIZATION")
print("="*70)
print("Optimizing schedule to minimize market impact...")

result = minimize(
    total_impact_objective,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 500}
)

optimized_schedule = result.x

print(f"✓ Optimization {'succeeded' if result.success else 'failed'}")
print(f"  Total quantity: {optimized_schedule.sum():,.0f} / {total_quantity:,}")

# ============================================================================
# PERFORMANCE EVALUATION
# ============================================================================

def evaluate_strategy(schedule, market_volume, prices, strategy_name):
    """Calculate total cost and metrics for a schedule."""
    total_cost = 0
    execution_prices = []
    participation_rates = []
    
    for t in range(n_minutes):
        if schedule[t] > 0:
            cost_t = market_impact_cost(schedule[t], market_volume[t], prices[t])
            total_cost += cost_t
            
            # Effective execution price
            impact_bps = 10 * ((schedule[t] / market_volume[t]) ** 0.6)
            exec_price = prices[t] * (1 + impact_bps / 10000)
            execution_prices.append(exec_price)
            
            participation_rates.append(schedule[t] / market_volume[t])
    
    # Average execution price
    avg_exec_price = np.average(execution_prices, weights=schedule[schedule > 0])
    
    # Market VWAP (benchmark)
    market_vwap = np.sum(prices * market_volume) / market_volume.sum()
    
    # Slippage vs arrival price
    arrival_price = prices[0]
    slippage_bps = (avg_exec_price - arrival_price) / arrival_price * 10000
    
    # Slippage vs market VWAP
    vwap_diff_bps = (avg_exec_price - market_vwap) / market_vwap * 10000
    
    # Participation stats
    avg_participation = np.mean(participation_rates)
    max_participation = np.max(participation_rates) if len(participation_rates) > 0 else 0
    
    results = {
        'total_cost': total_cost,
        'avg_exec_price': avg_exec_price,
        'arrival_price': arrival_price,
        'market_vwap': market_vwap,
        'slippage_bps': slippage_bps,
        'vwap_diff_bps': vwap_diff_bps,
        'avg_participation': avg_participation,
        'max_participation': max_participation
    }
    
    return results

print("\n" + "="*70)
print("STRATEGY COMPARISON")
print("="*70)

strategies = {
    'Uniform': uniform_schedule,
    'Volume-Weighted': volume_weighted_schedule,
    'Optimized': optimized_schedule
}

results_dict = {}

for name, schedule in strategies.items():
    results = evaluate_strategy(schedule, market_volume, prices, name)
    results_dict[name] = results
    
    print(f"\n{name} Schedule:")
    print(f"   Total Cost: ${results['total_cost']:,.2f}")
    print(f"   Avg Execution Price: ${results['avg_exec_price']:.2f}")
    print(f"   Slippage vs Arrival: {results['slippage_bps']:.2f} bps")
    print(f"   vs Market VWAP: {results['vwap_diff_bps']:+.2f} bps")
    print(f"   Avg Participation: {results['avg_participation']*100:.2f}%")
    print(f"   Max Participation: {results['max_participation']*100:.2f}%")

# Calculate improvements
uniform_cost = results_dict['Uniform']['total_cost']
optimized_cost = results_dict['Optimized']['total_cost']
improvement_pct = (uniform_cost - optimized_cost) / uniform_cost * 100

print(f"\n{'='*70}")
print(f"Optimized Schedule vs Uniform: {improvement_pct:+.1f}% cost reduction")
print(f"{'='*70}")

# ============================================================================
# VOLUME FORECAST ERROR SENSITIVITY
# ============================================================================

print("\n" + "="*70)
print("ROBUSTNESS: Volume Forecast Error Impact")
print("="*70)

# Simulate forecast errors
forecast_errors = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]  # Multipliers
forecast_costs = []

for error_mult in forecast_errors:
    # "Forecasted" volume (wrong)
    forecast_volume = market_volume * error_mult
    
    # Schedule based on forecast
    forecast_schedule = (forecast_volume / forecast_volume.sum()) * total_quantity
    
    # But execute in actual market
    results = evaluate_strategy(forecast_schedule, market_volume, prices, f'Forecast×{error_mult}')
    forecast_costs.append(results['total_cost'])
    
    if error_mult in [0.5, 1.0, 1.5]:
        print(f"\nForecast Error ×{error_mult}: Total Cost = ${results['total_cost']:,.2f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(3, 3, figsize=(18, 12))

# Time axis (minutes from open, convert to hours)
time_axis = np.arange(n_minutes) / 60  # Hours from open

# Plot 1: Intraday volume curve
ax1 = axes[0, 0]
ax1.fill_between(time_axis, 0, market_volume / 1000, alpha=0.3, color='gray', label='Market Volume')
ax1.plot(time_axis, market_volume / 1000, 'k-', linewidth=2)
ax1.set_xlabel('Hours from Open')
ax1.set_ylabel('Volume (thousands of shares)')
ax1.set_title('Intraday Volume Pattern (U-Shaped)')
ax1.axvline(x=1.5, color='red', linestyle='--', alpha=0.3, label='Midday Lull')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Price path
ax2 = axes[0, 1]
ax2.plot(time_axis, prices, 'b-', linewidth=2)
ax2.set_xlabel('Hours from Open')
ax2.set_ylabel('Price ($)')
ax2.set_title('Intraday Price Path')
ax2.grid(True, alpha=0.3)

# Plot 3: Schedule comparison
ax3 = axes[0, 2]
ax3.plot(time_axis, uniform_schedule, label='Uniform', linewidth=2, alpha=0.7)
ax3.plot(time_axis, volume_weighted_schedule, label='Volume-Weighted', linewidth=2, alpha=0.7)
ax3.plot(time_axis, optimized_schedule, label='Optimized', linewidth=2, alpha=0.7)
ax3.set_xlabel('Hours from Open')
ax3.set_ylabel('Shares per Minute')
ax3.set_title('Trade Schedules Comparison')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Participation rates
ax4 = axes[1, 0]
for name, schedule in strategies.items():
    participation = np.divide(schedule, market_volume, 
                             out=np.zeros_like(schedule), where=market_volume!=0) * 100
    ax4.plot(time_axis, participation, label=name, linewidth=2, alpha=0.7)
ax4.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% Threshold')
ax4.set_xlabel('Hours from Open')
ax4.set_ylabel('Participation Rate (%)')
ax4.set_title('Market Participation by Strategy')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Plot 5: Cumulative execution
ax5 = axes[1, 1]
for name, schedule in strategies.items():
    cumulative = np.cumsum(schedule) / total_quantity * 100
    ax5.plot(time_axis, cumulative, label=name, linewidth=2, alpha=0.7)
ax5.set_xlabel('Hours from Open')
ax5.set_ylabel('Cumulative % Executed')
ax5.set_title('Execution Progress')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Plot 6: Cost comparison bar chart
ax6 = axes[1, 2]
costs = [results_dict[name]['total_cost'] for name in strategies.keys()]
colors_bar = ['lightcoral', 'lightblue', 'lightgreen']
bars = ax6.bar(strategies.keys(), costs, color=colors_bar, edgecolor='black', alpha=0.7)
ax6.set_ylabel('Total Cost ($)')
ax6.set_title('Total Execution Cost by Strategy')
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, cost in zip(bars, costs):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'${cost:,.0f}', ha='center', va='bottom', fontsize=9)

# Plot 7: Volume concentration (pie chart)
ax7 = axes[2, 0]
# Divide day into segments
open_vol = market_volume[:60].sum()  # First hour
midday_vol = market_volume[60:300].sum()  # Hours 1-5
close_vol = market_volume[300:].sum()  # Last 1.5 hours
segments = [open_vol, midday_vol, close_vol]
labels = ['Opening\n(1st hour)', 'Midday\n(hours 1-5)', 'Closing\n(last 1.5h)']
colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
ax7.pie(segments, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
ax7.set_title('Volume Distribution by Session')

# Plot 8: Forecast error sensitivity
ax8 = axes[2, 1]
ax8.plot(forecast_errors, forecast_costs, 'o-', linewidth=2, markersize=8, color='purple')
ax8.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Perfect Forecast')
ax8.axhline(y=optimized_cost, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Optimized Cost')
ax8.set_xlabel('Forecast Error Multiplier')
ax8.set_ylabel('Total Cost ($)')
ax8.set_title('Robustness: Impact of Volume Forecast Errors')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# Plot 9: Market impact function
ax9 = axes[2, 2]
participation_range = np.linspace(0, 0.3, 100)
impact_bps = 10 * (participation_range ** 0.6)
ax9.plot(participation_range * 100, impact_bps, linewidth=3, color='darkred')
ax9.fill_between(participation_range * 100, 0, impact_bps, alpha=0.2, color='red')
ax9.set_xlabel('Participation Rate (%)')
ax9.set_ylabel('Market Impact (bps)')
ax9.set_title('Market Impact Function (Concave)')
ax9.grid(True, alpha=0.3)
ax9.annotate('Nonlinear: impact ∝ participation^0.6', 
            xy=(15, 20), fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('trade_scheduling_analysis.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved: trade_scheduling_analysis.png")
plt.show()
```

## 6. Challenge Round
When does scheduling optimization fail?
- Regime breaks: Historical U-curve invalid during crisis (volume persists all day) → Schedule wrong
- Flash events: News at 2pm creates volume spike → Pre-determined schedule misses opportunity or overtrades
- Crowding: All algos schedule close trading → Closing auction becomes expensive (self-fulfilling)
- Forecast error: Predicted volume = 2× actual → Schedule too passive, incomplete execution
- Endogeneity: Large order itself shifts volume curve → Assume exogenous volume is wrong

## 7. Key References
- [Admati & Pfleiderer (1988), "Theory of Intraday Patterns"](https://www.jstor.org/stable/2962302)
- [Kissell, "Optimal Trading Strategies", Ch. 8](https://www.amazon.com/Optimal-Trading-Strategies-Quantitative-Approaches/dp/0814407242)
- [Bialkowski et al. (2008), "Intraday Liquidity Patterns"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1107872)

---
**Status:** Foundational for algo execution | **Complements:** VWAP, POV, Almgren-Chriss | **Requires:** Volume forecasting