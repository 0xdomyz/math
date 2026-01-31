# Execution Costs in Algorithmic Trading

## 1. Concept Skeleton
**Definition:** Total cost of transforming investment decision into executed position; includes spread, market impact, opportunity cost, fees  
**Purpose:** Quantify trading friction; optimize execution to minimize total transaction cost; measure execution quality  
**Prerequisites:** Market microstructure, bid-ask spread, liquidity measurement, order types, transaction cost analysis (TCA)

## 2. Comparative Framing
| Cost Component | Explicit (Direct) | Implicit (Hidden) | Timing-Based | Measurement |
|----------------|-------------------|-------------------|--------------|-------------|
| **Exchange Fees** | ✓ | ✗ | ✗ | Invoice |
| **Bid-Ask Spread** | ✗ | ✓ | ✗ | Half-spread × quantity |
| **Market Impact** | ✗ | ✓ | ✗ | Price move during execution |
| **Opportunity Cost** | ✗ | ✓ | ✓ | Slippage from delay |
| **Timing Risk** | ✗ | ✗ | ✓ | Adverse price move |

## 3. Examples + Counterexamples

**Simple Example:**  
Buy 10,000 shares at bid=$100, ask=$100.10 → Pay spread $0.10 × 10,000 = $1,000 → Explicit cost visible

**Failure Case:**  
Aggressive execution: Buy 100K shares instantly → Price jumps $0.50 → Market impact $50K >> spread cost

**Edge Case:**  
Wait for better price → Market rallies $1 while waiting → Opportunity cost $100K > saved execution cost

## 4. Layer Breakdown
```
Execution Cost Taxonomy:
├─ I. EXPLICIT COSTS (Observable, Direct):
│   ├─ Commissions:
│   │   ├─ Broker Fees: $/share or % of notional
│   │   ├─ Range: 0.5-5 bps (basis points) typical
│   │   ├─ Bundled: Execution + research + custody
│   │   └─ Unbundled: Pure execution only
│   ├─ Exchange Fees:
│   │   ├─ Maker-Taker Fees:
│   │   │   ├─ Maker Rebate: ~0.2-0.3 bps (add liquidity)
│   │   │   ├─ Taker Fee: ~0.3-0.5 bps (remove liquidity)
│   │   │   └─ Net: Taker pays ~0.5 bps, maker earns ~0.2 bps
│   │   ├─ Routing Fees: Smart order router costs
│   │   └─ Dark Pool: Typically no fees (or small)
│   ├─ Clearing & Settlement:
│   │   ├─ DTC Fees: ~$0.002 per side
│   │   ├─ NSCC: Continuous net settlement
│   │   └─ SEF/DCO: Derivatives clearing (futures, swaps)
│   ├─ Market Data:
│   │   ├─ Level 1: Best bid/offer (~$10-50/month)
│   │   ├─ Level 2: Full order book (~$50-500/month)
│   │   └─ Tick Data: Historical trades ($$thousands)
│   └─ Total Explicit: Sum of all invoiced costs
│       Formula: Commissions + Exchange Fees + Clearing
├─ II. IMPLICIT COSTS (Hidden, Price-Based):
│   ├─ Bid-Ask Spread:
│   │   ├─ Definition: Ask - Bid (cost to round-trip)
│   │   ├─ One-Way Cost: Spread / 2
│   │   ├─ Typical: 1-50 bps depending on liquidity
│   │   ├─ Factors:
│   │   │   ├─ Adverse Selection: Information asymmetry
│   │   │   ├─ Inventory Risk: Market maker holding cost
│   │   │   └─ Order Processing: Fixed cost per trade
│   │   └─ Measurement: Quoted spread, effective spread, realized spread
│   ├─ Market Impact:
│   │   ├─ Definition: Price move caused by own order
│   │   ├─ Temporary Impact:
│   │   │   ├─ During execution: Liquidity demand shock
│   │   │   ├─ Reversal: Prices revert after completion
│   │   │   ├─ Duration: Minutes to hours
│   │   │   └─ Models: Square root, linear, power law
│   │   ├─ Permanent Impact:
│   │   │   ├─ Information: Order reveals new information
│   │   │   ├─ Persistent: Prices don't fully revert
│   │   │   ├─ Duration: Days to weeks
│   │   │   └─ Cause: Informed trading, price discovery
│   │   ├─ Impact Function:
│   │   │   ├─ Linear: Cost = λ × Q (constant slope)
│   │   │   ├─ Square Root: Cost = λ × √Q (concave, economies of scale)
│   │   │   ├─ Power Law: Cost = λ × Q^α (α ≈ 0.5-0.7)
│   │   │   └─ Almgren-Chriss: Temporary + permanent components
│   │   └─ Factors:
│   │       ├─ Order Size: Larger Q → higher impact
│   │       ├─ Liquidity: Low volume → higher impact
│   │       ├─ Volatility: High σ → wider spreads → impact
│   │       └─ Speed: Faster execution → more impact
│   ├─ Opportunity Cost:
│   │   ├─ Definition: Cost of not executing (delay)
│   │   ├─ Calculation: (Decision Price - Execution Price) × Quantity
│   │   ├─ Causes:
│   │   │   ├─ Patience: Waiting for better price
│   │   │   ├─ Limit Orders: Unfilled orders (adverse selection)
│   │   │   ├─ Partial Fills: Incomplete execution
│   │   │   └─ Market Moves Away: Adverse price movement
│   │   ├─ Tradeoff: Urgency vs Impact
│   │   │   ├─ Fast Execution: Low opportunity cost, high impact
│   │   │   ├─ Slow Execution: High opportunity cost, low impact
│   │   │   └─ Optimal: Balance via Almgren-Chriss framework
│   │   └─ Measurement: Arrival price benchmark
│   └─ Timing Risk:
│       ├─ Definition: Volatility during execution window
│       ├─ Cost: σ × √T × Q (random component)
│       ├─ Not Predictable: Market moves unrelated to order
│       └─ Mitigation: Faster execution reduces T
├─ III. TOTAL TRANSACTION COST (TTC):
│   ├─ Components:
│   │   TTC = Explicit + Spread + Impact + Opportunity Cost
│   ├─ Benchmark-Relative:
│   │   ├─ Arrival Price: Price at decision time
│   │   ├─ VWAP: Volume-weighted average price (period)
│   │   ├─ TWAP: Time-weighted average price
│   │   └─ Close: Official closing price
│   ├─ Pre-Trade Estimation:
│   │   ├─ Inputs: Quantity, ADV (avg daily volume), volatility
│   │   ├─ Models: Empirical impact curves, ML predictions
│   │   └─ Output: Expected TTC in bps
│   ├─ Real-Time Monitoring:
│   │   ├─ Slippage: Execution price vs arrival price
│   │   ├─ Fill Rate: Executed / total quantity
│   │   └─ Alerts: If TTC exceeds threshold
│   └─ Post-Trade Analysis (TCA):
│       ├─ Implementation Shortfall: Decision → execution
│       ├─ Price Impact: Execution → post-trade
│       ├─ Attribution: Spread vs impact vs timing
│       └─ Venue Analysis: Which exchanges performed best
├─ IV. COST MINIMIZATION STRATEGIES:
│   ├─ Order Type Selection:
│   │   ├─ Market Orders: Pay spread + impact (immediate)
│   │   ├─ Limit Orders: Save spread, risk non-fill
│   │   ├─ Iceberg: Hide quantity, reduce impact signal
│   │   └─ Dark Pool: Midpoint, no information leakage
│   ├─ Execution Algorithms:
│   │   ├─ TWAP: Minimize timing risk (uniform)
│   │   ├─ VWAP: Minimize benchmark deviation
│   │   ├─ POV: Blend into volume (low footprint)
│   │   ├─ Implementation Shortfall: Optimize urgency
│   │   └─ Adaptive: ML-based dynamic adjustment
│   ├─ Venue Selection:
│   │   ├─ Smart Order Routing: Best price across exchanges
│   │   ├─ Dark Pools: Large blocks, no pre-trade transparency
│   │   ├─ Lit Exchanges: Price discovery, maker rebates
│   │   └─ Internalization: Broker crosses internally
│   ├─ Timing:
│   │   ├─ Avoid: Open/close (volatile), news releases
│   │   ├─ Target: High-liquidity windows (11am-2pm)
│   │   └─ Patient: Use limit orders in thin periods
│   └─ Size Management:
│       ├─ Slice: Break large orders into smaller pieces
│       ├─ Randomize: Avoid predictable patterns
│       └─ Participate: Target <10% of volume
└─ V. MEASUREMENT & BENCHMARKS:
    ├─ Arrival Price:
    │   ├─ Definition: Mid-quote at decision time
    │   ├─ Use: Measure total implementation shortfall
    │   └─ Formula: (Exec Price - Arrival) / Arrival × 10,000 bps
    ├─ VWAP:
    │   ├─ Definition: Volume-weighted market price
    │   ├─ Use: Benchmark for execution algos
    │   └─ Formula: Σ(Price_i × Volume_i) / Σ(Volume_i)
    ├─ Close:
    │   ├─ Definition: Official closing price (4:00 PM)
    │   ├─ Use: NAV tracking (mutual funds)
    │   └─ Risk: Crowded (everyone trades close)
    ├─ Pre-Trade Estimate:
    │   ├─ Predicted TTC before execution
    │   ├─ Compare: Actual vs predicted
    │   └─ Improve: Refine cost models
    └─ Venue Analysis:
        ├─ Fill Rate: % executed per venue
        ├─ Price Improvement: Execution vs NBBO
        └─ Rebate Capture: Maker fee economics
```

**Interaction:** Order decision → Cost estimation → Algo selection → Execution → TCA → Model refinement

## 5. Mini-Project
Implement transaction cost breakdown and optimization:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

# ============================================================================
# MARKET PARAMETERS
# ============================================================================

initial_price = 100.0
daily_volume = 1_000_000  # Average daily volume (shares)
volatility = 0.02  # Daily volatility (2%)

# Order parameters
total_quantity = 50_000  # shares to execute
n_slices = 10  # Number of execution periods

print("="*70)
print("EXECUTION COST ANALYSIS")
print("="*70)
print(f"\nMarket Parameters:")
print(f"   Price: ${initial_price:.2f}")
print(f"   Avg Daily Volume: {daily_volume:,} shares")
print(f"   Daily Volatility: {volatility*100:.1f}%")
print(f"\nOrder:")
print(f"   Quantity: {total_quantity:,} shares")
print(f"   % of ADV: {total_quantity/daily_volume*100:.1f}%")

# ============================================================================
# COST COMPONENTS
# ============================================================================

def calculate_explicit_costs(quantity, price):
    """Calculate commissions and fees."""
    commission_bps = 0.5  # 0.5 basis points
    commission = quantity * price * (commission_bps / 10000)
    
    exchange_fee_bps = 0.3  # Taker fee
    exchange_fee = quantity * price * (exchange_fee_bps / 10000)
    
    clearing_fee = 0.002 * quantity  # $0.002 per share
    
    total_explicit = commission + exchange_fee + clearing_fee
    
    return {
        'commission': commission,
        'exchange_fee': exchange_fee,
        'clearing_fee': clearing_fee,
        'total': total_explicit
    }

def calculate_spread_cost(quantity, price, spread_bps=10):
    """Calculate bid-ask spread cost."""
    spread = price * (spread_bps / 10000)
    half_spread = spread / 2  # One-way cost
    spread_cost = quantity * half_spread
    return spread_cost

def calculate_market_impact(quantity, volume, price, alpha=0.6):
    """
    Calculate market impact using power law model.
    Impact = λ × (Q / V)^α
    α ≈ 0.6 (empirical, square root-like but slightly steeper)
    """
    lambda_param = 0.1  # Impact coefficient (calibrated)
    participation = quantity / volume
    impact_bps = lambda_param * (participation ** alpha) * 1000  # Scale to bps
    impact_cost = quantity * price * (impact_bps / 10000)
    return impact_cost, impact_bps

def calculate_opportunity_cost(quantity, price, price_move):
    """
    Opportunity cost from adverse price movement during delay.
    price_move: Price change between decision and execution
    """
    opportunity_cost = quantity * abs(price_move)
    return opportunity_cost

# ============================================================================
# EXECUTION STRATEGIES WITH COST BREAKDOWN
# ============================================================================

def aggressive_execution(quantity, price, volume):
    """Execute entire order immediately (one market order)."""
    explicit = calculate_explicit_costs(quantity, price)
    spread_cost = calculate_spread_cost(quantity, price, spread_bps=10)
    impact_cost, impact_bps = calculate_market_impact(quantity, volume, price, alpha=0.6)
    opportunity_cost = 0  # No delay
    
    total_cost = explicit['total'] + spread_cost + impact_cost + opportunity_cost
    total_bps = (total_cost / (quantity * price)) * 10000
    
    return {
        'strategy': 'Aggressive (Immediate)',
        'explicit': explicit['total'],
        'spread': spread_cost,
        'impact': impact_cost,
        'opportunity': opportunity_cost,
        'total': total_cost,
        'total_bps': total_bps,
        'impact_bps': impact_bps
    }

def patient_execution(quantity, price, volume, n_slices=10):
    """Execute gradually over multiple periods (TWAP-like)."""
    slice_size = quantity / n_slices
    
    total_explicit = 0
    total_spread = 0
    total_impact = 0
    total_opportunity = 0
    
    # Simulate price path
    price_path = [price]
    for _ in range(n_slices):
        # Random walk
        price_change = np.random.normal(0, volatility / np.sqrt(n_slices))
        new_price = price_path[-1] * (1 + price_change)
        price_path.append(new_price)
    
    for i in range(n_slices):
        exec_price = price_path[i]
        
        # Explicit costs per slice
        explicit = calculate_explicit_costs(slice_size, exec_price)
        total_explicit += explicit['total']
        
        # Spread cost per slice
        spread_cost = calculate_spread_cost(slice_size, exec_price, spread_bps=10)
        total_spread += spread_cost
        
        # Market impact per slice (smaller participation)
        impact_cost, _ = calculate_market_impact(slice_size, volume/n_slices, exec_price, alpha=0.6)
        total_impact += impact_cost
        
        # Opportunity cost (price move from arrival)
        price_move = exec_price - price
        opp_cost = slice_size * abs(price_move)
        total_opportunity += opp_cost
    
    total_cost = total_explicit + total_spread + total_impact + total_opportunity
    total_bps = (total_cost / (quantity * price)) * 10000
    
    return {
        'strategy': f'Patient ({n_slices} slices)',
        'explicit': total_explicit,
        'spread': total_spread,
        'impact': total_impact,
        'opportunity': total_opportunity,
        'total': total_cost,
        'total_bps': total_bps,
        'price_path': price_path
    }

# ============================================================================
# COST COMPARISON
# ============================================================================

print("\n" + "="*70)
print("EXECUTION STRATEGY COMPARISON")
print("="*70)

strategies = [
    aggressive_execution(total_quantity, initial_price, daily_volume),
    patient_execution(total_quantity, initial_price, daily_volume, n_slices=5),
    patient_execution(total_quantity, initial_price, daily_volume, n_slices=10),
    patient_execution(total_quantity, initial_price, daily_volume, n_slices=20)
]

for result in strategies:
    print(f"\n{result['strategy']}:")
    print(f"   Explicit Costs:    ${result['explicit']:,.2f} ({result['explicit']/result['total']*100:.1f}%)")
    print(f"   Spread Cost:       ${result['spread']:,.2f} ({result['spread']/result['total']*100:.1f}%)")
    print(f"   Market Impact:     ${result['impact']:,.2f} ({result['impact']/result['total']*100:.1f}%)")
    print(f"   Opportunity Cost:  ${result['opportunity']:,.2f} ({result['opportunity']/result['total']*100:.1f}%)")
    print(f"   ─────────────────────────────────────")
    print(f"   TOTAL COST:        ${result['total']:,.2f} ({result['total_bps']:.2f} bps)")

# Find optimal strategy
optimal = min(strategies, key=lambda x: x['total'])
print(f"\n** OPTIMAL STRATEGY: {optimal['strategy']} **")
print(f"   Total Cost: ${optimal['total']:,.2f} ({optimal['total_bps']:.2f} bps)")

# ============================================================================
# ALMGREN-CHRISS OPTIMAL EXECUTION
# ============================================================================

print("\n" + "="*70)
print("ALMGREN-CHRISS OPTIMAL EXECUTION")
print("="*70)

def almgren_chriss_optimal_schedule(Q, T, sigma, lambda_temp, risk_aversion=1e-6):
    """
    Calculate optimal execution schedule balancing market impact and timing risk.
    
    Q: Total quantity
    T: Time horizon (number of periods)
    sigma: Volatility
    lambda_temp: Temporary market impact coefficient
    risk_aversion: Trader's risk aversion (λ)
    """
    # Simplified: Optimal trajectory is linear decrease
    # n_t = n_0 × (1 - t/T) where n_0 = Q
    
    trajectory = []
    times = np.linspace(0, T, T+1)
    for t in times:
        remaining = Q * (1 - t / T)
        trajectory.append(remaining)
    
    # Calculate trading rate (v_t = -dn/dt)
    trading_rates = np.diff(trajectory) * (-1)
    
    return {
        'trajectory': np.array(trajectory),
        'trading_rates': trading_rates,
        'times': times
    }

ac_schedule = almgren_chriss_optimal_schedule(
    Q=total_quantity,
    T=10,
    sigma=volatility,
    lambda_temp=0.1,
    risk_aversion=1e-6
)

print(f"\nOptimal Execution Trajectory:")
print(f"   Initial Holdings: {ac_schedule['trajectory'][0]:,.0f} shares")
print(f"   Trading Rate: {ac_schedule['trading_rates'][0]:,.0f} shares/period")
print(f"   Final Holdings: {ac_schedule['trajectory'][-1]:.0f} shares")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cost breakdown by strategy
ax1 = axes[0, 0]
strategy_names = [s['strategy'] for s in strategies]
cost_components = {
    'Explicit': [s['explicit'] for s in strategies],
    'Spread': [s['spread'] for s in strategies],
    'Impact': [s['impact'] for s in strategies],
    'Opportunity': [s['opportunity'] for s in strategies]
}

x = np.arange(len(strategy_names))
width = 0.2
colors = ['gray', 'orange', 'red', 'purple']

for i, (component, values) in enumerate(cost_components.items()):
    ax1.bar(x + i*width, values, width, label=component, color=colors[i], alpha=0.7)

ax1.set_xlabel('Strategy')
ax1.set_ylabel('Cost ($)')
ax1.set_title('Execution Cost Breakdown by Component')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels([s.replace(' (', '\n(') for s in strategy_names], fontsize=8)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Total cost comparison
ax2 = axes[0, 1]
total_costs = [s['total'] for s in strategies]
total_bps = [s['total_bps'] for s in strategies]
bars = ax2.bar(strategy_names, total_costs, color=['red', 'orange', 'yellow', 'green'], 
              edgecolor='black', alpha=0.7)
ax2.set_ylabel('Total Cost ($)')
ax2.set_title('Total Transaction Cost Comparison')
ax2.set_xticklabels([s.replace(' (', '\n(') for s in strategy_names], fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# Add bps labels
for bar, bps in zip(bars, total_bps):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{bps:.1f} bps', ha='center', va='bottom', fontsize=9)

# Plot 3: Market impact vs slices
ax3 = axes[1, 0]
n_slices_range = range(1, 21)
impact_vs_slices = []
total_cost_vs_slices = []

for n in n_slices_range:
    if n == 1:
        result = aggressive_execution(total_quantity, initial_price, daily_volume)
    else:
        result = patient_execution(total_quantity, initial_price, daily_volume, n_slices=n)
    impact_vs_slices.append(result['impact'])
    total_cost_vs_slices.append(result['total'])

ax3.plot(n_slices_range, impact_vs_slices, 'o-', linewidth=2, markersize=6, label='Market Impact')
ax3_twin = ax3.twinx()
ax3_twin.plot(n_slices_range, total_cost_vs_slices, 's-', color='green', 
             linewidth=2, markersize=6, label='Total Cost')

ax3.set_xlabel('Number of Slices')
ax3.set_ylabel('Market Impact Cost ($)', color='blue')
ax3_twin.set_ylabel('Total Cost ($)', color='green')
ax3.set_title('Tradeoff: Slicing vs Costs')
ax3.tick_params(axis='y', labelcolor='blue')
ax3_twin.tick_params(axis='y', labelcolor='green')
ax3.grid(True, alpha=0.3)

# Plot 4: Almgren-Chriss optimal trajectory
ax4 = axes[1, 1]
ax4.plot(ac_schedule['times'], ac_schedule['trajectory'], 'b-', linewidth=3, 
        marker='o', markersize=6, label='Optimal Holdings')
ax4.fill_between(ac_schedule['times'], 0, ac_schedule['trajectory'], alpha=0.2)
ax4.set_xlabel('Time Period')
ax4.set_ylabel('Remaining Quantity (shares)')
ax4.set_title('Almgren-Chriss Optimal Execution Trajectory')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('execution_costs_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: execution_costs_analysis.png")
plt.show()
```

## 6. Challenge Round
When do execution cost models break down?
- Flash crash: Spread explodes 100x → Cost model assumes normal spreads → Catastrophic underestimate
- Illiquidity: Market depth vanishes → Impact model linear → Actually exponential at extremes
- Information leakage: Predictable algo pattern → Front-runners detect → Implicit costs surge
- Wrong benchmark: Use closing price → Market rallies all day → Looks like bad execution, actually unavoidable
- Hidden costs: Internalization with price improvement claimed → Actually worse than NBBO after selection bias

## 7. Key References
- [Almgren & Chriss (2001), "Optimal Execution"](https://www.jstor.org/stable/2645747) - Theoretical framework for cost-risk tradeoff
- [Kissell (2011), "Transaction Cost Analysis"](https://www.wiley.com/en-us/Algorithmic+Trading+Methods-p-9780470643112) - Practitioner TCA guide  
- [Perold (1988), "Implementation Shortfall"](https://www.jstor.org/stable/2328955) - Cost measurement methodology

---
**Status:** Fundamental to execution quality | **Complements:** Execution algorithms, TCA, market microstructure | **Drives:** Algo selection, venue routing