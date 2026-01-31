# Temporary Impact

## 1. Concept Skeleton
**Definition:** Short-term price movement from order flow that reverses; liquidity effect; mean-reverts to pre-trade level  
**Purpose:** Quantify execution costs separate from information effects; optimize execution timing; measure market impact decay  
**Prerequisites:** Bid-ask spreads, market microstructure, inventory dynamics, liquidity provision

## 2. Comparative Framing
| Component | Duration | Recovery | Cause | Reversibility |
|-----------|----------|----------|-------|--------------|
| **Temporary** | Seconds/minutes | Rapid | Liquidity/inventory | Reverts fully |
| **Permanent** | Hours/days | None | Information | Stays |
| **Transient** | Microseconds | Instant | Bid-ask bounce | Reverts instantly |
| **Latent** | Days/weeks | Gradual | Regime change | Partial reversion |

## 3. Examples + Counterexamples

**Temporary Impact Clear:**  
Large market order hits exchange → spreads widen $0.01 temporarily → liquidity providers step in → within 5 minutes, spreads return to normal → price where it was adjusted for true move only

**Permanent Mislabeled as Temporary:**  
Positive earnings surprise → stock jumps $5 → doesn't come back down → thought temporary but actually permanent (new information) → missed signal by calling it temporary

**Mixed Impact Confusion:**  
$50M sell order drops price $3 → $2 sticks (permanent, panic selling overshot) → $1 bounces back (temporary, liquidity dry up) → total effect $3, but composition unclear initially

**Liquidity Crisis:**  
Market stress → temporary impact becomes extreme $0.50 → takes hours to revert instead of minutes → temporary becomes semi-permanent (liquidity freeze) → normal reversal doesn't apply

## 4. Layer Breakdown
```
Temporary Impact Framework:
├─ Nature and Origin:
│   ├─ Inventory Dynamics:
│   │   - MM accumulates inventory from incoming trade
│   │   - Wants to rebalance (not hold risk)
│   │   - Widens quotes to encourage opposite trades
│   │   - After rebalancing, quotes normalize
│   │   - Effect: Price moves and reverts
│   ├─ Liquidity Provision:
│   │   - Limited depth at each price level
│   │   - Large order exhausts available liquidity
│   │   - Price must move to attract more sellers/buyers
│   │   - Once book replenishes, moves back
│   │   - Mechanism: Supply/demand imbalance clearing
│   ├─ Bid-Ask Bounce:
│   │   - Trade can occur on buy or sell side
│   │   - Observed price oscillates: ask → mid → bid
│   │   - Pure artifacts of trading mechanism
│   │   - Not fundamental price change
│   │   - Measurement: Use mid-quote instead of trade
│   ├─ Execution Timing:
│   │   - Order doesn't execute at single price
│   │   - Gets filled across multiple levels
│   │   - Execution quality varies intraday
│   │   - High volume periods: Large temporary impact
│   │   - Low volume periods: Small temporary impact
│   └─ Market Microstructure Noise:
│       - Definition: Price variation not driven by info
│       - Source: Bid-ask spread crossing, discretization
│       - Magnitude: 50-80% of high-frequency price moves
│       - Implication: Much HF price movement is noise
│
├─ Measurement Approaches:
│   ├─ Mean Reversion Detection:
│   │   - Hypothesis: Price moves are temporary
│   │   - Test: Check if price reverts to prior level
│   │   - Method: Correlation of returns at different frequencies
│   │   - Negative autocorrelation: Evidence of mean reversion
│   │   - Typical: Returns at 1-min frequency negatively correlated
│   ├─ Inventory Adjustment:
│   │   - Before: MM has target inventory
│   │   - Trade: Inventory changes
│   │   - After: MM rebalances
│   │   - Rebalancing: Quotes adjust to move inventory
│   │   - Measurement: Track MM inventory over time
│   ├─ Quote Dynamics:
│   │   - Bid-ask spread: Widens when MM absorbs order
│   │   - Recovery: Spread narrows as other traders arrive
│   │   - Timing: How fast does spread recover?
│   │   - Typical: 80% recovery in 10-30 seconds
│   ├─ Half-Life Measurement:
│   │   - Definition: Time for 50% of impact to revert
│   │   - Calculation: Regression of price level on time
│   │   - Exponential decay: p(t) = p_0 + impact × e^(-t/τ)
│   │   - τ = half-life; typical 10-100 milliseconds
│   ├─ Lagged Price Regression:
│   │   - Model: Price change = f(lagged changes)
│   │   - Negative coefficient: Mean reversion present
│   │   - Magnitude: Coefficient size = temporary fraction
│   │   - Advantage: Simple implementation
│   └─ Variance Decomposition:
│       - Total variance: Information + temporary noise
│       - Method: Filter data at different frequencies
│       - Result: Low-frequency variance = info, high-frequency = noise
│       - Application: Separate signal from noise
│
├─ Empirical Magnitudes:
│   ├─ Equity Markets:
│   │   - Large cap: ~20-30% of immediate impact reverts in 1 minute
│   │   - Small cap: ~10-20% reverts in 1 minute
│   │   - Highly liquid: Reverts faster (60% in 10 seconds)
│   │   - Illiquid: Reverts slower (takes hours)
│   │   - Typical half-life: 10-100 milliseconds
│   ├─ By Venue:
│   │   - Open/close auctions: Large temporary impact (low liquidity)
│   │   - Midday: Small temporary impact (high liquidity)
│   │   - After-hours: Extreme temporary impact (very illiquid)
│   │   - Overnight gaps: Partial recovery at open
│   ├─ By Order Type:
│   │   - Market orders: Highest temporary impact
│   │   - Iceberg orders: Lowest (hidden until close)
│   │   - VWAP sliced: Medium (average temporary)
│   │   - Block trades: Moderate if negotiated properly
│   ├─ By Market Conditions:
│   │   - Calm: ~25% temporary, 75% permanent
│   │   - Normal: ~30-40% temporary, 60-70% permanent
│   │   - Volatile: ~40-50% temporary, 50-60% permanent
│   │   - Crisis: ~50-60% temporary, 40-50% permanent
│   ├─ Asset Class Differences:
│   │   - Equities: 30-40% typical
│   │   - Options: 20-35% (different microstructure)
│   │   - Futures: 40-50% (more noise)
│   │   - FX: 35-45% (decentralized market)
│   └─ Time Decay Pattern:
│       - Immediate: 100% temporary impact present
│       - 10ms: ~70% remains
│       - 100ms: ~40% remains
│       - 1 second: ~15% remains
│       - 10 seconds: ~5% remains
│       - 1 minute: ~1-2% remains (mostly gone)
│
├─ Inventory-Based Models:
│   ├─ Stoll Model (1978):
│   │   - MM targets inventory level
│   │   - Trade changes inventory
│   │   - Spread adjusts to encourage rebalancing
│   │   - Temporary: Spread widening incentivizes reversal
│   │   - Formula: Spread = adverse_selection + 2×inventory_cost
│   ├─ Avellaneda-Stoikov (2008):
│   │   - Stochastic control approach
│   │   - Optimal bid-ask quotes minimize risk
│   │   - Quotes follow inventory level
│   │   - Temporary: Quotes return to baseline as inventory rebalances
│   │   - Application: Algorithm design for market making
│   ├─ Mean-Reversion Inventory Target:
│   │   - Target: Zero (or hedged level)
│   │   - Divergence: Current - Target
│   │   - Rebalancing force: Proportional to divergence
│   │   - Result: Exponential mean reversion
│   │   - Time constant: Reflects MM risk aversion
│   └─ Empirical Inventory Evidence:
│       - Hasbrouck (1991): ~60-70% of spread inventory component
│       - Stoll (1989): Inventory explains price changes over seconds
│       - Madhavan et al (1997): Confirmed inventory effect
│       - Recent: Still dominant effect in modern markets
│
├─ Execution Algorithm Applications:
│   ├─ Optimal Execution Timing:
│   │   - Objective: Minimize execution cost
│   │   - Trade-off: Market impact vs timing risk
│   │   - Insight: Temporary reverts quickly
│   │   - Strategy: Execute when temporary highest (low liquidity)
│   │   - Benefit: Temporary cost will revert before next trade
│   ├─ Slicing Strategy:
│   │   - Large order: Break into small pieces
│   │   - Benefit: Avoids large temporary impact
│   │   - Cost: Takes longer, exposed to price moves
│   │   - Optimal: Balance impact reduction vs timing risk
│   │   - Example: 100K order → 10 × 10K over 1 hour
│   ├─ Intraday Volume Curves:
│   │   - U-shaped: High volume open/close, low midday
│   │   - Reason: Rebalancing (portfolio adjustments)
│   │   - Implication: Execute during low volume (less temporary)
│   │   - Strategy: Avoid open/close (high temporary impact)
│   ├─ Liquidity Seeking:
│   │   - Passive approach: Post limit orders
│   │   - Benefit: Avoid temporary impact entirely
│   │   - Cost: Timing risk if not filled
│   │   - Hybrid: Aggressive when spreads tight, passive when wide
│   └─ Dark Pool Execution:
│       - Concept: Avoid public impact
│       - Temporary: Still occurs in dark pool (MM dynamics)
│       - Permanent: Delayed discovery reduces permanent
│       - Net effect: Lower total impact than public execution
│
├─ Technical Analysis and Temporary:
│   ├─ False Signals:
│   │   - Reversal patterns: Often temporary impact reversions
│   │   - Not true reversal: Just inventory rebalancing
│   │   - Confusion: Technical traders mistake temporary for signal
│   │   - Reality: Most intraday reversals are temporary
│   ├─ Support/Resistance:
│   │   - Psychological levels: May be real but weak
│   │   - Temporary: Many bounces from support
│   │   - Mechanism: Inventory rebalancing, not level support
│   │   - Caution: Don't over-weight technical levels
│   ├─ Momentum Illusion:
│   │   - Observed: Prices trending during heavy trading
│   │   - Misattribution: Thought to be fundamental momentum
│   │   - Reality: Temporary impact accumulation
│   │   - Reversion: When trades slow, temporary reverts
│   └─ Mean Reversion Strategies:
│       - Concept: Trade reversals of temporary impacts
│       - Setup: Price spikes extreme, bet on reversion
│       - Timing: Must execute within reversion half-life
│       - Risk: Real trends vs temporary (hard to distinguish)
│       - Returns: Low (competitive, traded heavily)
│
├─ Liquidity Measurement:
│   ├─ Roll Estimator:
│   │   - Formula: Spread = 2√(-cov(Δp_t, Δp_{t-1}))
│   │   - Intuition: Negative autocorrelation = bid-ask bounce
│   │   - Application: Estimate spread without trade data
│   │   - Advantage: Works from price data alone
│   │   - Limitation: Assumes covariance from bid-ask only
│   ├─ Effective Spread:
│   │   - Definition: 2 × |trade price - midpoint|
│   │   - Includes: Both bid-ask and price impact
│   │   - Temporary: Part of effective (mean-reverts)
│   │   - Permanent: Part of effective (doesn't revert)
│   │   - Measurement: Use mid-quote before/after trade
│   ├─ Amihud Illiquidity:
│   │   - Formula: |Return| / Volume (in dollars)
│   │   - Interpretation: Price impact per dollar traded
│   │   - Temporary: Correlates with illiquidity measure
│   │   - Application: Compare liquidity across assets/time
│   └─ High-Frequency Measures:
│       - VPIN: Volume-synchronized PIN (intraday)
│       - NTVOL: Noise-to-true-volatility ratio
│       - Advantage: Real-time liquidity monitoring
│       - Limitation: Requires high-frequency data
│
├─ Cross-Asset and Systemic Implications:
│   ├─ Liquidity Commonality:
│   │   - Concept: Liquidity dries up across assets simultaneously
│   │   - Cause: Market makers reduce inventory targets
│   │   - Effect: Temporary impact spikes for all assets
│   │   - Risk: Correlated liquidity shocks → cascades
│   ├─ Index Arbitrage:
│   │   - Strategy: Arbitrage S&P 500 futures vs ETF
│   │   - Mechanism: Execute when temporary impact highest
│   │   - Profit: Capture mean reversion of temporary
│   │   - Competition: Erodes profits as others do it too
│   ├─ Flash Crash Context:
│   │   - May 6, 2010: Temporary impact extreme (500ms collapse)
│   │   - Cause: Liquidity evaporation + algorithmic cascade
│   │   - Normally temporary: Reverts in minutes
│   │   - Crisis mode: Takes longer or partial recovery
│   └─ Correlation with Volatility:
│       - High volatility: Larger temporary impact
│       - Reason: MMs risk averse, quotes widen
│       - Measurement: Temporary × volatility correlate
│       - Implication: Temporary worse during stress
│
└─ Advanced Topics:
    ├─ Nonlinear Decay:
    │   - Assumption: Exponential decay
    │   - Reality: Often power-law decay
    │   - Implication: Slower reversion than assumed
    │   - Model: impact ∝ t^(-α), α ≈ 0.5-1.0
    ├─ State-Dependent Temporary:
    │   - Inventory level: Affects reversion speed
    │   - Market sentiment: Changes MM behavior
    │   - Liquidity: Low liquidity = slower reversion
    │   - Empirical: No single temporary impact size
    ├─ Predatory Strategies:
    │   - Front-running: Execute before known trades
    │   - Exploit temporary: Trade into dislocations
    │   - Regulation: Prohibited under Dodd-Frank
    │   - Detection: Hard due to market complexity
    └─ Permanent vs Temporary Blurring:
        - Time horizon dependent: Permanent at 1hr = temporary at 1yr
        - Definition: No clear boundary between
        - Practical: Use "temporary" for sub-minute, "permanent" for rest
        - Challenge: Categorization somewhat arbitrary
```

**Interaction:** Large market order fills across price levels → spread widens as MM inventory loaded → other traders arrive sensing opportunity → MM rebalances → spread narrows → price partially reverts to pre-order level

## 5. Mini-Project
Simulate temporary impact decay and optimal execution timing:
```python
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
```

## 6. Challenge Round
Why do algorithmic trading systems specifically target temporary impact if it reverts anyway—shouldn't the reversion happen automatically without intervention?

- **Speed advantage**: Reversion is automatic, but mean reversion takes time (seconds/minutes) → fast traders profit by stepping in first → front-run the reversion → extract value before slow traders
- **Information exploitation**: Other traders don't know temporary vs permanent → trade the reversion as if it's informative → fast traders use knowledge of mean reversion to profit from their mistakes
- **Cascade avoidance**: Reversion doesn't happen if no one steps in → prices can stick away from equilibrium → algos provide liquidity that enables reversion → profit from providing service
- **Latency arbitrage**: See temporary move on one venue → trade on another before reversion → profit from arbitrage → wouldn't exist if all venues updated simultaneously (but they don't due to latency)
- **Statistical opportunity**: Temporary impact is predictable → trade the pattern → becomes self-fulfilling prophesy → strategies work until everyone does it, then profits disappear (arms race)

## 7. Key References
- [Hasbrouck (1991) - Measuring Effects of Data Aggregation on Price Discovery](https://www.jstor.org/stable/2328955)
- [Roll (1984) - A Simple Implicit Measure of the Effective Bid-Ask Spread](https://www.jstor.org/stable/2327617)
- [Avellaneda & Stoikov (2008) - High-Frequency Trading in a Limit Order Book](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf)
- [Madhavan et al (1997) - Why Do Security Prices Change?](https://www.jstor.org/stable/2329541)

---
**Status:** Liquidity/inventory-driven short-term reversal | **Complements:** Permanent Impact, Mean Reversion, Inventory Dynamics, Execution Algorithms
