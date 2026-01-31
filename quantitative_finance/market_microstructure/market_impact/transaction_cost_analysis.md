# Transaction Cost Analysis (TCA)

## 1. Concept Skeleton
**Definition:** Systematic measurement and decomposition of execution costs comparing actual execution against benchmarks (VWAP, arrival price, implementation shortfall); identifies cost sources and trader performance  
**Purpose:** Evaluate execution quality, benchmark trading desks, optimize order routing decisions, quantify slippage and market impact contributions to total cost  
**Prerequisites:** Market microstructure, order execution algorithms, price benchmarking methodologies, statistical attribution techniques

## 2. Comparative Framing
| Benchmark | Formula | Use Case | Timing Issue | Performance Target |
|-----------|---------|----------|--------------|-------------------|
| **Arrival Price (AP)** | Cost = Trade_price - Entry_price | Evaluate post-trade decision quality | Known at trade time | 0 (no market impact) |
| **VWAP** | Cost = Σ(P×Q) / ΣQ benchmark | Compare vs volume-weighted execution | Calculated intraday | ±0.5-1% |
| **TWAP** | Cost = ΣP / n equally-weighted | Measure vs time-weighted execution | Simple averaging | ±1-2% |
| **Closing Price** | Cost = Trade_price - Close | Evaluate end-of-day fills | Known after close | ±1-2% |
| **Implementation Shortfall** | Cost = (Expected_return - Actual_return) × Notional | Isolate execution vs decision effect | Decomposes both | Minimize total |
| **Benchmark-relative** | Cost vs custom basket/index | Sector/factor-adjusted cost | Index-specific | Relative to peers |

## 3. Examples + Counterexamples

**VWAP Success (Liquid Stock):**  
Execute 100k shares of Apple; VWAP benchmark = $150.25; actual execution = $150.18.  
Cost: $150.18 - $150.25 = -$0.07 (savings, favorable execution); 7 basis points outperformance.

**VWAP Failure (Illiquid Stock):**  
Execute 50k shares of micro-cap; VWAP = $12.50; actual = $13.25.  
Cost: -$0.75 (unfavorable); market impact exceeded benchmark. VWAP not suitable benchmark for illiquid names.

**Implementation Shortfall Decomposition (Fund Manager):**  
Decision: Buy $10M position of tech ETF at $100 (closing price day before).  
Execution: Takes 3 days, average fill $101.25. Market moves to $102 close.  
Timing cost: $10M × ($100 - $102) / $100 = -$200k (bad timing, market moved higher).  
Execution cost: $10M × ($101.25 - $100) / $100 = +$125k (execution premium paid).  
Total shortfall: -$200k + $125k = -$75k (net loss vs original target).

**TWAP over VWAP (Momentum Stock):**  
Volume distribution U-shaped (high open/close, low midday). TWAP assumes equal time-weighting → executes more at midday (low volume) → higher impact cost than VWAP which follows volume curve.

**Arriving Too Late (Regret Case):**  
Client decision: Buy at $50. Actual execution at $52. Market continues to $55. Regret costs: Timing decision was right (bought before rally) but execution poor (expensive). TCA reveals timing good, execution bad (separate evaluation).

## 4. Layer Breakdown
```
Transaction Cost Analysis Framework:

├─ Benchmark Selection:
│  ├─ Pre-Trade Benchmarks (known at decision time):
│  │  ├─ Arrival Price (AP):
│  │  │  ├─ Definition: Price at first trade execution
│  │  │  ├─ Advantage: Simple, objective, known immediately
│  │  │  ├─ Disadvantage: Doesn't isolate market impact (trade causes price move)
│  │  │  ├─ Cost = E[Trade_price] - Arrival_price
│  │  │  ├─ Typical range: ±20-50 bps
│  │  │  └─ Usage: Quick, real-time performance metrics
│  │  ├─ Decision Price (Entry Price):
│  │  │  ├─ Price trader decided to enter (e.g., analyst recommendation)
│  │  │  ├─ Maximizes trader skill opportunity (includes all post-decision costs)
│  │  │  ├─ Subject to regret (market could move against trade)
│  │  │  ├─ Cost = Trade_price - Decision_price
│  │  │  ├─ Typical range: ±100-300 bps
│  │  │  └─ Usage: Evaluate total decision+execution quality
│  │  └─ Policy Price (Predetermined Level):
│  │     ├─ Pre-agreed execution target (e.g., "buy within 2% of close")
│  │     ├─ Incentivizes adherence to plan
│  │     └─ Cost = Trade_price - Policy_price
│  │
│  ├─ Intraday Benchmarks (observed during execution):
│  │  ├─ VWAP (Volume-Weighted Average Price):
│  │  │  ├─ Definition: Σ(Price × Volume) / ΣVolume
│  │  │  ├─ Advantage: Naturally reflects volume distribution
│  │  │  ├─ Disadvantage: Not known fully until close; subject to manipulation (spoofing)
│  │  ├─ TWAP (Time-Weighted Average Price):
│  │  │  ├─ Definition: (1/n) × ΣPrice_i equally-weighted
│  │  │  ├─ Advantage: Simple calculation; independent of volume
│  │  │  ├─ Disadvantage: Doesn't track market liquidity; suboptimal for volume-skewed days
│  │  ├─ Percentage of Volume (POV):
│  │  │  ├─ Definition: Participate at X% of market volume
│  │  │  ├─ Advantage: Adapts to market conditions; market-neutral
│  │  │  ├─ Disadvantage: Limited control; depends on market participation
│  │  └─ Market Impact Adjusted:
│  │     ├─ Benchmark includes expected market impact
│  │     ├─ Benchmark = VWAP + (expected_impact_bps × notional)
│  │     ├─ Advantage: Fair performance assessment (controls for trade size)
│  │     └─ Disadvantage: Requires impact model estimation
│  │
│  └─ Post-Trade Benchmarks (known after execution complete):
│     ├─ Day-End Close:
│     │  ├─ Definition: Price at market close of execution day
│     │  ├─ Advantage: Universally known; stable
│     │  ├─ Disadvantage: Subject to closing auction dynamics; not market-neutral
│     │  └─ Cost = Trade_price - Close_price
│     ├─ Next Day Open:
│     │  ├─ Definition: Gap-neutral after overnight market move
│     │  ├─ Advantage: Isolates intraday execution (excludes overnight)
│     │  ├─ Disadvantage: Includes after-hours trading effects
│     │  └─ Cost = Trade_price - Next_open
│     ├─ Time-Horizon Adjusted:
│     │  ├─ Definition: Price T minutes/hours after execution complete
│     │  ├─ Advantage: Distinguishes temporary vs permanent impact
│     │  ├─ Disadvantage: Arbitrary T selection
│     │  └─ Cost = Trade_price - Price_T_later
│     └─ Index/Sector Neutral:
│        ├─ Definition: (Trade_return - Index_return) normalized
│        ├─ Advantage: Controls for market moves; factor-neutral
│        ├─ Disadvantage: Requires beta estimation
│        └─ Cost = Relative_performance_vs_benchmark
│
├─ Cost Decomposition (Implementation Shortfall):
│  ├─ Components:
│  │  ├─ Timing Cost (Decision Effect):
│  │  │  ├─ Definition: Return if decided to NOT execute vs decision price
│  │  │  ├─ Cost_timing = (Market_return - 0) × Notional
│  │  │  ├─ Formula: (Final_price - Decision_price) × shares
│  │  │  ├─ Positive cost = Market moved against decision (regret)
│  │  │  ├─ Negative cost = Favorable market move (luck)
│  │  │  ├─ Uncontrollable by trader (market-dependent)
│  │  │  └─ Typical: ±100-500 bps
│  │  │
│  │  ├─ Execution Cost (Performance Effect):
│  │  │  ├─ Definition: Cost of execution vs arrival price
│  │  │  ├─ Cost_exec = (Execution_price - Arrival_price) × shares
│  │  │  ├─ Positive cost = Paid more than necessary (unfavorable execution)
│  │  │  ├─ Negative cost = Got better price (favorable execution)
│  │  │  ├─ Directly controllable by algo design
│  │  │  └─ Typical: ±10-50 bps (liquid), ±50-200 bps (illiquid)
│  │  │
│  │  ├─ Spread Cost:
│  │  │  ├─ Definition: Half-spread cost on each trade leg
│  │  │  ├─ Cost_spread = 0.5 × (Ask - Bid) / Midpoint
│  │  │  ├─ Unavoidable friction
│  │  │  ├─ Typical: 1-5 bps (large-cap), 10-100 bps (small-cap)
│  │  │  └─ Controllable via venue selection, order type
│  │  │
│  │  ├─ Market Impact Cost:
│  │  │  ├─ Definition: Permanent price move caused by order flow
│  │  │  ├─ Cost_impact = (Execution_mid - Arrival_mid) - 0.5×spread
│  │  │  ├─ Non-linear in order size (square-root law)
│  │  │  ├─ Controllable via execution pace (slow = lower)
│  │  │  └─ Typical: 5-50 bps for 1-5% of volume
│  │  │
│  │  ├─ Opportunity Cost (Incomplete Execution):
│  │  │  ├─ Definition: Unrealized profit if order not fully filled
│  │  │  ├─ Cost_oppty = (Final_price - Unfilled_price) × unfilled_shares
│  │  │  ├─ Tradeoff: Fill fast (pay impact) vs. patient (miss price)
│  │  │  ├─ Relevant for POV/passive algos
│  │  │  └─ Typical: ±50-200 bps
│  │  │
│  │  └─ Slippage (Aggregate):
│  │     ├─ Definition: Cost_exec = spread + impact + opportunity
│  │     ├─ Varies by venue, time, market conditions
│  │     └─ TCA decomposes these contributors
│  │
│  ├─ Calculation Framework:
│  │  ├─ Total Cost:
│  │  │   │   Cost_total = (Final_price - Decision_price) × shares
│  │  │   │   = Cost_timing + Cost_execution
│  │  │   │
│  │  ├─ Execution Cost (via arrival price):
│  │  │   │   Cost_execution = (Execution_avg - Arrival_price) × shares
│  │  │   │   = Cost_spread + Cost_impact + Cost_opportunity
│  │  │   │
│  │  ├─ Example Decomposition:
│  │  │   │   Decision price: $100.00
│  │  │   │   Arrival price: $100.50 (market moved while order was placed)
│  │  │   │   Execution price: $100.75 (filled at worse price)
│  │  │   │   Final price (30min later): $101.00 (price continued rising)
│  │  │   │
│  │  │   │   Timing cost = ($101.00 - $100.00) = $1.00 (unfavorable)
│  │  │   │   Execution cost = ($100.75 - $100.50) = $0.25 (unfavorable)
│  │  │   │   Total cost = $1.25
│  │  │   │
│  │  └─ Attribution Analysis:
│  │       │   Timing (80%): $1.00 / $1.25 = 80% (market move)
│  │       │   Execution (20%): $0.25 / $1.25 = 20% (algo performance)
│  │       │
│  └─ Benchmark Selection Tree:
│     ├─ If executing liquid large-cap: VWAP (standard)
│     ├─ If executing illiquid/small-cap: Arrival price + impact model
│     ├─ If performance-sensitive (mutual fund): Implementation shortfall
│     ├─ If real-time monitoring: TWAP or POV
│     └─ If risk management critical: Index-neutral or sector-neutral
│
├─ Advanced TCA Techniques:
│  ├─ Multi-Benchmark Analysis:
│  │  ├─ Compare cost vs. multiple benchmarks simultaneously
│  │  ├─ VWAP: 95 bps cost (above volume-weighted)
│  │  ├─ AP: 70 bps cost (above arrival)
│  │  ├─ TWAP: 110 bps cost (above equal-weight)
│  │  ├─ Venue-specific insights (which venue best?)
│  │  └─ Algorithm selection validation
│  │
│  ├─ Peer Benchmarking:
│  │  ├─ Compare execution cost vs. other traders/algos
│  │  ├─ Identify high-cost days/scenarios
│  │  ├─ Best-practice knowledge transfer
│  │  └─ Typically: Quartile rankings (1=best, 4=worst)
│  │
│  ├─ Conditional TCA (By Market State):
│  │  ├─ Segment by volatility (high vol → higher expected costs)
│  │  ├─ Segment by volume (low vol days → higher impact)
│  │  ├─ Segment by market stress (crises → wider spreads)
│  │  ├─ Adjust benchmarks for conditions
│  │  └─ Fair performance assessment
│  │
│  ├─ Pre/Post Comparison:
│  │  ├─ Expected cost (model prediction) vs. actual (realized)
│  │  ├─ Identifies consistently better/worse execution
│  │  ├─ Algorithm tuning feedback
│  │  └─ Model calibration
│  │
│  └─ Venue Analysis:
│     ├─ Which venue gives best execution?
│     ├─ Large-cap: Exchange (tighter spreads)
│     ├─ Small-cap: OTC/ATS (better depth)
│     ├─ Route optimization
│     └─ Rebate capture analysis
│
├─ TCA Reporting & Metrics:
│  ├─ Standard Metrics:
│  │  ├─ Execution cost (bps): Cost / Notional × 10000
│  │  ├─ Slippage (bps): (Trade_price - Arrival_price) / Arrival_price
│  │  ├─ Implementation shortfall: Cost_total / Notional
│  │  ├─ Fill rate: Executed_shares / Intended_shares
│  │  ├─ Execution time: Duration of order
│  │  └─ Price improvement: Positive bps vs. benchmark
│  │
│  ├─ Advanced Metrics:
│  │  ├─ Sharpe ratio of execution: Return_vs_benchmark / Volatility
│  │  ├─ Information ratio: α_execution / Tracking_error
│  │  ├─ Percentile ranking: vs. peer group or historical
│  │  ├─ Risk-adjusted cost: Cost / Volatility
│  │  └─ Cost per unit of impact: Cost / Market_impact_estimate
│  │
│  ├─ Segmentation & Drill-Down:
│  │  ├─ By: Time of day, market cap, sector, country
│  │  ├─ Identify high-cost segments
│  │  ├─ Targeted algorithm improvements
│  │  └─ Resource allocation
│  │
│  └─ Dashboards:
│     ├─ Real-time TCA (during trading)
│     ├─ Post-trade TCA (next day reporting)
│     ├─ Monthly TCA reviews
│     └─ Ad-hoc analysis (special requests)
│
├─ Practical Challenges:
│  ├─ Benchmark Selection Bias:
│  │  ├─ Different benchmarks → different conclusions
│  │  ├─ VWAP may be unfair if volume skewed
│  │  ├─ Arrival price inflated by queue delay
│  │  └─ Must justify choice
│  │
│  ├─ Market-Impact Confounding:
│  │  ├─ Own trades cause price move (permanent impact)
│  │  ├─ Separating impact from market move difficult
│  │  ├─ Need microstructure data (order-by-order)
│  │  └─ Risk: Over-crediting execution skill
│  │
│  ├─ Incomplete Information:
│  │  ├─ Dark pool executions (price hidden)
│  │  ├─ OTC trades (size unknown)
│  │  ├─ Synthetic benchmarking required
│  │  └─ Estimation error in TCA
│  │
│  ├─ Timing of Benchmarks:
│  │  ├─ When calculate VWAP? (full day, trading hours, trade period?)
│  │  ├─ Open/close auctions skew benchmarks
│  │  ├─ Different calculation → different conclusions
│  │  └─ Standardize definition
│  │
│  └─ Survivorship Bias:
│     ├─ Exclude cancelled/rejected orders from TCA
│     ├─ May have had poor execution (why cancelled?)
│     ├─ Biases results upward
│     └─ Include all attempts
│
└─ Technology & Systems:
   ├─ TCA Platforms:
   │  ├─ Bloomberg AFPT (Arrival Price TCA)
   │  ├─ ITG Posit Analytics
   │  ├─ Greenlight TCA Suite
   │  ├─ Abel Noser/Cass
   │  └─ Custom in-house systems
   │
   ├─ Data Requirements:
   │  ├─ Order-level: Submission time, size, price, fills
   │  ├─ Trade-level: Execution time, venue, price, size
   │  ├─ Reference data: Benchmark prices (VWAP, AP, TWAP)
   │  ├─ Market data: Quotes, trades, volume (tick data)
   │  └─ Latency: Real-time (msec), not delayed (data quality)
   │
   ├─ Data Challenges:
   │  ├─ Venue fragmentation (multi-exchange execution)
   │  ├─ Time synchronization (exchange clocks differ)
   │  ├─ Incomplete fills (partial execution across venues)
   │  ├─ FX conversion (international trades)
   │  └─ Adjustments (dividends, splits, corporate actions)
   │
   └─ Automation:
      ├─ Daily TCA computation
      ├─ Peer group comparisons
      ├─ Alert on outliers (high-cost executions)
      ├─ Trend analysis (improving/degrading over time)
      └─ Feedback loop to traders/algos
```

**Interaction:** Define benchmarks → Execute order → Measure costs → Decompose components → Compare to peers → Feedback for improvement.

## 5. Mini-Project
Implement TCA analysis comparing multiple benchmarks on intraday execution:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from datetime import datetime, timedelta

# Generate realistic intraday data with VWAP, trades, price movement
np.random.seed(42)
trading_minutes = 390  # 6.5 hours
minutes = np.arange(trading_minutes)

# Underlying price follows random walk with drift
true_drift = 0.0002
true_vol = 0.05 / np.sqrt(252 * 390)  # Annualized vol scaled to minutes
price_move = np.random.normal(true_drift, true_vol, trading_minutes)
base_price = 100.0
intraday_prices = base_price + np.cumsum(price_move)

# Volume follows U-shape (higher at open/close)
volume_base = 1000 * (1 + 0.5 * np.sin(np.pi * minutes / trading_minutes))
volume = volume_base + np.random.normal(0, 100, trading_minutes)
volume = np.maximum(volume, 100)

# VWAP calculation (cumulative)
vwap_cumsum = np.cumsum(intraday_prices * volume)
vwap_cumvol = np.cumsum(volume)
vwap = vwap_cumsum / vwap_cumvol

# TWAP calculation
twap_cumsum = np.cumsum(intraday_prices)
twap = twap_cumsum / (minutes + 1)

# Simulate execution strategy: execute 50k shares gradually
total_shares_to_execute = 50000
start_minute = 30
end_minute = 360

# Execution profile: gentle ramp up, then taper down (to minimize impact)
exec_minutes = np.arange(start_minute, end_minute)
exec_profile = np.exp(-2 * ((exec_minutes - (start_minute + end_minute) / 2) / (end_minute - start_minute))**2)
exec_shares_per_minute = (total_shares_to_execute / np.sum(exec_profile)) * exec_profile
exec_shares_per_minute = np.round(exec_shares_per_minute).astype(int)
exec_shares_per_minute = np.concatenate([
    np.zeros(start_minute),
    exec_shares_per_minute,
    np.zeros(trading_minutes - end_minute)
])

# Ensure total matches
diff = total_shares_to_execute - np.sum(exec_shares_per_minute)
exec_shares_per_minute[end_minute - 1] += diff

# Execution price per minute (midpoint + half-spread + market impact)
spread = 0.01  # $0.01 bid-ask
market_impact_coef = 0.0001  # Impact = coef * sqrt(volume_fraction)

exec_prices = []
cum_executed = 0
for i in range(trading_minutes):
    if exec_shares_per_minute[i] > 0:
        # Market impact: sqrt of order size relative to volume
        order_frac = exec_shares_per_minute[i] / volume[i]
        impact = market_impact_coef * np.sqrt(order_frac)
        
        # Execution price: midpoint + half-spread + impact
        exec_price = intraday_prices[i] + 0.5 * spread + impact
        
        exec_prices.append(exec_price)
        cum_executed += exec_shares_per_minute[i]

# Calculate benchmarks
arrival_price = intraday_prices[start_minute]
execution_avg_price = np.mean(exec_prices)
final_price = intraday_prices[-1]
vwap_benchmark = vwap[-1]
twap_benchmark = twap[-1]

print("="*100)
print("TRANSACTION COST ANALYSIS (TCA)")
print("="*100)

print(f"\nStep 1: Market Conditions")
print(f"-" * 50)
print(f"Start price: ${arrival_price:.2f}")
print(f"End price: ${final_price:.2f}")
print(f"Market move: ${final_price - arrival_price:.2f} ({(final_price - arrival_price) / arrival_price * 100:.2f}%)")
print(f"Intraday volatility: {np.std(intraday_prices) * 100:.2f}%")
print(f"Total volume: {np.sum(volume):,.0f} shares")
print(f"Execution period: Minute {start_minute} to {end_minute}")
print(f"Total shares executed: {total_shares_to_execute:,.0f}")

print(f"\nStep 2: Benchmark Prices")
print(f"-" * 50)
print(f"Arrival Price (AP): ${arrival_price:.2f}")
print(f"VWAP: ${vwap_benchmark:.2f}")
print(f"TWAP: ${twap_benchmark:.2f}")
print(f"Execution Avg: ${execution_avg_price:.2f}")
print(f"Final Price: ${final_price:.2f}")

print(f"\nStep 3: Cost Analysis (Per Share, in Basis Points)")
print(f"-" * 50)

# Costs in dollars and bps
ap_cost_pershare = execution_avg_price - arrival_price
vwap_cost_pershare = execution_avg_price - vwap_benchmark
twap_cost_pershare = execution_avg_price - twap_benchmark

ap_cost_bps = (ap_cost_pershare / arrival_price) * 10000
vwap_cost_bps = (vwap_cost_pershare / vwap_benchmark) * 10000
twap_cost_bps = (twap_cost_pershare / twap_benchmark) * 10000

costs_df = pd.DataFrame({
    'Benchmark': ['Arrival Price', 'VWAP', 'TWAP'],
    'Benchmark Price': [arrival_price, vwap_benchmark, twap_benchmark],
    'Cost per Share': [ap_cost_pershare, vwap_cost_pershare, twap_cost_pershare],
    'Cost (bps)': [ap_cost_bps, vwap_cost_bps, twap_cost_bps],
    'Total Cost': [ap_cost_pershare * total_shares_to_execute,
                   vwap_cost_pershare * total_shares_to_execute,
                   twap_cost_pershare * total_shares_to_execute],
})

print(costs_df.to_string(index=False))

print(f"\nStep 4: Implementation Shortfall (Decision-Based)")
print(f"-" * 50)

decision_price = arrival_price  # Assume decision at arrival
timing_cost = (final_price - decision_price) * total_shares_to_execute
execution_cost = (execution_avg_price - arrival_price) * total_shares_to_execute
total_shortfall = timing_cost + execution_cost

print(f"Decision Price: ${decision_price:.2f}")
print(f"Execution Avg: ${execution_avg_price:.2f}")
print(f"Final Price: ${final_price:.2f}")
print(f"")
print(f"Timing Cost (Market Move): ${timing_cost:,.2f} ({timing_cost/final_price/total_shares_to_execute*10000:.2f} bps)")
print(f"Execution Cost (Algo): ${execution_cost:,.2f} ({execution_cost/arrival_price/total_shares_to_execute*10000:.2f} bps)")
print(f"Total Shortfall: ${total_shortfall:,.2f}")
print(f"Total Shortfall (bps): {total_shortfall/decision_price/total_shares_to_execute*10000:.2f} bps")

if timing_cost != 0:
    timing_pct = abs(timing_cost) / (abs(timing_cost) + abs(execution_cost)) * 100
    execution_pct = abs(execution_cost) / (abs(timing_cost) + abs(execution_cost)) * 100
    print(f"")
    print(f"Attribution:")
    print(f"  Timing (market move): {timing_pct:.1f}%")
    print(f"  Execution (algo): {execution_pct:.1f}%")

print(f"\nStep 5: Cost Decomposition (Slippage Components)")
print(f"-" * 50)

# Estimate components
spread_cost = 0.5 * spread * total_shares_to_execute  # Half-spread per share
# Market impact: sqrt-law
avg_order_frac = np.mean([exec_shares_per_minute[i] / volume[i] 
                           for i in range(len(exec_shares_per_minute)) 
                           if exec_shares_per_minute[i] > 0])
market_impact = market_impact_coef * np.sqrt(avg_order_frac) * arrival_price * total_shares_to_execute

print(f"Estimated Spread Cost: ${spread_cost:,.2f}")
print(f"Estimated Market Impact: ${market_impact:,.2f}")
print(f"Total (Slippage): ${spread_cost + market_impact:,.2f}")

print(f"\nStep 6: Performance Ranking (vs Benchmarks)")
print(f"-" * 50)

rank_df = pd.DataFrame({
    'Benchmark': ['Arrival Price', 'VWAP', 'TWAP'],
    'Cost (bps)': [ap_cost_bps, vwap_cost_bps, twap_cost_bps],
    'Rank': [2, 1, 3]  # Rank by cost (1=best, 3=worst)
})
rank_df = rank_df.sort_values('Cost (bps)')
print(rank_df.to_string(index=False))

# VISUALIZATION
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Intraday price and benchmarks
ax = axes[0, 0]
ax.plot(minutes, intraday_prices, label='Intraday Price', linewidth=2, color='black')
ax.plot(minutes, vwap, label='VWAP', linewidth=2, color='blue', alpha=0.7)
ax.plot(minutes, twap, label='TWAP', linewidth=2, color='green', alpha=0.7)
ax.axhline(y=arrival_price, color='red', linestyle='--', label='Arrival Price')
ax.axhline(y=execution_avg_price, color='purple', linestyle='--', label='Execution Avg')
ax.set_xlabel('Minute')
ax.set_ylabel('Price ($)')
ax.set_title('Intraday Price vs Benchmarks')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Volume and execution
ax = axes[0, 1]
ax.bar(minutes, volume, alpha=0.5, label='Market Volume', edgecolor='black', width=1)
ax.bar(minutes, exec_shares_per_minute, alpha=0.7, label='Execution Volume', edgecolor='red', width=1)
ax.set_xlabel('Minute')
ax.set_ylabel('Shares')
ax.set_title('Execution Profile vs Market Volume')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Cost vs benchmarks (bar chart)
ax = axes[0, 2]
benchmarks = ['Arrival\nPrice', 'VWAP', 'TWAP']
costs = [ap_cost_bps, vwap_cost_bps, twap_cost_bps]
colors = ['green' if c < 0 else 'red' for c in costs]
bars = ax.bar(benchmarks, costs, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Cost (basis points)')
ax.set_title('Execution Cost vs Benchmarks')
ax.grid(alpha=0.3, axis='y')
for bar, cost in zip(bars, costs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{cost:.1f}', ha='center', va='bottom' if cost > 0 else 'top')

# Plot 4: Implementation shortfall breakdown
ax = axes[1, 0]
components = ['Timing Cost\n(Market)', 'Execution Cost\n(Algo)', 'Total Shortfall']
values = [timing_cost / 1000, execution_cost / 1000, total_shortfall / 1000]  # In thousands
colors = ['red' if v < 0 else 'green' for v in values]
bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Cost ($thousands)')
ax.set_title('Implementation Shortfall Decomposition')
ax.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${val:.1f}k', ha='center', va='bottom' if val > 0 else 'top')

# Plot 5: Cumulative execution and price
ax = axes[1, 1]
cum_shares = np.cumsum(exec_shares_per_minute)
ax2 = ax.twinx()
ax.plot(minutes, cum_shares / 1000, label='Cumulative Execution', linewidth=2, color='blue')
ax2.plot(minutes, intraday_prices, label='Price', linewidth=2, color='red', alpha=0.7)
ax.set_xlabel('Minute')
ax.set_ylabel('Cumulative Shares (thousands)', color='blue')
ax2.set_ylabel('Price ($)', color='red')
ax.set_title('Execution Progress vs Price Move')
ax.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')
ax.grid(alpha=0.3)

# Plot 6: Slippage decomposition (pie chart)
ax = axes[1, 2]
if spread_cost > 0 and market_impact > 0:
    slippage_components = [spread_cost, market_impact]
    labels = [f'Spread\n${spread_cost:,.0f}', f'Impact\n${market_impact:,.0f}']
    colors = ['skyblue', 'orange']
    wedges, texts, autotexts = ax.pie(slippage_components, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Slippage Decomposition')

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("INSIGHTS")
print(f"="*100)
print(f"- Best benchmark: VWAP ({vwap_cost_bps:.2f} bps) - reflects volume distribution")
print(f"- Execution timeline: {end_minute - start_minute} minutes, smooth ramp (minimize impact)")
print(f"- Implementation shortfall driven by: {'Timing' if abs(timing_cost) > abs(execution_cost) else 'Execution'}")
print(f"- Estimated market impact cost: ~{market_impact_coef*100:.3f}% × sqrt(volume_fraction)")
print(f"- Spread friction: ${spread_cost:,.0f} (transaction cost floor)")
```

## 6. Challenge Round
- Implement TCA analysis comparing VWAP vs TWAP on 3 different stocks (large/mid/small cap)
- Build conditional TCA: Segment by market volatility regime (normal vs. stress days); adjust benchmarks
- Evaluate implicit costs: Measure the cost of market impact using realized volatility signature plots
- Design optimal benchmark: Given stock characteristics (liquidity, size, sector), select best benchmark
- Create peer comparison: Rank 10 traders by average execution cost; identify top/bottom performers

## 7. Key References
- [Perold (1988), "The Implementation Shortfall," Journal of Portfolio Management](https://www.jstor.org/stable/4479223) — Foundational framework for decomposing timing vs execution costs
- [Kissell & Glantz (2003), "Optimal Trading Strategies," AMACOM](https://www.amazon.com/Optimal-Trading-Strategies-Quantitative-Approaches/dp/0814407242) — Comprehensive TCA methodology and benchmarks
- [Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions," Mathematical Finance](https://www.math.nyu.edu/faculty/chriss/optliq_f.pdf) — Optimal execution optimization balancing timing/impact
- [Bloomberg AFPT Documentation](https://www.bloomberg.com/) — Industry-standard TCA platform reference

---
**Status:** Core TCA methodology (ubiquitous in trading operations) | **Complements:** Market Impact, Execution Algorithms, Slippage, Liquidity Analysis
