# TWAP (Time Weighted Average Price)

## 1. Concept Skeleton
**Definition:** Algorithm that slices order uniformly across execution window; attempts to fill at average of prices over period; benchmark is time-weighted average price  
**Purpose:** Provide simple baseline for execution; minimize execution complexity; predictable pacing; benchmark against time-averaged market price  
**Prerequisites:** Execution algorithms, benchmark pricing, order slicing, market dynamics

## 2. Comparative Framing
| Aspect | TWAP | VWAP | POV | IS | Optimal |
|--------|------|------|-----|-----|---------|
| **Algorithm Complexity** | Trivial | Medium | Medium | High | Very High |
| **Predictability** | Highly predictable | Moderately predictable | Unpredictable | Adaptive | Dynamic |
| **Market Impact** | Medium-High | Low-Medium | Very Low | Low-Medium | Minimal |
| **Implementation Cost** | Minimal | Low | Low | Medium | High |
| **Information Leakage** | High (uniform pace) | Low (volume following) | Very Low | Low | Variable |
| **Best Case** | Calm flat market | Volume clustering | Liquid, normal | Uncertain vol | All cases |
| **Worst Case** | Trending market | Dry powder day | Volume drought | Complex params | Model error |

## 3. Examples + Counterexamples

**Simple Case:**  
Execute 100k shares over 1 hour. Slice into 12 x 8,333 shares per 5-minute bucket. Market prices: $100, $100.02, $99.98, $100.01, ... → TWAP ≈ $100.00. Fill whole order at ≈ $100.00.

**Predictable Front-Running:**  
Informed traders see uniform 5-min allocations → deduce total order size. They can front-run each slice knowing next 8,333 shares coming. TWAP's predictability becomes weakness.

**Favorable Trending Market:**  
Buy 50k shares. Market rallies from $100 to $102. TWAP forced to buy equal amounts each period at rising prices → avg fills $101. VWAP algorithm, if volume peaks late in rally, buys more at $102 → worse. Depends on vol timing.

**Unfavorable Trending Market:**  
Sell 50k shares. Market selling off: $100 → $98. TWAP sells equal 5k per period → avg $99. But volume might spike late (panic selling) → VWAP could be forced to sell at $98. Both suffer in trending market.

**Liquidity Heterogeneity:**  
Morning: Low liquidity. Afternoon: High liquidity. TWAP: Buy 1k each hour, struggles morning (high impact), overly passive afternoon. VWAP: Buys tiny amount morning (low vol), huge afternoon (high vol). Better if vol clustered.

**Simple Baseline:**  
Institutional fund, no time pressure, wants simple benchmark. TWAP: "Slice 1/6 per hour for 6-hour window." Implemented in Excel. Zero complexity, works adequately.

## 4. Layer Breakdown
```
TWAP Execution Framework:

├─ Definition:
│  ├─ Mathematical:
│  │   TWAP = (1/T) ∫₀ᵀ P(t) dt ≈ (1/n) Σᵢ₌₀ⁿ P(tᵢ)
│  │   Integral of price over time, normalized by period
│  ├─ Execution Mechanics:
│  │   Divide target order X into n equal slices
│  │   Each slice: x̂ = X / n
│  │   Schedule: One slice every Δt = T / n
│  │   Total time: T (fixed)
│  └─ Benchmark:
│      Daily TWAP: Full-day average price
│      Intraday TWAP: Partial-day average
│      Custom TWAP: User-defined interval
├─ Slicing Mechanisms:
│  ├─ Time-Based Slicing:
│  │   Divide time interval into equal buckets
│  │   Hourly buckets: 1 per hour
│  │   5-minute buckets: 78 per day (finer control)
│  │   Customizable to trading hours (RTH vs ETH)
│  ├─ Equal Quantity:
│  │   Each bucket: X / n shares
│  │   Deterministic, easy to compute
│  ├─ Volume-Following (TWAP Variant):
│  │   Add randomness within bucket
│  │   Arrive at time ∈ [t_i, t_i + Δt]
│  │   Perturb quantity: x̂ + ε noise
│  │   Reduces predictability
│  ├─ Adaptive Timing Within Bucket:
│  │   Front-load if favorable momentum
│  │   Back-load if adverse momentum
│  │   Keeps average allocation
│  └─ Precision:
│      Rounding: X / n might not be integer
│      Accumulate on last slice: x̂_n = X - Σᵢ₍ₙ₋₁₎ x̂ᵢ
│      Or: proportional adjustment across all
├─ Order Submission:
│  ├─ Limit Orders:
│  │   Place at mid-price or inside
│  │   Minimize spread cost
│  │   Risk: Non-execution if sliced too passively
│  ├─ Market Orders:
│  │   Guarantee execution (pay bid-ask spread)
│  │   No slippage, just spread cost
│  │   Common approach for simplicity
│  ├─ Hybrid Strategy:
│  │   Limit order first 90% of slice time
│  │   Market order last 10% (ensure fill)
│  │   Balances cost vs certainty
│  └─ Timing Precision:
│      Submit at beginning of bucket
│      Or: Randomly within bucket (reduce front-running)
│      Or: Follow market flow (react to volume/prices)
├─ Market Conditions Impact:
│  ├─ Flat Market:
│  │   Price ≈ constant over window
│  │   TWAP ≈ VWAP ≈ Arrival Price
│  │   Algorithm choice doesn't matter much
│  ├─ Trending Market (Adverse):
│  │   Price moves against execution continuously
│  │   TWAP forced to average up (buy) or average down (sell)
│  │   Results in worse execution than early market order
│  ├─ Trending Market (Favorable):
│  │   Price moves with execution
│  │   TWAP captures benefit of time-averaging
│  │   Better than catching peak/trough
│  ├─ Gapped Market (Open/Close):
│  │   Overnight news → price jumps
│  │   TWAP doesn't adjust → executes from previous close
│  │   Information risk
│  ├─ Low Liquidity:
│  │   Each slice liquidity-constrained
│  │   Fills may be sparse
│  │   Execution time stretches beyond planned window
│  └─ Volume Spikes:
│      Corporate action, index rebalancing
│      TWAP continues uniform → relatively passive
│      Misses opportunity during high-liquidity windows
├─ Benchmark Performance:
│  ├─ Price Slippage (Execution):
│  │   Exec_Avg_Price - TWAP = slippage
│  │   Positive = better than benchmark
│  │   Negative = underperformed
│  ├─ Information Ratio:
│  │   Mean(Slippage) / Std(Slippage)
│  │   Consistency measure
│  ├─ Worst/Best Days:
│  │   Track 10th/90th percentile performance
│  │   Symmetric or skewed?
│  ├─ Venue-Specific TWAP:
│  │   If split across exchanges, calculate venue-specific
│  │   Compare venue execution to venue TWAP
│  └─ Vs Alternatives:
│      TWAP vs VWAP (actual market weighted)
│      TWAP vs POV (reactive)
│      TWAP vs Arrival Price (pure timing)
├─ Risks & Challenges:
│  ├─ Predictable Signal:
│  │   Uniform allocation visible to market
│  │   Traders can deduce order size
│  │   Front-running incentivized
│  │   Signal: Large order every 5 minutes exactly
│  ├─ Adversarial Markets:
│  │   HFT algorithms detect TWAP pattern
│  │   Trade ahead of each slice
│  │   Extract profit from known future demand
│  ├─ Price Trending:
│  │   TWAP assumption: Price stationary
│  │   Reality: Trends, mean reversion, jumps
│  │   Breaks in trending markets
│  ├─ Liquidity Dry-Up:
│  │   Market volume depletes
│  │   TWAP slices can't fill at expected cost
│  │   Forced to accept worse prices or extend window
│  ├─ After-Hours Issues:
│  │   TWAP over 8am-midnight window with 9:30-4:00 market
│  │   Off-hours slices: Non-execution or extreme spreads
│  └─ Regulatory:
│      Potential market manipulation if transparent pattern
│      Must document algo choice
│      Fair execution requirements
├─ Variants & Enhancements:
│  ├─ TWAP + Random Arrival:
│  │   Same total allocation
│  │   Arrive randomly within bucket
│  │   Reduces pattern visibility
│  │   Execution time uncertain (+/- 5 min)
│  ├─ TWAP + Volume Scaling:
│  │   Scale quantities up/down by volume ratio
│  │   More passive in low vol
│  │   More aggressive in high vol
│  │   Closer to VWAP
│  ├─ TWAP + Momentum Overlay:
│  │   Increase allocation if price favorable
│  │   Decrease if adverse
│  │   Keep time-weighted property
│  ├─ TWAP + Volatility Regime:
│  │   Calm regime: Stick to plan
│  │   Volatile regime: Accelerate (reduce time at risk)
│  ├─ Multi-Slice TWAP:
│  │   Slice across multiple trading sessions
│  │   If multi-day order: distribute across days
│  │   Reduces daily market impact
│  └─ TWAP with Limit Trigger:
│      Accelerate if favorable limit prices hit
│      Decelerate if unfavorable
│      Still maintains average
├─ Comparison to Benchmarks:
│  ├─ vs VWAP:
│  │   TWAP simpler, more predictable
│  │   VWAP adapts to volume, better in liquid periods
│  │   TWAP better if vol uniformly distributed
│  │   VWAP better if vol clustered
│  ├─ vs Arrival Price:
│  │   TWAP time-averaged (reduces one-time impact)
│  │   Arrival Price point-in-time (instant impact)
│  │   TWAP better for large orders
│  │   Arrival Price better for urgent orders
│  ├─ vs Optimal Execution:
│  │   TWAP simple, doesn't optimize
│  │   Optimal uses market impact theory
│  │   TWAP easier to implement, fewer params
│      Optimal better cost if vol/impact calibrated correctly
│  └─ vs POV:
│      TWAP fixed time pace
│      POV fixed volume participation
│      TWAP predictable
│      POV adaptive
└─ Practical Implementation:
   ├─ Technology:
   │   Simple Excel model (non-professional)
   │   Python/C++ algo framework
   │   Broker algo (3rd party provider)
   ├─ Documentation:
   │   Record planned vs actual execution
   │   Benchmark TWAP from market data
   │   Calculate slippage attribution
   ├─ Client Communication:
   │   Set expectations: Simple, predictable
   │   Performance tied to market prices
   │   No guarantees vs VWAP/POV
   └─ Testing:
       Backtest on historical data
       Walk-forward validation
       Compare alternative algo performance
```

**Interaction:** Divide time window → uniform allocation per bucket → submit orders → average to TWAP benchmark.

## 5. Mini-Project
Simple TWAP vs VWAP comparison:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate intraday market
minutes = np.arange(1, 391)
hours = 9.5 + minutes / 60

# Generate prices with trend
base_price = 100
trend = np.linspace(0, 2, 390)  # Slight uptrend
prices = base_price + trend + np.random.normal(0, 0.3, 390)

# Generate volume (U-shaped)
base_vol = 1000
vol_morning = base_vol * 1.2
vol_lunch = base_vol * 0.6
vol_afternoon = base_vol * 1.3

volumes = np.concatenate([
    np.random.poisson(vol_morning, 90),
    np.random.poisson(vol_lunch, 120),
    np.random.poisson(vol_afternoon, 180)
])

# Create market data
market_data = pd.DataFrame({
    'minute': minutes,
    'hour': hours,
    'price': prices,
    'volume': volumes,
})

# Calculate benchmarks
market_data['cum_volume'] = market_data['volume'].cumsum()
market_data['dollar_volume'] = market_data['price'] * market_data['volume']
market_data['cum_dollar_volume'] = market_data['dollar_volume'].cumsum()
market_data['vwap'] = market_data['cum_dollar_volume'] / market_data['cum_volume']
market_data['twap'] = market_data['price'].expanding().mean()

# Execute order using TWAP and VWAP
target_order = 50000

# TWAP: Equal allocation per minute
twap_qty_per_min = np.full(390, target_order / 390)
twap_filled = (twap_qty_per_min * market_data['price'].values).sum()
twap_avg_price = twap_filled / target_order

# VWAP: Proportional to volume
vwap_qty_per_min = market_data['volume'].values * (target_order / market_data['volume'].sum())
vwap_filled = (vwap_qty_per_min * market_data['price'].values).sum()
vwap_avg_price = vwap_filled / target_order

final_vwap = market_data['vwap'].iloc[-1]
final_twap = market_data['twap'].iloc[-1]

print("="*60)
print("TWAP vs VWAP EXECUTION COMPARISON")
print("="*60)
print(f"\nBenchmarks:")
print(f"  Final VWAP:  ${final_vwap:.4f}")
print(f"  Final TWAP:  ${final_twap:.4f}")
print(f"\nExecution Results (50k shares):")
print(f"  TWAP Execution Avg: ${twap_avg_price:.4f}")
print(f"    vs TWAP benchmark: ${twap_avg_price - final_twap:.4f} ({(twap_avg_price - final_twap)/final_twap*10000:.1f} bps)")
print(f"\n  VWAP Execution Avg: ${vwap_avg_price:.4f}")
print(f"    vs VWAP benchmark: ${vwap_avg_price - final_vwap:.4f} ({(vwap_avg_price - final_vwap)/final_vwap*10000:.1f} bps)")
print(f"\nComparison:")
print(f"  VWAP Execution better by: ${twap_avg_price - vwap_avg_price:.4f} "
      f"({(twap_avg_price - vwap_avg_price)/vwap_avg_price*10000:.1f} bps)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Price trajectory
ax = axes[0, 0]
ax.plot(hours, prices, 'b-', linewidth=1.5, label='Market Price')
ax.axhline(y=twap_avg_price, color='g', linestyle='--', linewidth=2, label=f'TWAP Exec (${twap_avg_price:.2f})')
ax.axhline(y=vwap_avg_price, color='r', linestyle='--', linewidth=2, label=f'VWAP Exec (${vwap_avg_price:.2f})')
ax.set_title('Price Movement & Execution Prices')
ax.set_xlabel('Time')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Volume profile
ax = axes[0, 1]
colors = ['red' if v < np.mean(volumes) else 'green' for v in volumes]
ax.bar(hours, volumes, width=0.015, alpha=0.6, color=colors)
ax.axhline(y=np.mean(volumes), color='black', linestyle='--', linewidth=1, label='Avg Volume')
ax.set_title('Intraday Volume Profile')
ax.set_xlabel('Time')
ax.set_ylabel('Volume (shares)')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Cumulative execution
ax = axes[1, 0]
twap_cumsum = np.cumsum(twap_qty_per_min)
vwap_cumsum = np.cumsum(vwap_qty_per_min)
ax.plot(hours, twap_cumsum, 'g-', linewidth=2, label='TWAP Cumulative')
ax.plot(hours, vwap_cumsum, 'r-', linewidth=2, label='VWAP Cumulative')
ax.axhline(y=target_order, color='black', linestyle=':', linewidth=1, label='Target Order')
ax.set_title('Cumulative Execution Progress')
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Shares')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Per-minute slicing comparison
ax = axes[1, 1]
x_pos = np.arange(0, 390, 30)
ax.scatter(hours[x_pos], twap_qty_per_min[x_pos], s=50, alpha=0.6, label='TWAP Qty/Min', color='green')
ax.scatter(hours[x_pos], vwap_qty_per_min[x_pos], s=50, alpha=0.6, label='VWAP Qty/Min', color='red')
ax.set_title('Allocation per Minute (Sample)')
ax.set_xlabel('Time')
ax.set_ylabel('Shares per Minute')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
- Why is TWAP vulnerable to front-running? Design countermeasure
- How would you optimize TWAP for trending market?
- Calculate TWAP for order spanning multiple trading sessions
- Compare expected execution cost: TWAP vs optimal execution model
- Build TWAP + adaptive volume overlay variant

## 7. Key References
- [Kissell & Perlin, "The Art of Execution" (2012)](https://www.wiley.com/en-us/The+Art+of+Execution-p-9781118219584) — TWAP benchmarking
- [Almgren & Chriss, "Optimal Execution of Portfolio Transactions" (2001)](https://www.jstor.org/stable/2645747) — Theoretical optimality
- [SEC Market Microstructure Rules (Reg SHO, Rule 10b-5)](https://www.sec.gov/marketstructure)

---
**Status:** Simplest execution baseline | **Complements:** VWAP, POV, Optimal Execution, Arrival Price
