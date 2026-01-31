# VWAP (Volume Weighted Average Price)

## 1. Concept Skeleton
**Definition:** Algorithmic execution strategy targeting volume-weighted average price over a specified time window; minimizes deviation from VWAP benchmark  
**Purpose:** Execute large orders while minimizing market impact; reduce information leakage by matching market volume; provide fair execution benchmark  
**Prerequisites:** Order execution theory, market impact models, intraday volume patterns, optimization algorithms, transaction cost analysis

## 2. Comparative Framing
| Aspect | VWAP | TWAP | POV | IS | Arrival Price |
|--------|------|------|-----|-----|---------------|
| **Benchmark** | Volume-weighted avg | Time-weighted avg | Active participation | Start price | Opening price |
| **Volume Adaptation** | Yes (reactive) | No (fixed slice) | Yes (dynamic %) | Yes (adaptive) | No |
| **Slippage Risk** | Medium | High | Low | Medium | High |
| **Information Leakage** | Low (natural volume) | High (uniform) | Medium | Low | Low |
| **Use Case** | Large orders, liquid | Baseline, simple | Active, fast | Risk-averse | Opportunistic |
| **Execution Cost** | Lower if patient | Higher if fast | Market-dependent | Lowest | Depends on timing |

## 3. Examples + Counterexamples

**Simple Example:**  
Trade 100k shares of S&P 500 stock. Intraday volume profile: 20% (10am), 25% (11am), 30% (12pm), 25% (1pm). VWAP algorithm allocates: 20k→10am, 25k→11am, 30k→12pm, 25k→1pm. Execution mimics market volume, reducing price impact.

**Success Case:**  
Large hedge fund liquidating 500k shares quietly over day. VWAP tracks actual market participation; final executed price ≈ benchmark VWAP ± 2-3 bps. Successful execution.

**Poor Market Conditions:**  
Dry powder day (low volume); VWAP algo slices small, but prices move up 1% anyway. Small slices can't absorb, trader waits → higher fill prices. Benchmark VWAP becomes worse, algo underperforms naive approach.

**Intraday Vol Surprise:**  
Volume normally peaks at 11am. But earnings announced → volume crashes. VWAP algo allocates 25k shares to now-illiquid 11am window; market impact explodes. Benchmark miss.

**Sector Rotation:**  
Transitioning from Tech to Financials. Buy 50k Financial shares. VWAP follows Financial volume, not tech-correlated. Good isolation.

## 4. Layer Breakdown
```
VWAP Execution Framework:

├─ VWAP Calculation:
│  ├─ Definition:
│  │   VWAP = Σ(Price_i × Volume_i) / Σ(Volume_i)
│  │   where i = time buckets (minute, 5-min, hourly)
│  ├─ Cumulative VWAP:
│  │   VWAP(t) = Σ₀ᵗ (P_i × V_i) / Σ₀ᵗ V_i
│  │   Updates continuously as new trades occur
│  ├─ Closing VWAP:
│  │   Final daily benchmark; settlement reference
│  └─ Intraday buckets:
│      1-minute: 390 bars/day (liquid)
│      5-minute: 78 bars/day (standard)
│      Custom: Market-dependent granularity
├─ Volume Prediction:
│  ├─ Historical Pattern:
│  │   V̂(t) = ∑ wᵢ V(t, i) historical weight
│  │   Typically: Low open → ramp up → lunch dip → close ramp
│  ├─ Real-time Adjustment:
│  │   Observed vol vs predicted vol → scale allocation
│  │   Higher vol window → increase allocation
│  │   Lower vol window → reserve shares for later
│  ├─ Seasonality:
│  │   Day-of-week effects (Monday, Friday)
│  │   Month-end, quarter-end (rebalancing rush)
│  │   Ex-dividend dates (vol changes)
│  └─ Event Impact:
│      Economic data releases (high vol spikes)
│      Earnings announcements (structural breaks)
│      Fed announcements (regime changes)
├─ Allocation Strategy:
│  ├─ Proportional Allocation:
│  │   Allocate shares proportional to intraday volume
│  │   Simple: If vol at 11am is 25% of daily, buy 25% of order then
│  ├─ Adaptive Allocation (Real-time):
│  │   Compare actual vol to prediction
│  │   If actual > predicted → increase allocation aggressively
│  │   If actual < predicted → reduce, wait for catch-up
│  ├─ Price-Adaptive Allocation:
│  │   If price moving favorably → increase allocation (capture move)
│  │   If price moving adversely → reduce, wait
│  │   Tension: Balance vol participation vs price tracking
│  ├─ Participation Rate (POV Variant):
│  │   Participate at 20-25% of market volume per slice
│  │   Limits market impact per window
│  │   VWAP + POV = hybrid approach
│  └─ Urgency Levels:
│      Patient (spread over day): Volume-follow
│      Urgent (few hours): Accelerate, accept impact
│      Emergency (ASAP): Market order approximation
├─ Execution Mechanics:
│  ├─ Order Submission:
│  │   Calculate target quantity for bucket t
│  │   Submit slice as limit/market order
│  │   Monitor fill vs target
│  ├─ Market Orders vs Limits:
│  │   Market: Guaranteed execution, pay spread
│  │   Limit: Avoid spread, risk non-execution
│  │   Hybrid: Limit on momentum, market on weakness
│  ├─ Venue Selection:
│  │   Primary exchange (tightest spread)
│  │   Alternative venues (lit pools, ATS)
│  │   Dark pools (minimize impact in liquid periods)
│  └─ Timing:
│      Trade early in volume spike (front-run own benchmark)
│      Trade late in volume spike (ride volume up)
│      Trade middle (neutral participation)
├─ Performance Measurement:
│  ├─ VWAP Slippage:
│  │   Execution_Price_Avg - VWAP (fixed benchmark)
│  │   Positive slippage = better execution
│  │   Negative slippage = underperformance vs benchmark
│  ├─ Realized VWAP:
│  │   Actual weighted average of filled orders
│  │   Often tracks benchmark if algo working well
│  ├─ Information Ratio (VWAP):
│  │   (Avg Slippage) / (Std Dev Slippage)
│  │   Consistency measure
│  ├─ Breakeven Threshold:
│  │   VWAP + Market Spread ≈ client breakeven
│  │   If exec price ≤ VWAP + spread, worth it
│  └─ Attribution:
│      Market impact component
│      Timing (beat/miss volume forecasting)
│      Spread cost component
│      Timing delay cost (opportunity)
├─ Risk Factors:
│  ├─ Volume Forecasting Error:
│  │   Actual volume << forecast → allocated too aggressively
│  │   Actual volume >> forecast → allocated too passively
│  │   Market structure changes (ETF rebalancing)
│  ├─ Price Momentum:
│  │   Strong trending market → volume congregates at prices
│  │   VWAP algo follows vol, can miss directional move
│  ├─ Volatility Spikes:
│  │   Sudden events → vol dries up or explodes
│  │   VWAP assumption of normal patterns breaks
│  ├─ Liquidity Crises:
│  │   Bid-ask widens, volume evaporates
│  │   VWAP algo paralyzed (nothing to execute against)
│  └─ Information Leakage:
│      Large VWAP orders visible in order flow
│      Informed traders can trade ahead
│      Crossing networks can detect patterns
├─ Variations:
│  ├─ MVWAP (Modified VWAP):
│  │   Volume-weighted with price bounds
│  │   Reject volume at outlier prices
│  │   Smooths impact of flash crashes
│  ├─ Adaptive VWAP:
│  │   Adjust participation based on market conditions
│  │   High vol periods → accelerate
│  │   Low vol periods → slow down
│  ├─ Arrival Price (Volume-Aware):
│  │   VWAP + price momentum term
│  │   Balance volume participation + price tracking
│  └─ FVWAP (Futures VWAP):
│      Applied to future contracts
│      Accounts for carry, open interest
└─ Practical Considerations:
   ├─ Data Quality:
   │   Accurate volume feeds (trades vs bids/asks)
   │   Exchange connectivity (latency)
   │   Corporate actions (splits, dividends)
   ├─ Regulatory:
   │   FINRA rules on algo disclosure
   │   Market manipulation risks (layering, spoofing)
   │   Audit trail requirements
   ├─ Market Conditions:
   │   Applies best to high-liquidity assets
   │   Poor in low-volume stocks (VWAP undefined)
   │   Index constituent changes affect volume patterns
   └─ Benchmarking:
       Post-trade TCA analysis
       Compare to TWAP, POV alternatives
       Adjust parameters based on results
```

**Interaction:** Volume prediction → allocation sizing → execution sequencing → benchmark tracking → performance attribution.

## 5. Mini-Project
Implement VWAP execution simulation:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate intraday volume profile
np.random.seed(42)

# Trading day: 9:30am - 4:00pm (390 minutes)
minutes = np.arange(1, 391)
hours = 9.5 + minutes / 60

# Typical intraday volume pattern (U-shaped)
base_vol = 1000  # base volume per minute
morning_ramp = 1 + 0.5 * np.sin(np.pi * minutes / 100)  # 9:30-11:00
lunch_dip = 0.6  # 11:00-1:00 (around minute 90-210)
afternoon_ramp = 1 + 0.3 * np.sin(np.pi * (minutes - 210) / 180)  # 1:00-4:00

vol_multiplier = np.ones(390)
vol_multiplier[:90] = morning_ramp[:90]
vol_multiplier[90:210] = lunch_dip
vol_multiplier[210:] = afternoon_ramp[210:]

# Simulate intraday prices (random walk)
price_base = 100
price_changes = np.random.normal(0, 0.05, 390)  # ~5 bps volatility
prices = price_base + np.cumsum(price_changes)

# Daily volume
daily_volume = np.random.poisson(base_vol, 390) * vol_multiplier
daily_volume = daily_volume.astype(int)

total_daily_vol = daily_volume.sum()
print(f"Total Daily Volume: {total_daily_vol:,} shares")

# Create DataFrame
intraday_data = pd.DataFrame({
    'minute': minutes,
    'hour': hours,
    'price': prices,
    'volume': daily_volume,
    'cum_volume': daily_volume.cumsum(),
    'cum_dollar_volume': (prices * daily_volume).cumsum(),
})

# Calculate VWAP
intraday_data['vwap'] = intraday_data['cum_dollar_volume'] / intraday_data['cum_volume']

print("\nIntraday VWAP Profile (Sample):")
print(intraday_data[::60][['hour', 'price', 'volume', 'vwap']].to_string(index=False))

# VWAP Execution Strategy
target_order = 50000  # 50k shares to execute
volume_pct = intraday_data['volume'] / total_daily_vol

# Allocation: proportional to volume
target_qty = (volume_pct * target_order).values
target_qty = target_qty.astype(int)

# Simulate fills: assume we get filled at market price
filled_qty = target_qty.copy()
filled_qty[-1] += (target_order - target_qty.sum())  # Adjust last bar for rounding

filled_price = prices.copy()
execution_cost = (filled_qty * filled_price).sum()
avg_execution_price = execution_cost / target_order
final_vwap = intraday_data['vwap'].iloc[-1]

print(f"\n{'='*60}")
print("VWAP EXECUTION ANALYSIS")
print(f"{'='*60}")
print(f"Order Size: {target_order:,} shares")
print(f"Final VWAP: ${final_vwap:.4f}")
print(f"Avg Execution Price: ${avg_execution_price:.4f}")
print(f"Slippage: ${avg_execution_price - final_vwap:.4f}")
print(f"Slippage (bps): {(avg_execution_price - final_vwap) / final_vwap * 10000:.2f} bps")
print(f"Total Cost: ${execution_cost:,.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Price and VWAP
ax = axes[0, 0]
ax.plot(hours, prices, 'b-', linewidth=1.5, label='Price')
ax.plot(hours, intraday_data['vwap'], 'r--', linewidth=2, label='VWAP')
ax.fill_between(hours, prices, intraday_data['vwap'], alpha=0.2)
ax.set_title('Price vs VWAP')
ax.set_xlabel('Time')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Intraday Volume Profile
ax = axes[0, 1]
ax.bar(hours, daily_volume, width=0.015, alpha=0.7, color='green')
ax.set_title('Intraday Volume Profile')
ax.set_xlabel('Time')
ax.set_ylabel('Volume (shares)')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Allocation vs Actual Fill
ax = axes[1, 0]
ax.plot(hours, target_qty, 'b-', linewidth=2, marker='o', markersize=3, 
       label='Target Allocation')
ax.set_title('VWAP Allocation Schedule')
ax.set_xlabel('Time')
ax.set_ylabel('Shares per Minute')
ax.grid(alpha=0.3)
ax.legend()

# Plot 4: Cumulative execution vs volume
ax = axes[1, 1]
ax.plot(hours, intraday_data['cum_volume'].values, 'g-', linewidth=2, 
       label='Cumulative Market Volume')
ax.plot(hours, np.cumsum(filled_qty), 'b--', linewidth=2, 
       label='Cumulative Filled Qty')
ax.set_title('Execution Progress vs Market Volume')
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Volume')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Performance Summary
print(f"\nExecution Summary:")
print(f"Filled Qty: {filled_qty.sum():,} shares")
print(f"Avg Fill Price: ${avg_execution_price:.4f}")
print(f"VWAP Benchmark: ${final_vwap:.4f}")
print(f"Outperformance: ${(final_vwap - avg_execution_price) * target_order:,.2f}")
```

## 6. Challenge Round
- How would you predict intraday volume for a news-heavy stock?
- Why is VWAP suboptimal in trending markets?
- Design hybrid VWAP + POV strategy for volatile periods
- Explain volume manipulation and its VWAP impact
- Compute optimal participation rate for given market impact model

## 7. Key References
- [Kissell & Perlin, "Algorithmic Trading Methods" (2016)](https://www.wiley.com/en-us/Algorithmic+Trading+Methods-p-9780470643112) — VWAP detailed analysis
- [Almgren & Chriss, "Optimal Execution of Portfolio Transactions" (2001)](https://www.jstor.org/stable/2645747) — Theoretical foundations
- [Bouchaud et al, "Market Microstructure" (2009)](https://www.cambridge.org/core/books/) — Volume patterns

---
**Status:** Industry-standard execution benchmark | **Complements:** TWAP, POV, Market Impact, TCA
