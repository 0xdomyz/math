# Transaction Cost Analytics (TCA)

## Concept Skeleton

Transaction Cost Analytics (TCA) measures execution quality by comparing actual trade prices to benchmarks (arrival price, VWAP, closing price), decomposing costs into timing (delay), market impact (price movement from order), spread (bid-ask crossing), and opportunity cost (unfilled orders), enabling traders to optimize execution strategies and hold brokers/algorithms accountable. TCA operates across pre-trade (cost forecasting), intra-trade (real-time monitoring), and post-trade (performance attribution) phases.

**Core Components:**
- **Benchmarks**: Arrival price (decision price), VWAP (volume-weighted average price), TWAP (time-weighted), closing price
- **Cost components**: Spread (bid-ask), market impact (permanent + temporary), timing cost (delay between decision and execution)
- **Slippage**: Difference between expected price and actual fill price (positive slippage = worse than expected)
- **Implementation shortfall**: Total cost from decision to final fill, including opportunity cost of unfilled shares
- **Venue analysis**: Compare execution quality across exchanges, dark pools, brokers

**Why it matters:** Execution costs erode alpha; 10 bps slippage on 100% annual turnover consumes 10% of returns; TCA quantifies hidden costs, identifies inefficient execution, and justifies trading infrastructure investments (colocation, smart order routing).

---

## Comparative Framing

| Benchmark | **Arrival Price** | **VWAP** | **Closing Price** |
|-----------|-------------------|----------|-------------------|
| **Definition** | Price when decision made (order sent) | Volume-weighted average during execution window | Official closing price (4pm ET) |
| **Use case** | Measures total implementation shortfall | Typical institutional benchmark | Passive end-of-day trades |
| **Captures delay cost** | Yes (includes time between decision and first fill) | No (starts when trading begins) | Yes (if decided earlier) |
| **Benchmark difficulty** | Realistic for urgent trades | Achievable with VWAP algo | Hard for large orders (moves market by close) |
| **Gaming risk** | Low (can't control decision time retroactively) | Moderate (algo can "lean on" VWAP) | High (traders can manipulate closing auction) |

**Key insight:** Arrival price is gold standard for assessing total cost (decision to completion); VWAP is practical for comparing algo performance; closing price useful for passive strategies but manipulable.

---

## Examples & Counterexamples

### Examples of Transaction Cost Analytics

1. **Implementation Shortfall Calculation**  
   - Decision to buy 100,000 shares at 10:00 AM, arrival price $50.00  
   - VWAP algo executes 90,000 shares at avg price $50.15, 10,000 shares unfilled  
   - Stock closes at $50.25  
   - **Components:**  
     - Spread + impact: 90,000 × ($50.15 - $50.00) = $13,500 (15 bps)  
     - Opportunity cost: 10,000 × ($50.25 - $50.00) = $2,500 (25 bps on unfilled)  
     - Total shortfall: $16,000 (16 bps on full order)  

2. **Pre-Trade Cost Estimation**  
   - Order: Buy 50,000 shares, stock ADV (average daily volume) = 1M shares (5% participation)  
   - Estimated market impact: \(\text{Impact} = \alpha \times \sigma \times \sqrt{\frac{Q}{V}}\) where \(\sigma\) = 2% daily vol, \(\alpha = 0.5\)  
   - Impact = 0.5 × 0.02 × √(50k/1M) = 0.5 × 0.02 × 0.224 = 0.22% = 22 bps  
   - Pre-trade forecast: Expect 22 bps cost; if actual is 40 bps, execution was poor  

3. **Broker Comparison via TCA**  
   - Broker A: Average slippage 8 bps, fill rate 95%  
   - Broker B: Average slippage 12 bps, fill rate 98%  
   - Trade-off: Broker A cheaper per filled share but misses 5%; Broker B costlier but higher certainty  
   - Decision: Use Broker A for patient trades, Broker B for urgent orders  

4. **Dark Pool TCA**  
   - Order routed to 5 dark pools; 30% fills in Pool X at midpoint (0 bps spread cost)  
   - Remaining 70% goes to lit exchanges, pays half-spread (5 bps)  
   - Blended cost: 0.30 × 0 + 0.70 × 5 = 3.5 bps  
   - Compare to all-lit execution (5 bps): Dark pool saves 1.5 bps

### Non-Examples (or Misuses)

- **Benchmarking to prior day's close**: Ignores intraday price movement; artificially inflates/deflates costs.
- **Ignoring opportunity cost of unfilled orders**: Only measuring filled shares understates true cost.
- **Comparing VWAP slippage across different volatility regimes**: 10 bps slippage in calm market vs. volatile market not equivalent.

---

## Layer Breakdown

**Layer 1: Cost Decomposition**  
Total transaction cost = Spread + Temporary Impact + Permanent Impact + Delay Cost + Opportunity Cost.  

**Spread:** Crossing bid-ask spread when taking liquidity (market order).  
\[
\text{Spread Cost} = \frac{\text{Ask} - \text{Bid}}{2}
\]  

**Market Impact:**  
- **Temporary:** Price moves against order during execution, mean-reverts afterward (liquidity provision).  
- **Permanent:** Information leakage; market infers order reflects non-public information, price adjusts.  

**Delay Cost:** Price drift between decision and first execution (timing risk).  

**Opportunity Cost:** Value lost on unfilled portion (if price moves unfavorably and order canceled).

**Layer 2: Benchmark Selection**  
**Arrival Price (Decision Price):**  
Midpoint when order decision made; includes all subsequent costs. Best for assessing total implementation.  

**VWAP (Volume-Weighted Average Price):**  
\[
\text{VWAP} = \frac{\sum (\text{Price}_i \times \text{Volume}_i)}{\sum \text{Volume}_i}
\]  
Typical for institutional orders; neutral benchmark (not information-revealing).  

**Implementation Shortfall:**  
\[
\text{IS} = \frac{\sum (\text{Fill Price}_i - \text{Arrival Price}) \times \text{Shares}_i + (\text{Closing Price} - \text{Arrival Price}) \times \text{Unfilled Shares}}{\text{Arrival Price} \times \text{Total Shares}}
\]

**Layer 3: Pre-Trade Cost Forecasting**  
Use market impact models:  
**Almgren-Chriss Model:**  
\[
\text{Cost} = \frac{\sigma}{2} \sqrt{\frac{Q}{V}} + \gamma \left( \frac{Q}{V} \right)^{\beta}
\]  
where \(\sigma\) = volatility, \(Q\) = order size, \(V\) = market volume, \(\gamma, \beta\) = calibrated parameters.  

**Layer 4: Post-Trade Attribution and Reporting**  
Generate TCA report:  
- **Execution vs. Benchmark:** Actual fill price vs. VWAP, arrival price  
- **Cost breakdown:** Spread, impact, delay, opportunity cost (in bps)  
- **Peer comparison:** Percentile rank vs. similar orders (size, stock, time)  
- **Venue analysis:** Fill rate, avg price by exchange/dark pool  
- **Algo performance:** Which execution algo (VWAP, POV, Implementation Shortfall) achieved best results

---

## Mini-Project: Implementation Shortfall TCA

**Goal:** Calculate implementation shortfall for a simulated order execution.

```python
import numpy as np
import pandas as pd

# Simulate order execution
np.random.seed(123)
order_size = 100_000  # shares to buy
arrival_price = 50.00  # decision price at 10:00 AM
intraday_prices = 50.00 + np.cumsum(np.random.randn(100) * 0.02)  # Random walk

# Simulate fills using VWAP algo (executes over time)
fill_times = np.linspace(0, 99, 20).astype(int)  # 20 fills evenly spaced
fill_shares = order_size / len(fill_times)  # Equal-sized fills
fill_prices = intraday_prices[fill_times]

# Assume 5% of order unfilled
filled_shares = 0.95 * order_size
unfilled_shares = order_size - filled_shares
closing_price = intraday_prices[-1]

# Calculate implementation shortfall components
# 1. Executed cost (filled shares)
executed_value = fill_shares * fill_prices.sum()
arrival_value = arrival_price * filled_shares
execution_cost = executed_value - arrival_value
execution_cost_bps = (execution_cost / arrival_value) * 10000

# 2. Opportunity cost (unfilled shares)
opportunity_cost = unfilled_shares * (closing_price - arrival_price)
opportunity_cost_bps = (opportunity_cost / (arrival_price * order_size)) * 10000

# 3. Total implementation shortfall
total_cost = execution_cost + opportunity_cost
total_cost_bps = (total_cost / (arrival_price * order_size)) * 10000

# Average fill price
avg_fill_price = (fill_shares * fill_prices.sum()) / filled_shares

# Slippage vs VWAP
intraday_vwap = intraday_prices[:len(fill_times)].mean()  # Simplified VWAP
vwap_slippage_bps = ((avg_fill_price - intraday_vwap) / arrival_price) * 10000

print("=" * 70)
print("TRANSACTION COST ANALYTICS (Implementation Shortfall)")
print("=" * 70)
print(f"Order Details:")
print(f"  Total Order Size:              {order_size:>12,} shares")
print(f"  Arrival Price (Decision):      ${arrival_price:>11.2f}")
print(f"  Closing Price:                 ${closing_price:>11.2f}")
print(f"  Intraday Price Move:           {(closing_price - arrival_price):>11.2f} ({(closing_price/arrival_price - 1)*100:+.2f}%)")
print()
print(f"Execution Summary:")
print(f"  Filled Shares:                 {filled_shares:>12,.0f} ({filled_shares/order_size:.0%})")
print(f"  Unfilled Shares:               {unfilled_shares:>12,.0f} ({unfilled_shares/order_size:.0%})")
print(f"  Average Fill Price:            ${avg_fill_price:>11.2f}")
print(f"  Number of Fills:               {len(fill_times):>12}")
print()
print(f"Cost Breakdown:")
print(f"  Execution Cost (Filled):       ${execution_cost:>11,.2f}  ({execution_cost_bps:>6.1f} bps)")
print(f"  Opportunity Cost (Unfilled):   ${opportunity_cost:>11,.2f}  ({opportunity_cost_bps:>6.1f} bps)")
print(f"  Total Implementation Shortfall:${total_cost:>11,.2f}  ({total_cost_bps:>6.1f} bps)")
print()
print(f"Benchmark Comparison:")
print(f"  Slippage vs. Arrival Price:    {(avg_fill_price - arrival_price):>11.2f}  ({(avg_fill_price/arrival_price - 1)*10000:>6.1f} bps)")
print(f"  Slippage vs. Intraday VWAP:    {(avg_fill_price - intraday_vwap):>11.2f}  ({vwap_slippage_bps:>6.1f} bps)")
print()

# Interpretation
if total_cost_bps < 10:
    assessment = "✓ Excellent execution (< 10 bps)"
elif total_cost_bps < 25:
    assessment = "✓ Good execution (10-25 bps)"
elif total_cost_bps < 50:
    assessment = "⚠️  Acceptable execution (25-50 bps)"
else:
    assessment = "⚠️  Poor execution (> 50 bps)"

print(f"Assessment: {assessment}")
print("=" * 70)

# Visualize fills over time
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(intraday_prices, label='Intraday Price', linewidth=2)
plt.scatter(fill_times, fill_prices, color='red', s=100, zorder=5, label='Fill Prices')
plt.axhline(arrival_price, color='green', linestyle='--', linewidth=2, label=f'Arrival Price (${arrival_price:.2f})')
plt.axhline(intraday_vwap, color='orange', linestyle=':', linewidth=2, label=f'Intraday VWAP (${intraday_vwap:.2f})')
plt.xlabel('Time (Intraday Intervals)')
plt.ylabel('Price ($)')
plt.title('Execution TCA: Fill Prices vs. Benchmarks')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Expected Output (illustrative):**
```
======================================================================
TRANSACTION COST ANALYTICS (Implementation Shortfall)
======================================================================
Order Details:
  Total Order Size:                100,000 shares
  Arrival Price (Decision):          $50.00
  Closing Price:                     $50.32
  Intraday Price Move:                 0.32 (+0.64%)

Execution Summary:
  Filled Shares:                  95,000 (95%)
  Unfilled Shares:                 5,000 (5%)
  Average Fill Price:                $50.18
  Number of Fills:                       20

Cost Breakdown:
  Execution Cost (Filled):        $17,100.00  ( 36.0 bps)
  Opportunity Cost (Unfilled):     $1,600.00  (  3.2 bps)
  Total Implementation Shortfall: $18,700.00  ( 37.4 bps)

Benchmark Comparison:
  Slippage vs. Arrival Price:          0.18  ( 36.0 bps)
  Slippage vs. Intraday VWAP:          0.08  ( 16.0 bps)

Assessment: ⚠️  Acceptable execution (25-50 bps)
======================================================================
```

**Interpretation:**  
- 37.4 bps total cost: Within acceptable range for large order (10% ADV).  
- Most cost from execution (36 bps), minimal opportunity cost (3.2 bps, high fill rate).  
- Slippage vs. VWAP (16 bps): Algo underperformed intraday average—consider more passive approach or dark pool routing.

---

## Challenge Round

1. **Arrival Price vs. VWAP as Benchmark**  
   Trader receives order at 10 AM (arrival $50), waits until 11 AM to start executing (VWAP $50.50), fills at avg $50.55. Which benchmark shows better performance?

   <details><summary>Solution</summary>
   **Arrival Price:** Slippage = $50.55 - $50.00 = $0.55 (55 bps) → Poor.  
   **VWAP:** Slippage = $50.55 - $50.50 = $0.05 (5 bps) → Good.  
   **Reality:** Delay from 10-11 AM cost 50 bps (timing cost); algo execution from 11 AM was efficient (+5 bps vs. VWAP). Arrival price correctly captures total cost; VWAP obscures delay penalty. Use arrival for accountability.
   </details>

2. **Dark Pool Information Leakage**  
   Order routed to dark pool, no fills. Order then goes to lit exchange at worse price (market moved 10 bps). Did dark pool help or hurt?

   <details><summary>Solution</summary>
   Dark pool hurt: Information leakage—other participants saw order, inferred buy interest, front-ran on lit exchange. **Mitigation:** (1) Limit dark pool exposure time, (2) Use inverted venues (pay for liquidity provision, less info leakage), (3) Randomize routing to prevent pattern detection.
   </details>

3. **Pre-Trade vs. Post-Trade TCA**  
   Pre-trade forecast: 20 bps cost for 100k share order. Post-trade actual: 50 bps. Should trader be penalized?

   <details><summary>Hint</summary>
   Not necessarily. **Factors:** (1) Did market conditions change? (Volatility spike would increase cost unavoidably). (2) Was order more urgent than expected? (Rushed execution increases impact). (3) Model recalibration—if consistently underestimating, update model parameters. Evaluate trader on *best execution given constraints*, not vs. stale forecast.
   </details>

4. **Opportunity Cost of Cancellations**  
   Order to buy 50k shares at $100. Stock rallies; trader cancels at $102 (didn't chase). Stock closes $105. What is opportunity cost?

   <details><summary>Solution</summary>
   If order had executed at decision price ($100), value gained = 50k × ($105 - $100) = $250k.  
   By canceling, opportunity cost = $250k (forgone profit).  
   **Measurement:** Implementation shortfall includes unfilled shares at closing price: 50k × ($105 - $100) = $250k opportunity cost.  
   **Trade-off:** Trader avoided paying $102 (saved $100k vs. chasing) but lost $250k in upside → Net cost $150k. Context matters: If rally was unpredictable, canceling was reasonable; if signal suggested continued rally, should have executed.
   </details>

---

## Key References

- **Kissell (2013)**: *The Science of Algorithmic Trading and Portfolio Management* ([Academic Press](https://www.elsevier.com/))
- **Almgren & Chriss (2001)**: "Optimal Execution of Portfolio Transactions" ([Journal of Risk](https://www.jstor.org/))
- **Perold (1988)**: "The Implementation Shortfall" ([Journal of Portfolio Management](https://jpm.pm-research.com/))
- **ITG (Investment Technology Group)**: TCA best practices and white papers ([ITG.com](https://www.itg.com/))

**Further Reading:**  
- Market impact models: Square-root law, Almgren-Chriss, Kyle's lambda  
- Reg NMS and order routing optimization (best execution obligations)  
- TCA for derivatives (options, futures): Mid-price benchmarks, implied volatility slippage
