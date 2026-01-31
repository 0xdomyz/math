# Liquidity Risk

## Concept Skeleton

Liquidity risk in algorithmic trading arises when market depth insufficient to execute orders without significant price impact (market liquidity risk) or when funding constraints force liquidation at unfavorable prices (funding liquidity risk). Bid-ask spreads widen, volume dries up, and large orders move markets—particularly during stress events when correlations spike and natural buyers disappear. Algorithmic strategies must adapt execution (slice orders, use dark pools, wait for liquidity replenishment) or risk amplifying losses through forced selling into illiquid markets.

**Core Components:**
- **Market liquidity risk:** Inability to trade desired size without excessive price impact (wide spreads, thin order books)
- **Funding liquidity risk:** Insufficient cash/margin to maintain positions, forced liquidation (margin calls, cash flow mismatches)
- **Bid-ask spread widening:** Spreads expand from pennies to dollars during stress (cost of immediacy skyrockets)
- **Market impact:** Price moves against order direction as trade executes (permanent impact + temporary impact)
- **Liquidity black holes:** Extreme events where all buyers/sellers disappear (flash crashes, circuit breakers)

**Why it matters:** LTCM (1998) couldn't exit $100B positions without 50% haircuts; 2008 crisis saw credit spreads widen 10×, forcing mass deleveraging; liquidity risk is systemic (affects everyone simultaneously, non-diversifiable).

---

## Comparative Framing

| Dimension | **Normal Market** | **Stressed Market** | **Liquidity Crisis** |
|-----------|-------------------|---------------------|----------------------|
| **Bid-ask spread** | 1–2 cents (10–20 bps) | 5–10 cents (50–100 bps) | $0.50–$5.00 (500+ bps) |
| **Order book depth** | 50,000+ shares at best bid/ask | 5,000 shares (90% reduction) | 100 shares (stub quotes only) |
| **Market impact** | 5–10 bps per $1M trade | 50–100 bps per $1M trade | 500+ bps (unmeasurable, gapping) |
| **Correlation** | 0.3–0.5 (diversification works) | 0.7–0.9 (crowded exits) | 0.95+ (everything moves together) |
| **Execution time** | Minutes (VWAP, TWAP) | Hours (patient limit orders) | Days (no natural counterparty) |
| **Example** | Routine $10M S&P 500 trade | August 2015 ETF dislocations | March 2020 COVID crash (VIX 82) |

**Key insight:** Liquidity evaporates precisely when needed most (tail risk, non-linear); strategies profitable in calm markets can be catastrophic in stress if liquidity-dependent.

---

## Examples & Counterexamples

### Examples of Liquidity Risk Events

1. **LTCM (1998)—Convergence Trade Liquidity Crisis**  
   - **Strategy:** Long Russian bonds, short US Treasuries (spread convergence bet, 50:1 leverage).  
   - **Trigger:** Russia defaults (August 1998), flight-to-quality → spreads widen instead of converge.  
   - **Liquidity trap:** LTCM needs to exit $100B positions, but no buyers (all hedge funds doing same trade).  
   - **Market impact:** Attempting to sell would push spreads 10× wider, realize massive losses.  
   - **Outcome:** Federal Reserve-orchestrated bailout (14 banks inject $3.6B), positions unwound over months.  
   - **Lesson:** Position size must account for exit liquidity, not just entry; leverage amplifies liquidity risk.

2. **2008 Credit Crisis—Funding Liquidity Evaporation**  
   - **Scenario:** Investment-grade corporate bond spreads widen from 100 bps (normal) to 600 bps (October 2008).  
   - **Forced sellers:** Hedge funds receive margin calls, must liquidate bonds to raise cash.  
   - **Liquidity spiral:** Selling pressure pushes spreads wider → more margin calls → more forced selling.  
   - **Bid-ask spreads:** Some corporate bonds had $5–$10 spreads (5–10% round-trip cost, normally 0.1%).  
   - **Result:** Funds liquidating at 50–70 cents on dollar, many bankruptcies (not credit losses, liquidity losses).

3. **Flash Crash (May 6, 2010)—Market Liquidity Withdrawal**  
   - **Trigger:** $4.1B E-mini S&P futures sell program (automated execution).  
   - **Liquidity providers retreat:** HFT market makers detect unusual volatility, widen quotes or shut down.  
   - **Bid-ask spread:** S&P futures spread widened from 1 tick (0.25 bps) to 20+ ticks (5 bps).  
   - **Price impact:** Futures dropped 3% in 4 minutes (order exhausted liquidity, no natural buyers).  
   - **Cascade:** Equity markets followed (Dow -600 points), some stocks traded at $0.01 (no bids).  
   - **Recovery:** Liquidity returned after 20 minutes (algorithmic market makers resumed quoting), prices rebounded.

4. **March 2020 COVID Crash—Treasury Market Liquidity Breakdown**  
   - **Paradox:** Flight-to-safety, but even US Treasuries illiquid (bid-ask spreads widened 10×).  
   - **Cause:** Dealers couldn't warehouse positions (balance sheet constraints), leveraged funds forced to sell everything.  
   - **Fed intervention:** $1.5 trillion repo operations, unlimited QE → restored liquidity in days.  
   - **Impact on algos:** VWAP strategies unable to execute (no volume to blend into), required urgent orders at market (high impact).

### Non-Examples (or Liquidity Resilience)

- **S&P 500 large-cap stocks (AAPL, MSFT):** $5M order typically 5–10 bps impact (deep liquidity, absorbs without stress).
- **Patient execution (5-day VWAP):** Breaking $50M order into 1-day slices → minimal impact (spreads participation over time).
- **Dark pool usage:** 30% of order filled at midpoint (no spread cost, no information leakage → lower impact).

---

## Layer Breakdown

**Layer 1: Market Liquidity and Order Book Dynamics**  
**Order book structure:** Bids (buy orders) and asks (sell orders) stacked by price; top-of-book = best bid/ask (NBBO).

**Liquidity metrics:**  
- **Bid-ask spread:** Cost of immediacy (market order pays spread); tight spread (1–2 cents) = liquid, wide spread ($0.50+) = illiquid.  
- **Depth:** Total shares at best bid/ask; deep book (50k+ shares) = absorb large orders, thin book (100 shares) = price sensitive.  
- **Resilience:** Speed at which liquidity replenishes after large order; resilient market recovers in seconds, fragile market stays dislocated.

**Market impact model (Almgren-Chriss):**  
$$\text{Total Cost} = \text{Spread Cost} + \text{Temporary Impact} + \text{Permanent Impact}$$

- **Spread cost:** $\frac{1}{2} \times \text{spread} \times \text{shares}$ (pay half-spread on average if using limit orders)  
- **Temporary impact:** $\eta \times \frac{\text{trade size}}{\text{ADV}} \times \sigma$ (reverses after trade, $\eta$ = market impact parameter)  
- **Permanent impact:** $\gamma \times \frac{\text{trade size}}{\text{ADV}} \times \sigma$ (information leakage, price doesn't fully revert)

**Layer 2: Funding Liquidity and Margin Constraints**  
**Funding liquidity risk:** Inability to meet cash obligations (margin calls, collateral requirements, redemptions) without selling assets.

**Sources:**  
- **Leverage:** Portfolio worth $100M, borrowed $80M → if portfolio drops 10% to $90M, equity = $10M; if lenders demand higher margin (30% → 40%), must post $36M collateral (need to liquidate $26M).  
- **Redemptions:** Mutual fund/hedge fund investors withdraw capital → fund must sell positions to raise cash (even if markets unfavorable).  
- **Margin calls:** Futures position loses money, clearinghouse demands variation margin by end of day (must have cash, can't wait to "ride it out").

**Liquidity spiral (Brunnermeier-Pedersen):**  
1. Asset prices drop → Margin calls triggered → Forced selling  
2. Forced selling → Prices drop further → More margin calls  
3. Feedback loop amplifies initial shock (non-linear, self-reinforcing).

**Mitigation:**  
- **Liquidity buffers:** Hold 10% of portfolio in cash/T-bills (can meet margin calls without selling).  
- **Lower leverage:** 2:1 instead of 10:1 (less sensitive to small price moves).  
- **Staggered maturities:** Don't borrow short (30-day repo) to fund long-term positions (3-year bonds); maturity mismatch forces refinancing at worst times.

**Layer 3: Algorithmic Execution and Liquidity Adaptation**  
**Execution strategies for illiquid markets:**

**1. VWAP (Volume-Weighted Average Price):**  
- **Logic:** Split order to match intraday volume distribution (trade 20% of order in hour 1 if hour 1 typically 20% of daily volume).  
- **Benefit:** Blend into market flow, minimize footprint.  
- **Risk:** If market volume dries up (stressed conditions), VWAP struggles (can't find counterparty).

**2. TWAP (Time-Weighted Average Price):**  
- **Logic:** Execute equal slices every 5 minutes regardless of volume.  
- **Benefit:** Simple, predictable.  
- **Risk:** May trade during low-liquidity periods (lunch hour, end of day), incur higher impact.

**3. Participation Rate (Target X% of Volume):**  
- **Logic:** Trade 10% of market volume (if market volume spikes, increase trading; if dries up, slow down).  
- **Benefit:** Adaptive to liquidity availability.  
- **Risk:** Adverse selection (high volume often occurs during price moves against you).

**4. Liquidity-Seeking (Dark Pools, Iceberg Orders):**  
- **Logic:** Ping dark pools for hidden liquidity (midpoint fills, no spread), use iceberg orders (display 100, hide 9,900).  
- **Benefit:** Reduced information leakage, better prices.  
- **Risk:** Low fill rates (20–30% in dark pools), unpredictable execution time.

**5. Adaptive Urgency:**  
- **Logic:** Start patient (limit orders, wait for liquidity), increase urgency if time running out or market moving against.  
- **Implementation:** If half-day elapsed and only 20% filled, switch to more aggressive strategy (take liquidity with market orders).

**Layer 4: Liquidity Risk Measurement**  
**Metrics:**

**1. Bid-Ask Spread:**  
$$\text{Relative Spread} = \frac{\text{Ask} - \text{Bid}}{\text{Midpoint}} \times 10000 \text{ (bps)}$$  
Tight spread (<10 bps) = liquid, wide spread (>100 bps) = illiquid.

**2. Amihud Illiquidity Ratio:**  
$$\text{ILLIQ} = \frac{1}{T} \sum_{t=1}^{T} \frac{|\text{Return}_t|}{\text{Dollar Volume}_t}$$  
Measures price impact per dollar traded; higher ILLIQ = more illiquid (small volume causes large price moves).

**3. Turnover Ratio:**  
$$\text{Turnover} = \frac{\text{Trading Volume}}{\text{Shares Outstanding}}$$  
High turnover (>100% annually) = liquid, low turnover (<10%) = illiquid.

**4. Market Depth:**  
Sum of shares within 1% of midpoint (bid and ask sides). Deep market: 500k+ shares, shallow market: <10k shares.

**5. Liquidity-at-Risk (LaR):**  
Probability that liquidating portfolio exceeds X% price impact. Example: 95% LaR = 10% → 5% chance liquidation costs >10% of portfolio value.

**Layer 5: Stress Testing and Scenario Analysis**  
**Scenario 1: Bid-ask spreads widen 10×**  
- Normal: 2 cent spread, 10,000 share order → $200 spread cost.  
- Stress: 20 cent spread → $2,000 spread cost (10× higher).  
- **Mitigation:** Use limit orders (pay spread passively), increase execution horizon (5 days instead of 1).

**Scenario 2: Order book depth drops 90%**  
- Normal: 50,000 shares at best bid, can sell 10,000 with minimal impact.  
- Stress: 5,000 shares at best bid → 10,000 order exhausts level, walks down order book (5% slippage).  
- **Mitigation:** Slice order smaller (1,000 shares per fill), wait for liquidity replenishment between slices.

**Scenario 3: Funding liquidity crisis (margin call)**  
- Portfolio: $100M, leverage 5:1 ($80M borrowed).  
- Loss: -10% ($10M) → Equity down to $10M.  
- Margin requirement increase: 20% → 30% → Must post $30M collateral, have $10M equity → Need $20M.  
- **Forced liquidation:** Sell $20M in stressed market (10% impact) → Realize additional $2M loss.  
- **Mitigation:** Hold $15M cash buffer (can meet margin without selling), reduce leverage to 2:1.

---

## Mini-Project: Liquidity Risk and Market Impact Simulation

**Goal:** Simulate order execution under varying liquidity conditions.

```python
import numpy as np
import matplotlib.pyplot as plt

# Market parameters
np.random.seed(42)
initial_price = 100.0
n_steps = 100  # Time steps (minutes)

# Order to execute
order_size = 50000  # shares
order_side = 'sell'  # Selling pressure

# Scenario 1: Normal liquidity
normal_spread = 0.02  # 2 cents
normal_depth = 10000  # shares per price level
normal_resilience = 0.90  # 90% of liquidity replenishes each period

# Scenario 2: Stressed liquidity
stressed_spread = 0.20  # 20 cents (10× wider)
stressed_depth = 1000  # shares (90% reduction)
stressed_resilience = 0.50  # 50% replenishment (slower recovery)

def simulate_execution(order_size, spread, depth, resilience, n_steps):
    """Simulate order execution with market impact"""
    prices = [initial_price]
    remaining = order_size
    cumulative_cost = 0
    available_liquidity = depth
    
    for t in range(n_steps):
        if remaining <= 0:
            prices.append(prices[-1])
            continue
        
        # Determine execution size (min of remaining, available liquidity, or 1% of order)
        execution_size = min(remaining, available_liquidity, order_size * 0.01)
        
        # Market impact: temporary (proportional to size / depth) + spread cost
        temporary_impact = (execution_size / depth) * 0.50  # 50 cent max impact
        spread_cost = spread / 2  # Pay half-spread on average
        
        # Price moves down (selling pressure) by temporary impact
        new_price = prices[-1] - temporary_impact - spread_cost
        
        # Permanent impact (10% of temporary impact persists)
        permanent_impact = temporary_impact * 0.10
        
        # Update state
        remaining -= execution_size
        cumulative_cost += execution_size * (initial_price - new_price)
        prices.append(new_price - permanent_impact)
        
        # Liquidity replenishes partially
        available_liquidity = depth * resilience + (depth - available_liquidity) * (1 - resilience)
    
    # Average execution price
    avg_price = initial_price - (cumulative_cost / order_size)
    slippage = (initial_price - avg_price) / initial_price * 10000  # bps
    
    return prices, avg_price, slippage

# Run simulations
prices_normal, avg_price_normal, slippage_normal = simulate_execution(
    order_size, normal_spread, normal_depth, normal_resilience, n_steps
)

prices_stressed, avg_price_stressed, slippage_stressed = simulate_execution(
    order_size, stressed_spread, stressed_depth, stressed_resilience, n_steps
)

# Display results
print("=" * 80)
print("LIQUIDITY RISK: MARKET IMPACT SIMULATION")
print("=" * 80)
print(f"Order: SELL {order_size:,} shares")
print(f"Initial Price: ${initial_price:.2f}")
print()

print("SCENARIO 1: NORMAL LIQUIDITY")
print("-" * 80)
print(f"  Bid-Ask Spread:           ${normal_spread:.2f}  ({normal_spread/initial_price*10000:.0f} bps)")
print(f"  Order Book Depth:         {normal_depth:,} shares per level")
print(f"  Liquidity Resilience:     {normal_resilience:.0%} replenishment")
print(f"\n  Average Execution Price:  ${avg_price_normal:.4f}")
print(f"  Slippage:                 {slippage_normal:.1f} bps")
print(f"  Total Cost:               ${(initial_price - avg_price_normal) * order_size:,.0f}")
print()

print("SCENARIO 2: STRESSED LIQUIDITY (Crisis)")
print("-" * 80)
print(f"  Bid-Ask Spread:           ${stressed_spread:.2f}  ({stressed_spread/initial_price*10000:.0f} bps)  [10× wider]")
print(f"  Order Book Depth:         {stressed_depth:,} shares per level  [90% reduction]")
print(f"  Liquidity Resilience:     {stressed_resilience:.0%} replenishment  [slower recovery]")
print(f"\n  Average Execution Price:  ${avg_price_stressed:.4f}")
print(f"  Slippage:                 {slippage_stressed:.1f} bps")
print(f"  Total Cost:               ${(initial_price - avg_price_stressed) * order_size:,.0f}")
print()

print("COMPARISON")
print("-" * 80)
print(f"  Slippage Increase:        {slippage_stressed / slippage_normal:.1f}× higher in crisis")
print(f"  Additional Cost:          ${(avg_price_normal - avg_price_stressed) * order_size:,.0f}")
print("=" * 80)

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Price path
axes[0].plot(prices_normal, label='Normal Liquidity', linewidth=2, color='blue')
axes[0].plot(prices_stressed, label='Stressed Liquidity', linewidth=2, color='red')
axes[0].axhline(initial_price, color='black', linestyle='--', linewidth=1, label='Initial Price')
axes[0].axhline(avg_price_normal, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
axes[0].axhline(avg_price_stressed, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
axes[0].set_ylabel('Price ($)', fontsize=12, fontweight='bold')
axes[0].set_title('Liquidity Risk: Market Impact During Order Execution', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

# Slippage accumulation
cumulative_slippage_normal = (initial_price - np.array(prices_normal)) / initial_price * 10000
cumulative_slippage_stressed = (initial_price - np.array(prices_stressed)) / initial_price * 10000

axes[1].plot(cumulative_slippage_normal, label='Normal Liquidity', linewidth=2, color='blue')
axes[1].plot(cumulative_slippage_stressed, label='Stressed Liquidity', linewidth=2, color='red')
axes[1].set_ylabel('Cumulative Slippage (bps)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('liquidity_risk_simulation.png', dpi=150)
plt.show()

# Liquidity stress test summary
print("\n" + "=" * 80)
print("LIQUIDITY STRESS TEST SUMMARY")
print("=" * 80)
print("Portfolio: $50M position to liquidate")
print()
print(f"{'Scenario':<25} {'Slippage (bps)':<20} {'Cost ($)':<20} {'Days to Complete':<20}")
print("-" * 80)
print(f"{'Normal (5-day VWAP)':<25} {slippage_normal:<20.1f} ${(initial_price - avg_price_normal) * order_size:<19,.0f} {'5 days':<20}")
print(f"{'Stressed (Urgent)':<25} {slippage_stressed:<20.1f} ${(initial_price - avg_price_stressed) * order_size:<19,.0f} {'1 day (forced)':<20}")
print(f"{'Crisis (Firesale)':<25} {slippage_stressed * 2:<20.1f} ${(initial_price - avg_price_stressed) * order_size * 2:<19,.0f} {'Immediate (minutes)':<20}")
print("=" * 80)
```

**Expected Output (illustrative):**
```
================================================================================
LIQUIDITY RISK: MARKET IMPACT SIMULATION
================================================================================
Order: SELL 50,000 shares
Initial Price: $100.00

SCENARIO 1: NORMAL LIQUIDITY
--------------------------------------------------------------------------------
  Bid-Ask Spread:           $0.02  (2 bps)
  Order Book Depth:         10,000 shares per level
  Liquidity Resilience:     90% replenishment

  Average Execution Price:  $99.9650
  Slippage:                 35.0 bps
  Total Cost:               $1,750

SCENARIO 2: STRESSED LIQUIDITY (Crisis)
--------------------------------------------------------------------------------
  Bid-Ask Spread:           $0.20  (200 bps)  [10× wider]
  Order Book Depth:         1,000 shares per level  [90% reduction]
  Liquidity Resilience:     50% replenishment  [slower recovery]

  Average Execution Price:  $99.3200
  Slippage:                 680.0 bps
  Total Cost:               $34,000

COMPARISON
--------------------------------------------------------------------------------
  Slippage Increase:        19.4× higher in crisis
  Additional Cost:          $32,250
================================================================================

================================================================================
LIQUIDITY STRESS TEST SUMMARY
================================================================================
Portfolio: $50M position to liquidate

Scenario                  Slippage (bps)       Cost ($)             Days to Complete    
--------------------------------------------------------------------------------
Normal (5-day VWAP)       35.0                 $1,750               5 days              
Stressed (Urgent)         680.0                $34,000              1 day (forced)      
Crisis (Firesale)         1360.0               $68,000              Immediate (minutes) 
================================================================================
```

**Interpretation:**  
Stressed liquidity increases slippage 19× (35 bps → 680 bps). Forced liquidation in crisis catastrophic (>1,000 bps impact). Lesson: Maintain liquidity buffers to avoid forced selling; size positions for stressed-market exit (not normal-market capacity).

---

## Challenge Round

1. **LTCM Position Sizing Error**  
   LTCM held $100B positions with $4B equity (25:1 leverage). Average daily liquidity in their trades: $500M. Estimate days to unwind if no price impact.

   <details><summary>Hint</summary>**Calculation:** $100B / $500M per day = 200 days to unwind at normal pace. With price impact (each sale pushes prices down, reducing remaining position value), actual unwinding takes longer and incurs losses. Rule of thumb: Position size should be <10 days of average liquidity to exit without catastrophic impact. LTCM violated by 20× → unsustainable.</details>

2. **Liquidity Spiral Mechanics**  
   Portfolio drops 10% ($100M → $90M). Lender increases margin from 20% to 30%. Calculate forced liquidation size.

   <details><summary>Solution</summary>
   **Initial:** $100M assets, $80M debt, $20M equity (20% margin).  
   **After 10% loss:** $90M assets, $80M debt, $10M equity (11% margin).  
   **New requirement:** 30% margin → Need $27M equity ($90M × 30%).  
   **Shortfall:** $27M required - $10M have = $17M.  
   **Forced sale:** Sell $17M assets → Raise $17M cash → Post as collateral (alternatively, if lender demands debt paydown, must sell $17M, pay down debt to $63M, leaving $73M assets, $63M debt, $10M equity = 14% margin—still short, need more selling). Iterative deleveraging until 30% margin met.
   </details>

3. **Dark Pool vs. Lit Exchange Trade-Off**  
   $10M order: Dark pools offer midpoint (save spread), but only 30% fill rate. Lit exchange: full fill, pay spread (10 bps). Which cheaper?

   <details><summary>Solution</summary>
   **Dark pool:** 30% × $10M = $3M filled at midpoint (0 bps), 70% × $10M = $7M route to lit (10 bps). Weighted cost: 0.3 × 0 + 0.7 × 10 = **7 bps**.  
   **Lit only:** 10 bps.  
   **Winner:** Dark pool saves 3 bps. **But:** Dark pool adds execution delay (need to ping, wait, timeout, then route to lit)—opportunity cost if price moves 20 bps against you during delay. **Conclusion:** Use dark pools for patient orders (no urgency), lit exchanges for urgent (certainty > cost).
   </details>

4. **Amihud Illiquidity Ratio Interpretation**  
   Stock A: ILLIQ = 0.01 (1% return per $1M volume). Stock B: ILLIQ = 1.00 (1% return per $10K volume). Which riskier for $5M order?

   <details><summary>Solution</summary>
   **Stock A:** $5M order → Expected impact = 0.01 × 5 = **5% price move** (manageable).  
   **Stock B:** $5M order → Expected impact = 1.00 × 5 = **500% price move** (impossible—order exhausts all liquidity, trades at any price). Stock B catastrophically illiquid for $5M order.  
   **Rule:** Order size should be <10× inverse of ILLIQ. Stock A: Max order ~$10M. Stock B: Max order ~$10K. $5M order in Stock B is 500× too large.
   </details>

---

## Key References

- **Almgren & Chriss (2000)**: "Optimal Execution of Portfolio Transactions" ([NYU Stern](https://www.stern.nyu.edu/))
- **Amihud (2002)**: "Illiquidity and Stock Returns" (illiquidity measure) ([JFE](https://www.sciencedirect.com/science/article/pii/S0304405X01000781))
- **Brunnermeier & Pedersen (2009)**: "Market Liquidity and Funding Liquidity" (liquidity spirals) ([RFS](https://academic.oup.com/rfs))
- **Adrian & Shin (2010)**: "Liquidity and Leverage" (procyclical leverage dynamics) ([JFI](https://www.journals.uchicago.edu/))

**Further Reading:**  
- When Genius Failed (LTCM, liquidity risk at extreme leverage)  
- Flash crash (SEC-CFTC report): Liquidity provider withdrawal mechanisms  
- BIS Market Liquidity Reports: Post-crisis changes in market-making business models
