# Temporary vs Permanent Market Impact Components

## 1. Concept Skeleton
**Definition:** Decomposition of total price concession into temporary (bid-ask spread, dealer inventory cost; reverses post-trade) and permanent (information revelation, price adjustment; persistent) components  
**Purpose:** Understand cost structure; identify which costs are recoverable through timing; inform algorithm design  
**Prerequisites:** Market microstructure theory, bid-ask spread, adverse selection, order processing

## 2. Comparative Framing
| Component | **Temporary** | **Permanent** | **Total** |
|-----------|---------------|---------------|----------|
| **Persistence** | Reverts within seconds–minutes | Persists indefinitely | Sum of above |
| **Cause** | Bid-ask spread, inventory cost | Information leakage, price discovery | Combined microstructure |
| **Recoverable** | Partially (wait for reversion) | Not (lasting price move) | Determines true cost |
| **Timing Strategy** | Benefit from reversion | Cannot avoid | Optimize across both |
| **Example** | Buy at ask, revert to mid | Price jumps on info → stays up | Total cost to trader |

## 3. Examples + Counterexamples

**Simple Example (Buy Order):**  
Mid price $100, bid-ask spread $0.02 (2 cents = 2 bps).  
- Buy market order fills at ask = $100.02  
- Temporary impact: 2 bps (spread; reverts when dealer unwinds)  
- Permanent impact: 0 bps (no information reveal; routine buying)  
- Total cost: 2 bps

**Information Example (Insider):**  
Mid price $100; insider knows stock will move to $102 (takeover).  
- Insider buys $10M (2% of daily volume)  
- Temporary impact: Bid-ask spread + dealer cost = 3 bps  
- Permanent impact: 150 bps (information-driven price jump; doesn't revert)  
- Total cost: 153 bps

**Non-Example (Routine Rebalancing):**  
Fund quarterly rebalancing; sells $50M overweight tech stock.  
- Market perceives as uninformed (routine, not information-based)  
- Temporary impact: 2–3 bps (spread + inventory)  
- Permanent impact: 0–1 bps (minimal; no information content)  
- Total: 3–4 bps

**Counter-Example (Split Components):**  
Sell 1,000 shares at $100 mid. Execution:  
- Sell 500 at $99.98 (ask-side, spread = 2 bps temporary)  
- Market immediately reverts to $100.00 mid (spread reversion; temporary cost vanishes)  
- Sell remaining 500 at $99.95 (new dealer inventory concession = 5 bps temporary)  
- After trade: New mid = $99.98 (price moved 2 cents down; permanent component)  
- Permanent impact: 2 bps (reversion didn't occur fully; info content)  
- Total: 4 bps temporary + 2 bps permanent = 6 bps

**Edge Case (Highly Liquid Asset):**  
Blue-chip stock with tight spread (0.5 bps), high volume.  
- Permanent impact: Near-zero (liquidity so deep, trade doesn't signal info)  
- Temporary impact: 0.5 bps (spread only)  
- Total: 0.5 bps (split heavily toward temporary)

**Edge Case (Illiquid Asset):**  
Small-cap illiquid stock, wide spread (20 bps).  
- Temporary impact: 20 bps spread  
- Permanent impact: 30 bps (trade signals scarcity; dealers post wider for follow-on orders)  
- Total: 50 bps (significant permanent component)

## 4. Layer Breakdown
```
Market Impact Decomposition Framework:

├─ Total Market Impact Concept:
│   ├─ Definition:
│   │   ├─ Total Impact = Executed Price − Counterfactual "No Trade" Price
│   │   ├─ Usually measured vs mid-price (observable benchmark)
│   │   ├─ Example: Execute sell at $99.95; mid = $100 → Total Impact = 5 bps
│   │   └─ Paid by trader; benefit to dealer and market participants
│   │
│   ├─ Decomposition:
│   │   ├─ Total Impact = Temporary Impact + Permanent Impact
│   │   ├─ Notation: I_total = I_temp + I_perm
│   │   ├─ Interpretation: Immediate cost + Lasting price move
│   │   └─ Time: Temporary reverts in minutes; permanent persists
│   │
│   └─ Measurement Challenge:
│       ├─ Counterfactual "no-trade" price unknown (not observed)
│       ├─ Proxy: Use mid-price at execution as benchmark
│       ├─ Assumption: Mid-price represents price without trade
│       ├─ Reality: Mid-price itself changes (stale vs fresh quotes)
│       └─ Estimation: Use post-trade price reversion to extract components
│
├─ Temporary Impact Component:
│   ├─ Definition:
│   │   ├─ Price concession that reverts shortly after trade
│   │   ├─ Reflects immediate transaction costs (bid-ask spread, dealer fees)
│   │   ├─ Not information-based (routine trading)
│   │   └─ Typically reverts within minutes to hours
│   │
│   ├─ Sources:
│   │   ├─ Bid-Ask Spread:
│   │   │   ├─ Buy → fill above mid (pay ask premium)
│   │   │   ├─ Sell → fill below mid (pay bid discount)
│   │   │   ├─ Spread reflects dealer inventory cost, adverse selection
│   │   │   └─ Typical: 1–2 bps large-cap equity, 5–20 bps small-cap
│   │   │
│   │   ├─ Inventory Cost:
│   │   │   ├─ After trade, dealer has imbalanced position (long/short excess)
│   │   │   ├─ Dealer widen spread to incentivize offsetting orders
│   │   │   ├─ Trader bears cost of imbalance (concession to dealer)
│   │   │   ├─ Duration: Few minutes to hours (until dealer hedges)
│   │   │   └─ Magnitude: 1–5 bps depending on dealer size, market conditions
│   │   │
│   │   ├─ Price Discretion (Market Maker Behavior):
│   │   │   ├─ Dealer sees large order; quotes worse immediately
│   │   │   ├─ Reflects option value: Dealer wants protection on downside
│   │   │   ├─ Temporary: Once trade done, quotes reset
│   │   │   └─ Example: Buy 1,000 shares; dealer quotes 5 pips wider pre-trade, normal post
│   │   │
│   │   └─ Non-Information Component of Spread:
│   │       ├─ Mechanical spread: Order processing, adverse selection (pre-trade)
│   │       ├─ Reverts: As order flow normalizes, quotes tighten
│   │       └─ Timing: Minutes to several hours
│   │
│   ├─ Measurement:
│   │   ├─ Observed Immediate: Trade price vs mid-price at execution
│   │   │   ├─ Example: Sell 1,000 shares; execute at $99.95; mid at execution = $100
│   │   │   ├─ Observed price concession = 5 bps
│   │   │   └─ This mixes temporary and permanent (can't separate directly)
│   │   │
│   │   ├─ Post-Trade Reversion Method:
│   │   │   ├─ Measure price 1 min, 5 min, 30 min after trade
│   │   │   ├─ Temporary impact ≈ partial reversion (portion that comes back)
│   │   │   ├─ Example:
│   │   │   │   ├─ Execute at $99.95 (5 bps down from $100 mid)
│   │   │   │   ├─ 5 min later: Mid = $99.98 (reverts 3 bps, stays down 2 bps)
│   │   │   │   ├─ Estimated temp ≈ 3 bps (reverted portion)
│   │   │   │   ├─ Estimated perm ≈ 2 bps (persistent portion)
│   │   │   │   └─ Caveat: Other trades may have moved price; imperfect
│   │   │
│   │   └─ Regression Approach (High-Frequency Data):
│   │       ├─ Use limit order book data with millisecond timing
│   │       ├─ Regress future mid-price change on current order flow
│   │       ├─ Decompose: Immediate vs 1-hour-ahead vs permanent effect
│   │       └─ Requires detailed tick data; not always available
│   │
│   ├─ Characteristics:
│   │   ├─ Mean Reversion: Temporary cost tends to reverse quickly
│   │   ├─ Predictable: Can be partially anticipated (dealer behavior pattern)
│   │   ├─ Size-Dependent: Larger orders → wider spreads → more temporary impact
│   │   ├─ Volatile: Fluctuates with market conditions (bid-ask changes intraday)
│   │   └─ Recoverable: Waiting can capture reversion (but faces opportunity cost)
│   │
│   └─ Example Impact Across Liquid/Illiquid Spectrum:
│       ├─ Apple (very liquid): Temporary ≈ 2 bps, Permanent ≈ 0 bps
│       ├─ Mid-cap tech: Temporary ≈ 5 bps, Permanent ≈ 2 bps
│       ├─ Small-cap: Temporary ≈ 15 bps, Permanent ≈ 5 bps
│       └─ Illiquid micro-cap: Temporary ≈ 40 bps, Permanent ≈ 20 bps
│
├─ Permanent Impact Component:
│   ├─ Definition:
│   │   ├─ Price move that persists indefinitely post-trade
│   │   ├─ Reflects information revelation or market efficiency update
│   │   ├─ Trader's order implies information (even if unintentional)
│   │   ├─ Market interprets order size/timing as signal → adjusts expectations
│   │   └─ Permanent: Doesn't revert (reflects new equilibrium)
│   │
│   ├─ Rationale (Microstructure Theory):
│   │   ├─ Asymmetric Information Model (Kyle 1985):
│   │   │   ├─ Insiders trade on private info; uninformed participate routinely
│   │   │   ├─ Dealer can't distinguish; prices orders up by estimated info content
│   │   │   ├─ Large order → higher probability of insider info → bigger price move
│   │   │   ├─ Permanent component compensates dealer for risk
│   │   │   └─ Formula: Permanent = λ × (Order Size / Market Depth)
│   │   │
│   │   ├─ Price Discovery Process:
│   │   │   ├─ Order flow reveals demand/supply imbalance
│   │   │   ├─ Market maker updates quote; becomes new midpoint
│   │   │   ├─ Subsequent traders see new mid (don't experience reversion)
│   │   │   └─ Permanent impact: Equilibrium price adjustment
│   │   │
│   │   └─ Inventory Pressure (Ho & Stoll 1981):
│   │       ├─ Dealers adjust prices to offload inventory imbalance
│   │       ├─ If dealer has excess short position: Raises quotes (buy incentive)
│   │       ├─ This move reflects inventory pressure; permanent until balanced
│   │       └─ Differs from temporary spread (which tightens once dealer hedges)
│   │
│   ├─ Sources:
│   │   ├─ Information-Based Impact:
│   │   │   ├─ Informed trader signal: Insider buying → permanent price increase
│   │   │   ├─ Herding effect: If large buyer → others follow → price stays up
│   │   │   ├─ Momentum: Technical traders react; price continues direction
│   │   │   └─ Magnitude: Depends on transparency (dark pool vs lit exchange)
│   │   │
│   │   ├─ Inventory Adjustment:
│   │   │   ├─ Dealer carries imbalanced position temporarily
│   │   │   ├─ Prices stay adjusted until hedge executed (hours to days)
│   │   │   ├─ If hedge involves adverse price (market moved), stays embedded
│   │   │   └─ Magnitude: 1–10 bps depending on market depth
│   │   │
│   │   ├─ Volatility Increase:
│   │   │   ├─ Large order indicates high activity; uncertainty → wider quotes
│   │   │   ├─ Volatility shock persists (market reassesses risk)
│   │   │   ├─ Bid-ask spread widens (not temporary tightening)
│   │   │   └─ Permanent: Until volatility decays
│   │   │
│   │   └─ Correlation Impact:
│   │       ├─ Sell in correlated asset class → signals sector stress
│   │       ├─ Prices of related assets adjust; permanent revaluation
│   │       ├─ Example: Sell tech ETF → individual tech stocks fall
│       └─ Magnitude: Can be significant in crash scenarios
│   │
│   ├─ Measurement:
│   │   ├─ Residual Method:
│   │   │   ├─ Total Impact (observed) = Temporary + Permanent
│   │   │   ├─ Estimate Temporary → Permanent = Total − Temporary
│   │   │   ├─ Relies on accurate temporary estimation
│   │   │   └─ Noise in measurement → noise in permanent estimate
│   │   │
│   │   ├─ Long-Horizon Price Change:
│   │   │   ├─ Measure price 1 hour, 1 day, 1 week post-trade
│   │   │   ├─ If price stabilizes (no further reversion), change is permanent
│   │   │   ├─ Example: Post-trade price reverts 60% in 5 min; 5% in 1 hour
│   │   │   │   ├─ After 1 hour: 95% permanent (5% temporary)
│   │   │   │   ├─ After 1 day: Still 95% (assuming no reversion further)
│   │   │   │   └─ Permanent estimate ≈ 95%
│   │   │
│   │   └─ Regression on Order Flow (Econometric):
│   │       ├─ Model: ΔP_t = α + β × Order_Flow_t + ε_t
│   │       ├─ Where ΔP_t = price change over horizon (e.g., 1 hour)
│   │       ├─ β = permanent impact (price doesn't revert; reflected in ΔP)
│   │       ├─ High-frequency estimation: Can separate immediate vs cumulative
│   │       └─ Requires careful timing alignment (tick data methodology)
│   │
│   ├─ Characteristics:
│   │   ├─ Non-reverting: Doesn't decay; persists in new equilibrium
│   │   ├─ Information Content: Signals something about fundamentals/sentiment
│   │   ├─ Unpredictable: Can't be timed away; inherent to large trades
│   │   ├─ Scale-Dependent: Larger orders → more permanent impact (info signal)
│   │   ├─ Correlated: Across securities (especially correlated ones)
│   │   └─ Non-Recoverable: Can't capture via patience/timing strategies
│   │
│   └─ Example Impact Across Scenarios:
│       ├─ Routine rebalancing (low info): Permanent ≈ 0–1 bps
│       ├─ Earnings surprise flow: Permanent ≈ 10–50 bps
│       ├─ Forced liquidation (fire sale): Permanent ≈ 5–20 bps (signal of stress)
│       ├─ Insider trade (high info): Permanent ≈ 50–300+ bps
│       └─ Crash/panic: Permanent ≈ 100+ bps (market repricing)
│
└─ Practical Implications for Trading:
    ├─ Strategy Design:
    │   ├─ Patient execution (e.g., VWAP): Captures some temporary reversion
    │   │   ├─ Example: Split 10k share order over 10 min
    │   │   ├─ If 50% of impact is temporary, patience saves ~1–2 bps
    │   │   ├─ But faces opportunity cost (price could move +/- during window)
    │   │   └─ Net benefit: Depends on volatility
    │   │
    │   ├─ Permanent impact unavoidable: Just part of trading cost
    │   │   ├─ Don't waste time trying to time around it
    │   │   ├─ Focus on minimizing temporary component
    │   │   └─ Accept as cost of doing business
    │   │
    │   └─ Participation rate (POV, IS):
    │       ├─ Participation dampens immediate impact (spreads execution)
    │       ├─ Temporary component reduced (lower per-trade concession)
    │       ├─ But trades extend over longer period (opportunity cost increases)
    │       └─ Optimal: Balance temporary (benefit from patience) vs permanent (harm from delay)
    │
    ├─ Venue Selection:
    │   ├─ Lit exchange (visible): Higher permanent impact (transparent order flow)
    │   ├─ Dark pool (hidden): Lower permanent impact (order size hidden)
    │   ├─ High-frequency venues: Lower temporary (tight spreads) but high permanent (fast info)
    │   └─ Trade-off: Transparency vs info leakage
    │
    └─ Performance Measurement:
        ├─ Implementation Shortfall = Total Impact (temporary + permanent)
        ├─ Don't confuse with temporary alone
        ├─ Attribution: Break down into components if possible (improves strategy)
        └─ Benchmark: VWAP/TWAP includes both; compare execution price to benchmark
```

## 5. Mini-Project: Decomposing Temporary vs Permanent Impact

**Goal:** Use post-trade price data to estimate components; analyze by trade size and condition.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Generate synthetic trade execution data
np.random.seed(42)
n_trades = 500

# Trade parameters
trade_size_pct = np.random.uniform(0.05, 2.0, n_trades)  # % of daily volume
trade_direction = np.random.choice([-1, 1], n_trades)  # -1 = sell, +1 = buy
market_vol = np.random.uniform(0.8, 1.5, n_trades)  # Vol multiplier (normal=1)

# Price movements (bid-ask adjusted)
# Temporary: Spread + inventory cost (decays quickly)
# Permanent: Information (doesn't decay)

# Model: I_total = temp_component + permanent_component
# temp_component = 2 + 0.5 * size + 0.1 * vol (bid-ask + inventory)
# perm_component = 0.1 * size + 0.05 * vol (info-based)

temp_baseline = 2.0  # Bid-ask spread (bps)
temp_slope = 0.5  # Inventory cost per % volume
temp_vol_sensitivity = 0.1

perm_baseline = 0.1
perm_slope = 0.1  # Information content per % volume
perm_vol_sensitivity = 0.05

temp_impact_true = temp_baseline + temp_slope * trade_size_pct + temp_vol_sensitivity * market_vol
perm_impact_true = perm_baseline * trade_size_pct + perm_vol_sensitivity * market_vol

# Add noise
temp_noise = np.random.normal(0, 0.3, n_trades)
perm_noise = np.random.normal(0, 0.1, n_trades)

temp_impact_obs = np.maximum(temp_impact_true + temp_noise, 0.5)
perm_impact_obs = np.maximum(perm_impact_true + perm_noise, 0)

total_impact_obs = temp_impact_obs + perm_impact_obs

# Post-trade price data (simulated)
# Price reverts partially from temp; doesn't revert from perm
reversion_windows = [1, 5, 30]  # minutes
reversion_data = []

for window in reversion_windows:
    # Reversion factor: How much of temporary impact reverts in this window
    # Assume exponential decay: revert_pct = 1 - exp(-t / tau), tau = 10 min
    tau = 10  # Minutes for half-life-ish decay
    revert_pct = 1 - np.exp(-window / tau)
    
    # Observed price change (accounting for partial reversion)
    remaining_temp = temp_impact_obs * (1 - revert_pct)
    observed_price_change = remaining_temp + perm_impact_obs  # Temp partially reverted; perm stays
    
    reversion_data.append({
        'window_min': window,
        'revert_pct': revert_pct,
        'price_change': observed_price_change,
    })

# Create DataFrame
df = pd.DataFrame({
    'trade_size_pct': trade_size_pct,
    'market_vol': market_vol,
    'total_impact': total_impact_obs,
})

# Add post-trade price data
for rv in reversion_data:
    df[f'price_change_{rv["window_min"]}min'] = rv['price_change']

print(f"Trade data: {len(df)} trades\n")
print(f"Impact statistics:\n{df[['trade_size_pct', 'total_impact']].describe()}\n")

# Estimation Method 1: Linear regression on total impact
# Total = temp_base + temp_slope*size + perm_slope*size + vol_effect
# Try to disentangle by looking at reversion patterns

# Method 1: Use 1-min vs 30-min price changes
# Assumption: 1-min captures more temporary; 30-min is mostly permanent
price_1min = df['price_change_1min']
price_30min = df['price_change_30min']

# Regression: price_1min = α + β*size
slope_1min, intercept_1min, r2_1min, _, _ = linregress(df['trade_size_pct'], price_1min)

# Regression: price_30min = γ + δ*size
slope_30min, intercept_30min, r2_30min, _, _ = linregress(df['trade_size_pct'], price_30min)

print("ESTIMATION: Temporary vs Permanent (Regression on Size):")
print(f"1-min price change: Intercept={intercept_1min:.3f}, Slope={slope_1min:.3f}, R²={r2_1min:.3f}")
print(f"30-min price change: Intercept={intercept_30min:.3f}, Slope={slope_30min:.3f}, R²={r2_30min:.3f}")

# Estimate components (rough approximation)
est_temp_baseline = intercept_1min - intercept_30min  # What reverts in 1-30 min
est_temp_slope = slope_1min - slope_30min
est_perm_baseline = intercept_30min
est_perm_slope = slope_30min

print(f"\nEstimated Temporary Component:")
print(f"  Baseline: {est_temp_baseline:.3f} bps, Slope: {est_temp_slope:.3f} bps per % vol")
print(f"Estimated Permanent Component:")
print(f"  Baseline: {est_perm_baseline:.3f} bps, Slope: {est_perm_slope:.3f} bps per % vol")

print(f"\nTrue (Simulated) Parameters:")
print(f"  Temporary: {temp_baseline:.3f} + {temp_slope:.3f}*size")
print(f"  Permanent: {perm_baseline:.3f}*size + {perm_vol_sensitivity:.3f}*vol")

# Decomposition analysis
df['est_temp'] = est_temp_baseline + est_temp_slope * df['trade_size_pct']
df['est_perm'] = est_perm_baseline + est_perm_slope * df['trade_size_pct']
df['est_total'] = df['est_temp'] + df['est_perm']
df['perm_ratio'] = df['est_perm'] / df['est_total']

print(f"\nDecomposition of Total Impact:")
print(f"  Average temporary: {df['est_temp'].mean():.2f} bps ({df['est_temp'].mean()/df['est_total'].mean()*100:.0f}%)")
print(f"  Average permanent: {df['est_perm'].mean():.2f} bps ({df['est_perm'].mean()/df['est_total'].mean()*100:.0f}%)")
print(f"  Average total: {df['est_total'].mean():.2f} bps")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Total impact vs trade size with reversion windows
ax = axes[0, 0]
ax.scatter(df['trade_size_pct'], df['total_impact'], alpha=0.5, s=30, label='Total (t=0)', color='black')
ax.scatter(df['trade_size_pct'], df['price_change_1min'], alpha=0.5, s=30, label='1-min later', color='blue')
ax.scatter(df['trade_size_pct'], df['price_change_30min'], alpha=0.5, s=30, label='30-min later', color='green')
# Add trend lines
size_range = np.linspace(df['trade_size_pct'].min(), df['trade_size_pct'].max(), 100)
ax.plot(size_range, intercept_1min + slope_1min * size_range, '--', linewidth=2, color='blue')
ax.plot(size_range, intercept_30min + slope_30min * size_range, '--', linewidth=2, color='green')
ax.set_xlabel('Trade Size (% of daily volume)')
ax.set_ylabel('Price Change (bps)')
ax.set_title('Post-Trade Price Reversion Over Time')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Estimated temporary vs permanent by size
ax = axes[0, 1]
ax.scatter(df['trade_size_pct'], df['est_temp'], alpha=0.6, s=30, label='Temporary', color='orange')
ax.scatter(df['trade_size_pct'], df['est_perm'], alpha=0.6, s=30, label='Permanent', color='red')
ax.scatter(df['trade_size_pct'], df['est_total'], alpha=0.6, s=30, label='Total', color='black')
ax.set_xlabel('Trade Size (% of daily volume)')
ax.set_ylabel('Impact (bps)')
ax.set_title('Decomposed Impact Components')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Proportion permanent vs trade size
ax = axes[1, 0]
ax.scatter(df['trade_size_pct'], df['perm_ratio'] * 100, alpha=0.6, s=30, color='purple')
ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Trade Size (% of daily volume)')
ax.set_ylabel('Permanent Impact (% of total)')
ax.set_title('Permanent Impact Ratio vs Trade Size')
ax.grid(alpha=0.3)

# Plot 4: Reversion trajectory (average)
ax = axes[1, 1]
windows = [0, 1, 5, 30]
avg_prices = [df['total_impact'].mean()] + [df[f'price_change_{w}min'].mean() for w in [1, 5, 30]]
ax.plot(windows, avg_prices, 'o-', linewidth=2, markersize=8, color='steelblue')
# Add permanent component line (no further reversion after ~30 min)
perm_level = df['est_perm'].mean()
ax.axhline(perm_level, color='red', linestyle='--', linewidth=2, label=f'Estimated Permanent (~{perm_level:.2f} bps)')
ax.set_xlabel('Time Since Trade (minutes)')
ax.set_ylabel('Observed Price Impact (bps)')
ax.set_title('Price Reversion Trajectory (Average Trade)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n\nSUMMARY: How Much Impact is Recoverable?")
temp_fraction = df['est_temp'].mean() / df['est_total'].mean()
perm_fraction = 1 - temp_fraction
print(f"Temporary (recoverable via patience): {temp_fraction*100:.0f}%")
print(f"Permanent (unavoidable): {perm_fraction*100:.0f}%")
print(f"⟹ Optimal strategy: Reduce temporary via patient execution; accept permanent as cost")
```

**Key Insights:**
- Temporary impact typically 40–60% of total; decays in minutes
- Permanent impact 40–60% of total; reflects information content
- Patient execution (VWAP/TWAP) recovers mostly temporary; doesn't help permanent
- Larger trades → higher permanent component (signals information)
- Strategy: Optimize execution pace; accept permanent impact as unavoidable cost

## 6. Relationships & Dependencies
- **To Optimal Execution:** Decomposition informs risk-aversion parameter (λ) in Almgren-Chriss
- **To Algorithm Design:** Participation rate influenced by temporary/permanent split
- **To Implementation Shortfall:** IS benchmark includes both; decomposition aids performance analysis
- **To Cost Reduction:** Targeting temporary component (via patience) vs permanent (via venue/timing)

## References
- [Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=208282)
- [Kyle (1985) "Continuous Auctions and Insider Trading"](https://www.jstor.org/stable/1913210)
- [Hasbrouck (1991) "Measuring the Information Content of Stock Trades"](https://www.jstor.org/stable/2490519)

