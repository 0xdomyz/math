# Permanent Impact

## 1. Concept Skeleton
**Definition:** Long-term price change resulting from informed trading; information revelation; permanent adjustment of market prices  
**Purpose:** Quantify information incorporated into prices; assess fundamental value discovery; distinguish from temporary liquidity effects  
**Prerequisites:** Market microstructure, information asymmetry, price discovery, bid-ask spreads

## 2. Comparative Framing
| Impact Type | Duration | Cause | Recovery | Example |
|------------|----------|-------|----------|---------|
| **Permanent** | Long-term | Information | None (stays) | Insider buys → price up permanently |
| **Temporary** | Minutes | Liquidity | Reverts | Uninformed seller → bounces back |
| **Total** | Mixed | Info + liquidity | Partial | Market order → immediate + info decay |
| **Transient** | Microseconds | Order flow | Instant | Bid-ask bounce recovers in seconds |

## 3. Examples + Counterexamples

**Permanent Impact True:**  
Analyst upgrades Apple from $150 to $180 target → stock jumps $5 immediately → stays at elevated level for weeks → permanent price adjustment reflects new information

**Temporary Impact Misidentified as Permanent:**  
Large $50M uninformed sell order hits market → price drops $1 immediately → but rebounds within hours as inventory rebalances → thought permanent but was just temporary dislocation

**Mixed Impact:**  
$100M stock purchase by informed trader → price moves $2 immediately → $1.50 is permanent (new info) → $0.50 is temporary (demand/supply imbalance) → total $2 but only half sticks

**Information Revelation Failure:**  
CEO announces bankruptcy → stock drops 80% → days later, recovers 20% → suggests market over-shot → some of the drop was temporary → hard to disentangle until after fact

## 4. Layer Breakdown
```
Permanent Impact Framework:
├─ Conceptual Foundation:
│   ├─ Hasbrouck Decomposition:
│   │   - Total impact: Bid-ask + market impact
│   │   - Market impact components:
│   │     - Transient (temporary): Liquidity effect
│   │     - Permanent: Information effect
│   │   - Model: Price change = temporary + permanent + noise
│   │   - Empirical: ~60-70% permanent, 30-40% temporary (typical)
│   ├─ Information Theory Connection:
│   │   - Informed traders: Know more than market
│   │   - Trade: Execute when advantageous
│   │   - Result: Price moves toward their information
│   │   - Permanent: Market incorporates their info
│   │   - Inefficiency: Markets learning from informed trades
│   ├─ Kyle Model Intuition:
│   │   - Informed trader has edge
│   │   - Market maker sets spreads to compensate
│   │   - Spread widens if more information asymmetry
│   │   - Price moves gradually as MM updates beliefs
│   │   - Permanent impact: MM's belief change
│   └─ Efficient Markets Perspective:
│       - Semi-strong EMH: All public info reflected
│       - Private information: Creates opportunities
│       - Permanent impact: Price adjustment to private info
│       - Debate: How fast is incorporation? (seconds to hours)
│
├─ Measurement Techniques:
│   ├─ Vector Autoregression (VAR) Method:
│   │   - Price change model: Δp(t) = f(order_flow(t), lagged_prices)
│   │   - Impulse response: Shock of 1000 share buy
│   │   - Temporary: Response dies out (returns to baseline)
│   │   - Permanent: Response doesn't decay
│   │   - Estimation: Regress price changes on order flow
│   │   - Advantage: Can separate transient vs permanent
│   │   - Limitation: Requires high-frequency data
│   ├─ Run Test Method:
│   │   - Assumption: Order clustering indicates information
│   │   - Run: Sequence of same-side orders
│   │   - Longer run → more likely informed
│   │   - Permanent impact: Price change after run ends
│   │   - Temporary impact: Mean reversion of spread
│   │   - Advantage: Simple, intuitive
│   │   - Limitation: Doesn't account for order size
│   ├─ Partial Adjustment Method:
│   │   - Compare initial price move vs long-term adjustment
│   │   - Temporary: Initial price reverts partway
│   │   - Permanent: No reversion
│   │   - Model: p_final = p_initial + α × (p_moved - p_initial)
│   │   - α = 1: All permanent; α = 0: All temporary
│   │   - Typical: α = 0.6-0.8 (60-80% permanent)
│   ├─ Information Content Method:
│   │   - Realized volatility: How much price moves
│   │   - Order flow: Buy vs sell imbalance
│   │   - Correlation: Persistent price changes = information
│   │   - Non-persistent: Temporary liquidity effect
│   │   - PIN measure: Probability of informed trading
│   └─ VPIN Method (Easley et al):
│       - Volume-Synchronized PIN
│       - High VPIN: More informed trading likely
│       - Correlates with adverse selection costs
│       - Advance: Works in real-time
│       - Limitation: Requires high-frequency data
│
├─ Empirical Evidence:
│   ├─ Stock Market Studies:
│   │   - Hasbrouck (1991): 40-80% permanent (varies by stock)
│   │   - Large cap: ~70-80% permanent (efficient market)
│   │   - Small cap: ~40-60% permanent (less efficient)
│   │   - Bid-ask spread: $0.10 → $0.07 permanent, $0.03 temporary
│   │   - Finding: Liquidity effects die out in minutes
│   │   - Permanent: Settles over hours/days
│   ├─ Cross-Sectional Patterns:
│   │   - Volume: Highly liquid stocks → higher permanent impact
│   │   - Volatility: More volatile → higher permanent
│   │   - Firm size: Larger firms → higher permanent %
│   │   - Bid-ask spread: Wider spread → higher permanent
│   │   - Intuition: Efficiency correlates with information content
│   ├─ Market Regimes:
│   │   - Calm markets: ~75% permanent, ~25% temporary
│   │   - Volatile periods: ~70% permanent, ~30% temporary
│   │   - Crisis: ~60% permanent, ~40% temporary (liquidity dries)
│   │   - Overnight gaps: ~90% permanent (no trading to revert)
│   ├─ Asset Classes:
│   │   - Equities: 70-80% permanent
│   │   - Options: 60-70% permanent (more temporary effects)
│   │   - Futures: 80-90% permanent (information-driven)
│   │   - Currencies: 60-70% permanent (flows + rates)
│   │   - Commodities: 50-60% permanent (supply/demand shocks)
│   └─ Time of Day:
│       - Pre-open: Highest permanent impact (overnight accumulation)
│       - Morning: High permanent impact (news revelation)
│       - Midday: Moderate permanent impact
│       - Close: High permanent impact (portfolio adjustments)
│
├─ Theoretical Models:
│   ├─ Kyle (1985) Model:
│   │   - Single informed trader + uninformed flow
│   │   - Equilibrium: Market maker sets prices
│   │   - Permanent impact: Fully from informed trading
│   │   - Temporary impact: From MM inventory adjustment
│   │   - Prediction: All impact permanent in this model
│   │   - Limitation: Only one asset, two types of traders
│   ├─ Glosten-Milgrom (1985):
│   │   - Sequential trades with uncertain information
│   │   - Spreads widen as more sells (could be informed)
│   │   - Permanent: Belief updating by MM
│   │   - Model incorporates asymmetric information decay
│   │   - Empirical: Matches 60-70% permanent observations
│   ├─ Madhavan et al (1997):
│   │   - Three components: Adverse selection + inventory + order processing
│   │   - Decomposition: Can estimate each effect
│   │   - Permanent: Adverse selection component
│   │   - Temporary: Inventory + order processing components
│   │   - Method: Regression-based estimation
│   ├─ Almgren-Chriss (2000):
│   │   - Optimal execution framework
│   │   - Assumes linear permanent impact (√ volume)
│   │   - Temporary: Bid-ask bounce, execution risk
│   │   - Permanent: Price discovery from execution
│   │   - Application: Execution algorithm design
│   └─ Bouchaud et al (2004):
│       - Empirical market impact scaling laws
│       - Permanent impact ∝ √(Volume)
│       - Decay: Exponential + power-law components
│       - Temporary: Decays in 10-100 milliseconds
│       - Permanent: Slow decay over hours
│
├─ Practical Applications:
│   ├─ Execution Algorithm Design:
│   │   - VWAP/TWAP minimizes temporary costs
│   │   - But permanent unavoidable (information revelation)
│   │   - Timing: Execute when spreads tight (temporary ↓)
│   │   - Sequencing: Split into smaller chunks
│   │   - Cost: Permanent impact proportional to √(volume)
│   ├─ Block Trading:
│   │   - Large orders: Mostly permanent impact
│   │   - Negotiated price: Avoid permanent discovery
│   │   - Timing: Execute during active volume
│   │   - Size: Larger blocks → larger permanent impact
│   │   - Economics: Justify block premium
│   ├─ Risk Management:
│   │   - Position sizing: Account for permanent impact
│   │   - Liquidity: More liquid → lower permanent cost
│   │   - Slicing: Break into pieces to reduce impact
│   │   - Timing: Spread over time to let market adjust
│   ├─ Performance Attribution:
│   │   - Benchmark vs VWAP: Separates temporary/permanent
│   │   - Contribution: Permanent = strategy skill, Temporary = execution
│   │   - Analysis: Where did performance come from?
│   │   - Improvement: Target permanent (strategy), reduce temporary (execution)
│   ├─ Derivative Pricing:
│   │   - Option value: Larger orders cause permanent moves
│   │   - Adjustments: Add permanent impact to model
│   │   - Hedging: Account for execution costs
│   │   - Greeks: Delta/gamma affected by impact
│   └─ Regulatory Compliance:
│       - Best execution: Must minimize permanent impact
│       - Documentation: Show understanding of costs
│       - Venue selection: Optimize for liquidity
│       - Benchmarks: Compare to market alternatives
│
├─ Market Impact Decomposition (Stoll 1989):
│   ├─ Adverse Selection Component:
│   │   - Definition: Cost of trading with informed counterparties
│   │   - Permanent: Fully permanent (market learns)
│   │   - Magnitude: ~40-60% of bid-ask spread typically
│   │   - Drivers: Information asymmetry, uncertainty
│   │   - PIN measure: High PIN → high adverse selection
│   ├─ Inventory Component:
│   │   - Definition: MM risk from holding positions
│   │   - Temporary: Fully temporary (MM rebalances)
│   │   - Magnitude: ~20-40% of spread
│   │   - Drivers: Risk aversion, holding period
│   │   - Recovery: MM adjusts back to target inventory
│   ├─ Order Processing Component:
│   │   - Definition: Fixed costs of executing order
│   │   - Temporary: Fully temporary (fixed cost)
│   │   - Magnitude: ~5-15% of spread
│   │   - Drivers: Technology, operations
│   │   - Recovery: Immediate (cost already paid)
│   └─ Empirical Decomposition Example:
│       - Total spread: $0.10
│       - Adverse selection: $0.060 (60%)
│       - Inventory: $0.025 (25%)
│       - Order processing: $0.015 (15%)
│       - Permanent impact: $0.060 (60% of total)
│
├─ Advanced Topics:
│   ├─ Nonlinear Permanent Impact:
│   │   - Assumption: Impact linear in order size
│   │   - Reality: May be nonlinear at extremes
│   │   - Large orders: Disproportionate impact (convexity)
│   │   - Reason: Market depth finite, liquidity evaporates
│   │   - Model: Impact ∝ Volume^α, α > 1 (possible)
│   ├─ Asymmetric Impact:
│   │   - Buy vs sell: May have different permanent impacts
│   │   - Reason: Market sentiment, inventory dynamics
│   │   - Empirical: Often buy impact < sell impact (same size)
│   │   - Explanation: Dealers short-biased (prefer buyers)
│   ├─ Cross-Asset Impact:
│   │   - Correlated assets: Price move affects others
│   │   - Mechanism: Market makers hedge across assets
│   │   - Effect: Permanent impact extends to related assets
│   │   - Example: S&P 500 large buy → all component stocks move
│   ├─ Seasonal Patterns:
│   │   - Time of day: Morning → higher permanent
│   │   - Day of week: Monday → higher permanent (weekend news)
│   │   - Earnings: Permanent impact spike at announcements
│   │   - Rebalancing: Month-end → increased permanent impact
│   └─ Temporal Decay:
│       - Not instant: Price adjusts over hours
│       - Mechanism: MM learning, cascade of informed trading
│       - Empirical: 50% permanent by 10 seconds, 90% by 1 hour
│       - Implication: Execution over time reduces permanent reveal
│
└─ Challenges and Debates:
    ├─ Causality Question:
    │   - Does order flow cause price move or vice versa?
    │   - Endogeneity: Informed traders trade on predicted moves
    │   - Instrument: Need exogenous order flow
    │   - Challenge: Hard to find true exogeneity
    ├─ Time Scale Dependency:
    │   - Permanent over what horizon? (minutes, hours, days?)
    │   - Definition matters: May be permanent at 1hr, temporary at 1yr
    │   - Signal decay: All information eventually incorporated
    │   - Question: What's the right time scale?
    ├─ Microstructure vs Macroeconomics:
    │   - Microstructure effect: Market structure impacts
    │   - Macro shock: Fundamental value actually changed
    │   - Confounding: Hard to distinguish ex-post
    │   - Approach: Use order flow direction as proxy
    ├─ Model Assumptions:
    │   - Linearity: Assume impact linear (may not be)
    │   - Stationarity: Assume impact constant (regime-dependent)
    │   - Exogeneity: Assume order flow exogenous (it's not)
    │   - Reality: Real markets violate all assumptions
    └─ Measurement Error:
        - Bid-ask data: Often delayed or misreported
        - High-frequency: Noise dominates (TAQ data issues)
        - Definition: Trades inside/outside spread? (matters!)
        - Solution: Use best bid-ask only, filter data
```

**Interaction:** Informed trader buys 100K shares → price moves $0.10 → MM updates belief of fundamental value → price stays $0.05-0.10 higher → permanent adjustment reflects new information

## 5. Mini-Project
Simulate permanent vs temporary impact decomposition:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

np.random.seed(42)

class PermanentTemporaryDecomposition:
    def __init__(self):
        self.trades = []
        self.prices = []
        self.impact_decomposition = []
        
    def simulate_trade_sequence(self, order_sizes, num_days=20):
        """Simulate price evolution with permanent and temporary components"""
        price = 100.0
        prices_over_time = [price]
        permanent_levels = [0.0]
        
        for day in range(num_days):
            for order_size in order_sizes:
                # Permanent impact: Proportional to √(order size)
                permanent_component = 0.0005 * np.sqrt(order_size / 10000)
                
                # Temporary impact: Proportional to order size directly, mean-reverts
                temporary_component = 0.001 * (order_size / 100000)
                
                # Noise (random trade not affecting price long-term)
                noise = np.random.normal(0, 0.0005)
                
                # Total price impact
                total_impact = permanent_component + temporary_component + noise
                price += total_impact
                
                # Permanent shifts the baseline
                permanent_level = permanent_levels[-1] + permanent_component
                permanent_levels.append(permanent_level)
                
                # Temporary reverts partially each period
                temporary_reversion = -temporary_component * 0.3
                price += temporary_reversion
                
                prices_over_time.append(price)
                
                # Store for analysis
                self.trades.append({
                    'order_size': order_size,
                    'permanent': permanent_component,
                    'temporary': temporary_component,
                    'total': total_impact,
                    'price': price
                })
        
        return prices_over_time, permanent_levels
    
    def decompose_impact_vp(self, prices, order_flow, lags=20):
        """Vector autoregression decomposition of impacts"""
        # Price changes
        price_changes = np.diff(prices)
        
        # Order flow (1 = buy, -1 = sell)
        order_flow_array = np.array(order_flow)
        
        # Build lagged variables
        y = price_changes[lags:]
        X = np.ones((len(y), 1))
        
        for lag in range(1, lags + 1):
            X = np.column_stack([X, order_flow_array[lags - lag:-lag if lag < len(order_flow_array) else None]])
        
        # Regression
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        # Temporary: Sum of all coefficients (one-period effect)
        temporary = coeffs[1]
        
        # Permanent: Cumulative effect (sum all future periods)
        # Approximate by ratio of coefficient to variance
        permanent = np.sum(coeffs[1:]) / len(coeffs[1:])
        
        return temporary, permanent
    
    def partial_adjustment_method(self, initial_price, final_price, prior_price):
        """Estimate permanent fraction using partial adjustment"""
        # α = (p_t - p_{t-1}) / (p* - p_{t-1})
        # α = 1: all permanent, α = 0: all temporary
        
        adjustment = initial_price - prior_price
        total_change = final_price - prior_price
        
        if adjustment != 0:
            alpha = adjustment / total_change if total_change != 0 else 1.0
        else:
            alpha = 0.0
        
        permanent_fraction = alpha
        temporary_fraction = 1.0 - alpha
        
        return permanent_fraction, temporary_fraction

# Scenario 1: Varying order sizes and their impact decomposition
print("Scenario 1: Impact Decomposition by Order Size")
print("=" * 80)

sim = PermanentTemporaryDecomposition()
order_sizes = [10000, 50000, 100000, 250000, 500000]
permanent_impacts = []
temporary_impacts = []
total_impacts = []
permanent_fractions = []

for order_size in order_sizes:
    permanent = 0.0005 * np.sqrt(order_size / 10000)
    temporary = 0.001 * (order_size / 100000)
    total = permanent + temporary
    
    permanent_impacts.append(permanent * 10000)  # Convert to cents
    temporary_impacts.append(temporary * 10000)
    total_impacts.append(total * 10000)
    permanent_fractions.append(permanent / total if total > 0 else 0)
    
    print(f"Order Size: {order_size:>10,} shares")
    print(f"  Permanent Impact: {permanent*10000:>8.2f} cents ({permanent/total*100:>5.1f}%)")
    print(f"  Temporary Impact: {temporary*10000:>8.2f} cents ({temporary/total*100:>5.1f}%)")
    print(f"  Total Impact:     {total*10000:>8.2f} cents")
    print()

# Scenario 2: Price evolution with permanent and temporary components
print("Scenario 2: Price Evolution (20 days, random orders)")
print("=" * 80)

order_sequence = np.random.choice([10000, 50000, 100000], size=20)
prices, permanent_levels = sim.simulate_trade_sequence(order_sequence, num_days=1)

print(f"Initial Price: ${prices[0]:.2f}")
print(f"Final Price:   ${prices[-1]:.2f}")
print(f"Total Change:  ${prices[-1] - prices[0]:.2f}")
print(f"Permanent Drift: ${permanent_levels[-1]:.4f}")
print(f"\nMean Reversion: {(prices[-1] - permanent_levels[-1]):.4f}")

# Scenario 3: Simulating mean reversion (temporary component decay)
print(f"\n\nScenario 3: Temporary Component Mean Reversion")
print("=" * 80)

time_periods = np.arange(0, 100)
permanent_base = 0.05
temporary_initial = 0.15
reversion_rate = 0.95  # Each period, 95% of temporary remains

temporary_over_time = []
cumulative_price = permanent_base

for t in time_periods:
    temporary = temporary_initial * (reversion_rate ** t)
    cumulative_price += temporary * (1 - reversion_rate)
    temporary_over_time.append(temporary)

print(f"Initial Temporary Impact: {temporary_initial:.4f}")
print(f"After 1 second:           {temporary_over_time[1]:.4f}")
print(f"After 10 seconds:         {temporary_over_time[10]:.4f}")
print(f"After 100 periods:        {temporary_over_time[-1]:.6f}")
print(f"Half-life (periods):      {-np.log(0.5) / np.log(reversion_rate):.1f}")

# Scenario 4: Stoll decomposition example
print(f"\n\nScenario 4: Stoll Decomposition (Spread Components)")
print("=" * 80)

bid_ask_spread = 0.10  # $0.10 spread

# Typical decomposition
adverse_selection_pct = 0.60
inventory_pct = 0.25
order_processing_pct = 0.15

adverse_selection = bid_ask_spread * adverse_selection_pct
inventory = bid_ask_spread * inventory_pct
order_processing = bid_ask_spread * order_processing_pct

print(f"Total Bid-Ask Spread:   ${bid_ask_spread:.4f}")
print(f"  Adverse Selection:    ${adverse_selection:.4f} ({adverse_selection_pct*100:.0f}%) - PERMANENT")
print(f"  Inventory Cost:       ${inventory:.4f} ({inventory_pct*100:.0f}%) - TEMPORARY")
print(f"  Order Processing:     ${order_processing:.4f} ({order_processing_pct*100:.0f}%) - TEMPORARY")
print(f"\nPermanent Component:    ${adverse_selection:.4f} ({adverse_selection_pct*100:.0f}%)")
print(f"Temporary Component:    ${inventory + order_processing:.4f} ({(inventory_pct + order_processing_pct)*100:.0f}%)")

# Scenario 5: Cross-sectional comparison
print(f"\n\nScenario 5: Permanent Impact Across Asset Types")
print("=" * 80)

asset_types = [
    {'name': 'Large Cap Stock', 'permanent_pct': 0.75, 'sample_size': 50},
    {'name': 'Small Cap Stock', 'permanent_pct': 0.55, 'sample_size': 50},
    {'name': 'ETF', 'permanent_pct': 0.80, 'sample_size': 50},
    {'name': 'Option', 'permanent_pct': 0.65, 'sample_size': 50},
    {'name': 'Futures', 'permanent_pct': 0.85, 'sample_size': 50},
]

for asset in asset_types:
    print(f"{asset['name']:>20}: {asset['permanent_pct']*100:>5.1f}% permanent impact")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Impact decomposition by order size
axes[0, 0].bar(np.arange(len(order_sizes)), permanent_impacts, label='Permanent', alpha=0.7)
axes[0, 0].bar(np.arange(len(order_sizes)), temporary_impacts, bottom=permanent_impacts, label='Temporary', alpha=0.7)
axes[0, 0].set_xticks(np.arange(len(order_sizes)))
axes[0, 0].set_xticklabels([f'{s/1000:.0f}K' for s in order_sizes])
axes[0, 0].set_ylabel('Price Impact (cents)')
axes[0, 0].set_title('Scenario 1: Permanent vs Temporary Components')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Permanent fraction by order size
axes[0, 1].plot(order_sizes, np.array(permanent_fractions)*100, 'o-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Order Size (shares)')
axes[0, 1].set_ylabel('Permanent Impact (%)')
axes[0, 1].set_title('Scenario 1: Permanent Fraction of Total Impact')
axes[0, 1].set_xscale('log')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].axhline(y=70, color='r', linestyle='--', label='Typical (70%)', alpha=0.5)
axes[0, 1].legend()

# Plot 3: Mean reversion of temporary component
axes[1, 0].semilogy(time_periods, temporary_over_time, linewidth=2)
axes[1, 0].axhline(y=temporary_initial * 0.5, color='r', linestyle='--', label='Half-Life', alpha=0.5)
axes[1, 0].set_xlabel('Time Periods')
axes[1, 0].set_ylabel('Temporary Impact (log scale)')
axes[1, 0].set_title('Scenario 3: Mean Reversion of Temporary Component')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Asset type comparison
asset_names = [a['name'] for a in asset_types]
permanent_pcts = [a['permanent_pct']*100 for a in asset_types]
colors_assets = plt.cm.viridis(np.linspace(0, 1, len(asset_names)))

bars = axes[1, 1].barh(asset_names, permanent_pcts, color=colors_assets)
axes[1, 1].set_xlabel('Permanent Impact (%)')
axes[1, 1].set_title('Scenario 5: Permanent Impact by Asset Type')
axes[1, 1].set_xlim([0, 100])
axes[1, 1].grid(alpha=0.3, axis='x')

for bar, pct in zip(bars, permanent_pcts):
    width = bar.get_width()
    axes[1, 1].text(width + 1, bar.get_y() + bar.get_height()/2.,
                   f'{pct:.0f}%', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Average Permanent Impact: {np.mean(permanent_fractions)*100:.1f}%")
print(f"Range: {np.min(permanent_fractions)*100:.1f}% - {np.max(permanent_fractions)*100:.1f}%")
print(f"Typical decomposition: 70% permanent, 30% temporary")
print(f"Recovery time scale: Minutes to hours (temporary)")
print(f"Permanent: Persists indefinitely (no reversion)")
```

## 6. Challenge Round
If permanent impact represents true information discovery (beneficial market efficiency), why do regulators restrict certain trading practices that would speed up permanent price adjustment?

- **Fairness concern**: Speed-based advantages create winners/losers regardless of information quality → creates distributional inequality → retail loses to HFT even without information advantage
- **Cascade risk**: Rapid price discovery can trigger algorithmic cascades → flash crashes → systemic instability → faster isn't always better if destabilizing
- **Market quality**: Permanent impact during stress can be destabilizing → forced liquidations → permanent moves in wrong direction → not efficient just because permanent
- **Manipulation**: Not all permanent impact is information → spoofing creates artificial permanent moves → hard to distinguish manipulation from information until after
- **Stability vs efficiency**: Trade-off exists → slightly slower price discovery accepted if prevents cascades → regulators optimize for stability over efficiency (risk-averse choice)

## 7. Key References
- [Hasbrouck (1991) - Measuring the Effects of Data Aggregation on Price Discovery](https://www.jstor.org/stable/2328955)
- [Madhavan et al (1997) - Why do Security Prices Change?](https://www.jstor.org/stable/2329541)
- [Glosten & Milgrom (1985) - Bid, Ask and Transaction Prices](https://www.sciencedirect.com/science/article/abs/pii/0304405X85900443)
- [Bouchaud et al (2004) - Empirical Properties of Asset Returns: Stylized Facts and Universal Measures](https://arxiv.org/abs/cond-mat/0406224)

---
**Status:** Information-driven long-term price adjustment | **Complements:** Temporary Impact, Price Discovery, Market Efficiency, Information Asymmetry
