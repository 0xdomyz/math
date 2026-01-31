# Dark Pools

## 1. Concept Skeleton
**Definition:** Private off-exchange trading venues with minimal pre-trade transparency; price discovery from reference exchanges; used for block trading  
**Purpose:** Execute large orders without moving market; reduce information leakage; negotiate better prices for institutional traders  
**Prerequisites:** Order book mechanics, block trading, information asymmetry, market impact

## 2. Comparative Framing
| Venue Type | Pre-Trade Transparency | Post-Trade Transparency | Order Size | Venue |
|-----------|----------------------|------------------------|-----------|-------|
| **Dark Pool** | None (hidden) | Delayed (T+5 min typical) | Large blocks | Private |
| **Public Exchange** | Full (NBBO) | Immediate (tape) | Any size | Central |
| **Lit Exchange** | Full book | Immediate (tape) | Any size | Central |
| **ATS (ECN)** | Partial (top level) | Immediate | Any size | Private regulated |

## 3. Examples + Counterexamples

**Dark Pool Success:**  
Pension fund needs to buy 1M shares of Apple without moving market → routes 100K at a time through multiple dark pools → average price $180.00 vs VWAP $180.15 → saves $150K

**Dark Pool Failure:**  
Retail trader places market order in dark pool → no liquidity (only big institutions) → order rejected or fills at terrible price → better off on public exchange

**Information Leakage:**  
Large buy order on public exchange → everyone sees demand → price moves $0.50 before full order fills → permanent market impact → expensive execution

**Regulatory Crackdown:**  
SEC discovered dark pool operators providing trading info to select clients → unfair advantage → fined operators → new transparency rules → dark pools lost advantages

## 4. Layer Breakdown
```
Dark Pool Framework:
├─ Structure & Operations:
│   ├─ Private Nature:
│   │   - Not exchange (regulated differently)
│   │   - Private company operates
│   │   - Membership required: Institutional investors
│   │   - Limited transparency: Not public order book
│   ├─ Order Book (Hidden):
│   │   - Orders kept private: Not shown to market
│   │   - Matching algorithm: Same as exchanges typically
│   │   - Price discovery: Based on reference exchange (e.g., NBBO)
│   │   - Example: Buy order at NBBO -$0.01, sell at NBBO +$0.01
│   ├─ Pricing Mechanism:
│   │   - NBBO Reference: Orders reference National Best Bid Offer
│   │   - Advantages: Standard pricing vs negotiated
│   │   - Pegging: Orders peg to reference exchange prices
│   │   - Discretion: Brokers may improve standard peg
│   ├─ Execution Priority:
│   │   - Pro-rata: Share size among all orders
│   │   - FIFO: First in first out (most common)
│   │   - Price-time: Best price, then time priority
│   │   - Varies by pool: Each pool has own rules
│   └─ Settlement:
│       - T+2: Typical equity settlement
│       - Bilateral: Only parties involved know trade
│       - Delayed reporting: Post-trade data delayed 5+ minutes
│
├─ Market Impact Reduction:
│   ├─ Information Concealment:
│   │   - Demand hidden: Market doesn't see 1M share order
│   │   - Price stability: No advance price move
│   │   - Stealth execution: Execute gradually without detection
│   │   - Example: Execute 500K in 10 parcels through dark pools
│   ├─ Slicing Strategy:
│   │   - Break large order into smaller pieces
│   │   - Route pieces through multiple venues
│   │   - Dark pools + public exchanges
│   │   - Timing: Spread over hours or days
│   │   - Example: 1M order → 100K × 10 pieces, different times
│   ├─ Impact Measurement:
│   │   - VWAP (Volume-Weighted Average Price): Compare to volume average
│   │   - TWAP (Time-Weighted Average Price): Compare to time average
│   │   - Arrival Price: Midpoint when order submitted
│   │   - Empirical: Dark pool execution saves 1-5 bps on large orders
│   ├─ Block Trade Premium:
│   │   - Negotiated price: Brokers negotiate better than posted
│   │   - Off-exchange: Can execute below public spread
│   │   - Size discount: Larger blocks get better prices
│   │   - Relationship: Long-term clients get better treatment
│   └─ Cost Savings:
│       - vs public exchange impact: 1-3% cost savings
│       - vs announced large order: 5-10% savings
│       - vs market order slippage: 0.5-2% savings
│       - Institutional value: Huge for multi-billion $ managers
│
├─ Dark Pool Economics:
│   ├─ Participant Composition:
│   │   - Asset managers: 40%+ of flow
│   │   - Hedge funds: 30%+ of flow
│   │   - Brokers: 15%+ of flow
│   │   - Retail: <5% (most excluded)
│   ├─ Revenue Model:
│   │   - Subscription fees: Participants pay membership
│   │   - Transaction fees: Per-share fee for trades
│   │   - Volume rebates: Incentivize high-volume members
│   │   - Premium: Charge for better pricing/access
│   ├─ Profitability:
│   │   - High margins: $10-100M revenue per pool annually
│   │   - Low costs: Minimal infrastructure vs exchange
│   │   - Scale benefits: Larger pools more profitable
│   │   - Competition: Prices driven down over time
│   ├─ Broker Conflict of Interest:
│   │   - Broker owns dark pool: Revenue from order flow
│   │   - Incentive: Route orders to own pool (not best price)
│   │   - Conflict: Customer interest vs broker interest
│   │   - Regulation: Requires best execution despite conflict
│   └─ Market Microstructure Effects:
│       - Price discovery degradation: Less public information
│       - Volatility changes: Different liquidity patterns
│       - Bid-ask spread: May widen on public exchange (less volume)
│       - Empirical: Dark pool growth associated with wider spreads
│
├─ Liquidity Types in Dark Pools:
│   ├─ Passive Liquidity:
│   │   - Institutional orders waiting
│   │   - Algos trying to accumulate
│   │   - Buy-side orders resting
│   │   - Lower execution probability but passive
│   ├─ HFT Liquidity:
│   │   - HFT provides tight spreads
│   │   - Quote entire order book
│   │   - Profits from spread and prediction
│   │   - Controversial: Predatory or beneficial?
│   ├─ Broker Proprietary:
│   │   - Broker trades own account
│   │   - Can improve prices vs NBBO
│   │   - Can disadvantage customers
│   │   - Regulatory scrutiny: Conflicts of interest
│   └─ Liquidity Pools:
│       - Multiple small pools aggregate
│       - Each pool illiquid alone
│       - Combined create meaningful liquidity
│       - Aggregation challenges: Must check each pool
│
├─ Dark Pool Types:
│   ├─ Agency Models:
│   │   - Broker acts as agent only
│   │   - Matches customer orders
│   │   - No principal trading (broker doesn't trade own account)
│   │   - Example: Goldman Sachs Sigma X
│   ├─ Principal Models:
│   │   - Broker trades own account
│   │   - Broker principal provides liquidity
│   │   - Lower spreads but conflicts of interest
│   │   - Example: Liquidnet (now Instinet)
│   ├─ Hybrid Models:
│   │   - Both agency and principal trading
│   │   - Flexibility but more conflicts
│   │   - Most common model
│   │   - Example: Credit Suisse PTIX
│   └─ Consortium Models:
│       - Multiple brokers jointly operate
│       - Shared infrastructure
│       - Reduced conflicts (not single broker)
│       - Example: POSIT (Defunct, acquired)
│
├─ Trading Strategy Effects:
│   ├─ Momentum Strategies:
│   │   - Less visible orders
│   │   - Predatory algos harder to detect
│   │   - Better execution for momentum traders
│   │   - Increases profitability of strategies
│   ├─ Insider Trading:
│   │   - Dark pools used to hide inside info trades
│   │   - No pre-trade transparency
│   │   - Hard to detect manipulation
│   │   - Regulatory concern: Abuse potential
│   ├─ Predatory Trading:
│   │   - Front-running opportunities
│   │   - HFTs see orders in pools, trade ahead
│   │   - Less visibility makes prediction easier
│   │   - Controversial practice
│   └─ Information Leakage:
│       - Post-trade data reveals intent
│       - Brokers can infer buy/sell side
│       - Info sold to others
│       - Creates information asymmetry
│
├─ Regulatory Environment:
│   ├─ Regulation SHO (SEC):
│   │   - Applies to dark pools
│   │   - Tick-size rules: Prices in cents/eighths
│   │   - Locate requirement: Short-selling restrictions
│   │   - Fails to deliver: Naked short-selling restrictions
│   ├─ ATS Regulation (Reg ATS):
│   │   - Alternative Trading System rules
│   │   - Transparency requirements: Trade reporting
│   │   - Fair access: Can't discriminate
│   │   - Conflict of interest: Disclosure required
│   ├─ Pre-Trade Transparency:
│   │   - Not required for dark pools (by design)
│   │   - Orders don't have to be visible
│   │   - Exception: Orders within $0.0625 of NBBO
│   │   - Debate: Should require transparency?
│   ├─ Post-Trade Reporting:
│   │   - Trades must be reported (FINRA ADF)
│   │   - Timing: 10-15 second reporting
│   │   - Delayed: Not immediate (vs public exchanges)
│   │   - Block trades: May be reported later
│   ├─ Enforcement Actions:
│   │   - SEC fined Barclays (LXE) for misrepresentation
│   │   - Fined brokers for best execution violations
│   │   - Crackdowns on conflicts of interest
│   │   - Recent trend: Stricter scrutiny
│   └─ International:
│       - MiFID (EU): Stricter dark pool rules
│       - Transparency required above threshold
│       - Reporting: EMIR & other frameworks
│       - Trend: Less tolerance for dark trading
│
├─ Market Impact and Concerns:
│   ├─ Fragmentation:
│   │   - Liquidity split across venues
│   │   - Harder to find best price (must check all)
│   │   - Costs for HFTs (colocation fees, connectivity)
│   │   - Benefit for agencies (execution venues)
│   ├─ Price Discovery:
│   │   - Less visible order book
│   │   - Prices lag behind fundamental
│   │   - Information incorporation slower
│   │   - Empirical: Slight negative effect
│   ├─ Market Quality:
│   │   - Spreads on public exchanges widen
│   │   - Volume migrates to dark pools
│   │   - Retail traders hurt (worse spreads)
│   │   - Institutional traders benefit (better prices)
│   ├─ Systemic Risk:
│   │   - Hidden positions accumulate
│   │   - Crash can be sharper (no visibility)
│   │   - Circuit breakers insufficient
│   │   - Example: Flash crash, positions hidden in pools
│   └─ Fairness:
│       - Retail excluded: Only institutional access
│       - Better prices for insiders
│       - Information asymmetry: Brokers know more
│       - Democratic concern: Not all investors benefit
│
├─ Market Data and Intelligence:
│   ├─ Post-Trade Data Advantage:
│   │   - See where trades occurred
│   │   - Infer market sentiment
│   │   - HFTs parse data in microseconds
│   │   - Advantage: First to know trades
│   ├─ Information Selling:
│   │   - Dark pool operators sell anonymized data
│   │   - Buyers: HFTs, market makers, researchers
│   │   - Timing: Data sold with slight delay
│   │   - Revenue: Major pool revenue stream
│   ├─ Broker Information:
│   │   - Brokers know customer orders
│   │   - Can trade ahead (front-running)
│   │   - Can route to favored venues
│   │   - Regulatory burden: Must disclose conflicts
│   └─ Competitive Advantage:
│       - Information edge: Worth millions
│       - HFT profits depend on this
│       - Regulatory concern: Fairness issues
│       - Arms race: Faster data access
│
├─ Evolution and Trends:
│   ├─ Growth:
│   │   - Pre-2010: <5% of volume
│   │   - 2015: ~15-20% of volume
│   │   - 2020: ~20-25% of volume
│   │   - Trend: Plateauing as regulatory pressure increases
│   ├─ Consolidation:
│   │   - Many small pools merged
│   │   - Large pools grow (economies of scale)
│   │   - Oligopoly forming: Few dominant pools
│   │   - Trend: Further consolidation likely
│   ├─ Technology:
│   │   - Blockchain alternative trading systems emerging
│   │   - Smart contracts for matching
│   │   - Decentralized exchanges competing
│   │   - Trend: More competition from crypto venues
│   ├─ Regulatory Response:
│   │   - MiFID II: Stricter transparency rules
│   │   - Ban on certain strategies
│   │   - Position limits: Caps on exposures
│   │   - Trend: Global push for transparency
│   └─ Future Outlook:
│       - May decline if transparency required
│       - Or evolve to accept transparency
│       - Technology enabling better transparency
│       - Likely: Hybrid transparency pools emerge
│
└─ Strategic Use Cases:
    ├─ Block Trading:
    │   - Institutional trade: 100K+ shares
    │   - Negotiated pricing below NBBO
    │   - Brokers facilitate
    │   - Example: Pension fund portfolio rebalancing
    ├─ Liquidity Seeking:
    │   - Find contra-side interest
    │   - Passive execution
    │   - Low market impact
    │   - Example: Buy-side fund liquidating
    ├─ Conditional Algorithms:
    │   - Execute only if price levels
    │   - Avoid market detection
    │   - Minimize footprint
    │   - Example: "Don't show orders unless near VWAP"
    └─ Benchmark Trading:
        - Execute to VWAP/TWAP without leakage
        - Performance attribution easier
        - Transparent to clients
        - Example: "We beat VWAP by 2 bps"
```

**Interaction:** Large order submitted privately → matched against hidden contra-side → price based on NBBO → settlement occurs → post-trade data delayed

## 5. Mini-Project
Simulate dark pool execution vs public exchange market impact:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class MarketImpactSimulator:
    def __init__(self):
        self.trades = []
        self.prices_public = []
        self.prices_dark = []
        self.impact_public = []
        self.impact_dark = []
        
    def execute_public_exchange(self, order_size, nbbo=100.0):
        """Execute large order on public exchange (visible, causes impact)"""
        initial_price = nbbo
        
        # Market impact model: ΔP = λ * sqrt(Q/V)
        # Larger orders cause larger price moves
        price_impact = 0.001 * np.sqrt(order_size / 100000)
        
        # Price moves against order
        execution_price = initial_price + price_impact
        
        # Permanent impact: Price moves and stays
        final_price = initial_price + price_impact * 0.8  # 80% permanent
        
        # Temporary impact: Price rebounds slightly
        rebound_price = initial_price + price_impact * 0.3
        
        self.prices_public.append(execution_price)
        self.impact_public.append(execution_price - initial_price)
        
        return {
            'execution': execution_price,
            'final': final_price,
            'rebound': rebound_price,
            'total_impact': execution_price - initial_price,
            'permanent_impact': final_price - initial_price
        }
    
    def execute_dark_pool(self, order_size, nbbo=100.0):
        """Execute through dark pool (hidden, minimal impact)"""
        # Dark pool trades at NBBO or slight improvement
        # No market impact because hidden
        
        execution_price = nbbo - 0.001  # Slight improvement from NBBO
        final_price = nbbo  # No permanent impact
        
        # Small chance order doesn't fill (illiquid pool)
        fill_probability = max(0.3, 1 - order_size / 500000)
        
        if np.random.random() > fill_probability:
            # Partial fill or rejection
            filled_size = order_size * fill_probability
            # Must execute remainder on public exchange (worse)
            remainder_impact = self.execute_public_exchange(order_size - filled_size, nbbo)
            execution_price = (execution_price * filled_size + remainder_impact['execution'] * (order_size - filled_size)) / order_size
        
        self.prices_dark.append(execution_price)
        self.impact_dark.append(execution_price - nbbo)
        
        return {
            'execution': execution_price,
            'final': final_price,
            'rebound': final_price,
            'total_impact': execution_price - nbbo,
            'permanent_impact': final_price - nbbo
        }

# Scenario 1: Varying order sizes
print("Scenario 1: Impact of Order Size")
print("=" * 80)

sim = MarketImpactSimulator()
order_sizes = [10000, 50000, 100000, 250000, 500000, 1000000]

public_impacts = []
dark_impacts = []

for size in order_sizes:
    public = sim.execute_public_exchange(size, nbbo=100.0)
    dark = sim.execute_dark_pool(size, nbbo=100.0)
    
    public_impacts.append(abs(public['total_impact']))
    dark_impacts.append(abs(dark['total_impact']))
    
    print(f"Order Size: {size:>10,} shares")
    print(f"  Public Exchange Impact: {public['total_impact']:>8.4f} ({public['total_impact']*100/100:.2f}%)")
    print(f"  Dark Pool Impact:       {dark['total_impact']:>8.4f} ({dark['total_impact']*100/100:.2f}%)")
    print(f"  Savings:                ${abs(public['total_impact'] - dark['total_impact']) * size:>15,.0f}")
    print()

# Scenario 2: Slicing strategy
print("Scenario 2: Slicing Large Order (100K shares over 5 periods)")
print("=" * 80)

nbbo = 100.0
total_order = 100000
num_pieces = 5

print("Strategy 1: Execute entire order immediately (public exchange)")
immediate_result = sim.execute_public_exchange(total_order, nbbo)
immediate_cost = immediate_result['total_impact'] * total_order
print(f"  Execution Price: ${immediate_result['execution']:.4f}")
print(f"  Total Cost: ${immediate_cost:>15,.0f}")

print("\nStrategy 2: Slice into 5 pieces, execute through dark pool")
slice_size = total_order // num_pieces
slice_prices = []
total_dark_cost = 0

for i in range(num_pieces):
    result = sim.execute_dark_pool(slice_size, nbbo)
    slice_prices.append(result['execution'])
    total_dark_cost += result['total_impact'] * slice_size
    nbbo = result['final']  # Update reference price

avg_slice_price = np.mean(slice_prices)
print(f"  Average Execution Price: ${avg_slice_price:.4f}")
print(f"  Total Cost: ${total_dark_cost:>15,.0f}")
print(f"  Savings vs Immediate: ${immediate_cost - total_dark_cost:>15,.0f}")

# Scenario 3: Market impact recovery
print(f"\n\nScenario 3: Price Recovery After Large Order")
print("=" * 80)

nbbo = 100.0
large_order = 500000

public = sim.execute_public_exchange(large_order, nbbo)
print(f"Initial NBBO: ${nbbo:.4f}")
print(f"Execution Price: ${public['execution']:.4f}")
print(f"Immediate Impact: ${public['total_impact']:.4f} ({public['total_impact']*100:.2f}%)")
print(f"Price After Recovery: ${public['rebound']:.4f}")
print(f"Permanent Impact: ${public['permanent_impact']:.4f} ({public['permanent_impact']*100:.2f}%)")
print(f"Temporary Impact: ${public['total_impact'] - public['permanent_impact']:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Market impact vs order size
axes[0, 0].plot(order_sizes, [p*10000 for p in public_impacts], 'o-', linewidth=2, markersize=8, label='Public Exchange')
axes[0, 0].plot(order_sizes, [d*10000 for d in dark_impacts], 's-', linewidth=2, markersize=8, label='Dark Pool')
axes[0, 0].set_xlabel('Order Size (shares)')
axes[0, 0].set_ylabel('Price Impact (cents)')
axes[0, 0].set_title('Scenario 1: Market Impact Comparison')
axes[0, 0].set_xscale('log')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Execution cost comparison
cost_savings = [(p - d) * s for p, d, s in zip(public_impacts, dark_impacts, order_sizes)]
colors_cost = ['green' if s > 0 else 'red' for s in cost_savings]

axes[0, 1].bar(range(len(order_sizes)), [s/1000 for s in cost_savings], color=colors_cost, alpha=0.7)
axes[0, 1].set_xticks(range(len(order_sizes)))
axes[0, 1].set_xticklabels([f'{s/1000:.0f}K' for s in order_sizes], rotation=45)
axes[0, 1].set_ylabel('Cost Savings ($1000s)')
axes[0, 1].set_title('Scenario 1: Dark Pool Savings by Order Size')
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Slicing effectiveness
strategies = ['Immediate\nPublic', 'Sliced\nDark Pool']
costs = [immediate_cost/1000, total_dark_cost/1000]
colors_strat = ['red', 'green']

bars = axes[1, 0].bar(strategies, costs, color=colors_strat, alpha=0.7)
axes[1, 0].set_ylabel('Execution Cost ($1000s)')
axes[1, 0].set_title('Scenario 2: Slicing Strategy Comparison')
axes[1, 0].grid(alpha=0.3, axis='y')

for bar, cost in zip(bars, costs):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost:.1f}K', ha='center', va='bottom', fontweight='bold')

# Plot 4: Impact decomposition (permanent vs temporary)
impact_types = ['Immediate', 'Permanent', 'Temporary']
impact_values = [public['total_impact']*100, public['permanent_impact']*100, 
                (public['total_impact'] - public['permanent_impact'])*100]
colors_impact = ['blue', 'red', 'orange']

bars = axes[1, 1].bar(impact_types, impact_values, color=colors_impact, alpha=0.7)
axes[1, 1].set_ylabel('Price Impact (basis points)')
axes[1, 1].set_title('Scenario 3: Impact Decomposition (500K share order)')
axes[1, 1].grid(alpha=0.3, axis='y')

for bar, impact in zip(bars, impact_values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{impact:.1f}bps', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Average Impact (Public): {np.mean(public_impacts)*10000:.2f} cents per $10K traded")
print(f"Average Impact (Dark):   {np.mean(dark_impacts)*10000:.2f} cents per $10K traded")
print(f"Avg Savings:             {(np.mean(public_impacts) - np.mean(dark_impacts))*10000:.2f} cents per $10K traded")
print(f"Total Savings (100K order): ${(immediate_cost - total_dark_cost):,.0f}")
```

## 6. Challenge Round
Why do regulatory authorities worry about dark pools even though they appear to benefit institutions through lower market impact?

- **Price discovery erosion**: Less visible orders → prices don't reflect true supply/demand → fundamental prices lag → inefficient overall
- **Retail harm**: Dark pools benefit institutions → retail left on public exchanges → retail faces wider spreads (liquidity migrates to dark) → two-tier market
- **Information advantage**: Data selling creates asymmetry → HFTs profit from knowing where trades occurred → arms race disadvantages everyone else
- **Predatory behavior**: Lack of transparency enables front-running, spoofing, manipulation → brokers can route orders to harm customers
- **Systemic risk**: Positions accumulate hidden in pools → crash can be sharper when positions revealed → less visibility for regulators to prevent cascade

## 7. Key References
- [Zhu (2014) - Do Dark Pools Harm Price Discovery?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2420541)
- [SEC Report on Dark Pools](https://www.sec.gov/news/press/2012-26.htm)
- [Harris (2003) - Trading and Exchanges - Chapter on Block Trading](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708)
- [Barclays LXE Enforcement Action](https://www.sec.gov/litigation/litreleases/2014/lr23135.htm)

---
**Status:** Off-exchange hidden execution | **Complements:** Market Impact, Price Discovery, Information Asymmetry, Systemic Risk
