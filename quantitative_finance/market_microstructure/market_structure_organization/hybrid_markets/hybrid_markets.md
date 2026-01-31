# Hybrid Markets

## 1. Concept Skeleton
**Definition:** Trading venues combining order-driven electronic matching with quote-driven market makers; both mechanisms operate simultaneously  
**Purpose:** Blend price discovery efficiency of order-driven with liquidity certainty of quote-driven; optimize best of both worlds  
**Prerequisites:** Order-driven markets, quote-driven markets, market maker role, order book mechanics

## 2. Comparative Framing
| Market Type | Order-Driven Component | MM Component | Price Discovery | Liquidity | Example |
|------------|----------------------|--------------|-----------------|-----------|---------|
| **Pure Order-Driven** | 100% | None | Via order book | Uncertain | NASDAQ |
| **Hybrid** | 70-80% | 20-30% MM | Both mechanisms | Certain + competitive | NYSE |
| **Quote-Driven** | None | 100% | Via dealers | Certain | FX OTC |
| **Auction** | 100% batch | None | Periodic clearing | Periodic | Options open/close |

## 3. Examples + Counterexamples

**Hybrid Success:**  
NYSE: 50 Designated Market Makers (DMMs) + electronic order book → Small orders match electronically (tight spreads) → Large blocks negotiate with DMM (liquidity certain)

**Hybrid Advantage in Crisis:**  
March 2020 COVID crash: Electronic order book dries up → DMMs provide continuous bids → market doesn't completely collapse → slow recovery vs fast crash

**Hybrid Failure:**  
CME pit trading (hybrid) overcrowded → pit traders + electronic orders compete → confusion → trading halted → now mostly electronic

**Pure Order-Driven Alternative:**  
NASDAQ pure electronic: No MMs → tight spreads in normal markets → but during volatility, spreads widen 10x (no MM to absorb shocks)

## 4. Layer Breakdown
```
Hybrid Market Framework:
├─ Dual Mechanism Structure:
│   ├─ Electronic Order Book:
│   │   - Continuous limit order matching
│   │   - Small orders route here automatically
│   │   - Price-time priority (FIFO)
│   │   - Example: 100-1000 share orders
│   ├─ Market Maker Participation:
│   │   - MMs also post on order book
│   │   - MMs required to maintain minimum spread
│   │   - MMs earn spread on their orders
│   │   - MMs also negotiate large blocks
│   ├─ Order Routing:
│   │   - Retail orders → electronic (tight spreads)
│   │   - Large orders → DMM negotiation
│   │   - Broker choice: Route to best price
│   │   - Example: 10K order at $100.00 electronic, 1M order negotiated with MM
│   ├─ Price Priority:
│   │   - Best price: Electronic limit order book or MM quote
│   │   - NBBO (National Best Bid Offer): Includes both
│   │   - MM improvement: MM can improve best bid/ask
│   │   - Execution: Goes to best price regardless of source
│   └─ Block Execution:
│       - Large orders (100K+): Negotiated with MM directly
│       - Off-exchange: Can do on floor/OTC
│       - Price: May improve vs electronic for size
│       - Settlement: T+2 typical
│
├─ Designated Market Maker (DMM) Role:
│   ├─ DMM Obligations (NYSE):
│   │   - Maintain liquidity: Quote continuously
│   │   - Fair and orderly: Don't manipulate
│   │   - Price improvement: Step in if spread too wide
│   │   - Spread limits: Typical max $0.05 for large-cap
│   ├─ DMM Compensation:
│   │   - Bid-ask spread: DMM profit per share traded
│   │   - Rebates: May receive rebates for providing liquidity
│   │   - Fees: May pay fees for taking liquidity
│   │   - Block commissions: Negotiated % on large trades
│   ├─ DMM Inventory Management:
│   │   - Holding long: Too many buy orders
│   │   - Holding short: Too many sell orders
│   │   - Adjustment: Widened spread to balance inventory
│   │   - Hedging: Offset in derivatives or other stocks
│   ├─ DMM Duties:
│   │   - Opening procedures: Determine opening price
│   │   - Halt trading: Suspend if news pending
│   │   - Closing procedures: Determine closing price
│   │   - Fair dealing: Treat all customers equally
│   └─ Regulatory Oversight:
│       - SEC monitoring: Real-time surveillance
│       - Compliance: Strict rules on conduct
│       - Penalties: Fines for violations
│       - Example: NYSE fined Goldman Sachs for conflict of interest (2010)
│
├─ Hybrid Advantages:
│   ├─ Best of Both Worlds:
│   │   - Competitive spreads: Electronic orders compete
│   │   - Liquidity certainty: MM backstop ensures execution
│   │   - Large block accommodation: DMM negotiates
│   │   - Efficient discovery: Prices from both sources
│   ├─ Price Improvement:
│   │   - Retail: Benefit from tight electronic spreads
│   │   - Institutional: Benefit from DMM flexibility
│   │   - Competition: MM must improve vs order book
│   │   - Example: Electronic bid $99.99, MM improves to $100.00
│   ├─ Stability:
│   │   - Circuit breaker: DMM can stabilize in crisis
│   │   - Floor trading: Personal relationships help negotiation
│   │   - Capital: DMM provides capital when market stressed
│   │   - 1987 crash: Floor trading limited cascade damage
│   ├─ Information Aggregation:
│   │   - DMM sees all flow: Informed and uninformed
│   │   - Better pricing: DMM prices knowing flow patterns
│   │   - Price discovery: Faster than pure order-driven
│   │   - Market watch: DMM monitors for manipulation
│   └─ Regulatory Control:
│       - Centralized: Single DMM responsible for quality
│       - Oversight easier: Known counterparty
│       - Rules enforceable: DMM has compliance dept
│       - Accountability: Clear who to hold responsible
│
├─ Hybrid Disadvantages:
│   ├─ Complexity:
│   │   - Dual system: More complex than pure order-driven
│   │   - Routing rules: Must determine best execution
│   │   - Order flow: Harder to predict where order goes
│   │   - Technology: Expensive systems to manage both
│   ├─ Potential Conflicts:
│   │   - DMM advantage: Sees all flow, can trade ahead
│   │   - Front-running risk: DMM could abuse information
│   │   - Regulatory burden: Must prevent conflicts
│   │   - Customer protection: Rules limit DMM trading
│   ├─ Higher Costs:
│   │   - DMM compensation: Spreads may be wider
│   │   - Floor overhead: Floor trading infrastructure
│   │   - Technology: Dual systems cost more
│   │   - Example: NYSE floor costs $100M+ annually
│   ├─ Execution Uncertainty:
│   │   - Where does order go?: Electronic or MM?
│   │   - Timing: May execute at worse price if routed wrong
│   │   - Transparency: Not always clear why routed certain way
│   │   - Disputes: Broker must ensure best execution
│   └─ DMM Monopoly:
│       - Single designated MM: Limited alternatives
│       - Market power: DMM can extract rents
│       - Switching costs: Hard to switch MM
│       - Inefficiency: Single provider less efficient
│
├─ Hybrid Market Examples:
│   ├─ New York Stock Exchange (NYSE):
│   │   - 50 Designated Market Makers
│   │   - Electronic order book + floor trading
│   │   - Top 20% trades use floor negotiation
│   │   - Bottom 80% trades electronic
│   ├─ Tokyo Stock Exchange (TSE):
│   │   - Hybrid auction system
│   │   - Morning and afternoon sessions with specialists
│   │   - Electronic matching + specialist intervention
│   ├─ London Stock Exchange (LSE):
│   │   - Electronic order book (SETS)
│   │   - Market makers also provide liquidity
│   │   - Large cap electronic, small cap hybrid
│   └─ Deutsche Börse (Xetra):
│       - Electronic order book
│       - MMs (Designated Sponsors) participate
│       - MMs quoted continuously
│       - Strict spread and quote requirements
│
├─ Order Routing in Hybrid Markets:
│   ├─ Best Execution Requirement:
│   │   - Brokers must route to best price
│   │   - Consider: Speed, size, likelihood of execution
│   │   - Documentation: Must show best execution
│   │   - Example: 5K share order gets electronic spread, 500K negotiates with MM
│   ├─ Routing Algorithm:
│   │   - 1. Check electronic book for best price
│   │   - 2. Check MM quote at that venue
│   │   - 3. Determine probability of electronic execution
│   │   - 4. Route to most likely to execute at best price
│   │   - 5. Example: Electronic has 1000 shares @ $100.00, MM has 100K @ $100.01
│   │       - For 5K order: Route electronic (limited liquidity)
│   │       - For 200K order: Route to MM (guaranteed execution)
│   ├─ Payment for Order Flow:
│   │   - MM pays broker for order flow
│   │   - Creates incentive to route to MM
│   │   - Controversial: May hurt customer execution
│   │   - SEC debate: Is this best execution?
│   └─ Transparency Requirements:
│       - Disclosure: Must show routing destinations
│       - Performance: Report execution quality
│       - Conflicts: Disclose who pays for order flow
│       - FINRA: Requires quarterly reporting
│
├─ MM Trading Strategies in Hybrid:
│   ├─ Passive Liquidity Provision:
│   │   - Quote tight bid-ask: Earn spread
│   │   - Minimize adverse selection: Know market
│   │   - Inventory management: Mean-revert positions
│   │   - Profit model: Volume × spread
│   ├─ Active Trading:
│   │   - Position taking: MM bets on direction
│   │   - Insider info: Flow reveals patterns
│   │   - Scalping: Quick profit from temporary mispricings
│   │   - Front-running: Illegal, but tempting given information
│   ├─ Flow Prediction:
│   │   - Momentum: If buying volume high, buy first
│   │   - Reversal: If selling extreme, position for bounce
│   │   - Seasonal: Time-of-day patterns
│   │   - Leverage: AI models predict next order
│   └─ Regulatory Constraints:
│       - Best execution: Can't disadvantage customers
│       - Fair dealing: Can't front-run
│       - Spread caps: Must maintain tight spreads
│       - Position limits: Cap size of bets
│
├─ Evolution of Hybrid Markets:
│   ├─ Historical:
│   │   - Before 1990s: All floor trading (dealers)
│   │   - 1990s: Electronic systems introduce, floor remains
│   │   - 2000s: Electronic dominance, floor shrinking
│   │   - 2010s: Floor mostly ceremonial, electronic primary
│   │   - 2020s: Push to eliminate floor entirely
│   ├─ Technology Shift:
│   │   - Electronic faster: Microseconds vs seconds
│   │   - Floor advantages disappear: Tech reduces info advantage
│   │   - HFT favors pure electronic: Colocation, speed
│   │   - Pressure to close floor: Cost without benefit
│   ├─ Regulatory Pressure:
│   │   - Transparency: Post-2010 rules require full disclosure
│   │   - Conflicts: Strict rules on MM trading
│   │   - MiFID (EU): Pushes toward electronic
│   │   - SEC rules: Encourage automation
│   ├─ Future Trends:
│   │   - Pure electronic: NYSE may eliminate floor
│   │   - Algorithmic MM: Robots instead of humans
│   │   - Global integration: Reduced fragmentation
│   │   - Regulatory consolidation: Harmonized rules
│   └─ Emerging Markets Adoption:
│       - India: Introduced MM requirements
│       - Brazil: Hybrid model adopted
│       - China: Electronic with specialist oversight
│       - Southeast Asia: Gradual shift toward electronic
│
├─ Crisis Behavior:
│   ├─ 1987 Black Monday:
│   │   - Hybrid structure saved market
│   │   - Floor traders negotiated large blocks
│   │   - Circuit breakers halted cascade
│   │   - Recovery: Floor provided stability vs electronic
│   ├─ 2008 Financial Crisis:
│   │   - MMs provided liquidity initially
│   │   - Then withdrew (inventory blow up)
│   │   - Spreads widened dramatically (10x+)
│   │   - Lesson: MM guarantees limited in crisis
│   ├─ 2010 Flash Crash:
│   │   - Electronic order book failed
│   │   - Prices spiked 60% in seconds
│   │   - Circuit breakers halted trading
│   │   - MMs withdrew, not helpful
│   │   - Lesson: Hybrid not sufficient for extreme crash
│   └─ 2020 COVID Crash:
│       - Electronic spreads widened 20x
│       - MM presence helped early recovery
│       - Federal intervention key to stability
│       - Lesson: Both mechanisms needed plus external support
│
└─ Regulatory Framework:
    ├─ SEC Rules (US):
    │   - Regulation NMS: Best execution across venues
    │   - Rule 10b-5: No insider trading or front-running
    │   - Rule 15c3-1: Capital requirements for MM
    │   - Conflict rules: MM can't disadvantage customers
    ├─ NYSE Rules (Specific):
    │   - DMM Agreement: Required obligations
    │   - Spread limits: Max spread rules
    │   - Quoting rules: Min quote size and frequency
    │   - Conduct: Fair and orderly market
    ├─ FINRA Rules (Brokers):
    │   - Best execution: Quarterly documentation
    │   - Order routing: Disclosure of where orders go
    │   - Payment for order flow: Must disclose
    │   - Suitability: Recommendations must be suitable
    ├─ International:
    │   - MiFID (EU): Electronic execution preference
    │   - FCA (UK): Hybrid systems allowed but transparent
    │   - ASX (Australia): Hybrid model similar to NYSE
    │   - TSE (Japan): Specialist system being modernized
    └─ Post-Trade Surveillance:
        - Blotter monitoring: Track all MM trades
        - Execution review: Ensure best prices
        - Suspicious activity: Flag manipulation
        - Reporting: Monthly to regulators
```

**Interaction:** Order submitted → routed to best price source → executes on electronic or negotiates with MM → confirmation sent

## 5. Mini-Project
Simulate hybrid market with both electronic and MM execution:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

np.random.seed(42)

class HybridMarket:
    def __init__(self, initial_price=100.0):
        self.price = initial_price
        self.electronic_bid = initial_price - 0.005
        self.electronic_ask = initial_price + 0.005
        self.electronic_book = {'bid': {}, 'ask': {}}
        self.mm_bid = initial_price - 0.01
        self.mm_ask = initial_price + 0.01
        self.mm_inventory = 0
        self.trade_log = []
        self.routing_log = []
        self.spread_history = []
        self.price_history = []
        
    def get_nbbo(self):
        """Get National Best Bid Offer (best across both)"""
        best_bid = max(self.electronic_bid, self.mm_bid)
        best_ask = min(self.electronic_ask, self.mm_ask)
        return best_bid, best_ask
    
    def route_order(self, side, quantity, order_price=None):
        """Route order to best execution venue"""
        nbbo_bid, nbbo_ask = self.get_nbbo()
        
        if side == 'buy':
            # Compare electronic ask vs MM ask
            elec_ask = self.electronic_ask
            mm_ask = self.mm_ask
            
            if elec_ask <= mm_ask and np.random.random() < 0.7:
                # Likely execute electronic
                venue = 'electronic'
                execution_price = elec_ask
            else:
                venue = 'mm'
                execution_price = mm_ask
        else:  # sell
            # Compare electronic bid vs MM bid
            elec_bid = self.electronic_bid
            mm_bid = self.mm_bid
            
            if elec_bid >= mm_bid and np.random.random() < 0.7:
                venue = 'electronic'
                execution_price = elec_bid
            else:
                venue = 'mm'
                execution_price = mm_bid
        
        return venue, execution_price
    
    def execute_order(self, side, quantity):
        """Execute order and update market"""
        venue, price = self.route_order(side, quantity)
        
        # Update MM inventory
        if venue == 'mm':
            if side == 'buy':
                self.mm_inventory -= quantity
            else:
                self.mm_inventory += quantity
            
            # MM adjusts quotes based on inventory
            inventory_adjustment = 0.002 * self.mm_inventory / 10000
            self.mm_bid = self.price - 0.01 + inventory_adjustment
            self.mm_ask = self.price + 0.01 + inventory_adjustment
        
        # Update electronic book (simplified)
        self.electronic_bid = self.price - 0.005
        self.electronic_ask = self.price + 0.005
        
        # Update market price toward execution price
        self.price = 0.7 * self.price + 0.3 * price
        
        # Record trade
        self.trade_log.append({
            'side': side,
            'venue': venue,
            'price': price,
            'quantity': quantity,
            'mm_inventory': self.mm_inventory
        })
        
        self.routing_log.append(venue)
        spread = self.mm_ask - self.mm_bid
        self.spread_history.append(spread)
        self.price_history.append(self.price)
        
        return price

# Scenario 1: Normal hybrid market
print("Scenario 1: Normal Hybrid Market Routing")
print("=" * 80)

hybrid1 = HybridMarket()

for t in range(200):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000, 5000])
    price = hybrid1.execute_order(side, quantity)

electronic_ratio = hybrid1.routing_log.count('electronic') / len(hybrid1.routing_log) * 100
mm_ratio = 100 - electronic_ratio

print(f"Total Orders: {len(hybrid1.routing_log)}")
print(f"Electronic Routing: {electronic_ratio:.1f}%")
print(f"MM Routing: {mm_ratio:.1f}%")
print(f"Average Spread: ${np.mean(hybrid1.spread_history):.4f}")
print(f"MM Final Inventory: {hybrid1.mm_inventory:,.0f} shares")

# Scenario 2: Stress (wider MM quotes)
print(f"\n\nScenario 2: Market Stress (MM Widens Spreads)")
print("=" * 80)

hybrid2 = HybridMarket()

for t in range(100):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000])
    price = hybrid2.execute_order(side, quantity)

# Stress: MM reduces liquidity (widens spread)
for t in range(100):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000])
    
    # MM widens spread 10x in stress
    hybrid2.mm_bid = hybrid2.price - 0.05
    hybrid2.mm_ask = hybrid2.price + 0.05
    
    price = hybrid2.execute_order(side, quantity)

normal_period_spread = np.mean(hybrid2.spread_history[:100])
stress_period_spread = np.mean(hybrid2.spread_history[100:])

print(f"Normal Period Average Spread: ${normal_period_spread:.4f}")
print(f"Stress Period Average Spread: ${stress_period_spread:.4f}")
print(f"Spread Widening: {stress_period_spread / normal_period_spread:.1f}x")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Routing pie chart
routing_counts = [hybrid1.routing_log.count('electronic'), hybrid1.routing_log.count('mm')]
labels = ['Electronic', 'MM']
colors = ['blue', 'orange']

axes[0, 0].pie(routing_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[0, 0].set_title('Scenario 1: Order Routing Distribution')

# Plot 2: Price evolution
times = range(len(hybrid1.price_history))
axes[0, 1].plot(times, hybrid1.price_history, linewidth=2, marker='o', markersize=2)
axes[0, 1].set_xlabel('Trade Number')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].set_title('Scenario 1: Price Evolution')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Spread comparison
scenarios = ['Normal\nMarket', 'Stress\nMarket']
spreads = [normal_period_spread, stress_period_spread]
colors_stress = ['green', 'red']

bars = axes[1, 0].bar(scenarios, spreads, color=colors_stress, alpha=0.7)
axes[1, 0].set_ylabel('Average Spread ($)')
axes[1, 0].set_title('Scenario 2: Spread Widening in Stress')
axes[1, 0].grid(alpha=0.3, axis='y')

for bar, spread in zip(bars, spreads):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'${spread:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: MM inventory
inventory_path = [t['mm_inventory'] for t in hybrid1.trade_log]
times = range(len(inventory_path))

axes[1, 1].plot(times, inventory_path, linewidth=2, color='purple')
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 1].fill_between(times, 0, inventory_path, alpha=0.3)
axes[1, 1].set_xlabel('Trade Number')
axes[1, 1].set_ylabel('MM Inventory (shares)')
axes[1, 1].set_title('Scenario 1: Market Maker Inventory')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n\nHybrid Market Summary:")
print("=" * 80)
print(f"Average Price: ${np.mean(hybrid1.price_history):.2f}")
print(f"Price Range: ${min(hybrid1.price_history):.2f} - ${max(hybrid1.price_history):.2f}")
print(f"Final MM Inventory: {hybrid1.mm_inventory:,.0f} shares")
```

## 6. Challenge Round
Why do hybrid markets sometimes fail to provide liquidity in crisis when MMs withdraw, making them no better than pure order-driven?

- **Inventory limits**: MM holds maximum position size → in crisis, inventory fills up quickly → can't absorb more orders → withdrawal inevitable
- **Capital constraints**: MM needs capital to hold inventory → crisis wipes out capital → funding dries up → MM withdraws
- **Adverse selection**: In crisis, all flow is informed (panic sellers, everyone liquidating) → MM loses money on every trade → unprofitable to continue → withdrawal
- **Cascade risk**: MM withdrawal → spreads widen → triggers stop orders → more selling → more MM withdrawal → positive feedback loop
- **Regulatory capital**: MM must meet capital ratios → crisis means reduced value of collateral → forced to reduce positions → withdrawal

## 7. Key References
- [Hasbrouck & Schwandt (2012) - The 2010 Flash Crash: High-Frequency Trading in an Electronic Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2019262)
- [Harris (2003) - Trading and Exchanges - Chapter on Hybrid Markets](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708)
- [SEC Report on May 6, 2010 Flash Crash](https://www.sec.gov/news/press/2010-85.htm)
- [NYSE Market Quality Review](https://www.nyse.com/publicdocs/nyse/regulation/marketplace-regulation/market-information/faq.pdf)

---
**Status:** Dual mechanism execution | **Complements:** Order-Driven Markets, Quote-Driven Markets, Market Maker Role, Crisis Management
