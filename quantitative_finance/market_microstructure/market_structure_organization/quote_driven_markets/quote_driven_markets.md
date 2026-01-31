# Quote-Driven Markets

## 1. Concept Skeleton
**Definition:** Trading venues where market makers/dealers post continuous bid-ask quotes; counterparty supplied by dealer, not order matching  
**Purpose:** Provide liquidity certainty; absorb order flow without requiring matching counterparty; negotiate prices for large blocks  
**Prerequisites:** Market maker function, bid-ask spread, inventory management, adverse selection

## 2. Comparative Framing
| Market Type | Liquidity Provider | Price Discovery | Execution | Transparency |
|------------|-------------------|-----------------|-----------|--------------|
| **Quote-Driven** | Dealer posts quotes | Dealer sets prices | Bilateral negotiation | Quotes only |
| **Order-Driven** | Traders post limit orders | Order book matching | Automated execution | Full book visible |
| **Hybrid** | Both dealers and order book | Combination | Both mechanisms | Partial transparency |
| **Auction** | Central clearinghouse | Clearing price | Periodic batching | Full at auction |

## 3. Examples + Counterexamples

**Quote-Driven Success:**  
Foreign exchange (FX): Bank dealer posts EUR/USD bid $1.0900, ask $1.0902 → customer demands buy → gets $1.0902 immediately → certain execution, no waiting

**Quote-Driven Advantage in Crisis:**  
Market stress: Equity order book crashes (no liquidity) → FX quote-driven market stays functional (dealers provide continuous quotes)

**Quote-Driven Failure:**  
Corporate bond market illiquid → large dealer quotes $1000 bid, $1050 ask (5% spread) → trader negotiates, gets $1025 after haggling → inefficient price discovery

**Order-Driven Alternative:**  
NASDAQ electronic: Limit order book provides $100.00 bid, $100.01 ask (0.01% spread) → automatic matching → no negotiation needed

## 4. Layer Breakdown
```
Quote-Driven Market Framework:
├─ Core Mechanics:
│   ├─ Dealer Role:
│   │   - Dealer posts bid and ask (quote)
│   │   - Bid: Price dealer buys at (trader sells to dealer)
│   │   - Ask: Price dealer sells at (trader buys from dealer)
│   │   - Spread: Ask - Bid = dealer's compensation for risk
│   │   - Example: Dealer posts bid $100.00, ask $100.05
│   ├─ Trading Process:
│   │   1. Dealer posts continuous quotes (updated throughout day)
│   │   2. Customer can trade at quoted price
│   │   3. Execution: Bilateral agreement (customer + dealer)
│   │   4. No order book: No queue, no waiting
│   │   5. Settlement: T+2 or T+3 depending on market
│   ├─ Quote Adjustment:
│   │   - Dealer adjusts quotes based on:
│   │   - Inventory position (holding too much? Widen bid)
│   │   - Risk aversion (volatile? Widen spread)
│   │   - Competition (other dealers quoting tighter? Follow)
│   │   - Information (news arrived? Adjust quotes before trading)
│   ├─ Price Discovery:
│   │   - Dealer sets prices based on:
│   │   - Fundamental value (estimation of fair value)
│   │   - Inventory costs (cost to hold position)
│   │   - Adverse selection costs (fear of informed traders)
│   │   - Competitive pressure (other dealers' quotes)
│   │   - Formula: Quote = Fair Value + Inventory Adjustment + Adverse Selection
│   ├─ Negotiation:
│   │   - Large trades: Can negotiate away from posted quote
│   │   - Example: Posted $100.00-$100.05, negotiate $100.02 for 1M shares
│   │   - Discount for size: Larger orders get better prices
│   │   - Relationship factor: Long-term customers get better treatment
│   └─ Multiple Dealers:
│       - Typically multiple dealers in same security
│       - Each quotes own bid-ask
│       - Trader can shop around: "What's your bid on 1M shares?"
│       - Best execution: Choose tightest quote
│       - Negotiation: Dealers compete on price for large orders
│
├─ Advantages of Quote-Driven Markets:
│   ├─ Execution Certainty:
│   │   - Immediate counterparty: Dealer always buys/sells
│   │   - No waiting: Can execute instantly at quoted price
│   │   - No queue position: Order size irrelevant to priority
│   │   - Useful for urgent need: Always possible to execute
│   ├─ Large Order Accommodation:
│   │   - Block trades: Dealers negotiate large blocks
│   │   - Price improvement: Negotiate discount for size
│   │   - No market impact: Dealer absorbs order without public price move
│   │   - Example: Buy 1M shares without market knowing size
│   ├─ Stability:
│   │   - Dealer commitment: Provides continuous liquidity
│   │   - No liquidity crisis: Dealer may stay in market during stress
│   │   - Reduces cascades: Single dealer absorbs multiple orders
│   │   - Price stability: Less volatile than order-driven (dealer dampening)
│   ├─ Customization:
│   │   - Large trades negotiable: Can structure larger positions
│   │   - Terms flexible: Settlement dates, quantity, price all negotiable
│   │   - Relationship pricing: Long-term customers get better treatment
│   │   - Cross-trading: Dealer can offer swaps, repos, derivatives
│   ├─ Confidentiality:
│   │   - Bilateral trade: Only dealer and trader know size
│   │   - Order book not public: No front-running risk
│   │   - Privacy: Useful for large institutional traders
│   │   - Example: Pension fund can buy block without moving market
│   └─ Dealer Expertise:
│       - Market making experience: Dealer prices efficiently
│       - Inventory management: Dealer balances long/short
│       - Information integration: Dealer synthesizes market info
│       - Research: Dealer provides fundamental analysis
│
├─ Disadvantages of Quote-Driven Markets:
│   ├─ Wide Spreads:
│   │   - Dealer markup: Bid-ask spread is dealer profit
│   │   - Large-cap equities: 0.01% spread (order-driven) vs 0.05% (quote-driven)
│   │   - Corporate bonds: 0.5-2% spread common (order-driven unavailable)
│   │   - Emerging markets: Spreads can exceed 5% (thin dealer network)
│   ├─ Limited Transparency:
│   │   - Quotes only visible: Full order book not disclosed
│   │   - No depth information: Don't know dealer inventory
│   │   - Hidden trades: Block trades negotiated off-exchange
│   │   - Information disadvantage: Retail traders have less data
│   ├─ Adverse Selection:
│   │   - Informed traders exploit: Buy at bid from dealer, sell at ask elsewhere
│   │   - Dealer loses on informed trade: Takes other side of informed order
│   │   - Compensation: Dealer widens spreads to protect
│   │   - Result: All traders pay higher spreads due to informed traders
│   ├─ Slow Discovery:
│   │   - Dealer-dependent: Prices move as dealer adjusts quotes
│   │   - Slower than order-driven: No continuous competitive pressure
│   │   - Quote stickiness: Dealers slow to change quotes during uncertainty
│   │   - Information lag: Fundamental changes take time to appear in quotes
│   ├─ Dealer Power:
│   │   - Monopoly risk: Single dealer can exploit customers
│   │   - Price discrimination: Different customers get different prices
│   │   - Relationship dependence: Customers may not switch easily
│   │   - Opaque pricing: Not clear if price is "fair"
│   ├─ Order Processing Costs:
│   │   - Manual negotiation: Takes time for large trades
│   │   - Documentation: Legal agreements required
│   │   - Settlement complexity: T+2 or T+3, not instantaneous
│   │   - Relationship management: Ongoing interaction needed
│   └─ Concentration Risk:
│       - Market concentrated: Few dealers
│       - Systemic risk: If dealer fails, market disruption
│       - Crisis vulnerability: Dealers may withdraw in crisis
│       - Example: 2008 credit crisis, bond dealer bid-ask widened 10x+
│
├─ Quote-Driven Market Examples:
│   ├─ Foreign Exchange (FX):
│   │   - Bank dealers (JP Morgan, Goldman, Deutsche Bank) quote
│   │   - EUR/USD: Most liquid, tight spreads (1 pip ≈ 0.0001)
│   │   - Emerging pairs: Less liquid, wider spreads (10-50 pips)
│   │   - Negotiations: Large (>$50M) orders negotiated
│   ├─ Fixed Income:
│   │   - Corporate bonds: Dealer network quotes
│   │   - Government bonds: Large dealer markets (BrokerTec, eSpeed)
│   │   - Spreads: 0.5-2% for illiquid corporates
│   │   - Block trading: Large positions negotiated off-exchange
│   ├─ Over-the-Counter (OTC) Derivatives:
│   │   - Swaps, forwards, options: Bank dealers quote
│   │   - Customization: Can structure any terms
│   │   - Bilateral: Customer calls dealer for price
│   │   - Spreads: Dealer profit margin 0.1-1% depending on product
│   ├─ Treasury Markets:
│   │   - Government bonds: Dealer auction process
│   │   - Primary dealers: 24 authorized dealers
│   │   - Secondary market: Dealers quote continuously
│   │   - Spreads: Tightest in government (most liquid)
│   └─ Emerging Markets:
│       - Local currency: Limited liquidity, dealer network only
│       - Equities: Some emerging markets use dealer models
│       - Spreads: Much wider than developed markets
│       - Negotiation: More common for larger trades
│
├─ Dealer Inventory Management:
│   ├─ Inventory Position:
│   │   - Dealer holds inventory: Buys when imbalanced
│   │   - Long position: Dealer holds more buy orders than sell
│   │   - Negative quote: Dealer desperate to sell (widens ask, tightens bid)
│   │   - Short position: Dealer holds more sell orders than buy
│   │   - Positive quote: Dealer desperate to buy (tightens ask, widens bid)
│   ├─ Stoll Model (Inventory Costs):
│   │   - Quote = Fair Value + Inventory Adjustment
│   │   - Inventory Adjustment = (Inventory - Target) × λ
│   │   - If too much inventory: λ positive, quote widens
│   │   - If too little inventory: λ negative, quote tightens (to attract sells)
│   ├─ Dynamic Quoting:
│   │   - Continuous adjustment: Quotes updated every millisecond
│   │   - HFT dealers: Adjust quotes to manage inventory
│   │   - Mean reversion: Dealers mean-revert inventory to target
│   │   - Profit from spread: Difference between buy and sell prices
│   ├─ Risk Management:
│   │   - Position limits: Dealer sets max inventory size
│   │   - Stop loss: Dealer exits if losing on position
│   │   - Hedging: Dealer hedges positions in derivatives
│   │   - Collateral: Dealer posts margin if dealer is broker-dealer
│   └─ Empirical:
│       - Spreads widen when dealer inventory high (adverse selection)
│       - Spreads tighten when inventory normalized
│       - Quote stickiness: Prices don't adjust immediately to news
│
├─ Competitive Dynamics:
│   ├─ Multiple Dealers:
│   │   - Competition tightens spreads: More dealers = narrower spreads
│   │   - Price discovery: Dealers compete on price
│   │   - Trader shopping: Large customers compare dealers
│   │   - Winner: Take-all dynamics for each dealer
│   ├─ Dealer Concentration:
│   │   - Few dealers dominate: e.g., JP Morgan, Goldman Sachs, BofA
│   │   - Relationship power: Customers stick with dealer
│   │   - Information advantage: Large dealer sees all flow
│   │   - Profits: Concentrated dealer makes higher spreads
│   ├─ Price Discrimination:
│   │   - Dealer charges different prices to different customers
│   │   - Sophisticated customer: Better price (larger volume)
│   │   - Retail customer: Worse price (small volume)
│   │   - Relationship: Long-term customers may get better rates
│   │   - Information: Dealer penalizes informed customers
│   └─ Quote Stuffing:
│       - Dealer quote a very tight bid-ask
│       - Immediately withdraw before anyone trades
│       - Purpose: Signal activity without risk
│       - Result: Quote stuffing appears like liquidity but isn't
│       - Regulation: SEC fined dealers for quote stuffing
│
├─ Regulatory Framework:
│   ├─ FCA Rules (UK):
│   │   - Market maker obligation: Designated dealers must quote
│   │   - Quote standards: Min size, max spread
│   │   - Pre-trade transparency: Some quotes public
│   │   - Post-trade: All trades reported
│   ├─ SEC Rules (US):
│   │   - Dealer registration: Only registered dealers can operate
│   │   - Best execution: Must give customers best prices
│   │   - Conflict of interest: Dealer can't disadvantage customers
│   │   - Reporting: MiFID (EU) requires post-trade data
│   ├─ FINRA Rules (US):
│   │   - Markup rules: Dealer can't charge excessive markups
│   │   - Blotter: Track all quotes and trades
│   │   - Supervisory review: Audit dealer compliance
│   │   - Suitability: Dealer must recommend suitable products
│   └─ OTC Transparency:
│       - Pre-trade: Quotes may not be public (negotiated)
│       - Post-trade: Trades must be reported (TRACE for bonds)
│       - Block trades: Special reporting (doesn't include all details)
│       - Dark pools: Additional transparency rules post-2010
│
├─ Crisis Dynamics:
│   ├─ 2008 Credit Crisis:
│   │   - Dealer inventory explodes (unwanted)
│   │   - Spreads widen 10-20x normal
│   │   - Liquidity evaporates: Dealers refuse to quote
│   │   - Government intervention: Fed facilities restore liquidity
│   │   - Contagion: Multiple dealers fail (Lehman Brothers)
│   ├─ Dealer Failure Risk:
│   │   - Counterparty risk: Dealer might not settle
│   │   - Capital requirements: Dealer needs to absorb losses
│   │   - Leverage: Dealers often highly leveraged (30:1 typical)
│   │   - Systemic: Dealer failure affects entire market
│   ├─ Market Drying Up:
│   │   - Dealer risk aversion: Won't quote when uncertain
│   │   - Bid-ask explodes: From 1 pip to 100 pips (100x wider)
│   │   - Trade volume crashes: Few trades occur
│   │   - Price discovery broken: Prices stuck or no trades
│   │   - Recovery: Takes weeks to restore normal spreads
│   └─ Government Response:
│       - Central bank liquidity: Fed adds money to system
│       - Dealer support: Fed backstops dealer funding
│       - Price support: Government buys to stabilize prices
│       - Regulation: Stricter capital rules post-crisis
│
├─ Modern Evolution:
│   ├─ Electronic Communication Networks (ECNs):
│   │   - Reduce dealer power: Alternative execution venues
│   │   - Automated matching: ECNs match orders automatically
│   │   - Transparency: Order books visible (like order-driven)
│   │   - Example: ITCH (electronic bonds), Bloomberg PORT (FX)
│   ├─ Hybrid Models:
│   │   - Dealer + order-driven: Both mechanisms available
│   │   - Customer choice: Route to dealer or order book
│   │   - Tighter spreads: Competition between dealer and order-driven
│   │   - Example: CME (pit + electronic), NYSE (floor + electronic)
│   ├─ Algorithmic Dealers:
│   │   - AI-driven market making: Algorithms set quotes
│   │   - Faster than humans: Quote update every millisecond
│   │   - Better inventory mgmt: Algorithms predict flows
│   │   - Lower spreads: Algorithm competition tightens
│   └─ Decentralization:
│       - Blockchain order books: Decentralized exchanges
│       - No central dealer: Peer-to-peer matching
│       - Smart contracts: Automate settlement
│       - Future trend: May replace some OTC markets
│
└─ Comparative Advantages:
    ├─ Quote-Driven Best For:
    │   - Large blocks: Dealer can negotiate
    │   - Illiquid securities: Dealer provides liquidity
    │   - Customization: Swaps, derivatives, negotiated terms
    │   - Crisis periods: Dealer maintains quotes when order book empty
    ├─ Order-Driven Best For:
    │   - Liquid securities: Tight spreads via competition
    │   - Retail traders: Fair pricing, no dealer markup
    │   - Speed: Automated matching (microseconds)
    │   - Transparency: Full order book visible
    ├─ Future Convergence:
    │   - Hybrid model: Most markets trending toward both
    │   - Customer choice: Route to best execution venue
    │   - Technology: Reduces dealer power (ECNs, trading platforms)
    │   - Regulation: Pushes for transparency and best execution
    └─ Economic Efficiency:
        - Bid-ask spread: Ultimate measure of market quality
        - Order-driven narrower in liquid markets (0.01% vs 0.1%)
        - Quote-driven better in crisis (liquidity available)
        - Net: Order-driven preferable except during stress
```

**Interaction:** Dealer posts quote → trader demands at quote price → execution occurs → dealer manages position → adjusts next quote

## 5. Mini-Project
Simulate quote-driven market with dealer inventory management and adverse selection:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class QuoteDrivenMarket:
    def __init__(self, initial_price=100.0, fundamental_value=100.0):
        self.price = initial_price
        self.fundamental = fundamental_value
        self.dealer_inventory = 0
        self.dealer_inventory_target = 0
        self.dealer_pnl = 0
        self.quote_history = []
        self.trade_log = []
        self.spread_history = []
        self.inventory_history = []
        self.bid_history = []
        self.ask_history = []
        
    def get_spread(self, inventory, volatility=0.01):
        """Calculate spread based on Stoll's model"""
        # Spread components:
        # 1. Adverse selection (fear of informed traders)
        # 2. Inventory costs (compensation for holding risk)
        
        adverse_selection_cost = 0.005 * volatility  # 0.5% base
        inventory_cost = 0.0001 * abs(inventory)  # Cost per unit of inventory
        
        spread = adverse_selection_cost + inventory_cost
        return max(0.001, spread)  # Minimum spread
    
    def set_quotes(self, fundamental, inventory, volatility):
        """Dealer sets bid and ask based on fundamental and inventory"""
        mid = fundamental
        spread = self.get_spread(inventory, volatility)
        
        # Inventory adjustment (widen ask if short, widen bid if long)
        inventory_adjustment = 0.002 * inventory  # Adjust mid by inventory
        
        bid = mid - spread / 2 + inventory_adjustment
        ask = mid + spread / 2 + inventory_adjustment
        
        return bid, ask
    
    def execute_trade(self, side, quantity, informed_prob=0.1):
        """Execute trade from customer"""
        
        # Update fundamental with random news
        news = np.random.normal(0, 0.002)
        self.fundamental *= (1 + news)
        
        # Determine if order is informed
        is_informed = np.random.random() < informed_prob
        
        # Get current quotes
        volatility = abs(self.fundamental - self.price) / self.price + 0.01
        bid, ask = self.set_quotes(self.fundamental, self.dealer_inventory, volatility)
        
        if side == 'buy':
            # Customer buys at dealer's ask
            execution_price = ask
            
            # If informed, customer wants to buy near fundamental
            if is_informed and self.price < self.fundamental:
                execution_price = ask  # Pay dealer's ask anyway
            
            self.dealer_inventory -= quantity  # Dealer sells, inventory decreases
            self.dealer_pnl += quantity * (ask - self.fundamental)  # PnL from trade
            
        else:  # sell
            # Customer sells at dealer's bid
            execution_price = bid
            
            # If informed, customer wants to sell near fundamental
            if is_informed and self.price > self.fundamental:
                execution_price = bid  # Get dealer's bid anyway
            
            self.dealer_inventory += quantity  # Dealer buys, inventory increases
            self.dealer_pnl += quantity * (self.fundamental - bid)  # PnL from trade
        
        # Update market price (slowly move toward fundamental)
        self.price = 0.9 * self.price + 0.1 * self.fundamental
        
        # Record trade
        self.trade_log.append({
            'side': side,
            'price': execution_price,
            'quantity': quantity,
            'inventory': self.dealer_inventory,
            'informed': is_informed
        })
        
        self.spread_history.append(ask - bid)
        self.inventory_history.append(self.dealer_inventory)
        self.bid_history.append(bid)
        self.ask_history.append(ask)
        
        return execution_price

# Scenario 1: Dealer managing inventory
print("Scenario 1: Dealer Inventory Management")
print("=" * 80)

market1 = QuoteDrivenMarket()

# Simulate customer orders (random arrival)
for t in range(200):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000, 5000])
    
    price = market1.execute_trade(side, quantity, informed_prob=0.05)

print(f"Initial Price: $100.00")
print(f"Final Price: ${market1.price:.2f}")
print(f"Fundamental Value: ${market1.fundamental:.2f}")
print(f"Total Trades: {len(market1.trade_log)}")
print(f"Dealer Inventory: {market1.dealer_inventory:,.0f} shares")
print(f"Dealer PnL: ${market1.dealer_pnl:,.2f}")
print(f"Average Spread: ${np.mean(market1.spread_history):.4f}")
print(f"Max Spread: ${np.max(market1.spread_history):.4f}")

# Scenario 2: High informed trader proportion
print(f"\n\nScenario 2: High Informed Trader Proportion")
print("=" * 80)

market2 = QuoteDrivenMarket()

# More informed traders increase spread
for t in range(150):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000])
    
    price = market2.execute_trade(side, quantity, informed_prob=0.3)  # 30% informed

print(f"Initial Price: $100.00")
print(f"Final Price: ${market2.price:.2f}")
print(f"Dealer Inventory: {market2.dealer_inventory:,.0f} shares")
print(f"Dealer PnL: ${market2.dealer_pnl:,.2f}")
print(f"Average Spread: ${np.mean(market2.spread_history):.4f}")
print(f"Avg Spread vs Low Informed: {np.mean(market2.spread_history) / np.mean(market1.spread_history):.2f}x wider")

# Scenario 3: Crisis (fundamental drops sharply)
print(f"\n\nScenario 3: Crisis Scenario (Fundamental Crash)")
print("=" * 80)

market3 = QuoteDrivenMarket(fundamental_value=100.0)

# Normal period
for t in range(100):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000])
    price = market3.execute_trade(side, quantity, informed_prob=0.1)

# Crisis: Fundamental drops 20%
market3.fundamental = 80.0
spreads_crisis = []

for t in range(100):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000])
    price = market3.execute_trade(side, quantity, informed_prob=0.5)  # More informed in crisis
    spreads_crisis.append(market3.spread_history[-1])

print(f"Price Pre-Crisis: $100.00")
print(f"Fundamental Pre-Crisis: $100.00")
print(f"Price Post-Crisis: ${market3.price:.2f}")
print(f"Fundamental Post-Crisis: $80.00")
print(f"Dealer Inventory at Crisis: {market3.dealer_inventory:,.0f} shares")
print(f"Dealer PnL: ${market3.dealer_pnl:,.2f}")
print(f"Average Spread Pre-Crisis: ${np.mean(market1.spread_history):.4f}")
print(f"Average Spread Crisis: ${np.mean(spreads_crisis):.4f}")
print(f"Spread Widening: {np.mean(spreads_crisis) / np.mean(market1.spread_history):.1f}x")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Bid-Ask evolution
times = range(len(market1.bid_history))
axes[0, 0].fill_between(times, market1.bid_history, market1.ask_history, alpha=0.3, color='blue')
axes[0, 0].plot(times, market1.bid_history, linewidth=1, label='Bid', color='blue')
axes[0, 0].plot(times, market1.ask_history, linewidth=1, label='Ask', color='red')
axes[0, 0].axhline(y=market1.fundamental, color='green', linestyle='--', linewidth=2, label='Fundamental')
axes[0, 0].set_xlabel('Trade Number')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Scenario 1: Dealer Quotes (Bid-Ask Evolution)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Dealer inventory
times = range(len(market1.inventory_history))
axes[0, 1].plot(times, market1.inventory_history, linewidth=2, color='purple')
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Trade Number')
axes[0, 1].set_ylabel('Inventory (shares)')
axes[0, 1].set_title('Scenario 1: Dealer Inventory Management')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].fill_between(times, 0, market1.inventory_history, alpha=0.3)

# Plot 3: Spread comparison
scenarios = ['Low\nInformed\n(5%)', 'High\nInformed\n(30%)']
avg_spreads = [np.mean(market1.spread_history), np.mean(market2.spread_history)]
colors = ['blue', 'orange']

bars = axes[1, 0].bar(scenarios, avg_spreads, color=colors, alpha=0.7)
axes[1, 0].set_ylabel('Average Bid-Ask Spread ($)')
axes[1, 0].set_title('Spread Widening with Informed Traders')
axes[1, 0].grid(alpha=0.3, axis='y')

for bar, spread in zip(bars, avg_spreads):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'${spread:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Crisis spread spike
time_crisis = range(len(spreads_crisis))
axes[1, 1].plot(time_crisis, spreads_crisis, linewidth=2, color='red', label='Crisis Period')
axes[1, 1].axhline(y=np.mean(market1.spread_history), color='blue', linestyle='--', linewidth=2, label='Normal Average')
axes[1, 1].set_xlabel('Trade Number (Crisis Period)')
axes[1, 1].set_ylabel('Bid-Ask Spread ($)')
axes[1, 1].set_title('Scenario 3: Spread Widening During Crisis')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nComparative Summary:")
print("=" * 80)
print(f"{'Metric':<30} {'Scenario 1':<20} {'Scenario 2':<20}")
print("-" * 70)
print(f"{'Average Spread':<30} {f'${np.mean(market1.spread_history):.4f}':<20} {f'${np.mean(market2.spread_history):.4f}':<20}")
print(f"{'Max Spread':<30} {f'${np.max(market1.spread_history):.4f}':<20} {f'${np.max(market2.spread_history):.4f}':<20}")
print(f"{'Final Inventory':<30} {f'{market1.dealer_inventory:,.0f}':<20} {f'{market2.dealer_inventory:,.0f}':<20}")
print(f"{'Dealer PnL':<30} {f'${market1.dealer_pnl:,.2f}':<20} {f'${market2.dealer_pnl:,.2f}':<20}")
print(f"{'Total Trades':<30} {f'{len(market1.trade_log)}':<20} {f'{len(market2.trade_log)}':<20}")
```

## 6. Challenge Round
Why do quote-driven markets often have wider bid-ask spreads than order-driven markets for the same liquid security?

- **Dealer markup**: Dealer compensation requires spread; no competition from limit orders → spreads wider. Order-driven spreads result from competitive pressure only
- **Adverse selection**: Dealers fear informed traders (lemons problem) → widen spreads to protect. Order-driven spreads more purely reflect supply/demand
- **Inventory costs**: Dealers hold positions and must hedge → cost passed to customers. Order-driven has no inventory (automatic matching)
- **Market power**: Fewer dealers (oligopoly) than limit order providers (many retailers) → less competitive pressure → wider spreads
- **Customization costs**: Quote-driven involves negotiation and documentation → higher operational costs → wider spreads to recover costs

## 7. Key References
- [Stoll (1978) - The Supply of Dealer Services in Securities Markets](https://www.jstor.org/stable/2327007)
- [O'Hara (1995) - Market Microstructure Theory](https://www.amazon.com/Market-Microstructure-Theory-Maureen-OHara/dp/0631207619)
- [Harris (2003) - Trading and Exchanges - Chapter on Quote-Driven Markets](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708)
- [Grossman & Miller (1988) - Liquidity and Market Structure](https://www.jstor.org/stable/1913721)

---
**Status:** Dealer-supplied liquidity | **Complements:** Order-Driven Markets, Bid-Ask Spread, Inventory Costs, Adverse Selection
