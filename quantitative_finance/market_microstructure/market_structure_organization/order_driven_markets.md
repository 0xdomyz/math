# Order-Driven Markets

## 1. Concept Skeleton
**Definition:** Electronic trading venues where prices determined by continuous matching of buy/sell limit orders in centralized order book  
**Purpose:** Price discovery through supply/demand interaction; transparent order book; fair queue-based execution  
**Prerequisites:** Order book mechanics, limit orders, market orders, execution priority rules

## 2. Comparative Framing
| Market Type | Price Discovery | Execution | Transparency | Venue | Example |
|------------|----------------|-----------|--------------|-------|---------|
| **Order-Driven** | Limit order book | FIFO queue | Full book visible | Electronic exchange | NASDAQ, NYSE |
| **Quote-Driven** | Market maker spreads | Bilateral negotiation | MM quotes only | OTC dealer network | FX, bonds, swaps |
| **Hybrid** | Both order book + MMs | Both mechanisms | Partial transparency | Exchange + floor | NYSE floor + electronic |
| **Auction** | Single clearing price | Periodic matching | Full transparency at auction | Central clearing | Options open/close |

## 3. Examples + Counterexamples

**Pure Order-Driven Success:**  
NASDAQ electronic limit order book: All traders submit orders → book continuously matched → best bid/ask visible to all → price discovered by supply/demand only → no market maker required

**Order-Driven Failure:**  
Illiquid penny stock trading on order-driven market: Bid $5.00 (1000 shares), Ask $5.10 (100 shares) → massive spread → poor price discovery → traders flee to OTC markets with MM negotiation

**Hybrid Advantage:**  
NYSE: Both electronic order matching AND floor market makers → large orders can route to floor for negotiated blocks → liquidity pools combine → better execution than pure electronic

**Quote-Driven Advantage:**  
FX market (quote-driven): Bank dealer posts continuous bid/ask on EUR/USD → trader demands execution → immediate counterparty guaranteed → no order queue delay vs order-driven

## 4. Layer Breakdown
```
Order-Driven Market Framework:
├─ Core Mechanics:
│   ├─ Order Book Structure:
│   │   - Buy side: All limit buy orders sorted descending (highest price first)
│   │   - Sell side: All limit sell orders sorted ascending (lowest price first)
│   │   - Example: Buy orders: $100.00 (100K), $99.99 (50K), $99.98 (200K)
│   │   - Example: Sell orders: $100.01 (75K), $100.02 (100K), $100.03 (50K)
│   │   - Midpoint price: ($100.00 + $100.01) / 2 = $100.005
│   │   - Spread: $100.01 - $100.00 = $0.01
│   ├─ Price Discovery Mechanism:
│   │   - No external price setter (unlike market makers)
│   │   - Prices emerge from supply/demand interaction
│   │   - First match: Buy at $100.00 meets sell at $100.01 → negotiated $100.005 (or rules apply)
│   │   - Information: Every trade reveals transaction price
│   │   - Order flow signal: New buy orders push up price, sell orders push down
│   ├─ Execution Rules:
│   │   - Price-time priority: Best price first, then FIFO within price level
│   │   - Pro-rata allocation (some exchanges): Share fills by size
│   │   - Example: $100.00 bid with 100K buy orders → first 10K order fills first
│   │   - Partial fills possible: 100 share order can fill 50 @ $100.00, 50 @ $99.99
│   ├─ Queue Position Value:
│   │   - First in queue at best price: Gets filled before later orders
│   │   - Time stamp = execution priority (FIFO)
│   │   - Incentive: Traders compete to be first (race to place order)
│   │   - Cost: Trading costs money (commissions, spreads)
│   │   - Benefit: Guaranteed execution at queue price
│   └─ Continuous Matching:
│       - Every new order immediately matched against best opposite side
│       - If no match available: Order rests on book for future matching
│       - Millisecond-speed matching: Automated computers do it
│       - Example: Sell 10K @ market → matches 5K @ $100.00 bid, 3K @ $99.99, 2K @ $99.98
│
├─ Advantages of Order-Driven Markets:
│   ├─ Transparency:
│   │   - Full order book visible: All market participants see liquidity
│   │   - Bid-ask spread public: No information asymmetry about prices
│   │   - Trade prices public: Every transaction reported (tape)
│   │   - Volume public: See how much trading at each level
│   ├─ Price Discovery:
│   │   - Supply and demand determine prices
│   │   - No dealer markup: Spread reflects supply/demand imbalance only
│   │   - Efficient: Prices incorporate all information quickly
│   │   - Competitive: Multiple buyers/sellers compete → better prices
│   ├─ Low Costs:
│   │   - No dealer markup: Spread is only transaction cost
│   │   - Competition: Many traders compete → spreads narrow
│   │   - Large-cap equities: Spreads as low as $0.0001 (0.01 cents/share)
│   │   - vs. Quote-driven: Dealer markups typically 5-10x larger
│   ├─ Fairness:
│   │   - FIFO queue: First come first served (equal treatment)
│   │   - No dealer discrimination: All orders treated equally
│   │   - Public information: No hidden advantages
│   │   - Regulation: SEC enforces best execution rules
│   ├─ Depth Information:
│   │   - Level 2 book: Shows top 10-20 levels of bids and asks
│   │   - Level 3 book: Shows all orders (typically reserved for brokers)
│   │   - Traders can optimize: See full liquidity profile
│   │   - Algorithms use depth: Predict price moves based on order imbalance
│   └─ High Participation:
│       - Low barriers to entry: Retail traders can participate equally
│       - Institutional and retail on same platform
│       - Competitive advantage: Speed (HFT) and strategy, not dealer power
│
├─ Disadvantages of Order-Driven Markets:
│   ├─ Illiquidity Problem:
│   │   - Illiquid securities: Sparse order book
│   │   - Large spreads: When few buyers/sellers
│   │   - Order impact: Large orders move price significantly
│   │   - Execution uncertainty: Order might not fill
│   │   - Example: Penny stocks: $5.00 bid, $5.50 ask, 100 share depth
│   ├─ Inventory Risk:
│   │   - Market maker function absent: No one forced to provide liquidity
│   │   - During crisis: Liquidity evaporates
│   │   - Flash crash: Order-driven market vulnerable to cascades
│   │   - Bid-ask can widen 10x in panic (normally $0.01, panic $0.10)
│   ├─ Execution Complexity:
│   │   - Order not immediately filled: Must wait or accept worse price
│   │   - Partial fills: Can complicate portfolio tracking
│   │   - Multiple fill prices: Tax accounting complicated
│   │   - Algorithms needed: Professional traders use algorithms to execute large orders
│   ├─ Price Volatility:
│   │   - Large order impact: Single order can cause price spikes
│   │   - Cascade effect: Stop orders triggered together → volatility spike
│   │   - HFT predation: Algorithms detect and trade-ahead of large orders
│   │   - vs. Quote-driven: Market maker absorbs large orders without price moves
│   ├─ Information Leakage:
│   │   - Order book visible: Competitors see your demand/supply
│   │   - Queue position known: Others see you're ahead in queue
│   │   - Front-running risk: Others trade ahead of your order
│   │   - Predatory algorithms: Use order book info to extract profits
│   └─ Operational Complexity:
│       - System coordination needed: Complex order matching algorithms
│       - Technology investment: Exchanges invest millions in systems
│       - Latency races: Arms race for speed (co-location, HFT)
│       - Regulatory burden: More rules needed for order book integrity
│
├─ Order-Driven Market Examples:
│   ├─ Equities:
│   │   - NASDAQ: Pure order-driven, fully electronic
│   │   - NYSE: Hybrid (electronic + floor DMMs)
│   │   - LSE: Order-driven, electronic
│   │   - TSE (Tokyo): Order-driven, hybrid auctions
│   ├─ Futures:
│   │   - CME: Pit trading (floor, quote-driven) → electronic (order-driven)
│   │   - EUREX: Fully electronic order-driven
│   │   - ICE: Hybrid electronic/voice trading
│   ├─ Options:
│   │   - CBOE: Market maker system (quote-driven, not order-driven)
│   │   - Electronic options venues: Emerging order-driven options
│   └─ Crypto:
│       - Binance: Order-driven limit order book
│       - Coinbase: Order-driven (market orders + limit orders)
│
├─ Price Discovery Comparison:
│   ├─ Order-Driven Strengths:
│   │   - Competitive: Multiple participants compete for spreads
│   │   - Transparent: Order book reveals supply/demand
│   │   - Efficient: Prices adjust instantly to new information
│   │   - Fair: No dealer markup distortion
│   ├─ Quote-Driven Strengths:
│   │   - Stable: Market maker provides continuous prices
│   │   - Certain: Always can execute at quoted prices
│   │   - Negotiable: Large trades can negotiate better prices
│   ├─ Speed of Discovery:
│   │   - Order-driven: Seconds (continuous matching)
│   │   - Quote-driven: Minutes (negotiation process)
│   │   - Winner: Order-driven for information incorporation
│   └─ Price Efficiency:
│       - Order-driven: Prices reflect supply/demand + information
│       - Quote-driven: Prices reflect dealer views + information
│       - Winner: Order-driven generally more efficient
│
├─ Order-Driven Market Regulation:
│   ├─ SEC Regulation NMS (2007):
│   │   - Order Protection Rule: Must execute at best available price
│   │   - Access Rule: Brokers must reach all venues
│   │   - Fragmentation: Multiple venues required to coordinate
│   │   - Result: Best price protection across all order-driven markets
│   ├─ Trade-Through Rules:
│   │   - Can't execute at worse price if better price available elsewhere
│   │   - Broker liability: Must find best execution
│   │   - System-wide coordination: ISGs (Intermarket Sweep Groups) coordinate
│   ├─ Pre-Trade Transparency:
│   │   - NBBO (National Best Bid Offer) public: Everyone sees best prices
│   │   - Bid-ask quotes: Real-time feeds to public
│   │   - Order book: Limited access (not full book to public)
│   ├─ Post-Trade Transparency:
│   │   - Trades reported: Trade price, size, time public within seconds
│   │   - SHO Regulation: Restrictions on short selling
│   ├─ Equity Listing Standards:
│   │   - Market cap minimums: Large-cap stocks more likely order-driven
│   │   - Reporting standards: Audited financials required
│   │   - Voting rights: One share = one vote
│   └─ Volatility Controls:
│       - Circuit breakers: Halt trading if market falls 7% (Level 1)
│       - Pause orders: Orders paused during volatility events
│       - Aim: Prevent cascading crashes in order-driven markets
│
├─ Market Microstructure Dynamics:
│   ├─ Order Imbalance:
│   │   - More buy orders than sell → prices rise (positive imbalance)
│   │   - More sell orders than buy → prices fall (negative imbalance)
│   │   - Information signal: Imbalance reveals informed vs uninformed flow
│   │   - Predictive: Order imbalance predicts near-term returns
│   ├─ Book Depth:
│   │   - Top-of-book: Best bid-ask (tight spread)
│   │   - Deeper levels: Worse prices, larger quantities
│   │   - Resilience: After large trade, book refills with new orders
│   │   - Speed: milliseconds to minutes depending on market
│   ├─ Price Impact:
│   │   - Linear: ΔP ≈ λ × Q (Kyle model)
│   │   - Square-root: ΔP ≈ σ × sqrt(Q/V) (Almgren-Chriss)
│   │   - Permanent: Information-driven, persists
│   │   - Temporary: Liquidity-driven, reverts in minutes
│   ├─ Volatility Sources:
│   │   - Information: Fundamental news drives volatility
│   │   - Order flow: Large orders cause temporary volatility
│   │   - Cascade: Stops triggering each other amplifies volatility
│   │   - Leverage: Margin calls force selling → volatility spike
│   └─ Stability:
│       - Liquid markets: Stable (tight spreads, fast recovery)
│       - Illiquid markets: Fragile (wide spreads, slow recovery)
│       - Crisis conditions: Can fail entirely (no liquidity)
│       - Historical: 2008 crisis, 2020 COVID crash show order-driven fragility
│
└─ Evolutionary Dynamics:
    ├─ Historical Trends:
    │   - 1950s-1990s: Floor trading (specialists, quote-driven)
    │   - 1990s-2000s: Electronic emergence (Nasdaq, ECNs)
    │   - 2000s-2010s: Full electronic (best execution rules, Reg NMS)
    │   - 2010s-present: HFT dominance (speed critical), fragmentation
    ├─ Future Evolution:
    │   - Blockchain order books: Decentralized matching
    │   - AI-driven MM: Algorithms replace human dealers
    │   - Retail dominance: Retail order flow exceeds institutional
    │   - Global integration: 24-hour continuous trading
    ├─ Emerging Markets:
    │   - Developing economies: Mix order-driven + dealer networks
    │   - Gradual shift: Increasing towards order-driven as development increases
    │   - Technology: Reduces costs to operate order-driven venues
    └─ Regulatory Evolution:
        - Pre-2007: Local regulation (each exchange)
        - Post-2007: Coordinated markets (Reg NMS)
        - Future: International harmonization (IOSCO standards)
```

**Interaction:** Order submitted → book updated → matching algorithm checks → executes against opposite side → confirmation sent → price discovered by supply/demand

## 5. Mini-Project
Simulate order book dynamics and price discovery in pure order-driven market:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from heapq import heappush, heappop

np.random.seed(42)

class OrderDrivenMarket:
    def __init__(self, initial_price=100.0):
        self.current_price = initial_price
        # Min-heaps (Python default): buy heap stores negative prices
        self.buy_orders = []  # (price, -time, quantity) - negative price for max-heap
        self.sell_orders = []  # (price, time, quantity)
        self.trade_log = []
        self.price_history = [initial_price]
        self.spread_history = []
        self.order_count = 0
        self.time = 0
        
    def get_best_bid(self):
        """Get best (highest) buy price"""
        while self.buy_orders and self.buy_orders[0][2] <= 0:
            heappop(self.buy_orders)
        if self.buy_orders:
            return -self.buy_orders[0][0]  # Negative price stored
        return 0
    
    def get_best_ask(self):
        """Get best (lowest) sell price"""
        while self.sell_orders and self.sell_orders[0][2] <= 0:
            heappop(self.sell_orders)
        if self.sell_orders:
            return self.sell_orders[0][0]
        return float('inf')
    
    def get_spread(self):
        """Get bid-ask spread"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid > 0 and ask < float('inf'):
            return ask - bid
        return None
    
    def add_limit_order(self, side, price, quantity):
        """Add limit order to book"""
        self.order_count += 1
        self.time += 1
        
        # Try to match with opposite side
        executed = 0
        
        if side == 'buy':
            # Try to match against sell orders
            while quantity > 0 and self.sell_orders:
                best_ask = self.sell_orders[0]
                if best_ask[0] <= price:  # Can match
                    # Match occurs
                    ask_price = best_ask[0]
                    ask_quantity = best_ask[2]
                    
                    fill = min(quantity, ask_quantity)
                    executed += fill
                    quantity -= fill
                    
                    # Update sell order
                    new_quantity = ask_quantity - fill
                    if new_quantity <= 0:
                        heappop(self.sell_orders)
                    else:
                        heappop(self.sell_orders)
                        heappush(self.sell_orders, (ask_price, best_ask[1], new_quantity))
                    
                    # Record trade
                    self.trade_log.append({
                        'time': self.time,
                        'price': ask_price,
                        'quantity': fill,
                        'side': 'buy',
                        'buyer_price': price,
                        'seller_price': ask_price
                    })
                else:
                    break
            
            # If not fully filled, add to buy book
            if quantity > 0:
                heappush(self.buy_orders, (-price, self.time, quantity))
        
        else:  # sell
            # Try to match against buy orders
            while quantity > 0 and self.buy_orders:
                best_bid = self.buy_orders[0]
                if -best_bid[0] >= price:  # Can match
                    # Match occurs
                    bid_price = -best_bid[0]
                    bid_quantity = best_bid[2]
                    
                    fill = min(quantity, bid_quantity)
                    executed += fill
                    quantity -= fill
                    
                    # Update buy order
                    new_quantity = bid_quantity - fill
                    if new_quantity <= 0:
                        heappop(self.buy_orders)
                    else:
                        heappop(self.buy_orders)
                        heappush(self.buy_orders, (best_bid[0], best_bid[1], new_quantity))
                    
                    # Record trade
                    self.trade_log.append({
                        'time': self.time,
                        'price': bid_price,
                        'quantity': fill,
                        'side': 'sell',
                        'buyer_price': bid_price,
                        'seller_price': price
                    })
                else:
                    break
            
            # If not fully filled, add to sell book
            if quantity > 0:
                heappush(self.sell_orders, (price, self.time, quantity))
        
        # Update price to last trade
        if self.trade_log and self.trade_log[-1]['time'] == self.time:
            self.current_price = self.trade_log[-1]['price']
        
        spread = self.get_spread()
        if spread:
            self.spread_history.append(spread)
        
        self.price_history.append(self.current_price)
        
        return {
            'executed': executed,
            'remaining': quantity,
            'partial_fill': quantity > 0 and executed > 0
        }
    
    def get_depth(self, n_levels=5):
        """Get order book depth"""
        bid_volume = 0
        ask_volume = 0
        
        # Copy heaps without modifying
        buy_copy = sorted(self.buy_orders, reverse=True)[:n_levels]
        sell_copy = sorted(self.sell_orders)[:n_levels]
        
        for order in buy_copy:
            bid_volume += order[2]
        for order in sell_copy:
            ask_volume += order[2]
        
        return {'bid': bid_volume, 'ask': ask_volume}

# Scenario 1: Random order arrival (order-driven discovery)
print("Scenario 1: Random Order Arrival (Order-Driven Price Discovery)")
print("=" * 80)

market1 = OrderDrivenMarket(initial_price=100.0)

# Simulate traders submitting random orders
order_types = []

for step in range(200):
    # Random side, price, quantity
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    
    # Price with noise around current
    price_noise = np.random.normal(0, 0.5)
    price = market1.current_price + price_noise
    price = max(90, min(110, price))  # Keep in reasonable range
    
    quantity = np.random.choice([100, 500, 1000, 5000])
    
    result = market1.add_limit_order(side, price, quantity)
    order_types.append(side)

print(f"Initial Price: $100.00")
print(f"Final Price: ${market1.current_price:.2f}")
print(f"Total Orders Submitted: {len(order_types)}")
print(f"Total Trades: {len(market1.trade_log)}")
print(f"Buy Orders: {sum(1 for o in order_types if o == 'buy')}")
print(f"Sell Orders: {sum(1 for o in order_types if o == 'sell')}")

if market1.spread_history:
    print(f"Average Spread: ${np.mean(market1.spread_history):.4f}")
    print(f"Min Spread: ${np.min(market1.spread_history):.4f}")
    print(f"Max Spread: ${np.max(market1.spread_history):.4f}")

# Scenario 2: Informed vs uninformed traders
print(f"\n\nScenario 2: Informed Trading (Price Moves to Fundamental)")
print("=" * 80)

market2 = OrderDrivenMarket(initial_price=100.0)
fundamental_value = 100.0

# Fundamental value follows random walk
for step in range(150):
    # Fundamental changes
    fundamental_value *= (1 + np.random.normal(0, 0.002))
    
    # Mix of informed and uninformed orders
    if np.random.random() < 0.3:
        # Informed order (toward fundamental)
        side = 'buy' if fundamental_value > market2.current_price else 'sell'
        price = market2.current_price + np.random.normal(0, 0.1)
    else:
        # Uninformed order (random)
        side = 'buy' if np.random.random() < 0.5 else 'sell'
        price = market2.current_price + np.random.normal(0, 0.5)
    
    price = max(90, min(110, price))
    quantity = np.random.choice([100, 500, 1000, 2000])
    
    market2.add_limit_order(side, price, quantity)

print(f"Initial Price: $100.00")
print(f"Final Price: ${market2.current_price:.2f}")
print(f"Fundamental Value: ${fundamental_value:.2f}")
print(f"Price Discovery Gap: ${abs(market2.current_price - fundamental_value):.2f}")
print(f"Total Trades: {len(market2.trade_log)}")

# Scenario 3: Large order impact
print(f"\n\nScenario 3: Large Order Impact (Market Impact)")
print("=" * 80)

market3 = OrderDrivenMarket(initial_price=100.0)

# Build initial book
for _ in range(50):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    price = 100.0 + np.random.normal(0, 0.3)
    quantity = np.random.choice([100, 500])
    market3.add_limit_order(side, price, quantity)

price_before = market3.current_price
spread_before = market3.get_spread()

# Large order impact
market3.add_limit_order('sell', 100.5, 50000)  # Large sell

price_after = market3.current_price
spread_after = market3.get_spread()

price_impact = price_before - price_after
spread_impact = (spread_after - spread_before) / spread_before * 100 if spread_before else 0

print(f"Price Before Large Order: ${price_before:.2f}")
print(f"Price After Large Order: ${price_after:.2f}")
print(f"Price Impact: ${price_impact:.2f}")
print(f"Spread Before: ${spread_before:.4f}" if spread_before else "N/A")
print(f"Spread After: ${spread_after:.4f}" if spread_after else "N/A")
print(f"Spread Change: {spread_impact:+.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price evolution (random orders)
times = range(len(market1.price_history))
axes[0, 0].plot(times, market1.price_history, linewidth=2, marker='o', markersize=2)
axes[0, 0].set_xlabel('Order Number')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Scenario 1: Price Discovery from Random Orders')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Price discovery (informed trading)
times2 = range(len(market2.price_history))
axes[0, 1].plot(times2, market2.price_history, linewidth=2, label='Market Price', marker='o', markersize=2)
axes[0, 1].axhline(y=fundamental_value, color='red', linestyle='--', linewidth=2, label='Fundamental Value')
axes[0, 1].set_xlabel('Order Number')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].set_title('Scenario 2: Informed Trading Moves Price to Fundamental')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Spread evolution
times = range(len(market1.spread_history))
axes[1, 0].plot(times, market1.spread_history, linewidth=2, color='green')
axes[1, 0].set_xlabel('Order Number')
axes[1, 0].set_ylabel('Bid-Ask Spread ($)')
axes[1, 0].set_title('Scenario 1: Spread Dynamics from Order Flow')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Trade price distribution
if market1.trade_log:
    trade_prices = [t['price'] for t in market1.trade_log]
    axes[1, 1].hist(trade_prices, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].set_xlabel('Trade Price ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Scenario 1: Distribution of Trade Prices')
    axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Summary Statistics
print(f"\n\nSummary Statistics:")
print("=" * 80)

if market1.trade_log:
    trade_prices = [t['price'] for t in market1.trade_log]
    print(f"\nScenario 1 (Random Orders):")
    print(f"  Price Range: ${min(market1.price_history):.2f} - ${max(market1.price_history):.2f}")
    print(f"  Trade Price Mean: ${np.mean(trade_prices):.2f}")
    print(f"  Trade Price Std Dev: ${np.std(trade_prices):.2f}")
    print(f"  Total Volume: {sum(t['quantity'] for t in market1.trade_log):,.0f} shares")

if market2.trade_log:
    trade_prices2 = [t['price'] for t in market2.trade_log]
    print(f"\nScenario 2 (Informed Trading):")
    print(f"  Price Convergence: Final gap = ${abs(market2.current_price - fundamental_value):.2f}")
    print(f"  Trade Count: {len(market2.trade_log)}")
    print(f"  Total Volume: {sum(t['quantity'] for t in market2.trade_log):,.0f} shares")
```

## 6. Challenge Round
Why do order-driven markets sometimes fail to discover prices efficiently (wider spreads and slower adjustment) compared to quote-driven dealer markets?

- **Thin order books**: Illiquid securities have sparse orders → large gaps between bid/ask → slow discovery process. Dealers in quote-driven would guarantee continuous prices
- **Order clustering**: Orders cluster at technical levels (support, resistance) → when price reaches level, many orders execute together → volatile price jumps → inefficient discovery
- **Information asymmetry**: Informed traders use limit orders to extract profits → uninformed traders pay wide spreads → prices adjust slowly as informed orders accumulate
- **Adverse selection**: Market order traders pay spreads → limits who participates → less participation → sparser book → wider spreads → slower discovery
- **Cascade risk**: Stop orders trigger together → feedback loop → price crash → discovery breaks down. Quote-driven MM absorbs such events without breaks

## 7. Key References
- [O'Hara (1995) - Market Microstructure Theory](https://www.amazon.com/Market-Microstructure-Theory-Maureen-OHara/dp/0631207619)
- [Harris (2003) - Trading and Exchanges - Chapter on Market Types](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708)
- [SEC Regulation NMS - Order Protection Rule](https://www.sec.gov/rules/final/34-51808.pdf)
- [Hasbrouck (2007) - Empirical Market Microstructure](https://www.amazon.com/Empirical-Market-Microstructure-Structure-Financial/dp/019530165X)

---
**Status:** Supply/demand price discovery | **Complements:** Quote-Driven Markets, Order Book Depth, Price Discovery, Market Efficiency
