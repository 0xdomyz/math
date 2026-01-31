# Level 2 & Level 3 Data

## 1. Concept Skeleton
**Definition:** Order book depth beyond best bid/ask; Level 2 shows top N price levels with aggregate size; Level 3 shows individual orders (exchange-dependent)  
**Purpose:** Analyze market depth; predict short-term price movement; optimize order placement; measure liquidity; detect hidden orders  
**Prerequisites:** Order book structure, limit orders, price-time priority, market depth, liquidity concepts

## 2. Comparative Framing
| Data Level | Content | Visibility | Use Case | Latency | Cost |
|------------|---------|------------|----------|---------|------|
| **Level 1** | Best bid/ask + last trade | Public, free | Basic trading | Low | Free |
| **Level 2** | Top 5-10 price levels (aggregated) | Subscription | Depth analysis, algos | Low-Medium | $50-500/mo |
| **Level 3** | Full order book (individual orders) | Restricted/exchange | HFT, research | Very low | $5K-50K/mo |
| **Market Depth** | Cumulative volume by price | Derived from L2/L3 | Liquidity measurement | N/A | Computed |
| **Time & Sales** | Trades only (no book) | Subscription | Transaction analysis | Low | $50-200/mo |

## 3. Examples + Counterexamples

**Level 1 (Basic):**  
```
AAPL: Bid $150.00 (800 shares) | Ask $150.01 (1,200 shares) | Last $150.01 × 100
```

**Level 2 (Depth):**  
```
AAPL Order Book:
Ask: $150.05 (5,000 shares) ← Level 5
Ask: $150.04 (3,500 shares) ← Level 4
Ask: $150.03 (2,000 shares) ← Level 3
Ask: $150.02 (1,500 shares) ← Level 2
Ask: $150.01 (1,200 shares) ← Best Ask (Level 1)
-----------------------------------
Bid: $150.00 (  800 shares) ← Best Bid (Level 1)
Bid: $149.99 (1,000 shares) ← Level 2
Bid: $149.98 (1,500 shares) ← Level 3
Bid: $149.97 (2,500 shares) ← Level 4
Bid: $149.96 (4,000 shares) ← Level 5
```

**Level 3 (Individual Orders - Exchange-Specific):**  
```
Ask $150.01:
  Order 1: 500 shares (MM1, 09:30:15.123)
  Order 2: 300 shares (MM2, 09:30:18.456)
  Order 3: 400 shares (MM3, 09:30:20.789)
Total: 1,200 shares (what Level 2 shows)
```

**Iceberg Order (Hidden):**  
```
Level 2 shows: $150.00 Bid with 100 shares
Reality (Level 3): 100 shares displayed + 10,000 shares hidden (iceberg)
```

## 4. Layer Breakdown
```
Level 2 & Level 3 Data Framework:
├─ Level 2 Market Depth:
│   ├─ Structure:
│   │   - Price levels: Top 5-20 on each side
│   │   - Aggregate size: Total shares at each price
│   │   - Number of orders: Count of orders (sometimes)
│   │   - Timestamp: When depth snapshot taken
│   │   - Updates: Stream of add/modify/delete events
│   ├─ Bid Side:
│   │   - Best bid: Highest buy price (inside market)
│   │   - Level 2-N: Progressively lower prices
│   │   - Depth: Total shares available at each level
│   │   - Interpretation: Support levels, buying interest
│   ├─ Ask Side:
│   │   - Best ask: Lowest sell price (inside market)
│   │   - Level 2-N: Progressively higher prices
│   │   - Depth: Total shares available at each level
│   │   - Interpretation: Resistance levels, selling pressure
│   ├─ Spread:
│   │   - Inside spread: Best ask - best bid
│   │   - Width: Indicates liquidity (tight = liquid)
│   │   - Locked: Bid = ask (rare, temporary)
│   │   - Crossed: Bid > ask (arbitrage, fleeting)
│   └─ Update Types:
│       - Add: New order posted at price level
│       - Modify: Order size/price changed
│       - Delete: Order cancelled or filled
│       - Refresh: Full book snapshot (periodic)
│       - Frequency: Millisecond updates (active stocks)
│
├─ Level 3 Order-Level Data:
│   ├─ Individual Order Details:
│   │   - Order ID: Unique identifier
│   │   - Price: Limit price
│   │   - Size: Number of shares (displayed)
│   │   - Hidden size: Iceberg reserve (if visible)
│   │   - Timestamp: When order placed
│   │   - Participant ID: Market maker/firm (anonymized)
│   │   - Flags: IOC, AON, hidden, pegged, etc.
│   ├─ Queue Position:
│   │   - FIFO ranking: Time priority within price
│   │   - Your position: How many shares ahead
│   │   - Execution probability: Function of position
│   │   - Value: Earlier position = higher fill likelihood
│   ├─ Order Book Events:
│   │   - Order entry: Limit order posted
│   │   - Order cancellation: Limit order removed
│   │   - Order modification: Price/size change (loses priority)
│   │   - Execution: Partial or full fill
│   │   - Sequencing: Strict time-ordered log
│   ├─ Participant Information:
│   │   - Market maker ID: Some exchanges provide
│   │   - Firm anonymization: Most exchanges hide
│   │   - Retail vs institutional: Not disclosed
│   │   - Detection: Infer from order patterns
│   └─ Access Restrictions:
│       - Exchange members: Full access (expensive)
│       - Academic: LOBSTER database (historical)
│       - Public: Not available (proprietary)
│       - Regulatory: FINRA/SEC have access
│
├─ Data Providers & Feeds:
│   ├─ Exchange Direct Feeds:
│   │   - NASDAQ TotalView: Level 2 (top 5)
│   │   - NYSE OpenBook: Level 2 (full book)
│   │   - BATS Depth: Level 2 (full book)
│   │   - CME MDP 3.0: Futures depth (10 levels)
│   │   - Latency: Microseconds (colocation)
│   │   - Cost: $1,000-$10,000/month per exchange
│   ├─ Consolidated Feeds:
│   │   - OPRA: Options depth (all exchanges)
│   │   - CQS: Consolidated Quote System (Level 1)
│   │   - No consolidated Level 2: Must subscribe per venue
│   │   - Fragmentation: 13+ equity exchanges in US
│   ├─ Retail Brokers:
│   │   - Simplified L2: Top 3-5 levels, delayed
│   │   - Cost: Free with account (marketing)
│   │   - Latency: Seconds (adequate for retail)
│   │   - Examples: TD Ameritrade, Interactive Brokers
│   ├─ Data Vendors:
│   │   - Bloomberg Terminal: Multi-exchange L2
│   │   - Refinitiv: Global depth data
│   │   - IEX: Free Level 2 (IEX exchange only)
│   │   - Polygon.io: API-based L2 (affordable)
│   └─ Academic Sources:
│       - LOBSTER: Reconstruct NASDAQ L3 (historical)
│       - WRDS: Some depth data via TAQ
│       - Cryptocurrency: Many exchanges provide free L2 via API
│       - Limitations: Historical only, not real-time
│
├─ Analysis Techniques:
│   ├─ Order Book Imbalance:
│   │   - Formula: (Bid Volume - Ask Volume) / (Bid + Ask)
│   │   - Interpretation: Positive = buy pressure, negative = sell
│   │   - Predictive power: Short-term (seconds to minutes)
│   │   - Limitation: Can be spoofed (fake depth)
│   │   - Research: Cao et al (2009) show predictive ability
│   ├─ Liquidity Measurement:
│   │   - Depth at best: Shares at inside bid/ask
│   │   - Cumulative depth: Total within X cents of mid
│   │   - Volume at price: Depth profile visualization
│   │   - Resilience: Speed of depth replenishment
│   │   - Application: Estimate market impact
│   ├─ Support/Resistance Levels:
│   │   - Large depth clusters: Potential barriers
│   │   - Example: 50K shares at $150.00 = strong support
│   │   - Breakout: If level consumed, momentum continues
│   │   - False signals: Depth can be pulled (spoofing)
│   ├─ Spread Dynamics:
│   │   - Spread widening: Decreased liquidity, uncertainty
│   │   - Spread tightening: Increased competition, liquidity
│   │   - Intraday pattern: Wider at open/close
│   │   - Event-driven: News widens spread (risk aversion)
│   ├─ Quote Stuffing Detection:
│   │   - Rapid add/cancel: >1,000 orders/second
│   │   - No fills: Orders cancelled before execution
│   │   - Purpose: Slow competitors' systems
│   │   - Detection: Message rate analysis
│   ├─ Hidden Liquidity:
│   │   - Iceberg orders: Display 100, hide 10,000
│   │   - Dark pools: No displayed depth
│   │   - Detection: Trades > displayed size
│   │   - Impact: Underestimate true liquidity
│   └─ Price Impact Forecasting:
│       - Model: Impact = f(order size, depth)
│       - Square-root law: Impact ∝ √(size / daily volume)
│       - Refinement: Use depth at relevant price levels
│       - Application: Optimal execution algorithms
│
├─ Order Book Reconstruction (Level 3):
│   ├─ From Message Feed:
│   │   - Input: Stream of add/modify/delete messages
│   │   - Process: Maintain sorted data structure (heap, tree)
│   │   - Output: Snapshot at any timestamp
│   │   - Complexity: O(log n) per update
│   ├─ Data Structures:
│   │   - Price levels: Hash map (price → queue)
│   │   - Order queue: Linked list (FIFO within price)
│   │   - Best bid/ask: Priority queue or pointers
│   │   - Efficient: Handle millions of updates/second
│   ├─ Challenges:
│   │   - Out-of-order messages: Network reordering
│   │   - Sequence gaps: Packet loss, need recovery
│   │   - State consistency: Periodic snapshots for validation
│   │   - Storage: Terabytes per day (active stocks)
│   ├─ LOBSTER Database:
│   │   - Provider: Academic (free for research)
│   │   - Coverage: NASDAQ stocks (2007-present)
│   │   - Reconstruction: Full L3 order book from messages
│   │   - Files: Message file + order book file
│   │   - Use: Research on order book dynamics
│   └─ Real-Time Processing:
│       - Streaming: Kafka, Flink for high throughput
│       - In-memory: Redis, kdb+ for fast queries
│       - Latency: Microseconds to milliseconds
│       - Scale: Multi-core, distributed systems
│
├─ Practical Applications:
│   ├─ Algorithmic Trading:
│   │   - Order placement: Find depth pockets
│   │   - Execution: Split orders to minimize impact
│   │   - Market making: Quote around depth
│   │   - Arbitrage: Detect depth imbalances across venues
│   ├─ Optimal Execution:
│   │   - VWAP: Participate proportional to depth
│   │   - POV: Percent-of-volume strategies
│   │   - Adaptive: Adjust to real-time depth changes
│   │   - Risk: Balance impact vs timing risk
│   ├─ Short-Term Prediction:
│   │   - Direction: Order imbalance predicts next tick
│   │   - Volatility: Shallow depth → higher volatility
│   │   - Liquidity: Deep market → lower slippage
│   │   - Horizon: Seconds to minutes (not long-term)
│   ├─ Market Making:
│   │   - Quote skewing: Lean inventory toward mean-revert
│   │   - Depth management: Adjust displayed size
│   │   - Adverse selection: Avoid when depth disappears
│   │   - Penny jumping: Beat competitors by 1 tick
│   ├─ Risk Management:
│   │   - Liquidation analysis: Can you exit position?
│   │   - Depth shock: Sudden depth withdrawal (risk)
│   │   - Stress testing: What if depth halves?
│   │   - Position sizing: Limit to available liquidity
│   └─ Academic Research:
│       - Price discovery: Where does information enter?
│       - Market microstructure: Order flow dynamics
│       - Liquidity provision: MM behavior
│       - High-frequency trading: HFT strategies
│
├─ Limitations & Challenges:
│   ├─ Iceberg Orders:
│   │   - Hidden portion: Not visible in Level 2/3
│   │   - Discovery: Only after partial fills
│   │   - Impact: Underestimate true depth
│   │   - Prevalence: ~10-20% of institutional orders
│   ├─ Dark Pools:
│   │   - No pre-trade transparency: Depth unknown
│   │   - Volume: ~15-20% of US equity volume
│   │   - Post-trade reporting: Only after execution
│   │   - Challenge: Fragmented liquidity
│   ├─ Spoofing/Layering:
│   │   - Fake depth: Orders placed without intent
│   │   - Manipulation: Induce others to trade
│   │   - Cancellation: Pulled before execution
│   │   - Detection: Difficult in real-time
│   ├─ Latency Arbitrage:
│   │   - Stale quotes: Slow feed shows old depth
│   │   - Fast traders: Exploit stale information
│   │   - Victim: Slow traders pick off
│   │   - Arms race: Speed advantage critical
│   ├─ Flash Crashes:
│   │   - Depth evaporation: All orders pulled simultaneously
│   │   - Cause: Automated algorithms, feedback loops
│   │   - Example: 2010 Flash Crash (Dow -9% in minutes)
│   │   - Recovery: Depth returns after halt
│   └─ Data Quality:
│       - Message loss: Network packet drops
│       - Timestamp precision: Microsecond accuracy required
│       - Cross-venue: No consolidated book (US equities)
│       - Cost: Expensive for retail traders
│
└─ Regulatory Aspects:
    ├─ Transparency Requirements:
    │   - Reg NMS: Quote display obligations
    │   - Best execution: Must route to best price
    │   - NBBO: Consolidated best bid/offer (Level 1)
    │   - Depth: No consolidated Level 2 requirement (yet)
    ├─ Market Access Rule:
    │   - SEC Rule 15c3-5: Pre-trade risk controls
    │   - Order limits: Size, price checks
    │   - Kill switches: Emergency halt capability
    │   - Compliance: Firms must demonstrate controls
    ├─ Spoofing Prohibition:
    │   - Dodd-Frank Act: Illegal to manipulate via orders
    │   - Enforcement: CFTC, SEC prosecutions
    │   - Detection: Surveillance algorithms
    │   - Penalties: Criminal charges, fines
    └─ Market Data Fees:
        - Controversy: Exchanges charge high fees
        - Debate: Should depth data be free/cheap?
        - Alternatives: IEX offers free Level 2
        - Policy: Ongoing regulatory review
```

**Interaction:** Trader views Level 2 → sees 5,000 shares at $150.00 → decides to buy 1,000 → order routed → fills from multiple limit orders at that level → depth reduces to 4,000 → new orders arrive, replenish depth

## 5. Mini-Project
Simulate and analyze order book depth dynamics:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

np.random.seed(42)

class OrderBook:
    def __init__(self):
        # Price -> [(timestamp, order_id, size), ...]
        self.bids = defaultdict(list)  # Buy orders
        self.asks = defaultdict(list)  # Sell orders
        self.order_id_counter = 0
        
        # For priority queue (efficient best bid/ask)
        self.bid_prices = []  # Max heap (negative for max)
        self.ask_prices = []  # Min heap
        
    def add_order(self, side, price, size, timestamp):
        """Add limit order to book"""
        self.order_id_counter += 1
        order_id = self.order_id_counter
        
        if side == 'bid':
            self.bids[price].append((timestamp, order_id, size))
            heapq.heappush(self.bid_prices, -price)  # Negative for max heap
        else:  # ask
            self.asks[price].append((timestamp, order_id, size))
            heapq.heappush(self.ask_prices, price)
        
        return order_id
    
    def get_best_bid_ask(self):
        """Get inside market"""
        # Clean up empty prices
        while self.bid_prices and not self.bids[-self.bid_prices[0]]:
            heapq.heappop(self.bid_prices)
        while self.ask_prices and not self.asks[self.ask_prices[0]]:
            heapq.heappop(self.ask_prices)
        
        best_bid = -self.bid_prices[0] if self.bid_prices else None
        best_ask = self.ask_prices[0] if self.ask_prices else None
        
        return best_bid, best_ask
    
    def get_depth(self, side, levels=5):
        """Get Level 2 depth (top N levels)"""
        if side == 'bid':
            prices = sorted([p for p in self.bids.keys() if self.bids[p]], reverse=True)[:levels]
            depth = [(p, sum(order[2] for order in self.bids[p])) for p in prices]
        else:
            prices = sorted([p for p in self.asks.keys() if self.asks[p]])[:levels]
            depth = [(p, sum(order[2] for order in self.asks[p])) for p in prices]
        
        return depth
    
    def execute_market_order(self, side, size, timestamp):
        """Execute market order, consuming depth"""
        remaining = size
        executed = []
        
        if side == 'buy':  # Buy market order, consume asks
            prices = sorted([p for p in self.asks.keys() if self.asks[p]])
            for price in prices:
                if remaining == 0:
                    break
                
                while self.asks[price] and remaining > 0:
                    order = self.asks[price][0]
                    order_size = order[2]
                    
                    if order_size <= remaining:
                        # Full fill
                        self.asks[price].pop(0)
                        executed.append((price, order_size))
                        remaining -= order_size
                    else:
                        # Partial fill
                        self.asks[price][0] = (order[0], order[1], order_size - remaining)
                        executed.append((price, remaining))
                        remaining = 0
        
        else:  # Sell market order, consume bids
            prices = sorted([p for p in self.bids.keys() if self.bids[p]], reverse=True)
            for price in prices:
                if remaining == 0:
                    break
                
                while self.bids[price] and remaining > 0:
                    order = self.bids[price][0]
                    order_size = order[2]
                    
                    if order_size <= remaining:
                        self.bids[price].pop(0)
                        executed.append((price, order_size))
                        remaining -= order_size
                    else:
                        self.bids[price][0] = (order[0], order[1], order_size - remaining)
                        executed.append((price, remaining))
                        remaining = 0
        
        return executed, remaining
    
    def calculate_imbalance(self, levels=5):
        """Calculate order book imbalance"""
        bid_depth = self.get_depth('bid', levels)
        ask_depth = self.get_depth('ask', levels)
        
        bid_volume = sum(size for _, size in bid_depth)
        ask_volume = sum(size for _, size in ask_depth)
        
        total = bid_volume + ask_volume
        if total > 0:
            imbalance = (bid_volume - ask_volume) / total
        else:
            imbalance = 0
        
        return imbalance, bid_volume, ask_volume

# Scenario 1: Build order book with depth
print("Scenario 1: Order Book Construction (Level 2)")
print("=" * 80)

book = OrderBook()

# Initialize book with depth
mid_price = 100.0
spread = 0.01

# Add bid orders (5 levels)
for level in range(5):
    price = mid_price - spread/2 - level * 0.01
    size = (level + 1) * 500  # Increasing depth away from mid
    book.add_order('bid', price, size, 0)

# Add ask orders (5 levels)
for level in range(5):
    price = mid_price + spread/2 + level * 0.01
    size = (level + 1) * 500
    book.add_order('ask', price, size, 0)

# Display Level 2
best_bid, best_ask = book.get_best_bid_ask()
print(f"Inside Market:")
print(f"  Best Bid: ${best_bid:.2f} | Best Ask: ${best_ask:.2f}")
print(f"  Spread: ${best_ask - best_bid:.3f} ({(best_ask-best_bid)/mid_price*10000:.2f} bps)")

print(f"\nLevel 2 Depth (5 levels):")
print(f"{'ASK':>10} {'Price':>10} {'Size':>10}")
print("-" * 30)
ask_depth = book.get_depth('ask', 5)
for price, size in reversed(ask_depth):
    print(f"{'':>10} ${price:>9.2f} {size:>10,}")

print(f"{'':>10} {'-'*9:>10} {'-'*10:>10}")

bid_depth = book.get_depth('bid', 5)
for price, size in bid_depth:
    print(f"{'BID':>10} ${price:>9.2f} {size:>10,}")

# Scenario 2: Order book imbalance
print(f"\n\nScenario 2: Order Book Imbalance")
print("=" * 80)

imbalance, bid_vol, ask_vol = book.calculate_imbalance(levels=5)

print(f"Bid volume (top 5): {bid_vol:,} shares")
print(f"Ask volume (top 5): {ask_vol:,} shares")
print(f"Order imbalance: {imbalance:+.2%}")
print(f"\nInterpretation:")
if abs(imbalance) < 0.1:
    print(f"  Balanced book → neutral short-term outlook")
elif imbalance > 0:
    print(f"  Buy pressure → slight bullish bias")
else:
    print(f"  Sell pressure → slight bearish bias")

# Scenario 3: Market order impact on depth
print(f"\n\nScenario 3: Market Order Execution")
print("=" * 80)

# Execute buy market order
order_size = 1500
executed, remaining = book.execute_market_order('buy', order_size, 1)

print(f"Buy market order: {order_size:,} shares")
print(f"Execution breakdown:")
total_cost = 0
total_filled = 0
for price, size in executed:
    print(f"  {size:>6,} shares @ ${price:.2f}")
    total_cost += price * size
    total_filled += size

avg_price = total_cost / total_filled if total_filled > 0 else 0
print(f"\nTotal filled: {total_filled:,} shares")
print(f"Average price: ${avg_price:.4f}")
print(f"Unfilled: {remaining:,} shares")

# New state
new_best_bid, new_best_ask = book.get_best_bid_ask()
print(f"\nNew inside market:")
print(f"  Best Bid: ${new_best_bid:.2f} | Best Ask: ${new_best_ask:.2f}")
print(f"  Price impact: ${new_best_ask - best_ask:+.4f}")

# Scenario 4: Depth dynamics over time
print(f"\n\nScenario 4: Depth Evolution Simulation")
print("=" * 80)

# Rebuild book and simulate
book2 = OrderBook()
timestamps = []
bid_volumes = []
ask_volumes = []
imbalances = []
spreads = []

# Initial state
for level in range(5):
    book2.add_order('bid', 99.99 - level * 0.01, (level + 1) * 500, 0)
    book2.add_order('ask', 100.00 + level * 0.01, (level + 1) * 500, 0)

for t in range(100):
    # Random events
    event_type = np.random.choice(['add_bid', 'add_ask', 'market_buy', 'market_sell'], 
                                  p=[0.4, 0.4, 0.1, 0.1])
    
    best_bid, best_ask = book2.get_best_bid_ask()
    
    if event_type == 'add_bid':
        price = best_bid - np.random.randint(0, 3) * 0.01
        size = np.random.randint(100, 1000)
        book2.add_order('bid', price, size, t)
    
    elif event_type == 'add_ask':
        price = best_ask + np.random.randint(0, 3) * 0.01
        size = np.random.randint(100, 1000)
        book2.add_order('ask', price, size, t)
    
    elif event_type == 'market_buy':
        size = np.random.randint(100, 500)
        book2.execute_market_order('buy', size, t)
    
    else:  # market_sell
        size = np.random.randint(100, 500)
        book2.execute_market_order('sell', size, t)
    
    # Record state
    imb, bid_vol, ask_vol = book2.calculate_imbalance(levels=5)
    best_bid, best_ask = book2.get_best_bid_ask()
    
    timestamps.append(t)
    bid_volumes.append(bid_vol)
    ask_volumes.append(ask_vol)
    imbalances.append(imb)
    spreads.append(best_ask - best_bid if best_bid and best_ask else 0)

print(f"Simulated {len(timestamps)} time steps")
print(f"Average bid depth: {np.mean(bid_volumes):,.0f} shares")
print(f"Average ask depth: {np.mean(ask_volumes):,.0f} shares")
print(f"Average imbalance: {np.mean(imbalances):+.2%}")
print(f"Average spread: ${np.mean(spreads):.4f}")

# Scenario 5: Depth-based liquidity analysis
print(f"\n\nScenario 5: Liquidity Cost Estimation")
print("=" * 80)

# Rebuild fresh book
book3 = OrderBook()
for level in range(10):
    book3.add_order('bid', 99.95 - level * 0.01, (level + 1) * 1000, 0)
    book3.add_order('ask', 100.00 + level * 0.01, (level + 1) * 1000, 0)

# Estimate cost for different order sizes
order_sizes = [1000, 5000, 10000, 25000]

print(f"{'Order Size':>12} | {'Avg Price':>10} | {'Impact':>10} | {'Cost (bps)':>12}")
print("-" * 50)

for size in order_sizes:
    # Create copy of book
    book_copy = OrderBook()
    book_copy.bids = book3.bids.copy()
    book_copy.asks = book3.asks.copy()
    book_copy.bid_prices = book3.bid_prices.copy()
    book_copy.ask_prices = book3.ask_prices.copy()
    
    initial_mid = (book_copy.get_best_bid_ask()[0] + book_copy.get_best_bid_ask()[1]) / 2
    
    executed, remaining = book_copy.execute_market_order('buy', size, 0)
    
    if executed:
        total_cost = sum(p * s for p, s in executed)
        total_filled = sum(s for _, s in executed)
        avg_price = total_cost / total_filled
        
        impact = avg_price - initial_mid
        cost_bps = (impact / initial_mid) * 10000
        
        print(f"{size:>12,} | ${avg_price:>9.4f} | ${impact:>9.4f} | {cost_bps:>11.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Order book depth visualization
bid_prices_plot = [p for p, _ in bid_depth]
bid_sizes_plot = [s for _, s in bid_depth]
ask_prices_plot = [p for p, _ in ask_depth]
ask_sizes_plot = [s for _, s in ask_depth]

axes[0, 0].barh(bid_prices_plot, bid_sizes_plot, height=0.005, color='green', alpha=0.7, label='Bids')
axes[0, 0].barh(ask_prices_plot, ask_sizes_plot, height=0.005, color='red', alpha=0.7, label='Asks')
axes[0, 0].set_xlabel('Size (shares)')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Scenario 1: Order Book Depth (Level 2)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='x')

# Plot 2: Order imbalance over time
axes[0, 1].plot(timestamps, imbalances, linewidth=2, color='blue')
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].fill_between(timestamps, 0, imbalances, 
                        where=np.array(imbalances) > 0, 
                        color='green', alpha=0.3, label='Buy Pressure')
axes[0, 1].fill_between(timestamps, 0, imbalances, 
                        where=np.array(imbalances) < 0, 
                        color='red', alpha=0.3, label='Sell Pressure')
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Order Imbalance')
axes[0, 1].set_title('Scenario 4: Order Book Imbalance Dynamics')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Bid vs Ask depth evolution
axes[1, 0].plot(timestamps, bid_volumes, linewidth=2, color='green', label='Bid Depth', alpha=0.7)
axes[1, 0].plot(timestamps, ask_volumes, linewidth=2, color='red', label='Ask Depth', alpha=0.7)
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Total Volume (shares)')
axes[1, 0].set_title('Scenario 4: Bid/Ask Depth Evolution')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Spread dynamics
axes[1, 1].plot(timestamps, np.array(spreads) * 10000, linewidth=2, color='purple')
axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('Spread (basis points)')
axes[1, 1].set_title('Scenario 4: Bid-Ask Spread Over Time')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n\nSummary:")
print("=" * 80)
print(f"Level 2 data reveals market depth beyond best bid/ask (top 5-10 levels)")
print(f"Order book imbalance predicts short-term price direction")
print(f"Market orders consume depth, causing temporary price impact")
print(f"Depth evolves dynamically with new orders, cancellations, executions")
print(f"Liquidity cost scales nonlinearly with order size (depth depletion)")
```

## 6. Challenge Round
If Level 2 data shows deep liquidity (10,000+ shares at each level), why can't large institutional orders simply execute at displayed prices without significant impact?

- **Phantom liquidity (spoofing)**: Displayed depth pulled before execution → order sees 10K, submits 5K, depth vanishes → only 100 shares actually fill → manipulation (illegal but happens)
- **Iceberg orders (hidden size)**: Level 2 shows 1,000 shares, but 50,000 hidden behind → can't estimate true depth → underestimate available liquidity → depth depletes faster than expected
- **Adverse selection**: Deep depth attracts large orders → market makers detect institutional flow → pull quotes milliseconds before execution → "toxic flow" avoidance → depth evaporates when needed most
- **Cross-venue fragmentation**: Level 2 shows single exchange → but 13+ US exchanges → total liquidity dispersed → can't aggregate easily → routing delays cause slippage
- **Information leakage**: Large order revealed → HFTs front-run → buy ahead, force price up → institutional order pays higher price → "stepping ahead" phenomenon → depth nominal, not accessible at stated price

## 7. Key References
- [Harris (2003) - Trading and Exchanges: Market Microstructure for Practitioners](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708)
- [Cont, Stoikov & Talreja (2010) - A Stochastic Model for Order Book Dynamics](https://arxiv.org/abs/1003.3796)
- [Cao, Hansch & Wang (2009) - The Information Content of an Open Limit-Order Book](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2009.01469.x)
- [LOBSTER - Limit Order Book System](https://lobsterdata.com/)

---
**Status:** Order book depth beyond Level 1 | **Complements:** Order Book Dynamics, Market Depth, Liquidity Measurement
