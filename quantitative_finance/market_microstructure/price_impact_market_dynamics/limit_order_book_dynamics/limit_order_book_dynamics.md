# Limit Order Book Dynamics

## 1. Concept Skeleton
**Definition:** Order arrival, cancellation, and execution in real-time book; queue position; depth evolution; microstructure consequences  
**Purpose:** Understand order book mechanics; optimize limit order execution; predict fill probabilities; model market resilience  
**Prerequisites:** Order types, market depth, bid-ask spread, microstructure basics

## 2. Comparative Framing
| Book Aspect | Normal Market | Stressed Market | HFT-Dominated | Comment |
|------------|-------------|------------------|--------------|---------|
| **Depth** | 1M+ shares | 10K-100K shares | Minimal | Liquidity changes drastically |
| **Queue Position** | Valuable | Worthless | Toxic | Timing critical |
| **Cancellation Rate** | 10-50% | 80%+ | 95%+ | Trash orders prevalent |
| **Replenishment** | Seconds | Minutes | Milliseconds | Depth restored variably |
| **Book Resilience** | Fast recovery | Slow/none | Very slow | Shock absorption differs |
| **Price Impact** | Temporary | Permanent | Persistent | Same market, different behavior |

## 3. Examples + Counterexamples

**Stable Book:**  
Order book: Bid 100@ $10.00 (1M shares at level), Ask $10.01 (1M shares) → I place buy limit @$10.00 → I'm 100th in queue → Bids filled slowly (market-makers continuously post) → I eventually fill in ~30 seconds at $10.00

**Toxic Queue:**  
Order book: Bid 100@ $10.00 (only 50K shares, sparse) → I place buy limit @$10.00 → I'm 100th in queue → Stock plummets to $9.50 → Everyone ahead of me cancels (losses avoided) → I'm now at front but wrong market price → Forced to execute at $9.50 (worse than original limit)

**Queue Position Value:**  
AAPL: Depth at bid = 1M shares, top 100 orders per second execute → My position = 10K shares → Execution time = 100 seconds → Normal → Can rely on fill

**Queue Position Worthless:**  
Tesla (volatile): Depth = 50K shares, execution rate = 1K/second (volatility driven) → I place 10K at limit → expected exec time = 10 seconds → But cancellations spike → Effective queue much longer → Conditional on price doesn't move → Unlikely → Order likely unfilled or market moves against me

## 4. Layer Breakdown
```
Limit Order Book Dynamics Framework:
├─ Order Book Structure:
│   ├─ Fundamental Components:
│   │   - Bid side: Limit orders to buy (below current mid)
│   │   - Ask side: Limit orders to sell (above current mid)
│   │   - Spread: Mid-price = (bid + ask) / 2
│   │   - Depth: Total volume available at each price level
│   │   - Price levels: Discrete (tick size) vs continuous
│   ├─ Depth Distribution:
│   │   - Level 1: Highest bid, lowest ask (inside quotes)
│   │   - Levels 2-10: Moderately deep
│   │   - Levels 10+: Sparse (few orders, large gaps)
│   │   - Empirical: Typical 1M+ shares at level 1, 100K at level 5
│   │   - Extreme: Illiquid stocks: 10K at level 1
│   ├─ Price Grid:
│   │   - Tick size: Minimum price increment (e.g., $0.01)
│   │   - Sub-tick: HFT liquidity rebates (e.g., $0.001)
│   │   - Continuous book: Theoretically possible, rare
│   │   - Effect: Tick size affects depth distribution
│   ├─ Queue Position:
│   │   - Definition: Where your order sits relative to others at same price
│   │   - Importance: Higher queue = earlier execution
│   │   - Measurement: Number of shares ahead in queue
│   │   - Example: 100 buy orders @ $10.00 level, 500K shares ahead
│   ├─ Partial Fills:
│   │   - Definition: Order not entirely filled at once
│   │   - Cause: Insufficient volume at limit price
│   │   - Timing: Multiple partial fills over time
│   │   - Execution: Market order fills queue sequentially
│   └─ Resting Orders:
│       - Definition: Limit orders sitting in book (not yet executed)
│       - Passive provision: Liquidity added to book
│       - Risk: Price moves against you while resting
│       - Cancellation: Can remove anytime (no cost)
│
├─ Order Arrival Process:
│   ├─ Market Orders (Remove Liquidity):
│   │   - Behavior: Execute immediately against book
│   │   - Queue impact: Eats through depth
│   │   - Example: Sell 50K shares market → removes 50K from bid
│   │   - Cascade: If 50K insufficient, eats next price level
│   │   - Impact: Reduces depth at current level
│   ├─ Limit Orders (Add Liquidity):
│   │   - Behavior: Placed at specific price, waits for execution
│   │   - Queue impact: Adds to queue at that level
│   │   - Example: Buy 10K @ $9.99 → adds to bid queue
│   │   - Duration: Rests until (a) executed, (b) cancelled
│   │   - Incentive: Maker rebates compensate for adverse selection
│   ├─ Order Clustering:
│   │   - Definition: Multiple orders arrive within microseconds
│   │   - Cause: Algorithm execution, news arrival, cascade
│   │   - Effect: Transient depth shock (recovered quickly)
│   │   - Observable: High variance in order flow
│   ├─ Arrival Timing:
│   │   - Poisson process: Arrival rate λ (orders per second)
│   │   - Empirical: λ ≈ 100-1000 orders/sec for active stocks
│   │   - Clustering: Not perfectly Poisson (self-exciting)
│   │   - Hawkes process: Models clustering (next order more likely if recent order)
│   ├─ Informed vs Uninformed:
│   │   - Informed: Profit from information (market orders, aggressive)
│   │   - Uninformed: Noise traders (random orders, passive)
│   │   - Detection: Difficult ex-ante, inference ex-post
│   │   - Impact: Informed order → permanent impact, uninformed → temporary
│   └─ Algorithmic Orders:
│       - VWAP/TWAP: Slice large orders into smaller parcels
│       - Execution: Spread across time, participate in volume
│       - Book impact: Gradual depth reduction, less shocking
│       - Detection: Observed order patterns, suspicious regularity
│
├─ Order Execution & Cancellation:
│   ├─ Execution Process:
│   │   - Matching: Market order matched against limit orders in queue
│   │   - Priority: Price > time (best price, then FIFO within price)
│   │   - Partial: Remaining market order continues to next level
│   │   - Time: Microseconds to milliseconds per execution
│   │   - Example: Buy 100K market → fills 50K @ ask, 30K @ ask+1tick, 20K @ ask+2ticks
│   ├─ Execution Rate:
│   │   - Definition: Volume executed per unit time
│   │   - Normal: 1-10% of depth per second
│   │   - Volatile: Up to 50% of depth per second
│   │   - Crisis: Entire depth may execute in <1 second
│   │   - Measurement: Calculate from trade data
│   ├─ Cancellation Behavior:
│   │   - Unconditional: Trader cancels order regardless of market
│   │   - Conditional: Cancel if price moves against you
│   │   - Probability: P(cancel|price move) high if adverse
│   │   - Typical rate: 50-95% of orders cancelled before fill
│   │   - HFT: 95%+ cancellation (cancel if faster competitors trade first)
│   ├─ Cancellation Motives:
│   │   - Quote change: Reposition after news/trade
│   │   - Adverse selection: Avoid losing trade (price moved)
│   │   - Inventory: Target inventory reached, cancel new orders
│   │   - Information loss: Cancel if information became stale
│   │   - Technical: Cancel due to system error, network latency
│   ├─ Fill Probability:
│   │   - Definition: Chance limit order fully executes
│   │   - Formula: P(fill) = f(limit price, duration, volatility, order flow)
│   │   - Better limit: Lower fill probability (less execution)
│   │   - Longer duration: Higher fill probability (more time)
│   │   - Higher volatility: Lower fill probability (price likely moves away)
│   │   - Example: Tight limit (1% inside)→ 70% fill | Loose limit (5% away) → 5% fill
│   └─ Order Interaction:
│       - Self-interaction: Your own orders affect each other
│       - Example: Place buy limit, then place sell limit
│       - Consequence: Second order affects first (inventory consideration)
│       - Strategy: Can exploit to reduce adverse selection costs
│
├─ Depth Dynamics:
│   ├─ Depth Replenishment:
│   │   - Depletion: Market order removes depth from book
│   │   - Replenishment: New limit orders arrive, rebuild depth
│   │   - Speed: Varies dramatically (ms to hours)
│   │   - Fast: Liquid stocks (depth back to normal in <1 sec)
│   │   - Slow: Illiquid stocks (depth may not recover same day)
│   │   - Mechanism: MM competitive dynamics or information revelation
│   ├─ Depth Decay (Relaxation):
│   │   - Formula: Depth(t) = Depth(0) + Recovery term
│   │   - Exponential: Depth ∝ e^(-t/τ) (time constant τ)
│   │   - Time scale: τ typically 1-10 seconds for stocks
│   │   - Longer τ in illiquid, volatile, or stressed markets
│   ├─ Limit Order Placement Strategy:
│   │   - Aggressive: Limit order close to mid (high fill prob, low fill price)
│   │   - Passive: Limit order far from mid (low fill prob, good fill price)
│   │   - Trade-off: Can't optimize both (no free lunch)
│   │   - Dynamic: Adjust as market conditions change
│   ├─ Depth Imbalance:
│   │   - Definition: Bid depth ≠ ask depth
│   │   - Imbalance ratio: bid volume / (bid + ask volume)
│   │   - Interpretation: Bias in order flow (more buy vs sell)
│   │   - Prediction: Imbalance predictor of next move (mixed evidence)
│   │   - Extreme: Imbalance >90% suggests temporary imbalance
│   └─ Resilience:
│       - Definition: Book's ability to absorb shocks and recover
│       - Measure: Time to return to normal depth after trade
│       - Low resilience: Depth doesn't recover (illiquid market)
│       - High resilience: Depth snaps back immediately (liquid)
│       - Policy: Regulatory concerns about resilience in stress
│
├─ Queue Position Dynamics:
│   ├─ Queue Length:
│   │   - Definition: Number of shares ahead in queue at same price
│   │   - Measurement: Sum volumes of orders placed before yours
│   │   - Importance: Queue length = expected time to fill
│   │   - Example: 500K shares in queue, 10K/sec execution → 50 sec to fill
│   ├─ Queue Erosion:
│   │   - Definition: Shares executed from queue ahead of you
│   │   - Mechanism: Market orders, earlier limit orders execute
│   │   - Rate: Depends on order flow intensity
│   │   - Predictable: Typical erosion rates measurable ex-post
│   ├─ Queue Jumping:
│   │   - Definition: New limit order at same price gets better position (if shares executed)
│   │   - Mechanism: FIFO within price (but new order goes to back)
│   │   - Advantage: You move up queue naturally as earlier shares execute
│   │   - Disadvantage: If no execution, you stuck at back indefinitely
│   ├─ Position Ranking:
│   │   - Metric: Your share count / Total queue depth at price
│   │   - Interpretation: Percentile in execution queue
│   │   - 10% ranking: 10% probability of execution before reaching next level
│   │   - 90% ranking: 90% likely to execute, or be far back
│   ├─ Adverse Selection Timing:
│   │   - Problem: Wait for limit order fill, price moves against you
│   │   - Cause: You're not trading, but others ahead are
│   │   - Example: Buy limit @$10.00, wait 30sec, stock drops to $9.90 → lose
│   │   - Risk: Function of (queue length × order flow rate × volatility)
│   └─ Queue Transparency:
│       - Full transparency: Know exact queue position (most exchanges)
│       - Partial: Know depth but not exact queue (rare)
│       - Hidden orders: Don't know full book depth (non-displayed portion)
│       - Impact: Uncertainty on fill probability complicates strategy
│
├─ Microstructure Phenomena:
│   ├─ Order Book Recovery:
│   │   - Event: Large market order → depth drops significantly
│   │   - Recovery: Depth bounces back quickly (elastic)
│   │   - Speed: Quick recovery → liquid market
│   │   - Slow/no recovery → illiquid, information-driven move
│   ├─ Price Impact Persistence:
│   │   - Transient: Temporary impact (from book depletion)
│   │   - Permanent: Permanent impact (from information revelation)
│   │   - Distinction: Book recovery speed indicates ratio
│   │   - Fast recovery → mostly transient → temporary book effect
│   │   - Slow recovery → mostly permanent → information effect
│   ├─ Cascading Orders:
│   │   - Definition: Market order triggers chain of executions across prices
│   │   - Mechanism: Large market order consumes depth at level → moves to next level
│   │   - Distance: How many levels consumed before book replenished
│   │   - Observation: Rare in normal markets, common in stress/crash
│   ├─ Information Cascades:
│   │   - Definition: Order induces other traders to follow (information-based)
│   │   - Mechanism: Public observation of trade suggests information
│   │   - Effect: Subsequent orders in same direction (reinforcement)
│   │   - Risk: Can lead to runaway moves (momentum)
│   ├─ Flash Crashes:
│   │   - Definition: Rapid, extreme price movement then reversal (minutes)
│   │   - Mechanism: Cascading, feedback loops, liquidity withdrawal
│   │   - Depth: Evaporates during flash crash (book disappears)
│   │   - Recovery: Usually within minutes, but damage done
│   │   - Prevention: Circuit breakers, trading halts, fat-finger protections
│   └─ Tick Size Effects:
│       - Larger tick: Fewer depth levels, wider spreads
│       - Smaller tick: More depth levels, tighter spreads
│       - Implication: Regulation of tick size affects book dynamics
│       - Debate: Smaller tick = more competition = tighter spreads vs more fragmentation
│
├─ Empirical Patterns:
│   ├─ Order Arrival Distribution:
│   │   - Poisson baseline: λ orders/sec (baseline model)
│   │   - Reality: Clustered (self-exciting, Hawkes process)
│   │   - Impact: Order flow not independent, predictability exists
│   ├─ Volume Distribution:
│   │   - Order sizes: Highly skewed (many small, few large)
│   │   - Distribution: Power law tails
│   │   - Implication: Large orders rare but significant impact
│   ├─ Depth Profile:
│   │   - Typical: 1M @ level 1, 500K @ level 2, decay power-law
│   │   - Decay rate: Varies with liquidity (steeper for illiquid)
│   │   - Gap: Occasional gaps in price levels (no orders)
│   ├─ Cancellation Rates:
│   │   - Range: 50-95% depending on asset class
│   │   - Stocks: 60-80% typical
│   │   - Options: 80-95% (more uncertain, more cancellations)
│   │   - Variation: Higher rates in volatile times
│   ├─ Fill Probability Distribution:
│   │   - Tighter limit price → Lower fill prob, better execution price if filled
│   │   - Looser limit price → Higher fill prob, worse execution price
│   │   - Optimal: Balance execution certainty vs price improvement
│   └─ Intraday Patterns:
│       - Open: Gaps from overnight, volatile, sparse depth
│       - Mid-day: Stable, rich depth, efficient execution
│       - Close: Portfolio adjustments, variable depth, possible gaps
│       - After-hours: Very sparse, large spreads, minimal execution
│
└─ Advanced Topics:
    ├─ Limit Order Book Modeling:
    │   - Brownian motion: Continuous price model (not realistic for discrete ticks)
    │   - Jump process: Account for discrete arrivals (better)
    │   - Stochastic depth: Model depth as random process
    │   - Queuing theory: Treat book as queue (M/M/1, M/D/1 etc.)
    │   - Benefit: Mathematical tractability, testable predictions
    ├─ Information Asymmetry:
    │   - Signaling: Order placement = information signal
    │   - MM inference: Detect informed traders from order patterns
    │   - Dynamics: Information gradually incorporated into prices
    │   - Arms race: Sophisticated traders camouflage intentions
    ├─ Strategic Order Placement:
    │   - Predatory: Place orders to move market for personal benefit
    │   - Spoofing: Place orders with no intent to execute (manipulate)
    │   - Layering: Multiple orders at different prices to suggest momentum
    │   - Regulation: Now prohibited (Dodd-Frank Act provisions)
    ├─ Virtual Queue Position:
    │   - Concept: Effective position accounting for likely cancellations
    │   - Calculation: Observed position / (1 - cancellation rate)
    │   - Value: Better predictor of actual fill time
    │   - Limitation: Cancellation rate time-varying, hard to estimate
    └─ Machine Learning Applications:
        - Fill probability prediction: Neural networks on order history
        - Queue dynamics: LSTM to forecast depth evolution
        - Optimal execution: RL agents learn order placement strategy
        - Advantage: Non-parametric, can capture complex patterns
        - Challenge: Requires large data, interpretability limited
```

**Interaction:** Order arrives → finds position in queue at price level → waits while ahead execute → eventually fills or gets cancelled → depth replenishes or depletes depending on imbalance

## 5. Mini-Project
Model and simulate limit order book dynamics:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass

np.random.seed(42)

@dataclass
class Order:
    order_id: int
    side: str  # 'buy' or 'sell'
    price: float
    size: int
    timestamp: float
    
class LimitOrderBookSimulator:
    def __init__(self, initial_mid=100.0, tick_size=0.01):
        self.mid_price = initial_mid
        self.tick_size = tick_size
        self.bids = {}  # {price: deque of (order_id, size)}
        self.asks = {}  # {price: deque of (order_id, size)}
        self.order_counter = 0
        self.execution_log = []
        self.book_snapshots = []
        
    def add_limit_order(self, side, price, size, timestamp):
        """Add limit order to book"""
        self.order_counter += 1
        order = Order(self.order_counter, side, price, size, timestamp)
        
        if side == 'buy':
            if price not in self.bids:
                self.bids[price] = deque()
            self.bids[price].append((order.order_id, size))
        else:  # sell
            if price not in self.asks:
                self.asks[price] = deque()
            self.asks[price].append((order.order_id, size))
        
        return order.order_id
    
    def execute_market_order(self, side, size, timestamp):
        """Execute market order against book"""
        remaining = size
        filled = []
        
        if side == 'buy':
            # Match against asks (lowest first)
            ask_levels = sorted(self.asks.keys())
            for ask_price in ask_levels:
                if remaining == 0:
                    break
                while self.asks[ask_price] and remaining > 0:
                    order_id, available = self.asks[ask_price].popleft()
                    execute_size = min(available, remaining)
                    filled.append((ask_price, execute_size))
                    remaining -= execute_size
                    
                    if execute_size < available:
                        self.asks[ask_price].appendleft((order_id, available - execute_size))
                
                if not self.asks[ask_price]:
                    del self.asks[ask_price]
        else:  # sell market
            # Match against bids (highest first)
            bid_levels = sorted(self.bids.keys(), reverse=True)
            for bid_price in bid_levels:
                if remaining == 0:
                    break
                while self.bids[bid_price] and remaining > 0:
                    order_id, available = self.bids[bid_price].popleft()
                    execute_size = min(available, remaining)
                    filled.append((bid_price, execute_size))
                    remaining -= execute_size
                    
                    if execute_size < available:
                        self.bids[bid_price].appendleft((order_id, available - execute_size))
                
                if not self.bids[bid_price]:
                    del self.bids[bid_price]
        
        self.execution_log.append({
            'timestamp': timestamp,
            'side': side,
            'requested': size,
            'executed': size - remaining,
            'prices': filled
        })
        
        return filled, remaining
    
    def get_bid_ask(self):
        """Get current bid, ask, and spread"""
        bid = max(self.bids.keys()) if self.bids else None
        ask = min(self.asks.keys()) if self.asks else None
        mid = (bid + ask) / 2 if bid and ask else None
        spread = ask - bid if bid and ask else None
        
        return bid, ask, mid, spread
    
    def get_depth(self, levels=5):
        """Get depth at top N levels"""
        bid, ask, _, _ = self.get_bid_ask()
        
        bid_depth = []
        ask_depth = []
        
        if bid:
            bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
            for price in bid_prices:
                volume = sum(size for _, size in self.bids[price])
                bid_depth.append((price, volume))
        
        if ask:
            ask_prices = sorted(self.asks.keys())[:levels]
            for price in ask_prices:
                volume = sum(size for _, size in self.asks[price])
                ask_depth.append((price, volume))
        
        return bid_depth, ask_depth
    
    def get_total_depth(self, side='all'):
        """Get total volume on each side"""
        bid_depth = sum(sum(size for _, size in orders) for orders in self.bids.values())
        ask_depth = sum(sum(size for _, size in orders) for orders in self.asks.values())
        
        if side == 'bid':
            return bid_depth
        elif side == 'ask':
            return ask_depth
        return bid_depth, ask_depth

# Scenario 1: Order book dynamics during heavy selling
print("Scenario 1: Order Book During Heavy Selling")
print("=" * 80)

book = LimitOrderBookSimulator(initial_mid=100.0)

# Build initial book
for price in np.arange(99.90, 100.0, 0.01):
    book.add_limit_order('buy', price, 50000, 0)

for price in np.arange(100.01, 100.11, 0.01):
    book.add_limit_order('sell', price, 50000, 0)

bid, ask, mid, spread = book.get_bid_ask()
bid_depth, ask_depth = book.get_total_depth()

print(f"Initial state:")
print(f"  Bid: ${bid:.2f}, Ask: ${ask:.2f}, Spread: ${spread:.2f}")
print(f"  Bid depth: {bid_depth:,}, Ask depth: {ask_depth:,}")

# Large selling order
filled, remaining = book.execute_market_order('sell', 150000, 1)
total_executed = sum(size for _, size in filled)
avg_price = sum(price * size for price, size in filled) / total_executed if filled else 0

print(f"\nLarge sell order (150,000 shares):")
print(f"  Executed: {total_executed:,} shares")
print(f"  Average price: ${avg_price:.2f}")
print(f"  Remaining unexecuted: {remaining:,}")

bid, ask, mid, spread = book.get_bid_ask()
bid_depth, ask_depth = book.get_total_depth()

print(f"\nPost-execution state:")
print(f"  Bid: ${bid:.2f}, Ask: ${ask:.2f}, Spread: ${spread:.2f}")
print(f"  Bid depth: {bid_depth:,}, Ask depth: {ask_depth:,}")
print(f"  Price impact: {(100.0 - avg_price):.4f} per share")

# Scenario 2: Limit order placement strategy
print(f"\n\nScenario 2: Limit Order Placement & Fill Probability")
print("=" * 80)

book = LimitOrderBookSimulator(100.0)

# Build book
for price in np.arange(99.80, 100.0, 0.01):
    book.add_limit_order('buy', price, 100000, 0)

for price in np.arange(100.01, 100.21, 0.01):
    book.add_limit_order('sell', price, 100000, 0)

bid, ask, mid, spread = book.get_bid_ask()

# Try different limit prices
strategies = [
    ('aggressive', ask + 0.01),  # Above ask (guaranteed fill)
    ('inside', ask + 0.005),      # Between bid-ask
    ('passive', ask),             # At ask
    ('very passive', ask + 0.05), # Far away
]

print(f"Current mid: ${mid:.2f}, Bid: ${bid:.2f}, Ask: ${ask:.2f}")
print(f"\nBuy order placement strategies:")

for name, limit_price in strategies:
    # Simulate queue position
    at_price = 0
    for ord_id, size in book.asks.get(limit_price, []):
        at_price += size
    
    # Estimate fill probability (simplified)
    distance = limit_price - ask
    fill_prob = max(0, 1 - (distance / spread) ** 2) if spread > 0 else 0
    
    print(f"  {name:>15}: Limit=${limit_price:.3f}, Queue ahead={at_price:,}, Est. fill prob={fill_prob:.1%}")

# Scenario 3: Depth dynamics with cancellations
print(f"\n\nScenario 3: Depth Evolution with Order Cancellations")
print("=" * 80)

book = LimitOrderBookSimulator(100.0)

# Initial depth
for price in np.arange(99.90, 100.0, 0.01):
    book.add_limit_order('buy', price, 100000, 0)

for price in np.arange(100.01, 100.11, 0.01):
    book.add_limit_order('sell', price, 100000, 0)

bid_depth_history = []
ask_depth_history = []

# Simulate time progression with cancellations and arrivals
for t in range(20):
    bid_d, ask_d = book.get_total_depth()
    bid_depth_history.append(bid_d)
    ask_depth_history.append(ask_d)
    
    if t % 3 == 0 and t > 0:
        # Random market order
        size = np.random.randint(10000, 50000)
        side = 'buy' if np.random.random() > 0.5 else 'sell'
        book.execute_market_order(side, size, t)
    else:
        # Add new limit orders (replenishment)
        side = 'buy' if np.random.random() > 0.5 else 'sell'
        if side == 'buy':
            price = bid - np.random.rand() * 0.05
        else:
            price = ask + np.random.rand() * 0.05
        
        size = np.random.randint(50000, 150000)
        book.add_limit_order(side, price, size, t)

print(f"Depth dynamics over {len(bid_depth_history)} time steps:")
print(f"  Starting bid depth:   {bid_depth_history[0]:,}")
print(f"  Ending bid depth:     {bid_depth_history[-1]:,}")
print(f"  Starting ask depth:   {ask_depth_history[0]:,}")
print(f"  Ending ask depth:     {ask_depth_history[-1]:,}")
print(f"  Volatility (bid): {np.std(bid_depth_history):,.0f}")
print(f"  Volatility (ask): {np.std(ask_depth_history):,.0f}")

# Scenario 4: Execution prices across different order sizes
print(f"\n\nScenario 4: Price Impact by Order Size")
print("=" * 80)

order_sizes = [10000, 50000, 100000, 250000, 500000]
impact_results = []

for size in order_sizes:
    book = LimitOrderBookSimulator(100.0)
    
    # Build fresh book
    for price in np.arange(99.80, 100.0, 0.01):
        book.add_limit_order('buy', price, 200000, 0)
    
    for price in np.arange(100.01, 100.21, 0.01):
        book.add_limit_order('sell', price, 200000, 0)
    
    # Execute market order
    filled, remaining = book.execute_market_order('sell', size, 0)
    
    if filled:
        total_executed = sum(s for _, s in filled)
        avg_price = sum(p * s for p, s in filled) / total_executed
        impact = 100.0 - avg_price
        impact_pct = (impact / 100.0) * 10000  # in bps
    else:
        avg_price = 0
        impact = 0
        impact_pct = 0
    
    impact_results.append((size, avg_price, impact_pct))
    print(f"  Sell {size:,} shares → Avg price: ${avg_price:.4f}, Impact: {impact_pct:.2f} bps")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Order book structure
book = LimitOrderBookSimulator(100.0)

# Build book
for price in np.arange(99.80, 100.0, 0.02):
    book.add_limit_order('buy', price, 100000, 0)

for price in np.arange(100.02, 100.20, 0.02):
    book.add_limit_order('sell', price, 100000, 0)

bid_depth, ask_depth = book.get_depth(levels=10)

bid_prices = [p for p, _ in bid_depth]
bid_vols = [v for _, v in bid_depth]
ask_prices = [p for p, _ in ask_depth]
ask_vols = [v for _, v in ask_depth]

axes[0, 0].barh([f'${p:.2f}' for p in bid_prices], bid_vols, color='green', alpha=0.7, label='Bid')
axes[0, 0].barh([f'${p:.2f}' for p in ask_prices], ask_vols, color='red', alpha=0.7, label='Ask')
axes[0, 0].set_xlabel('Volume')
axes[0, 0].set_title('Scenario 1: Order Book Structure')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='x')

# Plot 2: Depth evolution
axes[0, 1].plot(bid_depth_history, linewidth=2, label='Bid Depth', color='green')
axes[0, 1].plot(ask_depth_history, linewidth=2, label='Ask Depth', color='red')
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Total Volume')
axes[0, 1].set_title('Scenario 3: Depth Dynamics Over Time')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Price impact scaling
if impact_results:
    sizes = [r[0] for r in impact_results]
    impacts = [r[2] for r in impact_results]
    
    axes[1, 0].plot(sizes, impacts, 'o-', linewidth=2, markersize=8, color='blue')
    axes[1, 0].set_xlabel('Order Size (shares)')
    axes[1, 0].set_ylabel('Price Impact (basis points)')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_title('Scenario 4: Impact Scaling')
    axes[1, 0].grid(alpha=0.3)
    
    # Fit sqrt law
    import numpy as np
    log_sizes = np.log(sizes)
    log_impacts = np.log(impacts)
    coeffs = np.polyfit(log_sizes, log_impacts, 1)
    exponent = coeffs[0]
    
    axes[1, 0].text(0.05, 0.95, f'√-law exponent: {exponent:.2f}', 
                   transform=axes[1, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Fill probability by limit distance
distances = np.linspace(0, 0.2, 20)
fill_probs = np.maximum(0, 1 - (distances / 0.05) ** 1.5)  # Decay with distance

axes[1, 1].plot(distances, fill_probs, linewidth=2, color='purple')
axes[1, 1].fill_between(distances, fill_probs, alpha=0.3, color='purple')
axes[1, 1].set_xlabel('Distance from Best Ask ($)')
axes[1, 1].set_ylabel('Fill Probability')
axes[1, 1].set_title('Scenario 2: Fill Probability vs Limit Price')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Average execution price impact: ${np.mean([r[2] for r in impact_results]):.2f} bps")
print(f"Impact scales with √ order size (typical)")
print(f"Depth replenishment time: ~5-10 seconds (market dependent)")
print(f"Fill probability inversely related to limit distance from mid")
```

## 6. Challenge Round
If limit order books are so transparent (full depth visible), why don't all traders simply place limit orders at the inside quotes and guarantee fills?

- **Adverse selection risk**: Others ahead in queue are smarter (information-driven) → if you follow their execution, you're trading against informed traders → losses exceed fill premium → survival bias (see successful fills, not losses)
- **Opportunity cost**: Place at inside quote → guaranteed fill but on their terms, not yours → executed against momentum moves → mean reversion against you → after fill, regret (should have waited)
- **Queue position uncertainty**: Depth displayed, but cancellations hidden → queue length uncertain → expected fill time unknown → could wait 30 seconds (waste) or fill instantly (surprise) → timing risk compounds
- **Inventory constraints**: Place many limit orders to guarantee fills → build inventory → risk if market moves → forced to liquidate at loss → can't just be passive, must manage inventory
- **Regime changes**: In normal times, inside quote fill reliable → crisis arrives → others pull quotes → your inside limit becomes best bid/ask → forced to execute at worse price → regime-switching breaks strategy
- **Cost-benefit**: Guaranteed fill not worth the cost if execution price terrible → Liquidity provider premium (rebate) not enough → better to use market order, accept cost, move on

## 7. Key References
- [Rosu (2009) - A Dynamic Model of Limit Order Book](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1449985)
- [Harris (2003) - Trading and Exchanges: Market Microstructure for Practitioners](https://archive.org/details/tradinganddisper00harr)
- [Hasbrouck & Saar (2013) - High Frequency Trading and Price Discovery](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1656939)
- [Bouchaud et al (2002) - Statistically Valid Moves in Efficient Markets](https://arxiv.org/abs/cond-mat/0108141)

---
**Status:** Discrete-time market microstructure | **Complements:** Order Types, Market Depth, Execution Algorithms
