# Limit Orders: Mechanics, Queuing, and Execution Strategies

## I. Concept Skeleton

**Definition:** A limit order is an instruction to buy or sell a security at a specified maximum (buy) or minimum (sell) price, and the order will only execute at that price or better. Unlike market orders that guarantee execution at current market price, limit orders guarantee price but sacrifice execution certainty—they may never fill if the price never reaches the limit level.

**Purpose:** Provide traders control over execution price (avoid adverse fills), enable patience-based strategies (get better prices by waiting), protect against market impact in large orders, and allow systematic algorithmic execution at predetermined price levels.

**Prerequisites:** Bid-ask mechanics, order book structure (LOB = limit order book), market microstructure (priority rules), probability and queueing theory (arrival processes), execution algorithms.

---

## II. Comparative Framing

| **Aspect** | **Limit Orders** | **Market Orders** | **Stop-Limit Orders** | **Iceberg Orders** | **Pegged Orders** | **VWAP Algorithms** |
|-----------|-----------------|-----------------|----------------------|-------------------|------------------|-------------------|
| **Price Control** | Full (guaranteed price) | None (market price) | Conditional (limit if triggered) | Full (hidden limit) | Dynamic (pegs to bid/ask) | None (follows market vol) |
| **Execution Guarantee** | Low (may not fill) | High (guaranteed) | Medium (if stop triggered) | Medium (depends on reveal) | Medium (if peg holds) | High (VWAP tracks) |
| **Placement** | Off-market (in queue) | Immediate (crosses spread) | Triggered on stop, then limit | Partially visible | Dynamic on mid-price | Real-time adaptive |
| **Cost** | Low (no spread crossing) | High (spread + impact) | Medium (if filled) | Low if partial | Variable | Low (passive) |
| **Fill Probability** | Price-dependent | ~100% | Stop-dependent | Volume-dependent | Bid-ask dependent | ~100% (by design) |
| **Typical Use** | Patient execution, scaling | Urgent orders, hedges | Risk management, stops | Hide order size | Track benchmark | Algorithmic execution |
| **Latency Sensitivity** | Low (can wait) | High (must be fast) | Medium (monitor price) | Low (hidden) | Medium (track midpoint) | Low (adaptive) |
| **Information Leakage** | High (visible in book) | Immediate (crosses spread) | Medium (stop visible) | Low (iceberg hidden) | Low (blends with market) | Low (follows market) |
| **Example** | Buy 10k at $100 max | Buy 10k market | Sell if drops to $99, then sell at $98 max | Buy 10k total, show 1k at time | Buy tracking midpoint±$0.10 | Execute 500k matching vol |

---

## III. Examples & Counterexamples

### Example 1: Patient Limit Order - Getting Better Fill
**Setup:**
- Current market: Bid $100.00, Ask $100.05 (spread $0.05)
- Your need: Buy 10,000 shares (willing to wait up to 5 minutes)
- Aggressive strategy: Market order at $100.05 = $1,000,500 cost
- Patient strategy: Place limit order at $100.01

**Timeline:**
| Time | Market Bid-Ask | Book Status | Your Order Status | Notes |
|------|------------|---------|-----------|---------|
| t=0 | $100.00 - $100.05 | Many sellers at $100.05 | Limit $100.01 posted, waiting | Queue: 23rd in line at $100.01 |
| t=30s | $100.00 - $100.05 | Same | Still in queue | Position: 15th (others canceled) |
| t=60s | $99.98 - $100.04 | Price dipped | Still in queue | Position: 8th (more cancellations) |
| t=90s | $100.01 - $100.06 | Seller aggressed down | **PARTIALLY FILLED** | 6,000 shares at $100.01 |
| t=120s | $100.01 - $100.05 | Market bounced | Remaining 4k in queue | Position: 12th (new sellers) |
| t=210s | $99.99 - $100.02 | Another dip | **PARTIALLY FILLED** | Final 4,000 at $99.99 |

**Analysis:**
- Total cost: 6,000 × $100.01 + 4,000 × $99.99 = $600,060 + $399,960 = **$1,000,020 total**
- Average execution: $100.002 per share
- Vs market order: $1,000,500
- **Savings: $480** (0.048% better, 48 bps)
- Wait time: 3.5 minutes (acceptable)
- Risk: Could have missed entire fill if price went up

**Interpretation:** Patience rewarded. Limit order at $100.01 (1 cent inside NBBO spread) successfully caught partial fills as market fluctuated.

---

### Example 2: Limit Order Failure - Overly Aggressive Pricing
**Setup:**
- Current market: Bid $100.00, Ask $100.05
- Your goal: Buy 50,000 shares, need execution reasonably fast
- Your limit: $99.95 (5 cents below current bid, very aggressive)
- Time: Market in downtrend

**What Happens:**
- Your order posted: Limit $99.95 in order book
- Market movement: Prices rose instead (trending up)
  - t=1min: Bid $100.05, Ask $100.10 (moved up)
  - t=3min: Bid $100.15, Ask $100.20
  - t=5min: Bid $100.25, Ask $100.30
  - Market drifted +$0.25

**Result:**
- Your limit order: **NEVER FILLED** (price never hit $99.95)
- Status after 5 minutes: 0 shares executed, order still in book
- What you should have done: Market order at $100.05 would have filled immediately at $1,000,250 + impact
- Actual realized cost if forced to buy now: $100.30 ask = $5,015,000
- **Opportunity loss: ~$15,000** (worse than if you'd used market order)

**Interpretation:** Over-aggressive limit pricing (too far from market) led to execution failure. Lesson: Limit price must be realistic relative to current market.

---

### Example 3: Counterexample—Limit Order Advantage in Volatile Market
**Setup:**
- Current market: Bid $100.00, Ask $100.05
- Large sell order 20,000 shares
- Strategy A: Market order (immediate)
- Strategy B: Limit order at $100.01 (sell at better price if possible)

**Market Volatility Event (Downward breakout):**
- Sudden downward price shock (negative news)
- Bid drops from $100.00 to $99.80 in 200ms
- Market order placed at t=-100ms: Executes at $100.00 ✓
- Limit order placed at t=-100ms: Doesn't execute at $99.95 (price fell below)

**Recovery (Bounce back):**
- Price recovers: Bid climbs to $100.10, Ask $100.15
- Your limit order at $100.01: **NOW FILLS** at $100.01 (you posted above the bid)

**Comparison:**
- Market order: Sold at $100.00 = $2,000,000
- Limit order (with patience): Sold at $100.01 = $2,000,200
- **Gain: $200** (20 bps)
- But with extra risk: If market never recovered, you'd still hold 20k shares

**Interpretation:** Limit orders can capture upside in volatile markets but require patience and market recovery.

---

## IV. Layer Breakdown

```
LIMIT ORDER FRAMEWORK

┌──────────────────────────────────────────────────────────────┐
│              LIMIT ORDER EXECUTION MODEL                      │
│                                                               │
│  Core Principle:                                             │
│  Execute at specified price (or better)                      │
│  but only if market allows                                   │
│                                                               │
│  BUY Limit Order:  Execute at Limit_Price ≤ (given price)   │
│  SELL Limit Order: Execute at Limit_Price ≥ (given price)   │
└────────────────────┬─────────────────────────────────────────┘
                     │
    ┌────────────────▼─────────────────────────────────┐
    │  1. ORDER BOOK POSITIONING                       │
    │                                                   │
    │  Limit Order Book (LOB):                         │
    │  ┌───────────────────────────────────────┐      │
    │  │ ASK (Sellers' Limits)                │      │
    │  │ $100.10 : 5,000 shares (1st in queue)│      │
    │  │ $100.05 : 15,000 shares              │      │
    │  │ $100.05 : 8,000 shares (YOU posted)  │      │
    │  │ ─────────────────────────────────────│      │
    │  │ BID (Buyers' Limits)                │      │
    │  │ $100.00 : 20,000 shares             │      │
    │  │ $99.95  : 10,000 shares             │      │
    │  │ $99.95  : 5,000 shares              │      │
    │  └───────────────────────────────────────┘      │
    │                                                   │
    │  Your Limit Order:                               │
    │  ├─ Posted price: $100.05                       │
    │  ├─ Quantity: 8,000 shares                      │
    │  ├─ Side: Sell (in ask queue)                   │
    │  ├─ Position in queue: 2nd at this price       │
    │  └─ Waiting: For aggressive buyers              │
    │                                                   │
    │  Key Concept: PRICE-TIME PRIORITY               │
    │  ├─ Execution by price first (best price first) │
    │  ├─ Within same price: by time (FIFO)          │
    │  └─ Your position: affects fill likelihood     │
    └────────────────┬────────────────────────────────┘
                     │
    ┌────────────────▼───────────────────────────────────┐
    │  2. QUEUEING DYNAMICS (Poisson Process)            │
    │                                                     │
    │  Aggressive Buyer Arrivals:                        │
    │  ├─ Market order to buy arrives                   │
    │  ├─ Crosses spread: buys from ask queue          │
    │  ├─ Hits sellers at $100.05                      │
    │  │   First seller: 5,000 filled                  │
    │  │   Second seller (YOU): Partially filled       │
    │  └─ Your limit order fills as queue advances     │
    │                                                     │
    │  Queue Position Evolution:                        │
    │  t=0:   You posted → Position 2 at $100.05      │
    │  t=1.2s: Buyer came → Position 1                │
    │  t=2.1s: Another buyer → Partially filled       │
    │  t=3.5s: More buyers → FULLY FILLED             │
    │                                                     │
    │  Filling Process:                                 │
    │  ├─ Aggressive order arrives (market order)      │
    │  ├─ Executes against best limit prices          │
    │  ├─ Your order is in sequence                    │
    │  └─ Fill happens when position reaches top      │
    │                                                     │
    │  Fill Probability:                               │
    │  ├─ Depends on: Arrival rate of aggressors     │
    │  ├─ Formula: P(fill) ≈ 1 - exp(-λ·T)          │
    │  ├─ λ = arrival rate of market orders           │
    │  ├─ T = time duration                            │
    │  └─ Example: λ=2/min, T=5min → P≈99%           │
    └────────────────┬───────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────────────┐
    │  3. EXECUTION STRATEGIES (Optimal Posting)          │
    │                                                      │
    │  Strategy A: AGILE (Improve Price)                 │
    │  ├─ Post inside the spread                         │
    │  ├─ Limit: $100.02 (between bid $100.00, ask $100.05)
    │  ├─ Benefit: Better price than market order        │
    │  ├─ Cost: Lower fill probability                   │
    │  ├─ Risk: Might not execute                        │
    │  └─ Use case: Patient execution, scaling           │
    │                                                      │
    │  Strategy B: PASSIVE (Trade for Sure)              │
    │  ├─ Post at the spread                             │
    │  ├─ Limit: $100.05 (at ask for buying)            │
    │  ├─ Benefit: High fill probability                 │
    │  ├─ Cost: Market price (no improvement)            │
    │  ├─ Execution: Like market order but queued        │
    │  └─ Use case: When certainty matters               │
    │                                                      │
    │  Strategy C: LAYERING (Scaled Entry/Exit)          │
    │  ├─ Multiple limit orders at different prices     │
    │  ├─ Buy: $100.00, $99.95, $99.90 (scaled)         │
    │  ├─ Benefit: Average down on dips                 │
    │  ├─ Cost: Administrative complexity                │
    │  └─ Use case: Algorithmic scaling (TWAP-like)      │
    │                                                      │
    │  Trade-off Curve:                                  │
    │  Price Improvement vs Fill Probability             │
    │  ┌──────────────────────────────────────────┐     │
    │  │ Fill Prob                                 │     │
    │  │   100% ├─ PASSIVE ($100.05)             │     │
    │  │        │      .                          │     │
    │  │    80% ├─ MODERATE ($100.02)            │     │
    │  │        │           ..                    │     │
    │  │    50% ├─ AGILE ($99.98)                │     │
    │  │        │         ...                     │     │
    │  │    10% └─ AGGRESSIVE ($99.90)           │     │
    │  │          └────────────────────────────   │     │
    │  │            Better Price ←                │     │
    │  └──────────────────────────────────────────┘     │
    └────────────────┬────────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────────────┐
    │  4. CANCELLATION & UPDATING                         │
    │                                                      │
    │  Why Cancel?                                       │
    │  ├─ Price moved too much (no longer attractive)   │
    │  ├─ Market conditions changed (vol spiked)        │
    │  ├─ Better opportunity elsewhere                  │
    │  └─ Time-of-day logic (near close)                │
    │                                                      │
    │  Cancellation Cost:                                │
    │  ├─ Regulatory: None (order cancellation free)    │
    │  ├─ Implicit: Lose position in queue              │
    │  ├─ Slippage: If resubmit at different price      │
    │  └─ Latency: Microseconds to process              │
    │                                                      │
    │  Order Update (Amend):                             │
    │  ├─ Change price: May lose queue position         │
    │  ├─ Change quantity: Typically allowed            │
    │  ├─ Impact: Often treated as cancel + new order   │
    │  └─ Tactic: Adjust limit price intraday           │
    └────────────────┬────────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────────────┐
    │  5. PRICING ANALYTICS (Queue Position Value)        │
    │                                                      │
    │  Queue Position Metrics:                           │
    │  ├─ Depth at limit: 15,000 shares ahead of you   │
    │  ├─ Expected fill time: E[T] = Depth / λ         │
    │  ├─ Poisson λ ≈ 0.5 market orders per second    │
    │  ├─ E[T] ≈ 15,000 / (0.5 × 60) ≈ 500 seconds    │
    │  └─ That's 8 minutes                              │
    │                                                      │
    │  Price Improvement Value:                          │
    │  ├─ Limit at $100.02 (inside spread by $0.03)   │
    │  ├─ Improvement: $0.03 per share                 │
    │  ├─ On 10k order: $300 benefit potential         │
    │  ├─ BUT: Only if fill happens                     │
    │  └─ Expected value: $300 × P(fill)               │
    │                                                      │
    │  Decision Rule:                                   │
    │  ├─ If E[Value] > Cost of capital during wait   │
    │  │  → POST LIMIT ORDER (agile)                   │
    │  ├─ Else:                                        │
    │  │  → POST AT SPREAD or MARKET ORDER (passive)   │
    │  └─ Cost of capital ≈ daily return × hold time  │
    └────────────────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### Basic Limit Order Fill Model

**Poisson arrival process for aggressive orders:**
$$N(t) \sim \text{Poisson}(\lambda t)$$

where $\lambda$ = arrival rate of market orders at this price level.

**Probability of at least one fill by time $t$:**
$$P(\text{filled by } t) = 1 - e^{-\lambda t}$$

**Expected fill time (given arrival rate $\lambda$):**
$$E[T_{\text{fill}}] = \frac{1}{\lambda}$$

### Queue Position Evolution

If queue has $n$ orders ahead of yours:
$$P(\text{your order is } k\text{-th}) = \frac{e^{-\lambda t}(\lambda t)^k}{k!}$$

**Expected time to reach top of queue:**
$$E[T_{\text{to top}}] = \frac{n}{\lambda}$$

### Price Improvement vs Fill Probability Trade-off

**Implied spread relationship:**
$$\text{Spread} = \text{Ask} - \text{Bid}$$

**Aggressiveness parameter:**
$$\alpha = \frac{\text{Limit Price} - \text{Bid}}{\text{Spread}}$$

where $\alpha = 0$ (at bid) means limit at lower end, $\alpha = 1$ (at ask) means limit at upper end.

**Fill probability approximation:**
$$P(\text{fill}) \approx e^{-\gamma(1-\alpha)T}$$

where $\gamma$ = market tightness (higher = harder to fill away from spread).

### Expected Value of Limit Order Strategy

**Cost of patience:**
$$V_{\text{cost}} = r \cdot S \cdot E[T]$$

where $r$ = risk-free rate, $S$ = position size, $E[T]$ = expected fill time.

**Benefit of price improvement:**
$$V_{\text{benefit}} = \Delta P \cdot S \cdot P(\text{fill})$$

where $\Delta P$ = price improvement vs market price.

**Net expected value:**
$$V_{\text{net}} = V_{\text{benefit}} - V_{\text{cost}}$$

Choose limit order if $V_{\text{net}} > 0$.

---

## VI. Python Mini-Project: Limit Order Execution Simulator

### Objective
1. Simulate limit order queue dynamics with Poisson arrivals
2. Model fill probability based on queue position and price aggressiveness
3. Compare strategies: passive vs agile vs aggressive
4. Analyze optimal pricing for different market conditions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, expon
from scipy.optimize import minimize_scalar

np.random.seed(42)

# ============================================================================
# LIMIT ORDER QUEUE SIMULATOR
# ============================================================================

class LimitOrderBook:
    """
    Simplified limit order book tracking ask-side queue
    """
    
    def __init__(self, bid_price=100.00, ask_price=100.05, bid_depth=20000, ask_depth=25000):
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_depth = bid_depth  # qty at best bid
        self.ask_depth = ask_depth  # qty at best ask
        
        self.spread = ask_price - bid_price
        self.mid_price = (bid_price + ask_price) / 2
        self.orders_at_ask = ask_depth  # Total orders at ask level
    
    def post_limit_order(self, qty, limit_price, side='buy'):
        """
        Post a limit order and return queue position
        side: 'buy' or 'sell'
        """
        if side == 'buy':
            if limit_price <= self.bid_price:
                # Order inside spread (agile)
                queue_position = 1  # Will be ahead of market order
                return {'status': 'posted', 'queue_position': queue_position, 'price': limit_price}
            elif limit_price <= self.ask_price:
                # Order at spread
                queue_position = 1
                return {'status': 'posted', 'queue_position': queue_position, 'price': limit_price}
            else:
                # Order above ask (too aggressive for buy)
                return {'status': 'error', 'reason': 'limit price above ask'}
        
        elif side == 'sell':
            if limit_price >= self.ask_price:
                # Order at or above ask (good for seller)
                queue_position = self.orders_at_ask + 1
                return {'status': 'posted', 'queue_position': queue_position, 'price': limit_price}
            elif limit_price >= self.bid_price:
                # Order between bid and ask (inside spread, agile)
                queue_position = 1  # Will execute against buyers
                return {'status': 'posted', 'queue_position': queue_position, 'price': limit_price}
            else:
                return {'status': 'error', 'reason': 'limit price below bid'}


class LimitOrderSimulator:
    """
    Simulate execution of limit orders with Poisson market arrivals
    """
    
    def __init__(self, order_qty=10000, arrival_rate_per_min=1.0, max_time_min=10):
        """
        Parameters:
        -----------
        order_qty: Your order quantity
        arrival_rate_per_min: Market order arrivals per minute
        max_time_min: Maximum wait time in minutes
        """
        self.order_qty = order_qty
        self.lambda_per_sec = arrival_rate_per_min / 60  # Convert to per-second
        self.max_time_sec = max_time_min * 60
        
        self.execution_record = []
        self.fill_time = None
        self.filled_qty = 0
    
    def simulate_fills_poisson(self, queue_position, avg_market_order_size=500):
        """
        Simulate market order arrivals and calculate fill times
        
        Assumes: Market orders arrive as Poisson process
                 Each order size drawn from exponential distribution
        """
        fills = []
        current_queue_pos = queue_position
        cumulative_filled = 0
        current_time = 0
        
        # Generate inter-arrival times (exponential with rate lambda)
        while cumulative_filled < self.order_qty and current_time < self.max_time_sec:
            # Inter-arrival time of next market order
            inter_arrival = np.random.exponential(1 / self.lambda_per_sec)
            current_time += inter_arrival
            
            if current_time >= self.max_time_sec:
                break
            
            # Market order size (exponential)
            market_order_size = int(np.random.exponential(avg_market_order_size))
            market_order_size = max(market_order_size, 100)  # Minimum order size
            
            # Reduce queue position
            if current_queue_pos <= market_order_size:
                # Your order gets filled (partially or fully)
                fill_qty = min(self.order_qty - cumulative_filled, 
                              market_order_size - (current_queue_pos - 1))
                
                if fill_qty > 0:
                    fills.append({
                        'time': current_time,
                        'qty': fill_qty,
                        'cumulative': cumulative_filled + fill_qty
                    })
                    cumulative_filled += fill_qty
                    current_queue_pos = 0  # Reset (now at top, wait for next order)
                
                if cumulative_filled >= self.order_qty:
                    self.fill_time = current_time
                    break
            else:
                current_queue_pos -= market_order_size
        
        self.execution_record = fills
        self.filled_qty = cumulative_filled
        
        return fills
    
    def fill_probability_by_time(self, time_seconds):
        """
        Probability of getting filled within time_seconds
        Using exponential distribution: P(T <= t) = 1 - exp(-λt)
        """
        return 1 - np.exp(-self.lambda_per_sec * time_seconds)
    
    def expected_fill_time_given_queue(self, queue_position):
        """
        Expected time to fill given queue position
        E[T] = Queue_Position / λ
        """
        return queue_position / self.lambda_per_sec


class LimitOrderStrategy:
    """
    Compare different limit order pricing strategies
    """
    
    def __init__(self, order_qty=10000, bid=100.00, ask=100.05):
        self.order_qty = order_qty
        self.bid = bid
        self.ask = ask
        self.spread = ask - bid
        self.mid = (bid + ask) / 2
    
    def strategy_passive(self):
        """Post at the spread (at ask for buy)"""
        return {
            'name': 'Passive (At Spread)',
            'limit_price': self.ask,
            'description': f'Limit at ask ${self.ask:.2f}'
        }
    
    def strategy_moderate(self):
        """Post inside spread (middle)"""
        mid_spread = self.bid + 0.75 * self.spread
        return {
            'name': 'Moderate (Inside Spread)',
            'limit_price': mid_spread,
            'description': f'Limit at ${mid_spread:.4f}'
        }
    
    def strategy_agile(self):
        """Post just inside spread"""
        agile_price = self.bid + 0.5 * self.spread
        return {
            'name': 'Agile (Mid-Spread)',
            'limit_price': agile_price,
            'description': f'Limit at ${agile_price:.4f}'
        }
    
    def strategy_aggressive(self):
        """Post far inside spread (risky)"""
        aggressive_price = self.bid + 0.1 * self.spread
        return {
            'name': 'Aggressive (Deep Inside)',
            'limit_price': aggressive_price,
            'description': f'Limit at ${aggressive_price:.4f}'
        }
    
    def compare_strategies(self):
        """Return all strategies for comparison"""
        return [
            self.strategy_passive(),
            self.strategy_moderate(),
            self.strategy_agile(),
            self.strategy_aggressive()
        ]


# ============================================================================
# ANALYSIS & SIMULATIONS
# ============================================================================

print("\n" + "="*80)
print("LIMIT ORDER EXECUTION ANALYSIS")
print("="*80)

# Setup
order_qty = 10000
bid_price = 100.00
ask_price = 100.05
arrival_rate = 2.0  # 2 market orders per minute on average

print(f"\n1. MARKET SETUP")
print(f"   Order size: {order_qty:,} shares")
print(f"   Market: Bid ${bid_price:.2f} | Ask ${ask_price:.2f}")
print(f"   Spread: ${ask_price - bid_price:.4f}")
print(f"   Arrival rate: {arrival_rate} market orders/minute")

# Initialize
book = LimitOrderBook(bid_price, ask_price)
strategies = LimitOrderStrategy(order_qty, bid_price, ask_price).compare_strategies()

print(f"\n2. LIMIT ORDER STRATEGIES")
for i, strat in enumerate(strategies, 1):
    print(f"\n   Strategy {i}: {strat['name']}")
    print(f"   └─ {strat['description']}")

# Simulate each strategy
print(f"\n3. EXECUTION SIMULATION (100 Monte Carlo runs)")
results_summary = []

for strat in strategies:
    fill_times = []
    fill_quantities = []
    fill_probs = []
    
    for sim in range(100):
        # Post limit order
        post_result = book.post_limit_order(order_qty, strat['limit_price'], side='sell')
        
        if post_result['status'] != 'posted':
            continue
        
        # Simulate Poisson arrivals
        simulator = LimitOrderSimulator(order_qty, arrival_rate, max_time_min=10)
        fills = simulator.simulate_fills_poisson(post_result['queue_position'])
        
        if simulator.fill_time is not None:
            fill_times.append(simulator.fill_time)
            fill_quantities.append(simulator.filled_qty)
            fill_probs.append(1.0 if simulator.filled_qty == order_qty else simulator.filled_qty / order_qty)
    
    # Statistics
    if fill_times:
        results_summary.append({
            'Strategy': strat['name'],
            'Limit Price': f"${strat['limit_price']:.4f}",
            'Avg Fill Time (sec)': np.mean(fill_times),
            'Avg Fill Time (min)': np.mean(fill_times) / 60,
            'Fill Prob': np.mean(fill_probs),
            'Partial Fill Rate': sum(1 for q in fill_quantities if q < order_qty) / len(fill_quantities),
            'Expected Value': (strat['limit_price'] - ask_price) * order_qty * np.mean(fill_probs)
        })

df_results = pd.DataFrame(results_summary)
print(df_results.to_string(index=False))

# Detailed example: Aggressive vs Passive
print(f"\n4. DETAILED EXAMPLE: Aggressive vs Passive")

simulator_passive = LimitOrderSimulator(order_qty, arrival_rate, max_time_min=5)
fills_passive = simulator_passive.simulate_fills_poisson(queue_position=100, avg_market_order_size=500)

simulator_aggressive = LimitOrderSimulator(order_qty, arrival_rate, max_time_min=5)
fills_aggressive = simulator_aggressive.simulate_fills_poisson(queue_position=1, avg_market_order_size=500)

print(f"\n   PASSIVE (At ask, queue pos 100):")
print(f"   ├─ Fill time: {simulator_passive.fill_time:.1f} sec ({simulator_passive.fill_time/60:.2f} min)")
print(f"   ├─ Filled qty: {simulator_passive.filled_qty:,} shares")
print(f"   ├─ Fill rate: {simulator_passive.filled_qty/order_qty*100:.1f}%")
print(f"   └─ Price improvement vs market: $0.00 (at market)")

print(f"\n   AGGRESSIVE (Inside spread, queue pos 1):")
print(f"   ├─ Fill time: {simulator_aggressive.fill_time:.1f} sec ({simulator_aggressive.fill_time/60:.2f} min)")
print(f"   ├─ Filled qty: {simulator_aggressive.filled_qty:,} shares")
print(f"   ├─ Fill rate: {simulator_aggressive.filled_qty/order_qty*100:.1f}%")
print(f"   └─ Price improvement vs market: $0.025 per share = ${0.025 * order_qty:.2f} total")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Panel 1: Fill probability over time
ax1 = axes[0, 0]
sim_template = LimitOrderSimulator(order_qty, arrival_rate, max_time_min=10)
times_range = np.linspace(0, 600, 100)  # 0 to 10 minutes
fill_probs = [sim_template.fill_probability_by_time(t) for t in times_range]

ax1.plot(times_range/60, fill_probs, linewidth=2.5, color='blue', label='Fill Probability')
ax1.fill_between(times_range/60, 0, fill_probs, alpha=0.3, color='blue')
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Prob')
ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Prob')
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Probability of Execution')
ax1.set_title('Panel 1: Fill Probability vs Wait Time\n(Poisson arrivals λ=2/min)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 10)

# Panel 2: Expected fill time given queue position
ax2 = axes[0, 1]
queue_positions = np.arange(0, 2001, 100)
expected_times = [LimitOrderSimulator(order_qty, arrival_rate).expected_fill_time_given_queue(q) for q in queue_positions]

ax2.plot(queue_positions, np.array(expected_times)/60, linewidth=2.5, color='green', marker='o', markersize=5)
ax2.fill_between(queue_positions, 0, np.array(expected_times)/60, alpha=0.2, color='green')
ax2.set_xlabel('Queue Position (# orders ahead)')
ax2.set_ylabel('Expected Fill Time (minutes)')
ax2.set_title('Panel 2: Expected Fill Time vs Queue Position\n(Linear relationship: E[T] = Q/λ)')
ax2.grid(True, alpha=0.3)

# Panel 3: Strategy comparison (fill prob vs price improvement)
ax3 = axes[1, 0]
strategies_list = LimitOrderStrategy(order_qty, bid_price, ask_price).compare_strategies()
strategy_names = [s['name'].split('(')[0].strip() for s in strategies_list]
fill_probs_strategy = [0.95, 0.72, 0.45, 0.15]  # Approximate
price_improvements = [0.00, 0.0125, 0.025, 0.0375]

colors = ['green', 'orange', 'red', 'darkred']
for i, (name, prob, improvement, color) in enumerate(zip(strategy_names, fill_probs_strategy, price_improvements, colors)):
    ax3.scatter(improvement*10000, prob*100, s=300, color=color, alpha=0.7, edgecolor='black', linewidth=2, label=name)

ax3.plot(np.array(price_improvements)*10000, np.array(fill_probs_strategy)*100, 'k--', linewidth=1, alpha=0.5)
ax3.set_xlabel('Price Improvement vs Market (basis points)')
ax3.set_ylabel('Fill Probability (%)')
ax3.set_title('Panel 3: Strategy Trade-off Curve\n(Better price → Lower fill prob)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-5, 45)

# Panel 4: Queue evolution over time
ax4 = axes[1, 1]
initial_queue = 1000
times_queue = np.linspace(0, 300, 100)
market_orders_cum = np.random.poisson(arrival_rate/60 * times_queue)
queue_evolution = initial_queue - market_orders_cum * 400  # Assume avg 400 per order
queue_evolution = np.maximum(queue_evolution, 0)

ax4.plot(times_queue/60, queue_evolution, linewidth=2.5, color='purple', label='Your Queue Pos')
ax4.fill_between(times_queue/60, 0, queue_evolution, alpha=0.2, color='purple')
ax4.axhline(y=0, color='red', linestyle='-', linewidth=2, label='Your Order Filled')
ax4.set_xlabel('Time (minutes)')
ax4.set_ylabel('Queue Position (shares ahead)')
ax4.set_title('Panel 4: Queue Position Evolution\n(Decreases as market orders arrive)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('limit_order_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("• Fill probability increases exponentially with wait time")
print("• Expected fill time = Queue Position / Arrival Rate (linear)")
print("• Agile pricing: trade small price improvement for significant fill risk")
print("• Passive pricing: guaranteed fill but no price improvement")
print("• Optimal strategy depends on: opportunity cost, urgency, market tightness")
print("• Queue position crucial: deep inside spread = fast fill, but uncertain")
print("="*80 + "\n")
```

### Output Explanation
- **Panel 1:** Fill probability reaches 90% after ~3 minutes; Poisson process with λ=2/min arrival rate.
- **Panel 2:** Expected fill time linear in queue position; position 1000 = ~8 minutes expected.
- **Panel 3:** Classic trade-off; aggressive (0.0375 bps improvement) = 15% fill prob; passive (0 improvement) = 95% fill prob.
- **Panel 4:** Queue position drops as market orders arrive; reaches zero when fully filled.

---

## VII. References & Key Design Insights

1. **Rosu, I. (2009).** "A dynamic model of the limit order book." Review of Financial Studies, 22(11), 4601-4641.
   - Queue dynamics; fill probabilities; optimal limit pricing

2. **Parlour, C. A. (1998).** "Price dynamics in limit order markets." Review of Financial Studies, 11(4), 789-816.
   - Limit order book structure; price formation; strategic posting

3. **Foucault, T. (1999).** "Order flow composition and information in the market." Journal of Financial Intermediation, 8(3-4), 199-231.
   - Aggressive vs passive orders; information revelation; execution decisions

4. **Ranaldo, A. (2004).** "Order aggressiveness in limit order book markets." Journal of Financial Markets, 7(1), 53-74.
   - Empirical fill rates; aggressiveness strategies; market conditions

**Key Design Concepts:**

- **Price-Time Priority:** Orders execute in price order (best first), then by time (FIFO). Queue position directly affects execution probability.
- **Poisson Process:** Market order arrivals well-modeled as Poisson; enables analytical fill probability calculations.
- **Patience Economics:** Cost of waiting (opportunity cost of capital) must be weighed against price improvement benefit. Optimal limit price balances both.
- **Information Content:** Posting limit order reveals your intentions to market; aggressive pricing signals urgency, passive pricing signals patience.

