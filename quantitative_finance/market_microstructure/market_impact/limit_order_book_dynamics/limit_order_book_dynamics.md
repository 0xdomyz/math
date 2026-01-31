# Limit Order Book Dynamics

## 1. Concept Skeleton
**Definition:** Stochastic processes governing order book evolution—arrival and cancellation of limit orders, price formation, and queue dynamics affecting execution probability and timing  
**Purpose:** Predict execution likelihood and timing for limit orders, model optimal queuing strategies for market makers, forecast price impact from order book imbalance  
**Prerequisites:** Order flow processes, stochastic calculus, queueing theory, market microstructure dynamics

## 2. Comparative Framing
| Aspect | Regular Queue | Priority Queue | Layered Book | Turbulent Book |
|--------|---------------|----------------|--------------|-----------------|
| **Order Arrivals** | Poisson (random) | Correlated | U-shaped (open/close) | Clustered (stress) |
| **Cancellation Rate** | 5-20% per minute | <5% (patient) | 20-50% (uncertain) | 50%+ (panic) |
| **Fill Probability** | 10-50% per minute | <10% (lower) | 5-15% (shallow) | >50% (desperate) |
| **Queue Position** | FIFO (first-in-first-out) | Price-time priority | Multiple levels | Stochastic |
| **Price Discovery** | Gradual (efficient) | Smooth (stable) | Volatile (light) | Discontinuous (crisis) |
| **Execution Speed** | Medium (stable) | Slow (patient) | Fast (competitive) | Very fast (panic) |

## 3. Examples + Counterexamples

**Stable Queue (Mid-Day, Liquid Stock):**  
Limit order book depth: $100.00 bid (5k shares), $100.01 ask (3k shares).  
Arrival process: 1 order per second average (Poisson rate λ=1).  
Your limit buy order at $100.00 queued. Queue size ahead: 100 shares.  
Expected wait: ~100 sec (100 shares / 1 share per second rate).  
Fill probability: High (95%) if patient.

**Shallow Book (Illiquid Stock, Thin Depth):**  
Limit order book: $50.00 bid (50 shares only), $50.10 ask (30 shares).  
Total depth: 80 shares (very light).  
Your limit buy: 200 shares at $50.00.  
Queue position: 50 ahead (entire depth), you're behind the book.  
Fill probability: Low (<20%) - market orders from other buyers will fill you first, book depletes slowly.

**Cancellation Cascade (Earnings Announcement):**  
Before announcement: 10k shares at bid (stable, patient sellers).  
Announcement released (positive surprise): Immediate 30% cancellation (sellers pulled offers).  
Your bid to buy: Falls to back of much shorter queue (lucky!).  
Fill probability: Increases due to cancellations, despite bid-ask widening.

**Fast Queue (Open, Competitive Bidding):**  
IPO opening auction: 100,000 orders awaiting matching (enormous queue).  
Clearing happens all-at-once (not FIFO).  
Your limit order: Executes at 50% probability (depends on clearing price).  
Very fast (milliseconds), not queue-dependent.

**Toxic Queue (High-Frequency Trading Front-Running):**  
Market order from institutional trader → 10k shares to buy.  
Visible in order book: 5k shares at ask.  
HFT algos detect: Buy ahead of institutional (predict price rise).  
Institutional order gets worse execution (price moved up).  
Toxic queue: HFT extracted liquidity/information.

**Queue Jumping (Penny Improvement):**  
Best bid: $100.00 (1000 shares queued).  
New bidder enters: $100.01 (price improvement by 1 cent).  
Jumps to front of queue (better price).  
Original bidders: Now at back (worse deal for them, queue jumped).

## 4. Layer Breakdown
```
Limit Order Book Dynamics Framework:

├─ Order Book Structure:
│  ├─ Two-Sided Order Book:
│  │  ├─ Bid side (sellers' prices): Buy intentions
│  │  │  ├─ Level 1: Best bid (highest buy price)
│  │  │  ├─ Level 2-5: Secondary bids (lower prices)
│  │  │  ├─ Level 10+: Deep bids (even lower)
│  │  │  └─ Volume at each level: Queue depth
│  │  │
│  │  ├─ Ask side (buyers' prices): Sell intentions
│  │  │  ├─ Level 1: Best ask (lowest sell price)
│  │  │  ├─ Level 2-5: Secondary asks (higher prices)
│  │  │  ├─ Level 10+: Deep asks (even higher)
│  │  │  └─ Volume at each level: Queue depth
│  │  │
│  │  └─ Mid-Quote: (Bid + Ask) / 2 (price reference)
│  │
│  ├─ Queue Position:
│  │  ├─ FIFO (First-In-First-Out):
│  │  │  ├─ Orders matched in submission time order
│  │  │  ├─ Your position = time submitted (earlier = higher priority)
│  │  │  ├─ Standard on most exchanges
│  │  │  └─ Incentive: Submit early for better queue position
│  │  │
│  │  ├─ Price-Time Priority:
│  │  │  ├─ Price improvement takes priority (better price bumps queue)
│  │  │  ├─ Within same price: FIFO applies
│  │  │  ├─ Example: Bid $100.05 jumps ahead of $100.00 bids
│  │  │  └─ Mechanism: Exchange allows price improvement
│  │  │
│  │  └─ Pro-Rata (Rare):
│  │     ├─ Execution share proportional to volume posted
│  │     ├─ Example: 3 orders at $100.00 (1k, 2k, 3k volumes)
│  │     ├─ Buy 2k market order → 333, 667, 1000 fill ratio
│  │     ├─ Used: CBOT (futures), some commodity exchanges
│  │     └─ Effect: Discourages small orders (unfair allocation)
│  │
│  ├─ Depth Terminology:
│  │  ├─ Top of book (best bid/ask):
│  │  │  ├─ Most aggressive orders (likely to fill first)
│  │  │  ├─ Quantity: 100-1000 shares (varies by stock)
│  │  │  └─ Tightest spreads (competition for queue position)
│  │  │
│  │  ├─ Visible depth (Level 1-5):
│  │  │  ├─ Liquid orders (likely to participate in next trades)
│  │  │  ├─ Quantity: 1-10k shares (medium depth)
│  │  │  └─ Typical: Visible to all traders (displayed)
│  │  │
│  │  └─ Deep book (Level 10+):
│  │     ├─ Patient orders (far from current price)
│  │     ├─ Quantity: 10-100k shares (very deep)
│  │     ├─ Relevance: Price moves needed to reach (less immediate)
│  │     └─ May be undisclosed (hidden in dark pools)
│  │
│  └─ Centralized vs. Decentralized:
│     ├─ Centralized (Stock exchanges):
│     │  ├─ Single order book (one view of liquidity)
│     │  ├─ FIFO matching (fairness, transparency)
│     │  ├─ Price formation: Matching at exchange prices
│     │  └─ Example: NYSE, NASDAQ
│     │
│     └─ Decentralized (OTC markets):
│        ├─ Multiple dealers with separate books
│        ├─ Bilateral negotiation (no strict queue)
│        ├─ Price discovery: Dealer-to-dealer quotes
│        └─ Example: Foreign exchange, bonds, OTC equity
│
├─ Order Arrival Dynamics:
│  ├─ Poisson Process (Base Model):
│  │  ├─ Assumption: Orders arrive randomly, independently
│  │  ├─ Arrival rate λ (orders per unit time)
│  │  ├─ Example: λ = 1 order/sec → average 60 orders/min
│  │  ├─ Probability of k arrivals in time τ: P(k) = e^(-λτ) × (λτ)^k / k!
│  │  ├─ Distribution: Exponential inter-arrival times
│  │  ├─ Properties:
│  │  │  ├─ Memoryless (past arrivals irrelevant to future)
│  │  │  ├─ Mean interval: 1/λ (average time between orders)
│  │  │  ├─ Variance: 1/λ² (standard deviation = 1/λ)
│  │  │  └─ Simple, analytically tractable
│  │  │
│  │  └─ Empirical Reality: Approximately Poisson mid-day, deviates at edges
│  │
│  ├─ Hawkes Process (Self-Exciting Arrivals):
│  │  ├─ Improvement: Clusters of orders (not independent)
│  │  ├─ Mechanism:
│  │  │  ├─ Trend-following: One order arrival increases probability of next
│  │  │  ├─ Example: Price jump up → algos notice → buy more (cascade)
│  │  │  ├─ Positive feedback: Orders beget orders (clustering)
│  │  │  └─ Duration: Minutes-hours (memory of prior flows)
│  │  │
│  │  ├─ Mathematical:
│  │  │  ├─ λ(t) = λ₀ + α∫ φ(t-τ) dN(τ)
│  │  │  ├─ λ₀ = baseline rate
│  │  │  ├─ α = self-excitation strength
│  │  │  ├─ φ(τ) = decay kernel (memory of past orders)
│  │  │  └─ N(τ) = cumulative order count
│  │  │
│  │  ├─ Implication: Clustering → Queue dynamics less predictable (bunching)
│  │  └─ Evidence: Flash crashes (self-exciting order cascade)
│  │
│  ├─ Jump Processes (Structural Breaks):
│  │  ├─ Sudden regime changes (earnings announcements, data releases)
│  │  ├─ Order rate jumps (λ → 10λ instantly)
│  │  ├─ Cancellation spikes coincident
│  │  ├─ Queue dynamics: Completely different (shallow → deep or vice versa)
│  │  └─ Modeling: Add Poisson jump component
│  │
│  ├─ Intraday Patterns:
│  │  ├─ Open (9:30-10:00 AM):
│  │  │  ├─ High order arrival rate (overnight accumulation)
│  │  │  ├─ λ_open ≈ 2-5 × λ_midday (peak)
│  │  │  ├─ Execution: Faster (many counterparts)
│  │  │  ├─ Queue time: Minutes (crowded)
│  │  │  └─ Fill probability: Higher (good for liquidity seeker)
│  │  │
│  │  ├─ Mid-Day (11:00 AM-3:00 PM):
│  │  │  ├─ Moderate order arrival rate (steady state)
│  │  │  ├─ λ_midday = baseline (normalized)
│  │  │  ├─ Execution: Steady
│  │  │  ├─ Queue time: 1-5 minutes (stable)
│  │  │  └─ Fill probability: Medium (dependent on price)
│  │  │
│  │  └─ Close (3:00-4:00 PM):
│  │     ├─ High order arrival rate (final positioning)
│  │     ├─ λ_close ≈ 3 × λ_midday
│  │     ├─ Execution: Very fast (closing auction)
│  │     ├─ Queue time: Seconds (all-in-one clearing)
│  │     └─ Fill probability: Depends on clearing price (all-or-none)
│  │
│  └─ Order Type Mix:
│     ├─ Market orders: 30-40% of flow (impatient, remove liquidity)
│     ├─ Limit orders: 60-70% of flow (patient, provide liquidity)
│     ├─ Ratio determines: Depth (more limits → deeper book), volatility (more markets → volatile)
│     ├─ Example: High HFT activity → majority limit orders (market making), fast cancellations
│     └─ Dynamics: Ratio varies intraday (open → more markets; mid-day → balanced)
│
├─ Order Cancellation Dynamics:
│  ├─ Cancellation Rates:
│  │  ├─ Normal periods:
│  │  │  ├─ Limit orders: 5-20% cancellation rate per minute
│  │  │  ├─ Example: Order submitted, 95-5% chance survives next minute
│  │  │  ├─ Varies by stock (AAPL ~5%, micro-cap ~30%)
│  │  │  └─ Interpretation: High rate → book turnover fast (shallow book)
│  │  │
│  │  ├─ Stressed periods:
│  │  │  ├─ Limit orders: 50%+ cancellation per minute
│  │  │  ├─ Reason: Traders revising prices, fear of execution
│  │  │  ├─ Example: VIX spike → 50% cancellations (risk-off)
│  │  │  ├─ Book becomes very shallow (illiquid)
│  │  │  └─ Flash crash risk increases
│  │  │
│  │  └─ Market Dynamics:
│  │     ├─ Mechanical cancellation: Price moves away (stale order, delete to avoid bad fill)
│  │     ├─ Strategic cancellation: Trader reconsidering offer price (market moved)
│  │     ├─ Inventory management: Market maker adjusting positions
│  │     └─ Gaming: Spoofing (quote then cancel to manipulate price)
│  │
│  ├─ Cancellation Drivers:
│  │  ├─ Price Movement:
│  │  │  ├─ If bid price moves UP → sellers cancel (orders obsolete)
│  │  │  ├─ If bid price moves DOWN → buyers likely to sell instead (better prices)
│  │  │  ├─ Larger moves → higher cancellation rate
│  │  │  ├─ Correlation: Volatility ↑ → cancellation rate ↑
│  │  │  └─ Example: 1% intraday vol → ~10% cancellation; 5% vol → ~40% cancellation
│  │  │
│  │  ├─ Information Events:
│  │  │  ├─ Earnings announcements: Bulk cancellations pre-announcement
│  │  │  ├─ Economic data releases: Flash spike in cancellations
│  │  │  ├─ Sector news: Relevant stocks see cancellation spikes
│  │  │  └─ Behavior: Traders reducing exposure (risk-off)
│  │  │
│  │  ├─ Market Stress:
│  │  │  ├─ Liquidity crises: Massive cancellations (50%+ per second)
│  │  │  ├─ Example: 2008 crash, 2020 COVID crash, 2015 China devaluation
│  │  │  ├─ Mechanism: Dealers withdraw (stop providing quotes)
│  │  │  ├─ Consequence: Book evaporates (zero depth)
│  │  │  └─ Recovery: Hours-days for order book to rebuild
│  │  │
│  │  └─ Trader Behavior:
│  │     ├─ Patient traders: Low cancellation (committed to price)
│  │     ├─ Impatient traders: High cancellation (constantly repricing)
│  │     ├─ HFT: Extremely high (re-quote 100s times/second)
│  │     └─ Institutional: Medium (hold orders longer, less frequent repricing)
│  │
│  └─ Implications:
│     ├─ High cancellation → Book shallow → Execution slow (fewer counterparts)
│     ├─ Low cancellation → Book stable → Execution fast
│     ├─ Predictor of stress: Monitor cancellation rate (early warning)
│     └─ Risk management: Abandon orders if cancellation rate spikes
│
├─ Fill Probability & Execution Time:
│  ├─ Queueing Model (M/M/1 Queue):
│  │  ├─ M: Poisson arrivals (λ)
│  │  ├─ M: Exponential service (departures, μ)
│  │  ├─ 1: Single server (one price level)
│  │  ├─ Utilization: ρ = λ / μ
│  │  ├─ Your Order = Server Capacity:
│  │  │  ├─ If ρ < 1 (arrivals < departures): Queue clears, you fill eventually
│  │  │  ├─ If ρ > 1 (arrivals > departures): Queue grows indefinitely
│  │  │  ├─ If ρ ≈ 1 (balanced): Queue fluctuates
│  │  │  └─ Example: μ = 2 orders/sec (departures), λ = 1 order/sec (arrivals) → ρ = 0.5 (good)
│  │  │
│  │  ├─ Mean Wait Time:
│  │  │  ├─ E[W] = λ / (μ(μ - λ))
│  │  │  ├─ Example: λ = 1, μ = 2 → E[W] = 1 / (2×1) = 0.5 units
│  │  │  ├─ Sensitivity: Small changes in λ or μ cause large W changes (unstable)
│  │  │  └─ Practical: Near capacity (ρ → 1) → wait times explode
│  │  │
│  │  └─ Fill Probability:
│  │     ├─ P(fill in time t) = 1 - (1 - e^(-(μ-λ)t))
│  │     ├─ Example: λ=0.5, μ=1, t=1 min → P(fill) ≈ 39%
│  │     ├─ Higher μ or lower λ → fills faster
│  │     └─ Practical use: Predict execution in 1, 5, 10 minutes
│  │
│  ├─ Queue Position Effect:
│  │  ├─ Queue size (n) ahead directly increases wait time
│  │  ├─ Each ahead order increases wait by ~1/μ
│  │  ├─ Example: 100 orders ahead, μ=1/sec → ~100 sec expected
│  │  ├─ Probability of front (fill soon): Exponential in queue depth
│  │  └─ Strategy: Submit when queue small (mid-day, off-peak)
│  │
│  ├─ Price Level Impact:
│  │  ├─ Best bid/ask: Highest λ (many counterparts)
│  │  ├─ Secondary levels: Lower λ (fewer potential executions)
│  │  ├─ Deep levels: Lowest λ (price must move significantly)
│  │  ├─ Price improvement cost: Slower execution (tradeoff)
│  │  └─ Strategy: Best bid has best execution speed, worst price; deep levels vice versa
│  │
│  ├─ Empirical Fill Times:
│  │  ├─ Best bid (AAPL, mid-day): 95% fill within 5 min
│  │  ├─ Secondary bid (10 cents away): 70% fill within 5 min
│  │  ├─ Deep bid (30 cents away): 30% fill within 5 min
│  │  ├─ Highly stock/time dependent
│  │  └─ Use as baseline for algorithm tuning
│  │
│  └─ Dynamic Update:
│     ├─ Cancellation changes probability (if ahead canceled, you move up)
│     ├─ New arrivals increase probability (but deepen queue ahead)
│     ├─ Price moves: May benefit (if price moves to you) or hurt (if away from you)
│     └─ Monitor in real-time to adjust expectations
│
├─ Practical Applications:
│  ├─ Market Maker Quoting:
│  │  ├─ Quote at best bid/ask to be first in queue (highest λ)
│  │  ├─ Set width: Spread = inventory cost + adverse selection + queue value
│  │  ├─ Adjust spread based on queue depth (deeper → wider spread acceptable)
│  │  ├─ Adjust spread based on cancellation rate (higher → wider to compensate)
│  │  └─ Dynamic: Use Avellaneda-Stoikov model to optimize
│  │
│  ├─ Execution Algorithm Design:
│  │  ├─ Slice large orders to stay in queue
│  │  ├─ Monitor queue position (move up as others cancel)
│  │  ├─ Repricing: Adjust limit price within spreads to optimize
│  │  ├─ Switch from limit to market if queue too deep
│  │  └─ Parameterize by arrival/cancellation rates
│  │
│  ├─ Venue Selection:
│  │  ├─ Compare queue dynamics across venues (consolidated vs. fragmented)
│  │  ├─ Route to venue with shortest queue at best prices
│  │  ├─ Track queue depth evolution (if growing, abandon venue)
│  │  └─ SOR (Smart Order Routing) automates this
│  │
│  ├─ Price Prediction:
│  │  ├─ Use order flow (arrivals, cancellations) to predict next price move
│  │  ├─ Imbalance metric: (Bid volume - Ask volume) predicts up/down (Kyle model)
│  │  ├─ Cancellation asymmetry: More ask cancels → price likely up
│  │  ├─ Front-running: Detect via order flow patterns (HFT detection)
│  │  └─ Features for ML: Queue depth, cancellation rate, arrival rate
│  │
│  └─ Risk Management:
│     ├─ Alert: If queue depth shrinks suddenly (cancellation spike)
│     ├─ Alert: If cancellation rate exceeds threshold (stress indicator)
│     ├─ Alert: If order arrival rate drops (illiquidity risk)
│     ├─ Reduce order size if queue deteriorates
│     └─ Abandon order book liquidity if conditions turn adverse
│
├─ Microstructure Phenomena:
│  ├─ Queue Fading:
│  │  ├─ Definition: As your queue position improves (others cancel), price moves away
│  │  ├─ Mechanism: Price movements and queue dynamics anti-correlated
│  │  ├─ Example: Bid queue shrinks but bid price drops too
│  │  ├─ Result: Your limit order at old price now way behind market
│  │  └─ Implication: Don't rely solely on queue position improving
│  │
│  ├─ Adverse Selection in Queue:
│  │  ├─ If you're behind best bid and price drops, you're now better off (below market)
│  │  ├─ But if price drops, it's likely bad news (informed selling)
│  │  ├─ Being "lucky" to fill (price moved to you) often means bad fill (informed)
│  │  ├─ Selection bias: Fills tend to be on adverse side
│  │  └─ Implication: Execute aggressively to avoid adverse selection
│  │
│  ├─ Front-Running & Toxicity:
│  │  ├─ HFT detects large limit order in queue
│  │  ├─ HFT trades ahead (predicting your order will move price)
│  │  ├─ Your limit order executes at worse price than if you went first
│  │  ├─ HFT profits from latency advantage
│  │  └─ Defense: Use dark pools, hide order size, vary execution timing
│  │
│  ├─ Queue Jumping (Penny Jumpers):
│  │  ├─ Submit order at best price + 1 tick (queue jump)
│  │  ├─ Execute first (at slight worse price, but higher probability)
│  │  ├─ Classic HFT strategy (speed matters more than price)
│  │  ├─ Profitable if: Speed gain > tick loss
│  │  └─ Regulation: Tick size changes impact strategy effectiveness
│  │
│  └─ Flash Crashes via Queue Evaporation:
│     ├─ Mechanism:
│     │  ├─ Large sell order hits bid (market order)
│     │  ├─ Queue at bid depletes (limit orders fill)
│     │  ├─ Spreads widen (fewer sellers willing to post)
│     │  ├─ Next market order faces wider spread → larger fill at worse price
│     │  ├─ Price drops more, panic selling begins
│     │  ├─ Queue at ask depletes too (cascading)
│     │  ├─ Book goes from deep to zero depth (seconds)
│     │  └─ Price crashes 10%+, recovers in minutes
│     │
│     ├─ Amplification:
│     │  ├─ Algorithmic execution algo (e.g., VWAP) accelerates (market moves against it)
│     │  ├─ Algos all selling simultaneously (crowding)
│     │  ├─ Queue can't absorb all orders
│     │  ├─ Circuit breakers kick in (trading halt)
│     │  └─ Then queue rebuilds, flash reverses
│     │
│     └─ Prevention:
│        ├─ Circuit breakers (halt trading if price drops >X%)
│        ├─ Speed limits (prevent algos accelerating uncontrollably)
│        ├─ Liquidity provider participation (required)
│        └─ Monitoring tools (early warning of queue collapse)
│
└─ Data & Measurement:
   ├─ Lobster Dataset:
   │  ├─ Academic limited order book data
   │  ├─ Full message history (orders, cancellations, trades)
   │  ├─ Time-stamped to nanoseconds
   │  ├─ 10 price levels tracked
   │  └─ Research standard (reproducible)
   │
   ├─ Metrics:
   │  ├─ Queue depth: Σ(quantity) at price level
   │  ├─ Queue position: Your order rank in queue
   │  ├─ Arrival rate: λ = orders per unit time
   │  ├─ Cancellation rate: % orders withdrawn per unit time
   │  ├─ Effective spread: 2×|trade_price - midpoint|
   │  ├─ Half-life: Time for half of queue to clear
   │  └─ Fill probability: P(execution within t minutes)
   │
   └─ Real-Time Monitoring:
      ├─ Update queue position/depth every message (nanosecond frequency)
      ├─ Compute running estimates of λ, μ, cancellation rates
      ├─ Alert if thresholds breached (deep → shallow, high cancellation)
      ├─ Adjust execution strategy based on current regime
      └─ Backtest algorithms against historical queue data
```

**Interaction:** Submit order → Queue forms → Monitor position/arrivals/cancellations → Update fill probability → Adjust strategy if needed → Execute or cancel.

## 5. Mini-Project
Implement LOB dynamics simulator and fill probability forecaster:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import stats as sp_stats

# Simulate limit order book dynamics
np.random.seed(42)

class LimitOrderBook:
    def __init__(self, initial_bid=100.00, initial_ask=100.01, depth=10):
        # Bid side: list of (price, quantity, order_id)
        self.bid_book = [(initial_bid - 0.01*i, 100*(1+i), i) 
                         for i in range(depth)]
        self.ask_book = [(initial_ask + 0.01*i, 100*(1+i), i+1000) 
                         for i in range(depth)]
        self.order_id_counter = 2000
        self.trade_history = []
        
    def best_bid(self):
        return self.bid_book[0][0] if self.bid_book else None
    
    def best_ask(self):
        return self.ask_book[0][0] if self.ask_book else None
    
    def mid_price(self):
        bb = self.best_bid()
        ba = self.best_ask()
        if bb and ba:
            return (bb + ba) / 2
        return None
    
    def add_limit_buy(self, price, quantity):
        """Add limit buy order"""
        # Passive: Order goes to queue
        for i, (p, q, oid) in enumerate(self.bid_book):
            if p == price:
                self.bid_book[i] = (p, q + quantity, oid)
                return
        # New price level
        self.bid_book.append((price, quantity, self.order_id_counter))
        self.bid_book.sort(reverse=True)
        self.order_id_counter += 1
    
    def cancel_orders(self, cancellation_rate=0.10):
        """Random cancellation"""
        new_bid_book = []
        for price, qty, oid in self.bid_book:
            if np.random.random() > cancellation_rate:
                new_bid_book.append((price, qty, oid))
        self.bid_book = new_bid_book
        
        new_ask_book = []
        for price, qty, oid in self.ask_book:
            if np.random.random() > cancellation_rate:
                new_ask_book.append((price, qty, oid))
        self.ask_book = new_ask_book
    
    def market_order(self, side, quantity):
        """Execute market order (consumes book)"""
        if side == 'buy':
            fills = []
            remaining = quantity
            for i, (p, q, oid) in enumerate(self.ask_book):
                fill_qty = min(q, remaining)
                fills.append((p, fill_qty))
                remaining -= fill_qty
                if remaining == 0:
                    break
            self.ask_book = self.ask_book[len(fills):]
            return fills
        else:
            fills = []
            remaining = quantity
            for i, (p, q, oid) in enumerate(self.bid_book):
                fill_qty = min(q, remaining)
                fills.append((p, fill_qty))
                remaining -= fill_qty
                if remaining == 0:
                    break
            self.bid_book = self.bid_book[len(fills):]
            return fills
    
    def book_depth(self, price, side='bid', levels=5):
        """Total quantity available at/better than price"""
        if side == 'bid':
            qty = sum(q for p, q, _ in self.bid_book if p >= price)
        else:
            qty = sum(q for p, q, _ in self.ask_book if p <= price)
        return qty
    
    def queue_size(self, price, side='bid'):
        """Quantity ahead in queue at specific price"""
        if side == 'bid':
            for i, (p, q, _) in enumerate(self.bid_book):
                if p == price:
                    return (i, q)  # Position, quantity at level
        else:
            for i, (p, q, _) in enumerate(self.ask_book):
                if p == price:
                    return (i, q)
        return (None, None)

# Simulate LOB evolution
print("="*100)
print("LIMIT ORDER BOOK DYNAMICS ANALYSIS")
print("="*100)

lob = LimitOrderBook()
n_periods = 100

# Track metrics
metrics = []

print(f"\nStep 1: LOB Initialization")
print(f"-" * 50)
print(f"Best bid: ${lob.best_bid():.2f}")
print(f"Best ask: ${lob.best_ask():.2f}")
print(f"Mid-price: ${lob.mid_price():.2f}")
print(f"Spread: {(lob.best_ask() - lob.best_bid()) * 100:.1f} cents")

# Submit a limit buy order
limit_buy_price = lob.best_bid()
limit_buy_qty = 500
lob.add_limit_buy(limit_buy_price, limit_buy_qty)
print(f"\nSubmitted limit buy order: {limit_buy_qty} @ ${limit_buy_price:.2f}")

# Simulate time evolution
print(f"\nStep 2: Simulating LOB Evolution (100 periods)")
print(f"-" * 50)

for t in range(n_periods):
    # Arrivals
    if np.random.random() < 0.6:  # 60% chance new order
        side = 'bid' if np.random.random() > 0.4 else 'ask'
        if side == 'bid':
            new_price = lob.best_bid() - np.random.uniform(0, 0.02)
            new_qty = 100 * np.random.randint(1, 5)
            lob.add_limit_buy(new_price, new_qty)
    
    # Market order (10% chance)
    if np.random.random() < 0.10:
        side = 'buy' if np.random.random() > 0.5 else 'sell'
        qty = 100 * np.random.randint(1, 3)
        fills = lob.market_order(side, qty)
    
    # Cancellations
    cancellation_rate = 0.05 + 0.10 * np.random.random()  # 5-15%
    lob.cancel_orders(cancellation_rate)
    
    # Queue position of our order
    position, qty_at_level = lob.queue_size(limit_buy_price, 'bid')
    
    metrics.append({
        'period': t,
        'mid_price': lob.mid_price(),
        'best_bid': lob.best_bid(),
        'best_ask': lob.best_ask(),
        'spread': (lob.best_ask() - lob.best_bid()) * 100 if lob.best_ask() else None,
        'bid_depth': lob.book_depth(limit_buy_price, 'bid'),
        'ask_depth': lob.book_depth(limit_buy_price, 'ask'),
        'queue_position': position,
        'qty_at_level': qty_at_level,
        'bid_book_size': len(lob.bid_book),
        'cancellation_rate': cancellation_rate,
    })

df_metrics = pd.DataFrame(metrics)

print(f"Total periods simulated: {len(df_metrics)}")
print(f"Final mid-price: ${df_metrics['mid_price'].iloc[-1]:.2f}")
print(f"Final queue position: {df_metrics['queue_position'].iloc[-1]}")
print(f"Final bid depth: {df_metrics['bid_depth'].iloc[-1]:.0f} shares")

print(f"\nStep 3: Fill Probability Estimation")
print(f"-" * 50)

# Estimate fill probability based on queue position
def estimate_fill_probability(queue_position, queue_qty, cancellation_rate, periods_remaining):
    """Estimate probability of fill in remaining periods"""
    if queue_position is None:
        return 0
    if queue_position == 0:
        return 1.0  # Already at front
    
    # Probability increases as queue ahead cancels
    # Poisson model: λ = queue_ahead * cancellation_rate
    # Probability of being first in time t: 1 - (1 - cancellation_rate)^t
    lambda_param = queue_position * cancellation_rate
    fill_prob = 1 - np.exp(-lambda_param * periods_remaining)
    return np.clip(fill_prob, 0, 1)

# Calculate fill probability for different horizons
horizons = [5, 10, 20, 50]
fill_probs = {}

for horizon in horizons:
    current_period = 0
    final_cancellation_rate = df_metrics['cancellation_rate'].iloc[-1]
    final_queue_pos = df_metrics['queue_position'].iloc[-1]
    
    fill_prob = estimate_fill_probability(final_queue_pos, None, final_cancellation_rate, horizon)
    fill_probs[f'{horizon} periods'] = fill_prob

for horizon, prob in fill_probs.items():
    print(f"Fill probability within {horizon}: {prob*100:.1f}%")

print(f"\nStep 4: Queue Dynamics Statistics")
print(f"-" * 50)

print(f"Average queue position: {df_metrics['queue_position'].mean():.1f}")
print(f"Min queue position: {df_metrics['queue_position'].min()}")
print(f"Max queue position: {df_metrics['queue_position'].max()}")
print(f"Average bid spread: {df_metrics['spread'].mean():.2f} cents")
print(f"Bid depth volatility: {df_metrics['bid_depth'].std():.0f} shares")

# VISUALIZATION
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Price evolution
ax = axes[0, 0]
ax.plot(df_metrics['period'], df_metrics['mid_price'], label='Mid-Price', linewidth=2)
ax.axhline(y=limit_buy_price, color='red', linestyle='--', label='Your Limit Price')
ax.set_xlabel('Period')
ax.set_ylabel('Price ($)')
ax.set_title('Price Evolution')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Spread evolution
ax = axes[0, 1]
ax.plot(df_metrics['period'], df_metrics['spread'], linewidth=2, color='purple')
ax.set_xlabel('Period')
ax.set_ylabel('Spread (cents)')
ax.set_title('Bid-Ask Spread Over Time')
ax.grid(alpha=0.3)

# Plot 3: Queue position
ax = axes[0, 2]
ax.plot(df_metrics['period'], df_metrics['queue_position'], linewidth=2, color='orange')
ax.set_xlabel('Period')
ax.set_ylabel('Queue Position (rank)')
ax.set_title('Your Limit Order Queue Position')
ax.grid(alpha=0.3)

# Plot 4: Bid depth
ax = axes[1, 0]
ax.plot(df_metrics['period'], df_metrics['bid_depth'], linewidth=2, color='green')
ax.set_xlabel('Period')
ax.set_ylabel('Bid Depth (shares)')
ax.set_title('Liquidity Available at Your Price')
ax.grid(alpha=0.3)

# Plot 5: Order book size
ax = axes[1, 1]
ax.plot(df_metrics['period'], df_metrics['bid_book_size'], linewidth=2, color='blue')
ax.set_xlabel('Period')
ax.set_ylabel('Number of Bid Levels')
ax.set_title('Order Book Depth (Price Levels)')
ax.grid(alpha=0.3)

# Plot 6: Cumulative fill probability
ax = axes[1, 2]
periods = df_metrics['period'].values
fill_probs_cumulative = []
for p in periods:
    remaining = n_periods - p
    final_cancel_rate = df_metrics['cancellation_rate'].iloc[-1]
    final_queue = df_metrics['queue_position'].iloc[p]
    fill_prob = estimate_fill_probability(final_queue, None, final_cancel_rate, remaining)
    fill_probs_cumulative.append(fill_prob * 100)

ax.plot(periods, fill_probs_cumulative, linewidth=2, color='red')
ax.set_xlabel('Period')
ax.set_ylabel('Fill Probability (%)')
ax.set_title('Estimated Fill Probability Over Time')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("INSIGHTS")
print(f"="*100)
print(f"- Queue position improved (closer to front) as others cancelled")
print(f"- Spread widened when order book became shallow")
print(f"- Fill probability estimated via Poisson cancellation model")
print(f"- Trade-off: Your limit price (chance to fill) vs. market price (certainty)")
print(f"- Key driver: Cancellation rate (high → faster fills, volatile environment)")
```

## 6. Challenge Round
- Build LOB predictor: Use order flow to forecast next price move (imbalance → directional signal)
- Model fill probability: Incorporate queue depth, cancellation rate, price movement; backtest accuracy
- Design "smart cancel" algorithm: Monitor queue; cancel if position deteriorates past threshold
- Analyze flash crash LOB: Simulate queue evaporation and cascade selling
- Compare execution strategies: Market order vs. limit order vs. VWAP based on LOB dynamics

## 7. Key References
- [Cont et al (2010), "The Price Impact of Order Book Events," Journal of Econometrics](https://arxiv.org/abs/1003.3796) — Empirical LOB dynamics and price impact
- [Foucault et al (2005), "Order Flow, Transaction Clock, and Normality of Asset Returns," Journal of Finance](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2005.00796.x) — Queue priority and execution dynamics
- [Mertens et al (2015), "Lobster: Limit Order Book Reconstruction System," IEEE](https://academic.oup.com/rfs/article-abstract/27/8/2267/1582754) — LOB data processing and analysis
- [Cartea et al (2015), "Algorithmic and High-Frequency Trading," Cambridge Press](https://www.amazon.com/Algorithmic-High-Frequency-Trading-Mathematics-Finance/dp/1107091144) — Practical LOB modeling

---
**Status:** Foundational microstructure concept (critical for market making & execution) | **Complements:** Market Impact, Execution Algorithms, Price Discovery, Order Types
