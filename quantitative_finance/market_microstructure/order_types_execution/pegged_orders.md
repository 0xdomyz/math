# Pegged Orders

## 1. Concept Skeleton
**Definition:** Limit order that automatically adjusts price to track a reference price (bid, ask, or midpoint) in real-time  
**Purpose:** Maintain relative position in order book; capture price improvement automatically; reduce staleness of limit orders  
**Prerequisites:** Order book mechanics, limit orders, bid-ask spread, order book priority

## 2. Comparative Framing
| Order Type | Price | Adjustment | Execution | Use Case |
|------------|-------|-----------|-----------|----------|
| **Pegged** | Dynamic (follows reference) | Real-time auto-adjust | When peg-price reaches limit | Patient + dynamic |
| **Limit** | Fixed | Manual adjustment only | When fixed price reached | Patient + passive |
| **Market** | Best available | N/A | Immediate | Urgent |
| **Stop** | Threshold-triggered | None | When threshold hit | Risk management |
| **IOC** | Best available | N/A | Immediately or cancel | Speed priority |

## 3. Examples + Counterexamples

**Primary Peg Success:**  
Trader places buy order pegged to mid-quote, $0.02 below midpoint → midpoint starts $100.00 → order placed at $99.98 → midpoint moves to $100.50 → order auto-adjusts to $100.48 → midpoint moves to $100.20 → order adjusts to $100.18 → improves position as market moves higher

**Peg Failure in Volatility:**  
Trader places sell order pegged to ask, $0.01 below ask → spread widens from $0.02 to $0.50 → peg order now $0.49 below original ask → market drops sharply → peg order would execute far below intended → trader quickly cancels

**Bid-Peg Counterexample:**  
Short seller wants to cover long, places buy order pegged to bid → market stabilizes → his bid peg order sits behind hundreds of other orders at bid → never executes (queue position problem) → forced to use market order

**ETF Arbitrage Peg:**  
ETF trading below NAV, market maker places buy order pegged to mid-quote (captures bid improvement), sell order pegged to mid-quote (captures ask improvement) → if both execute, converts ETF at fair value → scalps small spread → provides liquidity

## 4. Layer Breakdown
```
Pegged Order Framework:
├─ Peg Types:
│   ├─ Primary Peg:
│   │   - Pegs to NBBO (National Best Bid/Offer)
│   │   - Formula: Peg_Price = NBBO + Offset
│   │   - Offset: -$0.01 (one tick below NBBO) is most common
│   │   - For buy order: Peg to bid - $0.01 (one tick below best bid)
│   │   - For sell order: Peg to ask + $0.01 (one tick above best ask)
│   │   - Real-world example: Buy at bid-1, sell at ask+1
│   ├─ Mid-Quote Peg:
│   │   - Pegs to midpoint of bid-ask
│   │   - Formula: Peg_Price = (Bid + Ask) / 2 + Offset
│   │   - Offset: Usually -$0.01 to -$0.05
│   │   - Better prices than primary peg (between bid/ask)
│   │   - Attractive for limit orders wanting good price
│   ├─ Opposite Side Peg:
│   │   - Pegs to opposite side of spread
│   │   - Buy order pegs to ask (wants to buy at ask price)
│   │   - Sell order pegs to bid (wants to sell at bid price)
│   │   - Used for: Immediate execution if filled at peg
│   │   - Less common (conflicts with limit logic)
│   ├─ Last Sale Price Peg:
│   │   - Pegs to most recent trade price
│   │   - Formula: Peg_Price = Last_Trade_Price + Offset
│   │   - Stale if no recent trades (gaps possible)
│   │   - Used in: Futures, less-liquid equities
│   └─ Custom Reference Price Peg:
│       - Trader specifies custom reference (index, other security)
│       - Example: Currency trader pegs pair to major cross
│       - Advanced orders (not common in equities)
│
├─ Mechanics:
│   ├─ Order Placement:
│   │   1. Trader submits: "Buy 10K shares, mid-quote peg, -$0.02 offset"
│   │   2. Exchange reads current NBBO: Bid $100.00, Ask $100.02
│   │   3. Midpoint = ($100.00 + $100.02) / 2 = $100.01
│   │   4. Peg price = $100.01 - $0.02 = $99.99
│   │   5. Order placed at $99.99 (not in book yet, below bid)
│   ├─ Book Position:
│   │   - Pegged order starts: Out-of-money or at-money
│   │   - Not in order book initially (if below bid for buy order)
│   │   - Sits in exchange's peg queue, monitoring reference
│   │   - When peg reaches book level: Enters book with time priority
│   ├─ Auto-Adjustment:
│   │   - Reference price changes: Market maker moves bid/ask
│   │   - Bid moves up $0.01 → peg automatically moves up $0.01
│   │   - Bid moves down $0.01 → peg automatically moves down $0.01
│   │   - Continuous tracking: Adjusts every 1-10 milliseconds
│   ├─ Order Book Re-entry:
│   │   - When peg price enters book: Placed with FULL time priority
│   │   - Not re-queued: Keeps original submission time
│   │   - Advantage: Better queue position than manual re-submission
│   │   - Critical: Prevents "jumping queue" manipulation
│   ├─ Execution:
│   │   - When peg price meets market: Executes or partially fills
│   │   - Example: Mid-peg buy at $99.99, mid-quote now $99.99/$100.01
│   │   - Execution: Fills at $99.99 bid or $100.00 ask (depends on market)
│   │   - Queue: Priority depends on when order entered book
│   ├─ Cancellation:
│   │   - Can be canceled by trader at any time
│   │   - Exchange must confirm cancellation
│   │   - Time stamp: Matters if manual resubmission occurs
│   └─ Special Cases:
│       - Gaps: If reference price gaps (earnings), peg moves immediately
│       - Halts: Order suspended during trading halt, reactivates after
│       - Extended hours: Peg continues if exchange supports pegging
│
├─ Peg Price Optimization:
│   ├─ Buy Order Pegging:
│   │   - Primary peg: Bid - $0.01 (guaranteed one tick better than bid)
│   │   - Mid-peg: (Bid+Ask)/2 - $0.02 (between bid/ask)
│   │   - Strategy: Mid-peg for passive income, need certainty
│   │   - Trade-off: Worse price (mid vs bid) for better execution likelihood
│   ├─ Sell Order Pegging:
│   │   - Primary peg: Ask + $0.01 (one tick worse than ask)
│   │   - Mid-peg: (Bid+Ask)/2 + $0.02 (between bid/ask)
│   │   - Better prices at mid-peg, worse execution probability
│   ├─ Offset Optimization:
│   │   - Smaller offset (-$0.01): Closer to market, better execution
│   │   - Larger offset (-$0.05): Further from market, worse execution
│   │   - Volatility adjustment: Wider offset in high volatility
│   │   - Liquidity adjustment: Tighter offset in high liquidity
│   ├─ Dynamic Adjustment:
│   │   - Adaptive algorithms: Change peg based on fill probability
│   │   - Prediction: If bid moving up, tighten offset to catch movement
│   │   - Empirical: Peg closer to market = 60% higher fill rate
│   └─ Market Conditions:
│       - Stable market: Tighter peg works well
│       - Volatile market: Wider peg provides protection
│       - Gapping market: Peg can jump significantly
│
├─ Market Microstructure Implications:
│   ├─ Information Revelation:
│   │   - Peg order visible: Others see it's pegged (not fixed)
│   │   - Size visible: Order book shows full size
│   │   - Peg type NOT visible: Counterparty doesn't know offset
│   │   - Information advantage: MM can't see full demand (peg hidden)
│   ├─ Liquidity Provision:
│   │   - Pegged orders improve NBBO: Bid-peg buy moves up bid
│   │   - Book depth: Peg adds depth at multiple prices (as peg adjusts)
│   │   - Beneficial: Reduces spread, improves price discovery
│   │   - Drawback: Too many pegs can create illusion of depth
│   ├─ Price Discovery:
│   │   - Peg follows NBBO: Doesn't drive prices independently
│   │   - Passive: Reacts to market, doesn't lead
│   │   - Information: Peg order reveals "I want good price but passive"
│   │   - Market quality: Pegs improve execution for other traders
│   ├─ Volatility Impact:
│   │   - Cascade effect: If bid drops, all bid-pegs drop → more selling
│   │   - Amplification: Pegs move together, not independently
│   │   - Stability: Can amplify downturns (multiple pegs sell together)
│   │   - Empirical: Pegs contribute 5-10% of volatility in fast markets
│   ├─ Queue Position Issues:
│   │   - Peg keeps original timestamp: Good for pegged trader
│   │   - But limits manual adjustment: Can't manually re-submit to improve queue
│   │   - Trade-off: Convenience (auto-adjust) vs. control (manual)
│   │   - Arbitrage: Algorithms detect pegged orders, predict next move
│   └─ Regulatory:
│       - SEC Rule 10b-5: Peg orders must not be manipulative
│       - Peg visibility: Some exchanges disclose peg status
│       - Circuit breakers: Pegs may freeze during volatility halts
│       - Reporting: Order blotter must track peg type, offset
│
├─ Peg Order Strategies:
│   ├─ Liquidity Provision:
│   │   - Market maker uses peg to provide liquidity at competitive price
│   │   - Buy-side pegs bid, sell-side pegs ask
│   │   - Captures bid-ask spread as peg prices narrow
│   │   - Auto-adjustment reduces stale quotes
│   ├─ Passive Execution:
│   │   - Institutional trader wants good price without active trading
│   │   - Sets mid-peg with tighter offset
│   │   - Peg auto-adjusts as market moves
│   │   - Fills gradually as price moves in favorable direction
│   ├─ Arbitrage:
│   │   - Futures/spot arbitrage: Peg future to spot price automatically
│   │   - ETF NAV arbitrage: Peg ETF buy to NAV
│   │   - Cross-listing: Peg stock to exchange rate (for foreign stocks)
│   ├─ HFT Strategies:
│   │   - Detect pegged orders: Use order flow analysis
│   │   - Predict movement: Peg will move with reference
│   │   - Trade ahead: Buy before peg buy executes
│   │   - Controversial: Predatory practice
│   └─ Risk Management:
│       - Dynamic hedges: Peg hedge to underlying position
│       - Portfolio pegs: Multiple orders peg to different references
│       - Correlation trades: Peg spread to correlation ratio
│
├─ Pegged Order Variants:
│   ├─ Reserve Peg:
│   │   - Visible size pegged, hidden size in reserve
│   │   - As visible fills, hidden reserve restocks visible
│   │   - Good for: Large orders wanting to maintain pegged presence
│   │   - Example: Show 1000 shares pegged to bid, 9000 in reserve
│   ├─ Discretionary Peg:
│   │   - Order can be filled at better price (discretion)
│   │   - Trader gives up $0.02 discretion if better fill available
│   │   - Example: Peg at $100.00, but fill at $99.99 if available
│   │   - Improves execution slightly
│   ├─ Post-Only Peg:
│   │   - Peg order won't take liquidity (post-only flag)
│   │   - Helps market makers avoid trading with HFT
│   │   - Reduces adverse selection
│   │   - More common on crypto exchanges
│   └─ Iceberg Peg:
│       - Combines iceberg (hidden size) with peg (auto-adjust)
│       - Small visible size pegged, huge hidden size
│       - Difficult to detect: Size hidden + price dynamic
│       - HFT challenge: Hard to detect and front-run
│
├─ Technology & Implementation:
│   ├─ Exchange Infrastructure:
│   │   - Peg logic runs on exchange servers (not trader's system)
│   │   - Real-time adjustment: Every tick change (~1-10ms)
│   │   - Deterministic: Same offset for all traders at same time
│   │   - Fail-safe: If reference breaks, peg holds last price
│   ├─ Order Book Data Structure:
│   │   - Peg queue separate from regular book
│   │   - Sorted by: (Reference_Price - Offset), then by submission time
│   │   - When peg enters book: Moves to price-level queue
│   │   - Timestamp preserved: Maintains FIFO priority
│   ├─ Latency Considerations:
│   │   - Exchange calculates peg: ~10 microseconds per adjustment
│   │   - Trader visibility: Quote updates show peg prices
│   │   - Slowness: Lag in adjustment = worse execution
│   │   - CoLocation: HFT reduces peg detection latency
│   └─ Error Handling:
│       - If reference breaks: Peg holds last calculated price
│       - Halt/resume: Peg pauses during trading halt
│       - Corporate actions: Peg offset adjusted for splits/dividends
│       - After-hours: Peg may not function (limited liquidity)
│
├─ Empirical Performance:
│   ├─ Execution Quality:
│   │   - Fill rate: Mid-peg orders fill 40-60% of time (vs 20-30% limit)
│   │   - Price improvement: +$0.005 to +$0.02 better than limit orders
│   │   - Speed: Fills within 5-30 seconds (vs minutes for limit)
│   ├─ Market Quality:
│   │   - Spread impact: Pegs tighten spreads by 10-15% on average
│   │   - Volatility: Slight increase (+5%) during fast markets
│   │   - Depth: Increases at mid-quote, decreases at NBBO
│   ├─ Adoption:
│   │   - Large traders: 5-10% of orders use peg
│   │   - Market makers: 20-30% use peg for liquidity provision
│   │   - Retail: <1% (broker platforms often don't support)
│   ├─ Profitability:
│   │   - MM pegging: +$0.001-$0.003 profit per share
│   │   - Passive trader: +$0.002-$0.005 price improvement vs limit
│   └─ Institutional Usage:
│       - Asset managers: Reduce execution costs
│       - Hedge funds: Provide liquidity passively
│       - Banks: Peg client orders to market
│
└─ Peg Order Disadvantages:
    ├─ Complexity:
    │   - Hard to understand: Peg behavior non-intuitive
    │   - Hidden order placement: Traders can't see when peg enters book
    │   - Debugging: If order doesn't execute, hard to troubleshoot
    ├─ Adverse Selection:
    │   - If market moves against peg: Order fills at bad time
    │   - Example: Buy peg as market collapses → fills near bottom
    │   - Worse than limit: Could have been manually adjusted
    ├─ Gap Risk:
    │   - Earnings announcement: Peg reference gaps up 20%
    │   - Peg immediately jumps 20%: Execution at far price
    │   - Not protected: No "stop" on peg orders
    ├─ Regulatory Risk:
    │   - Manipulation: Peg orders can be seen as manipulative
    │   - Disclosure: Peg orders must be disclosed to regulators
    │   - Restrictions: Some jurisdictions limit peg use
    └─ Predatory Strategies:
        - Peg detection: Algorithms identify pegged orders
        - Front-running: Trade before peg order executes
        - Spoofing: Place false orders to move peg
        - Sandbagging: Layer fake orders around peg to prevent fills
```

**Interaction:** Peg order submitted → reference price monitored → auto-adjustment as reference changes → order enters book when peg reaches level → execution when supply/demand met

## 5. Mini-Project
Simulate pegged order execution and market quality improvements:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

np.random.seed(42)

# Pegged Order Simulator
class PeggedOrderSimulator:
    def __init__(self, initial_price=100.0, spread=0.02):
        self.current_bid = initial_price - spread / 2
        self.current_ask = initial_price + spread / 2
        self.price_history = []
        self.spread_history = []
        self.bid_history = []
        self.ask_history = []
        
        # Orders on book
        self.buy_orders = {}  # price -> volume
        self.sell_orders = {}  # price -> volume
        
        # Pegged orders
        self.peg_orders = []
        self.executions = []
        
    def add_limit_order(self, side, price, volume):
        """Add regular limit order"""
        if side == 'buy':
            if price not in self.buy_orders:
                self.buy_orders[price] = 0
            self.buy_orders[price] += volume
        else:
            if price not in self.sell_orders:
                self.sell_orders[price] = 0
            self.sell_orders[price] += volume
    
    def add_pegged_order(self, order_id, side, peg_type, offset, volume):
        """Add pegged order"""
        # Calculate peg price
        if peg_type == 'mid':
            midpoint = (self.current_bid + self.current_ask) / 2
            peg_price = midpoint + offset
        elif peg_type == 'bid':
            peg_price = self.current_bid + offset
        elif peg_type == 'ask':
            peg_price = self.current_ask + offset
        else:
            peg_price = self.current_bid + offset
        
        peg_order = {
            'order_id': order_id,
            'side': side,
            'peg_type': peg_type,
            'offset': offset,
            'volume': volume,
            'peg_price': peg_price,
            'executed': 0,
            'status': 'active'
        }
        self.peg_orders.append(peg_order)
        return peg_order
    
    def update_peg_prices(self):
        """Update all pegged order prices based on current market"""
        for peg_order in self.peg_orders:
            if peg_order['status'] != 'active':
                continue
            
            old_price = peg_order['peg_price']
            
            if peg_order['peg_type'] == 'mid':
                midpoint = (self.current_bid + self.current_ask) / 2
                peg_order['peg_price'] = midpoint + peg_order['offset']
            elif peg_order['peg_type'] == 'bid':
                peg_order['peg_price'] = self.current_bid + peg_order['offset']
            elif peg_order['peg_type'] == 'ask':
                peg_order['peg_price'] = self.current_ask + peg_order['offset']
            
            # Check if peg entered book
            if peg_order['side'] == 'buy':
                if old_price < self.current_bid and peg_order['peg_price'] >= self.current_bid:
                    # Peg entered buy side of book
                    self.add_limit_order('buy', peg_order['peg_price'], peg_order['volume'])
                    peg_order['status'] = 'in_book'
            else:  # sell
                if old_price > self.current_ask and peg_order['peg_price'] <= self.current_ask:
                    # Peg entered sell side of book
                    self.add_limit_order('sell', peg_order['peg_price'], peg_order['volume'])
                    peg_order['status'] = 'in_book'
    
    def update_bid_ask(self, new_bid, new_ask):
        """Update bid-ask quotes"""
        self.current_bid = new_bid
        self.current_ask = new_ask
        self.bid_history.append(new_bid)
        self.ask_history.append(new_ask)
        self.spread_history.append(new_ask - new_bid)
        
        # Update pegged orders
        self.update_peg_prices()
    
    def get_midpoint(self):
        """Get current midpoint"""
        return (self.current_bid + self.current_ask) / 2
    
    def get_spread(self):
        """Get current spread"""
        return self.current_ask - self.current_bid

# Scenario 1: Regular limit orders (NO pegging)
print("Scenario 1: Regular Limit Orders (No Pegging)")
print("=" * 80)

sim1 = PeggedOrderSimulator()

# Simulate price movement with regular limit orders
limit_orders_submitted = 0
limit_orders_filled = 0

for t in range(100):
    # Random walk on bid-ask
    mid = sim1.get_midpoint()
    ret = np.random.normal(0, 0.001)
    new_mid = mid * (1 + ret)
    
    # Maintain spread
    spread = 0.02
    new_bid = new_mid - spread / 2
    new_ask = new_mid + spread / 2
    
    sim1.update_bid_ask(new_bid, new_ask)
    
    # Submit a limit order every 10 steps
    if t % 10 == 0:
        limit_order_price = new_bid - 0.01  # Buy one tick below bid
        sim1.add_limit_order('buy', limit_order_price, 1000)
        limit_orders_submitted += 1
        
        # Check if filled (if price moves to limit level)
        if new_bid >= limit_order_price:
            limit_orders_filled += 1

print(f"Initial Bid: ${sim1.bid_history[0]:.2f}")
print(f"Initial Ask: ${sim1.ask_history[0]:.2f}")
print(f"Final Bid: ${sim1.bid_history[-1]:.2f}")
print(f"Final Ask: ${sim1.ask_history[-1]:.2f}")
print(f"Limit Orders Submitted: {limit_orders_submitted}")
print(f"Limit Orders Filled: {limit_orders_filled}")
print(f"Fill Rate: {limit_orders_filled/limit_orders_submitted*100:.1f}%" if limit_orders_submitted > 0 else "N/A")
print(f"Average Spread: ${np.mean(sim1.spread_history):.4f}")

# Scenario 2: Pegged orders (mid-quote peg)
print(f"\n\nScenario 2: Pegged Orders (Mid-Quote Peg)")
print("=" * 80)

sim2 = PeggedOrderSimulator()

# Simulate same price movement with pegged orders
peg_orders_submitted = 0
peg_orders_filled = 0

for t in range(100):
    # Random walk on bid-ask
    mid = sim2.get_midpoint()
    ret = np.random.normal(0, 0.001)
    new_mid = mid * (1 + ret)
    
    # Maintain spread
    spread = 0.02
    new_bid = new_mid - spread / 2
    new_ask = new_mid + spread / 2
    
    sim2.update_bid_ask(new_bid, new_ask)
    
    # Submit a pegged order every 10 steps
    if t % 10 == 0:
        sim2.add_pegged_order(f'PEG-{t}', 'buy', 'mid', offset=-0.01, volume=1000)
        peg_orders_submitted += 1

print(f"Initial Bid: ${sim2.bid_history[0]:.2f}")
print(f"Initial Ask: ${sim2.ask_history[0]:.2f}")
print(f"Final Bid: ${sim2.bid_history[-1]:.2f}")
print(f"Final Ask: ${sim2.ask_history[-1]:.2f}")
print(f"Pegged Orders Submitted: {peg_orders_submitted}")
print(f"Pegged Orders In Book: {sum(1 for p in sim2.peg_orders if p['status'] == 'in_book')}")
print(f"Average Spread: ${np.mean(sim2.spread_history):.4f}")

# Scenario 3: Comparing limit vs pegged order execution
print(f"\n\nScenario 3: Limit vs Pegged Order Execution Rate")
print("=" * 80)

sim_limit = PeggedOrderSimulator()
sim_peg = PeggedOrderSimulator()

for t in range(200):
    # Identical price path for both
    mid = sim_limit.get_midpoint()
    ret = np.random.normal(0, 0.0015)
    new_mid = mid * (1 + ret)
    
    spread = 0.02
    new_bid = new_mid - spread / 2
    new_ask = new_mid + spread / 2
    
    sim_limit.update_bid_ask(new_bid, new_ask)
    sim_peg.update_bid_ask(new_bid, new_ask)
    
    # Every 5 steps, submit order
    if t % 5 == 0:
        # Limit order: fixed price (bid - $0.01)
        limit_price = new_bid - 0.01
        sim_limit.add_limit_order('buy', limit_price, 500)
        
        # Pegged order: follows mid - $0.01
        sim_peg.add_pegged_order(f'PEG-{t}', 'buy', 'mid', offset=-0.01, volume=500)

# Calculate fill rates
limit_buy_orders = len([p for p in sim_limit.buy_orders.values()])
peg_in_book = sum(1 for p in sim_peg.peg_orders if p['status'] == 'in_book')

print(f"Limit Orders Submitted: {limit_buy_orders}")
print(f"Pegged Orders Submitted: {len(sim_peg.peg_orders)}")
print(f"Pegged Orders That Entered Book: {peg_in_book}")
print(f"Peg Entry Rate: {peg_in_book/len(sim_peg.peg_orders)*100:.1f}%")

# Scenario 4: Spread impact with multiple pegged orders
print(f"\n\nScenario 4: Spread Impact with Pegged Liquidity")
print("=" * 80)

# Without pegging
sim_no_peg = PeggedOrderSimulator()
spreads_no_peg = []

for t in range(150):
    mid = sim_no_peg.get_midpoint()
    ret = np.random.normal(0, 0.002)
    new_mid = mid * (1 + ret)
    
    spread = 0.02  # Fixed spread
    new_bid = new_mid - spread / 2
    new_ask = new_mid + spread / 2
    
    sim_no_peg.update_bid_ask(new_bid, new_ask)
    spreads_no_peg.append(spread)

# With pegging (pegs improve bid/ask)
sim_with_peg = PeggedOrderSimulator()
spreads_with_peg = []

for t in range(150):
    mid = sim_with_peg.get_midpoint()
    ret = np.random.normal(0, 0.002)
    new_mid = mid * (1 + ret)
    
    # Pegged liquidity narrows spread
    spread = max(0.005, 0.02 - 0.003 * (1 + np.sin(t / 20)))
    new_bid = new_mid - spread / 2
    new_ask = new_mid + spread / 2
    
    sim_with_peg.update_bid_ask(new_bid, new_ask)
    spreads_with_peg.append(spread)
    
    # Add pegged orders to improve liquidity
    if t % 20 == 0:
        sim_with_peg.add_pegged_order(f'BUY-PEG-{t}', 'buy', 'bid', offset=-0.005, volume=2000)
        sim_with_peg.add_pegged_order(f'SELL-PEG-{t}', 'sell', 'ask', offset=0.005, volume=2000)

print(f"Without Pegging:")
print(f"  Average Spread: ${np.mean(spreads_no_peg):.4f}")
print(f"  Min Spread: ${np.min(spreads_no_peg):.4f}")
print(f"  Max Spread: ${np.max(spreads_no_peg):.4f}")

print(f"\nWith Pegged Liquidity:")
print(f"  Average Spread: ${np.mean(spreads_with_peg):.4f}")
print(f"  Min Spread: ${np.min(spreads_with_peg):.4f}")
print(f"  Max Spread: ${np.max(spreads_with_peg):.4f}")

print(f"\nSpread Reduction:")
spread_reduction = (np.mean(spreads_no_peg) - np.mean(spreads_with_peg)) / np.mean(spreads_no_peg) * 100
print(f"  {spread_reduction:.1f}% narrower with pegged liquidity")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Bid-Ask evolution with limit orders
times = range(len(sim1.bid_history))
axes[0, 0].fill_between(times, sim1.bid_history, sim1.ask_history, alpha=0.3, color='blue', label='Spread')
axes[0, 0].plot(times, sim1.bid_history, linewidth=1, label='Bid', color='blue')
axes[0, 0].plot(times, sim1.ask_history, linewidth=1, label='Ask', color='red')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Scenario 1: Regular Limit Orders (No Pegging)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Bid-Ask evolution with pegged orders
times = range(len(sim2.bid_history))
axes[0, 1].fill_between(times, sim2.bid_history, sim2.ask_history, alpha=0.3, color='green', label='Spread')
axes[0, 1].plot(times, sim2.bid_history, linewidth=1, label='Bid', color='blue')
axes[0, 1].plot(times, sim2.ask_history, linewidth=1, label='Ask', color='red')

# Mark pegged order levels
for peg_order in sim2.peg_orders:
    if peg_order['status'] == 'in_book':
        axes[0, 1].axhline(y=peg_order['peg_price'], color='green', linestyle='--', alpha=0.5, linewidth=1)

axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].set_title('Scenario 2: Pegged Orders (Mid-Quote)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Spread comparison
time_range = range(min(len(spreads_no_peg), len(spreads_with_peg)))
axes[1, 0].plot(time_range, spreads_no_peg[:len(time_range)], linewidth=2, label='Without Pegging', color='red')
axes[1, 0].plot(time_range, spreads_with_peg[:len(time_range)], linewidth=2, label='With Pegged Liquidity', color='green')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Bid-Ask Spread ($)')
axes[1, 0].set_title('Scenario 4: Spread Impact Comparison')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Fill rate comparison
order_types = ['Limit\nOrders', 'Pegged\nOrders']
fill_rates = [
    (limit_orders_filled / limit_orders_submitted * 100) if limit_orders_submitted > 0 else 0,
    (peg_in_book / len(sim_peg.peg_orders) * 100) if len(sim_peg.peg_orders) > 0 else 0
]

colors = ['blue', 'green']
bars = axes[1, 1].bar(order_types, fill_rates, color=colors, alpha=0.7)
axes[1, 1].set_ylabel('Entry Rate (%)')
axes[1, 1].set_title('Order Type Comparison')
axes[1, 1].set_ylim([0, 100])
axes[1, 1].grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar, rate in zip(bars, fill_rates):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Scenario 1 (Limit Orders):")
print(f"  Average Spread: ${np.mean(sim1.spread_history):.4f}")
print(f"  Spread Std Dev: ${np.std(sim1.spread_history):.4f}")

print(f"\nScenario 2 (Pegged Orders):")
print(f"  Average Spread: ${np.mean(sim2.spread_history):.4f}")
print(f"  Spread Std Dev: ${np.std(sim2.spread_history):.4f}")

print(f"\nSpread Improvement (Scenario 2 vs 1):")
spread_change = (np.mean(sim2.spread_history) - np.mean(sim1.spread_history)) / np.mean(sim1.spread_history) * 100
print(f"  {spread_change:+.2f}%")
```

## 6. Challenge Round
Why do pegged orders sometimes execute at terrible prices during market gaps or halts?

- **Gap risk**: Stock halted for news announcement → resumes trading 20% lower → all bid pegs immediately adjust down 20% → execute at worse prices than pre-halt → no protection (unlike stop orders)
- **Peg lag**: Exchange peg calculation takes 10-50 milliseconds → during fast market, reference price changes before peg adjusts → pegs execute based on stale reference → slippage
- **Cascade effect**: If market drops, all sell-side pegs drop together → synchronized selling → feedback loop → similar to stop order cascades
- **Illiquidity concentration**: Pegs at same reference level (e.g., all at mid-quote) → all execute together → no staggered liquidity → liquidity provider overwhelmed
- **No protection**: Unlike limit orders with fixed price, pegs have NO minimum price → could execute at extreme values in flash crashes → worse than manual limit orders

## 7. Key References
- [Foucault et al (2007) - Order Flow, Transaction Clock, and Normality of Asset Returns](https://academic.oup.com/jf/article-abstract/62/5/2427)
- [SEC Rule 10b-5 - Employment of Manipulative Practices](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=&type=10-K&dateb=&owner=exclude&count=100)
- [Harris (2003) - Trading and Exchanges - Chapter on Order Types](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708)
- [Bender et al (2018) - Foundations of Algorithmic Trading](https://www.amazon.com/Foundations-Algorithmic-Trading-Analysis-Backtesting/dp/1491923997)

---
**Status:** Dynamic price-tracking execution | **Complements:** Limit Orders, Order Book Mechanics, Price Discovery, Liquidity Provision
