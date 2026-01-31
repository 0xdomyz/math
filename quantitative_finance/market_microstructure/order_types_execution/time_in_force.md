# Time-in-Force (TIF) Orders

## 1. Concept Skeleton
**Definition:** Rules governing how long an order remains active before cancellation (IOC, FOK, GTC, Day, GAT, etc.)  
**Purpose:** Control execution window; manage counterparty exposure; automate order management; reduce operational risk  
**Prerequisites:** Order types, market structure, execution algorithms, regulatory rules

## 2. Comparative Framing
| TIF Type | Execution | Cancellation | Use Case | Latency Sensitive |
|----------|-----------|--------------|----------|-------------------|
| **IOC** | Fill immediately, cancel remainder | Instant (~100ms) | Large orders needing partial fills | Yes |
| **FOK** | All or nothing | Instant if unfilled | Block trades, institutional liquidity | Yes |
| **GTC** | Until filled or canceled | Manual or 30/60/90 days | Passive limit orders | No |
| **Day** | Until market close | Auto-cancel at 4pm | Most retail orders | No |
| **GAT** | Until specified time | Auto-cancel at time | Risk management, expiration hedges | No |
| **MOC** | At market close | Executes or cancels | Portfolio rebalancing, index reconstitution | No |

## 3. Examples + Counterexamples

**IOC Success:**  
Hedge fund wants to unwind $50M position in illiquid security → places IOC market buy 100K shares → captures 50K at best ask, 30K at second level, 20K at third level → remainder auto-cancels → position reduced, no waiting

**FOK Failure:**  
Pension fund needs 1M shares of thinly traded stock, doesn't want partial fill → places FOK → only 500K shares available system-wide → entire order rejected → position unchanged → must place smaller IOC instead

**GTC Trap:**  
Trader places GTC sell limit at $50 for 1000 shares → forgets about order → stock reaches $50 six months later → executes at wrong tax year → creates unwanted tax loss harvest → tax inefficient

**Day Order Quirk:**  
After-hours news: Stock will surge tomorrow → Trader places day limit buy at $100 overnight → order doesn't execute after-hours → at market open, stock gaps to $110 → order canceled at 4pm unused → missed opportunity

## 4. Layer Breakdown
```
Time-in-Force Framework:
├─ IOC (Immediate-or-Cancel):
│   ├─ Mechanics:
│   │   - Order fills immediately at available prices
│   │   - Any unfilled portion canceled automatically (~1-100ms)
│   │   - Requires fast exchange processing
│   │   - No rest period: Order not held in book
│   ├─ Execution Algorithm:
│   │   1. Submit IOC order
│   │   2. Exchange matches against best ask/bid (immediately)
│   │   3. Fills at multiple levels if needed
│   │   4. Unexecuted quantity deleted (no cancellation required)
│   │   5. Confirmation sent to trader
│   ├─ Examples:
│   │   - Buy 50K shares IOC on illiquid stock
│   │   - Depth: 10K@$99.95, 15K@$99.90, 20K@$99.85, 5K@$99.80
│   │   - Execution: 10K@$99.95, 15K@$99.90, 20K@$99.85 (45K filled)
│   │   - Remainder: 5K canceled (NOT 5K@$99.80 attempted)
│   ├─ Advantages:
│   │   - Prevents "leakage": Counterparties see only immediate fills
│   │   - No information revelation: Unfilled quantity not visible
│   │   - Fast execution: Microseconds to seconds
│   │   - Certainty: Know outcome immediately
│   ├─ Disadvantages:
│   │   - Partial fills common: Rarely execute complete order
│   │   - Slippage risk: Execute at multiple worse levels
│   │   - Execution uncertainty: Doesn't guarantee % fill
│   │   - Need to resubmit: Multiple IOC orders for full execution
│   ├─ When to Use:
│   │   - Large orders needing liquidity profile
│   │   - Illiquid securities (accepting partial fill)
│   │   - Algorithmic execution (multiple IOCs, smaller sizes)
│   │   - Predatory algorithms (small IOCs to detect MM behavior)
│   ├─ Market Microstructure:
│   │   - IOC = "take what's available, don't leave footprint"
│   │   - Information asymmetry: Liquidity providers don't know order size
│   │   - Limits MM information about demand
│   │   - Benefits passive traders (limit orders)
│   │   - Hurts MM (incomplete view of book)
│   └─ Empirical:
│       - Large-cap equity: 95%+ fill rate on IOC
│       - Small-cap equity: 40-60% fill rate on IOC
│       - Illiquid bonds: 10-30% fill rate typical
│
├─ FOK (Fill-or-Kill):
│   ├─ Mechanics:
│   │   - MUST execute entire order or kill entire order
│   │   - No partial fills accepted
│   │   - Rejected instantly if insufficient liquidity
│   │   - Binary outcome: All-or-nothing
│   ├─ Execution Algorithm:
│   │   1. Submit FOK order (e.g., 100K shares)
│   │   2. Check: Is 100K+ liquidity available?
│   │   3a. YES: Fill entire 100K, send confirmation
│   │   3b. NO: Kill entire order, send rejection
│   │   4. No partial fills or resting period
│   ├─ Examples:
│   │   - Block trade: "I want 5M shares or nothing"
│   │   - Regulatory hedge: Need 500K shares to hedge position by EOD
│   │   - Merger arb: "Buy 1M shares if available, else decline"
│   ├─ Advantages:
│   │   - Certainty: Know immediately if position filled
│   │   - No partial fills: Cleaner accounting
│   │   - Good for block trades: Negotiate complete size
│   │   - Portfolio rebalancing: Either fully rebalanced or not
│   ├─ Disadvantages:
│   │   - Rejection common: Even liquid markets have gaps
│   │   - All-or-nothing risky: Miss partial execution
│   │   - Timing risk: Market conditions change between quote & order
│   │   - Slower fill: Requires coordination vs. IOC
│   ├─ When to Use:
│   │   - Block trades: Broker-to-broker 1M+ share deals
│   │   - Risk management: Need complete hedge or no hedge
│   │   - Regulatory compliance: Tax loss harvesting, position limits
│   │   - Options: Assignment/exercise requires exact shares
│   ├─ Comparison to IOC:
│   │   - IOC: Take what you can get
│   │   - FOK: Get everything or nothing
│   │   - Trade-off: Execution certainty vs. flexibility
│   └─ Market Maker Perspective:
│       - FOK orders are good (full commitment, no cancellation risk)
│       - IOC orders are bad (MM left holding partial inventory)
│
├─ GTC (Good-Till-Canceled):
│   ├─ Mechanics:
│   │   - Order remains active until: filled, manually canceled, or expires
│   │   - No time limit (some brokers: 30/60/90 day auto-cancel)
│   │   - Rests on order book continuously
│   │   - Multiple days/weeks of exposure
│   ├─ Activation:
│   │   - Submit GTC limit order (e.g., sell $50 limit, 1000 shares)
│   │   - Order placed on book (usually bottom if worse than NBBO)
│   │   - Remains 1 day, 7 days, 30 days, until canceled
│   │   - If executed, great. If not executed, remains until you cancel.
│   ├─ Examples:
│   │   - Place GTC sell limit $50 (stock at $48) → wait weeks for $50 touch
│   │   - Place GTC buy limit $45 (stock at $50) → might never execute
│   │   - Forgotten GTC: Trader places, leaves firm, order still active
│   ├─ Advantages:
│   │   - Passive execution: Wait for price (no active management)
│   │   - Limit price protection: Guaranteed price if filled
│   │   - Set and forget: No daily resubmission needed
│   │   - Good for wide targets: Willing to wait months
│   ├─ Disadvantages:
│   │   - Execution uncertainty: May never execute
│   │   - Forgotten orders: Can execute months later unexpectedly
│   │   - Corporate actions: Stock splits/dividends affect order
│   │   - Price gaps: Order might execute after big gap at bad price
│   ├─ Forgotten GTC Risks:
│   │   - Trader A places GTC sell $50 in March
│   │   - Trader A leaves firm in April
│   │   - Stock hits $50 in June → EXECUTES → firm liable for trade
│   │   - Tax implications: June execution vs. March intention
│   │   - Regulatory: Unsupervised order execution without oversight
│   ├─ Typical Broker Policies:
│   │   - Interactive Brokers: 30-day auto-cancel (unless renewed)
│   │   - E-TRADE: 60-day auto-cancel (monthly renewal available)
│   │   - Fidelity: 365-day active
│   │   - Institutional: No auto-cancel (trader responsibility)
│   ├─ When to Use:
│   │   - Patient limit orders: Willing to wait
│   │   - Tax-loss harvesting: Specific prices for tax purposes
│   │   - Long-term rebalancing: Multi-month execution
│   │   - Retail traders: Set buy/sell targets and wait
│   ├─ Market Impact:
│   │   - GTC orders = standing demand/supply on book
│   │   - Price discovery: GTC represents patient capital
│   │   - Spread impact: Too many GTC orders can widen spreads
│   │   - Book depth: GTC orders provide depth (good for MM)
│   └─ Empirical:
│       - Retail: ~30% of limit orders are GTC
│       - Institutional: <5% of limit orders are GTC (prefer daily refresh)
│       - Execution rate: 15-25% for GTC orders (many never execute)
│
├─ Day Orders:
│   ├─ Mechanics:
│   │   - Order active from order submission to 4:00 PM ET (market close)
│   │   - Auto-cancel at 4pm if not executed
│   │   - Most common order type for retail traders
│   │   - Prevents overnight holdings
│   ├─ Timing:
│   │   - 9:30 AM: Market opens, order becomes active
│   │   - 3:59 PM: Order still active (can execute at close)
│   │   - 4:00 PM: Order auto-canceled if not filled
│   │   - After-hours: Order not active (requires AON or separate order)
│   ├─ Examples:
│   │   - Day limit buy $100: Active 9:30-16:00, auto-cancel if unfilled
│   │   - Day market sell: Execute immediately or during day
│   │   - Day orders end at 4pm: Must resubmit next day if want continued
│   ├─ Advantages:
│   │   - Clear expiration: No forgotten orders
│   │   - Risk containment: Overnight gap risk avoided
│   │   - Simple to manage: Resubmit same order each day if needed
│   │   - Default: Most retail platforms default to day orders
│   ├─ Disadvantages:
│   │   - Resubmission required: Can't set and forget
│   │   - Miss opportunities: After-hours doesn't execute
│   │   - Earnings gaps: No protection if stock gaps overnight
│   │   - Time zone: 4pm ET is not same as day-trader time zone
│   ├─ Day Order Misconception:
│   │   - "Day order" ≠ "Trading hours only"
│   │   - Day orders DO execute at market close (last seconds of 4pm)
│   │   - Pre-market (8am-9:30am) requires separate setup
│   │   - After-hours (4pm-8pm) requires separate order
│   ├─ When to Use:
│   │   - Intraday trading: Buy and sell same day
│   │   - Default retail trading: Most common choice
│   │   - Short-term positions: Don't want overnight risk
│   │   - Earnings trades: Execute during session, no overnight
│   └─ Market Structure:
│       - Day orders: Entire day liquidity concentrated
│       - EOD crush: Volume spike 3:50pm-4:00pm (last-minute orders)
│       - Liquidity drying up: 4pm order may face worse price than 3pm
│
├─ GAT (Good-After-Time) / GTD (Good-Till-Date):
│   ├─ Mechanics:
│   │   - GAT: Order becomes active at specified time (e.g., 2pm)
│   │   - GTD: Order expires at specified date (e.g., Friday only)
│   │   - Combination: "Active Monday-Friday, 9:30am-4pm only"
│   │   - Advanced order type (not available on all platforms)
│   ├─ Examples:
│   │   - GAT for earnings: "Execute only after 4:30pm (earnings announcement)"
│   │   - GTD for expiration: "Good only until Friday (options expiration)"
│   │   - GAT+GTD: "Active only during 9:30-3:30pm window on Tuesday"
│   ├─ Use Cases:
│   │   - Risk management: "Don't execute until after volatility event"
│   │   - Scheduled execution: "Execute at specific time only"
│   │   - Regulatory: "Execute during best execution hours only"
│   │   - Options: "Execute Monday-Friday before expiration"
│   ├─ Advantages:
│   │   - Fine-grained control: Specify exact execution window
│   │   - Risk management: Avoid volatile periods
│   │   - Behavioral: Forces discipline (can't trade impulsively)
│   └─ Disadvantages:
│       - Complexity: Fewer platforms offer GAT
│       - Confusion: Multiple parameters (time, date, type)
│       - Admin burden: Need to set/verify for each order
│
├─ MOC (Market-on-Close):
│   ├─ Mechanics:
│   │   - Order executes at official market close (4:00pm ET)
│   │   - Uses closing auction mechanism
│   │   - Better-defined price vs. limit orders at close
│   │   - Bunches volume at 4pm
│   ├─ Execution:
│   │   - 3:45pm-3:50pm: MOC orders accumulate in exchange
│   │   - 3:50pm-4:00pm: Closing imbalance published
│   │   - 4:00pm: Auction mechanism determines clearing price
│   │   - Execution at single price (official close)
│   ├─ Examples:
│   │   - Index fund rebalancing: All funds place MOC orders
│   │   - ETF creation: Market makers place MOC orders to hedge
│   │   - End-of-month flows: Many investors use MOC for performance
│   ├─ Advantages:
│   │   - Single price execution: Not walking across spreads
│   │   - Large orders: Execute full size at closing auction
│   │   - Known timing: Execute at exact close (no uncertainty)
│   │   - Cost-effective: No intraday slippage
│   ├─ Disadvantages:
│   │   - Bunched volume: High competition at close
│   │   - Execution risk: Imbalance can cause price move
│   │   - Regulatory: MOC orders must be bona-fide (no gaming)
│   │   - Timing: If you miss MOC deadline, order not accepted
│   ├─ Market Microstructure:
│   │   - MOC volume: 10-20% of daily volume on major indices
│   │   - Index reconstitution: Causes massive MOC imbalance
│   │   - Day-trader trap: If short index, gets squeezed on MOC
│   │   - Liquidity concentration: MM can charge premium for MOC
│   └─ Empirical:
│       - MOC execution: ±0.5% of day's volume typically moves close
│       - Index days: ±2-3% moves possible if large rebalancing
│       - Volatility spike: MOC volume correlated with VIX moves
│
├─ Regulatory Considerations:
│   ├─ SEC Rules:
│   │   - Regulation SHO: IOC/FOK can short without locate
│   │   - Best Execution: Broker must achieve best TIF price
│   │   - Cancelled Orders: Broker must track canceled quantity
│   ├─ FINRA Rules:
│   │   - Order Blotter: Audit trail of all orders (including TIF)
│   │   - Supervisory Review: Brokers must review order patterns
│   │   - Fraudulent Orders: No orders designed to manipulate close
│   ├─ Exchange Rules:
│   │   - NASDAQ/NYSE: Different TIF rules per order type
│   │   - CME/CBOT: Futures have different TIF (e.g., GTC cancels Monday)
│   │   - International: Hong Kong stock exchange has different rules
│   └─ Compliance:
│       - Blotter management: Track all TIF expirations
│       - Audit trail: Regulatory requires 6 years record
│       - Reporting: Monthly order statistics to SEC
│
├─ Strategic Use of TIF:
│   ├─ Predatory Trading:
│   │   - Use IOC to detect order book depth
│   │   - Small IOC → see partial fill → infer depth
│   │   - Trade ahead using depth information
│   │   - Result: Illegal market manipulation
│   ├─ Smart Execution:
│   │   - IOC for liquidity seeking (participate)
│   │   - GTC for patient waiting (provide)
│   │   - Mix TIF types: Some IOC aggressive, some GTC passive
│   │   - Algorithmic: Combine multiple TIF in algorithm
│   ├─ Information Leakage:
│   │   - GTC orders visible on book: Others see your demand
│   │   - IOC orders hidden: Order size not revealed
│   │   - FOK orders: Signal intention (need exact size)
│   │   - Information asymmetry: Using TIF to gain edge
│   ├─ Cost Optimization:
│   │   - IOC + FOK: Higher cost (market impact), faster
│   │   - GTC + Day: Lower cost (wait for fill), slower
│   │   - Urgent orders: Use IOC/FOK (accept higher cost)
│   │   - Patient orders: Use GTC/Day (accept lower certainty)
│   └─ Operational Risks:
│       - GTC expiration: Forgotten orders execute wrong time
│       - FOK rejections: Too many FOK rejections = no execution
│       - MOC volume: Timing risk if MOC deadline missed
│       - Regulatory: Ensure compliance with order tracking
│
└─ Summary Table:
    ┌──────┬──────────────┬─────────────────┬─────────────────┐
    │ TIF  │ Execution    │ Liquidity Taker │ Best For        │
    ├──────┼──────────────┼─────────────────┼─────────────────┤
    │ IOC  │ Partial OK   │ YES (market)    │ Large size      │
    │ FOK  │ All-or-None  │ YES (market)    │ Block trades    │
    │ GTC  │ Until cancel │ NO (limit)      │ Patient traders │
    │ Day  │ Until 4pm    │ Both possible   │ Default retail  │
    │ GAT  │ At time      │ Both possible   │ Event trades    │
    │ MOC  │ At close     │ YES (auction)   │ Index funds     │
    └──────┴──────────────┴─────────────────┴─────────────────┘
```

**Interaction:** TIF selected → order submitted → exchange enforces TIF rules → execution/cancellation at specified time → confirmation sent to trader

## 5. Mini-Project
Simulate TIF order execution with different order types:
```python
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

np.random.seed(42)

# TIF Order Execution Simulator
class OrderExecutionSimulator:
    def __init__(self, market_open_time=930, market_close_time=1600):
        self.market_open = market_open_time
        self.market_close = market_close_time
        self.current_time = market_open_time
        self.orders = []
        self.executions = []
        self.price_history = []
        self.current_price = 100.0
        
    def place_order(self, order_id, tif_type, side, quantity, limit_price=None, 
                   time_window=None, expiration_date=None):
        """Place an order with specified TIF"""
        order = {
            'order_id': order_id,
            'tif_type': tif_type,  # IOC, FOK, GTC, Day, GAT, MOC
            'side': side,  # buy, sell
            'quantity': quantity,
            'remaining': quantity,
            'limit_price': limit_price,
            'time_window': time_window,  # For GAT: (start_time, end_time)
            'expiration_date': expiration_date,  # For GTD
            'status': 'active',
            'created_time': self.current_time,
            'executed_quantity': 0,
            'execution_prices': []
        }
        self.orders.append(order)
        return order
    
    def should_execute_order(self, order, current_time, current_price):
        """Check if order should execute based on TIF rules"""
        
        if order['status'] != 'active':
            return False
        
        # Check TIF expiration
        if order['tif_type'] == 'Day':
            if current_time >= self.market_close:
                order['status'] = 'canceled'
                return False
        
        elif order['tif_type'] == 'IOC':
            if current_time - order['created_time'] > 1:  # 1 minute window
                order['status'] = 'canceled'
                return False
        
        elif order['tif_type'] == 'FOK':
            if current_time - order['created_time'] > 1:  # 1 minute window
                order['status'] = 'rejected'
                return False
        
        elif order['tif_type'] == 'GTC':
            # No expiration (unless manual cancel)
            pass
        
        elif order['tif_type'] == 'GAT':
            # Active only in time window
            if order['time_window']:
                start_time, end_time = order['time_window']
                if current_time < start_time or current_time >= end_time:
                    return False
        
        elif order['tif_type'] == 'MOC':
            # Execute only at market close
            if current_time != self.market_close:
                return False
        
        # Check limit price (for limit orders)
        if order['limit_price']:
            if order['side'] == 'buy' and current_price > order['limit_price']:
                return False
            elif order['side'] == 'sell' and current_price < order['limit_price']:
                return False
        
        return True
    
    def execute_order(self, order, current_price, available_quantity):
        """Execute order based on TIF type"""
        
        if order['tif_type'] == 'IOC':
            # Fill what available, cancel rest
            fill_quantity = min(order['remaining'], available_quantity)
            order['executed_quantity'] += fill_quantity
            order['remaining'] -= fill_quantity
            order['execution_prices'].append(current_price)
            
            if order['remaining'] > 0:
                order['status'] = 'partially_filled_canceled'
            else:
                order['status'] = 'filled'
        
        elif order['tif_type'] == 'FOK':
            # All or nothing
            if available_quantity >= order['remaining']:
                fill_quantity = order['remaining']
                order['executed_quantity'] += fill_quantity
                order['remaining'] = 0
                order['execution_prices'].append(current_price)
                order['status'] = 'filled'
            else:
                order['status'] = 'rejected'
        
        elif order['tif_type'] in ['GTC', 'Day', 'GAT']:
            # Limit order: execute if available
            fill_quantity = min(order['remaining'], available_quantity)
            if fill_quantity > 0:
                order['executed_quantity'] += fill_quantity
                order['remaining'] -= fill_quantity
                order['execution_prices'].append(current_price)
                
                if order['remaining'] == 0:
                    order['status'] = 'filled'
                else:
                    order['status'] = 'partially_filled'
        
        elif order['tif_type'] == 'MOC':
            # Execute at close (auction)
            fill_quantity = min(order['remaining'], available_quantity)
            order['executed_quantity'] += fill_quantity
            order['remaining'] -= fill_quantity
            order['execution_prices'].append(current_price)
            order['status'] = 'filled'
        
        return order['executed_quantity']
    
    def simulate_trading_day(self, n_minutes=390):
        """Simulate one trading day"""
        
        # Generate price path
        for i in range(n_minutes):
            # Random price movement
            ret = np.random.normal(0, 0.001)
            self.current_price *= (1 + ret)
            self.price_history.append(self.current_price)
            
            # Random liquidity (available for immediate purchase)
            available_quantity = np.random.randint(500, 5000)
            
            # Process orders
            for order in self.orders:
                if self.should_execute_order(order, self.current_time, self.current_price):
                    self.execute_order(order, self.current_price, available_quantity)
                    
                    if order['executed_quantity'] > 0:
                        self.executions.append({
                            'order_id': order['order_id'],
                            'time': self.current_time,
                            'quantity': order['executed_quantity'],
                            'price': self.current_price,
                            'tif': order['tif_type']
                        })
            
            self.current_time += 1
            
            # Ensure we cap at market close
            if self.current_time >= self.market_close:
                break

# Scenario 1: Various TIF orders, normal market
print("Scenario 1: Normal Market with Different TIF Orders")
print("=" * 80)

sim1 = OrderExecutionSimulator()

# Place various orders
sim1.place_order('IOC-1', 'IOC', 'buy', 1000, limit_price=None)  # Market buy IOC
sim1.place_order('FOK-1', 'FOK', 'buy', 2000, limit_price=None)  # Block trade FOK
sim1.place_order('GTC-1', 'GTC', 'buy', 500, limit_price=98.0)   # Patient limit
sim1.place_order('DAY-1', 'Day', 'buy', 750, limit_price=99.0)   # Day limit
sim1.place_order('MOC-1', 'MOC', 'sell', 1500, limit_price=None) # Sell at close

sim1.simulate_trading_day()

print(f"Initial Price: $100.00")
print(f"Final Price: ${sim1.current_price:.2f}")
print(f"\nOrder Execution Results:")
print(f"{'Order ID':<15} {'TIF':<10} {'Status':<20} {'Executed':<15} {'Remaining':<15}")
print("-" * 75)

for order in sim1.orders:
    print(f"{order['order_id']:<15} {order['tif_type']:<10} {order['status']:<20} "
          f"{order['executed_quantity']:<15} {order['remaining']:<15}")

# Scenario 2: GTC orders with long-term exposure
print(f"\n\nScenario 2: GTC Orders with Long-Term Exposure")
print("=" * 80)

sim2 = OrderExecutionSimulator()

# Place GTC orders at various levels
for i, price in enumerate([98.0, 99.0, 99.5, 100.5, 101.0]):
    sim2.place_order(f'GTC-{i}', 'GTC', 'buy', 500, limit_price=price)

# Extended simulation (multiple days)
days = 0
for _ in range(2):  # Simulate 2 days
    days += 1
    sim2.market_open = 930 + days * 1000
    sim2.market_close = 1600 + days * 1000
    sim2.current_time = sim2.market_open
    
    for i in range(390):
        ret = np.random.normal(0, 0.0015)
        sim2.current_price *= (1 + ret)
        sim2.price_history.append(sim2.current_price)
        
        available_quantity = np.random.randint(500, 5000)
        
        for order in sim2.orders:
            if order['status'] == 'active' and order['limit_price']:
                if order['side'] == 'buy' and sim2.current_price <= order['limit_price']:
                    sim2.execute_order(order, sim2.current_price, available_quantity)
        
        sim2.current_time += 1

print(f"Simulated {days} trading days")
print(f"GTC Orders that Filled: {sum(1 for o in sim2.orders if o['executed_quantity'] > 0)}")
print(f"\nGTC Order Details:")
print(f"{'Order ID':<15} {'Limit Price':<15} {'Status':<20} {'Filled':<10}")
print("-" * 60)

for order in sim2.orders:
    print(f"{order['order_id']:<15} ${order['limit_price']:<14.2f} {order['status']:<20} "
          f"{order['executed_quantity']:<10}")

# Scenario 3: FOK orders with liquidity constraints
print(f"\n\nScenario 3: FOK Orders with Liquidity Constraints")
print("=" * 80)

sim3 = OrderExecutionSimulator()

# Place various FOK orders
fok_sizes = [500, 1000, 2000, 5000, 10000]
fok_results = []

for i, size in enumerate(fok_sizes):
    order = sim3.place_order(f'FOK-{size}', 'FOK', 'buy', size)
    
    # Simulate: Is this size available?
    # Assume available liquidity decreases with order size
    available = 8000 - size * 0.5
    
    if size <= available:
        order['status'] = 'filled'
        order['executed_quantity'] = size
        order['remaining'] = 0
        result = 'FILLED'
    else:
        order['status'] = 'rejected'
        result = 'REJECTED'
    
    fok_results.append({'size': size, 'result': result})

print(f"FOK Order Results (varying sizes):")
print(f"{'Size':<15} {'Available':<15} {'Result':<15}")
print("-" * 45)

for i, size in enumerate(fok_sizes):
    available = 8000 - size * 0.5
    print(f"{size:<15} {available:<15.0f} {fok_results[i]['result']:<15}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price evolution with order execution markers
times = list(range(len(sim1.price_history)))
prices = sim1.price_history

axes[0, 0].plot(times, prices, linewidth=2, label='Price', color='blue')
axes[0, 0].set_xlabel('Time (minutes)')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Scenario 1: Price Path with TIF Orders')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()

# Plot 2: GTC order execution over time
if sim2.price_history:
    times2 = list(range(len(sim2.price_history)))
    prices2 = sim2.price_history
    
    axes[0, 1].plot(times2, prices2, linewidth=2, label='Price', color='green')
    
    # Mark GTC execution levels
    for order in sim2.orders:
        if order['limit_price']:
            axes[0, 1].axhline(y=order['limit_price'], linestyle='--', alpha=0.5)
    
    axes[0, 1].set_xlabel('Time (minutes)')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Scenario 2: GTC Orders at Multiple Levels')
    axes[0, 1].grid(alpha=0.3)

# Plot 3: TIF execution success rates
tif_types = []
success_rates = []

if sim1.orders:
    for tif_type in ['IOC', 'FOK', 'GTC', 'Day', 'MOC']:
        orders_of_type = [o for o in sim1.orders if o['tif_type'] == tif_type]
        if orders_of_type:
            filled = sum(1 for o in orders_of_type if o['executed_quantity'] > 0)
            success_rate = filled / len(orders_of_type) * 100
            tif_types.append(tif_type)
            success_rates.append(success_rate)

if tif_types:
    axes[1, 0].bar(tif_types, success_rates, color=['red', 'orange', 'green', 'blue', 'purple'])
    axes[1, 0].set_ylabel('Fill Rate (%)')
    axes[1, 0].set_title('Scenario 1: TIF Order Fill Rates')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: FOK liquidity constraint
fok_sizes_plot = [f['size'] for f in fok_results]
fok_results_numeric = [1 if f['result'] == 'FILLED' else 0 for f in fok_results]

colors_fok = ['green' if r == 1 else 'red' for r in fok_results_numeric]
axes[1, 1].bar(fok_sizes_plot, fok_results_numeric, color=colors_fok, alpha=0.7)
axes[1, 1].set_xlabel('FOK Order Size (shares)')
axes[1, 1].set_ylabel('Execution (1=Filled, 0=Rejected)')
axes[1, 1].set_title('Scenario 3: FOK Execution vs. Order Size')
axes[1, 1].set_ylim([0, 1.2])
axes[1, 1].grid(alpha=0.3, axis='y')

# Add labels
for i, (size, result) in enumerate(zip(fok_sizes_plot, fok_results)):
    y_pos = 0.5 if result['result'] == 'FILLED' else 0.3
    axes[1, 1].text(size, y_pos, result['result'], ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary Statistics
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"\nScenario 1 (Mixed TIF):")
total_orders = len(sim1.orders)
filled_orders = sum(1 for o in sim1.orders if o['executed_quantity'] > 0)
print(f"  Total Orders: {total_orders}")
print(f"  Filled Orders: {filled_orders}")
print(f"  Fill Rate: {filled_orders/total_orders*100:.1f}%")
print(f"  Total Executed: {sum(o['executed_quantity'] for o in sim1.orders):.0f} shares")

print(f"\nScenario 2 (GTC Long-term):")
gtc_orders = [o for o in sim2.orders if o['tif_type'] == 'GTC']
gtc_filled = sum(1 for o in gtc_orders if o['executed_quantity'] > 0)
print(f"  Total GTC Orders: {len(gtc_orders)}")
print(f"  Filled: {gtc_filled}")
print(f"  Fill Rate: {gtc_filled/len(gtc_orders)*100:.1f}%" if gtc_orders else "  N/A")

print(f"\nScenario 3 (FOK Rejection):")
fok_filled = sum(1 for f in fok_results if f['result'] == 'FILLED')
print(f"  Total FOK Orders: {len(fok_results)}")
print(f"  Filled: {fok_filled}")
print(f"  Fill Rate: {fok_filled/len(fok_results)*100:.1f}%")
print(f"  Largest Rejected: {next((f['size'] for f in fok_results if f['result'] == 'REJECTED'), 'None')}")
```

## 6. Challenge Round
Why do GTC (Good-Till-Canceled) orders sometimes execute at unexpected times and prices?

- **Forgotten orders**: Trader places GTC six months ago at $50 limit, forgets about it. Stock hits $50 three months later, executes automatically → Tax consequences (executed in wrong tax year), Portfolio mismatch (position no longer desired)
- **Corporate actions**: Stock splits 2:1 → GTC order quantity adjusted automatically by exchange → execution now 2x size intended → forced to deal with doubled position
- **Gap risk**: GTC sell limit at $50 → overnight bankruptcy announcement → stock opens $5 → GTC executes at open → losses worse than expected
- **Information changes**: GTC buy limit at $50 placed when fundamentals strong. Months later fundamentals deteriorate → stock drops to $50 → GTC executes at wrong time (value lower than when order placed)
- **Regulatory**: FINRA requires order management systems to track all GTC expiration rules, auto-cancel after certain periods (30-90 days typical) → order effectiveness varies by broker

## 7. Key References
- [Harris (2003) - Trading and Exchanges - Chapter on Order Types](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708)
- [SEC Regulation NMS - Order Handling Rules](https://www.sec.gov/rules/final/34-51808.pdf)
- [FINRA Rule 5210 - Customer Account Information](https://www.finra.org/rules-guidance/rulebooks/finra-rules/5210)
- [CME Globex Order Types Guide](https://www.cmegroup.com/confluence/display/epicsandwich/Order+Types)

---
**Status:** Time-based execution rules | **Complements:** Market Orders, Limit Orders, Order Book Depth, Execution Algorithms
