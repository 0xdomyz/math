# Limit Orders

## 1. Concept Skeleton
**Definition:** Order to buy/sell at specified price or better, providing price protection with execution uncertainty  
**Purpose:** Control execution price, provide liquidity to market, avoid slippage from market impact  
**Prerequisites:** Order book mechanics, bid-ask spread, price-time priority, liquidity provision

## 2. Comparative Framing
| Feature | Limit Order | Market Order | Stop-Limit Order |
|---------|-------------|--------------|------------------|
| **Price Control** | Exact maximum/minimum | None | After trigger |
| **Execution** | Uncertain (queue position) | Immediate | Conditional |
| **Fees** | Maker rebate (sometimes) | Taker fee | Taker fee (post-trigger) |
| **Risk** | Non-execution, opportunity cost | Slippage, price uncertainty | Gap risk, non-fill |

## 3. Examples + Counterexamples

**Simple Example:**  
Buy limit $50.00, market at $50.05/$50.06: Order joins queue at $50.00, waits for price to trade through

**Failure Case:**  
Limit buy $50.00, price never reaches (stays $50.01-$50.10): Misses rally, opportunity cost exceeds saved spread

**Edge Case:**  
Flash crash: Limit sell $49.50 executes during brief collapse, price rebounds to $50, limit provided downside protection

## 4. Layer Breakdown
```
Limit Order Lifecycle:
├─ Order Submission:
│   ├─ Price Specification: Maximum buy / minimum sell
│   ├─ Quantity: Total shares to execute
│   ├─ Time-in-Force: Day, GTC, IOC, FOK
│   └─ Display: Visible vs iceberg (hidden quantity)
├─ Order Book Placement:
│   ├─ Priority Assignment: Price-time (FIFO) or pro-rata
│   ├─ Queue Position: Behind existing orders at same price
│   ├─ Pre-trade Transparency: Displayed to market (Level 2)
│   └─ Liquidity Provision: Narrows spread if inside NBBO
├─ Matching Events:
│   ├─ Incoming Market Order: Matched if at best price
│   ├─ Aggressive Limit Order: May cross your resting limit
│   ├─ Partial Fills: Execute available quantity, remainder stays
│   └─ Queue Dynamics: Position improves as ahead orders fill/cancel
├─ Execution Outcomes:
│   ├─ Full Fill: Entire quantity matched
│   ├─ Partial Fill: Some executed, remainder cancelled/remains
│   ├─ No Fill: Price never reached, order expires
│   └─ Adverse Selection: Filled when price moves against you
└─ Economic Impact:
    ├─ Saved Spread: Execution inside initial spread
    ├─ Maker Rebate: Exchange fee structure reward
    ├─ Opportunity Cost: If non-execution and price moves away
    └─ Selection Bias: Fill probability inversely related to alpha
```

**Interaction:** Submit at limit → Join queue → Wait for counterparty → Match → Fill or expire

## 5. Mini-Project
Simulate limit order execution dynamics with adverse selection:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

class LimitOrderBook:
    """Full limit order book with price-time priority"""
    def __init__(self):
        # Max heap for bids (negative prices for max heap)
        self.bids = defaultdict(list)  # {price: [(timestamp, order_id, size), ...]}
        self.asks = defaultdict(list)  # {price: [(timestamp, order_id, size), ...]}
        self.order_id_counter = 0
        self.timestamp = 0
        self.trade_history = []
        
    def add_limit_order(self, side, price, size):
        """Add limit order to book"""
        self.order_id_counter += 1
        order_id = self.order_id_counter
        
        if side == 'buy':
            self.bids[price].append((self.timestamp, order_id, size))
        else:
            self.asks[price].append((self.timestamp, order_id, size))
        
        self.timestamp += 1
        return order_id
    
    def get_best_bid(self):
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def get_best_ask(self):
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def get_mid_price(self):
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return None
    
    def match_market_order(self, side, size):
        """Execute market order against book"""
        remaining = size
        fills = []
        
        if side == 'buy':
            # Match against asks
            while remaining > 0 and self.asks:
                best_ask = self.get_best_ask()
                if not best_ask:
                    break
                    
                queue = self.asks[best_ask]
                while remaining > 0 and queue:
                    ts, oid, order_size = queue[0]
                    
                    if order_size <= remaining:
                        fills.append((best_ask, order_size, oid))
                        remaining -= order_size
                        queue.pop(0)
                    else:
                        fills.append((best_ask, remaining, oid))
                        queue[0] = (ts, oid, order_size - remaining)
                        remaining = 0
                
                if not queue:
                    del self.asks[best_ask]
        else:
            # Match against bids
            while remaining > 0 and self.bids:
                best_bid = self.get_best_bid()
                if not best_bid:
                    break
                    
                queue = self.bids[best_bid]
                while remaining > 0 and queue:
                    ts, oid, order_size = queue[0]
                    
                    if order_size <= remaining:
                        fills.append((best_bid, order_size, oid))
                        remaining -= order_size
                        queue.pop(0)
                    else:
                        fills.append((best_bid, remaining, oid))
                        queue[0] = (ts, oid, order_size - remaining)
                        remaining = 0
                
                if not queue:
                    del self.bids[best_bid]
        
        for fill in fills:
            self.trade_history.append((self.timestamp, fill[0], fill[1]))
        
        self.timestamp += 1
        return fills
    
    def get_depth(self, side, levels=5):
        """Get order book depth"""
        if side == 'buy':
            prices = sorted(self.bids.keys(), reverse=True)[:levels]
            return [(p, sum(q[2] for q in self.bids[p])) for p in prices]
        else:
            prices = sorted(self.asks.keys())[:levels]
            return [(p, sum(q[2] for q in self.asks[p])) for p in prices]
    
    def display(self, levels=5):
        """Display top levels"""
        asks = self.get_depth('sell', levels)
        bids = self.get_depth('buy', levels)
        
        print(f"\n{'PRICE':<10} {'SIZE':>8}")
        print("-" * 18)
        for price, size in reversed(asks):
            print(f"${price:<9.2f} {size:>8.0f}")
        print("=" * 18)
        mid = self.get_mid_price()
        if mid:
            print(f"Mid: ${mid:.2f}")
        print("=" * 18)
        for price, size in bids:
            print(f"${price:<9.2f} {size:>8.0f}")
        print("-" * 18 + "\n")

# Simulation: Limit order adverse selection
np.random.seed(42)

def simulate_limit_order_execution(n_periods=1000, true_value_change=0.05):
    """
    Simulate limit order execution with adverse selection:
    - True value follows random walk
    - Informed traders arrive with probability
    - Limit orders face adverse selection risk
    """
    book = LimitOrderBook()
    
    # Initial true value
    true_value = 100.0
    
    # Initialize book around true value
    spread_half = 0.01
    for i in range(5):
        book.add_limit_order('buy', true_value - spread_half - i*0.01, 
                           np.random.randint(100, 300))
        book.add_limit_order('sell', true_value + spread_half + i*0.01, 
                           np.random.randint(100, 300))
    
    # Track limit order performance
    limit_order_submissions = []
    limit_order_outcomes = []
    
    for t in range(n_periods):
        # True value random walk
        if np.random.random() < 0.1:  # 10% chance of information event
            true_value += np.random.choice([-1, 1]) * true_value_change
        
        # Informed trader arrives (knows true value)
        if np.random.random() < 0.3:  # 30% informed trading
            mid = book.get_mid_price()
            if mid and abs(true_value - mid) > 0.02:
                # Informed trade: market order in direction of true value
                if true_value > mid:
                    book.match_market_order('buy', np.random.randint(50, 150))
                else:
                    book.match_market_order('sell', np.random.randint(50, 150))
        
        # Uninformed limit orders (liquidity providers)
        if np.random.random() < 0.5:
            mid = book.get_mid_price()
            if mid:
                # Submit limit orders inside spread
                bid_price = mid - 0.005
                ask_price = mid + 0.005
                
                order_id_bid = book.add_limit_order('buy', bid_price, 100)
                order_id_ask = book.add_limit_order('sell', ask_price, 100)
                
                limit_order_submissions.append({
                    't': t,
                    'side': 'buy',
                    'price': bid_price,
                    'order_id': order_id_bid,
                    'true_value': true_value,
                    'mid_at_submit': mid
                })
                
                limit_order_submissions.append({
                    't': t,
                    'side': 'sell',
                    'price': ask_price,
                    'order_id': order_id_ask,
                    'true_value': true_value,
                    'mid_at_submit': mid
                })
        
        # Noise trading
        if np.random.random() < 0.3:
            side = np.random.choice(['buy', 'sell'])
            book.match_market_order(side, np.random.randint(20, 100))
        
        # Check limit order fills (simplified: check if price has traded through)
        for submission in limit_order_submissions:
            if submission.get('filled'):
                continue
                
            if submission['side'] == 'buy':
                # Check if any trades below our limit price
                recent_trades = [p for ts, p, sz in book.trade_history[-10:]]
                if recent_trades and min(recent_trades) <= submission['price']:
                    submission['filled'] = True
                    submission['fill_time'] = t
                    submission['true_value_at_fill'] = true_value
            else:
                recent_trades = [p for ts, p, sz in book.trade_history[-10:]]
                if recent_trades and max(recent_trades) >= submission['price']:
                    submission['filled'] = True
                    submission['fill_time'] = t
                    submission['true_value_at_fill'] = true_value
    
    return limit_order_submissions, book

# Run simulation
print("Running limit order adverse selection simulation...")
submissions, final_book = simulate_limit_order_execution(n_periods=2000)

# Analyze adverse selection
df = pd.DataFrame(submissions)
df['filled'] = df['filled'].fillna(False)

filled_buys = df[(df['side'] == 'buy') & (df['filled'] == True)]
filled_sells = df[(df['side'] == 'sell') & (df['filled'] == True)]

# Calculate adverse price movement after fill
if len(filled_buys) > 0:
    filled_buys['adverse_move'] = filled_buys['true_value_at_fill'] - filled_buys['price']
    avg_adverse_buy = filled_buys['adverse_move'].mean()
else:
    avg_adverse_buy = 0

if len(filled_sells) > 0:
    filled_sells['adverse_move'] = filled_sells['price'] - filled_sells['true_value_at_fill']
    avg_adverse_sell = filled_sells['adverse_move'].mean()
else:
    avg_adverse_sell = 0

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Fill rates
fill_rate = df['filled'].mean()
axes[0, 0].bar(['Filled', 'Not Filled'], 
               [df['filled'].sum(), (~df['filled']).sum()],
               color=['green', 'red'], alpha=0.6)
axes[0, 0].set_title(f'Limit Order Fill Rate: {fill_rate*100:.1f}%')
axes[0, 0].set_ylabel('Number of Orders')

# Plot 2: Adverse selection distribution
if len(filled_buys) > 0:
    axes[0, 1].hist(filled_buys['adverse_move'], bins=30, alpha=0.6, 
                    label=f'Buy Orders (avg: {avg_adverse_buy:.4f})', color='blue')
if len(filled_sells) > 0:
    axes[0, 1].hist(filled_sells['adverse_move'], bins=30, alpha=0.6,
                    label=f'Sell Orders (avg: {avg_adverse_sell:.4f})', color='red')
axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=2)
axes[0, 1].set_title('Adverse Selection: Price Move After Fill')
axes[0, 1].set_xlabel('True Value - Limit Price ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# Plot 3: Time to fill distribution
filled_orders = df[df['filled'] == True].copy()
if len(filled_orders) > 0:
    filled_orders['time_to_fill'] = filled_orders['fill_time'] - filled_orders['t']
    axes[1, 0].hist(filled_orders['time_to_fill'], bins=50, alpha=0.7, color='green')
    axes[1, 0].set_title(f'Time to Fill (median: {filled_orders["time_to_fill"].median():.0f} periods)')
    axes[1, 0].set_xlabel('Periods Until Fill')
    axes[1, 0].set_ylabel('Frequency')

# Plot 4: PnL distribution (filled vs opportunity cost)
filled_pnl = []
missed_opportunity = []

for _, row in df.iterrows():
    if row['filled']:
        # Realized PnL (negative adverse selection)
        if row['side'] == 'buy':
            pnl = row['true_value_at_fill'] - row['price']
        else:
            pnl = row['price'] - row['true_value_at_fill']
        filled_pnl.append(pnl)
    else:
        # Opportunity cost (what we would have made)
        filled_pnl.append(0)  # Placeholder

axes[1, 1].hist(filled_pnl, bins=30, alpha=0.7, color='purple')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Breakeven')
axes[1, 1].set_title('Limit Order PnL Distribution')
axes[1, 1].set_xlabel('Profit/Loss per Share ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"\nAdverse Selection Results:")
print(f"  Buy orders filled: {len(filled_buys)}, avg adverse move: ${avg_adverse_buy:.4f}")
print(f"  Sell orders filled: {len(filled_sells)}, avg adverse move: ${avg_adverse_sell:.4f}")
print(f"  Overall fill rate: {fill_rate*100:.1f}%")
print(f"  Negative selection: Limit orders fill when prices move against them")
```

## 6. Challenge Round
What are the strategic considerations for limit orders?
- **Queue position**: Earlier submission = better priority, but price may never reach
- **Penny jumping**: Improve price by minimum tick to gain priority, but lower profit
- **Hidden orders**: Avoid signaling large size, but lose time priority at price level
- **Maker-taker economics**: Rebates incentivize passive orders, but execution uncertain
- **Adverse selection**: Fill probability highest when you're wrong about direction
- **Opportunity cost**: Saved spread vs missed alpha from non-execution

## 7. Key References
- [Foucault, Pagano, Röell: Market Liquidity (Chapter 3: Limit Order Markets)](https://global.oup.com/academic/product/market-liquidity-9780199936243)
- [Parlour & Seppi: Limit Order Markets Survey](https://www.annualreviews.org/doi/abs/10.1146/annurev.financial.8.082507.104702)
- [Glosten: Is the Electronic Open Limit Order Book Inevitable?](https://onlinelibrary.wiley.com/doi/abs/10.1046/j.1540-6261.1994.tb04788.x)

---
**Status:** Core market making mechanism | **Complements:** Market Orders, Bid-Ask Spread, Price-Time Priority
