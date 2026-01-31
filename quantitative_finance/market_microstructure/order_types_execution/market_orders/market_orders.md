# Market Orders

## 1. Concept Skeleton
**Definition:** Instruction to buy or sell security immediately at best available price in market  
**Purpose:** Guarantee execution certainty, prioritize speed over price, remove inventory immediately  
**Prerequisites:** Order book structure, bid-ask spread, liquidity concepts

## 2. Comparative Framing
| Order Type | Market Order | Limit Order | Stop Order |
|-----------|--------------|-------------|------------|
| **Execution** | Immediate (best effort) | Conditional on price | Triggered at threshold |
| **Price Certainty** | None (slippage risk) | Guaranteed if filled | Market price after trigger |
| **Fill Certainty** | Very high | Uncertain | Uncertain after trigger |
| **Liquidity Role** | Takes liquidity | Provides liquidity | Takes (after trigger) |

## 3. Examples + Counterexamples

**Simple Example:**  
Buy 100 shares market order: Executes against best ask ($50.05), potentially walks up order book if insufficient depth

**Failure Case:**  
Market order during low liquidity (3am): Wide spread ($50.00-$50.50) leads to poor execution, should use limit order

**Edge Case:**  
Flash crash: Market order executes at extreme price ($45) before recovery, demonstrates price risk without limits

## 4. Layer Breakdown
```
Market Order Execution Flow:
├─ Order Submission:
│   ├─ Trader Intent: Immediate execution paramount
│   ├─ Order Details: Direction (buy/sell), quantity, security ID
│   ├─ No Price Limit: Accept any available price
│   └─ Routing: Smart order router to best venue
├─ Matching Process:
│   ├─ Priority Rules: Price-time priority in order book
│   ├─ Best Quote: Match against best bid (sell) or ask (buy)
│   ├─ Walk the Book: Consume multiple price levels if needed
│   └─ Partial Fills: May execute in multiple tranches
├─ Execution Outcomes:
│   ├─ Fill Price: Volume-weighted average of matched prices
│   ├─ Slippage: Difference from expected vs actual price
│   ├─ Market Impact: Price moves away from initial quote
│   └─ Transaction Costs: Spread + impact + fees
└─ Post-Trade:
    ├─ Confirmation: Trade report with execution details
    ├─ Settlement: T+2 (equities), varies by asset class
    └─ TCA Analysis: Benchmark against VWAP, arrival price
```

**Interaction:** Submit → Route to venue → Match against book → Execute → Confirm → Settle

## 5. Mini-Project
Simulate market order execution with order book impact:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

class OrderBook:
    """Simplified limit order book for market order simulation"""
    def __init__(self):
        # Bids: [(price, size), ...] sorted descending
        self.bids = deque()
        # Asks: [(price, size), ...] sorted ascending  
        self.asks = deque()
        
    def add_bid(self, price, size):
        """Add limit buy order"""
        self.bids.append((price, size))
        self.bids = deque(sorted(self.bids, key=lambda x: -x[0]))
        
    def add_ask(self, price, size):
        """Add limit sell order"""
        self.asks.append((price, size))
        self.asks = deque(sorted(self.asks, key=lambda x: x[0]))
    
    def get_best_bid(self):
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self):
        return self.asks[0] if self.asks else None
    
    def get_spread(self):
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return ask[0] - bid[0]
        return None
    
    def execute_market_buy(self, quantity):
        """Execute market buy order, walking up the book"""
        remaining = quantity
        total_cost = 0
        fills = []
        
        while remaining > 0 and self.asks:
            ask_price, ask_size = self.asks[0]
            
            if ask_size <= remaining:
                # Consume entire level
                fills.append((ask_price, ask_size))
                total_cost += ask_price * ask_size
                remaining -= ask_size
                self.asks.popleft()
            else:
                # Partial fill at this level
                fills.append((ask_price, remaining))
                total_cost += ask_price * remaining
                self.asks[0] = (ask_price, ask_size - remaining)
                remaining = 0
        
        if remaining > 0:
            print(f"WARNING: Only filled {quantity - remaining}/{quantity} shares")
            
        avg_price = total_cost / (quantity - remaining) if quantity > remaining else 0
        return avg_price, fills
    
    def execute_market_sell(self, quantity):
        """Execute market sell order, walking down the book"""
        remaining = quantity
        total_proceeds = 0
        fills = []
        
        while remaining > 0 and self.bids:
            bid_price, bid_size = self.bids[0]
            
            if bid_size <= remaining:
                fills.append((bid_price, bid_size))
                total_proceeds += bid_price * bid_size
                remaining -= bid_size
                self.bids.popleft()
            else:
                fills.append((bid_price, remaining))
                total_proceeds += bid_price * remaining
                self.bids[0] = (bid_price, bid_size - remaining)
                remaining = 0
        
        avg_price = total_proceeds / (quantity - remaining) if quantity > remaining else 0
        return avg_price, fills
    
    def display_book(self, levels=5):
        """Display top levels of order book"""
        print(f"\n{'ASK PRICE':<12} {'SIZE':>8}")
        print("-" * 20)
        for price, size in list(self.asks)[:levels]:
            print(f"${price:<11.2f} {size:>8.0f}")
        print("=" * 20)
        print(f"Spread: ${self.get_spread():.2f}")
        print("=" * 20)
        for price, size in list(self.bids)[:levels]:
            print(f"${price:<11.2f} {size:>8.0f}")
        print("-" * 20)
        print(f"{'BID PRICE':<12} {'SIZE':>8}\n")

# Initialize order book with realistic depth
np.random.seed(42)
book = OrderBook()

# Create bid side (descending from $100)
mid_price = 100.00
for i in range(10):
    price = mid_price - 0.01 * (i + 1)
    size = np.random.randint(500, 2000)
    book.add_bid(price, size)

# Create ask side (ascending from $100.01)
for i in range(10):
    price = mid_price + 0.01 * (i + 1)
    size = np.random.randint(500, 2000)
    book.add_ask(price, size)

print("INITIAL ORDER BOOK")
book.display_book()

# Scenario 1: Small market buy (fits in best ask)
print("\n" + "="*50)
print("SCENARIO 1: Small Market Buy (500 shares)")
print("="*50)

initial_ask = book.get_best_ask()[0]
avg_price_1, fills_1 = book.execute_market_buy(500)

print(f"\nExpected price (best ask): ${initial_ask:.2f}")
print(f"Actual execution price: ${avg_price_1:.2f}")
print(f"Slippage: ${avg_price_1 - initial_ask:.4f}")
print(f"\nFill details:")
for price, size in fills_1:
    print(f"  {size:.0f} shares @ ${price:.2f}")

# Scenario 2: Large market buy (walks the book)
print("\n" + "="*50)
print("SCENARIO 2: Large Market Buy (5,000 shares)")
print("="*50)

book_snapshot = OrderBook()
for price, size in book.bids:
    book_snapshot.add_bid(price, size)
for price, size in book.asks:
    book_snapshot.add_ask(price, size)

initial_ask_2 = book_snapshot.get_best_ask()[0]
avg_price_2, fills_2 = book_snapshot.execute_market_buy(5000)

print(f"\nExpected price (best ask): ${initial_ask_2:.2f}")
print(f"Actual execution price: ${avg_price_2:.2f}")
print(f"Slippage: ${avg_price_2 - initial_ask_2:.4f} ({(avg_price_2/initial_ask_2 - 1)*10000:.1f} bps)")
print(f"\nFill details (walked {len(fills_2)} levels):")
for price, size in fills_2:
    print(f"  {size:.0f} shares @ ${price:.2f}")

# Visualization: Price impact vs order size
print("\n" + "="*50)
print("SCENARIO 3: Price Impact Analysis")
print("="*50)

order_sizes = np.arange(500, 10000, 500)
execution_prices = []
slippages = []

for size in order_sizes:
    # Rebuild book for each test
    test_book = OrderBook()
    for price, qty in book.bids:
        test_book.add_bid(price, qty)
    for price, qty in book.asks:
        test_book.add_ask(price, qty)
    
    initial_ask = test_book.get_best_ask()[0]
    avg_price, _ = test_book.execute_market_buy(size)
    
    execution_prices.append(avg_price)
    slippages.append((avg_price - initial_ask) * 10000)  # bps

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Execution price vs order size
axes[0, 0].plot(order_sizes, execution_prices, 'o-', linewidth=2, markersize=6)
axes[0, 0].axhline(book.get_best_ask()[0], color='r', linestyle='--', 
                   label=f'Best Ask: ${book.get_best_ask()[0]:.2f}')
axes[0, 0].set_title('Execution Price vs Order Size')
axes[0, 0].set_xlabel('Order Size (shares)')
axes[0, 0].set_ylabel('Average Fill Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Slippage in basis points
axes[0, 1].plot(order_sizes, slippages, 'o-', color='red', linewidth=2, markersize=6)
axes[0, 1].set_title('Market Impact (Slippage)')
axes[0, 1].set_xlabel('Order Size (shares)')
axes[0, 1].set_ylabel('Slippage (basis points)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Square-root law fit
from scipy.optimize import curve_fit

def sqrt_impact(size, a):
    return a * np.sqrt(size)

popt, _ = curve_fit(sqrt_impact, order_sizes, slippages)
fitted_slippage = sqrt_impact(order_sizes, *popt)

axes[1, 0].scatter(order_sizes, slippages, alpha=0.6, label='Observed')
axes[1, 0].plot(order_sizes, fitted_slippage, 'r-', linewidth=2, 
                label=f'Square-root fit: {popt[0]:.4f}√size')
axes[1, 0].set_title('Square-Root Price Impact Law')
axes[1, 0].set_xlabel('Order Size (shares)')
axes[1, 0].set_ylabel('Slippage (bps)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Cumulative cost comparison
limit_order_cost = book.get_best_ask()[0] * order_sizes  # Patient limit order
market_order_cost = np.array(execution_prices) * order_sizes

opportunity_cost = (market_order_cost - limit_order_cost) / 1000  # in thousands

axes[1, 1].plot(order_sizes, opportunity_cost, 'o-', linewidth=2, markersize=6)
axes[1, 1].set_title('Opportunity Cost: Market vs Limit Order')
axes[1, 1].set_xlabel('Order Size (shares)')
axes[1, 1].set_ylabel('Extra Cost ($1000s)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nSquare-root law coefficient: {popt[0]:.4f}")
print(f"Price impact for 10,000 shares: {sqrt_impact(10000, *popt):.2f} bps")
```

## 6. Challenge Round
When should you avoid market orders?
- Low liquidity securities: Wide spreads, high slippage risk, use limit orders
- High volatility periods: Prices moving rapidly, use limit with buffer
- Large institutional orders: Market impact prohibitive, use VWAP/TWAP algorithms
- Opening/closing auctions: Order imbalances, wait for continuous trading
- News events: Spreads widen, liquidity withdraws, delay execution
- After-hours trading: Thin order books, use limit orders only

## 7. Key References
- [Harris, Trading and Exchanges (Chapter 4: Orders and Order Properties)](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708)
- [SEC: Order Types and Trading](https://www.sec.gov/fast-answers/answersordertypeshtm.html)
- [Hasbrouck, Empirical Market Microstructure (Chapter 3)](https://global.oup.com/academic/product/empirical-market-microstructure-9780195301649)

---
**Status:** Fundamental execution mechanism | **Complements:** Limit Orders, Bid-Ask Spread, Market Impact
