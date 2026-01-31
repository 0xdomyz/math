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
