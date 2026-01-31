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
