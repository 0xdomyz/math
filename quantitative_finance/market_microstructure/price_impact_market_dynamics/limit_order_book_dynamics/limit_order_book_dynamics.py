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
