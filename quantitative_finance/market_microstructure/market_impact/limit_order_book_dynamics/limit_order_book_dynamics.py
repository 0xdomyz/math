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
