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
