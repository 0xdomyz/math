import numpy as np
import matplotlib.pyplot as plt
from collections import deque

np.random.seed(42)

class HybridMarket:
    def __init__(self, initial_price=100.0):
        self.price = initial_price
        self.electronic_bid = initial_price - 0.005
        self.electronic_ask = initial_price + 0.005
        self.electronic_book = {'bid': {}, 'ask': {}}
        self.mm_bid = initial_price - 0.01
        self.mm_ask = initial_price + 0.01
        self.mm_inventory = 0
        self.trade_log = []
        self.routing_log = []
        self.spread_history = []
        self.price_history = []
        
    def get_nbbo(self):
        """Get National Best Bid Offer (best across both)"""
        best_bid = max(self.electronic_bid, self.mm_bid)
        best_ask = min(self.electronic_ask, self.mm_ask)
        return best_bid, best_ask
    
    def route_order(self, side, quantity, order_price=None):
        """Route order to best execution venue"""
        nbbo_bid, nbbo_ask = self.get_nbbo()
        
        if side == 'buy':
            # Compare electronic ask vs MM ask
            elec_ask = self.electronic_ask
            mm_ask = self.mm_ask
            
            if elec_ask <= mm_ask and np.random.random() < 0.7:
                # Likely execute electronic
                venue = 'electronic'
                execution_price = elec_ask
            else:
                venue = 'mm'
                execution_price = mm_ask
        else:  # sell
            # Compare electronic bid vs MM bid
            elec_bid = self.electronic_bid
            mm_bid = self.mm_bid
            
            if elec_bid >= mm_bid and np.random.random() < 0.7:
                venue = 'electronic'
                execution_price = elec_bid
            else:
                venue = 'mm'
                execution_price = mm_bid
        
        return venue, execution_price
    
    def execute_order(self, side, quantity):
        """Execute order and update market"""
        venue, price = self.route_order(side, quantity)
        
        # Update MM inventory
        if venue == 'mm':
            if side == 'buy':
                self.mm_inventory -= quantity
            else:
                self.mm_inventory += quantity
            
            # MM adjusts quotes based on inventory
            inventory_adjustment = 0.002 * self.mm_inventory / 10000
            self.mm_bid = self.price - 0.01 + inventory_adjustment
            self.mm_ask = self.price + 0.01 + inventory_adjustment
        
        # Update electronic book (simplified)
        self.electronic_bid = self.price - 0.005
        self.electronic_ask = self.price + 0.005
        
        # Update market price toward execution price
        self.price = 0.7 * self.price + 0.3 * price
        
        # Record trade
        self.trade_log.append({
            'side': side,
            'venue': venue,
            'price': price,
            'quantity': quantity,
            'mm_inventory': self.mm_inventory
        })
        
        self.routing_log.append(venue)
        spread = self.mm_ask - self.mm_bid
        self.spread_history.append(spread)
        self.price_history.append(self.price)
        
        return price

# Scenario 1: Normal hybrid market
print("Scenario 1: Normal Hybrid Market Routing")
print("=" * 80)

hybrid1 = HybridMarket()

for t in range(200):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000, 5000])
    price = hybrid1.execute_order(side, quantity)

electronic_ratio = hybrid1.routing_log.count('electronic') / len(hybrid1.routing_log) * 100
mm_ratio = 100 - electronic_ratio

print(f"Total Orders: {len(hybrid1.routing_log)}")
print(f"Electronic Routing: {electronic_ratio:.1f}%")
print(f"MM Routing: {mm_ratio:.1f}%")
print(f"Average Spread: ${np.mean(hybrid1.spread_history):.4f}")
print(f"MM Final Inventory: {hybrid1.mm_inventory:,.0f} shares")

# Scenario 2: Stress (wider MM quotes)
print(f"\n\nScenario 2: Market Stress (MM Widens Spreads)")
print("=" * 80)

hybrid2 = HybridMarket()

for t in range(100):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000])
    price = hybrid2.execute_order(side, quantity)

# Stress: MM reduces liquidity (widens spread)
for t in range(100):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000])
    
    # MM widens spread 10x in stress
    hybrid2.mm_bid = hybrid2.price - 0.05
    hybrid2.mm_ask = hybrid2.price + 0.05
    
    price = hybrid2.execute_order(side, quantity)

normal_period_spread = np.mean(hybrid2.spread_history[:100])
stress_period_spread = np.mean(hybrid2.spread_history[100:])

print(f"Normal Period Average Spread: ${normal_period_spread:.4f}")
print(f"Stress Period Average Spread: ${stress_period_spread:.4f}")
print(f"Spread Widening: {stress_period_spread / normal_period_spread:.1f}x")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Routing pie chart
routing_counts = [hybrid1.routing_log.count('electronic'), hybrid1.routing_log.count('mm')]
labels = ['Electronic', 'MM']
colors = ['blue', 'orange']

axes[0, 0].pie(routing_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[0, 0].set_title('Scenario 1: Order Routing Distribution')

# Plot 2: Price evolution
times = range(len(hybrid1.price_history))
axes[0, 1].plot(times, hybrid1.price_history, linewidth=2, marker='o', markersize=2)
axes[0, 1].set_xlabel('Trade Number')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].set_title('Scenario 1: Price Evolution')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Spread comparison
scenarios = ['Normal\nMarket', 'Stress\nMarket']
spreads = [normal_period_spread, stress_period_spread]
colors_stress = ['green', 'red']

bars = axes[1, 0].bar(scenarios, spreads, color=colors_stress, alpha=0.7)
axes[1, 0].set_ylabel('Average Spread ($)')
axes[1, 0].set_title('Scenario 2: Spread Widening in Stress')
axes[1, 0].grid(alpha=0.3, axis='y')

for bar, spread in zip(bars, spreads):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'${spread:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: MM inventory
inventory_path = [t['mm_inventory'] for t in hybrid1.trade_log]
times = range(len(inventory_path))

axes[1, 1].plot(times, inventory_path, linewidth=2, color='purple')
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 1].fill_between(times, 0, inventory_path, alpha=0.3)
axes[1, 1].set_xlabel('Trade Number')
axes[1, 1].set_ylabel('MM Inventory (shares)')
axes[1, 1].set_title('Scenario 1: Market Maker Inventory')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n\nHybrid Market Summary:")
print("=" * 80)
print(f"Average Price: ${np.mean(hybrid1.price_history):.2f}")
print(f"Price Range: ${min(hybrid1.price_history):.2f} - ${max(hybrid1.price_history):.2f}")
print(f"Final MM Inventory: {hybrid1.mm_inventory:,.0f} shares")
