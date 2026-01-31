import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(42)

# Stop Order Cascade Simulator
class MarketWithStops:
    def __init__(self, initial_price=100, daily_vol=0.02):
        self.price = initial_price
        self.daily_vol = daily_vol
        self.price_history = [initial_price]
        self.volume_history = [0]
        self.stop_orders = []  # List of (side, stop_price, quantity)
        self.trades = []
        self.cascades = []
        
    def add_stop_order(self, side, stop_price, quantity):
        """Add stop order"""
        self.stop_orders.append({
            'side': side,
            'stop_price': stop_price,
            'quantity': quantity,
            'triggered': False
        })
    
    def check_stops(self, current_price):
        """Check if any stops triggered"""
        triggered = []
        
        for i, stop in enumerate(self.stop_orders):
            if stop['triggered']:
                continue
            
            # Sell stops trigger below price; buy stops trigger above
            if stop['side'] == 'sell' and current_price <= stop['stop_price']:
                triggered.append(i)
                stop['triggered'] = True
            elif stop['side'] == 'buy' and current_price >= stop['stop_price']:
                triggered.append(i)
                stop['triggered'] = True
        
        return triggered
    
    def process_stop_cascade(self, triggered_indices, base_liquidity=1000):
        """Process triggered stops, model cascade effect"""
        cascade_volume = 0
        cascade_volume_history = []
        
        for idx in triggered_indices:
            stop = self.stop_orders[idx]
            cascade_volume += stop['quantity']
        
        # Market impact: ΔP ≈ sqrt(Q/V) effect
        # Large volume → large price drop → triggers more stops
        
        rounds = 0
        max_rounds = 50  # Prevent infinite loops
        
        while cascade_volume > 0 and rounds < max_rounds:
            rounds += 1
            
            # Market impact from cascade volume
            impact = (cascade_volume / base_liquidity) * self.daily_vol * 0.5
            self.price *= (1 - impact)  # Price drops
            
            cascade_volume_history.append(cascade_volume)
            
            # Check if new stops triggered
            new_triggered = self.check_stops(self.price)
            
            new_cascade = 0
            for idx in new_triggered:
                stop = self.stop_orders[idx]
                if not stop['triggered']:
                    new_cascade += stop['quantity']
                    stop['triggered'] = True
            
            cascade_volume = new_cascade
        
        return {
            'rounds': rounds,
            'final_price': self.price,
            'cascade_volume_history': cascade_volume_history
        }
    
    def simulate_price_path(self, n_periods=100, dt=1):
        """Simulate price path with random walk"""
        for _ in range(n_periods):
            # Random return
            ret = np.random.normal(0, self.daily_vol)
            self.price *= (1 + ret)
            
            # Check stops
            triggered = self.check_stops(self.price)
            
            if triggered:
                # Cascade event
                cascade_info = self.process_stop_cascade(triggered)
                self.cascades.append({
                    'time': len(self.price_history),
                    'price_before': self.price_history[-1] if self.price_history else self.price,
                    'price_after': cascade_info['final_price'],
                    'cascade_rounds': cascade_info['rounds'],
                    'cascade_depth': len(cascade_info['cascade_volume_history'])
                })
            
            self.price_history.append(self.price)
            self.volume_history.append(sum(s['quantity'] for s in self.stop_orders if s['triggered']))

# Scenario 1: Normal Market (few stops, no cascade)
print("Scenario 1: Normal Market (Few Stops)")
print("=" * 70)

market1 = MarketWithStops(initial_price=100, daily_vol=0.01)

# Distributed stops
for i in range(20):
    stop_price = 100 - np.random.uniform(0.5, 3)
    market1.add_stop_order('sell', stop_price, np.random.randint(100, 500))

market1.simulate_price_path(n_periods=200)

print(f"Initial Price: $100.00")
print(f"Final Price: ${market1.price:.2f}")
print(f"Total Stop Orders: {len(market1.stop_orders)}")
print(f"Triggered: {sum(1 for s in market1.stop_orders if s['triggered'])}")
print(f"Cascades: {len(market1.cascades)}")
if market1.cascades:
    for i, c in enumerate(market1.cascades):
        print(f"  Cascade {i+1}: {c['price_before']:.2f} → {c['price_after']:.2f}, Rounds: {c['cascade_rounds']}")

# Scenario 2: Clustered Stops (cascade risk)
print(f"\n\nScenario 2: Clustered Stops (Cascade Risk)")
print("=" * 70)

market2 = MarketWithStops(initial_price=100, daily_vol=0.015)

# Many stops at similar level (technical support)
stop_level = 95  # Support level
for i in range(100):  # Many traders place stops at support
    quantity = np.random.randint(500, 1000)
    market2.add_stop_order('sell', stop_level + np.random.normal(0, 0.2), quantity)

market2.simulate_price_path(n_periods=200)

print(f"Initial Price: $100.00")
print(f"Final Price: ${market2.price:.2f}")
print(f"Total Stop Orders: {len(market2.stop_orders)}")
print(f"Triggered: {sum(1 for s in market2.stop_orders if s['triggered'])}")
print(f"Cascades: {len(market2.cascades)}")
if market2.cascades:
    print(f"Cascade Details:")
    for i, c in enumerate(market2.cascades):
        pct_drop = (c['price_before'] - c['price_after']) / c['price_before'] * 100
        print(f"  Cascade {i+1}: {c['price_before']:.2f} → {c['price_after']:.2f} ({pct_drop:.1f}%), Rounds: {c['cascade_rounds']}")

# Scenario 3: Extreme Cascade (many clustered stops, fast crash)
print(f"\n\nScenario 3: Extreme Cascade (Flash Crash Simulation)")
print("=" * 70)

market3 = MarketWithStops(initial_price=100, daily_vol=0.025)

# VERY many stops at support level (institutional stops)
stop_level = 95
for i in range(500):
    quantity = np.random.randint(1000, 5000)
    market3.add_stop_order('sell', stop_level + np.random.normal(0, 0.3), quantity)

market3.simulate_price_path(n_periods=100)

print(f"Initial Price: $100.00")
print(f"Final Price: ${market3.price:.2f}")
print(f"Total Stop Orders: {len(market3.stop_orders)}")
print(f"Triggered: {sum(1 for s in market3.stop_orders if s['triggered'])}")
print(f"Cascades: {len(market3.cascades)}")
if market3.cascades:
    print(f"Extreme Cascade Details:")
    for i, c in enumerate(market3.cascades):
        pct_drop = (c['price_before'] - c['price_after']) / c['price_before'] * 100
        print(f"  Cascade {i+1}: {c['price_before']:.2f} → {c['price_after']:.2f} ({pct_drop:.1f}% drop), Cascade Depth: {c['cascade_depth']}")
        print(f"    Total Rounds: {c['cascade_rounds']}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Normal market price path
axes[0, 0].plot(market1.price_history, linewidth=2, color='blue')
axes[0, 0].axhline(y=95, color='red', linestyle='--', linewidth=1, label='Stop Level')
if market1.cascades:
    for cascade in market1.cascades:
        axes[0, 0].plot(cascade['time'], cascade['price_after'], 'ro', markersize=8)
axes[0, 0].set_xlabel('Time Period')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Scenario 1: Normal Market (Few Stops)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Clustered stops price path
axes[0, 1].plot(market2.price_history, linewidth=2, color='green')
axes[0, 1].axhline(y=95, color='red', linestyle='--', linewidth=1, label='Stop Level')
if market2.cascades:
    for cascade in market2.cascades:
        axes[0, 1].plot(cascade['time'], cascade['price_after'], 'ro', markersize=8, label='Cascade')
axes[0, 1].set_xlabel('Time Period')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].set_title('Scenario 2: Clustered Stops (100 orders at ~$95)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Extreme cascade price path
axes[1, 0].plot(market3.price_history, linewidth=2, color='red')
axes[1, 0].axhline(y=95, color='red', linestyle='--', linewidth=1, label='Stop Level')
if market3.cascades:
    for cascade in market3.cascades:
        axes[1, 0].plot(cascade['time'], cascade['price_after'], 'ko', markersize=10, label='Flash Crash')
axes[1, 0].set_xlabel('Time Period')
axes[1, 0].set_ylabel('Price ($)')
axes[1, 0].set_title('Scenario 3: Extreme Cascade (500 stops, ~$95)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Maximum drawdowns comparison
drawdowns1 = [min(market1.price_history[:i+1]) for i in range(len(market1.price_history))]
drawdowns2 = [min(market2.price_history[:i+1]) for i in range(len(market2.price_history))]
drawdowns3 = [min(market3.price_history[:i+1]) for i in range(len(market3.price_history))]

max_dd1 = (100 - min(drawdowns1)) / 100 * 100
max_dd2 = (100 - min(drawdowns2)) / 100 * 100
max_dd3 = (100 - min(drawdowns3)) / 100 * 100

scenarios = ['Normal\n(20 stops)', 'Clustered\n(100 stops)', 'Extreme\n(500 stops)']
max_drawdowns = [max_dd1, max_dd2, max_dd3]
colors = ['blue', 'green', 'red']

axes[1, 1].bar(scenarios, max_drawdowns, color=colors, alpha=0.7)
axes[1, 1].set_ylabel('Maximum Drawdown (%)')
axes[1, 1].set_title('Cascade Impact: Max Drawdown by Stop Clustering')
axes[1, 1].grid(alpha=0.3, axis='y')

for i, (sc, dd) in enumerate(zip(scenarios, max_drawdowns)):
    axes[1, 1].text(i, dd + 0.5, f'{dd:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary Statistics
print(f"\n\nComparative Analysis:")
print("=" * 70)
print(f"{'Metric':<30} {'Scenario 1':<20} {'Scenario 2':<20} {'Scenario 3':<20}")
print("-" * 90)
print(f"{'Initial Price':<30} {'$100.00':<20} {'$100.00':<20} {'$100.00':<20}")
print(f"{'Final Price':<30} {f'${market1.price:.2f}':<20} {f'${market2.price:.2f}':<20} {f'${market3.price:.2f}':<20}")
print(f"{'Max Drawdown':<30} {f'{max_dd1:.2f}%':<20} {f'{max_dd2:.2f}%':<20} {f'{max_dd3:.2f}%':<20}")
print(f"{'Stop Orders':<30} {f'{len(market1.stop_orders)}':<20} {f'{len(market2.stop_orders)}':<20} {f'{len(market3.stop_orders)}':<20}")
print(f"{'Cascades':<30} {f'{len(market1.cascades)}':<20} {f'{len(market2.cascades)}':<20} {f'{len(market3.cascades)}':<20}")

# Cascade statistics
if market3.cascades:
    print(f"\nMost Severe Cascade:")
    worst_cascade = max(market3.cascades, key=lambda x: (x['price_before'] - x['price_after']))
    pct_drop = (worst_cascade['price_before'] - worst_cascade['price_after']) / worst_cascade['price_before'] * 100
    print(f"  Price Drop: {pct_drop:.1f}%")
    print(f"  Cascade Depth: {worst_cascade['cascade_depth']} rounds")
