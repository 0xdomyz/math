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
