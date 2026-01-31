import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Pre-trade control system
class PreTradeControlSystem:
    def __init__(self):
        # Position limits
        self.position_limit_per_stock = 100_000  # shares
        self.notional_limit_per_order = 5_000_000  # $5M
        self.sector_limit = {'Technology': 100_000_000, 'Finance': 80_000_000}  # $100M, $80M
        
        # Current state
        self.positions = {'AAPL': 80_000, 'MSFT': 50_000, 'GOOGL': 60_000}  # shares
        self.sector_exposure = {'Technology': 75_000_000, 'Finance': 40_000_000}  # $
        
        # Fat-finger thresholds
        self.max_quantity = 500_000  # shares per order
        self.max_price_deviation = 0.10  # 10% from last trade
        
        # Order history (for duplicate detection)
        self.recent_orders = []
        
        # Audit log
        self.audit_log = []
    
    def validate_order(self, order):
        """Run all pre-trade checks"""
        checks = {
            'symbol_check': self.check_symbol(order),
            'quantity_check': self.check_quantity(order),
            'price_check': self.check_price(order),
            'notional_check': self.check_notional(order),
            'position_limit_check': self.check_position_limit(order),
            'sector_limit_check': self.check_sector_limit(order),
            'duplicate_check': self.check_duplicate(order),
            'fat_finger_check': self.check_fat_finger(order)
        }
        
        # Aggregate results
        passed = all(checks.values())
        failed_checks = [k for k, v in checks.items() if not v]
        
        # Log
        self.audit_log.append({
            'timestamp': datetime.now(),
            'order': order,
            'passed': passed,
            'failed_checks': failed_checks
        })
        
        return passed, failed_checks
    
    def check_symbol(self, order):
        """Validate symbol exists and is tradeable"""
        approved_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        return order['symbol'] in approved_symbols
    
    def check_quantity(self, order):
        """Validate quantity within bounds"""
        qty = order['quantity']
        return 0 < qty <= self.max_quantity
    
    def check_price(self, order):
        """Validate price not too far from market"""
        if order['order_type'] == 'MARKET':
            return True  # Market orders don't have price limit
        
        last_price = order['last_trade']
        limit_price = order['limit_price']
        
        if order['side'] == 'BUY':
            # Buy limit should not be way above market (overpaying)
            max_price = last_price * (1 + self.max_price_deviation)
            return limit_price <= max_price
        else:  # SELL
            # Sell limit should not be way below market (giving away)
            min_price = last_price * (1 - self.max_price_deviation)
            return limit_price >= min_price
    
    def check_notional(self, order):
        """Validate order notional within limit"""
        if order['order_type'] == 'MARKET':
            price = order['last_trade']
        else:
            price = order['limit_price']
        
        notional = order['quantity'] * price
        return notional <= self.notional_limit_per_order
    
    def check_position_limit(self, order):
        """Validate position after order would not exceed limit"""
        symbol = order['symbol']
        current_position = self.positions.get(symbol, 0)
        
        if order['side'] == 'BUY':
            projected_position = current_position + order['quantity']
        else:
            projected_position = current_position - order['quantity']
        
        return abs(projected_position) <= self.position_limit_per_stock
    
    def check_sector_limit(self, order):
        """Validate sector exposure after order"""
        sector = order.get('sector', 'Technology')  # Default
        
        if order['order_type'] == 'MARKET':
            price = order['last_trade']
        else:
            price = order['limit_price']
        
        order_notional = order['quantity'] * price
        
        if order['side'] == 'BUY':
            projected_exposure = self.sector_exposure.get(sector, 0) + order_notional
        else:
            projected_exposure = self.sector_exposure.get(sector, 0) - order_notional
        
        sector_limit = self.sector_limit.get(sector, 1e12)  # Default unlimited
        return abs(projected_exposure) <= sector_limit
    
    def check_duplicate(self, order):
        """Check for duplicate orders in last 5 seconds"""
        now = datetime.now()
        for recent in self.recent_orders:
            time_diff = (now - recent['timestamp']).total_seconds()
            
            if time_diff < 5:  # Within 5 seconds
                # Check if same symbol, side, quantity
                if (recent['symbol'] == order['symbol'] and
                    recent['side'] == order['side'] and
                    recent['quantity'] == order['quantity']):
                    return False  # Duplicate detected
        
        # Add to recent orders
        self.recent_orders.append({
            'timestamp': now,
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': order['quantity']
        })
        
        # Cleanup old orders (>1 minute)
        self.recent_orders = [o for o in self.recent_orders 
                             if (now - o['timestamp']).total_seconds() < 60]
        
        return True  # Not a duplicate
    
    def check_fat_finger(self, order):
        """Detect potential fat-finger errors"""
        # Check 1: Quantity ends in many zeros (e.g., 1,000,000 vs intended 10,000)
        qty = order['quantity']
        if qty >= 100_000 and qty % 100_000 == 0:
            # Likely fat-finger (e.g., 1,000,000; 500,000)
            return False
        
        # Check 2: Price is round number AND far from market (e.g., $100 when trading $50)
        if order['order_type'] != 'MARKET':
            limit = order['limit_price']
            last = order['last_trade']
            
            # Round number (e.g., $50.00, $100.00)
            if limit == int(limit) and abs(limit - last) / last > 0.20:
                return False  # 20% away AND round; suspicious
        
        return True
    
    def execute_order(self, order):
        """Execute order (update positions)"""
        symbol = order['symbol']
        quantity = order['quantity']
        
        if order['side'] == 'BUY':
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        # Update sector exposure (simplified)
        sector = order.get('sector', 'Technology')
        price = order.get('limit_price', order['last_trade'])
        notional = quantity * price
        
        if order['side'] == 'BUY':
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + notional
        else:
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) - notional

# Generate test orders (mix of valid and violations)
def generate_test_orders():
    orders = [
        # Order 1: Valid order
        {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 10_000,
            'order_type': 'LIMIT',
            'limit_price': 180.0,
            'last_trade': 178.0,
            'sector': 'Technology'
        },
        # Order 2: Position limit violation
        {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 50_000,  # Current 80k + 50k = 130k (exceeds 100k limit)
            'order_type': 'LIMIT',
            'limit_price': 180.0,
            'last_trade': 178.0,
            'sector': 'Technology'
        },
        # Order 3: Fat-finger (quantity)
        {
            'symbol': 'MSFT',
            'side': 'BUY',
            'quantity': 1_000_000,  # Suspicious (many zeros)
            'order_type': 'MARKET',
            'last_trade': 350.0,
            'sector': 'Technology'
        },
        # Order 4: Price deviation violation
        {
            'symbol': 'GOOGL',
            'side': 'SELL',
            'quantity': 5_000,
            'order_type': 'LIMIT',
            'limit_price': 120.0,  # Last $150; selling $120 = 20% below (likely error)
            'last_trade': 150.0,
            'sector': 'Technology'
        },
        # Order 5: Notional limit violation
        {
            'symbol': 'AMZN',
            'side': 'BUY',
            'quantity': 40_000,
            'order_type': 'LIMIT',
            'limit_price': 180.0,  # 40k × $180 = $7.2M (exceeds $5M limit)
            'last_trade': 178.0,
            'sector': 'Technology'
        },
        # Order 6: Unknown symbol
        {
            'symbol': 'ABCD',  # Not in approved list
            'side': 'BUY',
            'quantity': 1_000,
            'order_type': 'MARKET',
            'last_trade': 50.0,
            'sector': 'Technology'
        },
        # Order 7: Valid order (different stock)
        {
            'symbol': 'TSLA',
            'side': 'SELL',
            'quantity': 5_000,
            'order_type': 'LIMIT',
            'limit_price': 240.0,
            'last_trade': 242.0,
            'sector': 'Technology'
        }
    ]
    return orders

# Run simulation
system = PreTradeControlSystem()
orders = generate_test_orders()

results = []
for i, order in enumerate(orders, 1):
    passed, failed_checks = system.validate_order(order)
    
    results.append({
        'Order #': i,
        'Symbol': order['symbol'],
        'Side': order['side'],
        'Quantity': f"{order['quantity']:,}",
        'Type': order['order_type'],
        'Price': f"${order.get('limit_price', order['last_trade']):.2f}",
        'Passed': '✓' if passed else '✗',
        'Violations': ', '.join(failed_checks) if failed_checks else 'None'
    })
    
    if passed:
        # Execute order (update positions)
        system.execute_order(order)

df_results = pd.DataFrame(results)

print("="*100)
print("Pre-Trade Control System: Order Validation Results")
print("="*100)
print(df_results.to_string(index=False))

# Statistics
total_orders = len(orders)
passed_orders = sum(1 for r in results if r['Passed'] == '✓')
rejected_orders = total_orders - passed_orders

print(f"\n{'='*100}")
print("Summary Statistics")
print(f"{'='*100}")
print(f"Total orders submitted: {total_orders}")
print(f"Orders passed: {passed_orders} ({passed_orders/total_orders*100:.0f}%)")
print(f"Orders rejected: {rejected_orders} ({rejected_orders/total_orders*100:.0f}%)")
print(f"\nRejection reasons breakdown:")
for check_type in ['position_limit_check', 'fat_finger_check', 'price_check', 
                   'notional_check', 'symbol_check']:
    count = sum(1 for r in results if check_type in r['Violations'])
    if count > 0:
        print(f"  - {check_type.replace('_', ' ').title()}: {count}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Pass vs Reject
labels = ['Passed', 'Rejected']
sizes = [passed_orders, rejected_orders]
colors = ['green', 'red']
explode = (0.1, 0)

axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.0f%%',
            shadow=True, startangle=90)
axes[0].set_title('Order Approval Rate')

# Chart 2: Rejection reasons
rejection_reasons = {}
for r in results:
    if r['Passed'] == '✗':
        for violation in r['Violations'].split(', '):
            rejection_reasons[violation] = rejection_reasons.get(violation, 0) + 1

if rejection_reasons:
    reasons_sorted = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)
    reasons_labels = [r[0].replace('_check', '').replace('_', ' ').title() for r in reasons_sorted]
    reasons_counts = [r[1] for r in reasons_sorted]
    
    axes[1].barh(reasons_labels, reasons_counts, color='orange')
    axes[1].set_xlabel('Count')
    axes[1].set_title('Rejection Reasons')
    axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('pre_trade_controls.png', dpi=300, bbox_inches='tight')
plt.show()

# Final positions
print(f"\n{'='*100}")
print("Updated Positions (After Approved Orders)")
print(f"{'='*100}")
for symbol, position in sorted(system.positions.items()):
    utilization = abs(position) / system.position_limit_per_stock * 100
    status = 'OK' if utilization < 90 else 'WARNING'
    print(f"{symbol}: {position:>10,} shares ({utilization:>5.1f}% of limit) [{status}]")

print(f"\nSector Exposure:")
for sector, exposure in sorted(system.sector_exposure.items()):
    limit = system.sector_limit[sector]
    utilization = abs(exposure) / limit * 100
    status = 'OK' if utilization < 90 else 'WARNING'
    print(f"{sector}: ${exposure:>12,.0f} ({utilization:>5.1f}% of ${limit:,.0f}) [{status}]")