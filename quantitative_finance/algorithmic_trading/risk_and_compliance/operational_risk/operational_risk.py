import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate trading day with operational errors
np.random.seed(42)
n_seconds = 23400  # 6.5 hours = 23,400 seconds
timestamps = np.arange(n_seconds)

# Normal order flow: 5 orders/second
normal_order_rate = 5
normal_orders = np.random.poisson(normal_order_rate, n_seconds)

# Inject operational errors
def inject_errors(orders, timestamps):
    """Inject various operational error scenarios"""
    error_log = []
    
    # Scenario 1: Fat-finger at t=3600 (1 hour in)
    fat_finger_time = 3600
    orders[fat_finger_time:fat_finger_time+10] = 500  # 500 orders/sec for 10 seconds
    error_log.append({
        'time': fat_finger_time,
        'type': 'Fat Finger',
        'severity': 'High',
        'description': 'Trader entered 1,000,000 shares (intended 1,000)'
    })
    
    # Scenario 2: System glitch at t=10800 (3 hours in)
    glitch_time = 10800
    orders[glitch_time:glitch_time+600] = np.random.poisson(100, 600)  # 10-minute burst
    error_log.append({
        'time': glitch_time,
        'type': 'System Glitch',
        'severity': 'Critical',
        'description': 'Algo loop bug; duplicate order submissions'
    })
    
    # Scenario 3: Data feed error at t=18000 (5 hours in)
    data_error_time = 18000
    orders[data_error_time:data_error_time+300] = 0  # No orders for 5 minutes (stale data)
    error_log.append({
        'time': data_error_time,
        'type': 'Data Feed Failure',
        'severity': 'Moderate',
        'description': 'Market data feed disconnected; trading halted'
    })
    
    return orders, error_log

orders_with_errors, error_log = inject_errors(normal_orders.copy(), timestamps)

# Kill switch logic
def apply_kill_switch(orders, error_log):
    """Detect anomalies and trigger kill switch"""
    kill_switch_events = []
    
    # Rolling window: 60-second average
    window = 60
    rolling_avg = pd.Series(orders).rolling(window).mean()
    
    # Baseline: First 1 hour average (before errors)
    baseline = orders[:3600].mean()
    
    for t in range(window, len(orders)):
        # Threshold: 10× normal rate
        if rolling_avg[t] > baseline * 10:
            # Kill switch triggered
            kill_switch_events.append({
                'time': t,
                'rate': rolling_avg[t],
                'threshold': baseline * 10,
                'action': 'KILL SWITCH ACTIVATED'
            })
            
            # Halt for 300 seconds (5 minutes)
            orders[t:min(t+300, len(orders))] = 0
            
            # Reset baseline after recovery
            if t + 300 < len(orders):
                # Resume at normal rate
                orders[t+300:t+600] = np.random.poisson(normal_order_rate, 
                                                       min(300, len(orders)-t-300))
    
    return orders, kill_switch_events

orders_controlled, kill_switch_events = apply_kill_switch(orders_with_errors.copy(), error_log)

# Calculate cumulative P&L impact
# Assume: Normal orders = $0.01 profit/share; Errors = -$5 loss/share; High volume = high slippage
def calculate_pnl(orders, baseline_rate=5):
    """Calculate P&L; penalize anomalous order rates"""
    pnl = np.zeros(len(orders))
    
    for t in range(len(orders)):
        if orders[t] <= baseline_rate * 2:
            # Normal trading: Small profit
            pnl[t] = orders[t] * 100 * 0.01  # 100 shares/order, $0.01/share
        else:
            # Anomalous: Loss due to market impact
            pnl[t] = -orders[t] * 100 * 5  # $5 loss/share (slippage + adverse selection)
    
    return np.cumsum(pnl)

pnl_without_control = calculate_pnl(orders_with_errors)
pnl_with_control = calculate_pnl(orders_controlled)

print("="*70)
print("Operational Risk Simulation Results")
print("="*70)
print(f"Baseline order rate: {normal_order_rate} orders/second")
print(f"Trading duration: {n_seconds/3600:.1f} hours")
print(f"\nInjected Errors:")
for i, err in enumerate(error_log, 1):
    print(f"{i}. {err['type']} at t={err['time']}s ({err['time']/3600:.1f}h)")
    print(f"   Severity: {err['severity']}")
    print(f"   Description: {err['description']}")

print(f"\nKill Switch Activations: {len(kill_switch_events)}")
for i, ks in enumerate(kill_switch_events, 1):
    print(f"{i}. Time: {ks['time']}s ({ks['time']/3600:.1f}h)")
    print(f"   Rate: {ks['rate']:.1f} orders/sec (threshold: {ks['threshold']:.1f})")
    print(f"   Action: {ks['action']}")

print(f"\nP&L Impact:")
print(f"Without kill switch: ${pnl_without_control[-1]:,.0f}")
print(f"With kill switch: ${pnl_with_control[-1]:,.0f}")
print(f"Loss prevented: ${(pnl_without_control[-1] - pnl_with_control[-1]):,.0f}")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Order rate time series
axes[0].plot(timestamps/3600, orders_with_errors, label='Without Kill Switch', 
             color='red', alpha=0.6, linewidth=0.8)
axes[0].plot(timestamps/3600, orders_controlled, label='With Kill Switch', 
             color='green', alpha=0.8, linewidth=1.2)
axes[0].axhline(normal_order_rate, color='blue', linestyle='--', 
                label=f'Normal Rate ({normal_order_rate}/sec)')
axes[0].axhline(normal_order_rate * 10, color='orange', linestyle='--', 
                label='Kill Switch Threshold (10× normal)')

# Mark error events
for err in error_log:
    axes[0].axvline(err['time']/3600, color='purple', alpha=0.3, linestyle=':')
    axes[0].text(err['time']/3600, orders_with_errors.max()*0.9, 
                err['type'].split()[0], rotation=90, fontsize=8)

# Mark kill switch events
for ks in kill_switch_events:
    axes[0].scatter(ks['time']/3600, ks['rate'], color='red', s=100, 
                   marker='X', zorder=5, label='Kill Switch' if ks == kill_switch_events[0] else '')

axes[0].set_xlabel('Time (hours)')
axes[0].set_ylabel('Order Rate (orders/sec)')
axes[0].set_title('Operational Risk: Order Flow Anomalies & Kill Switch Response')
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

# Cumulative P&L
axes[1].plot(timestamps/3600, pnl_without_control/1e6, label='Without Kill Switch', 
             color='red', linewidth=2)
axes[1].plot(timestamps/3600, pnl_with_control/1e6, label='With Kill Switch', 
             color='green', linewidth=2)
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)

# Mark error events
for err in error_log:
    axes[1].axvline(err['time']/3600, color='purple', alpha=0.3, linestyle=':')

axes[1].set_xlabel('Time (hours)')
axes[1].set_ylabel('Cumulative P&L ($M)')
axes[1].set_title('P&L Impact: Kill Switch Prevents Catastrophic Losses')
axes[1].legend(loc='lower left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('operational_risk_kill_switch.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print(f"\n{'='*70}")
print("Statistical Summary")
print(f"{'='*70}")
print(f"Max order rate (without control): {orders_with_errors.max()} orders/sec")
print(f"Max order rate (with control): {orders_controlled.max()} orders/sec")
print(f"Total orders (without control): {orders_with_errors.sum():,.0f}")
print(f"Total orders (with control): {orders_controlled.sum():,.0f}")
print(f"Orders prevented: {(orders_with_errors.sum() - orders_controlled.sum()):,.0f}")
print(f"\nAverage order rate (without control): {orders_with_errors.mean():.2f}/sec")
print(f"Average order rate (with control): {orders_controlled.mean():.2f}/sec")