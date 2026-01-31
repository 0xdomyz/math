import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Generate realistic intraday volume profile
np.random.seed(42)

n_minutes = 120  # 2-hour execution window
initial_price = 100.0

# Market volume: U-shaped curve with random fluctuations
time_grid = np.arange(n_minutes)
base_volume = 1000 * (1 + 0.5*np.cos(2*np.pi*time_grid/n_minutes - np.pi))  # U-shape

# Add volume spikes
volume_spikes = np.zeros(n_minutes)
volume_spikes[30] = 5000  # Spike at minute 30
volume_spikes[75] = 3000  # Spike at minute 75

# Add random noise
volume_noise = np.random.lognormal(0, 0.5, n_minutes)
market_volume = (base_volume + volume_spikes) * volume_noise

# Price path (correlated with volume)
price_changes = np.random.normal(0.0001, 0.001, n_minutes)
price_changes += 0.00002 * (market_volume / market_volume.mean() - 1)  # Volume impact
prices = initial_price * np.exp(np.cumsum(price_changes))

# Order parameters
total_quantity = 100000  # shares to execute
target_participation = 0.10  # 10% POV target

# ============================================================================
# POV IMPLEMENTATION
# ============================================================================

def execute_pov(prices, market_volume, total_qty, target_rate=0.10, 
               min_rate=0.05, max_rate=0.20, min_slice=100):
    """
    Execute Participation Rate (POV) strategy.
    
    target_rate: Target % of market volume (e.g., 0.10 = 10%)
    min_rate, max_rate: Participation bounds
    min_slice: Minimum order size
    """
    n_periods = len(prices)
    
    execution_prices = []
    execution_times = []
    executed_qty = []
    participation_rates = []
    
    remaining = total_qty
    
    for t in range(n_periods):
        if remaining <= 0:
            break
        
        # Calculate target slice based on market volume
        target_slice = target_rate * market_volume[t]
        
        # Apply constraints
        target_slice = max(min_slice, target_slice)  # Minimum size
        target_slice = min(target_slice, max_rate * market_volume[t])  # Max participation
        target_slice = min(target_slice, remaining)  # Don't exceed remaining
        
        if target_slice >= min_slice:
            # Market impact proportional to participation
            actual_participation = target_slice / market_volume[t] if market_volume[t] > 0 else 0
            market_impact = 0.00005 * actual_participation * prices[t]  # Impact function
            
            execution_price = prices[t] * (1 + market_impact)
            
            execution_prices.append(execution_price)
            execution_times.append(t)
            executed_qty.append(target_slice)
            participation_rates.append(actual_participation)
            
            remaining -= target_slice
    
    execution_prices = np.array(execution_prices)
    executed_qty = np.array(executed_qty)
    participation_rates = np.array(participation_rates)
    
    total_executed = executed_qty.sum()
    avg_price = np.sum(execution_prices * executed_qty) / total_executed if total_executed > 0 else prices[0]
    arrival_price = prices[0]
    slippage_bps = (avg_price - arrival_price) / arrival_price * 10000
    
    # Calculate market VWAP for comparison
    market_vwap = np.sum(prices * market_volume) / market_volume.sum()
    vwap_diff_bps = (avg_price - market_vwap) / market_vwap * 10000
    
    return {
        'execution_prices': execution_prices,
        'execution_times': execution_times,
        'executed_qty': executed_qty,
        'participation_rates': participation_rates,
        'avg_price': avg_price,
        'arrival_price': arrival_price,
        'slippage_bps': slippage_bps,
        'total_executed': total_executed,
        'completion_rate': total_executed / total_qty,
        'avg_participation': participation_rates.mean(),
        'market_vwap': market_vwap,
        'vwap_diff_bps': vwap_diff_bps
    }

# ============================================================================
# TWAP BASELINE (for comparison)
# ============================================================================

def execute_twap_baseline(prices, total_qty, duration):
    """Simple TWAP for comparison."""
    slice_size = total_qty / duration
    
    execution_prices = []
    executed_qty = []
    
    for t in range(duration):
        impact = 0.00001 * slice_size * prices[t]
        execution_price = prices[t] * (1 + impact)
        execution_prices.append(execution_price)
        executed_qty.append(slice_size)
    
    execution_prices = np.array(execution_prices)
    executed_qty = np.array(executed_qty)
    
    avg_price = np.sum(execution_prices * executed_qty) / total_qty
    slippage_bps = (avg_price - prices[0]) / prices[0] * 10000
    
    return {
        'avg_price': avg_price,
        'slippage_bps': slippage_bps,
        'execution_prices': execution_prices,
        'executed_qty': executed_qty
    }

# ============================================================================
# EXECUTION SIMULATIONS
# ============================================================================

print("="*70)
print("POV (PARTICIPATION RATE) EXECUTION ANALYSIS")
print("="*70)

# Run POV at different participation rates
pov_rates = [0.05, 0.10, 0.15, 0.20]
pov_results = {}

for rate in pov_rates:
    result = execute_pov(prices, market_volume, total_quantity, 
                        target_rate=rate, min_rate=rate*0.5, max_rate=rate*2)
    pov_results[rate] = result
    
    print(f"\nPOV {rate*100:.0f}% Target:")
    print(f"   Completed: {result['total_executed']:,.0f} / {total_quantity:,} "
          f"({result['completion_rate']*100:.1f}%)")
    print(f"   Avg Participation: {result['avg_participation']*100:.2f}%")
    print(f"   Avg Execution Price: ${result['avg_price']:.2f}")
    print(f"   Slippage: {result['slippage_bps']:.2f} bps")
    print(f"   vs Market VWAP: {result['vwap_diff_bps']:+.2f} bps")

# TWAP baseline
twap_result = execute_twap_baseline(prices, total_quantity, n_minutes)
print(f"\nTWAP Baseline:")
print(f"   Avg Execution Price: ${twap_result['avg_price']:.2f}")
print(f"   Slippage: {twap_result['slippage_bps']:.2f} bps")

# ============================================================================
# ADAPTIVE POV WITH VOLUME SPIKE HANDLING
# ============================================================================

print("\n" + "="*70)
print("ADAPTIVE POV: Volume Spike Mitigation")
print("="*70)

def execute_adaptive_pov(prices, market_volume, total_qty, target_rate=0.10,
                        spike_threshold=3.0):
    """
    Adaptive POV that caps participation during volume spikes.
    spike_threshold: Multiple of average volume to classify as spike
    """
    avg_volume = market_volume.mean()
    
    execution_prices = []
    execution_times = []
    executed_qty = []
    participation_rates = []
    spike_flags = []
    
    remaining = total_qty
    
    for t in range(len(prices)):
        if remaining <= 0:
            break
        
        # Detect volume spike
        is_spike = market_volume[t] > spike_threshold * avg_volume
        
        # Reduce participation during spikes
        if is_spike:
            effective_rate = target_rate * 0.5  # Cut participation in half
        else:
            effective_rate = target_rate
        
        target_slice = effective_rate * market_volume[t]
        target_slice = max(100, min(target_slice, remaining))
        
        if target_slice >= 100:
            actual_participation = target_slice / market_volume[t]
            market_impact = 0.00005 * actual_participation * prices[t]
            execution_price = prices[t] * (1 + market_impact)
            
            execution_prices.append(execution_price)
            execution_times.append(t)
            executed_qty.append(target_slice)
            participation_rates.append(actual_participation)
            spike_flags.append(is_spike)
            
            remaining -= target_slice
    
    execution_prices = np.array(execution_prices)
    executed_qty = np.array(executed_qty)
    
    total_executed = executed_qty.sum()
    avg_price = np.sum(execution_prices * executed_qty) / total_executed
    slippage_bps = (avg_price - prices[0]) / prices[0] * 10000
    
    return {
        'execution_prices': execution_prices,
        'execution_times': execution_times,
        'executed_qty': executed_qty,
        'participation_rates': participation_rates,
        'spike_flags': spike_flags,
        'avg_price': avg_price,
        'slippage_bps': slippage_bps,
        'total_executed': total_executed,
        'completion_rate': total_executed / total_qty
    }

adaptive_pov = execute_adaptive_pov(prices, market_volume, total_quantity, 
                                   target_rate=0.10, spike_threshold=3.0)

standard_pov = pov_results[0.10]

print(f"\nStandard POV 10%:")
print(f"   Slippage: {standard_pov['slippage_bps']:.2f} bps")
print(f"   Avg Participation: {standard_pov['avg_participation']*100:.2f}%")

print(f"\nAdaptive POV (spike-aware):")
print(f"   Slippage: {adaptive_pov['slippage_bps']:.2f} bps")
print(f"   Avg Participation: {np.mean(adaptive_pov['participation_rates'])*100:.2f}%")
print(f"   Spike periods detected: {sum(adaptive_pov['spike_flags'])}")
print(f"   Slippage Improvement: {standard_pov['slippage_bps'] - adaptive_pov['slippage_bps']:.2f} bps")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Market volume with POV executions
ax1 = axes[0, 0]
ax1.bar(time_grid, market_volume, alpha=0.3, color='gray', label='Market Volume')
ax1_twin = ax1.twinx()

pov_10 = pov_results[0.10]
ax1_twin.scatter(pov_10['execution_times'], pov_10['executed_qty'], 
                c='blue', s=50, alpha=0.7, label='POV 10% Executions')
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Market Volume', color='gray')
ax1_twin.set_ylabel('POV Slice Size', color='blue')
ax1.set_title('POV Execution: Following Market Volume')
ax1.tick_params(axis='y', labelcolor='gray')
ax1_twin.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, alpha=0.3)

# Plot 2: Participation rate over time
ax2 = axes[0, 1]
ax2.plot(pov_10['execution_times'], np.array(pov_10['participation_rates'])*100, 
        'o-', linewidth=2, markersize=5, label='Actual Participation')
ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Target: 10%')
ax2.fill_between(pov_10['execution_times'], 5, 20, alpha=0.1, color='red')
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('Participation Rate (%)')
ax2.set_title('POV 10%: Actual vs Target Participation')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Cumulative execution comparison
ax3 = axes[0, 2]
for rate in [0.05, 0.10, 0.15, 0.20]:
    result = pov_results[rate]
    cumulative = np.cumsum(result['executed_qty'])
    ax3.plot(result['execution_times'], cumulative, 
            linewidth=2, label=f'POV {rate*100:.0f}%', marker='o', markersize=3)

ax3.axhline(y=total_quantity, color='black', linestyle='--', linewidth=2, label='Target')
ax3.set_xlabel('Time (minutes)')
ax3.set_ylabel('Cumulative Quantity')
ax3.set_title('POV: Completion Speed by Target Rate')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Price path with adaptive POV
ax4 = axes[1, 0]
ax4.plot(time_grid, prices, 'k-', linewidth=2, alpha=0.7, label='Market Price')
ax4.scatter(adaptive_pov['execution_times'], adaptive_pov['execution_prices'],
           c=['red' if spike else 'blue' for spike in adaptive_pov['spike_flags']],
           s=60, alpha=0.6, label='Executions (red=spike)')
ax4.axhline(y=adaptive_pov['avg_price'], color='green', linestyle='--', 
           linewidth=2, label=f'Avg: ${adaptive_pov["avg_price"]:.2f}')
ax4.set_xlabel('Time (minutes)')
ax4.set_ylabel('Price ($)')
ax4.set_title('Adaptive POV: Spike Mitigation')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Plot 5: Slippage comparison
ax5 = axes[1, 1]
strategies = ['TWAP', 'POV 5%', 'POV 10%', 'POV 15%', 'POV 20%', 'Adaptive\nPOV']
slippages = [
    twap_result['slippage_bps'],
    pov_results[0.05]['slippage_bps'],
    pov_results[0.10]['slippage_bps'],
    pov_results[0.15]['slippage_bps'],
    pov_results[0.20]['slippage_bps'],
    adaptive_pov['slippage_bps']
]
colors = ['gray', 'lightblue', 'blue', 'darkblue', 'navy', 'green']
bars = ax5.bar(strategies, slippages, color=colors, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Slippage (bps)')
ax5.set_title('Execution Strategy Comparison')
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, slip in zip(bars, slippages):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{slip:.1f}', ha='center', va='bottom', fontsize=9)

# Plot 6: Completion rate vs slippage tradeoff
ax6 = axes[1, 2]
completion_rates = [100] + [pov_results[r]['completion_rate']*100 for r in pov_rates] + [adaptive_pov['completion_rate']*100]
slippage_values = [twap_result['slippage_bps']] + [pov_results[r]['slippage_bps'] for r in pov_rates] + [adaptive_pov['slippage_bps']]
labels = ['TWAP'] + [f'POV {r*100:.0f}%' for r in pov_rates] + ['Adaptive']

ax6.scatter(completion_rates, slippage_values, s=150, alpha=0.7, c=colors)
for i, label in enumerate(labels):
    ax6.annotate(label, (completion_rates[i], slippage_values[i]), 
                fontsize=8, ha='right', va='bottom')
ax6.set_xlabel('Completion Rate (%)')
ax6.set_ylabel('Slippage (bps)')
ax6.set_title('Execution Efficiency Frontier')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pov_execution_analysis.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved: pov_execution_analysis.png")
plt.show()
