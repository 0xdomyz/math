import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# VPIN (Volume-Synchronized PIN) Simulation
class VPINCalculator:
    def __init__(self, bucket_volume=50000, n_buckets=50):
        """
        bucket_volume: Fixed volume per bucket (V̄)
        n_buckets: Rolling window size
        """
        self.bucket_volume = bucket_volume
        self.n_buckets = n_buckets
        self.buckets = []
        self.vpin_history = []
        
    def add_trade(self, volume, is_buy):
        """Add trade and update buckets"""
        # Create new bucket if needed
        if len(self.buckets) == 0 or self.buckets[-1]['volume'] >= self.bucket_volume:
            self.buckets.append({
                'buy_volume': 0,
                'sell_volume': 0,
                'volume': 0
            })
        
        # Add to current bucket
        if is_buy:
            self.buckets[-1]['buy_volume'] += volume
        else:
            self.buckets[-1]['sell_volume'] += volume
        
        self.buckets[-1]['volume'] += volume
    
    def calculate_vpin(self):
        """Calculate VPIN over last n_buckets"""
        if len(self.buckets) < self.n_buckets:
            return None
        
        # Take last n buckets
        recent_buckets = self.buckets[-self.n_buckets:]
        
        # Sum order imbalances
        total_imbalance = sum(abs(b['buy_volume'] - b['sell_volume']) 
                            for b in recent_buckets)
        total_volume = sum(b['volume'] for b in recent_buckets)
        
        if total_volume > 0:
            vpin = total_imbalance / total_volume
        else:
            vpin = 0
        
        return vpin

# Simulate market with toxicity events
def simulate_market_with_toxicity(n_trades=10000, toxicity_events=3):
    """
    Simulate order flow with periodic toxicity events
    """
    results = []
    price = 100.0
    
    # Schedule toxicity events
    event_times = np.random.choice(range(1000, n_trades-1000), 
                                   toxicity_events, replace=False)
    event_times = sorted(event_times)
    
    # VPIN calculator
    vpin_calc = VPINCalculator(bucket_volume=50000, n_buckets=50)
    
    for t in range(n_trades):
        # Check if we're in toxicity event
        in_event = any(abs(t - et) < 200 for et in event_times)
        
        if in_event:
            # Toxic flow: 80% informed selling
            informed_fraction = 0.8
            informed_direction = -1  # Selling pressure
        else:
            # Normal: 20% informed, random direction
            informed_fraction = 0.2
            informed_direction = 1 if np.random.random() < 0.5 else -1
        
        # Generate trade
        is_informed = np.random.random() < informed_fraction
        
        if is_informed:
            is_buy = informed_direction > 0
            volume = np.random.uniform(1000, 5000)  # Larger trades
        else:
            is_buy = np.random.random() < 0.5
            volume = np.random.uniform(100, 1000)  # Smaller trades
        
        # Price impact
        if is_informed:
            impact = 0.01 * (1 if is_buy else -1) * (volume / 1000)
        else:
            impact = 0.005 * (1 if is_buy else -1) * (volume / 1000)
        
        price += impact + np.random.normal(0, 0.002)  # Add noise
        
        # Update VPIN
        vpin_calc.add_trade(volume, is_buy)
        vpin = vpin_calc.calculate_vpin()
        
        results.append({
            'time': t,
            'price': price,
            'volume': volume,
            'is_buy': is_buy,
            'is_informed': is_informed,
            'in_event': in_event,
            'vpin': vpin
        })
    
    return results, event_times

# Run simulation
print("Order Flow Toxicity (VPIN) Simulation")
print("=" * 70)

n_trades = 10000
toxicity_events = 3

results, event_times = simulate_market_with_toxicity(n_trades, toxicity_events)

print(f"\nSimulation Parameters:")
print(f"Total Trades: {n_trades}")
print(f"Toxicity Events: {toxicity_events}")
print(f"Event Times: {event_times}")
print(f"Bucket Volume: 50,000 shares")
print(f"Rolling Window: 50 buckets")

# Extract data
times = np.array([r['time'] for r in results])
prices = np.array([r['price'] for r in results])
vpins = np.array([r['vpin'] if r['vpin'] is not None else np.nan for r in results])
in_events = np.array([r['in_event'] for r in results])

# Filter valid VPIN values
valid_vpins = vpins[~np.isnan(vpins)]
valid_times = times[~np.isnan(vpins)]

print(f"\nVPIN Statistics:")
print(f"Mean VPIN: {np.nanmean(vpins):.3f}")
print(f"Max VPIN: {np.nanmax(vpins):.3f}")
print(f"Min VPIN: {np.nanmin(vpins):.3f}")
print(f"Std Dev VPIN: {np.nanstd(vpins):.3f}")

# VPIN during vs outside events
event_vpins = vpins[in_events & ~np.isnan(vpins)]
normal_vpins = vpins[~in_events & ~np.isnan(vpins)]

if len(event_vpins) > 0 and len(normal_vpins) > 0:
    print(f"\nVPIN During Events: {event_vpins.mean():.3f}")
    print(f"VPIN Normal Times: {normal_vpins.mean():.3f}")
    
    t_stat, p_value = stats.ttest_ind(event_vpins, normal_vpins)
    print(f"t-test p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("Result: VPIN significantly higher during toxic events")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price path with events
axes[0, 0].plot(times, prices, linewidth=1, alpha=0.7)

# Mark toxicity events
for et in event_times:
    axes[0, 0].axvspan(et-200, et+200, alpha=0.2, color='red', label='Toxicity Event')

axes[0, 0].set_xlabel('Trade Number')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Price Path with Toxicity Events (Red = Toxic Flow)')
handles, labels = axes[0, 0].get_legend_handles_labels()
if len(handles) > 0:
    axes[0, 0].legend([handles[0]], ['Toxicity Event'])
axes[0, 0].grid(alpha=0.3)

# Plot 2: VPIN time series
axes[0, 1].plot(valid_times, valid_vpins, linewidth=1.5, color='purple')
axes[0, 1].axhline(0.8, color='red', linestyle='--', linewidth=2, 
                  label='Danger Threshold (0.8)')
axes[0, 1].axhline(np.nanmean(vpins), color='green', linestyle='--', 
                  linewidth=1, alpha=0.7, label=f'Mean: {np.nanmean(vpins):.3f}')

# Mark events
for et in event_times:
    axes[0, 1].axvspan(et-200, et+200, alpha=0.2, color='red')

axes[0, 1].set_xlabel('Trade Number')
axes[0, 1].set_ylabel('VPIN')
axes[0, 1].set_title('Volume-Synchronized PIN (VPIN) Over Time')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_ylim([0, 1])

# Plot 3: VPIN distribution
axes[1, 0].hist(valid_vpins, bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[1, 0].axvline(np.nanmean(vpins), color='green', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.nanmean(vpins):.3f}')
axes[1, 0].axvline(0.8, color='red', linestyle='--', linewidth=2,
                  label='Danger: 0.8')
axes[1, 0].set_xlabel('VPIN Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('VPIN Distribution')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: VPIN vs future returns
# Calculate forward returns (next 100 trades)
forward_window = 100
forward_returns = []
vpin_for_regression = []

for i in range(len(results) - forward_window):
    if results[i]['vpin'] is not None:
        future_price = results[i + forward_window]['price']
        current_price = results[i]['price']
        fwd_return = (future_price - current_price) / current_price
        
        forward_returns.append(fwd_return)
        vpin_for_regression.append(results[i]['vpin'])

if len(forward_returns) > 0:
    forward_returns = np.array(forward_returns)
    vpin_for_regression = np.array(vpin_for_regression)
    
    # Scatter plot
    axes[1, 1].scatter(vpin_for_regression, forward_returns * 100, 
                      alpha=0.3, s=10)
    
    # Regression line
    if vpin_for_regression.std() > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            vpin_for_regression, forward_returns * 100
        )
        
        vpin_range = np.linspace(vpin_for_regression.min(), 
                                vpin_for_regression.max(), 100)
        axes[1, 1].plot(vpin_range, slope * vpin_range + intercept, 
                       'r-', linewidth=2, 
                       label=f'Slope={slope:.3f}, R²={r_value**2:.3f}')
    
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('VPIN')
    axes[1, 1].set_ylabel(f'Forward Return (%) [next {forward_window} trades]')
    axes[1, 1].set_title('VPIN Predictive Power')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    print(f"\nPredictive Regression:")
    print(f"Forward Return ~ VPIN")
    print(f"Slope: {slope:.4f}%")
    print(f"R²: {r_value**2:.4f}")
    print(f"p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("Result: VPIN significantly predicts future returns")

plt.tight_layout()
plt.show()

# Alert system simulation
print(f"\nToxicity Alert System:")
danger_threshold = 0.8
warnings = 0
for r in results:
    if r['vpin'] is not None and r['vpin'] > danger_threshold:
        warnings += 1

print(f"Danger Threshold: {danger_threshold}")
print(f"Alerts Triggered: {warnings}/{len(results)} trades ({warnings/len(results)*100:.2f}%)")
print(f"Recommended Action: Widen spreads or withdraw quotes when VPIN > {danger_threshold}")

# Event detection performance
detected_events = 0
for et in event_times:
    event_window_vpins = [r['vpin'] for r in results 
                         if abs(r['time'] - et) < 100 and r['vpin'] is not None]
    if event_window_vpins and max(event_window_vpins) > danger_threshold:
        detected_events += 1

print(f"\nEvent Detection:")
print(f"True Events: {toxicity_events}")
print(f"Detected (VPIN > {danger_threshold}): {detected_events}")
print(f"Detection Rate: {detected_events/toxicity_events*100:.0f}%")
