"""
VPIN Analysis: Real-Time Toxicity Detection & High-Frequency Monitoring
Implements order flow toxicity measures for liquidity risk management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

# ============================================================================
# 1. VPIN CALCULATION & REAL-TIME MONITORING
# ============================================================================

class VPINCalculator:
    """
    Real-time VPIN calculation and toxicity monitoring.
    """
    
    def __init__(self, volume_bucket_size=100):
        """
        Parameters:
        - volume_bucket_size: Number of shares per volume bucket
        """
        self.bucket_size = volume_bucket_size
        self.current_bucket_buys = 0
        self.current_bucket_sells = 0
        self.vpin_history = []
        self.bucket_history = []
        self.timestamps = []
        
    def add_trade(self, size, side, timestamp=None):
        """
        Add a trade to the current bucket.
        
        side: 1 (BUY) or -1 (SELL)
        """
        if side == 1:
            self.current_bucket_buys += size
        else:
            self.current_bucket_sells += size
        
        # Check if bucket complete
        total_bucket = self.current_bucket_buys + self.current_bucket_sells
        
        if total_bucket >= self.bucket_size:
            # Calculate VPIN for completed bucket
            vpin = self._calculate_vpin()
            self.vpin_history.append(vpin)
            
            # Store bucket composition
            self.bucket_history.append({
                'buys': self.current_bucket_buys,
                'sells': self.current_bucket_sells,
                'total': total_bucket,
                'imbalance': self.current_bucket_buys - self.current_bucket_sells
            })
            
            if timestamp:
                self.timestamps.append(timestamp)
            
            # Reset bucket
            self.current_bucket_buys = 0
            self.current_bucket_sells = 0
    
    def _calculate_vpin(self):
        """Calculate VPIN for current bucket."""
        total = self.current_bucket_buys + self.current_bucket_sells
        if total == 0:
            return 0
        
        imbalance = np.abs(self.current_bucket_buys - self.current_bucket_sells)
        vpin = imbalance / total
        return vpin
    
    def get_vpin_signal(self, lookback_window=50):
        """
        Generate real-time alert signal based on VPIN history.
        """
        if len(self.vpin_history) < lookback_window:
            return {'signal': 'INSUFFICIENT_DATA', 'vpin': np.nan}
        
        recent_vpin = np.array(self.vpin_history[-lookback_window:])
        current_vpin = recent_vpin[-1]
        mean_vpin = recent_vpin[:-1].mean()
        std_vpin = recent_vpin[:-1].std()
        
        # Z-score
        if std_vpin > 0:
            z_score = (current_vpin - mean_vpin) / std_vpin
        else:
            z_score = 0
        
        # Classification
        if current_vpin > 0.70:
            signal = 'EXTREME_TOXICITY'
            urgency = 'IMMEDIATE_ACTION'
        elif current_vpin > 0.50:
            signal = 'HIGH_TOXICITY'
            urgency = 'URGENT'
        elif current_vpin > 0.35:
            signal = 'MODERATE_TOXICITY'
            urgency = 'CAUTION'
        elif z_score > 2.0:
            signal = 'SPIKE_DETECTED'
            urgency = 'MONITOR'
        else:
            signal = 'NORMAL'
            urgency = 'NONE'
        
        return {
            'signal': signal,
            'urgency': urgency,
            'vpin': current_vpin,
            'z_score': z_score,
            'mean': mean_vpin,
            'std': std_vpin
        }

# ============================================================================
# 2. TOXICITY DETECTION WITH MULTI-SCALE WINDOWS
# ============================================================================

def calculate_vpin_multiscale(buy_volume, sell_volume, window_sizes=[10, 50, 100]):
    """
    Calculate VPIN at multiple time scales to capture different regimes.
    """
    
    n_periods = len(buy_volume)
    
    vpin_multiscale = {}
    
    for window in window_sizes:
        vpin_series = []
        
        for t in range(n_periods):
            # Calculate VPIN over rolling window
            start_idx = max(0, t - window + 1)
            end_idx = t + 1
            
            window_buys = buy_volume[start_idx:end_idx].sum()
            window_sells = sell_volume[start_idx:end_idx].sum()
            
            total = window_buys + window_sells
            if total > 0:
                imbalance = np.abs(window_buys - window_sells)
                vpin = imbalance / total
            else:
                vpin = 0
            
            vpin_series.append(vpin)
        
        vpin_multiscale[f'VPIN_{window}'] = np.array(vpin_series)
    
    return vpin_multiscale

# ============================================================================
# 3. STRESS DETECTION & CASCADE RISK
# ============================================================================

def detect_toxicity_cascade(vpin_series, alert_threshold=0.40, 
                           cascade_window=10, min_consecutive=3):
    """
    Detect potential liquidity cascade events.
    Alert when VPIN exceeds threshold persistently.
    """
    
    alerts = []
    in_cascade = False
    cascade_start = None
    consecutive_high = 0
    
    for t, vpin in enumerate(vpin_series):
        if vpin > alert_threshold:
            consecutive_high += 1
            
            if consecutive_high >= min_consecutive and not in_cascade:
                cascade_start = t - min_consecutive + 1
                in_cascade = True
                alerts.append({
                    'type': 'CASCADE_START',
                    'time_index': cascade_start,
                    'vpin': vpin
                })
        else:
            if in_cascade and consecutive_high >= min_consecutive:
                alerts.append({
                    'type': 'CASCADE_END',
                    'time_index': t,
                    'duration': t - cascade_start
                })
                in_cascade = False
            consecutive_high = 0
    
    return alerts

# ============================================================================
# 4. SIMULATION: MARKET STRUCTURE WITH TOXICITY
# ============================================================================

def simulate_intraday_trading(n_trades=2000, 
                            informed_periods=[]):
    """
    Generate realistic intraday trading with informed bursts.
    """
    
    np.random.seed(42)
    
    buy_volume = []
    sell_volume = []
    timestamps = []
    
    minutes_elapsed = 0
    
    for trade_idx in range(n_trades):
        # Determine if in informed period
        minute = trade_idx // 10  # ~10 trades per minute
        is_informed_minute = any(start <= minute < end 
                                for start, end in informed_periods)
        
        if is_informed_minute:
            # Informed trading: skewed flow
            buy_vol = np.random.poisson(lam=15)
            sell_vol = np.random.poisson(lam=5)
        else:
            # Normal trading: balanced
            buy_vol = np.random.poisson(lam=10)
            sell_vol = np.random.poisson(lam=10)
        
        buy_volume.append(buy_vol)
        sell_volume.append(sell_vol)
        minutes_elapsed += 1/10
        timestamps.append(timedelta(minutes=minutes_elapsed))
    
    return np.array(buy_volume), np.array(sell_volume), timestamps

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*70)
print("VPIN: Volume-Synchronized Probability of Informed Trading")
print("="*70)

# Scenario 1: Normal market with informed spike
print("\n--- SCENARIO 1: Normal Trading with Informed Spike ---")

informed_periods_s1 = [(60, 80), (150, 170)]  # Two 20-minute informed bursts
buys_s1, sells_s1, timestamps_s1 = simulate_intraday_trading(2000, informed_periods_s1)

# Calculate VPIN with 100-share buckets
vpin_calc = VPINCalculator(volume_bucket_size=100)

for buy, sell in zip(buys_s1, sells_s1):
    vpin_calc.add_trade(buy, side=1)
    vpin_calc.add_trade(sell, side=-1)

vpin_s1 = np.array(vpin_calc.vpin_history)

print(f"Total Buckets: {len(vpin_s1)}")
print(f"Average VPIN: {vpin_s1.mean():.4f}")
print(f"Max VPIN: {vpin_s1.max():.4f}")
print(f"Min VPIN: {vpin_s1.min():.4f}")
print(f"Std VPIN: {vpin_s1.std():.4f}")

# Detect cascades
cascades_s1 = detect_toxicity_cascade(vpin_s1, alert_threshold=0.35, min_consecutive=3)
print(f"\nCascade Events Detected: {len(cascades_s1) // 2}")  # Divide by 2 for start/end pairs
if cascades_s1:
    print(f"  First alert at bucket: {cascades_s1[0]['time_index']}")

# Scenario 2: Flash crash scenario
print("\n" + "="*70)
print("--- SCENARIO 2: Flash Crash Scenario ---")

# Simulate rapid sell-off
n_crisis_trades = 1000
buy_crisis = np.random.poisson(lam=5, size=n_crisis_trades)
sell_crisis = np.random.poisson(lam=30, size=n_crisis_trades)  # 6x more selling

vpin_calc_crisis = VPINCalculator(volume_bucket_size=100)

for buy, sell in zip(buy_crisis, sell_crisis):
    vpin_calc_crisis.add_trade(buy, side=1)
    vpin_calc_crisis.add_trade(sell, side=-1)

vpin_crisis = np.array(vpin_calc_crisis.vpin_history)

print(f"\nFlash Crash VPIN Statistics:")
print(f"  Average VPIN: {vpin_crisis.mean():.4f}")
print(f"  Max VPIN: {vpin_crisis.max():.4f}")
print(f"  VPIN > 0.50 (extreme): {np.sum(vpin_crisis > 0.50)} buckets")
print(f"  VPIN > 0.60 (crisis): {np.sum(vpin_crisis > 0.60)} buckets")

# Scenario 3: Multi-scale VPIN analysis
print("\n" + "="*70)
print("--- SCENARIO 3: Multi-Scale VPIN Analysis ---")

vpin_multi = calculate_vpin_multiscale(buys_s1, sells_s1, window_sizes=[10, 50, 100])

for scale_name, vpin_series in vpin_multi.items():
    print(f"\n{scale_name}:")
    print(f"  Mean: {vpin_series.mean():.4f}")
    print(f"  Max: {vpin_series.max():.4f}")
    print(f"  Correlation with 100-bucket VPIN: {np.corrcoef(vpin_series, vpin_s1)[0,1]:.4f}")

# Scenario 4: Real-time alert simulation
print("\n" + "="*70)
print("--- SCENARIO 4: Real-Time Alert System ---")

alert_count_low = 0
alert_count_moderate = 0
alert_count_high = 0

for t, vpin in enumerate(vpin_s1):
    if vpin > 0.60:
        alert_count_high += 1
    elif vpin > 0.40:
        alert_count_moderate += 1
    elif vpin > 0.25:
        alert_count_low += 1

print(f"\nAlert Distribution:")
print(f"  Low toxicity (0.25 < VPIN < 0.40): {alert_count_low} buckets")
print(f"  Moderate toxicity (0.40 < VPIN < 0.60): {alert_count_moderate} buckets")
print(f"  High toxicity (VPIN > 0.60): {alert_count_high} buckets")

print(f"\nSample Current VPIN Signal:")
signal_sample = vpin_calc.get_vpin_signal(lookback_window=30)
for key, val in signal_sample.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.4f}")
    else:
        print(f"  {key}: {val}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Plot 1: VPIN time series with alerts
ax1 = plt.subplot(2, 3, 1)
colors = ['green' if x < 0.30 else 'orange' if x < 0.50 else 'red' for x in vpin_s1]
ax1.bar(range(len(vpin_s1)), vpin_s1, color=colors, alpha=0.7, width=1)
ax1.axhline(y=0.30, color='orange', linestyle='--', linewidth=1, label='Caution')
ax1.axhline(y=0.50, color='red', linestyle='--', linewidth=1, label='Alert')
ax1.set_xlabel('Volume Bucket')
ax1.set_ylabel('VPIN')
ax1.set_title('Scenario 1: VPIN with Alerts\n(Green=Normal, Orange=Caution, Red=Alert)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Flash crash VPIN
ax2 = plt.subplot(2, 3, 2)
ax2.plot(vpin_crisis, linewidth=1.5, color='red', label='VPIN (Crisis)')
ax2.axhline(y=0.50, color='black', linestyle='--', linewidth=1, label='Crisis Threshold')
ax2.fill_between(range(len(vpin_crisis)), vpin_crisis, alpha=0.2, color='red')
ax2.set_xlabel('Volume Bucket')
ax2.set_ylabel('VPIN')
ax2.set_title('Scenario 2: Flash Crash Toxicity\n(Extreme Sell-Off)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Multi-scale VPIN
ax3 = plt.subplot(2, 3, 3)
ax3.plot(vpin_multi['VPIN_10'], label='VPIN (10-bucket)', alpha=0.7, linewidth=0.8)
ax3.plot(vpin_multi['VPIN_50'], label='VPIN (50-bucket)', alpha=0.7, linewidth=1)
ax3.plot(vpin_multi['VPIN_100'], label='VPIN (100-bucket)', alpha=0.9, linewidth=1.5)
ax3.set_xlabel('Time Index')
ax3.set_ylabel('VPIN')
ax3.set_title('Multi-Scale VPIN Analysis')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Order flow composition
ax4 = plt.subplot(2, 3, 4)
bucket_data = np.array(vpin_calc.bucket_history)
ax4.bar(range(len(bucket_data)), bucket_data[:, 0], alpha=0.6, label='Buys', color='blue', width=1)
ax4.bar(range(len(bucket_data)), -bucket_data[:, 1], alpha=0.6, label='Sells', color='red', width=1)
ax4.axhline(y=0, color='black', linewidth=0.8)
ax4.set_xlabel('Volume Bucket')
ax4.set_ylabel('Volume')
ax4.set_title('Order Flow Composition')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: Cumulative toxicity
ax5 = plt.subplot(2, 3, 5)
cumul_toxicity = np.cumsum(vpin_s1)
ax5.plot(cumul_toxicity, linewidth=1.5, color='purple')
ax5.fill_between(range(len(cumul_toxicity)), cumul_toxicity, alpha=0.2, color='purple')
ax5.set_xlabel('Volume Bucket')
ax5.set_ylabel('Cumulative Toxicity')
ax5.set_title('Cumulative Toxicity Index')
ax5.grid(True, alpha=0.3)

# Plot 6: VPIN distribution
ax6 = plt.subplot(2, 3, 6)
ax6.hist(vpin_s1, bins=40, alpha=0.7, color='blue', edgecolor='black', label='Normal')
ax6.hist(vpin_crisis, bins=40, alpha=0.7, color='red', edgecolor='black', label='Crisis')
ax6.axvline(x=0.30, color='orange', linestyle='--', linewidth=1.5, label='Thresholds')
ax6.axvline(x=0.50, color='darkred', linestyle='--', linewidth=1.5)
ax6.set_xlabel('VPIN')
ax6.set_ylabel('Frequency')
ax6.set_title('VPIN Distribution: Normal vs Crisis')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('vpin_analysis.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved: vpin_analysis.png")

plt.show()

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. VPIN INTERPRETATION:
   - VPIN < 0.25: Balanced flow, low information asymmetry
   - VPIN 0.25-0.40: Elevated, caution warranted
   - VPIN 0.40-0.60: High toxicity, defensive spreads
   - VPIN > 0.60: Extreme, consider halting/withdrawal

2. MARKET MAKER BEHAVIOR:
   - VPIN correlates with spread widening (positive relationship)
   - Depth inversely correlates with VPIN
   - Algorithmic withdrawal cascades when VPIN > 0.50

3. PREDICTIVE VALUE:
   - VPIN predicts future volatility (persistence)
   - High VPIN → likely to remain high (3-5 buckets)
   - VPIN spikes often precede price movements (leading indicator)

4. LIMITATIONS:
   - Cannot distinguish informed trading from panic/correlation
   - Misses hidden orders and order-splitting strategies
   - Sensitive to bucket size choice (volume aggregation)
   - May trigger false alarms during liquidity shocks

5. PRACTICAL IMPLEMENTATION:
   - Update VPIN every 100-1000 shares (1-10 seconds typically)
   - Combine with other signals (order size, patterns)
   - Use adaptive thresholds based on regime/volatility
   - Implement tiered response system (caution → reduce → halt)
""")
