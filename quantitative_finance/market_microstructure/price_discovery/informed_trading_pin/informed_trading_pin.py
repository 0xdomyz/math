"""
PIN Estimation: Probability of Informed Trading Detection & Measurement
Identifies informed trading periods and estimates information asymmetry.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln
import seaborn as sns

# ============================================================================
# 1. PIN MODEL: LIKELIHOOD & ESTIMATION
# ============================================================================

class PINModel:
    """
    Estimate PIN (Probability of Informed Trading) using Easley et al method.
    """
    
    def __init__(self, buys, sells, period_label='Day'):
        """
        Parameters:
        - buys: Array of daily buy volumes
        - sells: Array of daily sell volumes
        - period_label: Label for time period (Day, Hour, etc.)
        """
        self.buys = np.array(buys)
        self.sells = np.array(sells)
        self.period_label = period_label
        self.n_periods = len(buys)
        
        # Parameters to estimate
        self.params = {
            'mu': None,      # Probability of information event (good news)
            'delta': None,   # Probability informed trader active
            'alpha_b': None, # Informed buy intensity
            'alpha_s': None, # Informed sell intensity
            'beta': None     # Uninformed order intensity
        }
        
        self.pin = None
    
    def log_likelihood(self, params_dict):
        """
        Calculate negative log-likelihood (for minimization).
        params: [mu, delta, alpha_b, alpha_s, beta]
        """
        mu, delta, alpha_b, alpha_s, beta = params_dict
        
        # Constrain parameters
        if mu < 0 or mu > 1 or delta < 0 or delta > 1 or alpha_b < 0 or alpha_s < 0 or beta < 0:
            return 1e10  # Large penalty for invalid params
        
        ll = 0
        
        for b, s in zip(self.buys, self.sells):
            # Three scenarios contribute to observed (b, s):
            # 1. Good news event (prob mu), informed buys (alpha_b), uninformed random (beta)
            # 2. Bad news event (prob 1-mu), informed sells (alpha_s), uninformed random (beta)
            # 3. No information (prob 1-delta), both uninformed random (beta)
            
            # Poisson log-PMF: ln P(X=k) = k*ln(λ) - λ - ln(k!)
            
            # Scenario 1: Good news, informed buys
            ll_good = (mu * delta * 
                      (b * np.log(alpha_b + 1e-10) - (alpha_b) +
                       s * np.log(beta + 1e-10) - beta))
            
            # Scenario 2: Bad news, informed sells
            ll_bad = ((1 - mu) * delta *
                     (b * np.log(beta + 1e-10) - beta +
                      s * np.log(alpha_s + 1e-10) - (alpha_s)))
            
            # Scenario 3: No information, both uninformed
            ll_none = ((1 - delta) *
                      (b * np.log(beta + 1e-10) - beta +
                       s * np.log(beta + 1e-10) - beta))
            
            # Log-sum-exp trick for numerical stability
            max_ll = max(ll_good, ll_bad, ll_none)
            ll_combined = max_ll + np.log(np.exp(ll_good - max_ll) + 
                                         np.exp(ll_bad - max_ll) + 
                                         np.exp(ll_none - max_ll))
            
            ll += ll_combined
        
        return -ll  # Negative for minimization
    
    def estimate_pin(self, initial_guess=None):
        """
        Estimate PIN parameters using MLE.
        """
        
        if initial_guess is None:
            # Data-driven initial guess
            avg_buy = self.buys.mean()
            avg_sell = self.sells.mean()
            
            initial_guess = [
                0.5,        # mu: equal prob good/bad news
                0.2,        # delta: 20% informed
                avg_buy * 1.2,   # alpha_b: slightly higher than average
                avg_sell * 1.2,  # alpha_s: slightly higher than average
                min(avg_buy, avg_sell) * 0.8  # beta: lower than average (noise)
            ]
        
        # Optimize
        bounds = [(0, 1), (0, 1), (0, 1000), (0, 1000), (0, 1000)]
        
        result = minimize(
            lambda p: self.log_likelihood(p),
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        params_opt = result.x
        
        self.params = {
            'mu': params_opt[0],
            'delta': params_opt[1],
            'alpha_b': params_opt[2],
            'alpha_s': params_opt[3],
            'beta': params_opt[4]
        }
        
        # Calculate PIN
        self.pin = self.params['delta'] * (self.params['alpha_s'] / 
                 (self.params['alpha_b'] + self.params['alpha_s'] + 1e-10))
        
        return self.pin

# ============================================================================
# 2. VPIN: VOLUME-SYNCHRONIZED PIN (HIGH-FREQUENCY VERSION)
# ============================================================================

def calculate_vpin(order_flow, volume_bars=100):
    """
    Calculate Volume-Synchronized PIN (VPIN).
    Real-time measure of order flow toxicity.
    
    order_flow: Sequence of +1 (buy) or -1 (sell)
    volume_bars: Number of volume buckets to use
    """
    
    n_trades = len(order_flow)
    bucket_size = n_trades // volume_bars
    
    vpin_series = []
    
    for i in range(volume_bars - 1):
        start_idx = i * bucket_size
        end_idx = (i + 1) * bucket_size
        
        bucket_flow = order_flow[start_idx:end_idx]
        buys = np.sum(bucket_flow == 1)
        sells = np.sum(bucket_flow == -1)
        
        # VPIN = |B - S| / (B + S)
        total = buys + sells
        if total > 0:
            vpin = np.abs(buys - sells) / total
        else:
            vpin = 0
        
        vpin_series.append(vpin)
    
    return np.array(vpin_series)

# ============================================================================
# 3. ORDER FLOW TOXICITY DETECTION
# ============================================================================

def detect_order_toxicity(buy_volume, sell_volume, n_lookback=20):
    """
    Detect periods of elevated order flow toxicity.
    High toxicity = potential informed trading.
    """
    
    n_periods = len(buy_volume)
    toxicity_scores = []
    
    for t in range(n_periods):
        # Order imbalance in current period
        total_vol = buy_volume[t] + sell_volume[t]
        if total_vol > 0:
            imbalance = np.abs(buy_volume[t] - sell_volume[t]) / total_vol
        else:
            imbalance = 0
        
        # Historical context
        if t >= n_lookback:
            window_buy = buy_volume[t-n_lookback:t]
            window_sell = sell_volume[t-n_lookback:t]
            window_total = window_buy.sum() + window_sell.sum()
            
            if window_total > 0:
                avg_imbalance = np.abs(window_buy.sum() - window_sell.sum()) / window_total
            else:
                avg_imbalance = 0
            
            # Toxicity = deviation from normal
            toxicity = imbalance - avg_imbalance
        else:
            toxicity = imbalance
        
        toxicity_scores.append(toxicity)
    
    return np.array(toxicity_scores)

# ============================================================================
# 4. SIMULATION: INFORMED vs UNINFORMED TRADING PERIODS
# ============================================================================

def generate_synthetic_order_flow(n_days=60, periods_informed=[]):
    """
    Generate synthetic order flow with informed trading episodes.
    
    periods_informed: List of (start_day, end_day) tuples with high informed trading
    """
    
    np.random.seed(42)
    
    buy_volume = []
    sell_volume = []
    
    for day in range(n_days):
        # Check if this day has informed trading
        is_informed_day = any(start <= day < end for start, end in periods_informed)
        
        if is_informed_day:
            # Informed trading: skewed order flow
            buy_vol = np.random.poisson(lam=150)  # Higher buys
            sell_vol = np.random.poisson(lam=50)   # Lower sells
        else:
            # Uninformed (noise) trading: balanced
            buy_vol = np.random.poisson(lam=100)
            sell_vol = np.random.poisson(lam=100)
        
        buy_volume.append(buy_vol)
        sell_volume.append(sell_vol)
    
    return np.array(buy_volume), np.array(sell_volume)

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*70)
print("PIN MODEL: Probability of Informed Trading Detection")
print("="*70)

# Scenario 1: Baseline PIN estimation
print("\n--- SCENARIO 1: Normal Market Conditions (Mostly Uninformed) ---")

# Generate order flow: 2 informed trading periods
periods_informed_scenario1 = [(10, 15), (40, 45)]
buys_s1, sells_s1 = generate_synthetic_order_flow(60, periods_informed_scenario1)

# Estimate PIN
pin_model_s1 = PINModel(buys_s1, sells_s1)
pin_s1 = pin_model_s1.estimate_pin()

print(f"\nEstimated PIN: {pin_s1:.4f} ({pin_s1*100:.2f}%)")
print(f"Interpretation: {pin_s1*100:.2f}% probability random trader is informed")

print(f"\nParameter Estimates:")
for param_name, param_value in pin_model_s1.params.items():
    print(f"  {param_name:10s}: {param_value:8.4f}")

# Calculate daily order imbalance
order_imb_s1 = (buys_s1 - sells_s1) / (buys_s1 + sells_s1 + 1e-10)

print(f"\nOrder Flow Statistics:")
print(f"  Average Buy Volume: {buys_s1.mean():.0f}")
print(f"  Average Sell Volume: {sells_s1.mean():.0f}")
print(f"  Average Order Imbalance: {order_imb_s1.mean():.4f}")
print(f"  Imbalance Std Dev: {order_imb_s1.std():.4f}")

# Scenario 2: High information asymmetry
print("\n" + "="*70)
print("--- SCENARIO 2: High Information Asymmetry Period ---")

periods_informed_scenario2 = [(1, 25)]  # Persistent informed trading
buys_s2, sells_s2 = generate_synthetic_order_flow(60, periods_informed_scenario2)

pin_model_s2 = PINModel(buys_s2, sells_s2)
pin_s2 = pin_model_s2.estimate_pin()

print(f"\nEstimated PIN: {pin_s2:.4f} ({pin_s2*100:.2f}%)")
print(f"Interpretation: {pin_s2*100:.2f}% probability random trader is informed")

print(f"\nParameter Estimates:")
for param_name, param_value in pin_model_s2.params.items():
    print(f"  {param_name:10s}: {param_value:8.4f}")

# Scenario 3: VPIN Analysis
print("\n" + "="*70)
print("--- SCENARIO 3: Volume-Synchronized PIN (VPIN) ---")

# Generate intraday order flow with burst of informed trades
np.random.seed(42)
n_trades = 2000
order_flow_intraday = np.random.choice([-1, 1], size=n_trades, p=[0.48, 0.52])

# Add informed trading burst (hour 3)
informed_burst = np.random.choice([-1, 1], size=200, p=[0.2, 0.8])  # Heavy buying
order_flow_intraday[600:800] = informed_burst

vpin_series = calculate_vpin(order_flow_intraday, volume_bars=50)

print(f"\nVPIN Statistics Across Day:")
print(f"  Average VPIN: {vpin_series.mean():.4f}")
print(f"  Max VPIN: {vpin_series.max():.4f}")
print(f"  Min VPIN: {vpin_series.min():.4f}")
print(f"  Std Dev VPIN: {vpin_series.std():.4f}")

# Identify high toxicity periods
high_toxicity_threshold = vpin_series.mean() + 1.5 * vpin_series.std()
high_toxicity_periods = np.where(vpin_series > high_toxicity_threshold)[0]

print(f"\nHigh Toxicity Detection:")
print(f"  Threshold: {high_toxicity_threshold:.4f}")
print(f"  Periods with high toxicity: {len(high_toxicity_periods)} / {len(vpin_series)}")

# Scenario 4: Order Toxicity Time Series
print("\n" + "="*70)
print("--- SCENARIO 4: Toxicity Evolution Analysis ---")

toxicity_s1 = detect_order_toxicity(buys_s1, sells_s1)

print(f"\nToxicity Statistics (Normal Conditions):")
print(f"  Average Toxicity: {toxicity_s1.mean():.4f}")
print(f"  Max Toxicity: {toxicity_s1.max():.4f}")
print(f"  Detected periods with high toxicity: {np.sum(toxicity_s1 > toxicity_s1.mean() + toxicity_s1.std())}")

# Compare PIN and toxicity correlation
print(f"\nComparison: PIN vs Order Imbalance")
print(f"  PIN estimate: {pin_s1:.4f}")
print(f"  Max daily order imbalance: {np.abs(order_imb_s1).max():.4f}")
print(f"  Correlation(Imbalance, Toxicity): {np.corrcoef(np.abs(order_imb_s1), toxicity_s1)[0,1]:.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Plot 1: Order flow - Normal conditions
ax1 = plt.subplot(2, 3, 1)
ax1.bar(range(len(buys_s1)), buys_s1, alpha=0.6, label='Buy Volume', color='blue', width=0.8)
ax1.bar(range(len(sells_s1)), -sells_s1, alpha=0.6, label='Sell Volume', color='red', width=0.8)
ax1.axhspan(10, 150, alpha=0.05, color='orange')  # Informed period 1
ax1.axhspan(-150, -50, alpha=0.05, color='orange')
ax1.axvspan(40, 45, alpha=0.1, color='orange')
ax1.set_xlabel('Day')
ax1.set_ylabel('Volume')
ax1.set_title('Scenario 1: Order Flow (Normal, Some Informed Bursts)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Order imbalance
ax2 = plt.subplot(2, 3, 2)
ax2.plot(order_imb_s1, linewidth=1.5, label='Order Imbalance')
ax2.fill_between(range(len(order_imb_s1)), order_imb_s1, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax2.set_xlabel('Day')
ax2.set_ylabel('Order Imbalance (B-S)/(B+S)')
ax2.set_title('Order Flow Direction')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: High asymmetry scenario
ax3 = plt.subplot(2, 3, 3)
ax3.bar(range(len(buys_s2)), buys_s2, alpha=0.6, label='Buy Volume', color='blue', width=0.8)
ax3.bar(range(len(sells_s2)), -sells_s2, alpha=0.6, label='Sell Volume', color='red', width=0.8)
ax3.axvspan(0, 25, alpha=0.15, color='orange', label='Informed Period')
ax3.set_xlabel('Day')
ax3.set_ylabel('Volume')
ax3.set_title('Scenario 2: Persistent Informed Trading')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: VPIN series
ax4 = plt.subplot(2, 3, 4)
ax4.plot(vpin_series, linewidth=1.5, color='purple', label='VPIN')
ax4.axhline(y=high_toxicity_threshold, color='red', linestyle='--', linewidth=1, 
            label=f'Alert Threshold')
ax4.fill_between(range(len(vpin_series)), vpin_series, alpha=0.2, color='purple')
ax4.axvspan(12, 20, alpha=0.1, color='orange', label='Informed Burst')
ax4.set_xlabel('Volume Bucket')
ax4.set_ylabel('VPIN')
ax4.set_title('Volume-Synchronized PIN (Intraday)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: Toxicity time series
ax5 = plt.subplot(2, 3, 5)
ax5.plot(toxicity_s1, linewidth=1, label='Toxicity Score', alpha=0.7)
mean_tox = toxicity_s1.mean()
std_tox = toxicity_s1.std()
ax5.axhline(y=mean_tox, color='green', linestyle='--', linewidth=1, label='Mean')
ax5.axhline(y=mean_tox + std_tox, color='red', linestyle='--', linewidth=1, label='Mean + 1σ')
ax5.fill_between(range(len(toxicity_s1)), mean_tox + std_tox, 1, alpha=0.15, color='red')
ax5.set_xlabel('Day')
ax5.set_ylabel('Toxicity')
ax5.set_title('Order Flow Toxicity Detection')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: PIN comparison
ax6 = plt.subplot(2, 3, 6)
scenarios = ['Normal\n(Some Informed)', 'High\nAsymmetry']
pins = [pin_s1, pin_s2]
colors = ['green', 'red']
bars = ax6.bar(scenarios, pins, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax6.set_ylabel('Estimated PIN')
ax6.set_title('PIN: Probability of Informed Trading')
ax6.set_ylim([0, 0.5])
for i, (bar, pin) in enumerate(zip(bars, pins)):
    ax6.text(i, pin + 0.01, f'{pin:.3f}', ha='center', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pin_analysis.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved: pin_analysis.png")

plt.show()

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. PIN INTERPRETATION GUIDE:
   - PIN < 0.10: Minimal information asymmetry (tight spreads justified)
   - PIN 0.10-0.25: Moderate asymmetry (normal market conditions)
   - PIN 0.25-0.40: Elevated asymmetry (spreads widen, depth decreases)
   - PIN > 0.40: Severe information asymmetry (liquidity evaporates)

2. ESTIMATION CHALLENGES:
   - Requires detailed microstructure data (high-frequency timestamps)
   - ML estimation sensitive to initialization
   - Parameters not directly identifiable; joint likelihood drives estimation
   - Small sample bias in short periods

3. PRACTICAL APPLICATIONS:
   - Liquidity providers use PIN to adjust rebates/fees
   - Market makers use VPIN for real-time risk management
   - Regulators monitor for unusual trading patterns
   - Researchers decompose order flow informativeness

4. RELATIONSHIP TO SPREADS:
   - Observed spread ≈ Order processing + Inventory + 2×PIN×Value_Spread
   - PIN increase → Immediate spread widening
   - Liquidity provider losses = PIN × Adverse Selection Cost per trade

5. LIMITATIONS & EXTENSIONS:
   - Base PIN assumes exogenous information events (discrete good/bad news)
   - Real information is continuous and partially revealed
   - Extensions: VPIN for real-time monitoring, arrival PIN for event windows
   - Cannot distinguish sophisticated information hiding from noise
""")
