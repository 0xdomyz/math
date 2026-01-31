"""
Price Discovery Process: Information Incorporation Dynamics
Models and measures how quickly markets incorporate information.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import seaborn as sns

# ============================================================================
# 1. PRICE DISCOVERY SIMULATION: Order Flow → Price Adjustment
# ============================================================================

def simulate_price_discovery(n_steps=1000, fundamental_value=100, 
                            information_revelation_rate=0.15,
                            noise_std=0.5, informed_fraction=0.3):
    """
    Simulate price discovery process where price converges to fundamental value
    through order flow incorporating both information and noise.
    
    Parameters:
    - n_steps: Number of time periods
    - fundamental_value: True intrinsic value of asset
    - information_revelation_rate: λ - speed of adjustment (0-1)
    - noise_std: Standard deviation of order flow noise
    - informed_fraction: Proportion of informed vs uninformed traders
    """
    
    prices = np.zeros(n_steps)
    prices[0] = fundamental_value - 5  # Start with 5% undervaluation
    
    order_flow = np.zeros(n_steps)
    information_component = np.zeros(n_steps)
    
    for t in range(1, n_steps):
        # Order flow = informed component + noise
        informed_flow = informed_fraction * (fundamental_value - prices[t-1])
        noise = np.random.normal(0, noise_std)
        order_flow[t] = informed_flow + noise
        
        # Price adjustment: Partial adjustment toward fundamental value
        # Price_t = Price_t-1 + λ * (OF_t / market_depth)
        market_depth = 100  # Liquidity parameter
        price_impact = information_revelation_rate * order_flow[t] / market_depth
        
        prices[t] = prices[t-1] + price_impact
        information_component[t] = prices[t] - prices[t-1]
    
    return prices, order_flow, information_component

# ============================================================================
# 2. MEASURE PRICE DISCOVERY: Half-Life & Information Content
# ============================================================================

def measure_discovery_speed(prices, fundamental_value):
    """
    Calculate half-life of adjustment to fundamental value.
    Half-life = time for price to close 50% of gap to fundamental.
    """
    
    initial_gap = np.abs(prices[0] - fundamental_value)
    half_target = fundamental_value - initial_gap / 2
    
    # Find time when price crosses halfway point
    if prices[0] < fundamental_value:
        crossing_idx = np.where(prices >= half_target)[0]
    else:
        crossing_idx = np.where(prices <= half_target)[0]
    
    if len(crossing_idx) == 0:
        half_life = np.nan
    else:
        half_life = crossing_idx[0]
    
    return half_life

def calculate_information_share(returns_y, returns_x):
    """
    Calculate information share of security Y based on X using 
    Hasbrouck's method: IS_Y = Cov(Y,X) / (Cov(X,X) + Cov(Y,Y))
    
    Interpretation: Fraction of price discovery occurring in Y vs X
    """
    
    cov_yx = np.cov(returns_y, returns_x)[0, 1]
    cov_xx = np.var(returns_x)
    cov_yy = np.var(returns_y)
    
    if cov_xx + cov_yy == 0:
        return 0
    
    info_share = cov_yx / (cov_xx + cov_yy)
    return info_share

def estimate_adjustment_speed(price_gaps, fundamental_values):
    """
    Estimate exponential adjustment speed: Gap_t = Gap_0 * exp(-λ*t)
    λ = adjustment speed parameter (higher = faster discovery)
    """
    
    time_periods = np.arange(len(price_gaps))
    
    # Remove zero gaps to avoid log issues
    valid_idx = price_gaps != 0
    time_periods = time_periods[valid_idx]
    log_gaps = np.log(np.abs(price_gaps[valid_idx]))
    
    # Fit exponential decay: ln|Gap_t| = ln|Gap_0| - λ*t
    slope, intercept, r_value, p_value, std_err = linregress(time_periods, log_gaps)
    
    lambda_param = -slope  # Adjustment speed
    half_life = np.log(2) / lambda_param if lambda_param > 0 else np.inf
    
    return lambda_param, half_life, r_value**2

# ============================================================================
# 3. LEAD-LAG ANALYSIS: Information Flow Direction Across Venues
# ============================================================================

def lead_lag_analysis(price_series_primary, price_series_secondary, max_lags=20):
    """
    Analyze lead-lag relationship between two venues to identify where
    price discovery originates. If primary leads secondary, primary
    market discovers prices first.
    """
    
    # Calculate returns
    returns_primary = np.diff(price_series_primary)
    returns_secondary = np.diff(price_series_secondary)
    
    # Lag correlations
    lag_correlations = []
    for lag in range(-max_lags, max_lags + 1):
        if lag < 0:
            # Primary leads (future returns correlate with past secondary)
            corr = np.corrcoef(returns_primary[:lag], returns_secondary[-lag:])[0, 1]
        elif lag > 0:
            # Secondary leads (future returns correlate with past primary)
            corr = np.corrcoef(returns_primary[lag:], returns_secondary[:-lag])[0, 1]
        else:
            # Contemporaneous correlation
            corr = np.corrcoef(returns_primary, returns_secondary)[0, 1]
        lag_correlations.append(corr)
    
    # Peak correlation lag indicates lead
    max_corr_lag = np.argmax(np.abs(lag_correlations)) - max_lags
    
    return np.array(lag_correlations), max_corr_lag

# ============================================================================
# 4. INFORMATION ASYMMETRY DETECTION: PIN Estimation (Simplified)
# ============================================================================

def estimate_pin_simple(buy_orders, sell_orders):
    """
    Simplified PIN (Probability of Informed Trading) estimation.
    PIN = (probability informed is active) * (informed trades against you)
    
    High PIN = more informed trading = wider spreads by market makers
    """
    
    total_orders = buy_orders + sell_orders
    buy_ratio = buy_orders / total_orders if total_orders > 0 else 0.5
    
    # If buy_ratio far from 0.5, suggests informed trading direction
    # PIN proxy: deviation from 50-50 split
    pin = np.abs(buy_ratio - 0.5)
    
    return pin, buy_ratio

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*70)
print("PRICE DISCOVERY PROCESS: Information Incorporation Analysis")
print("="*70)

# Scenario 1: Price discovery with different revelation rates
np.random.seed(42)

scenarios = {
    'Fast Market (HFT era)': {'revelation_rate': 0.35, 'noise_std': 0.3},
    'Medium Market': {'revelation_rate': 0.15, 'noise_std': 0.5},
    'Slow Market': {'revelation_rate': 0.05, 'noise_std': 0.8}
}

discovery_stats = []
prices_all = {}

for scenario_name, params in scenarios.items():
    prices, order_flow, info_comp = simulate_price_discovery(
        n_steps=500,
        fundamental_value=100,
        information_revelation_rate=params['revelation_rate'],
        noise_std=params['noise_std'],
        informed_fraction=0.25
    )
    
    prices_all[scenario_name] = prices
    
    # Calculate metrics
    initial_gap = np.abs(prices[0] - 100)
    final_gap = np.abs(prices[-1] - 100)
    gap_reduction = (initial_gap - final_gap) / initial_gap * 100
    
    half_life = measure_discovery_speed(prices, 100)
    
    price_gaps = 100 - prices
    lambda_est, half_life_fit, r2 = estimate_adjustment_speed(price_gaps, np.full_like(prices, 100))
    
    discovery_stats.append({
        'Scenario': scenario_name,
        'Initial Gap': f'${initial_gap:.2f}',
        'Final Gap': f'${final_gap:.2f}',
        'Gap Reduction %': f'{gap_reduction:.1f}%',
        'Half-Life (obs)': f'{half_life:.0f} periods',
        'Half-Life (fit)': f'{half_life_fit:.0f} periods',
        'Adjustment Speed λ': f'{lambda_est:.4f}',
        'Model R²': f'{r2:.4f}'
    })

stats_df = pd.DataFrame(discovery_stats)
print("\n--- SCENARIO COMPARISON: Price Discovery Speed ---")
print(stats_df.to_string(index=False))

# Scenario 2: Multi-venue price discovery simulation
print("\n" + "="*70)
print("SCENARIO 2: Multi-Venue Price Discovery (Information Fragmentation)")
print("="*70)

n_venues = 3
n_periods = 300

# Primary venue discovers information first
prices_venues = np.zeros((n_periods, n_venues))
prices_venues[0] = [99.5, 99.3, 99.1]  # Slight variation in initial prices

fundamental = 100
discovery_rates = [0.25, 0.12, 0.08]  # Primary fastest, secondary slower

for t in range(1, n_periods):
    for v in range(n_venues):
        gap_to_fundamental = fundamental - prices_venues[t-1, v]
        adjustment = discovery_rates[v] * gap_to_fundamental
        prices_venues[t, v] = prices_venues[t-1, v] + adjustment
        
        # Add venue-specific noise
        prices_venues[t, v] += np.random.normal(0, 0.1)

# Calculate information share (lead-lag)
print("\n--- Lead-Lag Analysis: Information Flow ---")
lead_lag_results = lead_lag_analysis(prices_venues[:, 0], prices_venues[:, 1], max_lags=10)
lags = np.arange(-10, 11)
max_lag = lead_lag_results[1]
print(f"Primary venue leads Secondary venue by: {max_lag} periods")
print(f"Maximum correlation: {np.max(np.abs(lead_lag_results[0])):.4f} at lag {max_lag}")

# Calculate information share (simplified)
returns_primary = np.diff(prices_venues[:, 0])
returns_secondary = np.diff(prices_venues[:, 1])
info_share = calculate_information_share(returns_secondary, returns_primary)
print(f"Information Share (Secondary): {info_share:.2%}")
print(f"Information Share (Primary): {(1 - info_share):.2%} (implied)")

# Scenario 3: Informed vs Uninformed Trading
print("\n" + "="*70)
print("SCENARIO 3: Order Flow Direction & PIN Detection")
print("="*70)

# Simulate order flows with information advantage
n_days = 100
buy_orders = np.random.poisson(lam=50, size=n_days)
sell_orders = np.random.poisson(lam=50, size=n_days)

# Informed trading on day 50 (forward-looking)
buy_orders[50:70] += np.random.poisson(lam=20, size=20)

pin_estimates = []
buy_ratios = []

for day in range(n_days):
    pin, buy_ratio = estimate_pin_simple(buy_orders[day], sell_orders[day])
    pin_estimates.append(pin)
    buy_ratios.append(buy_ratio)

pin_df = pd.DataFrame({
    'Day': np.arange(n_days),
    'Buy Orders': buy_orders,
    'Sell Orders': sell_orders,
    'Buy Ratio': buy_ratios,
    'PIN': pin_estimates
})

# Detect informed trading period
elevated_pin = np.array(pin_estimates) > np.mean(pin_estimates) + np.std(pin_estimates)
informed_periods = np.where(elevated_pin)[0]

print("\n--- Order Flow Summary ---")
print(f"Average Buy Ratio: {np.mean(buy_ratios):.4f} (0.5 = balanced)")
print(f"Average PIN: {np.mean(pin_estimates):.4f}")
print(f"Max PIN (detect informed trading): {np.max(pin_estimates):.4f}")
print(f"Elevated PIN periods detected: {len(informed_periods)} out of {n_days}")
if len(informed_periods) > 0:
    print(f"Informed trading detected on Days: {informed_periods[:10]} (showing first 10)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Plot 1: Price discovery speed comparison
ax1 = plt.subplot(2, 3, 1)
for scenario_name, prices in prices_all.items():
    ax1.plot(prices, linewidth=1.5, label=scenario_name, alpha=0.8)
ax1.axhline(y=100, color='black', linestyle='--', linewidth=2, label='Fundamental Value')
ax1.set_xlabel('Time Period')
ax1.set_ylabel('Price')
ax1.set_title('Price Discovery: Convergence Speed Scenarios')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative gap closing
ax2 = plt.subplot(2, 3, 2)
for scenario_name, prices in prices_all.items():
    gaps = np.abs(prices - 100)
    cum_gap_closed = (gaps[0] - gaps) / gaps[0] * 100
    ax2.plot(cum_gap_closed, linewidth=1.5, label=scenario_name)
ax2.set_xlabel('Time Period')
ax2.set_ylabel('Gap Closed (%)')
ax2.set_title('Information Incorporation Rate')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Lead-lag correlation
ax3 = plt.subplot(2, 3, 3)
lags = np.arange(-10, 11)
ax3.bar(lags, lead_lag_results[0], color=['red' if x < 0 else 'blue' for x in lags], alpha=0.7)
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.set_xlabel('Lag (periods, <0 = primary leads)')
ax3.set_ylabel('Correlation')
ax3.set_title('Lead-Lag: Multi-Venue Information Flow')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Multi-venue prices
ax4 = plt.subplot(2, 3, 4)
ax4.plot(prices_venues[:, 0], label='Primary Venue', linewidth=1.5)
ax4.plot(prices_venues[:, 1], label='Secondary Venue', linewidth=1.5)
ax4.plot(prices_venues[:, 2], label='Tertiary Venue', linewidth=1.5)
ax4.axhline(y=100, color='black', linestyle='--', linewidth=1.5, label='Fundamental')
ax4.set_xlabel('Time Period')
ax4.set_ylabel('Price')
ax4.set_title('Multi-Venue Price Discovery')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: PIN over time (informed trading detection)
ax5 = plt.subplot(2, 3, 5)
ax5.plot(pin_estimates, label='PIN', linewidth=1, alpha=0.8)
mean_pin = np.mean(pin_estimates)
ax5.axhline(y=mean_pin, color='green', linestyle='--', linewidth=1, label='Mean PIN')
ax5.axhline(y=mean_pin + np.std(pin_estimates), color='red', linestyle='--', linewidth=1, 
            label='Mean + 1σ (Alert)')
ax5.fill_between(range(len(pin_estimates)), mean_pin + np.std(pin_estimates), 0.5, 
                  alpha=0.2, color='red')
ax5.axvspan(50, 70, alpha=0.1, color='orange', label='Informed period')
ax5.set_xlabel('Day')
ax5.set_ylabel('PIN (Probability of Informed Trading)')
ax5.set_title('Order Flow Asymmetry Detection')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Buy ratio time series
ax6 = plt.subplot(2, 3, 6)
ax6.plot(buy_ratios, label='Buy Ratio', linewidth=1, color='blue', alpha=0.7)
ax6.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Neutral (50%)')
ax6.fill_between(range(len(buy_ratios)), 0.45, 0.55, alpha=0.1, color='gray')
ax6.axvspan(50, 70, alpha=0.1, color='orange')
ax6.set_xlabel('Day')
ax6.set_ylabel('Buy Ratio')
ax6.set_title('Order Flow Direction (Informed Trading Indicator)')
ax6.set_ylim([0.3, 0.7])
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('price_discovery_analysis.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved: price_discovery_analysis.png")

plt.show()

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. DISCOVERY SPEED HIERARCHY:
   - HFT markets: Half-life 50-100 periods (seconds to minutes)
   - Traditional markets: Half-life 200+ periods (hours to days)
   - Pre-electronic: Half-life 1000+ periods (days to weeks)

2. INFORMATION ASYMMETRY EFFECTS:
   - Order flow imbalance (PIN) signals informed trading
   - Spreads widen during high-PIN periods (adverse selection cost)
   - Market maker inventory adjusts defensively against informed flow

3. MULTI-VENUE DISCOVERY:
   - Primary venue (most liquid) discovers first (~60-70% of info)
   - Secondary venues lag by 5-15 periods depending on fragmentation
   - Reg NMS prevents trade-through but slows cross-venue convergence

4. MEASUREMENT CHALLENGES:
   - True fundamental value often unknown at discovery moment
   - Private information creates systematic mispricings
   - Estimation models (Kyle, Glosten-Milgrom) make simplifying assumptions
""")
