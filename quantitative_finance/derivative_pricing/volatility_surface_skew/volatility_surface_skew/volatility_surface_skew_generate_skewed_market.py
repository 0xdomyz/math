import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize, least_squares
from scipy.interpolate import CubicSpline, griddata
import warnings
def generate_skewed_market(S0, K_range, T_range, base_vol=0.20, skew=-0.15, curve=0.05):
    """Generate synthetic option prices with volatility skew"""
    market = []
    r = 0.05
    
    for T in T_range:
        for K in K_range:
            log_moneyness = np.log(K/S0)
            
            # Stylized skew: negative slope + some curvature
            # Term structure: slightly decreasing
            term_adj = 1.0 - 0.1 * (1 - np.exp(-T))
            iv = (base_vol + skew * log_moneyness + curve * log_moneyness**2) * term_adj
            iv = max(iv, 0.05)  # Floor
            
            # Calculate BS price
            d1 = (np.log(S0/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
            d2 = d1 - iv*np.sqrt(T)
            call_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            
            market.append({
                'K': K, 'T': T, 'price': call_price, 
                'true_iv': iv, 'type': 'call'
            })
    
    return market

# Scenario 1: Build volatility surface from market data
print("\n" + "="*60)
print("SCENARIO 1: Volatility Surface Construction")
print("="*60)

S0 = 100
r = 0.05

# Generate market data with skew
strikes = np.linspace(85, 115, 13)
maturities = [0.25, 0.5, 1.0]

market_data = generate_skewed_market(S0, strikes, maturities)

# Build surface
vol_surface = VolatilitySurface(S0, r)

for data in market_data:
    vol_surface.add_market_data(data['K'], data['T'], data['price'], data['type'])

iv_surface = vol_surface.build_iv_surface()

print(f"\nMarket: S=${S0}, r={r:.1%}")
print(f"Strikes: {len(strikes)} from ${strikes[0]:.0f} to ${strikes[-1]:.0f}")
print(f"Maturities: {maturities}")

for T in sorted(iv_surface.keys()):
    print(f"\nMaturity T={T}yr:")
    print(f"{'Strike':<10} {'Moneyness':<12} {'Impl Vol':<12} {'True IV':<12}")
    print("-" * 46)
    
    points = sorted(iv_surface[T], key=lambda x: x['K'])
    for i, point in enumerate(points):
        if i % 3 == 0:  # Show every 3rd point
            true_iv = next(d['true_iv'] for d in market_data 
                          if d['K'] == point['K'] and d['T'] == T)
            print(f"${point['K']:<9.0f} {point['moneyness']:<11.3f} "
                  f"{point['iv']*100:<11.2f}% {true_iv*100:<11.2f}%")

# Scenario 2: Analyze skew
print("\n" + "="*60)
print("SCENARIO 2: Volatility Skew Analysis")
print("="*60)

for T in sorted(iv_surface.keys()):
    points = sorted(iv_surface[T], key=lambda x: x['K'])
    
    # Find specific moneyness points
    otm_put = next((p for p in points if p['moneyness'] < 0.95), points[0])
    atm = min(points, key=lambda x: abs(x['moneyness'] - 1.0))
    otm_call = next((p for p in reversed(points) if p['moneyness'] > 1.05), points[-1])
    
    skew_measure = otm_put['iv'] - otm_call['iv']
    slope = (points[-1]['iv'] - points[0]['iv']) / (points[-1]['log_moneyness'] - points[0]['log_moneyness'])
    
    print(f"\nMaturity T={T}yr:")
    print(f"  OTM Put (K=${otm_put['K']:.0f}): IV={otm_put['iv']*100:.2f}%")
    print(f"  ATM (K=${atm['K']:.0f}): IV={atm['iv']*100:.2f}%")
    print(f"  OTM Call (K=${otm_call['K']:.0f}): IV={otm_call['iv']*100:.2f}%")
    print(f"  Skew (Put-Call): {skew_measure*100:.2f}%")
    print(f"  Slope (∂IV/∂ln(K)): {slope:.4f}")

# Scenario 3: Arbitrage checks
print("\n" + "="*60)
print("SCENARIO 3: Arbitrage-Free Constraints")
print("="*60)

for T in sorted(iv_surface.keys()):
    butterfly_ok, butterfly_viol = vol_surface.check_butterfly_arbitrage(iv_surface, T)
    
    print(f"\nMaturity T={T}yr:")
    if butterfly_ok:
        print(f"  ✓ No butterfly arbitrage detected")
    else:
        print(f"  ✗ Butterfly violations: {len(butterfly_viol)} points")
        for K, iv_actual, iv_expected in butterfly_viol[:3]:
            print(f"    K=${K:.0f}: IV={iv_actual*100:.2f}% vs expected {iv_expected*100:.2f}%")

calendar_ok, calendar_viol = vol_surface.check_calendar_arbitrage(iv_surface)
print(f"\nCalendar Arbitrage:")
if calendar_ok:
    print(f"  ✓ Total variance increasing with maturity")
else:
    print(f"  ✗ Calendar violations between maturities:")
    for T1, T2 in calendar_viol:
        print(f"    T={T1}yr → T={T2}yr")

# Scenario 4: SVI model calibration
print("\n" + "="*60)
print("SCENARIO 4: SVI Model Calibration")
print("="*60)

for T in sorted(iv_surface.keys()):
    points = sorted(iv_surface[T], key=lambda x: x['K'])
    
    log_moneyness = np.array([p['log_moneyness'] for p in points])
    ivs = np.array([p['iv'] for p in points])
    
    params, error = SVIModel.calibrate(log_moneyness, ivs, T)
    a, b, rho, m, sigma = params
    
    # Calculate fitted IVs
    fitted_ivs = [SVIModel.svi_implied_vol(k, T, *params) for k in log_moneyness]
    rmse = np.sqrt(np.mean((np.array(fitted_ivs) - ivs)**2))
    
    print(f"\nMaturity T={T}yr:")
    print(f"  SVI Parameters:")
    print(f"    a={a:.6f}, b={b:.6f}, ρ={rho:.4f}, m={m:.4f}, σ={sigma:.4f}")
    print(f"  Fit Quality:")
    print(f"    RMSE: {rmse*10000:.2f} bps")
    print(f"    Max Error: {max(abs(np.array(fitted_ivs) - ivs))*10000:.2f} bps")

# Scenario 5: Term structure analysis
print("\n" + "="*60)
print("SCENARIO 5: Volatility Term Structure")
print("="*60)

atm_term_structure = []

for T in sorted(iv_surface.keys()):
    points = iv_surface[T]
    atm = min(points, key=lambda x: abs(x['moneyness'] - 1.0))
    atm_term_structure.append((T, atm['iv'], atm['iv']**2 * T))

print(f"\nATM Implied Volatility Term Structure:")
print(f"{'Maturity':<12} {'IV':<12} {'Total Var':<15} {'Fwd Var':<15}")
print("-" * 54)

for i, (T, iv, total_var) in enumerate(atm_term_structure):
    if i == 0:
        fwd_var_str = "N/A"
    else:
        T_prev, _, total_var_prev = atm_term_structure[i-1]
        fwd_var = (total_var - total_var_prev) / (T - T_prev)
        fwd_vol = np.sqrt(fwd_var)
        fwd_var_str = f"{fwd_vol*100:.2f}%"
    
    print(f"{T:<11.2f}yr {iv*100:<11.2f}% {total_var:<14.6f} {fwd_var_str:<15}")

# Check if term structure is upward/downward sloping
if len(atm_term_structure) >= 2:
    if atm_term_structure[-1][1] > atm_term_structure[0][1]:
        print(f"\nTerm structure: Upward sloping (mean reversion expected)")
    elif atm_term_structure[-1][1] < atm_term_structure[0][1]:
        print(f"\nTerm structure: Downward sloping (event risk near-term)")
    else:
        print(f"\nTerm structure: Flat")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Volatility smile by maturity
ax = axes[0, 0]
for T in sorted(iv_surface.keys()):
    points = sorted(iv_surface[T], key=lambda x: x['K'])
    strikes_plot = [p['K'] for p in points]
    ivs_plot = [p['iv']*100 for p in points]
    ax.plot(strikes_plot, ivs_plot, 'o-', linewidth=2.5, markersize=8, label=f'T={T}yr')

ax.axvline(S0, color='k', linestyle='--', alpha=0.3, label='ATM')
ax.set_xlabel('Strike')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('Volatility Smile Across Maturities')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Smile by moneyness (normalized)
ax = axes[0, 1]
for T in sorted(iv_surface.keys()):
    points = sorted(iv_surface[T], key=lambda x: x['moneyness'])
    moneyness_plot = [p['moneyness'] for p in points]
    ivs_plot = [p['iv']*100 for p in points]
    ax.plot(moneyness_plot, ivs_plot, 'o-', linewidth=2.5, markersize=8, label=f'T={T}yr')

ax.axvline(1.0, color='k', linestyle='--', alpha=0.3, label='ATM')
ax.set_xlabel('Moneyness (K/S)')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('Volatility Smile by Moneyness')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Term structure (ATM)
ax = axes[0, 2]
T_vals = [t[0] for t in atm_term_structure]
iv_vals = [t[1]*100 for t in atm_term_structure]
ax.plot(T_vals, iv_vals, 'ro-', linewidth=2.5, markersize=10)
ax.set_xlabel('Maturity (years)')
ax.set_ylabel('ATM Implied Volatility (%)')
ax.set_title('ATM Volatility Term Structure')
ax.grid(alpha=0.3)

# Plot 4: 3D Surface
ax = axes[1, 0] = plt.subplot(2, 3, 4, projection='3d')
all_strikes = []
all_maturities = []
all_ivs = []

for T in sorted(iv_surface.keys()):
    for point in iv_surface[T]:
        all_strikes.append(point['K'])
        all_maturities.append(T)
        all_ivs.append(point['iv']*100)

ax.scatter(all_strikes, all_maturities, all_ivs, c=all_ivs, cmap='viridis', s=50)
ax.set_xlabel('Strike')
ax.set_ylabel('Maturity')
ax.set_zlabel('IV (%)')
ax.set_title('3D Volatility Surface')

# Plot 5: SVI fit for one maturity
ax = axes[1, 1]
T_fit = maturities[1]  # Middle maturity
points = sorted(iv_surface[T_fit], key=lambda x: x['K'])
log_moneyness_fit = np.array([p['log_moneyness'] for p in points])
ivs_market = np.array([p['iv']*100 for p in points])

# Calibrate SVI
params_fit, _ = SVIModel.calibrate(log_moneyness_fit, ivs_market/100, T_fit)
log_k_fine = np.linspace(log_moneyness_fit.min(), log_moneyness_fit.max(), 100)
ivs_svi = [SVIModel.svi_implied_vol(k, T_fit, *params_fit)*100 for k in log_k_fine]

ax.plot(log_moneyness_fit, ivs_market, 'ro', markersize=10, label='Market')
ax.plot(log_k_fine, ivs_svi, 'b-', linewidth=2.5, label='SVI Fit')
ax.set_xlabel('Log-Moneyness ln(K/S)')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title(f'SVI Model Fit (T={T_fit}yr)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Skew across maturities
ax = axes[1, 2]
skew_values = []
for T in sorted(iv_surface.keys()):
    points = sorted(iv_surface[T], key=lambda x: x['log_moneyness'])
    log_m = [p['log_moneyness'] for p in points]
    ivs = [p['iv'] for p in points]
    
    # Linear fit to get slope
    slope = np.polyfit(log_m, ivs, 1)[0]
    skew_values.append((T, slope))

T_skew = [s[0] for s in skew_values]
slopes = [s[1] for s in skew_values]

ax.plot(T_skew, slopes, 'mo-', linewidth=2.5, markersize=10)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Maturity (years)')
ax.set_ylabel('Skew Slope (∂IV/∂ln(K))')
ax.set_title('Skew Evolution with Maturity')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()