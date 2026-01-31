import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, newton
import time
import warnings
def true_vol_smile(K, S, T):
    """Stylized volatility smile - downward sloping"""
    moneyness = np.log(K/S)
    base_vol = 0.20
    skew = -0.15  # Negative skew for equities
    curvature = 0.05
    return base_vol + skew * moneyness + curvature * moneyness**2

implied_vols = []
deltas = []

print(f"\nConstructing smile (S=${S_smile}, T={T_smile}yr):")
print(f"{'Strike':<10} {'Market Price':<15} {'Impl Vol':<12} {'Delta':<10}")
print("-" * 47)

for K_smile in strikes_smile:
    true_vol = true_vol_smile(K_smile, S_smile, T_smile)
    market_price = BlackScholes.call_price(S_smile, K_smile, r_smile, T_smile, true_vol)
    
    iv, _ = ImpliedVolatility.newton_raphson(S_smile, K_smile, r_smile, T_smile, 
                                             market_price, 'call', initial_guess=0.2)
    
    # Calculate delta
    d1 = BlackScholes.d1(S_smile, K_smile, r_smile, T_smile, iv)
    delta = norm.cdf(d1)
    
    implied_vols.append(iv)
    deltas.append(delta)
    
    if K_smile in [80, 90, 100, 110, 120]:
        print(f"${K_smile:<9} ${market_price:<14.4f} {iv*100:<11.2f}% {delta:<10.4f}")

# Scenario 4: Term structure
print("\n" + "="*60)
print("SCENARIO 4: Volatility Term Structure")
print("="*60)

K_atm = 100
maturities = np.array([1/12, 3/12, 6/12, 1, 2])  # 1m, 3m, 6m, 1y, 2y

# Stylized term structure (downward sloping - event risk)
def term_structure_vol(T):
    """Term structure - mean reverting"""
    short_vol = 0.30  # High near-term vol (event)
    long_vol = 0.18   # Long-term mean reversion
    decay = 2.0
    return long_vol + (short_vol - long_vol) * np.exp(-decay * T)

term_vols = []

print(f"\nATM Volatility Term Structure (K=${K_atm}):")
print(f"{'Maturity':<15} {'Market Price':<15} {'Impl Vol':<12} {'Ann. Variance':<15}")
print("-" * 57)

for T_term in maturities:
    true_vol = term_structure_vol(T_term)
    market_price = BlackScholes.call_price(S, K_atm, r, T_term, true_vol)
    
    iv, _ = ImpliedVolatility.newton_raphson(S, K_atm, r, T_term, market_price, 'call')
    variance = iv**2 * T_term
    
    term_vols.append(iv)
    
    maturity_label = f"{T_term*12:.0f} months" if T_term < 1 else f"{T_term:.1f} years"
    print(f"{maturity_label:<15} ${market_price:<14.4f} {iv*100:<11.2f}% {variance:<15.4f}")

# Scenario 5: Error cases
print("\n" + "="*60)
print("SCENARIO 5: Error Handling and Edge Cases")
print("="*60)

test_cases = [
    ("Valid ATM call", 100, 100, 10.0, 'call', True),
    ("Deep ITM call", 100, 80, 21.0, 'call', True),
    ("Deep OTM call", 100, 130, 0.1, 'call', True),
    ("Price too high", 100, 100, 105.0, 'call', False),  # Violates upper bound
    ("Price too low", 100, 100, -1.0, 'call', False),    # Negative price
    ("Near expiry", 100, 100, 5.0, 'call', True),
]

print(f"\n{'Case':<20} {'Valid Bounds':<15} {'IV (NR)':<15} {'IV (Bisect)':<15}")
print("-" * 65)

T_test = 0.25

for case_name, S_test, K_test, price, opt_type, should_work in test_cases:
    # Check bounds
    bounds_ok = ImpliedVolatility.check_arbitrage_bounds(S_test, K_test, r, T_test, price, opt_type)
    
    if bounds_ok:
        iv_nr, _ = ImpliedVolatility.newton_raphson(S_test, K_test, r, T_test, price, opt_type)
        iv_bis, _ = ImpliedVolatility.bisection(S_test, K_test, r, T_test, price, opt_type)
        
        iv_nr_str = f"{iv_nr*100:.2f}%" if not np.isnan(iv_nr) else "FAILED"
        iv_bis_str = f"{iv_bis*100:.2f}%" if not np.isnan(iv_bis) else "FAILED"
    else:
        iv_nr_str = "N/A"
        iv_bis_str = "N/A"
    
    print(f"{case_name:<20} {'✓' if bounds_ok else '✗':<14} {iv_nr_str:<15} {iv_bis_str:<15}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Volatility smile
ax = axes[0, 0]
ax.plot(strikes_smile, np.array(implied_vols)*100, 'bo-', linewidth=2.5, markersize=8)
ax.axvline(S_smile, color='r', linestyle='--', alpha=0.5, label='ATM')
ax.set_xlabel('Strike Price')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('Volatility Smile (Equity Style)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Smile by delta
ax = axes[0, 1]
ax.plot(deltas, np.array(implied_vols)*100, 'go-', linewidth=2.5, markersize=8)
ax.set_xlabel('Delta')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('Volatility by Delta')
ax.grid(alpha=0.3)

# Plot 3: Term structure
ax = axes[0, 2]
ax.plot(maturities*12, np.array(term_vols)*100, 'ro-', linewidth=2.5, markersize=10)
ax.set_xlabel('Maturity (months)')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('ATM Volatility Term Structure')
ax.grid(alpha=0.3)

# Plot 4: Convergence comparison
ax = axes[1, 0]
iterations_nr = []
iterations_bis = []
test_strikes = np.linspace(85, 115, 15)

for K_test in test_strikes:
    market_price = BlackScholes.call_price(S, K_test, r, T, 0.25)
    _, iter_n = ImpliedVolatility.newton_raphson(S, K_test, r, T, market_price, 'call')
    _, iter_b = ImpliedVolatility.bisection(S, K_test, r, T, market_price, 'call')
    iterations_nr.append(iter_n)
    iterations_bis.append(iter_b)

ax.plot(test_strikes, iterations_nr, 'b-', linewidth=2.5, marker='o', label='Newton-Raphson')
ax.plot(test_strikes, iterations_bis, 'r-', linewidth=2.5, marker='s', label='Bisection')
ax.set_xlabel('Strike')
ax.set_ylabel('Iterations to Converge')
ax.set_title('Convergence Speed Comparison')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: IV surface (strike vs maturity)
ax = axes[1, 1]
strikes_surf = np.linspace(85, 115, 20)
maturities_surf = np.linspace(0.1, 2, 20)
IV_surface = np.zeros((len(maturities_surf), len(strikes_surf)))

for i, T_surf in enumerate(maturities_surf):
    for j, K_surf in enumerate(strikes_surf):
        true_vol = true_vol_smile(K_surf, S, T_surf) * term_structure_vol(T_surf) / 0.20
        market_price = BlackScholes.call_price(S, K_surf, r, T_surf, true_vol)
        iv, _ = ImpliedVolatility.newton_raphson(S, K_surf, r, T_surf, market_price, 'call')
        IV_surface[i, j] = iv * 100 if not np.isnan(iv) else 20

im = ax.contourf(strikes_surf, maturities_surf*12, IV_surface, levels=15, cmap='viridis')
ax.set_xlabel('Strike')
ax.set_ylabel('Maturity (months)')
ax.set_title('Implied Volatility Surface')
plt.colorbar(im, ax=ax, label='IV (%)')

# Plot 6: Vega profile (showing why Newton-Raphson works)
ax = axes[1, 2]
strikes_vega = np.linspace(70, 130, 50)
vegas = []

for K_vega in strikes_vega:
    vega = BlackScholes.vega(S, K_vega, r, T, 0.25)
    vegas.append(vega)

ax.plot(strikes_vega, vegas, 'purple', linewidth=2.5)
ax.axvline(S, color='r', linestyle='--', alpha=0.5, label='ATM (max Vega)')
ax.set_xlabel('Strike')
ax.set_ylabel('Vega')
ax.set_title('Vega Profile (Why NR Works)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()