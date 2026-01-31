import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from scipy.integrate import quad
def black_scholes_call(S, K, r, sigma, T):
    """Black-Scholes call price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_volatility(market_price, S, K, r, T):
    """Compute implied volatility via Newton-Raphson."""
    sigma = 0.2  # Initial guess
    for _ in range(50):
        price = black_scholes_call(S, K, r, sigma, T)
        vega = S * np.sqrt(T) * norm.pdf((np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))
        if vega < 1e-10:
            break
        diff = market_price - price
        if abs(diff) < 1e-6:
            break
        sigma += diff / vega
    return sigma

market_df['IV_Market'] = market_df.apply(
    lambda row: implied_volatility(row['Market_Price'], S0_true, row['Strike'], r_true, row['Maturity']),
    axis=1
)

market_df['IV_Model'] = market_df.apply(
    lambda row: implied_volatility(row['Model_Price'], S0_true, row['Strike'], r_true, row['Maturity']),
    axis=1
)

print("\nImplied Volatility Smile (T=1.0 year):")
T_slice = 1.0
iv_slice = market_df[market_df['Maturity'] == T_slice][['Strike', 'IV_Market', 'IV_Model']].copy()
iv_slice['Moneyness'] = iv_slice['Strike'] / S0_true
print(iv_slice.to_string(index=False))

# =====================================
# VISUALIZATION
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Model vs Market Prices
axes[0, 0].scatter(market_df['Market_Price'], market_df['Model_Price'], alpha=0.6, s=50)
price_range = [market_df['Market_Price'].min(), market_df['Market_Price'].max()]
axes[0, 0].plot(price_range, price_range, 'r--', linewidth=2, label='Perfect Fit')
axes[0, 0].set_xlabel('Market Price ($)')
axes[0, 0].set_ylabel('Model Price ($)')
axes[0, 0].set_title(f'Calibration Fit (RMSE=${rmse:.3f})')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Pricing Errors by Strike
for T in maturities:
    subset = market_df[market_df['Maturity'] == T]
    axes[0, 1].plot(subset['Strike'], subset['Pricing_Error'], marker='o', label=f'T={T}Y')
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Strike')
axes[0, 1].set_ylabel('Pricing Error ($)')
axes[0, 1].set_title('Pricing Errors by Strike and Maturity')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Implied Volatility Smile
for T in maturities:
    subset = market_df[market_df['Maturity'] == T].copy()
    subset = subset.sort_values('Strike')
    axes[1, 0].plot(subset['Strike'], subset['IV_Market']*100, marker='o', linestyle='--', 
                    label=f'Market T={T}Y', alpha=0.7)
    axes[1, 0].plot(subset['Strike'], subset['IV_Model']*100, marker='s', linestyle='-', 
                    label=f'Model T={T}Y')
axes[1, 0].set_xlabel('Strike')
axes[1, 0].set_ylabel('Implied Volatility (%)')
axes[1, 0].set_title('Implied Volatility Smile (Market vs Model)')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Parameter Convergence Visualization
param_names = ['v₀', 'κ', 'θ', 'σᵥ', 'ρ']
param_true = [v0_true, kappa_true, theta_true, sigma_v_true, rho_true]
param_cal = [v0_cal, kappa_cal, theta_cal, sigma_v_cal, rho_cal]

x_pos = np.arange(len(param_names))
width = 0.35
axes[1, 1].bar(x_pos - width/2, param_true, width, label='True', alpha=0.7)
axes[1, 1].bar(x_pos + width/2, param_cal, width, label='Calibrated', alpha=0.7)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(param_names)
axes[1, 1].set_ylabel('Parameter Value')
axes[1, 1].set_title('Calibrated vs True Parameters')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Heston calibration successful:")
print(f"• Parameters recovered within mean error: {np.mean([abs(v0_cal-v0_true), abs(kappa_cal-kappa_true), abs(theta_cal-theta_true), abs(sigma_v_cal-sigma_v_true), abs(rho_cal-rho_true)]):.4f}")
print(f"• Pricing RMSE: ${rmse:.4f} (vs typical option values $5-$20)")
print(f"• IV surface captured: Smile and term structure fitted")
print(f"• Method: Differential Evolution (global) avoided local minima")
print(f"• Feller condition: {'Satisfied' if 2*kappa_cal*theta_cal >= sigma_v_cal**2 else 'Violated'}")