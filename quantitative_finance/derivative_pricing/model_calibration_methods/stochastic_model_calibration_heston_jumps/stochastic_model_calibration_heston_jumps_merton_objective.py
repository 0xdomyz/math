import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import iv  # Bessel function
from scipy.integrate import quad
from scipy.stats import norm
def merton_objective(params):
    sigma_diff, lam, mu_J, sigma_J = params
    
    # Constraints
    if sigma_diff <= 0.01 or lam < 0 or sigma_J <= 0.01:
        return 1e10
    if mu_J < -0.2 or mu_J > 0.2:
        return 1e10
    if lam > 2:  # Reasonable jump frequency
        return 1e10
    
    # Compute prices
    prices_model = np.array([merton_price_approx(S0, k, t, r, q, sigma_diff, lam, mu_J, sigma_J)
                             for k, t in zip(K_flat*S0, T_flat)])
    
    # MSE
    mse = np.mean((prices_model - prices_flat)**2)
    
    return mse

x0_merton = [0.15, 0.05, -0.02, 0.1]

result_merton = minimize(merton_objective, x0_merton, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-8})

sigma_diff_opt, lam_opt, mu_J_opt, sigma_J_opt = result_merton.x

print(f"\nJump-Diffusion Parameters (Optimized):")
print("-"*70)
print(f"σ_diff (diffusion vol):  {sigma_diff_opt:8.4f}  [typical: 0.1-0.2]")
print(f"λ (jump intensity):      {lam_opt:8.4f}  [jumps/year; typical: 0.05-0.5]")
print(f"μ_J (mean jump size):    {mu_J_opt:8.4f}  [typical: -0.05 to 0]")
print(f"σ_J (jump volatility):   {sigma_J_opt:8.4f}  [typical: 0.1-0.5]")

# Fit quality
heston_prices = np.array([heston_price_approx(S0, k, t, r, q, kappa_opt, theta_opt, sigma_opt, rho_opt, v0_opt)
                          for k, t in zip(K_flat*S0, T_flat)])

merton_prices = np.array([merton_price_approx(S0, k, t, r, q, sigma_diff_opt, lam_opt, mu_J_opt, sigma_J_opt)
                          for k, t in zip(K_flat*S0, T_flat)])

heston_rmse = np.sqrt(np.mean((heston_prices - prices_flat)**2))
merton_rmse = np.sqrt(np.mean((merton_prices - prices_flat)**2))

heston_res = heston_prices - prices_flat
merton_res = merton_prices - prices_flat

print(f"\nFit Quality:")
print("-"*70)
print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'Max Error':<12} {'Objective':<12}")
print("-"*70)
print(f"{'Heston':<20} ${heston_rmse:<11.4f} ${np.mean(np.abs(heston_res)):<11.4f} " +
      f"${np.max(np.abs(heston_res)):<11.4f} {result_heston.fun:<12.2e}")
print(f"{'Jump-Diffusion':<20} ${merton_rmse:<11.4f} ${np.mean(np.abs(merton_res)):<11.4f} " +
      f"${np.max(np.abs(merton_res)):<11.4f} {result_merton.fun:<12.2e}")

# ===== VISUALIZATION =====

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Market vs Heston fit
ax = axes[0, 0]
maturities_sorted = sorted(set(T_flat))
for T in maturities_sorted:
    mask = T_flat == T
    K_T = K_flat[mask]
    price_market = prices_flat[mask]
    price_heston = heston_prices[mask]
    
    ax.plot(K_T, price_market, 'o-', label=f'Market {T:.2f}Y', linewidth=2)
    ax.plot(K_T, price_heston, 's--', label=f'Heston {T:.2f}Y', linewidth=2, alpha=0.7)

ax.set_xlabel('Moneyness (K/S)')
ax.set_ylabel('Option Price ($)')
ax.set_title('Market vs Heston Prices')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 2: Market vs Jump-Diffusion fit
ax = axes[0, 1]
for T in maturities_sorted:
    mask = T_flat == T
    K_T = K_flat[mask]
    price_market = prices_flat[mask]
    price_merton = merton_prices[mask]
    
    ax.plot(K_T, price_market, 'o-', label=f'Market {T:.2f}Y', linewidth=2)
    ax.plot(K_T, price_merton, '^--', label=f'JD {T:.2f}Y', linewidth=2, alpha=0.7)

ax.set_xlabel('Moneyness (K/S)')
ax.set_ylabel('Option Price ($)')
ax.set_title('Market vs Jump-Diffusion Prices')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 3: Residuals comparison
ax = axes[0, 2]
ax.scatter(prices_flat, heston_res, label='Heston', s=50, alpha=0.6)
ax.scatter(prices_flat, merton_res, label='Jump-Diffusion', s=50, alpha=0.6)
ax.axhline(0, color='black', linestyle='--', linewidth=1)

ax.set_xlabel('Market Price ($)')
ax.set_ylabel('Residuals ($)')
ax.set_title('Price Residuals (Model - Market)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: IV comparison by moneyness
ax = axes[1, 0]
heston_ivs = np.array([heston_iv_approx(S0, k, t, r, q, kappa_opt, theta_opt, sigma_opt, rho_opt, v0_opt)
                       for k, t in zip(K_flat*S0, T_flat)])
merton_ivs = np.array([merton_iv_approx(S0, k, t, r, q, sigma_diff_opt, lam_opt, mu_J_opt, sigma_J_opt)
                       for k, t in zip(K_flat*S0, T_flat)])

T_1y = 1.0
mask_1y = np.abs(T_flat - T_1y) < 0.01
K_1y = K_flat[mask_1y]
iv_market_1y = ivs_flat[mask_1y]
iv_heston_1y = heston_ivs[mask_1y]
iv_merton_1y = merton_ivs[mask_1y]

ax.plot(K_1y, iv_market_1y * 100, 'o-', label='Market', linewidth=2, markersize=8)
ax.plot(K_1y, iv_heston_1y * 100, 's-', label='Heston', linewidth=2, markersize=8, alpha=0.7)
ax.plot(K_1y, iv_merton_1y * 100, '^-', label='Jump-Diffusion', linewidth=2, markersize=8, alpha=0.7)

ax.set_xlabel('Moneyness (K/S)')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('IV Smile (1Y Maturity)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Model parameters interpretation
ax = axes[1, 1]
ax.axis('off')

summary_heston = f"""
Heston Model Interpretation:

κ = {kappa_opt:.4f}
  ├─ Mean reversion speed
  └─ Half-life ~ {0.693/kappa_opt:.1f} years

θ = {theta_opt:.4f}
  ├─ Long-run variance
  └─ Implied vol ≈ {np.sqrt(theta_opt):.1%}

σ = {sigma_opt:.4f}
  ├─ Volatility of volatility
  └─ Controls tail fatness

ρ = {rho_opt:.4f}
  ├─ Leverage effect
  └─ Negative = vol rises on down moves
"""

ax.text(0.05, 0.5, summary_heston, fontsize=9, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Plot 6: Jump model interpretation
ax = axes[1, 2]
ax.axis('off')

summary_merton = f"""
Jump-Diffusion Interpretation:

σ_diff = {sigma_diff_opt:.4f}
  └─ Diffusion volatility

λ = {lam_opt:.4f}
  ├─ Jump intensity
  └─ ~1 jump per {1/lam_opt:.1f} years

μ_J = {mu_J_opt:.4f}
  ├─ Mean jump size
  └─ {mu_J_opt*100:.2f}% average

σ_J = {sigma_J_opt:.4f}
  ├─ Jump size std dev
  └─ Tail fatness control
"""

ax.text(0.05, 0.5, summary_merton, fontsize=9, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('stochastic_model_calibration.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Heston captures smile structure via leverage (ρ < 0)")
print("2. Jump-Diffusion models tail risk; useful for crisis scenarios")
print("3. Parameter stability critical; recalibrate frequently")
print("4. Out-of-sample validation essential to detect overfitting")
print("5. Combine models (ensemble) for robust pricing")