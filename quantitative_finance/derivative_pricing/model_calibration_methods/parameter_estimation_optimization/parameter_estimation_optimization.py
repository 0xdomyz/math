
# Block 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.stats import norm

# Set seed
np.random.seed(42)

# Generate synthetic option data (ATM IV = 20%, smile structure)
K = np.array([0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20])  # Moneyness
T = 1.0  # 1 year
r = 0.03
S0 = 100
F = S0 * np.exp(r * T)  # Forward price

# True market IVs (synthetic; with smile)
atm_vol = 0.20
smile_curvature = 0.02
market_iv = atm_vol + smile_curvature * ((K/F - 1)**2)  # Parabolic smile
market_iv = market_iv.clip(0.05, 0.50)

print("="*70)
print("SABR Model Calibration: Volatility Smile Fitting")
print("="*70)
print(f"Forward Price: ${F:.2f}")
print(f"Time to Maturity: {T} year")
print(f"ATM IV: {atm_vol:.1%}")
print("")
print("Market Data (Strikes and IVs):")
print("-"*70)
for k, iv in zip(K, market_iv):
    print(f"Strike {k*F:7.1f} (Moneyness {k:4.2f}): IV = {iv:.2%}")
print("")

# SABR Model
def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """
    SABR model volatility (simplified; asymptotic formula)
    alpha: ATM volatility
    beta: CEV parameter (0=normal, 1=lognormal)
    rho: correlation (FX vol)
    nu: volatility of volatility
    """
    if K == F:
        # ATM: Simplifies to alpha
        return alpha
    
    # Log-moneyness
    x = np.log(F / K)
    
    # SABR formula (Hagan et al. 2002)
    # Approximation: σ_SAB ≈ α / (F*K)^((1-β)/2) × z / χ(z) × [ 1 + term2 + term3 ]
    
    # Simplified formula for beta close to 1 (lognormal)
    if beta >= 0.99:
        # Pure lognormal approximation
        z = nu / alpha * F**beta * np.log(F / K)
        if abs(z) < 1e-6:
            chi_z = 1.0
        else:
            chi_z = z / np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        vol = alpha * chi_z / (F*K)**(0.5 * (1 - beta)) * (
            1 +
            ((1-beta)**2 / 24) * (alpha / (F*K)**(1-beta))**2 * np.log(F/K)**2 +
            (rho * beta * nu * alpha / (4 * (F*K)**(0.5*(1-beta)))) * np.log(F/K) +
            ((2 - 3*rho**2) / 24) * nu**2 * T
        )
    else:
        # General CEV case (simpler approximation)
        vol = alpha
    
    return max(vol, 0.01)  # Floor at 1%

# Vectorized version
def sabr_vol_vec(K, F, T, alpha, beta, rho, nu):
    return np.array([sabr_vol(F, k, T, alpha, beta, rho, nu) for k in K])

# Calibration: Minimize MSE between model IV and market IV
def objective(params):
    alpha, beta, rho, nu = params
    
    # Constraints
    if alpha <= 0 or alpha > 1:
        return 1e10
    if beta < 0.5 or beta > 1:
        return 1e10
    if rho < -0.99 or rho > 0.99:
        return 1e10
    if nu <= 0 or nu > 2:
        return 1e10
    
    model_iv = sabr_vol_vec(K, F, T, alpha, beta, rho, nu)
    mse = np.mean((model_iv - market_iv)**2)
    return mse

# Initial guess
x0 = [0.20, 0.95, -0.2, 0.4]

# Optimize
result = minimize(objective, x0, method='Nelder-Mead', 
                  options={'maxiter': 5000, 'xatol': 1e-6})

alpha_opt, beta_opt, rho_opt, nu_opt = result.x

print("Optimization Results:")
print("-"*70)
print(f"Converged: {result.success}")
print(f"Iterations: {result.nit}")
print(f"Objective (MSE): {result.fun:.2e}")
print("")
print("Calibrated SABR Parameters:")
print("-"*70)
print(f"α (ATM vol):        {alpha_opt:.4f} ({alpha_opt:.2%})")
print(f"β (CEV parameter):  {beta_opt:.4f}")
print(f"ρ (correlation):    {rho_opt:.4f}")
print(f"ν (vol of vol):     {nu_opt:.4f}")
print("")

# Fit quality
model_iv_opt = sabr_vol_vec(K, F, T, alpha_opt, beta_opt, rho_opt, nu_opt)
residuals = model_iv_opt - market_iv
rmse = np.sqrt(np.mean(residuals**2))

print("Fit Quality:")
print("-"*70)
print(f"RMSE (IV):          {rmse:.2%}")
print(f"Max Error:          {np.max(np.abs(residuals)):.2%}")
print("")

# Detailed comparison
print("Calibration Results (Strike-by-Strike):")
print("-"*70)
print(f"{'Strike':<12} {'Moneyness':<12} {'Market IV':<12} {'Model IV':<12} {'Error':<12}")
print("-"*70)
for k, m_iv, md_iv, err in zip(K, market_iv, model_iv_opt, residuals):
    print(f"${k*F:8.1f}   {k:8.2f}       {m_iv:8.2%}     {md_iv:8.2%}     {err:+8.2%}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Smile fitting
ax = axes[0, 0]
strike_fine = np.linspace(0.70, 1.30, 100)
market_iv_fine = atm_vol + smile_curvature * ((strike_fine/F - 1)**2)
market_iv_fine = market_iv_fine.clip(0.05, 0.50)
model_iv_fine = sabr_vol_vec(strike_fine, F, T, alpha_opt, beta_opt, rho_opt, nu_opt)

ax.plot(strike_fine, market_iv_fine * 100, 'b-', linewidth=2, label='Market IV')
ax.plot(strike_fine, model_iv_fine * 100, 'r--', linewidth=2, label='SABR Fit')
ax.scatter(K, market_iv * 100, color='blue', s=100, marker='o', zorder=5, label='Market Data Points')
ax.scatter(K, model_iv_opt * 100, color='red', s=100, marker='x', zorder=5, label='Model Fit')

ax.set_xlabel('Moneyness (K/F)')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('SABR Model Calibration to Volatility Smile')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Residuals
ax = axes[0, 1]
ax.scatter(K, residuals * 100, color='green', s=100, alpha=0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.axhline(rmse * 100, color='red', linestyle=':', linewidth=1, label=f'RMSE: {rmse:.2%}')
ax.axhline(-rmse * 100, color='red', linestyle=':', linewidth=1)

ax.set_xlabel('Moneyness (K/F)')
ax.set_ylabel('Residuals (%)')
ax.set_title('Calibration Residuals (Model - Market)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Parameter sensitivity (vega)
ax = axes[1, 0]
alpha_range = np.linspace(0.15, 0.30, 20)
rmse_by_alpha = []

for alpha_test in alpha_range:
    iv_test = sabr_vol_vec(K, F, T, alpha_test, beta_opt, rho_opt, nu_opt)
    rmse_test = np.sqrt(np.mean((iv_test - market_iv)**2))
    rmse_by_alpha.append(rmse_test)

ax.plot(alpha_range * 100, np.array(rmse_by_alpha) * 100, 'purple', linewidth=2)
ax.axvline(alpha_opt * 100, color='red', linestyle='--', linewidth=2, label=f'Optimal: {alpha_opt:.2%}')

ax.set_xlabel('α (ATM Volatility, %)')
ax.set_ylabel('RMSE (%)')
ax.set_title('Objective Function Sensitivity to α')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Model parameters summary
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
SABR Model Calibration Summary

Calibrated Parameters:
  α (ATM volatility)      = {alpha_opt:.4f}
  β (CEV parameter)       = {beta_opt:.4f}
  ρ (correlation)         = {rho_opt:.4f}
  ν (volatility of vol)   = {nu_opt:.4f}

Fit Quality:
  RMSE (IV)               = {rmse:.2%}
  Max Absolute Error      = {np.max(np.abs(residuals)):.2%}
  R² (approx)             = {1 - np.sum(residuals**2)/np.sum((market_iv - np.mean(market_iv))**2):.4f}

Observations:
  Data Points             = {len(K)}
  Maturities              = {T} year
  Strike Range (K/F)      = {K.min():.2f} to {K.max():.2f}

Interpretation:
  β ≈ 1: Lognormal dynamics
  ρ < 0: Negative skew (volatility smile)
  ν > 0: Stochastic volatility regime
"""

ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('parameter_estimation_optimization.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*70)
print("Key Insights:")
print("="*70)
print("1. SABR calibration captures volatility smile structure")
print("2. Negative ρ reflects skew (lower strikes higher IV)")
print("3. Fit quality (RMSE) indicates model adequacy")
print("4. Parameter sensitivity important for risk management")
print("5. Leverage calibrated params for exotic pricing")