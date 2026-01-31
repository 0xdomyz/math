import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import iv  # Bessel function
from scipy.integrate import quad
from scipy.stats import norm

# Block 1

np.random.seed(42)

print("="*70)
print("Stochastic Model Calibration: Heston vs Jump-Diffusion")
print("="*70)

# Synthetic market data (S&P 500 ATM implied vols)
strikes = np.array([0.90, 0.95, 1.00, 1.05, 1.10])  # Moneyness K/S
maturities = np.array([0.25, 1.0, 2.0])  # T (years): 3M, 1Y, 2Y
S0 = 100
r = 0.02
q = 0.01
T_grid, K_grid = np.meshgrid(maturities, strikes)

# Synthetic IVs (realistic smile + term structure)
iv_surface = 0.20 + 0.06 * np.exp(-(T_grid - 1)**2 / 0.5) + \
             0.04 * ((K_grid - 1) ** 2)  # Smile effect

print("Market Data (Implied Volatility Surface):")
print("-"*70)
print(f"{'Maturity':<12} {'0.90 Put':<12} {'0.95 Put':<12} {'1.00 ATM':<12} {'1.05 Call':<12} {'1.10 Call':<12}")
print("-"*70)
for j, T in enumerate(maturities):
    row = [f"{T:.2f}Y"]
    for i, K in enumerate(strikes):
        row.append(f"{iv_surface[i, j]:.2%}")
    print(f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]:<12}")

# Convert IVs to prices (Black-Scholes)

def bs_price(S, K, T, r, q, sigma, call=True):
    """Black-Scholes call/put price"""
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if call:
        price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    
    return price

market_prices = np.zeros_like(iv_surface)
for i in range(len(strikes)):
    for j in range(len(maturities)):
        market_prices[i, j] = bs_price(S0, strikes[i]*S0, maturities[j], r, q, iv_surface[i, j])

print("")
print("Market Prices:")
print("-"*70)
print(f"{'Maturity':<12} {'0.90 Put':<12} {'0.95 Put':<12} {'1.00 ATM':<12} {'1.05 Call':<12} {'1.10 Call':<12}")
print("-"*70)
for j, T in enumerate(maturities):
    row = [f"{T:.2f}Y"]
    for i, K in enumerate(strikes):
        row.append(f"${market_prices[i, j]:.2f}")
    print(f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]:<12}")

# Heston pricing (characteristic function approach, simplified)
def heston_cf(u, S, K, T, r, q, kappa, theta, sigma, rho, v0):
    """Heston characteristic function for implied vol (simplified)"""
    # Use approximation: AtmVol from Heston ≈ sqrt(theta) with adjustment
    # Proper implementation uses Fourier inversion; here simplified for demo
    
    # Effective vol: incorporate mean reversion effect
    alpha = np.sqrt(kappa**2 + 2*sigma**2*u*(u+1j))
    exp_alpha = np.exp(-alpha*T/2)
    
    numerator = 2*alpha*exp_alpha / (1 - rho*sigma*u*1j)
    denominator = kappa - rho*sigma*u*1j + alpha*(1 + rho*sigma*u*1j*exp_alpha)
    
    frac = numerator / denominator
    
    # Characteristic exponent (simplified approximation)
    real_part = (u**2 + 0.5*u) * theta * T
    imag_part = u*rho*sigma*v0*T / kappa
    
    return np.exp(1j*u*np.log(S/K) + real_part + imag_part)

def heston_iv_approx(S, K, T, r, q, kappa, theta, sigma, rho, v0):
    """Approximate Heston IV (actual Heston requires Fourier; using SABR-like fit for demo)"""
    # Simplified: IV ≈ sqrt(theta) + smile adjustment
    moneyness = np.log(K/S)
    
    # ATM vol
    atm_vol = np.sqrt(theta)
    
    # Smile (quadratic in moneyness)
    smile = 0.03 * moneyness**2  # Calibrate to match market smile
    
    # Term structure (mean reversion effect)
    term_adj = -(sigma**2 / (2*kappa)) * (1 - np.exp(-2*kappa*T)) * (1 - rho*moneyness/10)
    
    iv = atm_vol + smile + term_adj
    
    return np.maximum(iv, 0.05)  # Floor at 5%

def heston_price_approx(S, K, T, r, q, kappa, theta, sigma, rho, v0):
    """Get Heston price via IV approximation + BS formula"""
    iv = heston_iv_approx(S, K, T, r, q, kappa, theta, sigma, rho, v0)
    return bs_price(S, K, T, r, q, iv)

# Jump-Diffusion pricing (Merton formula)
def merton_series_cf(u, T, sigma, lam, mu_J, sigma_J):
    """Merton characteristic function component"""
    lambda_t = lam * T
    m_J = np.log(1 + mu_J) - 0.5*sigma_J**2  # Adjust for jump size
    
    # Series expansion (truncate at N terms)
    cf = np.exp(-lambda_t)
    for n in range(50):
        sigma_n_sq = sigma**2 + n*sigma_J**2/T
        cf += (np.exp(-lambda_t + n*np.log(lambda_t) - np.log(np.math.factorial(n))) * 
               np.exp(1j*u*n*m_J - 0.5*u**2*sigma_n_sq*T))
    
    return cf

def merton_iv_approx(S, K, T, r, q, sigma, lam, mu_J, sigma_J):
    """Approximate Merton IV"""
    # Base volatility (diffusion component)
    vol_base = sigma
    
    # Jump adjustment (adds to effective vol)
    jump_effect = np.sqrt(lam * (mu_J**2 + sigma_J**2))
    
    # Moneyness effect (skew)
    moneyness = np.log(K/S)
    skew = -lam * mu_J * moneyness / 5
    
    iv = np.sqrt(vol_base**2 + jump_effect**2) + skew
    
    return np.maximum(iv, 0.05)

def merton_price_approx(S, K, T, r, q, sigma, lam, mu_J, sigma_J):
    """Get Merton price via IV approximation + BS formula"""
    iv = merton_iv_approx(S, K, T, r, q, sigma, lam, mu_J, sigma_J)
    return bs_price(S, K, T, r, q, iv)

# ===== CALIBRATION =====

# Flatten market data for optimization
K_flat = K_grid.flatten()
T_flat = T_grid.flatten()
prices_flat = market_prices.flatten()
ivs_flat = iv_surface.flatten()

# Heston calibration
def heston_objective(params):
    kappa, theta, sigma, rho, v0 = params
    
    # Constraints
    if kappa <= 0.01 or theta <= 0.01 or sigma <= 0.01 or v0 <= 0.01:
        return 1e10
    if rho < -0.99 or rho > 0.99:
        return 1e10
    if 2*kappa*theta < sigma**2:  # Feller condition
        return 1e10
    
    # Compute prices
    prices_model = np.array([heston_price_approx(S0, k, t, r, q, kappa, theta, sigma, rho, v0) 
                             for k, t in zip(K_flat*S0, T_flat)])
    
    # MSE
    mse = np.mean((prices_model - prices_flat)**2)
    
    return mse

# Initial guess
x0_heston = [1.0, 0.04, 0.5, -0.5, 0.04]

result_heston = minimize(heston_objective, x0_heston, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-8})

kappa_opt, theta_opt, sigma_opt, rho_opt, v0_opt = result_heston.x

print("")
print("="*70)
print("CALIBRATION RESULTS")
print("="*70)
print(f"\nHeston Parameters (Optimized):")
print("-"*70)
print(f"κ (mean reversion):     {kappa_opt:8.4f}  [typical: 0.5-2.0]")
print(f"θ (long-run variance):  {theta_opt:8.4f}  [√θ = {np.sqrt(theta_opt):.2%}]")
print(f"σ (vol of vol):         {sigma_opt:8.4f}  [typical: 0.3-1.0]")
print(f"ρ (leverage):           {rho_opt:8.4f}  [typical: -0.5 to -0.1]")
print(f"v₀ (initial variance):  {v0_opt:8.4f}")
print(f"\nFeller Condition Check: 2κθ = {2*kappa_opt*theta_opt:.4f} vs σ² = {sigma_opt**2:.4f} ✓ PASS" 
      if 2*kappa_opt*theta_opt >= sigma_opt**2 else "✗ FAIL")

# Jump-Diffusion calibration