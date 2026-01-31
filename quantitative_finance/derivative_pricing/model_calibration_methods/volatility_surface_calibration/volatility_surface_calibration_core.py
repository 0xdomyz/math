import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

# Block 1

np.random.seed(42)

print("="*70)
print("Interest Rate Swaption IV Surface Calibration")
print("="*70)

# Synthetic swaption IV surface (realistic shape)
# Tenors (years to expiration) and underlying swap lengths
tenors = np.array([0.5, 1, 2, 5, 10])  # T (years)
swaps = np.array([1, 2, 5, 10, 30])    # τ (swap length, years)

# Create mesh grid
T_grid, tau_grid = np.meshgrid(tenors, swaps)

# Synthetic IV surface (ATM implied vols)
# U-shaped term structure (short: high, medium: low, long: rising)
# Smile structure (OTM skew)
iv_atm = 0.15 + 0.05 * np.exp(-(T_grid - 1)**2 / 0.5) + 0.02 / np.sqrt(tau_grid)
atm_iv_flat = iv_atm.flatten()

print(f"ATM Swaption IVs (Sample):")
print("-"*70)
print(f"{'Tenor (y)':<12} {'5Y Swap':<12} {'10Y Swap':<12} {'30Y Swap':<12}")
print("-"*70)
for i, t in enumerate(tenors):
    iv_5y = iv_atm[2, i]  # 5Y swap row
    iv_10y = iv_atm[3, i]  # 10Y swap row
    iv_30y = iv_atm[4, i]  # 30Y swap row
    print(f"{t:<12.1f} {iv_5y:<12.2%} {iv_10y:<12.2%} {iv_30y:<12.2%}")
print("")

# Model 1: SABR (parametric)
def sabr_iv(F, K, T, alpha, beta, rho, nu):
    """SABR implied vol formula"""
    if K == F:
        return alpha
    
    x = np.log(F / K)
    z = nu / alpha * F**beta * x
    
    if abs(z) < 1e-6:
        chi_z = 1.0
    else:
        chi_z = z / np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
    
    if beta >= 0.99:
        vol = alpha * chi_z / (F*K)**(0.5*(1-beta)) * (
            1 + ((1-beta)**2/24) * np.log(F/K)**2 +
            (rho * beta * nu * alpha / 4) * np.log(F/K) +
            ((2 - 3*rho**2)/24) * nu**2 * T
        )
    else:
        vol = alpha
    
    return max(vol, 0.01)

def sabr_surface(T, tau, params):
    """SABR surface (ATM, beta fixed at 1)"""
    alpha, rho, nu = params
    beta = 1.0
    F = 0.03 + 0.01 * np.log(tau)  # Forward rate term-dependent
    return sabr_iv(F, F, T, alpha, rho, nu, beta)

# Model 2: Parametric smile (polynomial in log-moneyness + term structure)
def poly_smile_surface(T, tau, params):
    """Polynomial smile + exponential term structure"""
    a, b, c, d, e = params
    
    # Base ATM vol
    atm = a * np.exp(-b * T) + c / np.sqrt(tau) + d
    
    # Add smile (second-order)
    smile_effect = e * 0.01 * T * (0.1**2)  # Minimal smile for demo
    
    return np.maximum(atm + smile_effect, 0.05)

# Model 3: Spline interpolation (least smooth; highest fidelity)
def spline_surface(T, tau, T_data, tau_data, iv_data):
    """Interpolate via bivariate spline (simplified: separate splines per maturity)"""
    # For each tenor, fit tau structure
    try:
        cs = CubicSpline(tau_data, iv_data)
        return np.maximum(cs(tau), 0.05)
    except:
        return iv_data.mean() * np.ones_like(T)

# Calibration data: Market IVs at grid points
market_iv_data = iv_atm.flatten()
market_T = T_grid.flatten()
market_tau = tau_grid.flatten()

# Calibrate SABR (3 parameters: α, ρ, ν; β fixed = 1)
def sabr_objective(params):
    try:
        alpha, rho, nu = params
        if alpha <= 0 or alpha > 1:
            return 1e10
        if rho < -0.99 or rho > 0.99:
            return 1e10
        if nu <= 0 or nu > 2:
            return 1e10
        
        model_iv = np.array([sabr_surface(T, tau, params) 
                             for T, tau in zip(market_T, market_tau)])
        
        mse = np.mean((model_iv - market_iv_data)**2)
        return mse
    except:
        return 1e10

x0_sabr = [0.15, -0.3, 0.4]
result_sabr = minimize(sabr_objective, x0_sabr, method='Nelder-Mead',
                       options={'maxiter': 5000})
params_sabr = result_sabr.x

# Calibrate Polynomial
def poly_objective(params):
    try:
        a, b, c, d, e = params
        if a <= 0 or b < 0 or c < 0 or d < 0 or e < 0:
            return 1e10
        
        model_iv = np.array([poly_smile_surface(T, tau, params)
                             for T, tau in zip(market_T, market_tau)])
        
        mse = np.mean((model_iv - market_iv_data)**2)
        return mse
    except:
        return 1e10

x0_poly = [0.15, 0.1, 0.02, 0.12, 0.01]
result_poly = minimize(poly_objective, x0_poly, method='Nelder-Mead',
                       options={'maxiter': 5000})
params_poly = result_poly.x

# Fit spline
params_spline = None  # No parameters; just interpolation

print("Calibration Results:")
print("-"*70)
print(f"SABR MSE:         {result_sabr.fun:.2e}")
print(f"Polynomial MSE:   {result_poly.fun:.2e}")
print("")
print(f"SABR params:")
print(f"  α = {params_sabr[0]:.4f}, ρ = {params_sabr[1]:.4f}, ν = {params_sabr[2]:.4f}")
print(f"Poly params:")
print(f"  a={params_poly[0]:.4f}, b={params_poly[1]:.4f}, c={params_poly[2]:.4f}, " +
      f"d={params_poly[3]:.4f}, e={params_poly[4]:.4f}")
print("")

# Generate predictions on fine grid
T_fine = np.linspace(0.5, 10, 30)
tau_fine = np.linspace(1, 30, 30)
T_fine_grid, tau_fine_grid = np.meshgrid(T_fine, tau_fine)

# SABR surface
iv_sabr_fine = np.array([[sabr_surface(T, tau, params_sabr) 
                          for T in T_fine] for tau in tau_fine])

# Polynomial surface
iv_poly_fine = np.array([[poly_smile_surface(T, tau, params_poly)
                          for T in T_fine] for tau in tau_fine])

# Spline (interpolate only on original grid)
iv_spline_fine = np.array([[np.interp(tau, swaps, iv_atm[:, np.argmin(np.abs(tenors - T))])
                            for T in T_fine] for tau in tau_fine])

# Compute fit quality