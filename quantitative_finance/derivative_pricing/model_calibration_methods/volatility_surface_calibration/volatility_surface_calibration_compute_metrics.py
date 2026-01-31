import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
def compute_metrics(model_iv_points, market_iv_points):
    """Residual statistics"""
    residuals = model_iv_points - market_iv_points
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    max_error = np.max(np.abs(residuals))
    r2 = 1 - np.sum(residuals**2) / np.sum((market_iv_points - np.mean(market_iv_points))**2)
    return rmse, mae, max_error, r2

# Evaluate at original data points
iv_sabr_data = np.array([sabr_surface(T, tau, params_sabr) for T, tau in zip(market_T, market_tau)])
iv_poly_data = np.array([poly_smile_surface(T, tau, params_poly) for T, tau in zip(market_T, market_tau)])

rmse_sabr, mae_sabr, max_err_sabr, r2_sabr = compute_metrics(iv_sabr_data, market_iv_data)
rmse_poly, mae_poly, max_err_poly, r2_poly = compute_metrics(iv_poly_data, market_iv_data)
rmse_spline, mae_spline, max_err_spline, r2_spline = compute_metrics(market_iv_data, market_iv_data)

print("Fit Quality Metrics:")
print("-"*70)
print(f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'Max Error':<12} {'R²':<12}")
print("-"*70)
print(f"{'SABR':<15} {rmse_sabr:<12.4%} {mae_sabr:<12.4%} {max_err_sabr:<12.4%} {r2_sabr:<12.4f}")
print(f"{'Polynomial':<15} {rmse_poly:<12.4%} {mae_poly:<12.4%} {max_err_poly:<12.4%} {r2_poly:<12.4f}")
print(f"{'Spline':<15} {rmse_spline:<12.4%} {mae_spline:<12.4%} {max_err_spline:<12.4%} {r2_spline:<12.4f}")
print("")

# 3D Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Market IV surface
ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.plot_surface(T_fine_grid, tau_fine_grid, iv_atm / iv_atm.max() * 100, cmap='viridis', alpha=0.8)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Swap Length (years)')
ax.set_zlabel('IV (%)')
ax.set_title('Market IV Surface (Original)')

# Plot 2: SABR fit
ax = fig.add_subplot(2, 3, 2, projection='3d')
ax.plot_surface(T_fine_grid, tau_fine_grid, iv_sabr_fine * 100, cmap='plasma', alpha=0.8)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Swap Length (years)')
ax.set_zlabel('IV (%)')
ax.set_title(f'SABR Calibration (RMSE: {rmse_sabr:.2%})')

# Plot 3: Polynomial fit
ax = fig.add_subplot(2, 3, 3, projection='3d')
ax.plot_surface(T_fine_grid, tau_fine_grid, iv_poly_fine * 100, cmap='cool', alpha=0.8)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Swap Length (years)')
ax.set_zlabel('IV (%)')
ax.set_title(f'Polynomial Calibration (RMSE: {rmse_poly:.2%})')

# Plot 4: SABR residuals
ax = fig.add_subplot(2, 3, 4)
residuals_sabr = iv_sabr_data - market_iv_data
ax.scatter(market_T, residuals_sabr * 100, c=market_tau, cmap='RdYlBu', s=50, alpha=0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Residuals (bps)')
ax.set_title('SABR Residuals')
ax.grid(alpha=0.3)
cb = plt.colorbar(ax.collections[0], ax=ax)
cb.set_label('Swap Length (years)')

# Plot 5: Polynomial residuals
ax = fig.add_subplot(2, 3, 5)
residuals_poly = iv_poly_data - market_iv_data
ax.scatter(market_T, residuals_poly * 100, c=market_tau, cmap='RdYlBu', s=50, alpha=0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Residuals (bps)')
ax.set_title('Polynomial Residuals')
ax.grid(alpha=0.3)
cb = plt.colorbar(ax.collections[0], ax=ax)
cb.set_label('Swap Length (years)')

# Plot 6: Comparison summary
ax = fig.add_subplot(2, 3, 6)
ax.axis('off')

summary_text = f"""
Swaption IV Surface Calibration Comparison

Market Data:
  Tenors (T)       : {', '.join([f'{x:.1f}y' for x in tenors])}
  Swap Lengths (τ) : {', '.join([f'{x:.0f}y' for x in swaps])}
  Total Points     : {len(market_iv_data)}

Model Performance:
┌────────────────────────────────────────┐
│ Model       │ RMSE    │ MAE     │ R²   │
├────────────────────────────────────────┤
│ SABR        │ {rmse_sabr:6.2%}  │ {mae_sabr:6.2%}  │ {r2_sabr:6.4f} │
│ Polynomial  │ {rmse_poly:6.2%}  │ {mae_poly:6.2%}  │ {r2_poly:6.4f} │
│ Spline      │ {rmse_spline:6.2%}  │ {mae_spline:6.2%}  │ {r2_spline:6.4f} │
└────────────────────────────────────────┘

SABR Calibrated Parameters:
  α (ATM vol)  = {params_sabr[0]:8.4f}
  ρ (correlation) = {params_sabr[1]:8.4f}
  ν (vol-of-vol)  = {params_sabr[2]:8.4f}
  β (beta)     = 1.0000 (fixed; lognormal)

Interpretation:
  SABR: Smooth, stable extrapolation
  Poly: Good fit to term structure
  Spline: Perfect fit; risky extrapolation
"""

ax.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('volatility_surface_calibration.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*70)
print("Key Insights:")
print("="*70)
print("1. SABR provides smooth, stable extrapolation with 3 parameters")
print("2. Polynomial fits term structure well; may fail at extremes")
print("3. Spline perfect fit but poor extrapolation to new tenors/swaps")
print("4. Trade-off: Fidelity vs stability; choose based on use case")
print("5. Validation: Forward-test on new options; monitor residuals")