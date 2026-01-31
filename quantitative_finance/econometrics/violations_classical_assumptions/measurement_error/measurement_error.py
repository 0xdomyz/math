import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy import stats

# Simulation parameters
np.random.seed(42)
n = 500

# True parameters
beta_0, beta_1 = 2.0, 1.5
sigma_epsilon = 1.0
sigma_Xstar = 2.0
sigma_eta = 0.8  # Measurement error std

print("=" * 80)
print("MEASUREMENT ERROR: ATTENUATION BIAS & REMEDIES")
print("=" * 80)

# Generate true X*
Xstar = np.random.normal(5, sigma_Xstar, n)

# Generate measurement error (classical)
eta = np.random.normal(0, sigma_eta, n)

# Observed X
X_obs = Xstar + eta

# Generate Y from true model
epsilon = np.random.normal(0, sigma_epsilon, n)
Y = beta_0 + beta_1 * Xstar + epsilon

# Calculate signal-to-noise ratio and attenuation factor
var_Xstar = np.var(Xstar, ddof=1)
var_eta = np.var(eta, ddof=1)
SNR = var_Xstar / var_eta
lambda_factor = var_Xstar / (var_Xstar + var_eta)

print("\nMeasurement Error Quantification:")
print("-" * 80)
print(f"True parameters: Î²â‚€ = {beta_0}, Î²â‚ = {beta_1}")
print(f"Var(X*) = {var_Xstar:.4f}, Var(Î·) = {var_eta:.4f}")
print(f"Signal-to-Noise Ratio (SNR) = {SNR:.4f}")
print(f"Attenuation factor Î» = {lambda_factor:.4f}")
print(f"Expected attenuation: Î²Ì‚â‚ â‰ˆ {beta_1 * lambda_factor:.4f} (true Î²â‚ = {beta_1})")
print()

# Scenario 1: OLS with error-ridden X (biased)
print("Scenario 1: OLS with Measurement Error (Biased)")
print("-" * 80)

X_obs_design = np.column_stack([np.ones(n), X_obs])
beta_ols_me = inv(X_obs_design.T @ X_obs_design) @ (X_obs_design.T @ Y)

print(f"OLS with observed X:")
print(f"  Î²Ì‚â‚€ = {beta_ols_me[0]:.6f}, Î²Ì‚â‚ = {beta_ols_me[1]:.6f}")
print(f"  Expected Î²Ì‚â‚ â‰ˆ {beta_1 * lambda_factor:.6f}")
print(f"  Actual bias: {beta_ols_me[1] - beta_1:.6f}")
print(f"  Attenuation: {(1 - beta_ols_me[1]/beta_1)*100:.1f}% (expected {(1-lambda_factor)*100:.1f}%)")

# Scenario 2: OLS with true X* (unbiased baseline)
print("\nScenario 2: OLS with True X* (Unbiased Baseline)")
print("-" * 80)

X_true_design = np.column_stack([np.ones(n), Xstar])
beta_ols_true = inv(X_true_design.T @ X_true_design) @ (X_true_design.T @ Y)

print(f"OLS with true X*:")
print(f"  Î²Ì‚â‚€ = {beta_ols_true[0]:.6f}, Î²Ì‚â‚ = {beta_ols_true[1]:.6f}")
print(f"  Bias: {beta_ols_true[1] - beta_1:.6f} (negligible âœ“)")

# Scenario 3: IV estimation (using Xstar as instrument for X_obs)
# In practice, would use external instrument; here using true X* for illustration
print("\nScenario 3: IV Estimation (Using Exogenous Instrument)")
print("-" * 80)

# Use true X* as instrument (pretend it's exogenous to measurement error)
# In real application, would use different instrument (e.g., lagged X, policy variable)
Z = Xstar + np.random.normal(0, 0.5, n)  # Instrument: related to Xstar, not to eta

# 2SLS
# Stage 1: Regress X_obs on Z
Z_design = np.column_stack([np.ones(n), Z])
gamma_1 = inv(Z_design.T @ Z_design) @ (Z_design.T @ X_obs)
X_fitted = Z_design @ gamma_1

# Stage 2: Regress Y on X_fitted
X_fitted_design = np.column_stack([np.ones(n), X_fitted])
beta_iv = inv(X_fitted_design.T @ X_fitted_design) @ (X_fitted_design.T @ Y)

print(f"IV (2SLS) with instrument Z:")
print(f"  Î²Ì‚â‚€ = {beta_iv[0]:.6f}, Î²Ì‚â‚ = {beta_iv[1]:.6f}")
print(f"  Bias: {beta_iv[1] - beta_1:.6f}")
print(f"  Instrument strength (Corr(Z, X_obs)): {np.corrcoef(Z, X_obs)[0,1]:.4f}")
print(f"  Instrument exogeneity (Corr(Z, Î·)): {np.corrcoef(Z, eta)[0,1]:.4f}")

# Scenario 4: Errors-in-variables regression (if error variance known)
print("\nScenario 4: Errors-in-Variables Regression")
print("-" * 80)

# Adjust for known measurement error: Î²Ì‚_corrected = Î²Ì‚_OLS / Î»
beta_eiv = beta_ols_me.copy()
beta_eiv[1] = beta_ols_me[1] / lambda_factor

print(f"OLS corrected for measurement error:")
print(f"  Î²Ì‚â‚ (corrected) = Î²Ì‚â‚ (OLS) / Î» = {beta_ols_me[1]:.6f} / {lambda_factor:.4f} = {beta_eiv[1]:.6f}")
print(f"  Bias: {beta_eiv[1] - beta_1:.6f} (near zero! âœ“)")
print(f"  Note: Correction requires knowing true error variance (ÏƒÂ²â‚™)")

# Residuals comparison
residuals_ols = Y - X_obs_design @ beta_ols_me
residuals_true = Y - X_true_design @ beta_ols_true
residuals_iv = Y - X_fitted_design @ beta_iv

se_ols = np.sqrt(np.sum(residuals_ols**2) / (n - 2))
se_true = np.sqrt(np.sum(residuals_true**2) / (n - 2))
se_iv = np.sqrt(np.sum(residuals_iv**2) / (n - 2))

print("\n\nResidual Standard Error:")
print("-" * 80)
print(f"{'Estimator':<30} {'SE':<12} {'Interpretation':<30}")
print("-" * 80)
print(f"{'OLS with true X*':<30} {se_true:<12.4f} {'Baseline (unbiased)':<30}")
print(f"{'OLS with measured X':<30} {se_ols:<12.4f} {'Inflated (measurement noise)':<30}")
print(f"{'IV/2SLS':<30} {se_iv:<12.4f} {'Higher (efficiency loss)':<30}")

print("=" * 80)

# Sensitivity analysis: Effect of measurement error severity
error_scales = np.linspace(0, 3, 50)
beta_1_estimates = []

for scale in error_scales:
    eta_scaled = np.random.normal(0, sigma_eta * scale, n)
    X_scaled = Xstar + eta_scaled
    
    var_X_scaled = np.var(X_scaled, ddof=1)
    lambda_scaled = var_Xstar / (var_Xstar + np.var(eta_scaled, ddof=1))
    
    # OLS with scaled error
    X_scaled_design = np.column_stack([np.ones(n), X_scaled])
    beta_scaled = inv(X_scaled_design.T @ X_scaled_design) @ (X_scaled_design.T @ Y)
    
    beta_1_estimates.append(beta_scaled[1])

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Scatter X* vs observed X
axes[0, 0].scatter(Xstar, X_obs, alpha=0.5, s=20)
axes[0, 0].plot([Xstar.min(), Xstar.max()], [Xstar.min(), Xstar.max()], 'r--', linewidth=2, label='No error line')
axes[0, 0].set_xlabel('True X*', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Observed X = X* + Î·', fontsize=11, fontweight='bold')
axes[0, 0].set_title(f'Measurement Error Visualization (Ïƒ_Î· = {sigma_eta:.2f})', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Panel 2: Coefficient comparison
methods = ['OLS\n(True X*)', 'OLS\n(Measured X)', 'IV\n(2SLS)', 'EIV\n(Corrected)']
coefficients = [beta_ols_true[1], beta_ols_me[1], beta_iv[1], beta_eiv[1]]
colors = ['green', 'red', 'blue', 'orange']
axes[0, 1].bar(methods, coefficients, color=colors, alpha=0.7, edgecolor='black')
axes[0, 1].axhline(y=beta_1, color='black', linestyle='--', linewidth=2, label=f'True Î²â‚ = {beta_1}')
axes[0, 1].set_ylabel('Î²Ì‚â‚ Estimate', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Estimator Comparison: Measurement Error Remedies', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Panel 3: Regression lines
x_range = np.linspace(Xstar.min(), Xstar.max(), 100)
y_ols_true = beta_ols_true[0] + beta_ols_true[1] * x_range
y_ols_me = beta_ols_me[0] + beta_ols_me[1] * x_range
y_iv = beta_iv[0] + beta_iv[1] * x_range

axes[1, 0].scatter(Xstar, Y, alpha=0.2, s=20, label='Data (X* vs Y)', color='gray')
axes[1, 0].plot(x_range, y_ols_true, 'g-', linewidth=2, label=f'OLS true (Î²Ì‚â‚={beta_ols_true[1]:.3f})')
axes[1, 0].plot(x_range, y_ols_me, 'r-', linewidth=2, label=f'OLS measured (Î²Ì‚â‚={beta_ols_me[1]:.3f})')
axes[1, 0].plot(x_range, y_iv, 'b-', linewidth=2, label=f'IV (Î²Ì‚â‚={beta_iv[1]:.3f})')
axes[1, 0].set_xlabel('X* (True Value)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Y', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Regression Lines: Impact of Measurement Error', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# Panel 4: Sensitivity to error severity
axes[1, 1].plot(error_scales, beta_1_estimates, 'b-', linewidth=2, label='Î²Ì‚â‚(error scale)')
axes[1, 1].axhline(y=beta_1, color='black', linestyle='--', linewidth=2, label=f'True Î²â‚ = {beta_1}')
axes[1, 1].axhline(y=beta_ols_me[1], color='red', linestyle=':', linewidth=2, label=f'Current OLS â‰ˆ {beta_ols_me[1]:.3f}')
axes[1, 1].fill_between([0, 1], beta_1*0.8, beta_1, alpha=0.2, color='green', label='Acceptable range (Â±20%)')
axes[1, 1].set_xlabel('Measurement Error Scale (Ïƒ_Î· multiplier)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Î²Ì‚â‚ Estimate', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Sensitivity: OLS Î²Ì‚â‚ vs. Error Severity', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('measurement_error_attenuation.png', dpi=150)
plt.show()
