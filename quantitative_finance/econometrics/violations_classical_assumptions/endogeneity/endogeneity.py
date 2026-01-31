import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy import stats

# Simulation parameters
np.random.seed(42)
n = 500

# True parameters
beta_0_true, beta_1_true = 2.0, 1.5
corr_endogeneity = 0.6  # Correlation between X and Îµ (creates endogeneity)

# Scenario 1: Exogenous X (baseline)
print("=" * 80)
print("ENDOGENEITY: OLS BIAS vs. IV CORRECTION")
print("=" * 80)

print("\nScenario 1: EXOGENOUS X (Baseline - OLS Consistent)")
print("-" * 80)

# Generate exogenous X
X_exog = np.random.uniform(0, 10, n)
epsilon_exog = np.random.normal(0, 1, n)
Y_exog = beta_0_true + beta_1_true * X_exog + epsilon_exog

# OLS
X_exog_design = np.column_stack([np.ones(n), X_exog])
beta_ols_exog = inv(X_exog_design.T @ X_exog_design) @ (X_exog_design.T @ Y_exog)

print(f"OLS estimates: Î²Ì‚â‚€ = {beta_ols_exog[0]:.4f}, Î²Ì‚â‚ = {beta_ols_exog[1]:.4f}")
print(f"True parameters: Î²â‚€ = {beta_0_true:.4f}, Î²â‚ = {beta_1_true:.4f}")
print(f"Bias in Î²Ì‚â‚: {beta_ols_exog[1] - beta_1_true:.6f} (negligible âœ“)")

# Scenario 2: Endogenous X (omitted variable bias)
print("\nScenario 2: ENDOGENOUS X (Omitted Variable Bias)")
print("-" * 80)

# Generate unobserved confounding variable
confound = np.random.normal(0, 1, n)

# Generate X correlated with confound
X_endo = 3 + 0.5 * confound + np.random.normal(0, 1, n)

# Generate Y with confound omitted (enters error)
# Y = Î²â‚€ + Î²â‚X + Î²â‚‚confound + Îµ, but we omit confound
beta_2_confound = 0.8  # True effect of confounder
epsilon_endo = beta_2_confound * confound + np.random.normal(0, 1, n)  # Omitted variable in error
Y_endo = beta_0_true + beta_1_true * X_endo + epsilon_endo

# OLS on endogenous model
X_endo_design = np.column_stack([np.ones(n), X_endo])
beta_ols_endo = inv(X_endo_design.T @ X_endo_design) @ (X_endo_design.T @ Y_endo)

# Check endogeneity (correlation between X and Îµ)
residuals_ols = Y_endo - X_endo_design @ beta_ols_endo
corr_X_error = np.corrcoef(X_endo, residuals_ols)[0, 1]

print(f"OLS estimates: Î²Ì‚â‚€ = {beta_ols_endo[0]:.4f}, Î²Ì‚â‚ = {beta_ols_endo[1]:.4f}")
print(f"True parameters: Î²â‚€ = {beta_0_true:.4f}, Î²â‚ = {beta_1_true:.4f}")
print(f"Bias in Î²Ì‚â‚: {beta_ols_endo[1] - beta_1_true:.6f} (OLS BIASED!)")
print(f"Correlation(X, Îµ): {corr_X_error:.4f} (Endogeneity detected)")

# Omitted variable bias formula: E[Î²Ì‚â‚] = Î²â‚ + Î²â‚‚ Ã— Cov(X, confound) / Var(X)
cov_X_confound = np.cov(X_endo, confound)[0, 1]
var_X = np.var(X_endo, ddof=1)
predicted_bias = beta_2_confound * (cov_X_confound / var_X)
print(f"Predicted bias (formula): {predicted_bias:.6f}")
print(f"Actual bias (E[Î²Ì‚â‚] - Î²â‚): {beta_ols_endo[1] - beta_1_true:.6f} (match! âœ“)")

# Scenario 3: IV Correction
print("\nScenario 3: INSTRUMENTAL VARIABLE (IV) CORRECTION")
print("-" * 80)

# Create instrument Z: correlated with X but not with confound (exogenous)
Z = 0.7 * X_endo + np.random.normal(0, 1.5, n)  # Z related to X, not directly to confound

# Verify relevance and exogeneity
corr_Z_X = np.corrcoef(Z, X_endo)[0, 1]
corr_Z_error = np.corrcoef(Z, residuals_ols)[0, 1]

print(f"Instrument properties:")
print(f"  Correlation(Z, X): {corr_Z_X:.4f} (Relevance: |corr| > 0.3 âœ“)")
print(f"  Correlation(Z, Îµ): {corr_Z_error:.4f} (Exogeneity: |corr| â‰ˆ 0 âœ“)")

# IV estimation (2SLS)
# Stage 1: Regress X on Z
Z_design = np.column_stack([np.ones(n), Z])
gamma_stage1 = inv(Z_design.T @ Z_design) @ (Z_design.T @ X_endo)
X_fitted = Z_design @ gamma_stage1

# Stage 2: Regress Y on fitted X
X_fitted_design = np.column_stack([np.ones(n), X_fitted])
beta_iv = inv(X_fitted_design.T @ X_fitted_design) @ (X_fitted_design.T @ Y_endo)

print(f"\n2SLS (IV) estimates: Î²Ì‚â‚€ = {beta_iv[0]:.4f}, Î²Ì‚â‚ = {beta_iv[1]:.4f}")
print(f"True parameters: Î²â‚€ = {beta_0_true:.4f}, Î²â‚ = {beta_1_true:.4f}")
print(f"Bias in IV Î²Ì‚â‚: {beta_iv[1] - beta_1_true:.6f} (much smaller! âœ“)")

# Hausman test: OLS vs. IV
print(f"\nHausman Test (OLS vs. IV):")
residuals_iv = Y_endo - X_endo_design @ beta_iv
var_ols_endo = np.sum(residuals_ols**2) / (n - 2)
var_iv = np.sum(residuals_iv**2) / (n - 2)

# Simplified Hausman: compare Î²Ì‚ estimates
diff_beta = beta_ols_endo[1] - beta_iv[1]
# Compute variance of difference (requires variance estimates)
se_diff = np.sqrt(var_ols_endo / np.sum((X_endo - X_endo.mean())**2) + 
                  var_iv / np.sum((X_fitted - X_fitted.mean())**2))
t_hausman = diff_beta / se_diff
p_hausman = 2 * (1 - stats.t.cdf(np.abs(t_hausman), df=n-2))

print(f"  t-statistic: {t_hausman:.4f}, p-value: {p_hausman:.6f}")
if p_hausman < 0.05:
    print(f"  â†’ Reject Hâ‚€: Endogeneity present (OLS and IV differ significantly)")
else:
    print(f"  â†’ Fail to reject Hâ‚€: No significant endogeneity")

print("=" * 80)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scenario 1: Exogenous X
axes[0, 0].scatter(X_exog, Y_exog, alpha=0.5, s=20)
axes[0, 0].plot(X_exog, X_exog_design @ beta_ols_exog, 'r-', linewidth=2, label=f'OLS: Î²Ì‚â‚={beta_ols_exog[1]:.3f}')
axes[0, 0].set_xlabel('X (Exogenous)', fontsize=10, fontweight='bold')
axes[0, 0].set_ylabel('Y', fontsize=10, fontweight='bold')
axes[0, 0].set_title(f'Scenario 1: Exogenous X (OLS Unbiased)', fontsize=11, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Scenario 2: Endogenous X
axes[0, 1].scatter(X_endo, Y_endo, alpha=0.5, s=20)
axes[0, 1].plot(X_endo, X_endo_design @ beta_ols_endo, 'r-', linewidth=2, label=f'OLS: Î²Ì‚â‚={beta_ols_endo[1]:.3f} (BIASED)')
axes[0, 1].axhline(y=beta_1_true, color='green', linestyle='--', linewidth=2, label=f'True Î²â‚={beta_1_true:.3f}')
axes[0, 1].set_xlabel('X (Endogenous)', fontsize=10, fontweight='bold')
axes[0, 1].set_ylabel('Y', fontsize=10, fontweight='bold')
axes[0, 1].set_title(f'Scenario 2: Endogenous X (Omitted Variable Bias)', fontsize=11, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# IV correction
axes[1, 0].scatter(X_endo, Y_endo, alpha=0.5, s=20, label='Data')
axes[1, 0].plot(X_endo, X_endo_design @ beta_ols_endo, 'r-', linewidth=2, label=f'OLS: Î²Ì‚â‚={beta_ols_endo[1]:.3f}')
axes[1, 0].plot(X_endo, X_endo_design @ beta_iv, 'g-', linewidth=2, label=f'IV: Î²Ì‚â‚={beta_iv[1]:.3f}')
axes[1, 0].axhline(y=beta_1_true, color='black', linestyle='--', linewidth=2, label=f'True Î²â‚={beta_1_true:.3f}')
axes[1, 0].set_xlabel('X', fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('Y', fontsize=10, fontweight='bold')
axes[1, 0].set_title('Scenario 3: IV Correction (2SLS)', fontsize=11, fontweight='bold')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Bias comparison
methods = ['OLS\n(Exog X)', 'OLS\n(Endo X)', 'IV\n(2SLS)']
biases = [beta_ols_exog[1] - beta_1_true, beta_ols_endo[1] - beta_1_true, beta_iv[1] - beta_1_true]
colors = ['green', 'red', 'blue']
axes[1, 1].bar(methods, biases, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1, 1].set_ylabel('Bias (Î²Ì‚ - Î²_true)', fontsize=10, fontweight='bold')
axes[1, 1].set_title('Estimator Bias Comparison', fontsize=11, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('endogeneity_iv_correction.png', dpi=150)
plt.show()
