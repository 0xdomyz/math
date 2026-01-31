import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# Simulation parameters
np.random.seed(42)
n = 200
n_simulations = 1000

# True parameters
beta_true = np.array([2.0, 1.5])  # [intercept, slope]
sigma = 1.0

# Scenario 1: Homoscedastic errors (Gauss-Markov holds)
print("=" * 80)
print("GAUSS-MARKOV THEOREM VERIFICATION")
print("=" * 80)
print("Scenario 1: Homoscedastic Errors (ÏƒÂ² constant)")
print("-" * 80)

X_homo = np.column_stack([np.ones(n), np.random.uniform(0, 10, n)])

# Storage for OLS and alternative estimator
beta_ols_homo = np.zeros((n_simulations, 2))
beta_alt_homo = np.zeros((n_simulations, 2))  # Use only first/last observation pairs

for sim in range(n_simulations):
    # Homoscedastic errors
    epsilon = np.random.normal(0, sigma, n)
    Y = X_homo @ beta_true + epsilon
    
    # OLS (uses all observations)
    beta_ols_homo[sim] = inv(X_homo.T @ X_homo) @ (X_homo.T @ Y)
    
    # Alternative estimator: slope from first and last observation only
    X_diff = X_homo[-1, 1] - X_homo[0, 1]
    Y_diff = Y[-1] - Y[0]
    beta_1_alt = Y_diff / X_diff
    # Intercept from mean (ensures unbiasedness)
    beta_0_alt = Y.mean() - beta_1_alt * X_homo[:, 1].mean()
    beta_alt_homo[sim] = [beta_0_alt, beta_1_alt]

# Check unbiasedness
mean_ols = beta_ols_homo.mean(axis=0)
mean_alt = beta_alt_homo.mean(axis=0)
var_ols = beta_ols_homo.var(axis=0, ddof=1)
var_alt = beta_alt_homo.var(axis=0, ddof=1)

print(f"\nEstimator Performance (n={n} observations, {n_simulations} simulations):")
print(f"{'Estimator':<20} {'E[Î²Ì‚â‚€]':<12} {'E[Î²Ì‚â‚]':<12} {'Var(Î²Ì‚â‚€)':<12} {'Var(Î²Ì‚â‚)':<12}")
print("-" * 80)
print(f"{'True parameters':<20} {beta_true[0]:<12.4f} {beta_true[1]:<12.4f} {'-':<12} {'-':<12}")
print(f"{'OLS (all data)':<20} {mean_ols[0]:<12.4f} {mean_ols[1]:<12.4f} {var_ols[0]:<12.6f} {var_ols[1]:<12.6f}")
print(f"{'Alternative (2 obs)':<20} {mean_alt[0]:<12.4f} {mean_alt[1]:<12.4f} {var_alt[0]:<12.6f} {var_alt[1]:<12.6f}")
print()
print(f"Efficiency Ratio (Var(Alt)/Var(OLS)):")
print(f"  Î²Ì‚â‚€: {var_alt[0]/var_ols[0]:.2f}Ã—  |  Î²Ì‚â‚: {var_alt[1]/var_ols[1]:.2f}Ã—")
print(f"  â†’ OLS uses all {n} observations â†’ {var_alt[1]/var_ols[1]:.1f}Ã— more efficient for slope")
print("=" * 80)

# Scenario 2: Heteroscedastic errors (OLS no longer BLUE; WLS is BLUE)
print("\nScenario 2: Heteroscedastic Errors (Ïƒáµ¢Â² = ÏƒÂ² Ã— Xáµ¢)")
print("-" * 80)

X_hetero = np.column_stack([np.ones(n), np.random.uniform(1, 10, n)])  # X â‰¥ 1 (for heteroscedasticity)

beta_ols_hetero = np.zeros((n_simulations, 2))
beta_wls_hetero = np.zeros((n_simulations, 2))

for sim in range(n_simulations):
    # Heteroscedastic errors: Var(Îµáµ¢) = ÏƒÂ² Ã— X_i (increasing with X)
    epsilon_hetero = np.random.normal(0, 1, n) * np.sqrt(X_hetero[:, 1])  # Ïƒáµ¢ = âˆšXáµ¢
    Y_hetero = X_hetero @ beta_true + epsilon_hetero
    
    # OLS (ignores heteroscedasticity)
    beta_ols_hetero[sim] = inv(X_hetero.T @ X_hetero) @ (X_hetero.T @ Y_hetero)
    
    # WLS (correct weights: wáµ¢ = 1/Ïƒáµ¢Â² = 1/Xáµ¢)
    W = np.diag(1 / X_hetero[:, 1])  # Weight matrix
    beta_wls_hetero[sim] = inv(X_hetero.T @ W @ X_hetero) @ (X_hetero.T @ W @ Y_hetero)

# Compare OLS vs. WLS
mean_ols_hetero = beta_ols_hetero.mean(axis=0)
mean_wls_hetero = beta_wls_hetero.mean(axis=0)
var_ols_hetero = beta_ols_hetero.var(axis=0, ddof=1)
var_wls_hetero = beta_wls_hetero.var(axis=0, ddof=1)

print(f"\nEstimator Performance under Heteroscedasticity:")
print(f"{'Estimator':<20} {'E[Î²Ì‚â‚€]':<12} {'E[Î²Ì‚â‚]':<12} {'Var(Î²Ì‚â‚€)':<12} {'Var(Î²Ì‚â‚)':<12} {'BLUE':<8}")
print("-" * 80)
print(f"{'True parameters':<20} {beta_true[0]:<12.4f} {beta_true[1]:<12.4f} {'-':<12} {'-':<12} {'-':<8}")
print(f"{'OLS (unweighted)':<20} {mean_ols_hetero[0]:<12.4f} {mean_ols_hetero[1]:<12.4f} "
      f"{var_ols_hetero[0]:<12.6f} {var_ols_hetero[1]:<12.6f} {'No':<8}")
print(f"{'WLS (wáµ¢=1/Xáµ¢)':<20} {mean_wls_hetero[0]:<12.4f} {mean_wls_hetero[1]:<12.4f} "
      f"{var_wls_hetero[0]:<12.6f} {var_wls_hetero[1]:<12.6f} {'Yes':<8}")
print()
print(f"Efficiency Gain (Var(OLS)/Var(WLS) - 1):")
print(f"  Î²Ì‚â‚€: {(var_ols_hetero[0]/var_wls_hetero[0] - 1)*100:.1f}%  |  Î²Ì‚â‚: {(var_ols_hetero[1]/var_wls_hetero[1] - 1)*100:.1f}%")
print(f"  â†’ WLS corrects heteroscedasticity, achieves BLUE (lower variance)")
print("=" * 80)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Homoscedasticity: OLS vs. Alternative
axes[0, 0].hist(beta_ols_homo[:, 1], bins=40, alpha=0.7, label='OLS (all data)', color='blue', density=True)
axes[0, 0].hist(beta_alt_homo[:, 1], bins=40, alpha=0.5, label='Alternative (2 obs)', color='red', density=True)
axes[0, 0].axvline(beta_true[1], color='green', linestyle='--', linewidth=2, label='True Î²â‚')
axes[0, 0].set_xlabel('Î²Ì‚â‚ (Slope)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Gauss-Markov: OLS vs. Alternative (Homoscedastic)', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Variance comparison plot
axes[0, 1].bar(['OLS', 'Alternative'], [var_ols[1], var_alt[1]], color=['blue', 'red'], alpha=0.7)
axes[0, 1].set_ylabel('Variance of Î²Ì‚â‚', fontsize=11, fontweight='bold')
axes[0, 1].set_title(f'OLS is BLUE: {var_alt[1]/var_ols[1]:.1f}Ã— lower variance', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Heteroscedasticity: OLS vs. WLS
axes[1, 0].hist(beta_ols_hetero[:, 1], bins=40, alpha=0.7, label='OLS (inefficient)', color='orange', density=True)
axes[1, 0].hist(beta_wls_hetero[:, 1], bins=40, alpha=0.7, label='WLS (BLUE)', color='purple', density=True)
axes[1, 0].axvline(beta_true[1], color='green', linestyle='--', linewidth=2, label='True Î²â‚')
axes[1, 0].set_xlabel('Î²Ì‚â‚ (Slope)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Heteroscedasticity: OLS vs. WLS', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Efficiency gain bar plot
axes[1, 1].bar(['OLS', 'WLS'], [var_ols_hetero[1], var_wls_hetero[1]], color=['orange', 'purple'], alpha=0.7)
axes[1, 1].set_ylabel('Variance of Î²Ì‚â‚', fontsize=11, fontweight='bold')
axes[1, 1].set_title(f'WLS Efficiency: {(var_ols_hetero[1]/var_wls_hetero[1] - 1)*100:.1f}% lower variance', 
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('gauss_markov_efficiency.png', dpi=150)
plt.show()
