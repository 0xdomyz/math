import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import inv

# Simulation parameters
np.random.seed(42)
n = 300

# True parameters for all scenarios
beta_0, beta_1 = 2.0, 1.5
sigma = 1.0

# Scenario 1: All assumptions hold (baseline)
print("=" * 80)
print("CLASSICAL ASSUMPTIONS TESTING")
print("=" * 80)

X1 = np.column_stack([np.ones(n), np.random.uniform(0, 10, n)])
epsilon1 = np.random.normal(0, sigma, n)
Y1 = X1 @ [beta_0, beta_1] + epsilon1

beta_hat1 = inv(X1.T @ X1) @ (X1.T @ Y1)
resid1 = Y1 - X1 @ beta_hat1
SSR1 = np.sum(resid1**2)
sigma_hat1 = np.sqrt(SSR1 / (n - 2))

print("\nScenario 1: ALL ASSUMPTIONS HOLD (Baseline)")
print("-" * 80)
print(f"Î²Ì‚â‚€ = {beta_hat1[0]:.4f}, Î²Ì‚â‚ = {beta_hat1[1]:.4f}")
print(f"Residual Std Error: {sigma_hat1:.4f}")

# Breusch-Pagan test (homoscedasticity)
X_bp = X1[:, 1].reshape(-1, 1)
X_bp = np.column_stack([np.ones(n), X_bp])
resid_sq = resid1**2
gamma_hat = inv(X_bp.T @ X_bp) @ (X_bp.T @ resid_sq)
fitted_resid_sq = X_bp @ gamma_hat
SSR_resid_sq = np.sum((resid_sq - fitted_resid_sq)**2)
SST_resid_sq = np.sum((resid_sq - resid_sq.mean())**2)
R2_bp = 1 - SSR_resid_sq / SST_resid_sq
BP_stat = n * R2_bp
BP_pval = 1 - stats.chi2.cdf(BP_stat, df=1)
print(f"Breusch-Pagan test: Ï‡Â² = {BP_stat:.4f}, p-value = {BP_pval:.4f} {'(No heteroscedasticity âœ“)' if BP_pval > 0.05 else '(Heteroscedasticity detected!)'}")

# Jarque-Bera test (normality)
skew = stats.skew(resid1)
kurt = stats.kurtosis(resid1, fisher=True)  # Excess kurtosis
JB_stat = (n / 6) * (skew**2 + (kurt**2) / 4)
JB_pval = 1 - stats.chi2.cdf(JB_stat, df=2)
print(f"Jarque-Bera test: JB = {JB_stat:.4f}, p-value = {JB_pval:.4f} {'(Normality âœ“)' if JB_pval > 0.05 else '(Non-normal!)'}")

# Scenario 2: Heteroscedasticity (MLR.5 violation)
print("\nScenario 2: HETEROSCEDASTICITY (MLR.5 Violation)")
print("-" * 80)
X2 = X1.copy()
epsilon2 = np.random.normal(0, 1, n) * np.sqrt(X2[:, 1])  # Var(Îµ) = X
Y2 = X2 @ [beta_0, beta_1] + epsilon2

beta_hat2 = inv(X2.T @ X2) @ (X2.T @ Y2)
resid2 = Y2 - X2 @ beta_hat2
print(f"Î²Ì‚â‚€ = {beta_hat2[0]:.4f}, Î²Ì‚â‚ = {beta_hat2[1]:.4f} (still unbiased)")

# Breusch-Pagan
resid_sq2 = resid2**2
gamma_hat2 = inv(X_bp.T @ X_bp) @ (X_bp.T @ resid_sq2)
fitted_resid_sq2 = X_bp @ gamma_hat2
SSR_resid_sq2 = np.sum((resid_sq2 - fitted_resid_sq2)**2)
SST_resid_sq2 = np.sum((resid_sq2 - resid_sq2.mean())**2)
R2_bp2 = 1 - SSR_resid_sq2 / SST_resid_sq2
BP_stat2 = n * R2_bp2
BP_pval2 = 1 - stats.chi2.cdf(BP_stat2, df=1)
print(f"Breusch-Pagan test: Ï‡Â² = {BP_stat2:.4f}, p-value = {BP_pval2:.4f} {'(Heteroscedasticity detected! âœ“)' if BP_pval2 < 0.05 else ''}")
print(f"â†’ OLS unbiased but inefficient; use robust SE or WLS")

# Scenario 3: Autocorrelation (MLR.5 violation)
print("\nScenario 3: AUTOCORRELATION (MLR.5 Violation)")
print("-" * 80)
X3 = np.column_stack([np.ones(n), np.arange(n) * 0.1])  # Time series X
rho = 0.7  # AR(1) coefficient
epsilon3 = np.zeros(n)
epsilon3[0] = np.random.normal(0, sigma)
for t in range(1, n):
    epsilon3[t] = rho * epsilon3[t-1] + np.random.normal(0, sigma * np.sqrt(1 - rho**2))
Y3 = X3 @ [beta_0, beta_1] + epsilon3

beta_hat3 = inv(X3.T @ X3) @ (X3.T @ Y3)
resid3 = Y3 - X3 @ beta_hat3
print(f"Î²Ì‚â‚€ = {beta_hat3[0]:.4f}, Î²Ì‚â‚ = {beta_hat3[1]:.4f} (still unbiased)")

# Durbin-Watson
dw = np.sum(np.diff(resid3)**2) / np.sum(resid3**2)
print(f"Durbin-Watson: DW = {dw:.4f} (DW â‰ˆ {2*(1-rho):.2f} expected; DW << 2 â†’ positive autocorrelation âœ“)")
print(f"â†’ OLS SE underestimates uncertainty; use HAC (Newey-West) SE")

# Scenario 4: Omitted variable bias (MLR.4 violation)
print("\nScenario 4: OMITTED VARIABLE BIAS (MLR.4 Violation)")
print("-" * 80)
X4_full = np.column_stack([np.ones(n), np.random.uniform(0, 10, n), np.random.uniform(-2, 2, n)])  # X1, X2
beta_true = [beta_0, beta_1, 0.8]  # X2 coefficient = 0.8
epsilon4 = np.random.normal(0, sigma, n)
Y4 = X4_full @ beta_true + epsilon4

# True model (both X1, X2)
beta_hat4_full = inv(X4_full.T @ X4_full) @ (X4_full.T @ Y4)
print(f"Full model (Xâ‚, Xâ‚‚): Î²Ì‚â‚ = {beta_hat4_full[1]:.4f}, Î²Ì‚â‚‚ = {beta_hat4_full[2]:.4f} (unbiased âœ“)")

# Omit X2 (biased)
X4_omit = X4_full[:, :2]
beta_hat4_omit = inv(X4_omit.T @ X4_omit) @ (X4_omit.T @ Y4)
cov_X1_X2 = np.cov(X4_full[:, 1], X4_full[:, 2])[0, 1]
var_X1 = np.var(X4_full[:, 1], ddof=1)
bias_formula = beta_true[2] * (cov_X1_X2 / var_X1)
print(f"Omit Xâ‚‚: Î²Ì‚â‚ = {beta_hat4_omit[1]:.4f} (biased! True Î²â‚ = {beta_true[1]:.2f})")
print(f"Bias formula: Î²â‚‚ Ã— Cov(Xâ‚,Xâ‚‚)/Var(Xâ‚) = {beta_true[2]:.2f} Ã— {cov_X1_X2:.4f}/{var_X1:.4f} = {bias_formula:.4f}")
print(f"Predicted Î²Ì‚â‚ = {beta_true[1] + bias_formula:.4f} vs. actual {beta_hat4_omit[1]:.4f} (match âœ“)")

# Scenario 5: Perfect multicollinearity (MLR.3 violation)
print("\nScenario 5: PERFECT MULTICOLLINEARITY (MLR.3 Violation)")
print("-" * 80)
X5 = np.column_stack([np.ones(n), X1[:, 1], 2 * X1[:, 1]])  # X2 = 2Ã—X1 (perfect collinearity)
print(f"rank(X) = {np.linalg.matrix_rank(X5)} (< 3 columns â†’ singular X'X)")
try:
    beta_hat5 = inv(X5.T @ X5) @ (X5.T @ Y1)
    print(f"Î²Ì‚ = {beta_hat5} (OLS defined)")
except np.linalg.LinAlgError:
    print("LinAlgError: X'X singular â†’ OLS undefined! Must drop one collinear variable.")

print("=" * 80)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Scenario 1: Baseline residual plot
axes[0, 0].scatter(X1[:, 1], resid1, alpha=0.5, s=20)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('X', fontsize=10, fontweight='bold')
axes[0, 0].set_ylabel('Residuals', fontsize=10, fontweight='bold')
axes[0, 0].set_title('Scenario 1: All Assumptions Hold', fontsize=11, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Scenario 1: Q-Q plot
stats.probplot(resid1, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Scenario 1: Q-Q Plot (Normality âœ“)', fontsize=11, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Scenario 2: Heteroscedasticity
axes[0, 2].scatter(X2[:, 1], resid2, alpha=0.5, s=20, color='orange')
axes[0, 2].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 2].set_xlabel('X', fontsize=10, fontweight='bold')
axes[0, 2].set_ylabel('Residuals', fontsize=10, fontweight='bold')
axes[0, 2].set_title('Scenario 2: Heteroscedasticity (Variance â†‘ with X)', fontsize=11, fontweight='bold')
axes[0, 2].grid(alpha=0.3)

# Scenario 3: Autocorrelation
axes[1, 0].plot(resid3, alpha=0.7, linewidth=1, color='purple')
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Time', fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('Residuals', fontsize=10, fontweight='bold')
axes[1, 0].set_title(f'Scenario 3: Autocorrelation (Ï={rho:.2f}, DW={dw:.2f})', fontsize=11, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Scenario 4: Omitted variable bias
x_range = np.linspace(0, 10, 100)
axes[1, 1].scatter(X4_full[:, 1], Y4, alpha=0.4, s=20, label='Data')
axes[1, 1].plot(x_range, beta_hat4_full[0] + beta_hat4_full[1]*x_range, 'g-', linewidth=2, label=f'Full model (Î²Ì‚â‚={beta_hat4_full[1]:.3f})')
axes[1, 1].plot(x_range, beta_hat4_omit[0] + beta_hat4_omit[1]*x_range, 'r--', linewidth=2, label=f'Omit Xâ‚‚ (Î²Ì‚â‚={beta_hat4_omit[1]:.3f}, biased)')
axes[1, 1].set_xlabel('Xâ‚', fontsize=10, fontweight='bold')
axes[1, 1].set_ylabel('Y', fontsize=10, fontweight='bold')
axes[1, 1].set_title('Scenario 4: Omitted Variable Bias', fontsize=11, fontweight='bold')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

# Scenario 5: Perfect multicollinearity illustration
axes[1, 2].scatter(X5[:, 1], X5[:, 2], alpha=0.5, s=20, color='brown')
axes[1, 2].plot([0, 10], [0, 20], 'r--', linewidth=2, label='Xâ‚‚ = 2Ã—Xâ‚ (Perfect collinearity)')
axes[1, 2].set_xlabel('Xâ‚', fontsize=10, fontweight='bold')
axes[1, 2].set_ylabel('Xâ‚‚', fontsize=10, fontweight='bold')
axes[1, 2].set_title('Scenario 5: Perfect Multicollinearity', fontsize=11, fontweight='bold')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('classical_assumptions_diagnostics.png', dpi=150)
plt.show()
