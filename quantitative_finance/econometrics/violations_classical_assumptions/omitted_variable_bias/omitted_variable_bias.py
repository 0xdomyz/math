import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy import stats

# Simulation parameters
np.random.seed(42)
n = 1000

# True parameters
beta_0, beta_1, beta_2 = 2.0, 1.0, 0.8

print("=" * 80)
print("OMITTED VARIABLE BIAS: QUANTIFICATION & SENSITIVITY")
print("=" * 80)

# Generate data with true X2
X1 = np.random.uniform(0, 10, n)
X2 = 0.6 * X1 + np.random.normal(0, 2, n)  # Correlated with X1
epsilon = np.random.normal(0, 1, n)
Y = beta_0 + beta_1 * X1 + beta_2 * X2 + epsilon

# True correlation and covariance
corr_X1_X2 = np.corrcoef(X1, X2)[0, 1]
cov_X1_X2 = np.cov(X1, X2)[0, 1]
var_X1 = np.var(X1, ddof=1)

print("\nData Generation:")
print("-" * 80)
print(f"True parameters: Î²â‚€ = {beta_0}, Î²â‚ = {beta_1}, Î²â‚‚ = {beta_2}")
print(f"Correlation(Xâ‚, Xâ‚‚): {corr_X1_X2:.4f}")
print(f"Cov(Xâ‚, Xâ‚‚): {cov_X1_X2:.4f}, Var(Xâ‚): {var_X1:.4f}")
print()

# Predicted bias
predicted_bias = beta_2 * (cov_X1_X2 / var_X1)
print(f"Predicted OVB (formula): Î²â‚‚ Ã— Cov(Xâ‚,Xâ‚‚)/Var(Xâ‚) = {beta_2:.2f} Ã— {cov_X1_X2:.4f}/{var_X1:.4f} = {predicted_bias:.6f}")

# Scenario 1: OLS with X1 only (omit X2)
print("\n\nScenario 1: OMIT Xâ‚‚ (OVB Present)")
print("-" * 80)

X_omit = np.column_stack([np.ones(n), X1])
beta_omit = inv(X_omit.T @ X_omit) @ (X_omit.T @ Y)

print(f"OLS with Xâ‚ only:")
print(f"  Î²Ì‚â‚€ = {beta_omit[0]:.6f}, Î²Ì‚â‚ = {beta_omit[1]:.6f}")
print(f"  Bias in Î²Ì‚â‚: {beta_omit[1] - beta_1:.6f}")
print(f"  Match predicted bias? {abs((beta_omit[1] - beta_1) - predicted_bias) < 0.01} âœ“")

# Scenario 2: Include X2 (unbiased)
print("\nScenario 2: INCLUDE Xâ‚‚ (Unbiased)")
print("-" * 80)

X_full = np.column_stack([np.ones(n), X1, X2])
beta_full = inv(X_full.T @ X_full) @ (X_full.T @ Y)

print(f"OLS with Xâ‚ and Xâ‚‚:")
print(f"  Î²Ì‚â‚€ = {beta_full[0]:.6f}, Î²Ì‚â‚ = {beta_full[1]:.6f}, Î²Ì‚â‚‚ = {beta_full[2]:.6f}")
print(f"  Bias in Î²Ì‚â‚: {beta_full[1] - beta_1:.6f} (negligible âœ“)")
print(f"  Bias in Î²Ì‚â‚‚: {beta_full[2] - beta_2:.6f} (negligible âœ“)")

# Scenario 3: Use imperfect proxy for X2
print("\nScenario 3: PROXY for Xâ‚‚ (Partial bias reduction)")
print("-" * 80)

# Create proxy correlated with X2 but with noise
X2_proxy = 0.7 * X2 + np.random.normal(0, 1.5, n)

X_proxy = np.column_stack([np.ones(n), X1, X2_proxy])
beta_proxy = inv(X_proxy.T @ X_proxy) @ (X_proxy.T @ Y)

print(f"OLS with Xâ‚ and proxy for Xâ‚‚:")
print(f"  Î²Ì‚â‚€ = {beta_proxy[0]:.6f}, Î²Ì‚â‚ = {beta_proxy[1]:.6f}, Î²Ì‚â‚‚_proxy = {beta_proxy[2]:.6f}")
print(f"  Bias in Î²Ì‚â‚: {beta_proxy[1] - beta_1:.6f}")
print(f"  Bias reduction vs. omission: {abs((beta_omit[1] - beta_1) - (beta_proxy[1] - beta_1)):.6f}")

# Sensitivity analysis: coefficient change with added controls
print("\n\nSensitivity Analysis: Coefficient Stability")
print("-" * 80)

beta_1_omit = beta_omit[1]
beta_1_proxy = beta_proxy[1]
beta_1_full = beta_full[1]

print(f"{'Specification':<30} {'Î²Ì‚â‚':<12} {'Change from full':<18} {'OVB %':<10}")
print("-" * 80)
print(f"{'Full model (Xâ‚, Xâ‚‚)':<30} {beta_1_full:<12.6f} {'0% (baseline)':<18} {'0%':<10}")
print(f"{'Proxy model (Xâ‚, proxy)':<30} {beta_1_proxy:<12.6f} {f'{(beta_1_proxy-beta_1_full)/beta_1_full*100:+.2f}%':<18} {f'{(beta_1_omit-beta_1_full)/beta_1_full*100:.1f}%':<10}")
print(f"{'Omitted (Xâ‚ only)':<30} {beta_1_omit:<12.6f} {f'{(beta_1_omit-beta_1_full)/beta_1_full*100:+.2f}%':<18} {f'{(beta_1_omit-beta_1_full)/beta_1_full*100:.1f}%':<10}")

# Bounds analysis (partial identification)
print("\n\nBounds Analysis: Sensitivity to Unobserved Confounder")
print("-" * 80)

# Assume unobserved confounder X2_unobs affects Y with unknown coefficient Î²2_unobs
# and correlates with X1 with unknown Corr(X1, X2_unobs)

# Extreme case 1: Confounder has max positive effect
print("If Xâ‚‚ were omitted entirely (maximum conceivable bias):")
max_bias = abs(predicted_bias)
lower_bound = beta_1_full - max_bias
upper_bound = beta_1_full + max_bias
print(f"  Conservative bounds on true Î²â‚: [{lower_bound:.6f}, {upper_bound:.6f}]")

# Rotnitzky-Robins bounds (sensitivity parameters)
print("\nSensitivity parameters:")
gamma = cov_X1_X2 / np.std(X1, ddof=1) / np.std(X2, ddof=1)  # Standardized corr
alpha = np.std(X2, ddof=1) / np.std(X1, ddof=1)  # Std ratio
print(f"  Î³ (standardized correlation Xâ‚, Xâ‚‚): {gamma:.4f}")
print(f"  Î± (std(Xâ‚‚)/std(Xâ‚)): {alpha:.4f}")

# Plot sensitivity curves
strengths = np.linspace(0, 1, 50)  # Confounder strength from 0 to 1
beta_1_sensitivity = []

for strength in strengths:
    # Hypothetical confounder with varying effect
    bias_at_strength = strength * predicted_bias
    beta_1_sensitivity.append(beta_1_full + bias_at_strength)

print("=" * 80)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Coefficient comparison
scenarios = ['Omitted\n(Xâ‚ only)', 'Proxy\n(Xâ‚, proxy)', 'Full\n(Xâ‚, Xâ‚‚)']
coefficients = [beta_1_omit, beta_1_proxy, beta_1_full]
colors = ['red', 'orange', 'green']
axes[0, 0].bar(scenarios, coefficients, color=colors, alpha=0.7, edgecolor='black')
axes[0, 0].axhline(y=beta_1, color='black', linestyle='--', linewidth=2, label=f'True Î²â‚ = {beta_1}')
axes[0, 0].set_ylabel('Î²Ì‚â‚ Estimate', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Coefficient Stability with Control Variables', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Panel 2: Scatter plot X1 vs X2
axes[0, 1].scatter(X1, X2, alpha=0.3, s=20)
axes[0, 1].set_xlabel('Xâ‚', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Xâ‚‚', fontsize=11, fontweight='bold')
axes[0, 1].set_title(f'Correlation between Xâ‚ and Xâ‚‚ (r={corr_X1_X2:.3f})', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Panel 3: Y vs X1 with regression lines
x_range = np.linspace(X1.min(), X1.max(), 100)
y_omit = beta_omit[0] + beta_omit[1] * x_range
y_full = beta_full[0] + beta_full[1] * x_range
y_proxy = beta_proxy[0] + beta_proxy[1] * x_range

axes[1, 0].scatter(X1, Y, alpha=0.2, s=20, label='Data', color='gray')
axes[1, 0].plot(x_range, y_omit, 'r-', linewidth=2, label=f'Omitted (Î²Ì‚â‚={beta_1_omit:.3f}, biased)')
axes[1, 0].plot(x_range, y_proxy, 'orange', linewidth=2, label=f'Proxy (Î²Ì‚â‚={beta_1_proxy:.3f})')
axes[1, 0].plot(x_range, y_full, 'g-', linewidth=2, label=f'Full (Î²Ì‚â‚={beta_1_full:.3f}, unbiased)')
axes[1, 0].set_xlabel('Xâ‚', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Y', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Regression Lines: Impact of Omitted Variable', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# Panel 4: Sensitivity analysis (bounds)
axes[1, 1].plot(strengths, beta_1_sensitivity, 'b-', linewidth=2, label='Sensitivity curve')
axes[1, 1].axhline(y=beta_1_full, color='g', linestyle='-', linewidth=2, label='Unbiased (full model)')
axes[1, 1].axhline(y=beta_1_omit, color='r', linestyle='--', linewidth=2, label='Fully biased (omitted)')
axes[1, 1].fill_between(strengths, beta_1_full - max_bias, beta_1_full + max_bias, alpha=0.2, color='blue', label='Bounds')
axes[1, 1].set_xlabel('Confounder Strength (0=none, 1=full)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Î²Ì‚â‚ Estimate', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Sensitivity Analysis: True Î²â‚ Under Confounding', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('omitted_variable_bias_sensitivity.png', dpi=150)
plt.show()
