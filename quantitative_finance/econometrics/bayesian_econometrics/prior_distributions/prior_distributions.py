import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set seed
np.random.seed(42)

# Generate synthetic data
n = 50
X = np.random.normal(0, 1, n)
X_design = np.column_stack([np.ones(n), X])
true_beta = np.array([1.0, 2.0])
sigma_true = 0.5
y = X_design @ true_beta + np.random.normal(0, sigma_true, n)

# Frequentist OLS
XtX = X_design.T @ X_design
Xty = X_design.T @ y
beta_ols = np.linalg.solve(XtX, Xty)
residuals = y - X_design @ beta_ols
sigma_ols = np.sqrt(np.sum(residuals**2) / (n - 2))

print("="*70)
print("Prior Sensitivity Analysis: Bayesian Linear Regression")
print("="*70)
print(f"OLS estimates: Î²â‚€={beta_ols[0]:.4f}, Î²â‚={beta_ols[1]:.4f}, Ïƒ={sigma_ols:.4f}")
print("")

# Define prior specifications to test
priors = {
    'Weak (ÏƒÂ²=100)': {'Sigma_0': np.diag([100, 100])},
    'Moderate (ÏƒÂ²=10)': {'Sigma_0': np.diag([10, 10])},
    'Strong (ÏƒÂ²=1)': {'Sigma_0': np.diag([1, 1])},
    'Informative (Î¼=[1,2])': {'Sigma_0': np.diag([1, 1]), 'beta_0': np.array([1.0, 2.0])},
    'Misspecified (Î¼=[0,5])': {'Sigma_0': np.diag([1, 1]), 'beta_0': np.array([0.0, 5.0])},
}

# Compute posteriors for each prior
results = {}

for prior_name, prior_spec in priors.items():
    beta_0_prior = prior_spec.get('beta_0', np.array([0.0, 0.0]))
    Sigma_0_prior = prior_spec['Sigma_0']
    
    # Posterior (conjugate Normal-Normal)
    Sigma_0_inv = np.linalg.inv(Sigma_0_prior)
    Sigma_n_inv = Sigma_0_inv + XtX / (sigma_ols**2)
    Sigma_n = np.linalg.inv(Sigma_n_inv)
    
    beta_n = Sigma_n @ (Sigma_0_inv @ beta_0_prior + Xty / (sigma_ols**2))
    
    se_posterior = np.sqrt(np.diag(Sigma_n) * sigma_ols**2)
    
    results[prior_name] = {
        'beta_n': beta_n,
        'se': se_posterior,
        'Sigma_n': Sigma_n
    }
    
    print(f"{prior_name}:")
    print(f"  Î²â‚€ posterior mean: {beta_n[0]:.4f} Â± {1.96*se_posterior[0]:.4f}")
    print(f"  Î²â‚ posterior mean: {beta_n[1]:.4f} Â± {1.96*se_posterior[1]:.4f}")

print("\n" + "="*70)
print("Sensitivity Assessment:")
print("="*70)

# Extract Î²â‚ posteriors (main parameter of interest)
beta1_means = [results[name]['beta_n'][1] for name in priors.keys()]
beta1_ses = [results[name]['se'][1] for name in priors.keys()]

mean_beta1 = np.mean(beta1_means)
std_beta1 = np.std(beta1_means)
range_beta1 = np.max(beta1_means) - np.min(beta1_means)

print(f"\nÎ²â‚ Posterior Sensitivity:")
print(f"  Mean: {mean_beta1:.4f}")
print(f"  Std Dev: {std_beta1:.4f}")
print(f"  Range: {range_beta1:.4f}")
print(f"  CV (Coefficient of Variation): {std_beta1/mean_beta1*100:.2f}%")

if range_beta1 / mean_beta1 < 0.1:
    print(f"  Interpretation: ROBUST (range < 10% of mean)")
elif range_beta1 / mean_beta1 < 0.25:
    print(f"  Interpretation: MODERATE sensitivity (range 10-25% of mean)")
else:
    print(f"  Interpretation: SENSITIVE (range > 25% of mean)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Posterior means for Î²â‚ across priors
prior_names = list(priors.keys())
ax = axes[0, 0]
colors = ['blue' if 'Mis' not in name and 'Inform' not in name else 'red' if 'Mis' in name else 'green' 
          for name in prior_names]

bars = ax.bar(range(len(prior_names)), beta1_means, color=colors, alpha=0.6, edgecolor='black')
ax.axhline(beta_ols[1], color='black', linestyle='--', linewidth=2, label='OLS estimate')
ax.axhline(true_beta[1], color='green', linestyle='--', linewidth=2, label='True value')

# Add error bars
for i, (mean, se) in enumerate(zip(beta1_means, beta1_ses)):
    ax.plot([i, i], [mean - 1.96*se, mean + 1.96*se], 'k-', linewidth=2)

ax.set_ylabel('Î²â‚ Posterior Mean')
ax.set_title('Prior Sensitivity: Î²â‚ Posterior Estimates')
ax.set_xticks(range(len(prior_names)))
ax.set_xticklabels(prior_names, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Prior vs Posterior for Î²â‚ (moderate prior)
ax = axes[0, 1]
prior_name = 'Moderate (ÏƒÂ²=10)'
beta_0_prior = priors[prior_name].get('beta_0', np.array([0.0, 0.0]))
Sigma_0_prior = priors[prior_name]['Sigma_0']

# Prior distribution
x_range = np.linspace(beta_ols[1] - 2, beta_ols[1] + 2, 100)
prior_dist = stats.norm.pdf(x_range, beta_0_prior[1], np.sqrt(Sigma_0_prior[1, 1]))

# Posterior distribution
posterior_dist = stats.norm.pdf(x_range, results[prior_name]['beta_n'][1], 
                                results[prior_name]['se'][1])

ax.plot(x_range, prior_dist, 'b-', linewidth=2, label='Prior')
ax.plot(x_range, posterior_dist, 'r-', linewidth=2, label='Posterior')
ax.axvline(true_beta[1], color='green', linestyle='--', linewidth=2, label='True Î²â‚')
ax.set_xlabel('Î²â‚')
ax.set_ylabel('Density')
ax.set_title(f'Prior vs Posterior: {prior_name}')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Posterior uncertainty (SE across priors)
ax = axes[1, 0]
ax.bar(range(len(prior_names)), beta1_ses, color='purple', alpha=0.6, edgecolor='black')
ax.set_ylabel('Posterior SE (Î²â‚)')
ax.set_title('Posterior Uncertainty Across Priors')
ax.set_xticks(range(len(prior_names)))
ax.set_xticklabels(prior_names, rotation=45, ha='right', fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Prior vs Posterior credible intervals
ax = axes[1, 1]
y_pos = np.arange(len(prior_names))
credible_width = [1.96 * se for se in beta1_ses]

# Color code: misspecified vs correct
colors_cred = ['red' if 'Mis' in name else 'green' if 'Inform' in name else 'blue' 
               for name in prior_names]

for i, (mean, width, color) in enumerate(zip(beta1_means, credible_width, colors_cred)):
    ax.plot([mean - width/2, mean + width/2], [i, i], 'o-', linewidth=3, color=color, markersize=6)
    ax.plot(mean, i, 'o', markersize=8, color=color)

ax.axvline(beta_ols[1], color='black', linestyle='--', linewidth=2, label='OLS')
ax.set_yticks(y_pos)
ax.set_yticklabels(prior_names, fontsize=8)
ax.set_xlabel('Î²â‚')
ax.set_title('95% Posterior Credible Intervals for Î²â‚')
ax.legend()
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('prior_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Weak priors (large ÏƒÂ²): Posteriors dominated by likelihood â†’ Similar to OLS")
print("   â†’ Less regularization; higher posterior uncertainty")
print("")
print("2. Strong priors (small ÏƒÂ²): Posteriors pulled toward prior")
print("   â†’ More regularization; lower posterior uncertainty")
print("")
print("3. Misspecified informative prior: Posterior biased toward wrong prior")
print("   â†’ Small sample can't overcome strong wrong belief")
print("")
print("4. Correct informative prior: Posterior improved (lower SE)")
print("   â†’ When domain knowledge reliable; use informative priors")
print("")
print("5. Sensitivity robust: Most priors â†’ Similar posteriors (data informative)")
print("   â†’ Recommend reporting range of reasonable priors")
