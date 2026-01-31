import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set seed
np.random.seed(42)

# Generate hierarchical data
J = 30  # Number of schools
n_students = 50  # Students per school

# True hyperparameters
mu_true = 70  # Overall mean score
sigma_school_true = 5  # Between-school SD
sigma_within_true = 10  # Within-school SD

# Generate school effects
school_effects = np.random.normal(mu_true, sigma_school_true, J)

# Generate student scores
y = []
school_idx = []

for j in range(J):
    y_j = np.random.normal(school_effects[j], sigma_within_true, n_students)
    y.extend(y_j)
    school_idx.extend([j] * n_students)

y = np.array(y)
school_idx = np.array(school_idx)
n_total = len(y)

print("="*70)
print("Hierarchical Model: Test Scores Across Schools")
print("="*70)
print(f"Schools: {J}")
print(f"Students per school: {n_students}")
print(f"Total students: {n_total}")
print(f"True hyperparameters:")
print(f"  Overall mean: {mu_true}")
print(f"  Between-school SD: {sigma_school_true}")
print(f"  Within-school SD: {sigma_within_true}")
print("")

# Fit hierarchical model (simplified Bayesian via conjugate priors)
# Prior: Î¼ ~ N(70, 10Â²), ÏƒÂ²_school ~ IG(2, 1), ÏƒÂ²_within ~ IG(2, 50)

# Estimate school effects using empirical Bayes (simpler than full MCMC)
# Posterior: Î±_j | Î¼, ÏƒÂ²_school, ÏƒÂ²_within is normal (mixture of data & prior)

# Step 1: Estimate ÏƒÂ²_within (within-school variance)
within_vars = []
for j in range(J):
    y_j = y[school_idx == j]
    var_j = np.var(y_j, ddof=1)
    within_vars.append(var_j)

sigma_within_est = np.sqrt(np.mean(within_vars))

# Step 2: Estimate school means
school_means = np.array([y[school_idx == j].mean() for j in range(J)])

# Step 3: Estimate between-school variance (method of moments)
grand_mean = y.mean()
between_var_est = np.var(school_means, ddof=1) - sigma_within_est**2 / n_students
sigma_school_est = np.sqrt(max(between_var_est, 0.01))  # Ensure positive

mu_est = grand_mean

print("Empirical Bayes Estimation:")
print("-"*70)
print(f"Estimated overall mean: {mu_est:.2f}")
print(f"Estimated between-school SD: {sigma_school_est:.2f}")
print(f"Estimated within-school SD: {sigma_within_est:.2f}")
print("")

# Compute posterior school effects (hierarchical shrinkage)
shrinkage_factor = (sigma_school_est**2) / (sigma_school_est**2 + sigma_within_est**2 / n_students)
alpha_posterior = mu_est + shrinkage_factor * (school_means - mu_est)
se_posterior = np.sqrt(sigma_school_est**2 * (1 - shrinkage_factor))

# Compare methods
print("Posterior Estimates (First 10 Schools):")
print("-"*70)
print(f"{'School':<8} {'No Pooling':<15} {'Pooled':<15} {'Hierarchical':<15} {'SE':<10}")
print("-"*70)

for j in range(10):
    no_pool = school_means[j]
    pool = mu_est
    hier = alpha_posterior[j]
    se = se_posterior[j]
    print(f"{j+1:<8} {no_pool:>13.2f}   {pool:>13.2f}   {hier:>13.2f}   {se:>8.2f}")

# Shrinkage analysis
print("\n" + "="*70)
print("Shrinkage Analysis:")
print("-"*70)
print(f"Shrinkage factor: {shrinkage_factor:.3f}")
print(f"  â†’ Each school estimate pulled {shrinkage_factor*100:.1f}% toward overall mean")
print(f"Posterior SE (hierarchy): {np.mean(se_posterior):.2f}")
print(f"No-pooling SE (group-specific): ~{sigma_within_est/np.sqrt(n_students):.2f}")
print(f"  â†’ Hierarchy reduces uncertainty by pooling")
print("")

# Variance decomposition
icc = sigma_school_est**2 / (sigma_school_est**2 + sigma_within_est**2)
print(f"Intraclass Correlation (ICC): {icc:.3f}")
print(f"  â†’ {icc*100:.1f}% of variance is between schools")
print(f"  â†’ {(1-icc)*100:.1f}% of variance is within schools")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: School effects comparison
ax = axes[0, 0]
schools = np.arange(J)
colors = ['red' if abs(sm - mu_est) > 2*se for sm, se in zip(school_means, se_posterior)] else 'blue' for sm, se in zip(school_means, se_posterior)]

ax.scatter(schools, school_means, alpha=0.5, s=50, color='blue', label='No-pooling (group-specific)')
ax.scatter(schools, alpha_posterior, alpha=0.7, s=50, color='red', marker='^', label='Hierarchical (shrunk)')
ax.axhline(mu_est, color='black', linestyle='--', linewidth=2, label='Estimated mean')
ax.axhline(mu_true, color='green', linestyle=':', linewidth=2, label='True mean')

# Add error bars
for j in range(J):
    ax.plot([j, j], [alpha_posterior[j] - 1.96*se_posterior[j], alpha_posterior[j] + 1.96*se_posterior[j]], 
           'r-', linewidth=1, alpha=0.3)

ax.set_xlabel('School')
ax.set_ylabel('Mean Test Score')
ax.set_title('School Effects: No-Pooling vs Hierarchical')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: Shrinkage visualization
ax = axes[0, 1]
ax.scatter(school_means, alpha_posterior, alpha=0.6, s=50, color='blue')

# Reference lines
min_val = min(school_means.min(), alpha_posterior.min())
max_val = max(school_means.max(), alpha_posterior.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5, label='No shrinkage')
ax.axhline(mu_est, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(mu_est, color='red', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('No-Pooling Estimate')
ax.set_ylabel('Hierarchical Posterior')
ax.set_title('Shrinkage: Points off diagonal show pulling toward mean')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Distribution of school effects
ax = axes[1, 0]
ax.hist(school_means - mu_est, bins=15, alpha=0.5, label='No-pooling residuals', color='blue')
ax.hist(alpha_posterior - mu_est, bins=15, alpha=0.5, label='Hierarchical residuals', color='red')

# Overlay true distribution
x_range = np.linspace(-20, 20, 100)
true_dist = stats.norm.pdf(x_range, 0, sigma_school_true)
ax.plot(x_range, true_dist * J * 3, 'g-', linewidth=2, label='True between-school dist')

ax.set_xlabel('Deviation from Mean')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of School Effects')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 4: Posterior SD by school (uncertainty)
ax = axes[1, 1]
ax.scatter(school_means, se_posterior, alpha=0.6, s=50, color='purple')
ax.axhline(np.mean(se_posterior), color='red', linestyle='--', linewidth=2, label='Average posterior SE')

# Color by school size effect (though all equal in this case)
ax.set_xlabel('School Mean (No-Pooling)')
ax.set_ylabel('Posterior SE (Hierarchical)')
ax.set_title('Uncertainty in School Effects')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hierarchical_model.png', dpi=300, bbox_inches='tight')
plt.show()

# Cross-validation: Predict for new school
print("\n" + "="*70)
print("Prediction for New School (Not in Data):")
print("-"*70)
new_school_mean_posterior = np.random.normal(mu_est, sigma_school_est)
new_students = np.random.normal(new_school_mean_posterior, sigma_within_est, 10)

print(f"Predicted new school effect: {new_school_mean_posterior:.2f}")
print(f"Expected new school mean: â‰ˆ {mu_est:.2f} (with uncertainty {sigma_school_est:.2f})")
print(f"10 new student predictions: {new_students.round(1)}")
print(f"Average of 10 new students: {new_students.mean():.2f}")

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Shrinkage: Extreme school estimates pulled toward mean")
print("   â†’ Reduces overfitting; improves out-of-sample predictions")
print("")
print("2. Partial pooling: Uses both within-school and between-school info")
print("   â†’ Balances group-specific estimates with shared structure")
print("")
print("3. ICC indicates: Most variation within schools, less between")
print("   â†’ But hierarchy still beneficial (regularization)")
print("")
print("4. New school prediction uses hyperprior (Î¼, ÏƒÂ²_school)")
print("   â†’ Accounts for uncertainty about where new school falls")
