import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(789)

# ===== Simulate Regression Data =====
print("="*80)
print("BAYESIAN INFERENCE IN ECONOMETRICS")
print("="*80)

n = 100
p = 3

X = np.column_stack([
    np.ones(n),
    np.random.randn(n),
    np.random.randn(n)
])

beta_true = np.array([2.0, 1.5, -0.8])
sigma_true = 1.0

Y = X @ beta_true + np.random.randn(n) * sigma_true

print(f"Simulation Setup:")
print(f"  Sample size: {n}")
print(f"  Parameters: {p}")
print(f"  True Î²: {beta_true}")
print(f"  True Ïƒ: {sigma_true}")

# ===== Frequentist OLS =====
print("\n" + "="*80)
print("FREQUENTIST OLS")
print("="*80)

beta_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
residuals_ols = Y - X @ beta_ols
sigma2_ols = np.sum(residuals_ols**2) / (n - p)
sigma_ols = np.sqrt(sigma2_ols)

# Standard errors
V_ols = sigma2_ols * np.linalg.inv(X.T @ X)
se_ols = np.sqrt(np.diag(V_ols))

# Confidence intervals
ci_ols_lower = beta_ols - 1.96 * se_ols
ci_ols_upper = beta_ols + 1.96 * se_ols

print(f"OLS Estimates:")
for i, (b, b_true, se, ci_l, ci_u) in enumerate(
    zip(beta_ols, beta_true, se_ols, ci_ols_lower, ci_ols_upper)
):
    print(f"  Î²{i}: {b:7.4f} (SE={se:.4f}, true={b_true:.2f})")
    print(f"       95% CI: [{ci_l:.4f}, {ci_u:.4f}]")

print(f"\nÏƒ: {sigma_ols:.4f} (true={sigma_true:.2f})")

# ===== Bayesian Inference with Conjugate Prior =====
print("\n" + "="*80)
print("BAYESIAN: CONJUGATE PRIOR (ANALYTICAL)")
print("="*80)

# Prior: Î²|ÏƒÂ² ~ N(Î²â‚€, ÏƒÂ²Vâ‚€), ÏƒÂ² ~ InvGamma(Î½â‚€/2, sâ‚€Â²Î½â‚€/2)
beta_0 = np.zeros(p)  # Prior mean
V_0 = np.eye(p) * 100  # Prior covariance (weakly informative)
nu_0 = 2  # Prior degrees of freedom
s2_0 = 1.0  # Prior scale

print(f"Prior Specification:")
print(f"  Î²â‚€: {beta_0}")
print(f"  Vâ‚€: diag({np.diag(V_0)[0]}) (weakly informative)")
print(f"  Î½â‚€: {nu_0}, sâ‚€Â²: {s2_0}")

# Posterior parameters (conditional on ÏƒÂ²)
V_0_inv = np.linalg.inv(V_0)
XtX = X.T @ X
V_n = np.linalg.inv(V_0_inv + XtX)
beta_n = V_n @ (V_0_inv @ beta_0 + X.T @ Y)

# Posterior for ÏƒÂ² (marginal)
nu_n = nu_0 + n
SS = np.sum((Y - X @ beta_n)**2)
s2_n = (nu_0 * s2_0 + SS + 
        (beta_ols - beta_n).T @ XtX @ (beta_ols - beta_n)) / nu_n

print(f"\nPosterior (Analytical):")
print(f"  Posterior mean Î²: {beta_n}")
print(f"  Posterior variance scale: ÏƒÂ²Vâ‚™ where Vâ‚™ = (Vâ‚€â»Â¹ + X'X)â»Â¹")

# Posterior standard deviations (approximate)
post_var_beta = s2_n * np.diag(V_n)
post_sd_beta = np.sqrt(post_var_beta)

print(f"\nBayesian Point Estimates (Posterior Mean):")
for i, (b, b_true, sd) in enumerate(zip(beta_n, beta_true, post_sd_beta)):
    print(f"  Î²{i}: {b:7.4f} (SD={sd:.4f}, true={b_true:.2f})")

print(f"\nPosterior ÏƒÂ²: InvGamma({nu_n/2:.1f}, {s2_n*nu_n/2:.4f})")
post_mean_sigma2 = (s2_n * nu_n / 2) / (nu_n / 2 - 1)
post_sd_sigma2 = np.sqrt((s2_n * nu_n / 2)**2 / ((nu_n / 2 - 1)**2 * (nu_n / 2 - 2)))
print(f"  E[ÏƒÂ²|Y] = {post_mean_sigma2:.4f}")
print(f"  SD[ÏƒÂ²|Y] = {post_sd_sigma2:.4f}")

# ===== Gibbs Sampling (MCMC) =====
print("\n" + "="*80)
print("BAYESIAN: GIBBS SAMPLING (MCMC)")
print("="*80)

n_iter = 10000
burn_in = 2000

# Storage
beta_samples = np.zeros((n_iter, p))
sigma2_samples = np.zeros(n_iter)

# Initialize
beta_current = beta_ols.copy()
sigma2_current = sigma2_ols

print(f"MCMC Settings:")
print(f"  Iterations: {n_iter}")
print(f"  Burn-in: {burn_in}")
print(f"  Posterior samples: {n_iter - burn_in}")

print(f"\nRunning Gibbs Sampler...")

for t in range(n_iter):
    # Sample Î²|ÏƒÂ², Y
    V_beta = sigma2_current * V_n
    mean_beta = beta_n
    beta_current = np.random.multivariate_normal(mean_beta, V_beta)
    beta_samples[t] = beta_current
    
    # Sample ÏƒÂ²|Î², Y
    residuals = Y - X @ beta_current
    SS_current = np.sum(residuals**2)
    
    # Posterior: ÏƒÂ² ~ InvGamma((Î½â‚€+n)/2, (Î½â‚€sâ‚€Â²+SS)/2)
    alpha_post = (nu_0 + n) / 2
    beta_post = (nu_0 * s2_0 + SS_current) / 2
    sigma2_current = 1 / np.random.gamma(alpha_post, 1/beta_post)
    sigma2_samples[t] = sigma2_current
    
    if (t + 1) % 2000 == 0:
        print(f"  Iteration {t+1}/{n_iter}")

print(f"âœ“ MCMC completed")

# Discard burn-in
beta_samples_post = beta_samples[burn_in:]
sigma2_samples_post = sigma2_samples[burn_in:]
sigma_samples_post = np.sqrt(sigma2_samples_post)

# Posterior summaries
beta_post_mean = np.mean(beta_samples_post, axis=0)
beta_post_sd = np.std(beta_samples_post, axis=0)
beta_post_median = np.median(beta_samples_post, axis=0)

# Credible intervals (95%)
beta_ci_lower = np.percentile(beta_samples_post, 2.5, axis=0)
beta_ci_upper = np.percentile(beta_samples_post, 97.5, axis=0)

sigma_post_mean = np.mean(sigma_samples_post)
sigma_post_sd = np.std(sigma_samples_post)
sigma_ci = np.percentile(sigma_samples_post, [2.5, 97.5])

print(f"\nPosterior Summaries (MCMC):")
for i, (mean, median, sd, ci_l, ci_u, b_true) in enumerate(
    zip(beta_post_mean, beta_post_median, beta_post_sd, 
        beta_ci_lower, beta_ci_upper, beta_true)
):
    print(f"  Î²{i}:")
    print(f"    Mean: {mean:7.4f}, Median: {median:7.4f}, SD: {sd:.4f}")
    print(f"    95% Credible: [{ci_l:.4f}, {ci_u:.4f}] (true={b_true:.2f})")

print(f"\n  Ïƒ:")
print(f"    Mean: {sigma_post_mean:.4f}, SD: {sigma_post_sd:.4f}")
print(f"    95% Credible: [{sigma_ci[0]:.4f}, {sigma_ci[1]:.4f}] (true={sigma_true:.2f})")

# ===== Prior Sensitivity Analysis =====
print("\n" + "="*80)
print("PRIOR SENSITIVITY ANALYSIS")
print("="*80)

# Compare different priors
priors = {
    'Flat (Improper)': {'V_0': np.eye(p) * 1e6, 'beta_0': np.zeros(p)},
    'Weakly Informative': {'V_0': np.eye(p) * 100, 'beta_0': np.zeros(p)},
    'Informative (Correct)': {'V_0': np.eye(p) * 1, 'beta_0': beta_true},
    'Informative (Wrong)': {'V_0': np.eye(p) * 1, 'beta_0': np.array([0, 0, 0])},
}

results_sensitivity = {}

for prior_name, prior_spec in priors.items():
    V_0_sens = prior_spec['V_0']
    beta_0_sens = prior_spec['beta_0']
    
    V_0_inv_sens = np.linalg.inv(V_0_sens)
    V_n_sens = np.linalg.inv(V_0_inv_sens + XtX)
    beta_n_sens = V_n_sens @ (V_0_inv_sens @ beta_0_sens + X.T @ Y)
    
    results_sensitivity[prior_name] = beta_n_sens

print(f"Posterior Means with Different Priors:")
print(f"{'Prior':<25} {'Î²â‚€':>8} {'Î²â‚':>8} {'Î²â‚‚':>8}")
print("-" * 55)
print(f"{'True':<25} {beta_true[0]:8.4f} {beta_true[1]:8.4f} {beta_true[2]:8.4f}")
print(f"{'OLS':<25} {beta_ols[0]:8.4f} {beta_ols[1]:8.4f} {beta_ols[2]:8.4f}")
for prior_name, beta_sens in results_sensitivity.items():
    print(f"{prior_name:<25} {beta_sens[0]:8.4f} {beta_sens[1]:8.4f} {beta_sens[2]:8.4f}")

# ===== Posterior Predictive Distribution =====
print("\n" + "="*80)
print("POSTERIOR PREDICTIVE DISTRIBUTION")
print("="*80)

# New observation
X_new = np.array([1, 0.5, -0.3])

# Frequentist prediction
y_pred_freq = X_new @ beta_ols
se_pred_freq = sigma_ols * np.sqrt(1 + X_new @ np.linalg.inv(XtX) @ X_new)
ci_pred_freq = [y_pred_freq - 1.96 * se_pred_freq, 
                y_pred_freq + 1.96 * se_pred_freq]

# Bayesian predictive
y_pred_samples = X_new @ beta_samples_post.T + \
                 np.random.randn(len(beta_samples_post)) * sigma_samples_post

y_pred_mean = np.mean(y_pred_samples)
y_pred_sd = np.std(y_pred_samples)
y_pred_ci = np.percentile(y_pred_samples, [2.5, 97.5])

# True value
y_true_new = X_new @ beta_true

print(f"Prediction for X_new = {X_new}:")
print(f"  True Y: {y_true_new:.4f}")
print(f"\n  Frequentist:")
print(f"    Point: {y_pred_freq:.4f}, SE: {se_pred_freq:.4f}")
print(f"    95% PI: [{ci_pred_freq[0]:.4f}, {ci_pred_freq[1]:.4f}]")
print(f"\n  Bayesian:")
print(f"    Mean: {y_pred_mean:.4f}, SD: {y_pred_sd:.4f}")
print(f"    95% PI: [{y_pred_ci[0]:.4f}, {y_pred_ci[1]:.4f}]")

# ===== Convergence Diagnostics =====
print("\n" + "="*80)
print("MCMC CONVERGENCE DIAGNOSTICS")
print("="*80)

# Effective sample size (simple autocorrelation-based)
def effective_sample_size(samples):
    """Estimate ESS using autocorrelation"""
    n = len(samples)
    centered = samples - np.mean(samples)
    acf = np.correlate(centered, centered, mode='full')[n-1:] / np.var(samples) / n
    
    # Sum until autocorrelation drops
    tau = 1
    for k in range(1, min(len(acf), n//2)):
        if acf[k] < 0.05:
            break
        tau += 2 * acf[k]
    
    ess = n / tau
    return ess

ess_beta = [effective_sample_size(beta_samples_post[:, i]) for i in range(p)]
ess_sigma = effective_sample_size(sigma_samples_post)

print(f"Effective Sample Size:")
for i, ess in enumerate(ess_beta):
    print(f"  Î²{i}: {ess:.0f} / {len(beta_samples_post)} ({ess/len(beta_samples_post)*100:.1f}%)")
print(f"  Ïƒ: {ess_sigma:.0f} / {len(sigma_samples_post)} ({ess_sigma/len(sigma_samples_post)*100:.1f}%)")

# ===== Visualizations =====
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Plot 1-3: Trace plots for Î²
for i in range(p):
    axes[0, i].plot(beta_samples[:, i], alpha=0.7, linewidth=0.5)
    axes[0, i].axhline(beta_true[i], color='red', linestyle='--', 
                      linewidth=2, label='True')
    axes[0, i].axhline(beta_post_mean[i], color='blue', linestyle='-', 
                      linewidth=2, label='Posterior Mean')
    axes[0, i].axvline(burn_in, color='gray', linestyle=':', 
                      linewidth=1, label='Burn-in')
    axes[0, i].set_ylabel(f'Î²{i}')
    axes[0, i].set_xlabel('Iteration')
    axes[0, i].set_title(f'Trace Plot: Î²{i}')
    axes[0, i].legend(fontsize=8)
    axes[0, i].grid(alpha=0.3)

# Plot 4-6: Posterior distributions for Î²
for i in range(p):
    axes[1, i].hist(beta_samples_post[:, i], bins=50, density=True, 
                   alpha=0.6, label='Posterior')
    
    # Overlay normal approximation
    x_grid = np.linspace(beta_samples_post[:, i].min(), 
                        beta_samples_post[:, i].max(), 200)
    pdf_normal = stats.norm.pdf(x_grid, beta_post_mean[i], beta_post_sd[i])
    axes[1, i].plot(x_grid, pdf_normal, 'r-', linewidth=2, 
                   label='Normal Approx')
    
    axes[1, i].axvline(beta_true[i], color='green', linestyle='--', 
                      linewidth=2, label='True')
    axes[1, i].axvline(beta_ols[i], color='orange', linestyle=':', 
                      linewidth=2, label='OLS')
    axes[1, i].set_xlabel(f'Î²{i}')
    axes[1, i].set_ylabel('Density')
    axes[1, i].set_title(f'Posterior: Î²{i}')
    axes[1, i].legend(fontsize=8)
    axes[1, i].grid(alpha=0.3)

# Plot 7: Ïƒ trace
axes[2, 0].plot(sigma_samples, alpha=0.7, linewidth=0.5)
axes[2, 0].axhline(sigma_true, color='red', linestyle='--', 
                  linewidth=2, label='True')
axes[2, 0].axhline(sigma_post_mean, color='blue', linestyle='-', 
                  linewidth=2, label='Posterior Mean')
axes[2, 0].axvline(burn_in, color='gray', linestyle=':', 
                  linewidth=1, label='Burn-in')
axes[2, 0].set_ylabel('Ïƒ')
axes[2, 0].set_xlabel('Iteration')
axes[2, 0].set_title('Trace Plot: Ïƒ')
axes[2, 0].legend(fontsize=8)
axes[2, 0].grid(alpha=0.3)

# Plot 8: Ïƒ posterior
axes[2, 1].hist(sigma_samples_post, bins=50, density=True, alpha=0.6)
axes[2, 1].axvline(sigma_true, color='red', linestyle='--', 
                  linewidth=2, label='True')
axes[2, 1].axvline(sigma_ols, color='orange', linestyle=':', 
                  linewidth=2, label='OLS')
axes[2, 1].set_xlabel('Ïƒ')
axes[2, 1].set_ylabel('Density')
axes[2, 1].set_title('Posterior: Ïƒ')
axes[2, 1].legend(fontsize=8)
axes[2, 1].grid(alpha=0.3)

# Plot 9: Predictive distribution
axes[2, 2].hist(y_pred_samples, bins=50, density=True, alpha=0.6, 
               label='Bayesian Predictive')
x_pred_grid = np.linspace(y_pred_samples.min(), y_pred_samples.max(), 200)
pdf_freq = stats.norm.pdf(x_pred_grid, y_pred_freq, se_pred_freq)
axes[2, 2].plot(x_pred_grid, pdf_freq, 'r-', linewidth=2, 
               label='Frequentist')
axes[2, 2].axvline(y_true_new, color='green', linestyle='--', 
                  linewidth=2, label='True')
axes[2, 2].set_xlabel('Predicted Y')
axes[2, 2].set_ylabel('Density')
axes[2, 2].set_title('Posterior Predictive Distribution')
axes[2, 2].legend(fontsize=8)
axes[2, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_inference_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Frequentist vs Bayesian:")
print(f"   OLS and Bayesian posterior mean similar (weak prior)")
print(f"   Credible intervals â‰ˆ Confidence intervals numerically")
print(f"   Interpretation differs: P(Î¸ âˆˆ CI|data) vs P(CI contains Î¸)")

print("\n2. Prior Influence:")
print(f"   Weakly informative: Minimal impact on posterior")
print(f"   Strong informative (wrong): Pulls estimates away from truth")
print(f"   Data overwhelms prior as n increases")

print("\n3. Uncertainty Quantification:")
print(f"   Posterior distribution: Full characterization")
print(f"   Credible intervals straightforward")
print(f"   Predictive intervals account for parameter uncertainty")

print("\n4. MCMC Convergence:")
print(f"   Trace plots show mixing")
print(f"   ESS: {np.mean(ess_beta):.0f} effective samples ({np.mean(ess_beta)/len(beta_samples_post)*100:.0f}%)")
print(f"   Burn-in: {burn_in} iterations discarded")

print("\n5. Practical Advantages:")
print("   â€¢ Incorporate prior knowledge (regularization)")
print("   â€¢ Natural prediction intervals")
print("   â€¢ Hierarchical models straightforward")
print("   â€¢ Sequential updating")
print("   â€¢ Missing data via data augmentation")

print("\n6. Practical Considerations:")
print("   âš  Prior specification requires thought")
print("   âš  MCMC diagnostics essential")
print("   âš  Computational cost (10k iterations here)")
print("   âš  Interpretation differences from frequentist")
print("   â€¢ Use weakly informative priors by default")
print("   â€¢ Perform prior sensitivity analysis")

print("\n7. Software Recommendations:")
print("   â€¢ Stan/PyStan: NUTS sampler, efficient HMC")
print("   â€¢ PyMC: Python-friendly, flexible")
print("   â€¢ JAGS: BUGS-like, Gibbs-based")
print("   â€¢ brms: R package, formula interface for Stan")
