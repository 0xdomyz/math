# Bayesian Inference in Econometrics

## 1. Concept Skeleton
**Definition:** Statistical framework updating prior beliefs with data via Bayes' theorem; posterior ∝ prior × likelihood; quantifies parameter uncertainty probabilistically  
**Purpose:** Incorporate prior information, regularization via priors, coherent uncertainty quantification, hierarchical modeling, sequential updating  
**Prerequisites:** Probability theory, Bayes' theorem, likelihood functions, conjugate distributions, integration techniques

## 2. Comparative Framing
| Method | Bayesian | Frequentist MLE | Bootstrap | Fiducial | Likelihood Principle | Empirical Bayes |
|--------|----------|-----------------|-----------|----------|---------------------|-----------------|
| **Parameter** | Random variable | Fixed unknown | Fixed unknown | Random | Fixed | Random (estimated) |
| **Uncertainty** | Posterior distribution | Sampling distribution | Resampling | Fiducial distribution | Likelihood function | Posterior with data-driven prior |
| **Prior** | Required (explicit) | None | None | None | None | Estimated from data |
| **Inference** | Credible intervals | Confidence intervals | Percentile CI | Fiducial limits | Profile likelihood | Shrinkage estimates |
| **Interpretation** | P(θ ∈ CI\|data) | P(CI contains θ\|repeat) | Empirical | Controversial | Relative support | Pragmatic |
| **Computation** | MCMC, VI | Optimization | Resampling | Analytical | Optimization | Two-stage |

## 3. Examples + Counterexamples

**Classic Example:**  
Linear regression with normal prior β~N(0, 10²I): Posterior mean shrinks OLS toward zero (Ridge-like). 95% credible interval [0.8, 1.6] directly interprets as "95% probability β ∈ [0.8, 1.6] given data." Frequentist CI same numerically but different interpretation.

**Failure Case:**  
Strong informative prior β~N(5, 0.1²) conflicts with data (MLE β̂=1.2). Posterior dominated by prior (posterior mean ≈ 4.8). Solution: Prior sensitivity analysis, robust priors, or increase prior variance.

**Edge Case:**  
Improper prior p(σ²) ∝ 1/σ² (Jeffreys) gives proper posterior for n>2 but undefined prior predictive. Useful for objective Bayes but conceptually problematic.

## 4. Layer Breakdown
```
Bayesian Inference Framework:
├─ Bayes' Theorem:
│   ├─ Posterior ∝ Prior × Likelihood
│   ├─ p(θ|Y) = p(Y|θ)·p(θ) / p(Y)
│   ├─ p(Y) = ∫ p(Y|θ)·p(θ) dθ (marginal likelihood/evidence)
│   └─ Normalizing constant often intractable
├─ Components:
│   ├─ Prior p(θ):
│   │   ├─ Encodes beliefs before seeing data
│   │   ├─ Informative: Strong beliefs (narrow distribution)
│   │   ├─ Weakly Informative: Mild regularization
│   │   ├─ Non-Informative: Vague/flat (Jeffreys, reference priors)
│   │   └─ Improper: ∫p(θ)dθ = ∞ (must yield proper posterior)
│   ├─ Likelihood p(Y|θ):
│   │   ├─ Same as frequentist likelihood
│   │   ├─ Probability of data given parameters
│   │   └─ Links model to observations
│   ├─ Posterior p(θ|Y):
│   │   ├─ Updated beliefs after seeing data
│   │   ├─ Complete inference about θ
│   │   └─ Point estimates: Mean, median, mode (MAP)
│   └─ Marginal Likelihood p(Y):
│       ├─ Evidence for model
│       ├─ Used in model comparison (Bayes factors)
│       └─ Computationally challenging
├─ Point Estimation:
│   ├─ Posterior Mean:
│   │   ├─ E[θ|Y] = ∫ θ·p(θ|Y) dθ
│   │   ├─ Minimizes MSE loss
│   │   └─ Optimal under squared error
│   ├─ Posterior Median:
│   │   ├─ 50th percentile of posterior
│   │   ├─ Minimizes absolute error loss
│   │   └─ Robust to outliers
│   ├─ Maximum A Posteriori (MAP):
│   │   ├─ mode(p(θ|Y))
│   │   ├─ Equivalent to penalized MLE
│   │   └─ Prior acts as regularization penalty
│   └─ Decision-Theoretic:
│       ├─ Minimize expected loss: δ = argmin E[L(θ,a)|Y]
│       └─ Different losses → different estimates
├─ Uncertainty Quantification:
│   ├─ Credible Intervals:
│   │   ├─ θ_L, θ_U: P(θ_L ≤ θ ≤ θ_U|Y) = 1-α
│   │   ├─ Direct probability statement (not frequentist)
│   │   ├─ Equal-tailed: α/2 in each tail
│   │   └─ Highest Posterior Density (HPD): Shortest interval
│   ├─ Posterior Standard Deviation:
│   │   ├─ SD(θ|Y) = √Var(θ|Y)
│   │   └─ Analogous to standard error
│   ├─ Posterior Variance:
│   │   ├─ Var(θ|Y) = E[θ²|Y] - (E[θ|Y])²
│   │   └─ Quantifies remaining uncertainty
│   └─ Predictive Distribution:
│       ├─ p(Ỹ|Y) = ∫ p(Ỹ|θ)·p(θ|Y) dθ
│       ├─ Integrates parameter uncertainty
│       └─ Prediction intervals naturally account for all sources
├─ Conjugate Priors:
│   ├─ Definition:
│   │   ├─ Prior and posterior from same family
│   │   ├─ Analytical tractability
│   │   └─ Closed-form updates
│   ├─ Normal-Normal:
│   │   ├─ Likelihood: Y|μ ~ N(μ, σ²) (σ² known)
│   │   ├─ Prior: μ ~ N(μ₀, τ₀²)
│   │   ├─ Posterior: μ|Y ~ N(μₙ, τₙ²)
│   │   ├─ μₙ = (τ₀⁻²μ₀ + nσ⁻²Ȳ)/(τ₀⁻² + nσ⁻²) (precision-weighted)
│   │   └─ τₙ² = 1/(τ₀⁻² + nσ⁻²)
│   ├─ Beta-Binomial:
│   │   ├─ Likelihood: Y|p ~ Binomial(n, p)
│   │   ├─ Prior: p ~ Beta(α, β)
│   │   ├─ Posterior: p|Y ~ Beta(α+Y, β+n-Y)
│   │   └─ Hyperparameters interpretable as pseudo-counts
│   ├─ Gamma-Poisson:
│   │   ├─ Likelihood: Y|λ ~ Poisson(λ)
│   │   ├─ Prior: λ ~ Gamma(α, β)
│   │   ├─ Posterior: λ|Y ~ Gamma(α+ΣYᵢ, β+n)
│   │   └─ Negative binomial marginal
│   ├─ Inverse-Gamma for Variance:
│   │   ├─ Likelihood: Y|σ² ~ N(μ, σ²)
│   │   ├─ Prior: σ² ~ InvGamma(α, β)
│   │   └─ Posterior: σ²|Y ~ InvGamma(α+n/2, β+SS/2)
│   └─ Normal-Inverse-Wishart (Multivariate):
│       ├─ Joint prior for (μ, Σ)
│       └─ Multivariate normal likelihood
├─ Non-Conjugate Cases:
│   ├─ Numerical Integration:
│   │   ├─ Grid approximation (low dimensions)
│   │   ├─ Quadrature methods
│   │   └─ Laplace approximation (Gaussian at mode)
│   ├─ Markov Chain Monte Carlo (MCMC):
│   │   ├─ Metropolis-Hastings algorithm
│   │   ├─ Gibbs sampling (conditional conjugacy)
│   │   ├─ Hamiltonian Monte Carlo (HMC)
│   │   └─ No-U-Turn Sampler (NUTS)
│   └─ Variational Inference:
│       ├─ Approximate posterior with simpler family
│       ├─ Minimize KL divergence
│       └─ Faster than MCMC, less accurate
├─ Linear Regression (Bayesian):
│   ├─ Model:
│   │   ├─ Y = Xβ + ε where ε ~ N(0, σ²I)
│   │   ├─ Likelihood: Y|β,σ² ~ N(Xβ, σ²I)
│   │   └─ Parameters: (β, σ²)
│   ├─ Conjugate Prior:
│   │   ├─ β|σ² ~ N(β₀, σ²V₀)
│   │   ├─ σ² ~ InvGamma(ν₀/2, s₀²ν₀/2)
│   │   └─ Normal-Inverse-Gamma conjugate family
│   ├─ Posterior (Conditional):
│   │   ├─ β|σ²,Y ~ N(β̃, σ²Vₙ)
│   │   ├─ Vₙ = (V₀⁻¹ + X'X)⁻¹
│   │   ├─ β̃ = Vₙ(V₀⁻¹β₀ + X'Y)
│   │   └─ σ²|Y ~ InvGamma(νₙ/2, sₙ²νₙ/2)
│   ├─ Special Cases:
│   │   ├─ Flat prior (V₀→∞I): β̃ = (X'X)⁻¹X'Y (OLS)
│   │   ├─ Ridge prior (V₀=λI): β̃ = (X'X+λI)⁻¹X'Y
│   │   └─ g-prior: V₀ = g(X'X)⁻¹ (unit information)
│   ├─ Predictive Distribution:
│   │   ├─ p(Ỹ|Y) = ∫∫ p(Ỹ|β,σ²)·p(β,σ²|Y) dβdσ²
│   │   ├─ Student-t distribution
│   │   └─ Wider than plug-in due to parameter uncertainty
│   └─ Marginal Likelihood:
│       ├─ p(Y) analytically available (conjugate)
│       └─ Used for model comparison
├─ Prior Elicitation:
│   ├─ Subjective:
│   │   ├─ Expert knowledge
│   │   ├─ Historical data
│   │   └─ Domain-specific constraints
│   ├─ Objective (Non-Informative):
│   │   ├─ Jeffreys Prior: p(θ) ∝ √|I(θ)| (Fisher information)
│   │   ├─ Reference Priors: Maximize expected information
│   │   ├─ Flat: p(θ) ∝ 1 (often improper)
│   │   └─ Maximum Entropy
│   ├─ Weakly Informative:
│   │   ├─ Regularization without strong beliefs
│   │   ├─ Example: β ~ N(0, 10²) (wide but not flat)
│   │   └─ Recommended for most applications
│   ├─ Data-Dependent:
│   │   ├─ Empirical Bayes: Estimate prior from data
│   │   ├─ Hierarchical models: Share information
│   │   └─ Not fully Bayesian (uses data twice)
│   └─ Robust Priors:
│       ├─ Heavy-tailed (Student-t, Cauchy)
│       ├─ Less sensitive to specification
│       └─ Mixture priors (spike-and-slab)
├─ Model Comparison:
│   ├─ Bayes Factors:
│   │   ├─ BF₁₂ = p(Y|M₁)/p(Y|M₂)
│   │   ├─ Ratio of marginal likelihoods
│   │   ├─ Interpretation: BF>10 strong evidence
│   │   └─ Automatic Occam's razor (penalizes complexity)
│   ├─ Posterior Odds:
│   │   ├─ Odds₁₂ = BF₁₂ × Prior_odds₁₂
│   │   └─ Updates prior model probabilities
│   ├─ Information Criteria:
│   │   ├─ DIC: Deviance Information Criterion
│   │   ├─ WAIC: Watanabe-Akaike IC (fully Bayesian)
│   │   └─ Lower is better
│   ├─ Posterior Predictive Checks:
│   │   ├─ Generate Y_rep ~ p(Y|θ⁽ⁱ⁾) from posterior draws
│   │   ├─ Compare to observed data
│   │   └─ Visual and test statistics
│   └─ Cross-Validation:
│       ├─ Leave-one-out CV (LOO-CV)
│       ├─ PSIS-LOO (Pareto-smoothed importance sampling)
│       └─ Approximate without refitting
├─ Computational Methods:
│   ├─ Gibbs Sampling:
│   │   ├─ Full conditionals: p(θⱼ|θ₋ⱼ, Y)
│   │   ├─ Iterate: θⱼ⁽ᵗ⁺¹⁾ ~ p(θⱼ|θ₋ⱼ⁽ᵗ⁾, Y)
│   │   ├─ Converges to joint posterior
│   │   └─ Effective when conditionals conjugate
│   ├─ Metropolis-Hastings:
│   │   ├─ Propose θ* ~ q(·|θ⁽ᵗ⁾)
│   │   ├─ Accept with prob α = min(1, [p(θ*|Y)·q(θ⁽ᵗ⁾|θ*)]/[p(θ⁽ᵗ⁾|Y)·q(θ*|θ⁽ᵗ⁾)])
│   │   ├─ Random walk or adaptive proposals
│   │   └─ General but can be slow
│   ├─ Hamiltonian Monte Carlo (HMC):
│   │   ├─ Uses gradient information
│   │   ├─ Explores posterior more efficiently
│   │   ├─ Requires differentiable log-posterior
│   │   └─ Stan uses NUTS variant
│   ├─ Convergence Diagnostics:
│   │   ├─ R̂ (Gelman-Rubin): Compare chains, want R̂<1.1
│   │   ├─ Effective Sample Size (ESS): Account for autocorrelation
│   │   ├─ Trace plots: Visual inspection
│   │   └─ Geweke diagnostic: Compare chain segments
│   └─ Burn-In & Thinning:
│       ├─ Discard initial samples (burn-in)
│       ├─ Thin to reduce autocorrelation (controversial)
│       └─ Long chains better than thinning
├─ Advantages:
│   ├─ Coherent Uncertainty:
│   │   ├─ Probability statements about parameters
│   │   └─ Natural prediction intervals
│   ├─ Incorporate Prior Information:
│   │   ├─ Expert knowledge
│   │   ├─ Small sample regularization
│   │   └─ Sequential updating
│   ├─ Hierarchical Modeling:
│   │   ├─ Partial pooling
│   │   ├─ Share strength across groups
│   │   └─ Natural shrinkage estimation
│   ├─ Missing Data:
│   │   ├─ Treat as latent variables
│   │   └─ MCMC imputes seamlessly
│   ├─ Model Averaging:
│   │   ├─ Weight models by posterior probabilities
│   │   └─ Automatic via Bayesian model averaging (BMA)
│   └─ Flexible Inference:
│       ├─ Any function of posterior
│       └─ Nonlinear transformations straightforward
├─ Limitations:
│   ├─ Prior Sensitivity:
│   │   ├─ Results depend on prior choice
│   │   ├─ Mitigate: Sensitivity analysis, robust priors
│   │   └─ Less issue with large data
│   ├─ Computational Cost:
│   │   ├─ MCMC can be slow (hours to days)
│   │   ├─ High-dimensional challenging
│   │   └─ Convergence diagnosis required
│   ├─ Subjective Elements:
│   │   ├─ Prior specification
│   │   └─ Loss function choice
│   ├─ Marginal Likelihood:
│   │   ├─ Often intractable
│   │   ├─ Harmonic mean estimator unstable
│   │   └─ Bridge sampling or thermodynamic integration
│   └─ Interpretation:
│       ├─ Frequentist properties not guaranteed
│       └─ Philosophical differences
└─ Applications in Econometrics:
    ├─ Panel Data: Hierarchical priors for unit effects
    ├─ Time Series: State-space models, BVAR
    ├─ Macroeconomics: DSGE estimation with priors
    ├─ Finance: Portfolio optimization, risk management
    ├─ Causal Inference: Incorporate expert knowledge
    ├─ High-Dimensional: Spike-and-slab for variable selection
    └─ Structural Models: Constrain parameters via priors
```

**Interaction:** Specify prior p(θ) → Observe data Y → Compute likelihood p(Y|θ) → Derive posterior p(θ|Y) ∝ p(Y|θ)·p(θ) → MCMC sampling → Summarize posterior

## 5. Mini-Project
Implement Bayesian linear regression with MCMC and compare to frequentist:
```python
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
print(f"  True β: {beta_true}")
print(f"  True σ: {sigma_true}")

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
    print(f"  β{i}: {b:7.4f} (SE={se:.4f}, true={b_true:.2f})")
    print(f"       95% CI: [{ci_l:.4f}, {ci_u:.4f}]")

print(f"\nσ: {sigma_ols:.4f} (true={sigma_true:.2f})")

# ===== Bayesian Inference with Conjugate Prior =====
print("\n" + "="*80)
print("BAYESIAN: CONJUGATE PRIOR (ANALYTICAL)")
print("="*80)

# Prior: β|σ² ~ N(β₀, σ²V₀), σ² ~ InvGamma(ν₀/2, s₀²ν₀/2)
beta_0 = np.zeros(p)  # Prior mean
V_0 = np.eye(p) * 100  # Prior covariance (weakly informative)
nu_0 = 2  # Prior degrees of freedom
s2_0 = 1.0  # Prior scale

print(f"Prior Specification:")
print(f"  β₀: {beta_0}")
print(f"  V₀: diag({np.diag(V_0)[0]}) (weakly informative)")
print(f"  ν₀: {nu_0}, s₀²: {s2_0}")

# Posterior parameters (conditional on σ²)
V_0_inv = np.linalg.inv(V_0)
XtX = X.T @ X
V_n = np.linalg.inv(V_0_inv + XtX)
beta_n = V_n @ (V_0_inv @ beta_0 + X.T @ Y)

# Posterior for σ² (marginal)
nu_n = nu_0 + n
SS = np.sum((Y - X @ beta_n)**2)
s2_n = (nu_0 * s2_0 + SS + 
        (beta_ols - beta_n).T @ XtX @ (beta_ols - beta_n)) / nu_n

print(f"\nPosterior (Analytical):")
print(f"  Posterior mean β: {beta_n}")
print(f"  Posterior variance scale: σ²Vₙ where Vₙ = (V₀⁻¹ + X'X)⁻¹")

# Posterior standard deviations (approximate)
post_var_beta = s2_n * np.diag(V_n)
post_sd_beta = np.sqrt(post_var_beta)

print(f"\nBayesian Point Estimates (Posterior Mean):")
for i, (b, b_true, sd) in enumerate(zip(beta_n, beta_true, post_sd_beta)):
    print(f"  β{i}: {b:7.4f} (SD={sd:.4f}, true={b_true:.2f})")

print(f"\nPosterior σ²: InvGamma({nu_n/2:.1f}, {s2_n*nu_n/2:.4f})")
post_mean_sigma2 = (s2_n * nu_n / 2) / (nu_n / 2 - 1)
post_sd_sigma2 = np.sqrt((s2_n * nu_n / 2)**2 / ((nu_n / 2 - 1)**2 * (nu_n / 2 - 2)))
print(f"  E[σ²|Y] = {post_mean_sigma2:.4f}")
print(f"  SD[σ²|Y] = {post_sd_sigma2:.4f}")

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
    # Sample β|σ², Y
    V_beta = sigma2_current * V_n
    mean_beta = beta_n
    beta_current = np.random.multivariate_normal(mean_beta, V_beta)
    beta_samples[t] = beta_current
    
    # Sample σ²|β, Y
    residuals = Y - X @ beta_current
    SS_current = np.sum(residuals**2)
    
    # Posterior: σ² ~ InvGamma((ν₀+n)/2, (ν₀s₀²+SS)/2)
    alpha_post = (nu_0 + n) / 2
    beta_post = (nu_0 * s2_0 + SS_current) / 2
    sigma2_current = 1 / np.random.gamma(alpha_post, 1/beta_post)
    sigma2_samples[t] = sigma2_current
    
    if (t + 1) % 2000 == 0:
        print(f"  Iteration {t+1}/{n_iter}")

print(f"✓ MCMC completed")

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
    print(f"  β{i}:")
    print(f"    Mean: {mean:7.4f}, Median: {median:7.4f}, SD: {sd:.4f}")
    print(f"    95% Credible: [{ci_l:.4f}, {ci_u:.4f}] (true={b_true:.2f})")

print(f"\n  σ:")
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
print(f"{'Prior':<25} {'β₀':>8} {'β₁':>8} {'β₂':>8}")
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
    print(f"  β{i}: {ess:.0f} / {len(beta_samples_post)} ({ess/len(beta_samples_post)*100:.1f}%)")
print(f"  σ: {ess_sigma:.0f} / {len(sigma_samples_post)} ({ess_sigma/len(sigma_samples_post)*100:.1f}%)")

# ===== Visualizations =====
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Plot 1-3: Trace plots for β
for i in range(p):
    axes[0, i].plot(beta_samples[:, i], alpha=0.7, linewidth=0.5)
    axes[0, i].axhline(beta_true[i], color='red', linestyle='--', 
                      linewidth=2, label='True')
    axes[0, i].axhline(beta_post_mean[i], color='blue', linestyle='-', 
                      linewidth=2, label='Posterior Mean')
    axes[0, i].axvline(burn_in, color='gray', linestyle=':', 
                      linewidth=1, label='Burn-in')
    axes[0, i].set_ylabel(f'β{i}')
    axes[0, i].set_xlabel('Iteration')
    axes[0, i].set_title(f'Trace Plot: β{i}')
    axes[0, i].legend(fontsize=8)
    axes[0, i].grid(alpha=0.3)

# Plot 4-6: Posterior distributions for β
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
    axes[1, i].set_xlabel(f'β{i}')
    axes[1, i].set_ylabel('Density')
    axes[1, i].set_title(f'Posterior: β{i}')
    axes[1, i].legend(fontsize=8)
    axes[1, i].grid(alpha=0.3)

# Plot 7: σ trace
axes[2, 0].plot(sigma_samples, alpha=0.7, linewidth=0.5)
axes[2, 0].axhline(sigma_true, color='red', linestyle='--', 
                  linewidth=2, label='True')
axes[2, 0].axhline(sigma_post_mean, color='blue', linestyle='-', 
                  linewidth=2, label='Posterior Mean')
axes[2, 0].axvline(burn_in, color='gray', linestyle=':', 
                  linewidth=1, label='Burn-in')
axes[2, 0].set_ylabel('σ')
axes[2, 0].set_xlabel('Iteration')
axes[2, 0].set_title('Trace Plot: σ')
axes[2, 0].legend(fontsize=8)
axes[2, 0].grid(alpha=0.3)

# Plot 8: σ posterior
axes[2, 1].hist(sigma_samples_post, bins=50, density=True, alpha=0.6)
axes[2, 1].axvline(sigma_true, color='red', linestyle='--', 
                  linewidth=2, label='True')
axes[2, 1].axvline(sigma_ols, color='orange', linestyle=':', 
                  linewidth=2, label='OLS')
axes[2, 1].set_xlabel('σ')
axes[2, 1].set_ylabel('Density')
axes[2, 1].set_title('Posterior: σ')
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
print(f"   Credible intervals ≈ Confidence intervals numerically")
print(f"   Interpretation differs: P(θ ∈ CI|data) vs P(CI contains θ)")

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
print("   • Incorporate prior knowledge (regularization)")
print("   • Natural prediction intervals")
print("   • Hierarchical models straightforward")
print("   • Sequential updating")
print("   • Missing data via data augmentation")

print("\n6. Practical Considerations:")
print("   ⚠ Prior specification requires thought")
print("   ⚠ MCMC diagnostics essential")
print("   ⚠ Computational cost (10k iterations here)")
print("   ⚠ Interpretation differences from frequentist")
print("   • Use weakly informative priors by default")
print("   • Perform prior sensitivity analysis")

print("\n7. Software Recommendations:")
print("   • Stan/PyStan: NUTS sampler, efficient HMC")
print("   • PyMC: Python-friendly, flexible")
print("   • JAGS: BUGS-like, Gibbs-based")
print("   • brms: R package, formula interface for Stan")
```

## 6. Challenge Round
When does Bayesian inference fail or mislead?
- **Prior-data conflict**: Strong informative prior contradicts data → Posterior between prior and likelihood; sensitivity analysis critical; robust priors (heavy-tailed) help
- **Improper posterior**: Improper prior + insufficient data → Non-integrable posterior; check propriety especially with flat priors and small n
- **MCMC non-convergence**: Complex posteriors with multimodality → Chains stuck in local modes; multiple chains, longer runs, better samplers (HMC)
- **Label switching**: Mixture models have permutation invariance → Post-processing needed; relabeling algorithms
- **Marginal likelihood sensitivity**: Bayes factors highly sensitive to prior for parameters not well-identified by data → Use BMA cautiously; prefer predictive performance
- **Computational intractability**: High dimensions or complex likelihoods → MCMC slow; variational inference approximates

## 7. Key References
- [Gelman et al. (2013) - Bayesian Data Analysis, 3rd Edition](http://www.stat.columbia.edu/~gelman/book/)
- [Robert (2007) - The Bayesian Choice](https://link.springer.com/book/10.1007/0-387-71599-1)
- [Koop (2003) - Bayesian Econometrics](https://www.wiley.com/en-us/Bayesian+Econometrics-p-9780470845677)

---
**Status:** Alternative paradigm to frequentist | **Complements:** MLE, MCMC, Hierarchical Models, Prior Elicitation
