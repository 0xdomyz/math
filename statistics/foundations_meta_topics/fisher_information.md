# Fisher Information

## 1.1 Concept Skeleton
**Definition:** Measure of information observable data carries about unknown parameter; curvature of log-likelihood  
**Purpose:** Quantify precision of parameter estimates, derive asymptotic variance, design experiments  
**Prerequisites:** Likelihood function, calculus (derivatives), probability theory

## 1.2 Comparative Framing
| Concept | Fisher Information I(θ) | Variance Var(θ̂) | Cramér-Rao Bound |
|---------|----------------------|----------------|-----------------|
| **Meaning** | Expected information | Estimator uncertainty | Minimum achievable variance |
| **Formula** | -E[d²ℓ/dθ²] | E[(θ̂-θ)²] | 1/I(θ) |
| **Use** | Measure data quality | Assess estimator | Lower bound for any unbiased estimator |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Bernoulli(p), n trials: I(p) = n/(p(1-p)). Maximum information at p=0.5 (most uncertainty)

**Failure Case:**  
Uniform distribution U(0,θ): Fisher information undefined (discontinuous likelihood at boundary)

**Edge Case:**  
Large n: I(θ) → ∞, Var(θ̂) → 0 (perfect information asymptotically)

## 1.4 Layer Breakdown
```
Fisher Information Framework:
├─ Single Parameter:
│   ├─ Definition: I(θ) = -E[d²ℓ(θ)/dθ²]
│   ├─ Alternative: I(θ) = E[(dℓ(θ)/dθ)²]
│   ├─ Relation: ℓ(θ) = log L(θ) (log-likelihood)
│   └─ Units: [θ]⁻²
├─ Sample Fisher Information:
│   ├─ Observed: Iₒ(θ) = -d²ℓ(θ)/dθ²
│   ├─ Expected: I(θ) = E[Iₒ(θ)]
│   └─ For n iid observations: Iₙ(θ) = n·I₁(θ)
├─ Multiple Parameters:
│   ├─ Information Matrix: I(θ) = -E[H(ℓ)]
│   ├─ H: Hessian matrix of second derivatives
│   └─ Inverse: I(θ)⁻¹ = asymptotic covariance matrix
├─ Cramér-Rao Lower Bound:
│   ├─ Statement: Var(θ̂) ≥ 1/I(θ) for unbiased estimators
│   ├─ Efficient: Var(θ̂) = 1/I(θ) (achieves bound)
│   └─ MLE: Asymptotically efficient (achieves bound as n→∞)
├─ Properties:
│   ├─ Additivity: I(θ₁,...,θₙ) = Σ I(θᵢ) for independent data
│   ├─ Reparameterization: I(η) = I(θ)·(dθ/dη)²
│   └─ Large sample: √n·(θ̂ - θ) → N(0, 1/I(θ))
└─ Applications:
    ├─ Confidence intervals: θ̂ ± z_α/2 / √I(θ̂)
    ├─ Optimal design: Maximize I(θ) through data collection
    └─ Power analysis: Larger I(θ) → easier to detect effects
```

## 1.5 Mini-Project
Compute and visualize Fisher information:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

np.random.seed(42)

# Example 1: Bernoulli/Binomial
def fisher_bernoulli(p):
    """Fisher information for single Bernoulli trial"""
    if p <= 0 or p >= 1:
        return np.inf
    return 1 / (p * (1 - p))

def fisher_bernoulli_n(p, n):
    """Fisher information for n Bernoulli trials"""
    return n * fisher_bernoulli(p)

# Example 2: Normal with unknown mean
def fisher_normal_mu(sigma, n):
    """Fisher information for normal mean (known variance)"""
    return n / sigma**2

# Example 3: Normal with unknown variance
def fisher_normal_sigma(sigma, n):
    """Fisher information for normal variance (known mean)"""
    return 2 * n / sigma**4

# Simulate Bernoulli and verify Cramér-Rao bound
true_p = 0.3
n_trials = 100
n_sims = 1000

estimates = []
for _ in range(n_sims):
    data = np.random.binomial(1, true_p, n_trials)
    p_hat = np.mean(data)
    estimates.append(p_hat)

estimates = np.array(estimates)
empirical_var = np.var(estimates)
theoretical_var = 1 / fisher_bernoulli_n(true_p, n_trials)

print("Bernoulli Example (p=0.3, n=100):")
print(f"  Fisher Information: {fisher_bernoulli_n(true_p, n_trials):.3f}")
print(f"  Cramér-Rao bound (variance): {theoretical_var:.6f}")
print(f"  Empirical variance: {empirical_var:.6f}")
print(f"  Ratio (should be ≈1): {empirical_var/theoretical_var:.3f}")

# Normal example
true_mu = 5.0
true_sigma = 2.0
n_obs = 50

data_normal = np.random.normal(true_mu, true_sigma, n_obs)
mu_hat = np.mean(data_normal)

fisher_info_mu = fisher_normal_mu(true_sigma, n_obs)
se_mu = 1 / np.sqrt(fisher_info_mu)

print(f"\nNormal Example (μ={true_mu}, σ={true_sigma}, n={n_obs}):")
print(f"  Fisher Information for μ: {fisher_info_mu:.3f}")
print(f"  SE(μ̂) = 1/√I(μ): {se_mu:.3f}")
print(f"  Theoretical SE: σ/√n = {true_sigma/np.sqrt(n_obs):.3f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Fisher information for Bernoulli vs p
p_range = np.linspace(0.01, 0.99, 100)
I_p = [fisher_bernoulli(p) for p in p_range]

axes[0, 0].plot(p_range, I_p, 'b-', linewidth=2)
axes[0, 0].axvline(0.5, color='r', linestyle='--', label='Max I(p) at p=0.5')
axes[0, 0].set_xlabel('p')
axes[0, 0].set_ylabel('Fisher Information I(p)')
axes[0, 0].set_title('Fisher Information: Bernoulli(p)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Variance (Cramér-Rao bound) vs p
var_bound = [1/fisher_bernoulli(p) for p in p_range]

axes[0, 1].plot(p_range, var_bound, 'g-', linewidth=2)
axes[0, 1].set_xlabel('p')
axes[0, 1].set_ylabel('Minimum Variance')
axes[0, 1].set_title('Cramér-Rao Lower Bound: Bernoulli(p)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Information vs sample size
n_range = np.arange(10, 500, 10)
I_n = [fisher_bernoulli_n(0.3, n) for n in n_range]

axes[0, 2].plot(n_range, I_n, 'r-', linewidth=2)
axes[0, 2].set_xlabel('Sample Size n')
axes[0, 2].set_ylabel('Fisher Information')
axes[0, 2].set_title('Information vs Sample Size (p=0.3)')
axes[0, 2].grid(True, alpha=0.3)

# 4. Empirical vs theoretical variance
axes[1, 0].hist(estimates, bins=40, density=True, alpha=0.7, edgecolor='black', 
                label='Empirical')
x = np.linspace(estimates.min(), estimates.max(), 100)
axes[1, 0].plot(x, stats.norm.pdf(x, true_p, np.sqrt(theoretical_var)), 
                'r-', linewidth=2, label='Theoretical N(p, 1/I(p))')
axes[1, 0].axvline(true_p, color='g', linestyle='--', label='True p')
axes[1, 0].legend()
axes[1, 0].set_title('Distribution of p̂ (Bernoulli)')
axes[1, 0].set_xlabel('Estimated p')

# 5. Log-likelihood curvature (normal example)
mu_range = np.linspace(true_mu - 2, true_mu + 2, 100)
log_likelihood = []

for mu in mu_range:
    ll = np.sum(stats.norm.logpdf(data_normal, mu, true_sigma))
    log_likelihood.append(ll)

log_likelihood = np.array(log_likelihood)
# Numerical second derivative at MLE
idx_mle = np.argmax(log_likelihood)
h = mu_range[1] - mu_range[0]
second_deriv = (log_likelihood[idx_mle+1] - 2*log_likelihood[idx_mle] + 
                log_likelihood[idx_mle-1]) / h**2
observed_info = -second_deriv

axes[1, 1].plot(mu_range, log_likelihood - np.max(log_likelihood), 'b-', linewidth=2)
axes[1, 1].axvline(mu_hat, color='r', linestyle='--', label='MLE')
axes[1, 1].axvline(true_mu, color='g', linestyle='--', label='True μ')
axes[1, 1].set_xlabel('μ')
axes[1, 1].set_ylabel('Log-Likelihood (normalized)')
axes[1, 1].set_title(f'Curvature = Fisher Info\nObserved: {observed_info:.2f}, Expected: {fisher_info_mu:.2f}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Confidence interval coverage simulation
coverage_sim = []
nominal_levels = np.linspace(0.80, 0.99, 20)

for conf_level in nominal_levels:
    z = stats.norm.ppf((1 + conf_level) / 2)
    covered = 0
    
    for _ in range(500):
        sim_data = np.random.binomial(1, true_p, n_trials)
        p_hat_sim = np.mean(sim_data)
        se = 1 / np.sqrt(fisher_bernoulli_n(p_hat_sim, n_trials))
        ci_lower = p_hat_sim - z * se
        ci_upper = p_hat_sim + z * se
        
        if ci_lower <= true_p <= ci_upper:
            covered += 1
    
    coverage_sim.append(covered / 500)

axes[1, 2].plot(nominal_levels * 100, nominal_levels * 100, 'r--', 
                linewidth=2, label='Nominal')
axes[1, 2].plot(nominal_levels * 100, np.array(coverage_sim) * 100, 'bo-', 
                linewidth=2, label='Actual')
axes[1, 2].set_xlabel('Nominal Coverage %')
axes[1, 2].set_ylabel('Actual Coverage %')
axes[1, 2].set_title('CI Coverage Using Fisher Information')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nFisher Information Properties:")
print(f"  Additivity: I(θ) for n obs = n × I(θ) for 1 obs")
print(f"  Curvature: High curvature → high information → low variance")
print(f"  Efficiency: MLE achieves Cramér-Rao bound asymptotically")
```

## 1.6 Challenge Round
When is Fisher information not useful?
- **Small samples**: Asymptotic results don't apply; exact inference needed
- **Non-regular models**: Likelihood discontinuous or non-differentiable (e.g., uniform distribution)
- **Biased estimators**: Cramér-Rao bound only for unbiased; biased may have lower MSE
- **Model misspecification**: Fisher info assumes correct model; wrong model invalidates bounds
- **Robust inference**: Fisher info optimal under model, but not robust to violations

## 1.7 Key References
- [Wikipedia - Fisher Information](https://en.wikipedia.org/wiki/Fisher_information)
- [Cramér-Rao Bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound)
- [Asymptotic Theory of MLE](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation#Properties)
- Thinking: High Fisher info = sharp likelihood peak = precise estimates; Curvature of log-likelihood measures information; MLE asymptotically achieves minimum variance bound

---
**Status:** Fundamental measure of statistical information | **Complements:** MLE, Cramér-Rao Bound, Asymptotic Theory, Optimal Design
