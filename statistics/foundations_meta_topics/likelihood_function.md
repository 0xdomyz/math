# Likelihood Function

## 1.1 Concept Skeleton
**Definition:** Function L(θ|data) expressing probability of observed data given parameter θ; foundation of parameter estimation  
**Purpose:** Quantify which parameter values make observed data most plausible, enable MLE  
**Prerequisites:** Probability distributions, conditional probability, calculus optimization

## 1.2 Comparative Framing
| Concept | Probability P(data\|θ) | Likelihood L(θ\|data) | Posterior P(θ\|data) |
|---------|----------------------|---------------------|-------------------|
| **Fixed** | θ is known | Data is fixed | Data is fixed |
| **Variable** | Data varies | θ varies | θ varies |
| **Interpretation** | Probability of future data | Plausibility of parameter | Updated belief (Bayesian) |

## 1.3 Examples + Counterexamples

**Simple Example:**  
10 coin flips: 7 heads, 3 tails. L(p=0.7) = C(10,7)×0.7⁷×0.3³ = 0.267 (most plausible p)

**Failure Case:**  
Uniform prior assumption when true parameter at boundary (p=0 or p=1) → MLE fails

**Edge Case:**  
Multiple local maxima: Likelihood has several peaks → optimization finds local, not global maximum

## 1.4 Layer Breakdown
```
Likelihood Components:
├─ Definition:
│   ├─ Discrete: L(θ|x₁,...,xₙ) = ∏P(xᵢ|θ)
│   └─ Continuous: L(θ|x₁,...,xₙ) = ∏f(xᵢ|θ)
├─ Log-Likelihood:
│   ├─ ℓ(θ) = log L(θ) = Σ log f(xᵢ|θ)
│   ├─ Advantage: Sum instead of product (numerical stability)
│   └─ Monotonic: max L(θ) = max ℓ(θ)
├─ Maximum Likelihood Estimation:
│   ├─ MLE: θ̂ = argmax L(θ|data)
│   ├─ Method: Set dℓ/dθ = 0, solve for θ
│   └─ Properties: Asymptotically unbiased, efficient, normal
├─ Likelihood Ratio:
│   ├─ LR = L(θ₁|data) / L(θ₀|data)
│   ├─ Use: Hypothesis testing (Neyman-Pearson)
│   └─ Test: -2 log(LR) ~ χ² under H₀
└─ Profile Likelihood:
    ├─ Fix parameter of interest, maximize over nuisance parameters
    └─ Use: Confidence intervals for one parameter
```

## 1.5 Mini-Project
Compute and visualize likelihood:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

np.random.seed(42)

# Generate data from known distribution
true_mu = 5.0
true_sigma = 2.0
data = np.random.normal(true_mu, true_sigma, 50)

# Define likelihood functions
def neg_log_likelihood_normal(params, data):
    """Negative log-likelihood for normal distribution"""
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    n = len(data)
    ll = -n/2 * np.log(2*np.pi) - n*np.log(sigma) - np.sum((data - mu)**2)/(2*sigma**2)
    return -ll  # Return negative for minimization

# Compute likelihood surface
mu_range = np.linspace(3, 7, 100)
sigma_range = np.linspace(0.5, 4, 100)
MU, SIGMA = np.meshgrid(mu_range, sigma_range)
LL = np.zeros_like(MU)

for i in range(len(mu_range)):
    for j in range(len(sigma_range)):
        LL[j, i] = -neg_log_likelihood_normal([MU[j, i], SIGMA[j, i]], data)

# Find MLE
result = minimize(neg_log_likelihood_normal, x0=[4, 1], args=(data,), 
                  method='Nelder-Mead')
mle_mu, mle_sigma = result.x

print(f"True Parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE Estimates: μ̂={mle_mu:.3f}, σ̂={mle_sigma:.3f}")
print(f"Sample Mean: {np.mean(data):.3f}")
print(f"Sample SD: {np.std(data, ddof=1):.3f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Data histogram with fitted distribution
axes[0, 0].hist(data, bins=20, density=True, alpha=0.7, edgecolor='black', label='Data')
x = np.linspace(data.min(), data.max(), 100)
axes[0, 0].plot(x, stats.norm.pdf(x, true_mu, true_sigma), 'r-', 
                linewidth=2, label=f'True N({true_mu},{true_sigma})')
axes[0, 0].plot(x, stats.norm.pdf(x, mle_mu, mle_sigma), 'g--', 
                linewidth=2, label=f'MLE N({mle_mu:.2f},{mle_sigma:.2f})')
axes[0, 0].legend()
axes[0, 0].set_title('Data with True and MLE Distributions')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')

# 2. Likelihood contour plot
contour = axes[0, 1].contourf(MU, SIGMA, LL, levels=20, cmap='viridis')
axes[0, 1].contour(MU, SIGMA, LL, levels=10, colors='white', alpha=0.3)
axes[0, 1].plot(true_mu, true_sigma, 'r*', markersize=15, label='True')
axes[0, 1].plot(mle_mu, mle_sigma, 'w*', markersize=15, label='MLE')
axes[0, 1].set_xlabel('μ')
axes[0, 1].set_ylabel('σ')
axes[0, 1].set_title('Log-Likelihood Surface')
axes[0, 1].legend()
plt.colorbar(contour, ax=axes[0, 1])

# 3. Profile likelihood for μ
ll_profile_mu = []
for mu in mu_range:
    # Maximize over sigma for each mu
    res = minimize(lambda s: neg_log_likelihood_normal([mu, s], data), 
                   x0=[2], method='Nelder-Mead')
    ll_profile_mu.append(-res.fun)

axes[1, 0].plot(mu_range, ll_profile_mu, 'b-', linewidth=2)
axes[1, 0].axvline(mle_mu, color='r', linestyle='--', label='MLE')
axes[1, 0].axvline(true_mu, color='g', linestyle='--', label='True')
axes[1, 0].set_xlabel('μ')
axes[1, 0].set_ylabel('Profile Log-Likelihood')
axes[1, 0].set_title('Profile Likelihood for μ')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Likelihood ratio test
# Test H0: μ = 4 vs H1: μ ≠ 4
mu_0 = 4.0
ll_0 = -neg_log_likelihood_normal([mu_0, mle_sigma], data)
ll_mle = -neg_log_likelihood_normal([mle_mu, mle_sigma], data)
lr_statistic = -2 * (ll_0 - ll_mle)
p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)

test_range = np.linspace(3, 7, 100)
lr_values = []
for mu_test in test_range:
    ll_test = -neg_log_likelihood_normal([mu_test, mle_sigma], data)
    lr = -2 * (ll_test - ll_mle)
    lr_values.append(lr)

axes[1, 1].plot(test_range, lr_values, 'b-', linewidth=2)
axes[1, 1].axhline(stats.chi2.ppf(0.95, df=1), color='r', linestyle='--', 
                   label='95% threshold (χ²)')
axes[1, 1].axvline(mle_mu, color='g', linestyle='--', label='MLE')
axes[1, 1].set_xlabel('μ')
axes[1, 1].set_ylabel('LR Test Statistic')
axes[1, 1].set_title('Likelihood Ratio Test')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nLikelihood Ratio Test (H0: μ=4):")
print(f"  LR Statistic: {lr_statistic:.3f}")
print(f"  P-value: {p_value:.4f}")
```

## 1.6 Challenge Round
When is likelihood the wrong approach?
- **Uninformative data**: Flat likelihood → all parameters equally plausible (need priors/Bayesian)
- **Model misspecification**: Likelihood assumes wrong distribution → biased estimates
- **Boundary parameters**: MLE at edge of parameter space (e.g., variance = 0)
- **Complex models**: High-dimensional likelihood intractable (use MCMC, variational methods)
- **Small samples**: MLE biased; use bias-corrected estimators or Bayesian methods

## 1.7 Key References
- [Wikipedia - Likelihood Function](https://en.wikipedia.org/wiki/Likelihood_function)
- [Wikipedia - Maximum Likelihood Estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
- [Likelihood Ratio Test](https://en.wikipedia.org/wiki/Likelihood-ratio_test)
- Thinking: Likelihood treats data as fixed, θ as variable (opposite of probability); Log-likelihood preferred for computation; MLE is foundation of frequentist inference

---
**Status:** Foundation of statistical inference | **Complements:** MLE, Hypothesis Testing, Bayesian Inference
