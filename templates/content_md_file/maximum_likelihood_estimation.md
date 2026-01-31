# Maximum Likelihood Estimation (MLE)

## 1.1 Concept Skeleton
**Definition:** Parameter estimation method finding θ that maximizes likelihood of observing sample data  
**Purpose:** Obtain best-fit parameters with desirable statistical properties (consistency, efficiency, asymptotic normality)  
**Prerequisites:** Probability distributions, calculus (derivatives), conditional probability, logarithms

## 1.2 Comparative Framing
| Method | Maximum Likelihood (MLE) | Method of Moments | Bayesian MAP |
|--------|-------------------------|-------------------|-------------|
| **Approach** | Maximize P(data\|θ) | Match sample moments to theoretical | Maximize posterior P(θ\|data) |
| **Prior Required** | No | No | Yes |
| **Computation** | Optimization (derivative=0) | Algebraic equations | Integration/MCMC |
| **Asymptotic** | Efficient, unbiased | Less efficient | Incorporates domain knowledge |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Estimate λ for Poisson data [3, 5, 4, 6, 2]. Likelihood L(λ) = ∏(e^(-λ)λ^xᵢ/xᵢ!). Log-likelihood derivative → λ̂ = mean(x) = 4

**Failure Case:**  
Uniform(0, θ) distribution: MLE gives θ̂ = max(sample), biased estimate. Derivative-based approach fails at boundary

**Edge Case:**  
Small sample with outliers: MLE sensitive to extremes (not robust). Single outlier shifts normal distribution mean estimate drastically

## 1.4 Layer Breakdown
```
MLE Process:
├─ 1. Likelihood Function: L(θ|x) = P(x₁, x₂, ..., xₙ | θ) = ∏P(xᵢ|θ)
├─ 2. Log-Likelihood: ℓ(θ) = log L(θ) = Σ log P(xᵢ|θ)
│       └─ Converts product to sum (easier optimization)
├─ 3. Score Function: S(θ) = dℓ/dθ
│       └─ Gradient of log-likelihood
├─ 4. Optimization: Set S(θ) = 0, solve for θ̂
│       ├─ Analytical: Closed-form solution
│       └─ Numerical: Newton-Raphson, gradient descent
├─ 5. Verification: Check d²ℓ/dθ² < 0 (maximum, not minimum)
└─ Properties:
    ├─ Consistency: θ̂ → θ₀ as n → ∞
    ├─ Asymptotic Normality: √n(θ̂ - θ₀) ~ N(0, I⁻¹(θ₀))
    └─ Efficiency: Minimum variance among unbiased estimators
```

## 1.5 Mini-Project
MLE for normal distribution parameters:
```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, 100)

# Method 1: Analytical solution (closed-form)
mu_hat = np.mean(data)
sigma_hat = np.std(data, ddof=0)  # MLE uses n, not n-1
print(f"Analytical MLE: μ̂={mu_hat:.3f}, σ̂={sigma_hat:.3f}")

# Method 2: Numerical optimization
def neg_log_likelihood(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return -np.sum(norm.logpdf(data, mu, sigma))

# Initial guess
initial = [0, 1]
result = minimize(neg_log_likelihood, initial, args=(data,), 
                  bounds=[(None, None), (0.01, None)])
mu_opt, sigma_opt = result.x
print(f"Numerical MLE: μ̂={mu_opt:.3f}, σ̂={sigma_opt:.3f}")

# Visualize likelihood surface
mu_range = np.linspace(4, 6, 100)
sigma_range = np.linspace(1.5, 2.5, 100)
MU, SIGMA = np.meshgrid(mu_range, sigma_range)
log_lik = np.zeros_like(MU)

for i in range(len(mu_range)):
    for j in range(len(sigma_range)):
        log_lik[j, i] = -neg_log_likelihood([MU[j, i], SIGMA[j, i]], data)

plt.contourf(MU, SIGMA, log_lik, levels=20, cmap='viridis')
plt.colorbar(label='Log-Likelihood')
plt.plot(mu_hat, sigma_hat, 'r*', markersize=15, label='MLE')
plt.plot(true_mu, true_sigma, 'w+', markersize=15, label='True')
plt.xlabel('μ')
plt.ylabel('σ')
plt.title('Log-Likelihood Surface')
plt.legend()
plt.show()

# Confidence intervals using Fisher Information
# For normal: I(μ) = n/σ², I(σ²) = n/(2σ⁴)
se_mu = sigma_hat / np.sqrt(len(data))
se_sigma = sigma_hat / np.sqrt(2 * len(data))
print(f"95% CI for μ: [{mu_hat - 1.96*se_mu:.3f}, {mu_hat + 1.96*se_mu:.3f}]")
print(f"95% CI for σ: [{sigma_hat - 1.96*se_sigma:.3f}, {sigma_hat + 1.96*se_sigma:.3f}]")
```

## 1.6 Challenge Round
When is MLE the wrong choice?
- Small samples: Biased estimates (e.g., σ² MLE uses n, not n-1)
- Prior knowledge available: Bayesian methods incorporate domain expertise
- Robust estimation needed: MLE sensitive to outliers; use M-estimators
- Boundary constraints: MLE can hit parameter boundaries (use constrained optimization)
- Model misspecification: MLE optimizes wrong model; more fragile than moment methods

## 1.7 Key References
- [MLE Theory and Examples](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) - Comprehensive coverage, asymptotic properties
- [Likelihood Function Explained](https://stats.stackexchange.com/questions/112451/maximum-likelihood-estimation-mle-in-layman-terms) - Intuitive explanation with examples
- [MLE vs Other Estimators](https://www.statlect.com/fundamentals-of-statistics/maximum-likelihood) - Comparison table, worked examples

---
**Status:** Core estimation method | **Complements:** Bayesian Inference, Hypothesis Testing, Fisher Information
