# Unbiased Estimators

## 1. Concept Skeleton
**Definition:** Statistic whose expected value equals the true population parameter: E(θ̂) = θ  
**Purpose:** Ensure long-run accuracy, avoid systematic error, provide theoretical foundation for inference  
**Prerequisites:** Expected value, sampling distributions, bias-variance tradeoff, consistency

## 2. Comparative Framing
| Property | Unbiased Estimator | Biased Estimator | Consistent Estimator |
|----------|-------------------|-----------------|---------------------|
| **Criterion** | E(θ̂) = θ | E(θ̂) ≠ θ | θ̂ → θ as n → ∞ |
| **Focus** | Correctness on average | Systematic error present | Long-run convergence |
| **Trade-off** | May have high variance | Lower variance possible | May be biased for finite n |
| **Example** | x̄ for μ | Uncorrected s² for σ² | MLE (often biased but consistent) |

## 3. Examples + Counterexamples

**Simple Example:**  
Sample mean x̄ is unbiased for μ. If true μ=50, E(x̄) from repeated sampling equals 50.

**Failure Case:**  
Sample variance with n divisor: s² = Σ(xᵢ-x̄)²/n systematically underestimates σ². Must use n-1 (Bessel's correction).

**Edge Case:**  
Sample median is unbiased for population median but inefficient for normal data (higher variance than x̄).

## 4. Layer Breakdown
```
Unbiased Estimator Framework:
├─ Definition: E(θ̂) = θ
│   ├─ Bias: E(θ̂) - θ
│   ├─ Unbiased: Bias = 0
│   └─ Asymptotically unbiased: Bias → 0 as n → ∞
├─ Common Unbiased Estimators:
│   ├─ Population Mean: x̄ = Σxᵢ/n → E(x̄) = μ
│   ├─ Population Proportion: p̂ = X/n → E(p̂) = p
│   ├─ Variance (corrected): s² = Σ(xᵢ-x̄)²/(n-1) → E(s²) = σ²
│   ├─ Difference in Means: x̄₁ - x̄₂ → E(x̄₁ - x̄₂) = μ₁ - μ₂
│   └─ OLS Regression: β̂ → E(β̂) = β (under assumptions)
├─ Biased Estimators (Common):
│   ├─ Uncorrected variance: Σ(xᵢ-x̄)²/n underestimates by factor (n-1)/n
│   ├─ Sample SD: s underestimates σ (Jensen's inequality: √E[s²] > E[s])
│   ├─ Maximum: max(X) underestimates upper bound of uniform
│   ├─ Ratio estimator: E(X/Y) ≠ E(X)/E(Y) unless independent
│   └─ MLE can be biased in finite samples
├─ Bias-Variance Tradeoff:
│   ├─ MSE = Bias² + Variance
│   ├─ Sometimes biased estimator has lower MSE
│   ├─ Ridge/Lasso: Accept bias to reduce variance (regularization)
│   └─ James-Stein: Shrinkage improves MSE despite bias
├─ Desirable Properties:
│   ├─ Unbiased: E(θ̂) = θ
│   ├─ Consistent: θ̂ →ᵖ θ as n → ∞
│   ├─ Efficient: Lowest variance among unbiased estimators
│   └─ Sufficient: Captures all information about θ
└─ Testing for Bias:
    ├─ Analytical: Compute E(θ̂) directly
    ├─ Simulation: Average θ̂ across many samples
    └─ Bootstrap: Estimate bias empirically
```

## 5. Mini-Project
Demonstrate bias in estimators through simulation:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Population parameters
mu = 50
sigma = 10
n = 20
n_simulations = 10000

# Simulate sampling distributions
sample_means = []
sample_vars_biased = []
sample_vars_unbiased = []
sample_sds = []

for _ in range(n_simulations):
    sample = np.random.normal(mu, sigma, n)
    sample_means.append(np.mean(sample))
    sample_vars_biased.append(np.var(sample, ddof=0))  # Biased (divide by n)
    sample_vars_unbiased.append(np.var(sample, ddof=1))  # Unbiased (divide by n-1)
    sample_sds.append(np.std(sample, ddof=1))

# Calculate expected values
expected_mean = np.mean(sample_means)
expected_var_biased = np.mean(sample_vars_biased)
expected_var_unbiased = np.mean(sample_vars_unbiased)
expected_sd = np.mean(sample_sds)

# Calculate bias
bias_mean = expected_mean - mu
bias_var_biased = expected_var_biased - sigma**2
bias_var_unbiased = expected_var_unbiased - sigma**2
bias_sd = expected_sd - sigma

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sample mean (unbiased)
axes[0, 0].hist(sample_means, bins=50, density=True, alpha=0.7, label='Empirical')
axes[0, 0].axvline(mu, color='red', linestyle='--', linewidth=2, label=f'True μ={mu}')
axes[0, 0].axvline(expected_mean, color='green', linestyle='--', linewidth=2, 
                   label=f'E(x̄)={expected_mean:.2f}')
axes[0, 0].set_title(f'Sample Mean (Unbiased)\nBias={bias_mean:.4f}')
axes[0, 0].set_xlabel('x̄')
axes[0, 0].legend()

# Biased variance (n divisor)
axes[0, 1].hist(sample_vars_biased, bins=50, density=True, alpha=0.7, label='Empirical', color='orange')
axes[0, 1].axvline(sigma**2, color='red', linestyle='--', linewidth=2, label=f'True σ²={sigma**2}')
axes[0, 1].axvline(expected_var_biased, color='green', linestyle='--', linewidth=2, 
                   label=f'E(s²)={expected_var_biased:.2f}')
axes[0, 1].set_title(f'Sample Variance (Biased, n divisor)\nBias={bias_var_biased:.2f}')
axes[0, 1].set_xlabel('s² (biased)')
axes[0, 1].legend()

# Unbiased variance (n-1 divisor)
axes[1, 0].hist(sample_vars_unbiased, bins=50, density=True, alpha=0.7, label='Empirical', color='purple')
axes[1, 0].axvline(sigma**2, color='red', linestyle='--', linewidth=2, label=f'True σ²={sigma**2}')
axes[1, 0].axvline(expected_var_unbiased, color='green', linestyle='--', linewidth=2, 
                   label=f'E(s²)={expected_var_unbiased:.2f}')
axes[1, 0].set_title(f'Sample Variance (Unbiased, n-1 divisor)\nBias={bias_var_unbiased:.2f}')
axes[1, 0].set_xlabel('s² (unbiased)')
axes[1, 0].legend()

# Sample SD (slightly biased)
axes[1, 1].hist(sample_sds, bins=50, density=True, alpha=0.7, label='Empirical', color='brown')
axes[1, 1].axvline(sigma, color='red', linestyle='--', linewidth=2, label=f'True σ={sigma}')
axes[1, 1].axvline(expected_sd, color='green', linestyle='--', linewidth=2, 
                   label=f'E(s)={expected_sd:.2f}')
axes[1, 1].set_title(f'Sample SD (Slightly Biased)\nBias={bias_sd:.2f}')
axes[1, 1].set_xlabel('s')
axes[1, 1].legend()

plt.suptitle(f'Bias in Common Estimators (n={n}, {n_simulations} simulations)', fontsize=14)
plt.tight_layout()
plt.show()

print("=== Bias Analysis ===")
print(f"Sample size: {n}")
print(f"True parameters: μ={mu}, σ²={sigma**2}, σ={sigma}\n")
print(f"Sample Mean:")
print(f"  E(x̄) = {expected_mean:.4f}, Bias = {bias_mean:.4f} ✓ Unbiased")
print(f"\nSample Variance (n divisor):")
print(f"  E(s²) = {expected_var_biased:.2f}, Bias = {bias_var_biased:.2f} ✗ Biased")
print(f"  Correction factor: {expected_var_biased/sigma**2:.4f} ≈ {(n-1)/n:.4f}")
print(f"\nSample Variance (n-1 divisor):")
print(f"  E(s²) = {expected_var_unbiased:.2f}, Bias = {bias_var_unbiased:.2f} ✓ Unbiased")
print(f"\nSample SD:")
print(f"  E(s) = {expected_sd:.2f}, Bias = {bias_sd:.2f} ✗ Slightly Biased")

# Bias-variance tradeoff example
print("\n\n=== Bias-Variance Tradeoff Example ===")

# Estimate mean of exponential (different estimators)
true_lambda = 2
true_mean = 1 / true_lambda
n = 50
n_sims = 5000

# Three estimators
mse_sample_mean = []
mse_mle = []
mse_shrinkage = []

for _ in range(n_sims):
    sample = np.random.exponential(1/true_lambda, n)
    
    # Estimator 1: Sample mean (unbiased)
    est_mean = np.mean(sample)
    mse_sample_mean.append((est_mean - true_mean)**2)
    
    # Estimator 2: MLE (unbiased for exponential)
    est_mle = np.mean(sample)
    mse_mle.append((est_mle - true_mean)**2)
    
    # Estimator 3: Shrinkage toward prior (biased but lower variance)
    prior_mean = 0.6  # Assume we have prior knowledge
    shrinkage_factor = 0.2
    est_shrinkage = (1 - shrinkage_factor) * np.mean(sample) + shrinkage_factor * prior_mean
    mse_shrinkage.append((est_shrinkage - true_mean)**2)

print(f"True mean: {true_mean:.3f}")
print(f"\n1. Sample Mean (Unbiased):")
print(f"   MSE: {np.mean(mse_sample_mean):.6f}")
print(f"\n2. MLE (Unbiased):")
print(f"   MSE: {np.mean(mse_mle):.6f}")
print(f"\n3. Shrinkage Estimator (Biased):")
print(f"   MSE: {np.mean(mse_shrinkage):.6f}")
print(f"   Note: Sometimes bias reduces MSE through lower variance")
```

## 6. Challenge Round
When is unbiasedness not the primary goal?
- High-dimensional estimation (shrinkage/regularization reduces MSE)
- Small samples with high variance (biased estimator may have lower MSE)
- Prediction focus (bias-variance tradeoff favors lower MSE)
- Computational constraints (biased but fast approximation preferred)
- Sequential decision-making (Bayes estimators may be biased but optimal)

## 7. Key References
- [Wikipedia: Bias of an Estimator](https://en.wikipedia.org/wiki/Bias_of_an_estimator)
- [Khan Academy: Unbiased Estimators](https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library/sample-means/v/review-and-intuition-why-we-divide-by-n-1-for-the-unbiased-sample-variance)
- [Bias-Variance Tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)

---
**Status:** Estimator quality criterion | **Complements:** Sampling Distributions, Consistency, Efficiency, MSE
