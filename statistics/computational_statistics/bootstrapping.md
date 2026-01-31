# Bootstrapping

## 1. Concept Skeleton
**Definition:** Resampling method creating many samples (with replacement) from observed data to estimate sampling distribution of statistic  
**Purpose:** Estimate uncertainty (confidence intervals, standard errors) without distributional assumptions  
**Prerequisites:** Basic statistics, sampling theory, computational programming

## 2. Comparative Framing
| Method | Bootstrapping | Parametric CI | Jackknife |
|--------|--------------|--------------|-----------|
| **Assumptions** | None (non-parametric) | Normal distribution | Smooth statistic |
| **Sample Generation** | Resample with replacement | Analytic formulas | Leave-one-out |
| **Computational Cost** | High (1000+ iterations) | Instant | Moderate (n iterations) |
| **Applicability** | Any statistic | Mean, proportion only | Bias estimation focus |

## 3. Examples + Counterexamples

**Simple Example:**  
Estimate 95% CI for median: Resample 1000 times, take 2.5th and 97.5th percentiles of bootstrap medians

**Failure Case:**  
Small sample (n<30) with outliers: Bootstrap may not capture true tail behavior, CI too narrow

**Edge Case:**  
Time series data: Naive bootstrap breaks temporal dependence → use block bootstrap instead

## 4. Layer Breakdown
```
Bootstrap Process:
├─ Original Sample: n observations {x₁, x₂, ..., xₙ}
├─ Bootstrap Sample: Draw n observations with replacement
│   └─ Some xᵢ appear multiple times, others omitted (~37% excluded)
├─ Compute Statistic: Calculate θ̂* on bootstrap sample
├─ Repeat: B times (typically 1000-10000)
├─ Bootstrap Distribution: {θ̂₁*, θ̂₂*, ..., θ̂ᴮ*}
└─ Inference:
    ├─ Standard Error: SD of bootstrap distribution
    ├─ Percentile CI: Quantiles of bootstrap distribution
    ├─ Bias Correction: E(θ̂*) - θ̂
    └─ Hypothesis Test: Check if null value in CI
```

**Interaction:** Data → Resample → Statistic → Aggregate → Distribution estimate

## 5. Mini-Project
Bootstrap confidence interval for correlation coefficient:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Generate correlated data
np.random.seed(42)
n = 50
x = np.random.normal(0, 1, n)
y = 0.6 * x + np.random.normal(0, 0.8, n)

# Original correlation
r_observed, p_value = pearsonr(x, y)
print(f"Observed correlation: {r_observed:.3f}")

# Bootstrap
n_bootstrap = 10000
bootstrap_rs = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    # Resample indices with replacement
    indices = np.random.choice(n, size=n, replace=True)
    x_boot = x[indices]
    y_boot = y[indices]
    
    # Calculate statistic
    bootstrap_rs[i], _ = pearsonr(x_boot, y_boot)

# Bootstrap inference
se_boot = bootstrap_rs.std()
ci_lower = np.percentile(bootstrap_rs, 2.5)
ci_upper = np.percentile(bootstrap_rs, 97.5)
bias = bootstrap_rs.mean() - r_observed

print(f"Bootstrap SE: {se_boot:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"Bias: {bias:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Original data
axes[0].scatter(x, y, alpha=0.6)
axes[0].set_title(f'Original Data (r={r_observed:.3f})')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Bootstrap distribution
axes[1].hist(bootstrap_rs, bins=50, density=True, alpha=0.7)
axes[1].axvline(r_observed, color='r', linewidth=2, label='Observed')
axes[1].axvline(ci_lower, color='g', linestyle='--', label='95% CI')
axes[1].axvline(ci_upper, color='g', linestyle='--')
axes[1].set_title('Bootstrap Distribution')
axes[1].set_xlabel('Correlation')
axes[1].legend()

plt.tight_layout()
plt.show()

# Compare to Fisher z-transform (parametric)
fisher_se = 1 / np.sqrt(n - 3)
fisher_ci_lower = np.tanh(np.arctanh(r_observed) - 1.96 * fisher_se)
fisher_ci_upper = np.tanh(np.arctanh(r_observed) + 1.96 * fisher_se)
print(f"\nFisher z-transform CI: [{fisher_ci_lower:.3f}, {fisher_ci_upper:.3f}]")
```

## 6. Challenge Round
When is bootstrapping the wrong tool?
- Tiny samples (n<10): Not enough diversity to resample from
- Estimating extreme quantiles (99th percentile): Cannot extrapolate beyond observed range
- Complex dependence structures (spatial, temporal): Need specialized bootstrap
- Real-time systems: Too computationally expensive
- When parametric method exists and assumptions hold (use faster analytic solution)

## 7. Key References
- [Bootstrap Methods Overview (Wikipedia)](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [Efron & Tibshirani, An Introduction to the Bootstrap](https://www.routledge.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317)
- [scikit-learn Bootstrap Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)

---
**Status:** Core resampling technique | **Complements:** Confidence Intervals, Jackknife, Permutation Tests
