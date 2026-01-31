# Standard Error

## 1. Concept Skeleton
**Definition:** Standard deviation of a sampling distribution; measures precision of sample statistic as estimate of population parameter  
**Purpose:** Quantify sampling variability, construct confidence intervals, standardize test statistics  
**Prerequisites:** Standard deviation, sampling distributions, square root properties

## 2. Comparative Framing
| Measure | Standard Deviation (σ, s) | Standard Error (SE) | Confidence Interval |
|---------|--------------------------|-------------------|-------------------|
| **Measures** | Variability of individual data points | Variability of sample statistics | Range likely containing parameter |
| **Decreases With** | N/A (population property) | Larger sample size (∝ 1/√n) | Smaller SE |
| **Formula (mean)** | √[Σ(x-μ)²/N] | σ/√n | x̄ ± z*SE |
| **Interpretation** | Spread of data | Precision of estimate | Plausible parameter values |

## 3. Examples + Counterexamples

**Simple Example:**  
IQ scores: σ = 15. Sample n=100 → SE = 15/√100 = 1.5. Sample mean precise to ±1.5 points.

**Failure Case:**  
Using s (sample SD) instead of SE when describing estimate precision. SD describes data spread, not estimate uncertainty.

**Edge Case:**  
Finite population correction: SE = (σ/√n) × √[(N-n)/(N-1)] when sampling >5% of population. SE smaller than uncorrected.

## 4. Layer Breakdown
```
Standard Error Framework:
├─ Core Formula: SE = σ / √n
│   ├─ σ: Population standard deviation (often unknown)
│   ├─ n: Sample size
│   └─ √n: Efficiency increases with square root of sample size
├─ Common Standard Errors:
│   ├─ Mean: SE(x̄) = σ/√n  (or s/√n if σ unknown)
│   ├─ Proportion: SE(p̂) = √[p(1-p)/n]
│   ├─ Difference in Means: SE(x̄₁-x̄₂) = √(σ₁²/n₁ + σ₂²/n₂)
│   ├─ Regression Coefficient: SE(β̂) = σ/√[Σ(xᵢ-x̄)²]
│   └─ Correlation: SE(r) ≈ √[(1-r²)/(n-2)]
├─ Key Properties:
│   ├─ Decreases with √n (doubling precision requires 4× sample)
│   ├─ Independent of population size (for large N)
│   ├─ Increases with population variability σ
│   └─ Used to construct: CI = estimate ± multiplier*SE
├─ Estimation:
│   ├─ Known σ: Use population SD directly
│   ├─ Unknown σ: Use sample SD (s) → t-distribution
│   └─ Bootstrap: Resample to estimate SE empirically
└─ Applications:
    ├─ Hypothesis testing: z = (x̄ - μ₀)/SE
    ├─ Confidence intervals: x̄ ± 1.96*SE (95% CI)
    ├─ Sample size planning: n = (z*σ/E)²
    └─ Comparing estimates: SE(diff) tells if difference significant
```

## 5. Mini-Project
Calculate and visualize standard errors:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Demonstrate SE vs SD
population = np.random.normal(100, 15, 100000)  # Large population
pop_mean = np.mean(population)
pop_std = np.std(population, ddof=0)

sample_sizes = [10, 20, 50, 100, 200, 500]
n_simulations = 1000

# Calculate empirical SE for each sample size
empirical_se = []
theoretical_se = []

for n in sample_sizes:
    sample_means = [np.mean(np.random.choice(population, n)) for _ in range(n_simulations)]
    empirical_se.append(np.std(sample_means))
    theoretical_se.append(pop_std / np.sqrt(n))

# Plot SE vs sample size
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SE decreases with √n
axes[0, 0].plot(sample_sizes, empirical_se, 'o-', label='Empirical SE', linewidth=2)
axes[0, 0].plot(sample_sizes, theoretical_se, 's--', label='Theoretical (σ/√n)', linewidth=2)
axes[0, 0].set_xlabel('Sample Size (n)')
axes[0, 0].set_ylabel('Standard Error')
axes[0, 0].set_title('Standard Error Decreases with √n')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# SE on log scale shows 1/√n relationship
axes[0, 1].loglog(sample_sizes, empirical_se, 'o-', label='Empirical SE', linewidth=2)
axes[0, 1].loglog(sample_sizes, theoretical_se, 's--', label='Theoretical (σ/√n)', linewidth=2)
axes[0, 1].set_xlabel('Sample Size (n)')
axes[0, 1].set_ylabel('Standard Error')
axes[0, 1].set_title('Log-Log: SE ∝ 1/√n')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Confidence intervals using SE
n = 50
n_samples = 100
ci_contains_mu = 0

for i in range(n_samples):
    sample = np.random.choice(population, n)
    sample_mean = np.mean(sample)
    se = pop_std / np.sqrt(n)
    ci_lower = sample_mean - 1.96 * se
    ci_upper = sample_mean + 1.96 * se
    
    color = 'green' if ci_lower <= pop_mean <= ci_upper else 'red'
    if ci_lower <= pop_mean <= ci_upper:
        ci_contains_mu += 1
    
    if i < 50:  # Plot first 50 only
        axes[1, 0].plot([i, i], [ci_lower, ci_upper], color=color, alpha=0.5)
        axes[1, 0].plot(i, sample_mean, 'o', color=color, markersize=3)

axes[1, 0].axhline(pop_mean, color='blue', linestyle='--', linewidth=2, label=f'True μ={pop_mean:.1f}')
axes[1, 0].set_xlabel('Sample Number')
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title(f'95% CIs using SE (n={n})\n{ci_contains_mu}/{n_samples} contain μ')
axes[1, 0].legend()

# SE for proportions
true_p = 0.3
sample_sizes_prop = np.arange(10, 500, 10)
se_proportions = [np.sqrt(true_p * (1 - true_p) / n) for n in sample_sizes_prop]

axes[1, 1].plot(sample_sizes_prop, se_proportions, linewidth=2)
axes[1, 1].set_xlabel('Sample Size (n)')
axes[1, 1].set_ylabel('SE(p̂)')
axes[1, 1].set_title(f'Standard Error of Proportion\n(p = {true_p})')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Practical example: comparing two groups
print("\n=== Practical Example: Two-Sample Comparison ===")
group_a = np.random.normal(100, 15, 30)
group_b = np.random.normal(105, 15, 30)

mean_a = np.mean(group_a)
mean_b = np.mean(group_b)
sd_a = np.std(group_a, ddof=1)
sd_b = np.std(group_b, ddof=1)

se_a = sd_a / np.sqrt(len(group_a))
se_b = sd_b / np.sqrt(len(group_b))
se_diff = np.sqrt(se_a**2 + se_b**2)

diff = mean_b - mean_a
t_stat = diff / se_diff
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=58))

print(f"Group A: mean={mean_a:.2f}, SE={se_a:.2f}")
print(f"Group B: mean={mean_b:.2f}, SE={se_b:.2f}")
print(f"Difference: {diff:.2f} ± {1.96*se_diff:.2f} (95% CI)")
print(f"t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
print(f"\nNote: SD measures data spread, SE measures estimate precision")
print(f"SD(A)={sd_a:.2f} (spread of values), SE(A)={se_a:.2f} (precision of mean)")
```

## 6. Challenge Round
When is standard error misleading or insufficient?
- Non-random samples (SE assumes random sampling)
- Severely skewed distributions with small n (SE assumes approximate normality)
- Clustered/hierarchical data (need cluster-robust SE)
- Model misspecification (SE conditional on correct model)
- Multiple comparisons (SE doesn't account for inflated Type I error)

## 7. Key References
- [Wikipedia: Standard Error](https://en.wikipedia.org/wiki/Standard_error)
- [Khan Academy: Standard Error of the Mean](https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library/sample-means/v/standard-error-of-the-mean)
- [Standard Error vs Standard Deviation](https://www.statisticshowto.com/probability-and-statistics/standard-deviation/standard-error-of-the-mean/)

---
**Status:** Precision quantifier | **Complements:** Sampling Distribution, Confidence Intervals, Hypothesis Testing
