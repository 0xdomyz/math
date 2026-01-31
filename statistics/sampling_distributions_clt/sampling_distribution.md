# Sampling Distribution

## 1. Concept Skeleton
**Definition:** Probability distribution of a statistic (mean, proportion, variance) computed from repeated random samples  
**Purpose:** Quantify sampling variability, construct confidence intervals, enable hypothesis testing  
**Prerequisites:** Random sampling, population vs sample distinction, basic probability distributions

## 2. Comparative Framing
| Distribution Type | Population Distribution | Sample Data | Sampling Distribution |
|------------------|------------------------|-------------|----------------------|
| **Definition** | True parameter values | One observed dataset | Distribution of statistics across samples |
| **Variability** | Fixed (unknown) | Fixed (known) | Random (depends on sample size) |
| **Use Case** | Theoretical target | Point estimate | Inference about uncertainty |
| **Example** | μ = 100, σ = 15 (IQ) | x̄ = 102 from n=30 | Distribution of x̄ if repeated sampling |

## 3. Examples + Counterexamples

**Simple Example:**  
Roll die 10 times, compute mean. Repeat 1000 times → sampling distribution of x̄ centers near 3.5, spread depends on n.

**Failure Case:**  
Non-random sampling (e.g., convenience sample). Sampling distribution biased, doesn't represent population parameter.

**Edge Case:**  
Small samples from highly skewed population. Sampling distribution may not be normal until n is large (CLT hasn't kicked in).

## 4. Layer Breakdown
```
Sampling Distribution Framework:
├─ Process:
│   ├─ 1. Define population with parameter θ (e.g., μ, p)
│   ├─ 2. Draw random sample of size n
│   ├─ 3. Calculate sample statistic T (e.g., x̄, p̂, s²)
│   ├─ 4. Repeat steps 2-3 many times
│   └─ 5. Distribution of T values = sampling distribution
├─ Key Properties:
│   ├─ Center: E(T) = θ if unbiased estimator
│   ├─ Spread: SE(T) = σ_T (standard error)
│   ├─ Shape: Often normal by CLT (n large)
│   └─ Sample size effect: SE decreases as n increases
├─ Common Statistics:
│   ├─ Sample Mean: x̄ ~ N(μ, σ/√n)
│   ├─ Sample Proportion: p̂ ~ N(p, √(p(1-p)/n))
│   ├─ Difference in Means: (x̄₁ - x̄₂) ~ N(μ₁-μ₂, √(σ₁²/n₁ + σ₂²/n₂))
│   └─ Sample Variance: (n-1)s²/σ² ~ χ²(n-1)
└─ Applications:
    ├─ Confidence intervals: x̄ ± z*SE
    ├─ Hypothesis testing: (x̄ - μ₀)/SE ~ standard distribution
    └─ Power analysis: Probability of detecting effect
```

## 5. Mini-Project
Simulate sampling distributions for different scenarios:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Population parameters
pop_mean = 50
pop_std = 10
sample_sizes = [5, 10, 30, 100]
n_samples = 10000

# Simulate sampling distributions for different sample sizes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, n in enumerate(sample_sizes):
    # Draw many samples and compute means
    sample_means = []
    for _ in range(n_samples):
        sample = np.random.normal(pop_mean, pop_std, n)
        sample_means.append(np.mean(sample))
    
    # Calculate theoretical standard error
    theoretical_se = pop_std / np.sqrt(n)
    empirical_se = np.std(sample_means)
    
    # Plot histogram
    axes[idx].hist(sample_means, bins=50, density=True, alpha=0.7, label='Empirical')
    
    # Overlay theoretical normal distribution
    x = np.linspace(min(sample_means), max(sample_means), 100)
    axes[idx].plot(x, stats.norm.pdf(x, pop_mean, theoretical_se), 
                   'r-', linewidth=2, label='Theoretical Normal')
    
    axes[idx].axvline(pop_mean, color='green', linestyle='--', linewidth=2, label=f'μ = {pop_mean}')
    axes[idx].set_title(f'n = {n}\nSE(theory)={theoretical_se:.2f}, SE(empirical)={empirical_se:.2f}')
    axes[idx].set_xlabel('Sample Mean')
    axes[idx].set_ylabel('Density')
    axes[idx].legend()

plt.suptitle('Sampling Distribution of Sample Mean\n(Population: N(50, 10))', fontsize=14, y=1.00)
plt.tight_layout()
plt.show()

# Demonstrate different population shapes
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Uniform population
uniform_samples = [np.mean(np.random.uniform(0, 10, 30)) for _ in range(5000)]
axes[0, 0].hist(np.random.uniform(0, 10, 1000), bins=30, alpha=0.7)
axes[0, 0].set_title('Population: Uniform')
axes[1, 0].hist(uniform_samples, bins=50, alpha=0.7)
axes[1, 0].set_title('Sampling Dist (n=30)')

# Exponential population
exp_samples = [np.mean(np.random.exponential(2, 30)) for _ in range(5000)]
axes[0, 1].hist(np.random.exponential(2, 1000), bins=30, alpha=0.7)
axes[0, 1].set_title('Population: Exponential')
axes[1, 1].hist(exp_samples, bins=50, alpha=0.7)
axes[1, 1].set_title('Sampling Dist (n=30)')

# Bimodal population
bimodal = np.concatenate([np.random.normal(2, 1, 500), np.random.normal(8, 1, 500)])
bimodal_samples = [np.mean(np.random.choice(bimodal, 30)) for _ in range(5000)]
axes[0, 2].hist(bimodal, bins=30, alpha=0.7)
axes[0, 2].set_title('Population: Bimodal')
axes[1, 2].hist(bimodal_samples, bins=50, alpha=0.7)
axes[1, 2].set_title('Sampling Dist (n=30)')

plt.suptitle('Central Limit Theorem: Various Population Shapes → Normal Sampling Distribution', fontsize=13)
plt.tight_layout()
plt.show()

print("Key Observations:")
print("1. Sampling distributions become more normal as n increases")
print("2. Standard error decreases with √n")
print("3. Even non-normal populations produce normal sampling distributions (CLT)")
```

## 6. Challenge Round
When is sampling distribution concept insufficient?
- Dependent observations (time series, clusters) violate independence assumption
- Extremely small samples from heavy-tailed distributions (CLT requires large n)
- Non-random sampling methods (convenience, voluntary response)
- Estimating complex statistics (e.g., median, correlation) where theory less developed
- Bootstrap methods may be better when theoretical distribution unknown

## 7. Key References
- [Wikipedia: Sampling Distribution](https://en.wikipedia.org/wiki/Sampling_distribution)
- [Khan Academy: Sampling Distributions](https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library)
- [OnlineStatBook: Sampling Distributions](http://onlinestatbook.com/2/sampling_distributions/intro_samp_dist.html)

---
**Status:** Foundation for inference | **Complements:** Central Limit Theorem, Standard Error, Confidence Intervals
