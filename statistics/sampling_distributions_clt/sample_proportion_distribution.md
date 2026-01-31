# Sample Proportion Distribution

## 1. Concept Skeleton
**Definition:** Sampling distribution of p̂ (sample proportion) from repeated samples when population proportion is p  
**Purpose:** Test hypotheses about proportions, construct CIs for rates/percentages, compare groups (A/B testing)  
**Prerequisites:** Binomial distribution, sampling distributions, Central Limit Theorem, normal approximation

## 2. Comparative Framing
| Statistic | Sample Mean (x̄) | Sample Proportion (p̂) | Sample Count (X) |
|-----------|-----------------|---------------------|-----------------|
| **Type** | Continuous | Continuous (0 to 1) | Discrete (0 to n) |
| **Formula** | Σxᵢ/n | X/n (successes/n) | Σ(1 if success) |
| **Distribution** | Normal by CLT | Normal if np≥10, n(1-p)≥10 | Binomial(n, p) |
| **Mean** | μ | p | np |
| **SE** | σ/√n | √[p(1-p)/n] | √[np(1-p)] |

## 3. Examples + Counterexamples

**Simple Example:**  
Election poll: 520/1000 favor candidate → p̂ = 0.52. SE = √[0.52×0.48/1000] = 0.016 → 95% CI: 0.52 ± 0.031.

**Failure Case:**  
Rare event (p=0.01, n=50). np=0.5 < 10 → normal approximation invalid. Must use exact binomial or increase n.

**Edge Case:**  
p̂=0 or p̂=1 (0% or 100% success). SE formula breaks down; use Wilson score interval instead of standard Wald CI.

## 4. Layer Breakdown
```
Sample Proportion Framework:
├─ Definition: p̂ = X/n where X ~ Binomial(n, p)
├─ Sampling Distribution (Large n):
│   ├─ Mean: E(p̂) = p (unbiased)
│   ├─ Standard Error: SE(p̂) = √[p(1-p)/n]
│   ├─ Shape: Approximately N(p, √[p(1-p)/n])
│   └─ Conditions: np ≥ 10 AND n(1-p) ≥ 10
├─ Confidence Intervals:
│   ├─ Wald (standard): p̂ ± z*√[p̂(1-p̂)/n]
│   ├─ Wilson score: Better for small n or extreme p̂
│   ├─ Exact binomial: When normal approx fails
│   └─ Agresti-Coull: Add 2 successes, 2 failures (conservative)
├─ Hypothesis Testing:
│   ├─ One-sample: z = (p̂ - p₀)/√[p₀(1-p₀)/n]
│   ├─ Two-sample: z = (p̂₁ - p̂₂)/√[p̂(1-p̂)(1/n₁ + 1/n₂)]
│   │   where p̂ = (X₁ + X₂)/(n₁ + n₂) (pooled)
│   └─ Chi-square equivalent: χ² = z²
└─ Applications:
    ├─ A/B testing (conversion rates)
    ├─ Quality control (defect rates)
    ├─ Epidemiology (disease prevalence)
    └─ Political polling (voter preferences)
```

## 5. Mini-Project
Simulate and test proportion distributions:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Demonstrate CLT for proportions
true_p = 0.3
sample_sizes = [10, 30, 100, 500]
n_simulations = 10000

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, n in enumerate(sample_sizes):
    # Simulate many samples
    sample_props = []
    for _ in range(n_simulations):
        sample = np.random.binomial(1, true_p, n)  # n Bernoulli trials
        sample_props.append(np.mean(sample))
    
    # Theoretical parameters
    theoretical_se = np.sqrt(true_p * (1 - true_p) / n)
    
    # Plot histogram
    axes[idx].hist(sample_props, bins=50, density=True, alpha=0.7, 
                   label=f'Empirical (n={n})')
    
    # Overlay theoretical normal
    x = np.linspace(0, 1, 100)
    axes[idx].plot(x, stats.norm.pdf(x, true_p, theoretical_se), 
                   'r-', linewidth=2, label='Normal Approximation')
    
    axes[idx].axvline(true_p, color='green', linestyle='--', linewidth=2, label=f'p={true_p}')
    axes[idx].set_title(f'n={n}, SE={theoretical_se:.3f}\nnp={n*true_p:.0f}, n(1-p)={n*(1-true_p):.0f}')
    axes[idx].set_xlabel('Sample Proportion (p̂)')
    axes[idx].set_ylabel('Density')
    axes[idx].legend()
    axes[idx].set_xlim(0, 0.6)

plt.suptitle(f'Sampling Distribution of Proportion (True p = {true_p})', fontsize=14)
plt.tight_layout()
plt.show()

# Confidence interval comparison methods
n = 100
successes = 30
p_hat = successes / n

print("=== 95% Confidence Intervals for Proportion ===")
print(f"Sample: {successes}/{n} = {p_hat:.2f}\n")

# 1. Wald (standard)
se_wald = np.sqrt(p_hat * (1 - p_hat) / n)
ci_wald_lower = p_hat - 1.96 * se_wald
ci_wald_upper = p_hat + 1.96 * se_wald
print(f"Wald:         [{ci_wald_lower:.3f}, {ci_wald_upper:.3f}]")

# 2. Wilson Score
z = 1.96
ci_wilson_lower = (p_hat + z**2/(2*n) - z*np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / (1 + z**2/n)
ci_wilson_upper = (p_hat + z**2/(2*n) + z*np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / (1 + z**2/n)
print(f"Wilson:       [{ci_wilson_lower:.3f}, {ci_wilson_upper:.3f}]")

# 3. Agresti-Coull
n_adj = n + 4
p_adj = (successes + 2) / n_adj
se_adj = np.sqrt(p_adj * (1 - p_adj) / n_adj)
ci_ac_lower = p_adj - 1.96 * se_adj
ci_ac_upper = p_adj + 1.96 * se_adj
print(f"Agresti-Coull:[{ci_ac_lower:.3f}, {ci_ac_upper:.3f}]")

# Two-sample comparison (A/B test)
print("\n=== A/B Test Example ===")
n_a, n_b = 1000, 1000
p_a_true, p_b_true = 0.10, 0.12  # 2% absolute difference

# Simulate data
conversions_a = np.random.binomial(n_a, p_a_true)
conversions_b = np.random.binomial(n_b, p_b_true)

p_a_hat = conversions_a / n_a
p_b_hat = conversions_b / n_b

# Pooled proportion (under H₀: p_a = p_b)
p_pooled = (conversions_a + conversions_b) / (n_a + n_b)
se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))

# Test statistic
diff = p_b_hat - p_a_hat
z_stat = diff / se_diff
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"Control (A): {conversions_a}/{n_a} = {p_a_hat:.3f}")
print(f"Treatment (B): {conversions_b}/{n_b} = {p_b_hat:.3f}")
print(f"Difference: {diff:.3f} (B - A)")
print(f"SE(difference): {se_diff:.4f}")
print(f"z-statistic: {z_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

# Visualize sampling distribution under null
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(-0.05, 0.05, 1000)
y = stats.norm.pdf(x, 0, se_diff)
ax.plot(x, y, 'b-', linewidth=2, label='Null Distribution\n(p_A = p_B)')
ax.axvline(diff, color='red', linestyle='--', linewidth=2, label=f'Observed diff = {diff:.3f}')
ax.axvline(-diff, color='red', linestyle='--', linewidth=2)
ax.fill_between(x[x >= abs(diff)], y[x >= abs(diff)], alpha=0.3, color='red', label='p-value region')
ax.fill_between(x[x <= -abs(diff)], y[x <= -abs(diff)], alpha=0.3, color='red')
ax.set_xlabel('Difference in Proportions (p_B - p_A)')
ax.set_ylabel('Density')
ax.set_title(f'Two-Sample Proportion Test\np-value = {p_value:.4f}')
ax.legend()
plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is sample proportion analysis inappropriate?
- Dependent observations (clustered surveys, repeated measures)
- Very small samples or extreme proportions (use exact binomial)
- Multiple comparisons without correction (inflated Type I error)
- Non-binary outcomes (use multinomial or continuous methods)
- Stratified/complex sampling designs (need survey weights)

## 7. Key References
- [Wikipedia: Binomial Proportion Confidence Interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
- [Khan Academy: Sample Proportions](https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library/sample-proportions)
- [Wilson Score Interval Explained](https://www.ucl.ac.uk/english-usage/staff/sean/resources/binomialpoisson.pdf)

---
**Status:** Categorical data inference | **Complements:** Binomial Distribution, Hypothesis Testing, A/B Testing
