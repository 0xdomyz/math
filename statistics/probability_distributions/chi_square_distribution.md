# Chi-Square Distribution

## 1.1 Concept Skeleton
**Definition:** Distribution of sum of squared independent standard normal variables; models variance and categorical data  
**Purpose:** Test goodness-of-fit, independence in contingency tables, confidence intervals for variance  
**Prerequisites:** Normal distribution, sampling distributions, hypothesis testing

## 1.2 Comparative Framing
| Distribution | Chi-Square χ²(k) | Normal N(μ,σ²) | t-distribution |
|--------------|----------------|----------------|---------------|
| **Shape** | Right-skewed (k small), approaches normal (k large) | Symmetric bell curve | Symmetric, heavier tails |
| **Support** | [0, ∞) (positive only) | (-∞, ∞) | (-∞, ∞) |
| **Use Case** | Variance tests, categorical data | Continuous measurements | Small sample means |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Sample variance of n=10 from N(0,1): (n-1)s²/σ² ~ χ²(9). Used for CI on population variance

**Failure Case:**  
Low expected cell counts (<5) in chi-square test → p-values unreliable; use Fisher's exact test instead

**Edge Case:**  
df=1: Highly right-skewed, mode at 0. df→∞: Approaches normal distribution by CLT

## 1.4 Layer Breakdown
```
Chi-Square Distribution Components:
├─ Definition:
│   ├─ Z₁,...,Zₖ ~ N(0,1) independent
│   ├─ χ² = Z₁² + Z₂² + ... + Zₖ²
│   └─ Degrees of freedom: k (number of squared normals)
├─ PDF:
│   ├─ f(x) = (1/(2^(k/2)Γ(k/2))) x^(k/2-1) e^(-x/2)
│   ├─ Support: x ≥ 0
│   └─ Shape: Right-skewed, approaches normal as k increases
├─ Parameters:
│   ├─ Mean: E[χ²(k)] = k
│   ├─ Variance: Var[χ²(k)] = 2k
│   └─ Mode: max(k-2, 0)
├─ Properties:
│   ├─ Additivity: χ²(k₁) + χ²(k₂) = χ²(k₁+k₂) if independent
│   ├─ Relationship: (Z ~ N(0,1)) ⟹ Z² ~ χ²(1)
│   └─ Approximation: χ²(k) ≈ N(k, 2k) for large k
├─ Applications:
│   ├─ Chi-Square Test of Independence:
│   │   ├─ Contingency tables (categorical × categorical)
│   │   ├─ Test statistic: Σ(Observed - Expected)²/Expected
│   │   └─ df = (rows-1) × (cols-1)
│   ├─ Goodness-of-Fit Test:
│   │   ├─ Does data match theoretical distribution?
│   │   └─ df = categories - parameters - 1
│   ├─ Variance Test:
│   │   ├─ (n-1)s²/σ² ~ χ²(n-1)
│   │   └─ CI for σ²: [(n-1)s²/χ²_upper, (n-1)s²/χ²_lower]
│   └─ Distribution Building Block:
│       ├─ F-distribution: ratio of two chi-squares
│       └─ t-distribution: involves chi-square in denominator
└─ Related Distributions:
    ├─ Non-central χ²: When normals have non-zero mean
    └─ Gamma distribution: χ²(k) = Gamma(k/2, 1/2)
```

## 1.5 Mini-Project
Explore chi-square distribution and tests:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

np.random.seed(42)

# Visualize chi-square distributions with different df
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. PDF for various degrees of freedom
x = np.linspace(0, 30, 1000)
dfs = [1, 2, 3, 5, 10, 20]

for i, df in enumerate(dfs):
    row, col = i // 3, i % 3
    axes[row, col].plot(x, stats.chi2.pdf(x, df), 'b-', linewidth=2)
    axes[row, col].axvline(df, color='r', linestyle='--', label=f'Mean = {df}')
    axes[row, col].set_title(f'χ²({df}): Mean={df}, Var={2*df}')
    axes[row, col].set_xlabel('x')
    axes[row, col].set_ylabel('Density')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 2. Demonstrate sum of squared normals
n_samples = 10000
df_test = 5

# Generate df independent standard normals, square and sum
normals = np.random.standard_normal((n_samples, df_test))
chi_square_samples = np.sum(normals**2, axis=1)

print(f"Chi-Square from Squared Normals (df={df_test}):")
print(f"  Theoretical Mean: {df_test}")
print(f"  Sample Mean: {np.mean(chi_square_samples):.3f}")
print(f"  Theoretical Variance: {2*df_test}")
print(f"  Sample Variance: {np.var(chi_square_samples):.3f}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(chi_square_samples, bins=50, density=True, alpha=0.7, 
             edgecolor='black', label='Sum of Squared Normals')
x_plot = np.linspace(0, chi_square_samples.max(), 200)
axes[0].plot(x_plot, stats.chi2.pdf(x_plot, df_test), 'r-', 
             linewidth=2, label=f'χ²({df_test})')
axes[0].legend()
axes[0].set_title('Chi-Square as Sum of Squared Normals')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')

# Q-Q plot
stats.probplot(chi_square_samples, dist=stats.chi2(df_test), plot=axes[1])
axes[1].set_title('Q-Q Plot: Sample vs χ²(5)')

plt.tight_layout()
plt.show()

# 3. Chi-Square Test of Independence
print("\n=== Chi-Square Test of Independence ===")
# Contingency table: Treatment vs Outcome
observed = np.array([[30, 10],   # Treatment A: Success, Failure
                     [20, 40]])   # Treatment B: Success, Failure

# Perform chi-square test
chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)

print("Observed Frequencies:")
print(observed)
print("\nExpected Frequencies (under independence):")
print(expected)
print(f"\nChi-square statistic: {chi2_stat:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.4f}")
print(f"Reject H₀ (independence): {p_value < 0.05}")

# Manual calculation
row_totals = observed.sum(axis=1, keepdims=True)
col_totals = observed.sum(axis=0, keepdims=True)
total = observed.sum()
expected_manual = (row_totals @ col_totals) / total
chi2_manual = np.sum((observed - expected_manual)**2 / expected_manual)
print(f"\nManual χ² calculation: {chi2_manual:.4f}")

# 4. Goodness-of-Fit Test
print("\n=== Chi-Square Goodness-of-Fit Test ===")
# Test if dice is fair
observed_rolls = np.array([18, 15, 22, 19, 16, 10])  # Observed counts for 1-6
expected_fair = np.array([100/6] * 6)  # Expected if fair

chi2_gof = np.sum((observed_rolls - expected_fair)**2 / expected_fair)
df_gof = len(observed_rolls) - 1  # No estimated parameters
p_value_gof = 1 - stats.chi2.cdf(chi2_gof, df_gof)

print(f"Observed: {observed_rolls}")
print(f"Expected (fair): {expected_fair}")
print(f"Chi-square statistic: {chi2_gof:.4f}")
print(f"Degrees of freedom: {df_gof}")
print(f"P-value: {p_value_gof:.4f}")
print(f"Dice appears fair: {p_value_gof > 0.05}")

# 5. Confidence Interval for Variance
print("\n=== Confidence Interval for Variance ===")
data = np.random.normal(100, 15, 25)  # n=25, true σ=15
n = len(data)
sample_var = np.var(data, ddof=1)
sample_std = np.std(data, ddof=1)

# 95% CI for variance
alpha = 0.05
chi2_lower = stats.chi2.ppf(alpha/2, n-1)
chi2_upper = stats.chi2.ppf(1-alpha/2, n-1)

ci_var_lower = (n-1) * sample_var / chi2_upper
ci_var_upper = (n-1) * sample_var / chi2_lower

ci_std_lower = np.sqrt(ci_var_lower)
ci_std_upper = np.sqrt(ci_var_upper)

print(f"Sample size: {n}")
print(f"Sample variance: {sample_var:.2f}")
print(f"Sample std dev: {sample_std:.2f}")
print(f"95% CI for σ²: [{ci_var_lower:.2f}, {ci_var_upper:.2f}]")
print(f"95% CI for σ: [{ci_std_lower:.2f}, {ci_std_upper:.2f}]")
print(f"True σ² = {15**2} is in CI: {ci_var_lower <= 15**2 <= ci_var_upper}")

# 6. Power curve for chi-square test
print("\n=== Power Analysis ===")
sample_sizes = np.arange(50, 500, 20)
power_values = []

for n in sample_sizes:
    # Simulate power for detecting difference in proportions
    n_sims = 1000
    rejects = 0
    
    for _ in range(n_sims):
        # Alternative hypothesis: different proportions
        group1 = np.random.binomial(1, 0.6, n//2)
        group2 = np.random.binomial(1, 0.4, n//2)
        
        observed = np.array([[group1.sum(), len(group1)-group1.sum()],
                            [group2.sum(), len(group2)-group2.sum()]])
        
        _, p, _, _ = stats.chi2_contingency(observed)
        if p < 0.05:
            rejects += 1
    
    power_values.append(rejects / n_sims)

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, power_values, 'b-', linewidth=2)
plt.axhline(0.8, color='r', linestyle='--', label='80% Power')
plt.xlabel('Total Sample Size')
plt.ylabel('Power')
plt.title('Power Curve: Chi-Square Test of Independence')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

## 1.6 Challenge Round
When is chi-square the wrong choice?
- **Small expected frequencies (<5)**: Use Fisher's exact test or combine categories
- **Continuous outcomes**: Use t-test or ANOVA, not chi-square
- **Ordered categories**: Chi-square ignores order; use trend tests or ordinal methods
- **Paired data**: Use McNemar's test, not standard chi-square
- **Non-normal variance estimates**: Chi-square test for variance assumes normality; robust to violations only with large n

## 1.7 Key References
- [Wikipedia - Chi-Squared Distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution)
- [Wikipedia - Chi-Squared Test](https://en.wikipedia.org/wiki/Chi-squared_test)
- [Khan Academy - Chi-Square Tests](https://www.khanacademy.org/math/statistics-probability/inference-categorical-data-chi-square-tests)
- Thinking: Sum of squared normals; Right-skewed but approaches normal; Essential for categorical data and variance inference

---
**Status:** Fundamental for categorical data analysis | **Complements:** Contingency Tables, Goodness-of-Fit, Variance Testing
