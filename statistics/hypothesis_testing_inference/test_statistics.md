# Test Statistics (t, z, χ², F)

## 1.1 Concept Skeleton
**Definition:** Standardized measures converting sample data to values on known probability distributions for hypothesis testing  
**Purpose:** Quantify evidence against null hypothesis; determine p-values via reference distribution  
**Prerequisites:** Probability distributions, standard error, degrees of freedom, hypothesis testing framework

## 1.2 Comparative Framing
| Statistic | z-statistic | t-statistic | χ² (Chi-square) | F-statistic |
|-----------|------------|-------------|----------------|-------------|
| **Use Case** | Known σ, large n | Unknown σ, small n | Categorical data, variance tests | Compare variances, ANOVA |
| **Distribution** | Standard Normal N(0,1) | t-distribution (df) | χ² distribution (df) | F-distribution (df1, df2) |
| **Formula** | (x̄ - μ₀)/(σ/√n) | (x̄ - μ₀)/(s/√n) | Σ(O - E)²/E | Variance₁/Variance₂ |
| **Sensitivity** | Less conservative | Wider tails (small n) | Right-skewed | Right-skewed |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Test μ = 100 with sample x̄=105, s=15, n=25. t = (105-100)/(15/√25) = 5/3 = 1.67, df=24. Compare to t-table

**Failure Case:**  
Using z-test with n=10 and unknown σ. Underestimates variability → inflated Type I error. Must use t-test

**Edge Case:**  
χ² goodness-of-fit with expected counts <5. Distribution approximation breaks down → use Fisher's exact test or combine categories

## 1.4 Layer Breakdown
```
Test Statistics Ecosystem:
├─ z-statistic (Standard Normal):
│   ├─ When: σ known OR n ≥ 30 (CLT)
│   ├─ Formula: z = (x̄ - μ₀)/(σ/√n)
│   ├─ Distribution: N(0,1) always
│   └─ Critical values: ±1.96 (α=0.05, two-tailed)
├─ t-statistic (Student's t):
│   ├─ When: σ unknown AND n < 30
│   ├─ Formula: t = (x̄ - μ₀)/(s/√n)
│   ├─ Distribution: t(df=n-1)
│   ├─ Approaches z as n→∞
│   └─ Variants: One-sample, two-sample, paired
├─ χ² (Chi-square):
│   ├─ Goodness-of-fit: Compare observed to expected frequencies
│   │   └─ χ² = Σ(Oᵢ - Eᵢ)²/Eᵢ, df = categories - 1
│   ├─ Independence: Test association in contingency table
│   │   └─ χ² = ΣΣ(Oᵢⱼ - Eᵢⱼ)²/Eᵢⱼ, df = (r-1)(c-1)
│   └─ Variance test: Test σ² against σ₀²
│       └─ χ² = (n-1)s²/σ₀², df = n-1
└─ F-statistic (Fisher's F):
    ├─ Variance ratio: F = s₁²/s₂²
    ├─ ANOVA: F = MSbetween/MSwithin
    ├─ Distribution: F(df1, df2)
    └─ Regression: F = (R²/k)/((1-R²)/(n-k-1))
```

## 1.5 Mini-Project
Compare test statistics across scenarios:
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Scenario 1: z-test (known σ)
print("=== z-test Example ===")
population_sigma = 15
sample = np.random.normal(105, population_sigma, 100)
x_bar = np.mean(sample)
n = len(sample)
mu_0 = 100

z_stat = (x_bar - mu_0) / (population_sigma / np.sqrt(n))
p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
print(f"z = {z_stat:.3f}, p-value = {p_value_z:.4f}")

# Scenario 2: t-test (unknown σ)
print("\n=== t-test Example ===")
small_sample = np.random.normal(105, 15, 25)
t_stat, p_value_t = stats.ttest_1samp(small_sample, mu_0)
print(f"t = {t_stat:.3f}, p-value = {p_value_t:.4f}")
print(f"df = {len(small_sample)-1}")

# Scenario 3: χ² test (independence)
print("\n=== χ² Test Example ===")
# Contingency table: Treatment vs Outcome
observed = np.array([[30, 10], [20, 40]])
chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(observed)
print(f"χ² = {chi2_stat:.3f}, p-value = {p_value_chi2:.4f}, df = {dof}")
print("Expected frequencies:")
print(expected)

# Scenario 4: F-test (ANOVA)
print("\n=== F-test Example ===")
group1 = np.random.normal(100, 15, 30)
group2 = np.random.normal(105, 15, 30)
group3 = np.random.normal(110, 15, 30)
f_stat, p_value_f = stats.f_oneway(group1, group2, group3)
print(f"F = {f_stat:.3f}, p-value = {p_value_f:.4f}")

# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: z-distribution
x = np.linspace(-4, 4, 200)
axes[0, 0].plot(x, stats.norm.pdf(x), 'b-', label='Standard Normal')
axes[0, 0].axvline(z_stat, color='r', linestyle='--', label=f'z={z_stat:.2f}')
axes[0, 0].axvline(-z_stat, color='r', linestyle='--')
axes[0, 0].fill_between(x, stats.norm.pdf(x), where=(np.abs(x) >= abs(z_stat)), 
                         alpha=0.3, color='red', label=f'p={p_value_z:.4f}')
axes[0, 0].set_title('z-test Distribution')
axes[0, 0].legend()
axes[0, 0].set_xlabel('z-value')
axes[0, 0].set_ylabel('Density')

# Plot 2: t-distribution comparison
df_values = [1, 5, 24, 100]
x = np.linspace(-4, 4, 200)
for df in df_values:
    if df == 100:
        axes[0, 1].plot(x, stats.norm.pdf(x), label=f'z (df=∞)', linewidth=2)
    else:
        axes[0, 1].plot(x, stats.t.pdf(x, df), label=f't (df={df})')
axes[0, 1].axvline(t_stat, color='r', linestyle='--', label=f'Observed t={t_stat:.2f}')
axes[0, 1].set_title('t-distribution vs z (Heavier Tails)')
axes[0, 1].legend()
axes[0, 1].set_xlabel('t-value')
axes[0, 1].set_ylabel('Density')

# Plot 3: χ² distribution
x = np.linspace(0, 20, 200)
df_values = [1, 2, 3, 5, 10]
for df in df_values:
    axes[1, 0].plot(x, stats.chi2.pdf(x, df), label=f'df={df}')
axes[1, 0].axvline(chi2_stat, color='r', linestyle='--', label=f'Observed χ²={chi2_stat:.2f}')
axes[1, 0].set_title('χ² Distribution (Right-Skewed)')
axes[1, 0].legend()
axes[1, 0].set_xlabel('χ² value')
axes[1, 0].set_ylabel('Density')

# Plot 4: F-distribution
x = np.linspace(0, 5, 200)
df_pairs = [(1, 10), (5, 20), (10, 50), (30, 100)]
for df1, df2 in df_pairs:
    axes[1, 1].plot(x, stats.f.pdf(x, df1, df2), label=f'df1={df1}, df2={df2}')
axes[1, 1].axvline(f_stat, color='r', linestyle='--', label=f'Observed F={f_stat:.2f}')
axes[1, 1].set_title('F-distribution (Variance Ratios)')
axes[1, 1].legend()
axes[1, 1].set_xlabel('F-value')
axes[1, 1].set_ylabel('Density')

plt.tight_layout()
plt.show()

# Critical values comparison
print("\n=== Critical Values (α=0.05, two-tailed) ===")
print(f"z: ±{stats.norm.ppf(0.975):.3f}")
print(f"t (df=24): ±{stats.t.ppf(0.975, 24):.3f}")
print(f"χ² (df=3, upper): {stats.chi2.ppf(0.95, 3):.3f}")
print(f"F (df1=2, df2=87, upper): {stats.f.ppf(0.95, 2, 87):.3f}")
```

## 1.6 Challenge Round
When is each test statistic inappropriate?
- **z-test**: Small sample (n<30) with unknown σ → overconfident; use t-test instead
- **t-test**: Severely non-normal data with small n → bootstrap or non-parametric tests
- **χ²**: Expected frequencies <5 → Fisher's exact test; continuous data → binning loses information
- **F-test**: Unequal variances in ANOVA → Welch's ANOVA; non-normal residuals → transformations or GLM
- **All**: Dependent observations (time series, clusters) → specialized methods (GEE, mixed models)

## 1.7 Key References
- [Test Statistics Overview](https://en.wikipedia.org/wiki/Test_statistic) - Definitions, distributions, relationship to hypothesis tests
- [t vs z Decision Guide](https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/t-score-vs-z-score/) - When to use each, worked examples
- [Chi-square and F-distributions](https://www.stat.yale.edu/Courses/1997-98/101/chigf.htm) - Visual comparison, degrees of freedom effects

---
**Status:** Foundation for all hypothesis tests | **Complements:** Hypothesis Testing, P-values, Confidence Intervals
