# Non-Parametric Statistics

## 1. Concept Skeleton
**Definition:** Statistical methods not requiring assumptions about population distribution parameters  
**Purpose:** Analyze data when parametric assumptions violated or ordinal/nominal data used  
**Prerequisites:** Basic hypothesis testing, distribution concepts, median/quartiles

## 2. Comparative Framing
| Method | Parametric | Non-Parametric | Robust Statistics |
|--------|-----------|----------------|------------------|
| **Assumptions** | Normal distribution, known parameters | Distribution-free | Resistant to outliers |
| **Data Type** | Interval/ratio, continuous | Ordinal, ranks, any | Continuous with contamination |
| **Power** | Higher when assumptions met | Lower but consistent | Moderate |
| **Example** | t-test, ANOVA | Mann-Whitney, Kruskal-Wallis | Trimmed mean, MAD |

## 3. Examples + Counterexamples

**Simple Example:**  
Survey ranks satisfaction (1-5 scale). Use Mann-Whitney U instead of t-test since ordinal data violates normality assumption.

**Failure Case:**  
Large sample (n>1000) with near-normal data. Parametric test has more power to detect small effects; non-parametric wastes information.

**Edge Case:**  
Data with many ties (repeated ranks). Requires tie-correction formulas; standard rank methods may fail.

## 4. Layer Breakdown
```
Non-Parametric Framework:
├─ Core Principle: Use ranks/order instead of raw values
├─ Hypothesis Tests:
│   ├─ Location: Mann-Whitney U, Wilcoxon signed-rank
│   ├─ Multiple Groups: Kruskal-Wallis, Friedman
│   ├─ Association: Spearman's ρ, Kendall's τ
│   └─ Goodness-of-fit: Kolmogorov-Smirnov, runs test
├─ Advantages:
│   ├─ No distribution assumptions
│   ├─ Robust to outliers
│   ├─ Works with ordinal data
│   └─ Valid for small samples
└─ Limitations:
    ├─ Less powerful when parametric valid
    ├─ Harder to interpret effect sizes
    └─ Limited software for complex designs
```

## 5. Mini-Project
Compare parametric vs non-parametric on skewed data:
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Generate skewed data (two groups)
np.random.seed(42)
group1 = np.random.exponential(2, 30)  # Skewed distribution
group2 = np.random.exponential(2.8, 30)

# Parametric test (assumes normality)
t_stat, t_pval = stats.ttest_ind(group1, group2)

# Non-parametric test (distribution-free)
u_stat, u_pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')

print(f"t-test p-value: {t_pval:.4f}")
print(f"Mann-Whitney U p-value: {u_pval:.4f}")

# Visualize distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(group1, bins=10, alpha=0.5, label='Group 1')
axes[0].hist(group2, bins=10, alpha=0.5, label='Group 2')
axes[0].set_title('Raw Data (Skewed)')
axes[0].legend()

axes[1].boxplot([group1, group2], labels=['Group 1', 'Group 2'])
axes[1].set_title('Boxplot Comparison')
axes[1].set_ylabel('Value')

plt.tight_layout()
plt.show()

# Check normality assumption
stat1, norm_p1 = stats.shapiro(group1)
stat2, norm_p2 = stats.shapiro(group2)
print(f"\nShapiro-Wilk normality test:")
print(f"Group 1 p-value: {norm_p1:.4f} {'(Normal)' if norm_p1 > 0.05 else '(Not Normal)'}")
print(f"Group 2 p-value: {norm_p2:.4f} {'(Normal)' if norm_p2 > 0.05 else '(Not Normal)'}")
```

## 6. Challenge Round
When are non-parametric methods the wrong choice?
- Large samples with near-normal data (parametric more powerful)
- Need specific parameter estimates (mean, variance)
- Complex models with covariates (use GLM or robust regression instead)
- Require precise confidence intervals for parameters
- Interaction effects critical (parametric models handle better)

## 7. Key References
- [Wikipedia: Nonparametric Statistics](https://en.wikipedia.org/wiki/Nonparametric_statistics)
- [Practical Guide to Choosing Tests](https://www.statmethods.net/stats/nonparametric.html)
- [When to Use Non-Parametric Tests](https://statistics.laerd.com/statistical-guides/types-of-variable.php)

---
**Status:** Distribution-free foundation | **Complements:** Hypothesis Testing, Rank-Based Tests, Robust Methods
