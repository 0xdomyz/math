# Analysis of Variance (ANOVA)

## 1. Concept Skeleton
**Definition:** Statistical method comparing means across three or more groups simultaneously using variance decomposition and F-statistic  
**Purpose:** Test equality of means, identify group differences, partition total variance into within/between group components  
**Prerequisites:** Hypothesis testing, normal distribution, variance concepts, t-tests

## 2. Comparative Framing
| Method | One-Way ANOVA | Two-Way ANOVA | Multiple t-tests |
|--------|---------------|---------------|------------------|
| **Groups Compared** | 3+ levels, 1 factor | Multiple factors + interaction | Pairwise only |
| **Type I Error** | Controlled at α | Controlled at α | Inflates with comparisons |
| **Information** | Overall test only | Main effects + interaction | All pairwise differences |
| **Assumptions** | Normality, equal variance | Same + balanced design | Same as ANOVA |

## 3. Examples + Counterexamples

**Simple Example:**  
Compare average test scores across 4 teaching methods: ANOVA tests H₀: μ₁=μ₂=μ₃=μ₄ vs at least one differs

**Failure Case:**  
Severely unequal group sizes (n₁=5, n₂=100) with heteroscedasticity: Violates equal variance assumption, inflates Type I error

**Edge Case:**  
Significant F-test but small effect size: Statistically significant due to large n, but practically negligible group differences

## 4. Layer Breakdown
```
ANOVA Components:
├─ Variance Decomposition:
│   ├─ Total Variance: SST = Σ(yᵢⱼ - ȳ)²
│   ├─ Between-Group: SSB = Σnⱼ(ȳⱼ - ȳ)²
│   └─ Within-Group: SSW = Σ(yᵢⱼ - ȳⱼ)²
│   └─ Relationship: SST = SSB + SSW
├─ Mean Squares:
│   ├─ MSB = SSB / (k-1) [k = number of groups]
│   └─ MSW = SSW / (N-k) [N = total sample size]
├─ F-Statistic: F = MSB / MSW
│   └─ Distribution: F ~ F(k-1, N-k) under H₀
├─ Assumptions:
│   ├─ Independence: Observations independent
│   ├─ Normality: Residuals normally distributed
│   └─ Homoscedasticity: Equal variance across groups
└─ Post-hoc Tests:
    ├─ Tukey HSD: All pairwise comparisons
    ├─ Bonferroni: Conservative correction
    └─ Dunnett: Compare all to control group
```

**Interaction:** Partition variance → Compute F-ratio → Test significance → Identify specific differences

## 5. Mini-Project
One-way ANOVA with post-hoc testing:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Generate data for 4 groups with different means
np.random.seed(42)
group1 = np.random.normal(10, 2, 30)
group2 = np.random.normal(12, 2, 30)
group3 = np.random.normal(12.5, 2, 30)
group4 = np.random.normal(15, 2, 30)

# Combine data
data = np.concatenate([group1, group2, group3, group4])
groups = ['A']*30 + ['B']*30 + ['C']*30 + ['D']*30

# One-way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3, group4)

print("One-Way ANOVA Results:")
print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {p_value:.6f}")

# Effect size (eta-squared)
grand_mean = data.mean()
ssb = sum([len(g) * (g.mean() - grand_mean)**2 for g in [group1, group2, group3, group4]])
sst = sum((data - grand_mean)**2)
eta_squared = ssb / sst
print(f"Eta-squared (effect size): {eta_squared:.3f}")

# Post-hoc Tukey HSD
df = pd.DataFrame({'score': data, 'group': groups})
tukey_result = pairwise_tukeyhsd(df['score'], df['group'], alpha=0.05)
print("\nTukey HSD Post-hoc Test:")
print(tukey_result)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Box plots
axes[0, 0].boxplot([group1, group2, group3, group4], labels=['A', 'B', 'C', 'D'])
axes[0, 0].set_title('Group Comparison (Box Plot)')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_xlabel('Group')
axes[0, 0].axhline(grand_mean, color='r', linestyle='--', label='Grand Mean')
axes[0, 0].legend()

# Strip plot with means
for i, (group, label) in enumerate(zip([group1, group2, group3, group4], ['A', 'B', 'C', 'D'])):
    x = np.random.normal(i+1, 0.04, len(group))
    axes[0, 1].scatter(x, group, alpha=0.5, s=30)
    axes[0, 1].plot([i+0.8, i+1.2], [group.mean(), group.mean()], 'r-', linewidth=3)

axes[0, 1].set_xticks([1, 2, 3, 4])
axes[0, 1].set_xticklabels(['A', 'B', 'C', 'D'])
axes[0, 1].set_title('Individual Data Points with Means')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_xlabel('Group')

# Residuals vs Fitted
fitted_values = np.concatenate([
    np.full(30, group1.mean()),
    np.full(30, group2.mean()),
    np.full(30, group3.mean()),
    np.full(30, group4.mean())
])
residuals = data - fitted_values

axes[1, 0].scatter(fitted_values, residuals, alpha=0.6)
axes[1, 0].axhline(0, color='r', linestyle='--')
axes[1, 0].set_title('Residuals vs Fitted Values')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('Residuals')

# Q-Q plot for normality
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Check)')

plt.tight_layout()
plt.show()

# Check assumptions
# Levene's test for homogeneity of variance
levene_stat, levene_p = stats.levene(group1, group2, group3, group4)
print(f"\nLevene's Test for Equal Variance:")
print(f"Statistic: {levene_stat:.3f}, P-value: {levene_p:.4f}")
print("Equal variances assumption:", "OK" if levene_p > 0.05 else "VIOLATED")

# Shapiro-Wilk test for normality
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\nShapiro-Wilk Test for Normality:")
print(f"Statistic: {shapiro_stat:.3f}, P-value: {shapiro_p:.4f}")
print("Normality assumption:", "OK" if shapiro_p > 0.05 else "VIOLATED")
```

## 6. Challenge Round
When is ANOVA the wrong tool?
- Only 2 groups: Use t-test (simpler, equivalent)
- Ordinal outcomes: Use Kruskal-Wallis (non-parametric alternative)
- Repeated measures: Use repeated measures ANOVA or mixed models
- Severely non-normal data: Use transformation or non-parametric tests
- Different research question: ANOVA tests "any difference", not specific patterns

## 7. Key References
- [ANOVA Explained (Wikipedia)](https://en.wikipedia.org/wiki/Analysis_of_variance)
- [One-Way ANOVA (Khan Academy)](https://www.khanacademy.org/math/statistics-probability/analysis-of-variance-anova-library)
- [statsmodels ANOVA Documentation](https://www.statsmodels.org/stable/anova.html)

---
**Status:** Core multi-group comparison method | **Complements:** t-tests, Regression, Post-hoc Tests
