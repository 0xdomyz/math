# Rank-Based Tests

## 1. Concept Skeleton
**Definition:** Hypothesis tests using data ranks rather than raw values to compare groups or measure association  
**Purpose:** Robust alternatives to t-tests/ANOVA when normality violated or ordinal data present  
**Prerequisites:** Hypothesis testing framework, median concepts, basic sorting algorithms

## 2. Comparative Framing
| Test | Mann-Whitney U | Wilcoxon Signed-Rank | Kruskal-Wallis H |
|------|----------------|---------------------|-----------------|
| **Parametric Equivalent** | Independent t-test | Paired t-test | One-way ANOVA |
| **Groups** | 2 independent | 1 paired/repeated | 3+ independent |
| **Hypothesis** | Median₁ ≠ Median₂ | Median difference ≠ 0 | All medians equal |
| **Test Statistic** | U (sum of ranks) | W (signed-rank sum) | H (chi-square approx) |

## 3. Examples + Counterexamples

**Simple Example:**  
Drug A vs Placebo recovery days: [5,7,8,10,15] vs [12,14,16,18,22]. Rank all: 1,2,3,4,5,6,7,8,9,10. Mann-Whitney compares rank sums.

**Failure Case:**  
Many tied ranks with small sample. Ties reduce test power; correction formulas needed but may be unstable.

**Edge Case:**  
Zero differences in Wilcoxon signed-rank. Must exclude zeros before ranking, reducing effective sample size.

## 4. Layer Breakdown
```
Rank-Based Test Framework:
├─ Mann-Whitney U Test (Independent Samples):
│   ├─ Step 1: Combine groups, rank all values
│   ├─ Step 2: Sum ranks for each group (R₁, R₂)
│   ├─ Step 3: U = n₁n₂ + n₁(n₁+1)/2 - R₁
│   ├─ Step 4: Compare to U distribution or normal approx
│   └─ Interpretation: Smaller U → group 1 tends lower
├─ Wilcoxon Signed-Rank (Paired Data):
│   ├─ Step 1: Compute differences d = x₁ - x₂
│   ├─ Step 2: Rank absolute differences |d|
│   ├─ Step 3: Restore signs, sum positive ranks (W⁺)
│   ├─ Step 4: Compare to Wilcoxon distribution
│   └─ Handles: Before-after, matched pairs
├─ Kruskal-Wallis H (Multiple Groups):
│   ├─ Extension of Mann-Whitney to 3+ groups
│   ├─ H = [12/(N(N+1))] Σ(R²ᵢ/nᵢ) - 3(N+1)
│   ├─ Approximately chi-square with k-1 df
│   └─ Post-hoc: Dunn's test for pairwise comparisons
└─ Tie Corrections:
    └─ Adjust formulas when many identical values present
```

## 5. Mini-Project
Perform Mann-Whitney and Wilcoxon tests:
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Mann-Whitney U Test (Independent samples)
np.random.seed(42)
control = np.random.lognormal(2, 0.5, 25)  # Skewed data
treatment = np.random.lognormal(2.3, 0.5, 25)

u_stat, u_pval = stats.mannwhitneyu(control, treatment, alternative='two-sided')
print("Mann-Whitney U Test:")
print(f"  U-statistic: {u_stat:.2f}")
print(f"  p-value: {u_pval:.4f}")
print(f"  Interpretation: {'Reject H₀' if u_pval < 0.05 else 'Fail to reject H₀'}")

# Wilcoxon Signed-Rank Test (Paired samples)
before = np.array([85, 92, 78, 88, 95, 82, 79, 90, 87, 93])
after = np.array([87, 95, 80, 90, 97, 84, 81, 93, 89, 96])

w_stat, w_pval = stats.wilcoxon(before, after, alternative='two-sided')
print("\nWilcoxon Signed-Rank Test:")
print(f"  W-statistic: {w_stat:.2f}")
print(f"  p-value: {w_pval:.4f}")
print(f"  Interpretation: {'Significant improvement' if w_pval < 0.05 else 'No significant change'}")

# Kruskal-Wallis H Test (Multiple groups)
group_a = np.random.exponential(2, 20)
group_b = np.random.exponential(2.5, 20)
group_c = np.random.exponential(3, 20)

h_stat, h_pval = stats.kruskal(group_a, group_b, group_c)
print("\nKruskal-Wallis H Test:")
print(f"  H-statistic: {h_stat:.2f}")
print(f"  p-value: {h_pval:.4f}")
print(f"  Interpretation: {'Groups differ' if h_pval < 0.05 else 'No significant difference'}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Mann-Whitney
axes[0].boxplot([control, treatment], labels=['Control', 'Treatment'])
axes[0].set_title(f'Mann-Whitney U\np={u_pval:.4f}')
axes[0].set_ylabel('Value')

# Wilcoxon
axes[1].plot(before, 'o-', label='Before', alpha=0.6)
axes[1].plot(after, 's-', label='After', alpha=0.6)
for i in range(len(before)):
    axes[1].plot([i, i], [before[i], after[i]], 'k-', alpha=0.3)
axes[1].set_title(f'Wilcoxon Signed-Rank\np={w_pval:.4f}')
axes[1].set_xlabel('Pair')
axes[1].legend()

# Kruskal-Wallis
axes[2].boxplot([group_a, group_b, group_c], labels=['A', 'B', 'C'])
axes[2].set_title(f'Kruskal-Wallis H\np={h_pval:.4f}')
axes[2].set_ylabel('Value')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When are rank-based tests inappropriate?
- Need to estimate actual mean difference (ranks lose magnitude info)
- Censored data with complex patterns (use survival analysis)
- Multivariate outcomes (use MANOVA or permutation tests)
- Longitudinal data with missing values (use mixed models)
- Require confidence intervals for medians (use bootstrap instead)

## 7. Key References
- [Mann-Whitney U Test Explained](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
- [Wilcoxon Signed-Rank Assumptions](https://statistics.laerd.com/statistical-guides/wilcoxon-signed-rank-test-statistical-guide.php)
- [Kruskal-Wallis vs ANOVA](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/kruskal-wallis/)

---
**Status:** Robust hypothesis testing | **Complements:** Non-Parametric Statistics, Hypothesis Testing, Effect Size
