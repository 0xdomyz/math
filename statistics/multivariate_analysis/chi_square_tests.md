# Chi-Square Tests

## 1. Concept Skeleton
**Definition:** Non-parametric tests comparing observed vs expected frequencies in categorical data using χ² distribution  
**Purpose:** Test independence between variables, goodness-of-fit to theoretical distribution, homogeneity across groups  
**Prerequisites:** Categorical data concepts, probability distributions, hypothesis testing, contingency tables

## 2. Comparative Framing
| Test Type | Goodness-of-Fit | Test of Independence | Fisher's Exact Test |
|-----------|-----------------|---------------------|---------------------|
| **Purpose** | One variable vs theory | Association between 2 variables | Independence (small samples) |
| **Data Structure** | Frequency counts | Contingency table (2D) | 2×2 table |
| **Sample Size** | All expected ≥5 | All expected ≥5 | Any size (exact) |
| **Computation** | Fast (asymptotic) | Fast (asymptotic) | Slow (combinatorial) |

## 3. Examples + Counterexamples

**Simple Example:**  
Fair die test: Roll 600 times, expect 100 per face. Observed: [98, 105, 92, 110, 89, 106]. χ² test checks if deviations due to chance.

**Failure Case:**  
2×2 table with expected cell count of 2: Violates minimum expected frequency requirement, χ² approximation invalid → use Fisher's exact

**Edge Case:**  
Large sample finds statistically significant association with Cramér's V=0.05: Trivial practical effect despite low p-value

## 4. Layer Breakdown
```
Chi-Square Test Components:
├─ Test Statistic:
│   └─ χ² = Σ[(Observed - Expected)² / Expected]
├─ Goodness-of-Fit Test:
│   ├─ H₀: Data follows specified distribution
│   ├─ Expected: Theoretical probabilities × n
│   └─ df = categories - 1 - parameters estimated
├─ Independence Test:
│   ├─ H₀: Row and column variables independent
│   ├─ Expected: (row total × column total) / grand total
│   └─ df = (rows - 1) × (columns - 1)
├─ Assumptions:
│   ├─ Random sampling
│   ├─ Expected frequency ≥5 in 80%+ cells
│   ├─ No expected frequency <1
│   └─ Independent observations
├─ Effect Size Measures:
│   ├─ Cramér's V: √(χ²/n×min(r-1,c-1))
│   ├─ Phi coefficient: For 2×2 tables
│   └─ Odds Ratio: For 2×2 tables
└─ Post-hoc Analysis:
    └─ Standardized residuals: (O-E)/√E identify contributing cells
```

**Interaction:** Count frequencies → Calculate expected → Compute χ² → Assess significance → Interpret patterns

## 5. Mini-Project
Chi-square tests for goodness-of-fit and independence:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chisquare, fisher_exact
import pandas as pd
import seaborn as sns

# Example 1: Goodness-of-Fit Test (Fair die)
print("=" * 50)
print("Chi-Square Goodness-of-Fit Test")
print("=" * 50)

observed_rolls = np.array([98, 105, 92, 110, 89, 106])
expected_prob = np.array([1/6] * 6)
n_rolls = observed_rolls.sum()
expected_rolls = expected_prob * n_rolls

chi2_stat, p_value = chisquare(observed_rolls, expected_rolls)

print(f"Observed: {observed_rolls}")
print(f"Expected: {expected_rolls}")
print(f"χ² statistic: {chi2_stat:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Conclusion: {'Fair die' if p_value > 0.05 else 'Biased die'} (α=0.05)")

# Example 2: Test of Independence (Gender vs Preference)
print("\n" + "=" * 50)
print("Chi-Square Test of Independence")
print("=" * 50)

# Contingency table: Rows=Gender, Columns=Product Preference
observed = np.array([
    [30, 45, 25],  # Male
    [50, 35, 15]   # Female
])

chi2, p_value, dof, expected = chi2_contingency(observed)

print("Observed frequencies:")
print(pd.DataFrame(observed, 
                   index=['Male', 'Female'], 
                   columns=['Product A', 'Product B', 'Product C']))
print("\nExpected frequencies:")
print(pd.DataFrame(expected, 
                   index=['Male', 'Female'], 
                   columns=['Product A', 'Product B', 'Product C']))

print(f"\nχ² statistic: {chi2:.3f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.4f}")

# Effect size (Cramér's V)
n = observed.sum()
min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"Cramér's V (effect size): {cramers_v:.3f}")

# Standardized residuals
std_residuals = (observed - expected) / np.sqrt(expected)
print("\nStandardized Residuals (|z|>2 indicates significant contribution):")
print(pd.DataFrame(std_residuals, 
                   index=['Male', 'Female'], 
                   columns=['Product A', 'Product B', 'Product C']).round(2))

# Example 3: Fisher's Exact Test (small sample)
print("\n" + "=" * 50)
print("Fisher's Exact Test (Small Sample)")
print("=" * 50)

small_table = np.array([
    [3, 1],  # Treatment success/failure
    [1, 4]   # Control success/failure
])

oddsratio, p_value_fisher = fisher_exact(small_table)

print("2×2 Contingency Table:")
print(pd.DataFrame(small_table, 
                   index=['Treatment', 'Control'], 
                   columns=['Success', 'Failure']))
print(f"\nOdds Ratio: {oddsratio:.3f}")
print(f"P-value (two-tailed): {p_value_fisher:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Observed vs Expected (Die rolls)
x_pos = np.arange(1, 7)
axes[0, 0].bar(x_pos - 0.2, observed_rolls, width=0.4, label='Observed', alpha=0.8)
axes[0, 0].bar(x_pos + 0.2, expected_rolls, width=0.4, label='Expected', alpha=0.8)
axes[0, 0].set_xlabel('Die Face')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Goodness-of-Fit: Observed vs Expected')
axes[0, 0].legend()
axes[0, 0].set_xticks(x_pos)

# Plot 2: Heatmap of observed frequencies
sns.heatmap(observed, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Product A', 'Product B', 'Product C'],
            yticklabels=['Male', 'Female'], ax=axes[0, 1], cbar_kws={'label': 'Count'})
axes[0, 1].set_title('Observed Frequencies')

# Plot 3: Heatmap of standardized residuals
sns.heatmap(std_residuals, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=['Product A', 'Product B', 'Product C'],
            yticklabels=['Male', 'Female'], ax=axes[1, 0], 
            cbar_kws={'label': 'Std Residual'})
axes[1, 0].set_title('Standardized Residuals\n(Red=More than expected, Blue=Less)')

# Plot 4: Mosaic-style representation
from matplotlib.patches import Rectangle

ax = axes[1, 1]
row_totals = observed.sum(axis=1)
col_totals = observed.sum(axis=0)
grand_total = observed.sum()

# Normalized positions
y_positions = np.cumsum([0] + list(row_totals / grand_total))
colors = ['lightblue', 'lightcoral', 'lightgreen']

for i in range(observed.shape[0]):
    x_start = 0
    for j in range(observed.shape[1]):
        width = observed[i, j] / row_totals[i]
        height = row_totals[i] / grand_total
        rect = Rectangle((x_start, y_positions[i]), width, height, 
                         facecolor=colors[j], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        # Add text
        ax.text(x_start + width/2, y_positions[i] + height/2, 
                f'{observed[i,j]}', ha='center', va='center', fontsize=10)
        x_start += width

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_yticks([y_positions[i] + (y_positions[i+1] - y_positions[i])/2 
               for i in range(len(row_totals))])
ax.set_yticklabels(['Male', 'Female'])
ax.set_xticks([])
ax.set_title('Mosaic Plot (Width ∝ conditional proportion)')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=f'Product {chr(65+i)}') 
                   for i in range(3)]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is chi-square the wrong tool?
- Small expected frequencies (<5): Use Fisher's exact or combine categories
- Continuous outcomes: Use ANOVA or regression
- Matched pairs: Use McNemar's test
- Ordinal categories: Consider ordinal tests (Cochran-Armitage trend)
- Need causal inference: Chi-square tests association, not causation

## 7. Key References
- [Chi-Square Test (Wikipedia)](https://en.wikipedia.org/wiki/Chi-squared_test)
- [Contingency Table Analysis](https://www.stat.berkeley.edu/~stark/SticiGui/Text/chiSquare.htm)
- [scipy.stats Chi-Square Functions](https://docs.scipy.org/doc/scipy/reference/stats.html#contingency-table-functions)

---
**Status:** Essential categorical data analysis | **Complements:** Fisher's Exact, Logistic Regression, Cramér's V
