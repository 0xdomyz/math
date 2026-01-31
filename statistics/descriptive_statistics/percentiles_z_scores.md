# Percentiles & Z-scores

## 5.1 Concept Skeleton
**Definition:** Percentiles indicate position in sorted distribution; Z-scores measure distance from mean in SD units  
**Purpose:** Standardize comparisons, locate values, assess extremity  
**Prerequisites:** Mean, standard deviation, normal distribution

## 5.2 Comparative Framing
| Measure | Percentile | Z-score | Rank |
|---------|-----------|---------|------|
| **Scale** | 0-100 (percentage) | Standardized (SD units) | 1 to n (ordinal) |
| **Distribution** | Non-parametric (any) | Assumes normality | Distribution-free |
| **Interpretation** | % below value | Distance from mean | Position in order |

## 5.3 Examples + Counterexamples

**Simple Example:**  
Score 85 in exam. 90th percentile (beat 90%), Z-score +1.5 (1.5 SD above mean)

**Failure Case:**  
Non-normal data: Z-score ±3 doesn't correspond to expected percentile (0.13% vs actual)

**Edge Case:**  
Tied values: Multiple values at same percentile rank (use midpoint or interpolation)

## 5.4 Layer Breakdown
```
Position Measures:
├─ Percentiles:
│   ├─ Definition: Value below which P% of data falls
│   ├─ Special Cases:
│   │   ├─ 25th: Q1 (lower quartile)
│   │   ├─ 50th: Median
│   │   └─ 75th: Q3 (upper quartile)
│   ├─ Calculation: Sort data, find position P×(n+1)/100
│   └─ Use: Compare across different scales/distributions
├─ Z-scores:
│   ├─ Formula: z = (x - μ) / σ
│   ├─ Properties:
│   │   ├─ Mean of z-scores = 0
│   │   ├─ SD of z-scores = 1
│   │   └─ Dimensionless (standardized)
│   ├─ Interpretation:
│   │   ├─ z = 0: At mean
│   │   ├─ z = +1: 1 SD above mean (~84th percentile)
│   │   └─ z = -2: 2 SD below mean (~2.3rd percentile)
│   └─ Use: Compare across different measurements
└─ Standard Normal Distribution:
    ├─ Z ~ N(0, 1)
    ├─ Z-table: Convert z to percentile/probability
    └─ Empirical Rule: 68-95-99.7 rule
```

## 5.5 Mini-Project
Calculate percentiles and z-scores:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
# Generate exam scores
scores = np.random.normal(75, 10, 200)

def analyze_position(data, value):
    # Percentile calculation
    percentile = stats.percentileofscore(data, value)
    
    # Z-score calculation
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    z_score = (value - mean) / std
    
    # Percentile from z-score (assuming normality)
    percentile_from_z = stats.norm.cdf(z_score) * 100
    
    print(f"\nValue: {value:.2f}")
    print(f"  Mean: {mean:.2f}, SD: {std:.2f}")
    print(f"  Actual Percentile: {percentile:.2f}%")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  Percentile from Z (normal): {percentile_from_z:.2f}%")
    
    return percentile, z_score

# Test specific scores
test_values = [65, 75, 85, 95]
for val in test_values:
    analyze_position(scores, val)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Distribution with percentiles
axes[0, 0].hist(scores, bins=30, alpha=0.7, edgecolor='black', density=True)
mean = np.mean(scores)
std = np.std(scores, ddof=1)
x = np.linspace(mean - 4*std, mean + 4*std, 100)
axes[0, 0].plot(x, stats.norm.pdf(x, mean, std), 'r-', lw=2)
for p in [25, 50, 75]:
    val = np.percentile(scores, p)
    axes[0, 0].axvline(val, linestyle='--', label=f'{p}th %ile')
axes[0, 0].legend()
axes[0, 0].set_title('Distribution with Percentiles')
axes[0, 0].set_xlabel('Score')

# Z-score transformation
z_scores = (scores - mean) / std
axes[0, 1].hist(z_scores, bins=30, alpha=0.7, edgecolor='black', density=True)
x_z = np.linspace(-4, 4, 100)
axes[0, 1].plot(x_z, stats.norm.pdf(x_z, 0, 1), 'r-', lw=2)
axes[0, 1].axvline(0, color='g', linestyle='--', label='Mean (z=0)')
axes[0, 1].legend()
axes[0, 1].set_title('Standardized Scores (Z-distribution)')
axes[0, 1].set_xlabel('Z-score')

# Percentile plot
percentiles = np.arange(1, 100)
values = np.percentile(scores, percentiles)
axes[1, 0].plot(percentiles, values)
axes[1, 0].set_title('Percentile Function')
axes[1, 0].set_xlabel('Percentile')
axes[1, 0].set_ylabel('Score')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot (check normality)
stats.probplot(scores, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Check)')

plt.tight_layout()
plt.show()

# Common z-score percentiles
print("\nZ-score to Percentile Reference:")
for z in [-3, -2, -1, 0, 1, 2, 3]:
    p = stats.norm.cdf(z) * 100
    print(f"  z = {z:+.0f}  →  {p:.2f}th percentile")
```

## 5.6 Challenge Round
When are these measures the wrong choice?
- **Percentiles**: Discrete data with many ties (use cumulative distribution function)
- **Z-scores**: Non-normal distributions (use percentile ranks or robust z-scores)
- **Both**: Ordinal data without meaningful intervals (use ranks only)
- **Z-scores**: Outliers inflate SD, distorting z-values (use modified z-scores with MAD)
- **Comparing percentiles across groups**: Interpretation depends on group distribution shape

## 5.7 Key References
- [Khan Academy - Z-scores](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/z-scores)
- [Wikipedia - Percentile](https://en.wikipedia.org/wiki/Percentile)
- [Wikipedia - Standard Score](https://en.wikipedia.org/wiki/Standard_score)
- Thinking: Z-scores assume normality; Percentiles distribution-free; Both enable cross-scale comparisons

---
**Status:** Standardization foundation | **Complements:** Normal Distribution, Confidence Intervals, Hypothesis Testing
