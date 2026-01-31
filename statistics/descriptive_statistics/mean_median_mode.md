# Mean, Median, Mode

## 1.1 Concept Skeleton
**Definition:** Central tendency measures indicating typical/center value in dataset  
**Purpose:** Summarize location, compare groups, detect distribution shape  
**Prerequisites:** Basic arithmetic, data types (continuous vs discrete)

## 1.2 Comparative Framing
| Measure | Mean | Median | Mode |
|---------|------|--------|------|
| **Definition** | Sum/count | Middle value | Most frequent |
| **Outlier Sensitivity** | High (pulled by extremes) | Low (resistant) | None (purely frequency) |
| **Use Case** | Symmetric distributions | Skewed data | Categorical data |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Dataset: [2, 3, 3, 5, 7, 10]. Mean = 5, Median = 4, Mode = 3

**Failure Case:**  
Income data: [30k, 35k, 40k, 500k]. Mean = 151k misrepresents typical income (use median)

**Edge Case:**  
Bimodal: [1, 1, 1, 10, 10, 10]. Two modes exist; mean/median miss dual peaks

## 1.4 Layer Breakdown
```
Central Tendency Components:
├─ Mean (μ or x̄):
│   ├─ Formula: Σxᵢ / n
│   ├─ Properties: Minimizes squared deviations
│   └─ Affected by: Every data point, especially extremes
├─ Median (M):
│   ├─ Formula: Middle value when sorted (or average of two middle)
│   ├─ Properties: 50th percentile, resistant statistic
│   └─ Skewness: Mean < Median (left skew), Mean > Median (right skew)
└─ Mode:
    ├─ Definition: Peak of frequency distribution
    ├─ Properties: Can have multiple modes (bimodal, multimodal)
    └─ Application: Best for categorical data
```

## 1.5 Mini-Project
Calculate and compare measures:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Symmetric data
symmetric = np.random.normal(50, 10, 1000)
# Right-skewed data
skewed = np.random.exponential(20, 1000)

def analyze_central_tendency(data, name):
    mean = np.mean(data)
    median = np.median(data)
    mode = stats.mode(data, keepdims=True)[0][0]
    
    print(f"\n{name} Distribution:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Median: {median:.2f}")
    print(f"  Mode: {mode:.2f}")
    
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
    plt.legend()
    plt.title(f'{name} Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

analyze_central_tendency(symmetric, "Symmetric")
analyze_central_tendency(skewed, "Right-Skewed")
```

## 1.6 Challenge Round
When are these measures the wrong choice?
- **Mean**: Skewed distributions, outliers present (use trimmed mean or median)
- **Median**: Need mathematical properties (sum of deviations) or theoretical distribution
- **Mode**: Continuous data with no repeats (use kernel density peak)
- **All three**: Uniform distribution (no meaningful center), multimodal (need multiple summaries)

## 1.7 Key References
- [Khan Academy - Measures of Center](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/mean-median-basics)
- [Wikipedia - Central Tendency](https://en.wikipedia.org/wiki/Central_tendency)
- Thinking: Symmetric → Mean=Median=Mode; Skewed → Mean pulled toward tail

---
**Status:** Foundation of descriptive statistics | **Complements:** Variance, IQR, Data Visualization
