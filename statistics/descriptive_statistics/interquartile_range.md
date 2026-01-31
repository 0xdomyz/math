# Interquartile Range (IQR)

## 3.1 Concept Skeleton
**Definition:** Spread of middle 50% of data; difference between 75th and 25th percentiles (Q3 - Q1)  
**Purpose:** Robust spread measure unaffected by outliers, identify outliers via 1.5×IQR rule  
**Prerequisites:** Percentiles, quartiles, sorted data

## 3.2 Comparative Framing
| Measure | IQR | Standard Deviation | Range |
|---------|-----|-------------------|-------|
| **Outlier Sensitivity** | Resistant (robust) | Sensitive | Extremely sensitive |
| **Use Case** | Skewed data, outliers | Symmetric data | Quick approximation |
| **Focus** | Middle 50% | All data weighted by distance | Extremes only |

## 3.3 Examples + Counterexamples

**Simple Example:**  
Dataset: [1, 3, 5, 7, 9, 11, 13, 15, 17]. Q1=4, Q3=14, IQR=10

**Failure Case:**  
Heavy-tailed: [1, 2, 2, 3, 3, 3, 4, 4, 5, 100]. IQR=2 ignores extreme tail structure

**Edge Case:**  
Uniform data: [5, 5, 5, 5, 5]. IQR=0, all quartiles equal (no spread)

## 3.4 Layer Breakdown
```
IQR Components:
├─ Quartiles:
│   ├─ Q1 (25th percentile): 25% of data below
│   ├─ Q2 (50th percentile): Median
│   └─ Q3 (75th percentile): 75% of data below
├─ IQR Calculation:
│   ├─ Formula: IQR = Q3 - Q1
│   └─ Interpretation: Width of box in boxplot
├─ Outlier Detection:
│   ├─ Lower Fence: Q1 - 1.5×IQR
│   ├─ Upper Fence: Q3 + 1.5×IQR
│   └─ Outliers: Values beyond fences
└─ Five-Number Summary:
    ├─ Minimum (excluding outliers)
    ├─ Q1
    ├─ Median (Q2)
    ├─ Q3
    └─ Maximum (excluding outliers)
```

## 3.5 Mini-Project
Calculate IQR and detect outliers:
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data with outliers
np.random.seed(42)
normal_data = np.random.normal(50, 10, 100)
outliers = np.array([10, 90, 95])
data = np.concatenate([normal_data, outliers])

def analyze_iqr(data):
    Q1 = np.percentile(data, 25)
    Q2 = np.median(data)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_fence) | (data > upper_fence)]
    
    print(f"Five-Number Summary:")
    print(f"  Min: {np.min(data):.2f}")
    print(f"  Q1 (25%): {Q1:.2f}")
    print(f"  Median (50%): {Q2:.2f}")
    print(f"  Q3 (75%): {Q3:.2f}")
    print(f"  Max: {np.max(data):.2f}")
    print(f"\nIQR: {IQR:.2f}")
    print(f"Lower Fence: {lower_fence:.2f}")
    print(f"Upper Fence: {upper_fence:.2f}")
    print(f"Outliers: {outliers}")
    
    # Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].boxplot(data, vert=True)
    axes[0].set_title('Boxplot with Outliers')
    axes[0].set_ylabel('Value')
    axes[0].axhline(lower_fence, color='r', linestyle='--', label='Fences')
    axes[0].axhline(upper_fence, color='r', linestyle='--')
    axes[0].legend()
    
    axes[1].hist(data, bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(Q1, color='g', linestyle='--', label='Q1')
    axes[1].axvline(Q2, color='b', linestyle='--', label='Median')
    axes[1].axvline(Q3, color='g', linestyle='--', label='Q3')
    axes[1].axvline(lower_fence, color='r', linestyle=':', label='Fences')
    axes[1].axvline(upper_fence, color='r', linestyle=':')
    axes[1].legend()
    axes[1].set_title('Histogram with Quartiles')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

analyze_iqr(data)

# Manual quartile calculation
sorted_sample = sorted([1, 3, 5, 7, 9, 11, 13, 15, 17])
print(f"\nManual: Q1={sorted_sample[2]}, Q3={sorted_sample[6]}, IQR={sorted_sample[6]-sorted_sample[2]}")
```

## 3.6 Challenge Round
When is IQR the wrong choice?
- **Symmetric normal data**: SD more efficient and standard for hypothesis tests
- **Need all data**: IQR ignores tails; use range or percentile ratios (P90/P10)
- **Very small samples**: Quartiles unstable with n < 10
- **Bimodal distributions**: IQR misses dual peaks; need density plots
- **Comparing across scales**: IQR not normalized; use relative measures

## 3.7 Key References
- [Khan Academy - IQR](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/box-whisker-plots)
- [Wikipedia - Interquartile Range](https://en.wikipedia.org/wiki/Interquartile_range)
- Thinking: 1.5×IQR rule balances Type I/II errors; Tukey's method for boxplots; Robust alternative to SD

---
**Status:** Robust spread measure | **Complements:** Boxplots, Outlier Detection, Percentiles
