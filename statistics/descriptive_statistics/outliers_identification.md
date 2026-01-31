# Outliers & Identification

## 4.1 Concept Skeleton
**Definition:** Data points significantly deviating from other observations; potential errors or extreme variability  
**Purpose:** Detect anomalies, assess data quality, prevent distorted analysis  
**Prerequisites:** IQR, standard deviation, distribution understanding

## 4.2 Comparative Framing
| Method | 1.5×IQR Rule | Z-score (±3σ) | Modified Z-score |
|--------|-------------|--------------|-----------------|
| **Basis** | Quartiles (robust) | Mean & SD (parametric) | Median & MAD (robust) |
| **Threshold** | Q1-1.5×IQR, Q3+1.5×IQR | \|z\| > 3 | \|Modified z\| > 3.5 |
| **Use Case** | Skewed data, small samples | Normal distributions | Non-normal, outlier-resistant |

## 4.3 Examples + Counterexamples

**Simple Example:**  
Dataset: [10, 12, 13, 14, 15, 50]. Value 50 exceeds Q3 + 1.5×IQR → outlier

**Failure Case:**  
Natural variation: CEO salary in company salary data. High but legitimate; removal distorts reality

**Edge Case:**  
Masked outliers: [1, 2, 3, 4, 5, 100, 105]. First outlier inflates IQR, masking second outlier

## 4.4 Layer Breakdown
```
Outlier Detection Framework:
├─ Statistical Methods:
│   ├─ IQR Method:
│   │   ├─ Lower: < Q1 - 1.5×IQR
│   │   └─ Upper: > Q3 + 1.5×IQR
│   ├─ Z-score Method:
│   │   ├─ z = (x - μ) / σ
│   │   └─ Outlier if |z| > 3 (or 2.5)
│   └─ Modified Z-score:
│       ├─ Mᵢ = 0.6745(xᵢ - x̃) / MAD
│       └─ Outlier if |Mᵢ| > 3.5
├─ Visual Methods:
│   ├─ Boxplot: Points beyond whiskers
│   ├─ Scatter plot: Isolated points
│   └─ Histogram: Extreme bins
├─ Domain Knowledge:
│   ├─ Physical limits (e.g., negative age)
│   ├─ Measurement errors
│   └─ Contextual plausibility
└─ Action:
    ├─ Investigate: Verify data entry/collection
    ├─ Keep: Valid extreme values
    ├─ Transform: Log/sqrt for skewed data
    └─ Remove: Only if justifiable error
```

## 4.5 Mini-Project
Detect outliers using multiple methods:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
# Normal data + outliers
normal = np.random.normal(50, 10, 100)
outliers = np.array([10, 15, 90, 95, 100])
data = np.concatenate([normal, outliers])

def detect_outliers(data):
    # IQR method
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_iqr = Q1 - 1.5 * IQR
    upper_iqr = Q3 + 1.5 * IQR
    outliers_iqr = data[(data < lower_iqr) | (data > upper_iqr)]
    
    # Z-score method
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    z_scores = np.abs((data - mean) / std)
    outliers_z = data[z_scores > 3]
    
    # Modified Z-score (MAD)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    outliers_mod_z = data[np.abs(modified_z) > 3.5]
    
    print("Outlier Detection Results:")
    print(f"  IQR Method: {len(outliers_iqr)} outliers")
    print(f"    Values: {sorted(outliers_iqr)}")
    print(f"  Z-score Method: {len(outliers_z)} outliers")
    print(f"    Values: {sorted(outliers_z)}")
    print(f"  Modified Z-score: {len(outliers_mod_z)} outliers")
    print(f"    Values: {sorted(outliers_mod_z)}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Boxplot
    axes[0, 0].boxplot(data, vert=True)
    axes[0, 0].set_title('Boxplot (IQR Method)')
    axes[0, 0].set_ylabel('Value')
    
    # Histogram with thresholds
    axes[0, 1].hist(data, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(lower_iqr, color='r', linestyle='--', label='IQR Fences')
    axes[0, 1].axvline(upper_iqr, color='r', linestyle='--')
    axes[0, 1].legend()
    axes[0, 1].set_title('Histogram with IQR Fences')
    axes[0, 1].set_xlabel('Value')
    
    # Z-scores
    axes[1, 0].scatter(range(len(data)), z_scores, alpha=0.6)
    axes[1, 0].axhline(3, color='r', linestyle='--', label='±3σ threshold')
    axes[1, 0].legend()
    axes[1, 0].set_title('Z-scores')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('|Z-score|')
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    
    plt.tight_layout()
    plt.show()

detect_outliers(data)
```

## 4.6 Challenge Round
When is outlier removal wrong?
- **Heavy-tailed distributions**: Extremes are expected (use robust methods, not removal)
- **Small samples**: Legitimate variation can appear extreme
- **Scientific discovery**: Outliers may be new phenomena (e.g., exoplanets)
- **Fraud detection**: Outliers are the signal, not noise
- **No investigation**: Automatic removal without checking data quality is dangerous

## 4.7 Key References
- [Khan Academy - Outliers](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/box-whisker-plots)
- [Wikipedia - Outlier](https://en.wikipedia.org/wiki/Outlier)
- Thinking: Always investigate before removing; IQR method most common; Context matters more than statistics alone

---
**Status:** Critical data quality check | **Complements:** IQR, Boxplots, Robust Statistics
