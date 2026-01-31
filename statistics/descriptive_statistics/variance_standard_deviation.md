# Variance & Standard Deviation

## 2.1 Concept Skeleton
**Definition:** Measures of spread/dispersion quantifying variation around mean  
**Purpose:** Assess consistency, compare variability, statistical inference foundation  
**Prerequisites:** Mean, squared deviations, square roots

## 2.2 Comparative Framing
| Measure | Variance (σ²) | Standard Deviation (σ) | Range |
|---------|--------------|----------------------|-------|
| **Units** | Squared original units | Original units | Original units |
| **Interpretation** | Abstract spread | Typical distance from mean | Max - Min |
| **Use Case** | Mathematical derivations | Reporting, comparisons | Quick rough estimate |

## 2.3 Examples + Counterexamples

**Simple Example:**  
Dataset: [2, 4, 6, 8, 10]. Mean = 6. Variance = 10. SD = 3.16

**Failure Case:**  
Outlier: [5, 5, 5, 5, 100]. SD = 42.4 dominated by single extreme value

**Edge Case:**  
Constant data: [7, 7, 7, 7]. Variance = 0, SD = 0 (no spread)

## 2.4 Layer Breakdown
```
Variance Components:
├─ Population Variance (σ²):
│   ├─ Formula: Σ(xᵢ - μ)² / N
│   └─ Use: True parameter of full population
├─ Sample Variance (s²):
│   ├─ Formula: Σ(xᵢ - x̄)² / (n-1)
│   ├─ Bessel's Correction: n-1 for unbiased estimator
│   └─ Use: Estimate from sample data
├─ Standard Deviation:
│   ├─ Formula: √Variance
│   ├─ Properties: Same units as data, additive under independence
│   └─ Interpretation: ~68% within ±1σ for normal distribution
└─ Calculation Steps:
    1. Compute mean
    2. Calculate deviations (xᵢ - mean)
    3. Square deviations
    4. Average squared deviations
    5. Take square root (for SD)
```

## 2.5 Mini-Project
Compute and visualize spread:
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate datasets with different spreads
np.random.seed(42)
low_spread = np.random.normal(50, 5, 1000)
high_spread = np.random.normal(50, 20, 1000)

def analyze_spread(data, name):
    mean = np.mean(data)
    var_pop = np.var(data, ddof=0)  # Population variance
    var_sample = np.var(data, ddof=1)  # Sample variance
    std = np.std(data, ddof=1)
    
    print(f"\n{name}:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Population Variance: {var_pop:.2f}")
    print(f"  Sample Variance: {var_sample:.2f}")
    print(f"  Standard Deviation: {std:.2f}")
    print(f"  Range [mean±1σ]: [{mean-std:.2f}, {mean+std:.2f}]")
    
    # Visual
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(mean - std, color='g', linestyle=':', label=f'±1 SD')
    plt.axvline(mean + std, color='g', linestyle=':')
    plt.legend()
    plt.title(f'{name} (SD = {std:.2f})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

analyze_spread(low_spread, "Low Spread (σ=5)")
analyze_spread(high_spread, "High Spread (σ=20)")

# Manual calculation verification
sample = [2, 4, 6, 8, 10]
mean_manual = sum(sample) / len(sample)
var_manual = sum((x - mean_manual)**2 for x in sample) / (len(sample) - 1)
std_manual = var_manual ** 0.5
print(f"\nManual: Mean={mean_manual}, Var={var_manual:.2f}, SD={std_manual:.2f}")
```

## 2.6 Challenge Round
When is variance/SD the wrong tool?
- **Skewed distributions**: Use IQR or MAD (median absolute deviation) instead
- **Outlier-prone data**: Use robust measures (IQR, trimmed SD)
- **Different units**: Cannot compare raw SD across different scales (use CV)
- **Non-normal distributions**: SD alone doesn't describe shape fully
- **Comparing groups**: Need coefficient of variation (CV = SD/Mean) for different means

## 2.7 Key References
- [Khan Academy - Variance & SD](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/variance-standard-deviation-population)
- [Wikipedia - Standard Deviation](https://en.wikipedia.org/wiki/Standard_deviation)
- Thinking: Bessel's correction (n-1) adjusts for using sample mean; Population uses N; SD preferred for reporting due to units

---
**Status:** Core spread measure | **Complements:** Mean, IQR, Z-scores, Confidence Intervals
