# Empirical Rule

## 1.1 Concept Skeleton
**Definition:** For normal distribution: ~68% within ±1σ, ~95% within ±2σ, ~99.7% within ±3σ from mean  
**Purpose:** Quick probability estimation, outlier detection, intuitive understanding of spread  
**Prerequisites:** Normal distribution, standard deviation, z-scores

## 1.2 Comparative Framing
| Rule | Empirical (68-95-99.7) | Chebyshev's Inequality | Exact Normal |
|------|----------------------|---------------------|--------------|
| **Applies To** | Normal distributions only | Any distribution | Normal only |
| **±1σ** | ~68% | ≥0% (not useful) | 68.27% |
| **±2σ** | ~95% | ≥75% | 95.45% |
| **±3σ** | ~99.7% | ≥88.9% | 99.73% |

## 1.3 Examples + Counterexamples

**Simple Example:**  
IQ scores: μ=100, σ=15. Empirical rule predicts ~68% between 85-115, ~95% between 70-130

**Failure Case:**  
Skewed income data: Empirical rule fails; only applies to symmetric, normal-like distributions

**Edge Case:**  
Small samples (n<30): Rule approximation less accurate; use t-distribution instead

## 1.4 Layer Breakdown
```
Empirical Rule Components:
├─ Three Key Intervals:
│   ├─ μ ± 1σ:
│   │   ├─ Contains: ~68.27% of data
│   │   ├─ Z-scores: -1 to +1
│   │   └─ Use: Common range, "typical" values
│   ├─ μ ± 2σ:
│   │   ├─ Contains: ~95.45% of data
│   │   ├─ Z-scores: -2 to +2
│   │   └─ Use: Outlier detection threshold
│   └─ μ ± 3σ:
│       ├─ Contains: ~99.73% of data
│       ├─ Z-scores: -3 to +3
│       └─ Use: Extreme outliers, control limits
├─ Exact Calculations:
│   ├─ P(μ - σ ≤ X ≤ μ + σ) = 0.6827
│   ├─ P(μ - 2σ ≤ X ≤ μ + 2σ) = 0.9545
│   ├─ P(μ - 3σ ≤ X ≤ μ + 3σ) = 0.9973
│   └─ Tails: P(|X - μ| > 3σ) = 0.0027 (~1 in 370)
├─ Assumptions:
│   ├─ Normal distribution (symmetric bell curve)
│   ├─ Known μ and σ (or good estimates)
│   └─ Large sample (n≥30 for approximation)
├─ Applications:
│   ├─ Quality Control:
│   │   ├─ Control charts use ±3σ limits
│   │   └─ Six Sigma: 3.4 defects per million
│   ├─ Outlier Detection:
│   │   ├─ Values beyond ±2σ or ±3σ flagged
│   │   └─ Quick screening without formal tests
│   ├─ Confidence Intervals:
│   │   ├─ ~95% CI: x̄ ± 2·SE (approximate)
│   │   └─ Exact: x̄ ± 1.96·SE
│   └─ Data Description:
│       ├─ Communicate spread intuitively
│       └─ Compare variability across groups
└─ Mnemonics:
    ├─ "68-95-99.7" rule
    ├─ "1-2-3 sigma rule"
    └─ Visual: Bell curve with shaded regions
```

## 1.5 Mini-Project
Verify and apply the empirical rule:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Generate normal data
mu, sigma = 100, 15
n = 10000
data = np.random.normal(mu, sigma, n)

# Calculate actual percentages
within_1sigma = np.sum((data >= mu - sigma) & (data <= mu + sigma)) / n * 100
within_2sigma = np.sum((data >= mu - 2*sigma) & (data <= mu + 2*sigma)) / n * 100
within_3sigma = np.sum((data >= mu - 3*sigma) & (data <= mu + 3*sigma)) / n * 100

print("=== Empirical Rule Verification ===")
print(f"Distribution: Normal(μ={mu}, σ={sigma}), n={n}")
print(f"\nWithin μ ± 1σ [{mu-sigma}, {mu+sigma}]:")
print(f"  Empirical rule: ~68%")
print(f"  Actual: {within_1sigma:.2f}%")
print(f"  Exact (theory): {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f} = 68.27%")

print(f"\nWithin μ ± 2σ [{mu-2*sigma}, {mu+2*sigma}]:")
print(f"  Empirical rule: ~95%")
print(f"  Actual: {within_2sigma:.2f}%")
print(f"  Exact (theory): {stats.norm.cdf(2) - stats.norm.cdf(-2):.4f} = 95.45%")

print(f"\nWithin μ ± 3σ [{mu-3*sigma}, {mu+3*sigma}]:")
print(f"  Empirical rule: ~99.7%")
print(f"  Actual: {within_3sigma:.2f}%")
print(f"  Exact (theory): {stats.norm.cdf(3) - stats.norm.cdf(-3):.4f} = 99.73%")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram with empirical rule intervals
axes[0, 0].hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
axes[0, 0].plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', 
                linewidth=2, label='Normal PDF')

# Shade regions
colors = ['lightblue', 'lightgreen', 'lightyellow']
labels = ['68%', '95%', '99.7%']
for i, (mult, color, label) in enumerate(zip([1, 2, 3], colors, labels)):
    x_fill = x_range[(x_range >= mu - mult*sigma) & (x_range <= mu + mult*sigma)]
    axes[0, 0].fill_between(x_fill, 0, stats.norm.pdf(x_fill, mu, sigma), 
                            alpha=0.3, color=color, label=f'±{mult}σ: ~{label}')

axes[0, 0].axvline(mu, color='black', linestyle='--', linewidth=2, label='μ')
for i in [1, 2, 3]:
    axes[0, 0].axvline(mu + i*sigma, color='gray', linestyle=':', alpha=0.7)
    axes[0, 0].axvline(mu - i*sigma, color='gray', linestyle=':', alpha=0.7)

axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Empirical Rule: 68-95-99.7')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Z-score interpretation
z_scores = (data - mu) / sigma
axes[0, 1].hist(z_scores, bins=50, density=True, alpha=0.7, edgecolor='black')
z_range = np.linspace(-4, 4, 1000)
axes[0, 1].plot(z_range, stats.norm.pdf(z_range, 0, 1), 'r-', linewidth=2)

for z in [1, 2, 3]:
    axes[0, 1].axvline(z, color='g', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(-z, color='g', linestyle='--', alpha=0.5)
    axes[0, 1].text(z, 0.4 - z*0.05, f'{z}σ', fontsize=10, ha='center')

axes[0, 1].set_xlabel('Z-score')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Standardized (Z-scores)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Comparison with non-normal (skewed) data
skewed_data = np.random.exponential(scale=20, size=n)
mu_skewed = np.mean(skewed_data)
sigma_skewed = np.std(skewed_data)

within_1sigma_skewed = np.sum((skewed_data >= mu_skewed - sigma_skewed) & 
                              (skewed_data <= mu_skewed + sigma_skewed)) / n * 100
within_2sigma_skewed = np.sum((skewed_data >= mu_skewed - 2*sigma_skewed) & 
                              (skewed_data <= mu_skewed + 2*sigma_skewed)) / n * 100

axes[1, 0].hist(skewed_data, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(mu_skewed, color='r', linestyle='--', linewidth=2, label='Mean')
axes[1, 0].axvline(mu_skewed + sigma_skewed, color='g', linestyle=':', alpha=0.7)
axes[1, 0].axvline(mu_skewed - sigma_skewed, color='g', linestyle=':', alpha=0.7)
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_title(f'Skewed Data: ±1σ contains {within_1sigma_skewed:.1f}% (not 68%!)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

print(f"\n=== Non-Normal Distribution ===")
print(f"Exponential distribution (right-skewed):")
print(f"  Within ±1σ: {within_1sigma_skewed:.1f}% (vs 68% for normal)")
print(f"  Within ±2σ: {within_2sigma_skewed:.1f}% (vs 95% for normal)")
print(f"  Empirical rule does NOT apply!")

# 4. Practical application: Quality control
process_mean = 500
process_std = 10
measurements = np.random.normal(process_mean, process_std, 200)

# Control limits (±3σ)
ucl = process_mean + 3 * process_std  # Upper Control Limit
lcl = process_mean - 3 * process_std  # Lower Control Limit

axes[1, 1].plot(measurements, 'bo-', alpha=0.6, markersize=4)
axes[1, 1].axhline(process_mean, color='g', linewidth=2, label='Target Mean')
axes[1, 1].axhline(ucl, color='r', linestyle='--', linewidth=2, label='UCL (μ+3σ)')
axes[1, 1].axhline(lcl, color='r', linestyle='--', linewidth=2, label='LCL (μ-3σ)')

# Shade warning zones (±2σ to ±3σ)
axes[1, 1].axhspan(process_mean + 2*process_std, ucl, alpha=0.2, color='yellow')
axes[1, 1].axhspan(lcl, process_mean - 2*process_std, alpha=0.2, color='yellow')

# Identify out-of-control points
outliers = (measurements > ucl) | (measurements < lcl)
if np.any(outliers):
    axes[1, 1].plot(np.where(outliers)[0], measurements[outliers], 
                    'r*', markersize=15, label='Out of Control')

axes[1, 1].set_xlabel('Sample Number')
axes[1, 1].set_ylabel('Measurement')
axes[1, 1].set_title('Quality Control Chart (±3σ Limits)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Real-world examples
print("\n=== Real-World Applications ===")

# Example 1: IQ scores
iq_mean, iq_std = 100, 15
print(f"\n1. IQ Scores (μ={iq_mean}, σ={iq_std}):")
print(f"   68% between {iq_mean-iq_std} and {iq_mean+iq_std}")
print(f"   95% between {iq_mean-2*iq_std} and {iq_mean+2*iq_std}")
print(f"   99.7% between {iq_mean-3*iq_std} and {iq_mean-3*iq_std}")
print(f"   IQ > 130 (2σ above): {(1 - stats.norm.cdf(2))*100:.2f}% of population")

# Example 2: SAT scores
sat_mean, sat_std = 1050, 200
print(f"\n2. SAT Scores (μ={sat_mean}, σ={sat_std}):")
print(f"   68% between {sat_mean-sat_std} and {sat_mean+sat_std}")
print(f"   95% between {sat_mean-2*sat_std} and {sat_mean+2*sat_std}")
print(f"   Score > 1450 (2σ above): {(1 - stats.norm.cdf(2))*100:.2f}% of test-takers")

# Example 3: Heights
height_mean, height_std = 170, 10  # cm
print(f"\n3. Heights (μ={height_mean}cm, σ={height_std}cm):")
print(f"   68% between {height_mean-height_std}cm and {height_mean+height_std}cm")
print(f"   Height > 190cm (2σ above): {(1 - stats.norm.cdf(2))*100:.2f}% of people")

# Outlier detection example
print("\n=== Outlier Detection Using Empirical Rule ===")
test_data = np.random.normal(50, 5, 100)
test_data = np.append(test_data, [75, 80])  # Add outliers

mean_test = np.mean(test_data)
std_test = np.std(test_data)
outliers_2sigma = test_data[(test_data > mean_test + 2*std_test) | 
                            (test_data < mean_test - 2*std_test)]
outliers_3sigma = test_data[(test_data > mean_test + 3*std_test) | 
                            (test_data < mean_test - 3*std_test)]

print(f"Data: mean={mean_test:.2f}, std={std_test:.2f}")
print(f"Outliers beyond ±2σ [{mean_test-2*std_test:.1f}, {mean_test+2*std_test:.1f}]: {outliers_2sigma}")
print(f"Outliers beyond ±3σ [{mean_test-3*std_test:.1f}, {mean_test+3*std_test:.1f}]: {outliers_3sigma}")
```

## 1.6 Challenge Round
When is the empirical rule the wrong choice?
- **Non-normal distributions**: Rule only accurate for symmetric, normal-like data
- **Small samples**: Need t-distribution; empirical rule assumes known σ
- **Skewed data**: Use percentiles/IQR instead of ±σ intervals
- **Heavy-tailed distributions**: More extreme values than normal; use robust methods
- **Precise inference**: Use exact calculations (z-scores, CDF) instead of approximations

## 1.7 Key References
- [Khan Academy - Empirical Rule](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/normal-distributions-library/v/empirical-rule)
- [Wikipedia - 68-95-99.7 Rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)
- [Normal Distribution Properties](https://en.wikipedia.org/wiki/Normal_distribution)
- Thinking: Quick mental approximation for normal data; ~95% within 2σ is most commonly used; Outlier detection uses 2σ or 3σ threshold; Only applies to normal/symmetric distributions

---
**Status:** Practical approximation for normal data | **Complements:** Normal Distribution, Z-scores, Outlier Detection, Quality Control
