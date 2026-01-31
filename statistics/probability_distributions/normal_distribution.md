# Normal Distribution

## 1. Concept Skeleton
**Definition:** Continuous probability distribution with symmetric bell curve, characterized by mean μ and standard deviation σ; P(X) = (1/(σ√2π)) × exp(-(x-μ)²/(2σ²))  
**Purpose:** Model naturally occurring phenomena; foundation for hypothesis testing and confidence intervals; Central Limit Theorem links sample means to normal  
**Prerequisites:** Continuous distributions, probability density functions, standardization (z-scores)

## 2. Comparative Framing
| Property | Normal | Poisson | Exponential | Binomial |
|----------|--------|---------|-------------|----------|
| **Type** | Continuous | Discrete | Continuous | Discrete |
| **Parameters** | μ, σ | λ | λ | n, p |
| **Support** | (-∞, +∞) | 0,1,2,... | [0, ∞) | 0,1,...,n |
| **Symmetry** | Symmetric | Right-skewed | Right-skewed | Depends on p |
| **Use** | Natural phenomena | Rare counts | Waiting times | Success/failure trials |
| **Skewness** | 0 | λ^(-0.5) | 2 | Depends on p |

## 3. Examples + Counterexamples

**Simple Example:**  
Human heights: μ=170cm, σ=8cm. Approximately 95% of heights fall within 170±16cm (μ±2σ). Use table/formula to find P(height > 185cm).

**Perfect Fit:**  
Test scores on well-designed standardized test: Distribution closely normal due to many independent factors combining (CLT).

**Poor Fit:**  
Income distribution: Right-skewed (few very high earners), not normal. Violates symmetry assumption; transforms (log-income) often more normal.

**Edge Case:**  
Standard normal Z~N(0,1): Used as reference; any normal converts via z = (x-μ)/σ. Tables/software built for standard normal only.

## 4. Layer Breakdown
```
Normal Distribution Structure:

├─ Probability Density Function (PDF):
│  ├─ f(x) = (1/(σ√2π)) × exp(-(x-μ)²/(2σ²))
│  ├─ Peak at x = μ with height 1/(σ√2π)
│  ├─ Inflection points at μ ± σ
│  └─ Tails asymptote to zero (never touch axis)
├─ Cumulative Distribution Function (CDF):
│  ├─ Φ(x) = P(X ≤ x)
│  ├─ Increases from 0 to 1
│  ├─ S-shaped sigmoid curve
│  ├─ Φ(μ) = 0.5 (median = mean)
│  └─ No closed form; requires numerical integration/tables
├─ Properties:
│  ├─ Mean = Median = Mode = μ
│  ├─ Variance = σ², SD = σ
│  ├─ 68-95-99.7 Rule:
│  │   ├─ P(μ-σ ≤ X ≤ μ+σ) ≈ 0.68
│  │   ├─ P(μ-2σ ≤ X ≤ μ+2σ) ≈ 0.95
│  │   └─ P(μ-3σ ≤ X ≤ μ+3σ) ≈ 0.997
│  ├─ Skewness = 0 (symmetric)
│  ├─ Kurtosis = 3 (moderate tail weight)
│  └─ Linear transformations: aX+b ~ N(aμ+b, a²σ²)
├─ Standardization:
│  ├─ Z = (X - μ) / σ → Z ~ N(0, 1)
│  ├─ Converts any normal to standard normal
│  ├─ Enables use of single Z-table
│  └─ Inverse: X = μ + σZ
├─ Relationship to Other Distributions:
│  ├─ If X₁, X₂ ind ~ N(μ, σ²), then X₁+X₂ ~ N(2μ, 2σ²)
│  ├─ Central Limit Theorem: Sample mean X̄ → N(μ, σ²/n)
│  ├─ If X ~ N(μ, σ²), then X² ~ χ²(df=1, noncentrality)
│  └─ Basis for t, F distributions
└─ Inference:
   ├─ Hypothesis tests (one/two-sample z-test)
   ├─ Confidence intervals: μ ± z* × σ/√n
   ├─ Prediction intervals: wider, account for individual variability
   └─ Regression errors assumed normal
```

**Interaction:** PDF height determined by σ (larger σ → flatter, wider); location determined by μ; total area under curve = 1.

## 5. Mini-Project
Explore normal distribution properties and testing normality:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, shapiro, normaltest, anderson

# Generate data from different distributions
np.random.seed(42)
n = 1000

# True normal data
normal_data = np.random.normal(loc=100, scale=15, size=n)

# Non-normal comparison: exponential
exp_data = np.random.exponential(scale=2, size=n)

# Non-normal comparison: t-distribution (heavy tails)
t_data = np.random.standard_t(df=3, size=n)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# === Visualization of Normal Distribution ===
# Plot 1: PDF and CDF
x_range = np.linspace(70, 130, 200)
pdf = norm.pdf(x_range, loc=100, scale=15)
cdf = norm.cdf(x_range, loc=100, scale=15)

ax1 = plt.subplot(3, 3, 1)
ax1.plot(x_range, pdf, 'b-', linewidth=2.5, label='PDF')
ax1.fill_between(x_range, pdf, alpha=0.3)
ax1.set_title('Normal PDF (μ=100, σ=15)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
ax2.plot(x_range, cdf, 'r-', linewidth=2.5, label='CDF')
ax2.axhline(0.5, color='k', linestyle='--', alpha=0.5)
ax2.axvline(100, color='k', linestyle='--', alpha=0.5, label='μ=100')
ax2.set_title('Normal CDF (μ=100, σ=15)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cumulative Probability')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: 68-95-99.7 Rule
x_rule = np.linspace(55, 145, 200)
y_rule = norm.pdf(x_rule, loc=100, scale=15)

ax3 = plt.subplot(3, 3, 3)
ax3.plot(x_rule, y_rule, 'b-', linewidth=2.5)

# Shade regions for 68-95-99.7
regions = [
    (85, 115, 0.68, 'green', '±1σ: 68%'),
    (70, 130, 0.95, 'orange', '±2σ: 95%'),
    (55, 145, 0.997, 'red', '±3σ: 99.7%')
]

colors_shade = ['green', 'orange', 'red']
alphas = [0.15, 0.10, 0.05]
for i, (lower, upper, prob, color, label) in enumerate(regions):
    x_fill = x_rule[(x_rule >= lower) & (x_rule <= upper)]
    y_fill = norm.pdf(x_fill, loc=100, scale=15)
    ax3.fill_between(x_fill, y_fill, alpha=alphas[i], color=color, label=label)

ax3.set_title('68-95-99.7 Rule', fontsize=11, fontweight='bold')
ax3.set_ylabel('Density')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# === Histogram and Q-Q Plots ===
# Plot 4: Histogram with theoretical normal overlay
ax4 = plt.subplot(3, 3, 4)
ax4.hist(normal_data, bins=30, density=True, alpha=0.6, color='blue', label='Data')
x_theo = np.linspace(normal_data.min(), normal_data.max(), 200)
ax4.plot(x_theo, norm.pdf(x_theo, normal_data.mean(), normal_data.std()), 
         'r-', linewidth=2.5, label='Theoretical')
ax4.set_title('Normal Data vs Theoretical', fontsize=11, fontweight='bold')
ax4.set_ylabel('Density')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Q-Q Plot for normal data
ax5 = plt.subplot(3, 3, 5)
stats.probplot(normal_data, dist="norm", plot=ax5)
ax5.set_title('Q-Q Plot: Normal Data', fontsize=11, fontweight='bold')
ax5.grid(alpha=0.3)

# Plot 6: Q-Q Plot for non-normal data (exponential)
ax6 = plt.subplot(3, 3, 6)
stats.probplot(exp_data, dist="norm", plot=ax6)
ax6.set_title('Q-Q Plot: Exponential Data', fontsize=11, fontweight='bold')
ax6.grid(alpha=0.3)

# === Normality Tests ===
# Plot 7: Comparison of data distributions
ax7 = plt.subplot(3, 3, 7)
ax7.hist(normal_data, bins=30, alpha=0.5, label='Normal', density=True, color='blue')
ax7.hist(exp_data, bins=30, alpha=0.5, label='Exponential', density=True, color='red')
ax7.hist(t_data, bins=30, alpha=0.5, label='t(df=3)', density=True, color='green')
ax7.set_title('Data Distribution Comparison', fontsize=11, fontweight='bold')
ax7.set_ylabel('Density')
ax7.legend()
ax7.grid(alpha=0.3)

# Plot 8: Standardized normal data
ax8 = plt.subplot(3, 3, 8)
normal_standardized = (normal_data - normal_data.mean()) / normal_data.std()
ax8.hist(normal_standardized, bins=30, alpha=0.6, density=True, color='blue', label='Data')
x_std = np.linspace(-4, 4, 200)
ax8.plot(x_std, norm.pdf(x_std, 0, 1), 'r-', linewidth=2.5, label='Standard Normal')
ax8.set_title('Standardized Normal (Z~N(0,1))', fontsize=11, fontweight='bold')
ax8.set_ylabel('Density')
ax8.legend()
ax8.grid(alpha=0.3)

# Plot 9: Normality test results (text table)
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

# Run normality tests
shapiro_stat, shapiro_p = shapiro(normal_data)
ks_stat, ks_p = normaltest(normal_data)
anderson_result = anderson(normal_data)

test_results = [
    ['Test', 'Statistic', 'p-value', 'Normal?'],
    ['Shapiro-Wilk', f'{shapiro_stat:.4f}', f'{shapiro_p:.4f}', 'Yes' if shapiro_p > 0.05 else 'No'],
    ['D\'Agostino-Pearson', f'{ks_stat:.4f}', f'{ks_p:.4f}', 'Yes' if ks_p > 0.05 else 'No'],
    ['Anderson-Darling', f'{anderson_result.statistic:.4f}', '(see CV)', 'Yes' if anderson_result.statistic < anderson_result.critical_values[2] else 'No'],
]

table = ax9.table(cellText=test_results, cellLoc='center', loc='center',
                 colWidths=[0.3, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color header row
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax9.set_title('Normality Test Results (α=0.05)', fontsize=11, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# === Practical Calculations ===
print("="*60)
print("NORMAL DISTRIBUTION CALCULATIONS")
print("="*60)

mu, sigma = 100, 15

# Calculate probabilities
print(f"\nGiven X ~ N(μ={mu}, σ={sigma}):")
print(f"\n1. Find P(X < 115):")
prob_less_115 = norm.cdf(115, loc=mu, scale=sigma)
z_115 = (115 - mu) / sigma
print(f"   z-score: (115 - {mu}) / {sigma} = {z_115:.2f}")
print(f"   P(X < 115) = {prob_less_115:.4f} ({prob_less_115*100:.2f}%)")

print(f"\n2. Find P(85 < X < 115):")
prob_between = norm.cdf(115, mu, sigma) - norm.cdf(85, mu, sigma)
print(f"   P(85 < X < 115) = {prob_between:.4f} ({prob_between*100:.2f}%)")
print(f"   (This is the ±1σ interval: {prob_between:.4f} ≈ 0.68)")

print(f"\n3. Find 95th percentile:")
p95 = norm.ppf(0.95, loc=mu, scale=sigma)
print(f"   x₉₅ = {p95:.2f}")
print(f"   95% of data falls below {p95:.2f}")

print(f"\n4. Confidence interval for sample mean (n=25, observed mean=102):")
se = sigma / np.sqrt(25)
ci_lower = 102 - 1.96 * se
ci_upper = 102 + 1.96 * se
print(f"   Standard error: σ/√n = {sigma}/√25 = {se:.2f}")
print(f"   95% CI: 102 ± 1.96 × {se:.2f} = [{ci_lower:.2f}, {ci_upper:.2f}]")

print("\n" + "="*60)
print("TRANSFORMATION PROPERTIES")
print("="*60)

print(f"\nIf X ~ N({mu}, {sigma}²) and Y = 2X + 10:")
print(f"  Then Y ~ N(2×{mu}+10, (2²)×{sigma}²)")
print(f"  Y ~ N({2*mu + 10}, {(2**2)*(sigma**2)})")
```

## 6. Challenge Round
1. **Birthday Problem Analog:** n=23 people normally distributed heights. Find P(two people within 1cm). Involves integration/convolution of normal distributions.

2. **Mixture of Normals:** Data from two groups: Group A ~ N(100,15), Group B ~ N(120,15). Observed data could be mixture. How to detect bimodality? (Dip test, mixture models)

3. **Extreme Value Tail:** What's the probability of observing value > μ + 5σ? Extremely rare (< 3×10⁻⁷). Relevance: financial crashes, climate extremes—"black swan" events.

4. **Non-normality Rescue:** Box-Cox transformation can normalize skewed data. Apply to exponential-like data, then analyze. Compare results before/after transformation.

5. **CLT Verification:** Sample from exponential (very non-normal). Plot histogram of 10000 sample means (each from n=30 samples). Observe convergence to normality despite exponential parent.

## 7. Key References
- [Khan Academy: Normal Distribution](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/normal-distributions-library)
- [Z-Table and CDF Reference](https://en.wikipedia.org/wiki/Standard_normal_table)
- [Shapiro-Wilk & Normality Testing](https://www.jstor.org/stable/2347986)
- [Box-Cox Transformation](https://www.jstor.org/stable/2286746)

---
**Status:** Foundation distribution | **Complements:** Central Limit Theorem, Hypothesis Testing, Confidence Intervals
