# Effect Size

## 4.1 Concept Skeleton
**Definition:** Quantitative measure of magnitude of phenomenon; standardized mean difference or strength of relationship  
**Purpose:** Assess practical significance independent of sample size; enable meta-analysis comparisons  
**Prerequisites:** Mean, standard deviation, correlation, hypothesis testing, statistical vs practical significance

## 4.2 Comparative Framing
| Measure | Cohen's d | Pearson's r | Odds Ratio | η² (Eta-squared) |
|---------|-----------|------------|-----------|-----------------|
| **Use Case** | Mean difference | Linear association | Binary outcome risk | Variance explained |
| **Range** | -∞ to +∞ | -1 to +1 | 0 to +∞ | 0 to 1 |
| **Interpretation** | 0.2/0.5/0.8 small/medium/large | 0.1/0.3/0.5 small/medium/large | 1=no effect, >1 increases odds | Proportion of variance |
| **Sample Size Effect** | None (standardized) | None | None | None |

## 4.3 Examples + Counterexamples

**Simple Example:**  
Treatment group mean=110, control=100, SD=15. Cohen's d = (110-100)/15 = 0.67 (medium effect)

**Failure Case:**  
Reporting only "p<0.001" with n=10,000. Tiny effect (d=0.05) statistically significant but meaningless

**Edge Case:**  
Cohen's d=2.0 with high overlap in distributions. Extreme effect but not perfect separation; individuals still vary

## 4.4 Layer Breakdown
```
Effect Size Taxonomy:
├─ Mean Difference Family:
│   ├─ Cohen's d: (M₁ - M₂) / SD_pooled
│   │   ├─ Small: d ≈ 0.2
│   │   ├─ Medium: d ≈ 0.5
│   │   └─ Large: d ≈ 0.8
│   ├─ Hedges' g: Bias-corrected d for small samples
│   │   └─ g = d × (1 - 3/(4N-9))
│   └─ Glass's Δ: Uses control SD only
│       └─ Δ = (M₁ - M₂) / SD_control
├─ Correlation Family:
│   ├─ Pearson's r: Linear association strength
│   │   ├─ Small: r ≈ 0.10
│   │   ├─ Medium: r ≈ 0.30
│   │   └─ Large: r ≈ 0.50
│   ├─ r² (R-squared): Proportion of variance explained
│   └─ Point-biserial: Correlation with binary variable
├─ Variance Explained:
│   ├─ η² (Eta-squared): SSbetween / SStotal
│   ├─ ω² (Omega-squared): Unbiased η²
│   └─ R²: Regression variance explained
├─ Risk/Odds Family:
│   ├─ Odds Ratio: Odds(outcome|exposed) / Odds(outcome|unexposed)
│   ├─ Risk Ratio: P(outcome|exposed) / P(outcome|unexposed)
│   └─ Number Needed to Treat (NNT): 1 / |risk₁ - risk₂|
└─ Properties:
    ├─ Independent of sample size (unlike p-value)
    ├─ Standardized (comparable across studies)
    ├─ Required for power analysis
    └─ Essential for meta-analysis
```

## 4.5 Mini-Project
Calculate and interpret multiple effect size measures:
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate two groups with known effect size
true_effect = 0.6  # Cohen's d
n1, n2 = 50, 50
group1 = np.random.normal(100, 15, n1)
group2 = np.random.normal(100 + true_effect * 15, 15, n2)

# Calculate effect sizes
print("=== Effect Size Calculations ===")

# 1. Cohen's d
mean1, mean2 = np.mean(group1), np.mean(group2)
sd1, sd2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
cohens_d = (mean2 - mean1) / pooled_sd

print(f"Cohen's d: {cohens_d:.3f}")
print(f"Interpretation: {'Small' if abs(cohens_d)<0.5 else 'Medium' if abs(cohens_d)<0.8 else 'Large'}")

# 2. Hedges' g (bias-corrected)
correction_factor = 1 - (3 / (4*(n1+n2) - 9))
hedges_g = cohens_d * correction_factor
print(f"Hedges' g: {hedges_g:.3f}")

# 3. Correlation coefficient (point-biserial)
combined_data = np.concatenate([group1, group2])
group_labels = np.concatenate([np.zeros(n1), np.ones(n2)])
r_pb, _ = stats.pearsonr(group_labels, combined_data)
print(f"Point-biserial r: {r_pb:.3f}")
print(f"r²: {r_pb**2:.3f} (variance explained)")

# 4. Statistical test for comparison
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"\nt-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Note: p-value tells significance, d tells magnitude")

# Demonstration: Effect size vs sample size
print("\n=== Effect Size Independence from Sample Size ===")
true_d = 0.5
sample_sizes = [10, 30, 50, 100, 200, 500]
results = []

for n in sample_sizes:
    d_values = []
    p_values = []
    
    for _ in range(1000):
        g1 = np.random.normal(100, 15, n)
        g2 = np.random.normal(100 + true_d * 15, 15, n)
        
        # Cohen's d
        sd_pooled = np.sqrt(((n-1)*np.std(g1, ddof=1)**2 + 
                             (n-1)*np.std(g2, ddof=1)**2) / (2*n-2))
        d = (np.mean(g2) - np.mean(g1)) / sd_pooled
        d_values.append(d)
        
        # p-value
        _, p = stats.ttest_ind(g1, g2)
        p_values.append(p)
    
    results.append({
        'n': n,
        'd_mean': np.mean(d_values),
        'd_std': np.std(d_values),
        'p_median': np.median(p_values),
        'power': np.mean(np.array(p_values) < 0.05)
    })

plt.figure(figsize=(14, 10))

# Plot 1: Effect size vs sample size
plt.subplot(2, 3, 1)
ns = [r['n'] for r in results]
ds = [r['d_mean'] for r in results]
d_errs = [r['d_std'] for r in results]
plt.errorbar(ns, ds, yerr=d_errs, fmt='o-', capsize=5, linewidth=2)
plt.axhline(true_d, color='r', linestyle='--', label=f'True d={true_d}')
plt.xlabel('Sample Size per Group')
plt.ylabel("Cohen's d")
plt.title("Effect Size Independent of n\n(Unbiased estimator)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: p-value vs sample size
plt.subplot(2, 3, 2)
ps = [r['p_median'] for r in results]
plt.plot(ns, ps, 'o-', linewidth=2)
plt.axhline(0.05, color='r', linestyle='--', label='α=0.05')
plt.xlabel('Sample Size per Group')
plt.ylabel('Median P-value')
plt.title('P-value Decreases with n\n(Same true effect)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Distribution overlap visualization
plt.subplot(2, 3, 3)
x = np.linspace(60, 140, 200)
for d in [0.2, 0.5, 0.8, 1.5]:
    pdf1 = stats.norm.pdf(x, 100, 15)
    pdf2 = stats.norm.pdf(x, 100 + d*15, 15)
    plt.plot(x, pdf1, 'b-', alpha=0.3)
    plt.plot(x, pdf2, 'r-', alpha=0.3)
    plt.fill_between(x, 0, np.minimum(pdf1, pdf2), alpha=0.2, 
                     label=f'd={d:.1f}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Distribution Overlap by Effect Size')
plt.legend()

# Plot 4: Effect size interpretation scale
plt.subplot(2, 3, 4)
d_values = np.array([0.2, 0.5, 0.8, 1.2, 2.0])
labels = ['Small\nd=0.2', 'Medium\nd=0.5', 'Large\nd=0.8', 'Very Large\nd=1.2', 'Huge\nd=2.0']
# Calculate overlap coefficient (proportion of overlap)
overlaps = []
for d in d_values:
    # Overlap = 2 * Φ(-|d|/2) where Φ is standard normal CDF
    overlap = 2 * stats.norm.cdf(-abs(d)/2)
    overlaps.append(overlap * 100)

colors = ['lightgreen', 'yellow', 'orange', 'red', 'darkred']
plt.barh(range(len(d_values)), overlaps, color=colors, edgecolor='black')
plt.yticks(range(len(d_values)), labels)
plt.xlabel('Distribution Overlap (%)')
plt.title('Effect Size Interpretation Scale')
for i, overlap in enumerate(overlaps):
    plt.text(overlap, i, f'  {overlap:.1f}%', va='center')
plt.grid(True, alpha=0.3, axis='x')

# Plot 5: Power analysis with effect size
plt.subplot(2, 3, 5)
effect_sizes = np.linspace(0.1, 1.5, 30)
alpha = 0.05
n = 50
power_values = []

for d in effect_sizes:
    # Simplified power calculation
    noncentrality = d * np.sqrt(n/2)
    critical_z = stats.norm.ppf(1 - alpha/2)
    power = 1 - stats.norm.cdf(critical_z - noncentrality)
    power_values.append(power)

plt.plot(effect_sizes, power_values, linewidth=2)
plt.axhline(0.80, color='r', linestyle='--', label='Target power=0.80')
plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
plt.xlabel('Effect Size (Cohen\'s d)')
plt.ylabel('Statistical Power')
plt.title(f'Power Analysis (n={n}, α={alpha})')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Meta-analysis forest plot simulation
plt.subplot(2, 3, 6)
n_studies = 8
true_effect = 0.5
study_effects = []
study_ses = []

for i in range(n_studies):
    n_study = np.random.randint(30, 150)
    g1 = np.random.normal(100, 15, n_study)
    g2 = np.random.normal(100 + true_effect*15, 15, n_study)
    
    sd_pooled = np.sqrt(((n_study-1)*np.std(g1, ddof=1)**2 + 
                         (n_study-1)*np.std(g2, ddof=1)**2) / (2*n_study-2))
    d = (np.mean(g2) - np.mean(g1)) / sd_pooled
    se = np.sqrt(2/n_study)  # Approximate SE of Cohen's d
    
    study_effects.append(d)
    study_ses.append(se)

study_effects = np.array(study_effects)
study_ses = np.array(study_ses)

# Meta-analytic mean (inverse variance weighted)
weights = 1 / study_ses**2
meta_effect = np.sum(weights * study_effects) / np.sum(weights)
meta_se = np.sqrt(1 / np.sum(weights))

# Forest plot
y_pos = np.arange(n_studies)
plt.errorbar(study_effects, y_pos, xerr=1.96*study_ses, fmt='o', 
             capsize=5, label='Individual studies')
plt.plot([meta_effect-1.96*meta_se, meta_effect+1.96*meta_se], 
         [-1, -1], 'r-', linewidth=3)
plt.plot(meta_effect, -1, 'ro', markersize=12, label=f'Meta d={meta_effect:.2f}')
plt.axvline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(true_effect, color='green', linestyle='--', alpha=0.5, 
            label=f'True d={true_effect}')
plt.yticks(list(y_pos) + [-1], [f'Study {i+1}' for i in range(n_studies)] + ['Meta'])
plt.xlabel('Effect Size (Cohen\'s d)')
plt.title('Meta-Analysis Forest Plot')
plt.legend()
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Practical significance vs statistical significance
print("\n=== Practical vs Statistical Significance ===")
scenarios = [
    {'n': 10, 'd': 0.8, 'desc': 'Small sample, large effect'},
    {'n': 1000, 'd': 0.1, 'desc': 'Large sample, small effect'},
    {'n': 50, 'd': 0.5, 'desc': 'Moderate sample, moderate effect'}
]

for scenario in scenarios:
    n = scenario['n']
    d = scenario['d']
    
    # Simulate
    g1 = np.random.normal(100, 15, n)
    g2 = np.random.normal(100 + d*15, 15, n)
    _, p = stats.ttest_ind(g1, g2)
    
    stat_sig = "YES" if p < 0.05 else "NO"
    practical_sig = "YES" if abs(d) >= 0.5 else "NO"
    
    print(f"\n{scenario['desc']}:")
    print(f"  n={n}, d={d:.1f}, p={p:.4f}")
    print(f"  Statistically significant: {stat_sig}")
    print(f"  Practically significant: {practical_sig}")
```

## 4.6 Challenge Round
When is effect size misleading?
- **Restricted range**: Correlation attenuated if variable truncated; d underestimated with ceiling/floor effects
- **Heterogeneous populations**: Single d masks subgroup variation; consider interaction effects
- **Different SDs across groups**: Glass's Δ or control group SD more appropriate than pooled SD
- **Ordinal/categorical outcomes**: Cohen's d inappropriate; use odds ratios, risk ratios, or Cramér's V
- **Non-linear relationships**: r and d assume linearity; use η² or other measures for curves

## 4.7 Key References
- [Effect Size Primer](https://www.statisticshowto.com/probability-and-statistics/effect-size/) - Comprehensive overview of all major measures
- [Cohen's Conventions Critique](https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full) - Context-dependent interpretation, not universal cutoffs
- [Reporting Effect Sizes (APA)](https://apastyle.apa.org/learn/faqs/report-effect-size) - Publication standards for transparency

---
**Status:** Essential companion to p-values | **Complements:** Statistical vs Practical Significance, Power Analysis, Meta-Analysis
