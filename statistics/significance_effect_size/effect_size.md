# Effect Size

## 1. Concept Skeleton
**Definition:** Quantitative measure of magnitude of difference or relationship, independent of sample size; expresses practical significance of findings  
**Purpose:** Complement p-values which only indicate statistical significance; communicate real-world importance; enable meta-analysis and power calculations  
**Prerequisites:** Hypothesis testing, distributions, statistical significance, type I/II errors

## 2. Comparative Framing
| Effect Size | Formula | Range | Interpretation | Context |
|------------|---------|-------|-----------------|---------|
| **Cohen's d** | (M₁ - M₂) / SD_pooled | -∞ to +∞ | d=0.2 (small), 0.5 (medium), 0.8 (large) | Two-sample t-test |
| **Pearson's r** | Cov(X,Y) / (SD_x × SD_y) | -1 to +1 | r=0.1 (small), 0.3 (medium), 0.5 (large) | Correlation |
| **R²** | Explained variance ratio | 0 to 1 | Proportion of variance explained | Regression |
| **Odds Ratio (OR)** | odds₁ / odds₀ | 0 to +∞ | OR=1 (no effect), >1 (increased odds) | Binary outcomes |
| **Cramér's V** | √(χ²/n) | 0 to 1 | V=0.1 (small), 0.3 (medium), 0.5 (large) | Categorical association |
| **η (eta)** | √(SS_between/SS_total) | 0 to 1 | Effect in ANOVA; 0-0.06 (small) | ANOVA |

## 3. Examples + Counterexamples

**Simple Example:**  
Two teaching methods: Method A M=75, Method B M=77, SD_pooled=10. Cohen's d = 0.2 (small effect). Statistically significant (p<0.05) with n=1000, but educationally trivial.

**Misleading Significance:**  
Drug trial: n=50,000. Difference 0.5 mm vs 0.4 mm blood pressure reduction, p<0.001. Effect size? d = 0.05 (negligible). Millions spent; clinical meaningfulness questionable.

**Practical Significance:**  
Safety intervention: Reduces accident rate from 10% to 8%. OR = 0.78. Small effect size but massive real-world impact if 100 million workers → prevents 2 million accidents/year.

**Effect Size Without Significance:**  
Small study: Medium effect d=0.5, but p=0.15 (underpowered). Real effect masked by noise. Effect size reveals truth; p-value tells insufficient evidence story.

## 4. Layer Breakdown
```
Effect Size Ecosystem:

├─ Conceptual Foundations:
│  ├─ Independence from sample size (invariant to n)
│  ├─ Measures genuine magnitude, not just detectability
│  ├─ Enables comparison across studies with different n
│  └─ Separates statistical significance from practical importance
├─ Common Types:
│  ├─ Standardized Differences:
│  │   ├─ Cohen's d: (mean difference) / pooled SD
│  │   ├─ Hedges' g: d adjusted for small samples
│  │   └─ Glass's Δ: Uses control group SD
│  ├─ Association Strength:
│  │   ├─ Pearson's r: Correlation coefficient
│  │   ├─ R²: Proportion variance explained
│  │   └─ Spearman's ρ: Rank-based correlation
│  ├─ Categorical/Binary:
│  │   ├─ Odds Ratio (OR)
│  │   ├─ Risk Ratio (RR)
│  │   ├─ Number Needed to Treat (NNT)
│  │   └─ Cramér's V
│  └─ Variance Components:
│      ├─ η² (eta-squared): ANOVA proportion
│      ├─ ω² (omega-squared): Unbiased estimator
│      └─ ε² (epsilon-squared): Alternative
├─ Interpretation Frameworks:
│  ├─ Cohen's Conventional Benchmarks:
│  │   ├─ Small: d=0.2, r=0.1, η²=0.01
│  │   ├─ Medium: d=0.5, r=0.3, η²=0.06
│  │   └─ Large: d=0.8, r=0.5, η²=0.14
│  ├─ Context-Dependent Interpretation:
│  │   ├─ Medical: Even small effect with health consequence = large
│  │   ├─ Psychology: Medium effects often most common
│  │   └─ Physics: Tiny effects with theory confirmation = large
│  └─ Publication Bias:
│      ├─ Selective reporting of "large" effect sizes
│      ├─ File drawer problem skews literature
│      └─ Meta-analysis pools effects across studies
└─ Applications:
   ├─ Power Analysis: Plan sample size from target effect
   ├─ Meta-Analysis: Combine effect sizes across studies
   ├─ Clinical Decision: Weigh effect size vs. side effects
   └─ Resource Allocation: Cost-benefit analysis
```

**Interaction:** Effect size + sample size + variability → statistical significance. Small effect + large n → p<0.05 but d=0.1. Large effect + small n → p≈0.1 but d=1.2.

## 5. Mini-Project
Calculate and compare effect sizes across scenarios:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency

# Scenario 1: Two-sample t-test - Cohen's d
print("="*60)
print("SCENARIO 1: Two Treatment Methods - Cohen's d")
print("="*60)

np.random.seed(42)

# Method A: Traditional teaching
method_a = np.random.normal(loc=70, scale=12, size=100)

# Method B: New teaching approach
method_b = np.random.normal(loc=74, scale=12, size=100)

# Calculate descriptive stats
m_a = method_a.mean()
m_b = method_b.mean()
sd_a = method_a.std(ddof=1)
sd_b = method_b.std(ddof=1)

# Pooled SD
n_a, n_b = len(method_a), len(method_b)
sd_pooled = np.sqrt(((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / (n_a + n_b - 2))

# Cohen's d
cohens_d = (m_b - m_a) / sd_pooled

# Hedge's g (bias-corrected for small n)
correction_factor = 1 - (3 / (4 * (n_a + n_b - 2) - 1))
hedges_g = cohens_d * correction_factor

# t-test
t_stat, p_value = ttest_ind(method_a, method_b)

# Compute 95% CI for effect size
se_d = np.sqrt((n_a + n_b) / (n_a * n_b) + cohens_d**2 / (2 * (n_a + n_b - 2)))
ci_lower = cohens_d - 1.96 * se_d
ci_upper = cohens_d + 1.96 * se_d

print(f"Method A: M={m_a:.2f}, SD={sd_a:.2f}")
print(f"Method B: M={m_b:.2f}, SD={sd_b:.2f}")
print(f"Mean difference: {m_b - m_a:.2f}")
print(f"\nCohen's d: {cohens_d:.3f}")
print(f"95% CI for d: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"Hedge's g (bias-corrected): {hedges_g:.3f}")
print(f"t({n_a + n_b - 2}) = {t_stat:.3f}, p = {p_value:.4f}")
print(f"Interpretation: {'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'} effect")

# Scenario 2: Correlation - r and R²
print("\n" + "="*60)
print("SCENARIO 2: Correlation Between Study Hours and Exam Score")
print("="*60)

study_hours = np.random.normal(loc=5, scale=2, size=50)
# Exam score correlated with study hours + noise
exam_score = 40 + 8 * study_hours + np.random.normal(0, 10, size=50)

# Pearson correlation
r = np.corrcoef(study_hours, exam_score)[0, 1]
r_squared = r**2

# Significance test
r_t_stat = r * np.sqrt(50 - 2) / np.sqrt(1 - r**2)
r_p_value = 2 * (1 - stats.t.cdf(abs(r_t_stat), df=48))

print(f"Sample size: 50")
print(f"Pearson's r: {r:.3f}")
print(f"R²: {r_squared:.3f} ({r_squared*100:.1f}% variance explained)")
print(f"t(48) = {r_t_stat:.3f}, p = {r_p_value:.4f}")
print(f"Interpretation: {'Negligible' if abs(r) < 0.1 else 'Small' if abs(r) < 0.3 else 'Medium' if abs(r) < 0.5 else 'Large'} correlation")

# Scenario 3: Categorical Association - Cramér's V
print("\n" + "="*60)
print("SCENARIO 3: Categorical Association - Cramér's V")
print("="*60)

# Contingency table: Treatment (Yes/No) vs Outcome (Recovered/Not)
contingency = np.array([[45, 5], [35, 15]])  # Rows=Treatment, Cols=Outcome
print(f"Contingency table:")
print(f"             Recovered  Not Recovered")
print(f"Treatment       {contingency[0, 0]:3d}        {contingency[0, 1]:3d}")
print(f"Control         {contingency[1, 0]:3d}        {contingency[1, 1]:3d}")

chi2, p_cat, dof, expected = chi2_contingency(contingency)

# Cramér's V
n_total = contingency.sum()
min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n_total * min_dim))

# Odds Ratio
odds_treatment = contingency[0, 0] / contingency[0, 1]
odds_control = contingency[1, 0] / contingency[1, 1]
odds_ratio = odds_treatment / odds_control

# Risk Ratio
risk_treatment = contingency[0, 0] / (contingency[0, 0] + contingency[0, 1])
risk_control = contingency[1, 0] / (contingency[1, 0] + contingency[1, 1])
risk_ratio = risk_treatment / risk_control

print(f"\nχ²({dof}) = {chi2:.3f}, p = {p_cat:.4f}")
print(f"Cramér's V: {cramers_v:.3f}")
print(f"Odds Ratio: {odds_ratio:.3f}")
print(f"Risk Ratio: {risk_ratio:.3f}")
print(f"Number Needed to Treat (NNT): {1/(risk_treatment - risk_control):.1f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution comparison
axes[0, 0].hist(method_a, bins=20, alpha=0.6, label=f'Method A (M={m_a:.1f})', color='blue')
axes[0, 0].hist(method_b, bins=20, alpha=0.6, label=f'Method B (M={m_b:.1f})', color='red')
axes[0, 0].axvline(m_a, color='blue', linestyle='--', linewidth=2)
axes[0, 0].axvline(m_b, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_title(f"Cohen's d = {cohens_d:.3f}\np = {p_value:.4f}")
axes[0, 0].set_xlabel('Exam Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Plot 2: Scatter with correlation
axes[0, 1].scatter(study_hours, exam_score, alpha=0.6, s=50)
# Add regression line
z = np.polyfit(study_hours, exam_score, 1)
p_line = np.poly1d(z)
x_line = np.linspace(study_hours.min(), study_hours.max(), 100)
axes[0, 1].plot(x_line, p_line(x_line), 'r-', linewidth=2)
axes[0, 1].set_title(f"Pearson's r = {r:.3f}, R² = {r_squared:.3f}")
axes[0, 1].set_xlabel('Study Hours')
axes[0, 1].set_ylabel('Exam Score')

# Plot 3: Effect size benchmarks
benchmarks = {'Small': 0.2, 'Medium': 0.5, 'Large': 0.8}
axes[1, 0].barh(list(benchmarks.keys()), list(benchmarks.values()), color=['green', 'orange', 'red'], alpha=0.6)
axes[1, 0].axvline(abs(cohens_d), color='blue', linestyle='--', linewidth=2, label=f'Observed d = {abs(cohens_d):.3f}')
axes[1, 0].set_title("Cohen's d Benchmarks vs Observed")
axes[1, 0].set_xlabel("Cohen's d")
axes[1, 0].legend()

# Plot 4: Effect size vs sample size relationship
sample_sizes = [20, 50, 100, 200, 500, 1000]
p_values_list = []
true_effect = 0.5

for n in sample_sizes:
    # Simulate data with true effect of 0.5
    group1 = np.random.normal(0, 1, size=n)
    group2 = np.random.normal(true_effect, 1, size=n)
    _, p = ttest_ind(group1, group2)
    p_values_list.append(p)

axes[1, 1].plot(sample_sizes, p_values_list, 'o-', linewidth=2, markersize=8, label='p-value')
axes[1, 1].axhline(0.05, color='r', linestyle='--', linewidth=2, label='α = 0.05')
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].set_title(f'P-value vs Sample Size\n(True Cohen\'s d = {true_effect})')
axes[1, 1].set_xlabel('Sample Size (log scale)')
axes[1, 1].set_ylabel('P-value (log scale)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary table
print("\n" + "="*60)
print("EFFECT SIZE INTERPRETATION SUMMARY")
print("="*60)
print(f"{'Measure':<15} {'Value':<10} {'Interpretation':<20}")
print("-" * 45)
print(f"{'Cohen\'s d':<15} {abs(cohens_d):<10.3f} {'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large':<20}")
print(f"{'Pearson\'s r':<15} {r:<10.3f} {'Negligible' if abs(r) < 0.1 else 'Small' if abs(r) < 0.3 else 'Medium' if abs(r) < 0.5 else 'Large':<20}")
print(f"{'R²':<15} {r_squared:<10.3f} {f'{r_squared*100:.1f}% variance':<20}")
print(f"{'Cramér\'s V':<15} {cramers_v:<10.3f} {'Small' if cramers_v < 0.3 else 'Medium' if cramers_v < 0.5 else 'Large':<20}")
print(f"{'Odds Ratio':<15} {odds_ratio:<10.3f} {'No effect' if odds_ratio == 1 else 'Increased odds' if odds_ratio > 1 else 'Decreased odds':<20}")
```

## 6. Challenge Round
1. **Publication Bias Detective:** Collect 10 papers on same topic. Plot effect sizes vs sample size. Do small studies report larger effects? (Likely yes—file drawer effect)

2. **Clinical vs Statistical:** Drug reduces cholesterol by 2 mg/dL (tiny effect, p<0.001, n=100,000). Would you take it? Why effect size alone insufficient—consider cost, side effects, alternatives.

3. **Meta-Analysis:** Three studies: d=0.3 (n=50), d=0.6 (n=100), d=0.8 (n=30). Compute weighted average effect. Larger samples weighted more. Quantify heterogeneity (I²).

4. **Replication Crisis:** Original study claims large effect (d=0.8), fails to replicate. Possible explanations: p-hacking, true effect smaller, measurement error, publication bias, contextual differences.

5. **Power via Effect Size:** Want 80% power to detect d=0.4. How large must n be? Use G*Power or manual calculation: n ≈ 2[(z₀.₀₅ + z₀.₂₀)/d]².

## 7. Key References
- [Cohen, Statistical Power Analysis](https://www.routledge.com/Statistical-Power-Analysis-for-the-Behavioral-Sciences/Cohen/p/book/9780805802832)
- [Effect Size Guide (APA)](https://www.apa.org/science/research-methods/effect-size)
- [Cumming & Finch, Inference by Eye](https://www.jstor.org/stable/1449639)
- [Grissom & Kim, Effect Sizes for Research](https://www.routledge.com/Effect-Sizes-for-Research-A-Broad-Practical-Approach/Grissom-Kim/p/book/9780805861761)

---
**Status:** Core significance concept | **Complements:** Hypothesis Testing, Statistical Power, Meta-Analysis
