# Significance Level (α)

## 3.1 Concept Skeleton
**Definition:** Pre-specified threshold probability for rejecting null hypothesis; maximum acceptable Type I error rate  
**Purpose:** Control false positive rate; balance between detecting true effects and avoiding spurious findings  
**Prerequisites:** Hypothesis testing, Type I/II errors, p-values, power analysis

## 3.2 Comparative Framing
| Threshold | α = 0.10 | α = 0.05 | α = 0.01 | α = 0.001 |
|-----------|---------|---------|---------|-----------|
| **Stringency** | Lenient | Conventional | Stringent | Very stringent |
| **Type I Error** | 10% false positives | 5% false positives | 1% false positives | 0.1% false positives |
| **Power (typically)** | Higher | Moderate | Lower | Much lower |
| **Use Case** | Exploratory research | General science | High-stakes decisions | Particle physics |

## 3.3 Examples + Counterexamples

**Simple Example:**  
Clinical trial α=0.05: Accept ≤5% chance of declaring ineffective drug effective. Protects against false hope

**Failure Case:**  
Running 100 tests at α=0.05 without correction. Expected 5 false positives even if all nulls true (multiple testing problem)

**Edge Case:**  
Asymmetric costs: Cancer screening (false negative costly) vs spam filter (false positive tolerable). Adjust α to context

## 3.4 Layer Breakdown
```
Significance Level Framework:
├─ Definition: α = P(Reject H₀ | H₀ true) = Type I error rate
├─ Decision Rule:
│   ├─ If p-value < α: Reject H₀ (declare "significant")
│   └─ If p-value ≥ α: Fail to reject H₀ (insufficient evidence)
├─ Common Conventions:
│   ├─ α = 0.05: Default in social/biological sciences
│   ├─ α = 0.01: More conservative (medicine)
│   ├─ α = 0.10: Exploratory studies
│   └─ α = 5×10⁻⁸: Genome-wide association studies (GWAS)
├─ Trade-offs:
│   ├─ Lower α → Fewer false positives (good)
│   ├─ Lower α → More false negatives, β increases (bad)
│   ├─ Lower α → Need larger sample for same power
│   └─ Optimal α depends on costs of Type I vs II errors
├─ Multiple Testing Corrections:
│   ├─ Bonferroni: α_adjusted = α / n_tests
│   │   └─ Very conservative; controls family-wise error rate (FWER)
│   ├─ Holm-Bonferroni: Sequential Bonferroni (less conservative)
│   ├─ Benjamini-Hochberg: Controls false discovery rate (FDR)
│   │   └─ α_i = (i/m) × α for i-th smallest p-value
│   └─ Sidak: α_adjusted = 1 - (1-α)^(1/n)
└─ One-tailed vs Two-tailed:
    ├─ One-tailed: α entirely in one tail (directional hypothesis)
    └─ Two-tailed: α/2 in each tail (non-directional)
```

## 3.5 Mini-Project
Explore significance level effects and multiple testing:
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Demonstration 1: Effect of α on decisions
print("=== Significance Level Decision Example ===")
sample = np.random.normal(102, 15, 50)
t_stat, p_value = stats.ttest_1samp(sample, 100)

alpha_levels = [0.10, 0.05, 0.01, 0.001]
print(f"Observed p-value: {p_value:.4f}")
for alpha in alpha_levels:
    decision = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT H₀"
    print(f"α = {alpha}: {decision}")

# Demonstration 2: Power vs α trade-off
print("\n=== Power vs Alpha Trade-off ===")
n = 50
true_effect = 0.5  # Cohen's d
sigma = 15
true_mean = 100 + true_effect * sigma

alpha_range = np.linspace(0.001, 0.20, 50)
power_values = []

for alpha in alpha_range:
    # Calculate power: P(reject H₀ | H₁ true)
    critical_t = stats.t.ppf(1 - alpha/2, n-1)
    se = sigma / np.sqrt(n)
    noncentrality = (true_mean - 100) / se
    power = 1 - stats.nct.cdf(critical_t, n-1, noncentrality)
    power_values.append(power)

plt.figure(figsize=(14, 10))

# Plot 1: Power vs α
plt.subplot(2, 3, 1)
plt.plot(alpha_range, power_values, linewidth=2)
plt.axvline(0.05, color='r', linestyle='--', label='Conventional α=0.05')
plt.xlabel('Significance Level (α)')
plt.ylabel('Statistical Power (1-β)')
plt.title(f'Power Increases with α\n(n={n}, d={true_effect})')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Type I vs Type II error trade-off
plt.subplot(2, 3, 2)
type_I = alpha_range
type_II = 1 - np.array(power_values)
plt.plot(alpha_range, type_I, label='Type I Error (α)', linewidth=2)
plt.plot(alpha_range, type_II, label='Type II Error (β)', linewidth=2)
plt.axvline(0.05, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Significance Level (α)')
plt.ylabel('Error Rate')
plt.title('Error Trade-off: α vs β')
plt.legend()
plt.grid(True, alpha=0.3)

# Demonstration 3: Multiple testing problem
print("\n=== Multiple Testing Inflation ===")
n_tests = 20
n_simulations = 10000
false_positive_counts = []

for _ in range(n_simulations):
    # All nulls are true
    p_values = []
    for _ in range(n_tests):
        sample = np.random.normal(100, 15, 30)
        _, p_val = stats.ttest_1samp(sample, 100)
        p_values.append(p_val)
    
    # Count false positives
    false_positives = np.sum(np.array(p_values) < 0.05)
    false_positive_counts.append(false_positives)

false_positive_counts = np.array(false_positive_counts)
prob_any_false_positive = np.mean(false_positive_counts > 0)

print(f"Number of tests: {n_tests}")
print(f"Nominal α per test: 0.05")
print(f"Expected false positives per experiment: {np.mean(false_positive_counts):.2f}")
print(f"Probability of ≥1 false positive: {prob_any_false_positive:.3f}")
print(f"Theoretical (1-(1-0.05)^{n_tests}): {1-(1-0.05)**n_tests:.3f}")

# Plot 3: Multiple testing false positive distribution
plt.subplot(2, 3, 3)
plt.hist(false_positive_counts, bins=range(0, max(false_positive_counts)+2), 
         alpha=0.7, edgecolor='black', density=True)
plt.axvline(np.mean(false_positive_counts), color='r', linestyle='--', 
            label=f'Mean={np.mean(false_positive_counts):.2f}')
plt.xlabel('Number of False Positives (out of 20 tests)')
plt.ylabel('Probability')
plt.title('Multiple Testing: False Positive Distribution')
plt.legend()

# Demonstration 4: Multiple testing corrections
print("\n=== Multiple Testing Corrections ===")
n_tests = 20
true_p_values = [0.001, 0.01, 0.02, 0.04, 0.06] + [0.5] * 15  # 4 true effects
true_p_values.sort()

# No correction
alpha = 0.05
significant_uncorrected = [p < alpha for p in true_p_values]

# Bonferroni
alpha_bonferroni = alpha / n_tests
significant_bonferroni = [p < alpha_bonferroni for p in true_p_values]

# Benjamini-Hochberg (FDR)
sorted_indices = np.argsort(true_p_values)
significant_bh = [False] * n_tests
for i in range(n_tests):
    threshold = (i + 1) / n_tests * alpha
    if true_p_values[i] < threshold:
        significant_bh[i] = True
    else:
        break

print(f"Uncorrected (α={alpha}): {sum(significant_uncorrected)} significant")
print(f"Bonferroni (α={alpha_bonferroni:.4f}): {sum(significant_bonferroni)} significant")
print(f"Benjamini-Hochberg: {sum(significant_bh)} significant")

# Plot 4: Multiple testing correction comparison
plt.subplot(2, 3, 4)
x_pos = np.arange(n_tests)
colors = ['red' if i < 4 else 'blue' for i in range(n_tests)]
plt.bar(x_pos, true_p_values, color=colors, alpha=0.6, edgecolor='black')
plt.axhline(alpha, color='black', linestyle='-', linewidth=2, label='Uncorrected α=0.05')
plt.axhline(alpha_bonferroni, color='red', linestyle='--', linewidth=2, 
            label=f'Bonferroni α={alpha_bonferroni:.4f}')
plt.xlabel('Test Index (sorted by p-value)')
plt.ylabel('P-value')
plt.title('Multiple Testing Corrections\n(Red=true effects, Blue=nulls)')
plt.legend()
plt.yscale('log')

# Plot 5: Sample size needed for different α
plt.subplot(2, 3, 5)
effect_size = 0.5  # Cohen's d
desired_power = 0.80
alpha_values = [0.10, 0.05, 0.01, 0.001]
sample_sizes = []

for alpha in alpha_values:
    # Approximate sample size calculation
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(desired_power)
    n_needed = ((z_alpha + z_beta) / effect_size) ** 2
    sample_sizes.append(n_needed)

plt.barh(range(len(alpha_values)), sample_sizes, edgecolor='black')
plt.yticks(range(len(alpha_values)), [f'α={a}' for a in alpha_values])
plt.xlabel('Sample Size Required')
plt.title(f'Sample Size vs α\n(d={effect_size}, power={desired_power})')
for i, n in enumerate(sample_sizes):
    plt.text(n, i, f'  n≈{int(n)}', va='center')
plt.grid(True, alpha=0.3, axis='x')

# Plot 6: α in different fields
plt.subplot(2, 3, 6)
fields = ['Exploratory\nStudies', 'Social\nSciences', 'Medical\nTrials', 'Particle\nPhysics', 'GWAS']
alphas = [0.10, 0.05, 0.01, 5e-8, 5e-8]
colors_field = ['lightblue', 'skyblue', 'orange', 'red', 'darkred']

plt.barh(range(len(fields)), alphas, color=colors_field, edgecolor='black')
plt.yticks(range(len(fields)), fields)
plt.xlabel('Significance Level (α)')
plt.title('Conventional α by Field')
plt.xscale('log')
for i, (field, alpha_val) in enumerate(zip(fields, alphas)):
    if alpha_val < 1e-5:
        plt.text(alpha_val, i, f'  {alpha_val:.0e}', va='center', fontsize=9)
    else:
        plt.text(alpha_val, i, f'  {alpha_val}', va='center')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Demonstration 5: Decision theory approach
print("\n=== Decision Theory: Optimal α ===")
# Cost-based approach
cost_false_positive = 1000  # Type I error cost
cost_false_negative = 5000  # Type II error cost
effect_size = 0.5
n = 50

optimal_alpha_found = None
min_expected_cost = float('inf')

for alpha in np.linspace(0.001, 0.20, 100):
    # Calculate beta for this alpha
    critical_t = stats.t.ppf(1 - alpha/2, n-1)
    se = 15 / np.sqrt(n)
    noncentrality = (effect_size * 15) / se
    power = 1 - stats.nct.cdf(critical_t, n-1, noncentrality)
    beta = 1 - power
    
    expected_cost = alpha * cost_false_positive + beta * cost_false_negative
    
    if expected_cost < min_expected_cost:
        min_expected_cost = expected_cost
        optimal_alpha_found = alpha

print(f"Cost of Type I error: ${cost_false_positive}")
print(f"Cost of Type II error: ${cost_false_negative}")
print(f"Optimal α: {optimal_alpha_found:.3f}")
print(f"Conventional α=0.05 may not minimize expected loss")
```

## 3.6 Challenge Round
When is the conventional α=0.05 inappropriate?
- **High-stakes decisions**: Medical device approval → use α=0.01 or stricter; false positive costly
- **Many comparisons**: Genomics (millions of tests) → α=5×10⁻⁸; Bonferroni too conservative, use FDR
- **Asymmetric costs**: If false negative far worse than false positive → increase α (e.g., 0.10)
- **Exploratory research**: Hypothesis generation → α=0.10 acceptable; confirmatory needs 0.05
- **Sequential testing**: Checking data multiple times → adjust α for interim analyses (spend α conservatively)

## 3.7 Key References
- [Choosing Significance Levels](https://en.wikipedia.org/wiki/Statistical_significance) - Historical context, field conventions, criticisms
- [Multiple Testing Corrections](https://www.statisticshowto.com/familywise-error-rate/) - FWER, FDR, when to use each method
- [Fisher vs Neyman-Pearson](https://errorstatistics.com/2014/11/17/s-senn-fisher-and-the-p-value-vs-neyman-and-the-alpha-level/) - Philosophical debate on pre-specified α

---
**Status:** Arbitrary but essential convention | **Complements:** P-values, Type I/II Errors, Power Analysis
