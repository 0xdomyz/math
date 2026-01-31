# P-values

## 2.1 Concept Skeleton
**Definition:** Probability of observing data as extreme as sample (or more) assuming null hypothesis true  
**Purpose:** Quantify strength of evidence against H₀; decision criterion for hypothesis testing  
**Prerequisites:** Probability, hypothesis testing framework, test statistics, sampling distributions

## 2.2 Comparative Framing
| Measure | P-value | Confidence Interval | Effect Size | Bayes Factor |
|---------|---------|-------------------|------------|-------------|
| **Meaning** | P(data\|H₀) | Range for parameter | Magnitude of effect | P(data\|H₁)/P(data\|H₀) |
| **Null Dependent** | Yes (assumes H₀) | Yes (test-based) | No (descriptive) | No (compares both) |
| **Sample Size Effect** | Decreases with large n | Narrows with large n | Unchanged | More decisive |
| **Interpretation** | Evidence strength | Plausible values | Practical importance | Hypothesis support |

## 2.3 Examples + Counterexamples

**Simple Example:**  
Coin flip test: 65 heads in 100 flips. p-value = 0.003 → strong evidence coin biased (if H₀: p=0.5)

**Failure Case:**  
p=0.049 → "significant", p=0.051 → "not significant". Treating threshold as absolute; effects may be identical

**Edge Case:**  
Very large sample: Mean difference = 0.01, p < 0.001. Statistically significant but trivially small effect (practical irrelevance)

## 2.4 Layer Breakdown
```
P-value Interpretation Framework:
├─ Definition: P(observe data or more extreme | H₀ true)
├─ Calculation Steps:
│   ├─ 1. Specify H₀ and H₁
│   ├─ 2. Choose test statistic (t, z, χ², F)
│   ├─ 3. Compute observed statistic from data
│   ├─ 4. Determine reference distribution under H₀
│   └─ 5. Calculate tail probability beyond observed value
├─ Tail Types:
│   ├─ Two-tailed: P(|statistic| ≥ |observed|)
│   ├─ Right-tailed: P(statistic ≥ observed)
│   └─ Left-tailed: P(statistic ≤ observed)
├─ What P-value IS:
│   ├─ Measure of data-H₀ compatibility
│   ├─ Continuous evidence metric (not binary)
│   └─ Conditional on H₀ being true
├─ What P-value IS NOT:
│   ├─ NOT P(H₀ true | data)
│   ├─ NOT P(result due to chance)
│   ├─ NOT size of effect
│   ├─ NOT importance of result
│   └─ NOT 1 - P(replication)
└─ Common Thresholds:
    ├─ p < 0.001: Very strong evidence
    ├─ p < 0.01: Strong evidence
    ├─ p < 0.05: Moderate evidence (conventional)
    ├─ p < 0.10: Weak evidence (exploratory)
    └─ p ≥ 0.05: Insufficient evidence (NOT proof of H₀)
```

## 2.5 Mini-Project
Explore p-value behavior and misconceptions:
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Demonstration 1: P-value calculation (manual vs library)
print("=== P-value Calculation Example ===")
# H₀: μ = 100, H₁: μ ≠ 100
sample = np.array([102, 105, 98, 110, 95, 103, 99, 107, 101, 104])
x_bar = np.mean(sample)
s = np.std(sample, ddof=1)
n = len(sample)
mu_0 = 100

# Manual calculation
t_stat = (x_bar - mu_0) / (s / np.sqrt(n))
df = n - 1
p_value_manual = 2 * (1 - stats.t.cdf(abs(t_stat), df))  # Two-tailed

# Library calculation
t_stat_lib, p_value_lib = stats.ttest_1samp(sample, mu_0)

print(f"Sample mean: {x_bar:.2f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"P-value (manual): {p_value_manual:.4f}")
print(f"P-value (scipy): {p_value_lib:.4f}")
print(f"Interpretation: {'Reject H₀' if p_value_lib < 0.05 else 'Fail to reject H₀'} at α=0.05")

# Demonstration 2: P-value distribution under H₀
print("\n=== P-value Distribution Under H₀ ===")
n_simulations = 10000
p_values = []

for _ in range(n_simulations):
    # Generate data from H₀: μ = 100
    sim_sample = np.random.normal(100, 15, 30)
    _, p_val = stats.ttest_1samp(sim_sample, 100)
    p_values.append(p_val)

p_values = np.array(p_values)
false_positives = np.sum(p_values < 0.05) / n_simulations
print(f"False positive rate (α=0.05): {false_positives:.3f}")
print(f"Expected: 0.050")

plt.figure(figsize=(14, 10))

# Plot 1: P-value histogram under H₀
plt.subplot(2, 3, 1)
plt.hist(p_values, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.axhline(1, color='r', linestyle='--', label='Uniform(0,1)')
plt.axvline(0.05, color='g', linestyle='--', label='α=0.05')
plt.xlabel('P-value')
plt.ylabel('Density')
plt.title('P-value Distribution Under H₀ (Uniform)')
plt.legend()

# Plot 2: P-value vs sample size
plt.subplot(2, 3, 2)
true_mean = 105  # True population mean
sample_sizes = [10, 20, 50, 100, 200, 500]
p_values_by_n = []

for n in sample_sizes:
    p_vals_n = []
    for _ in range(1000):
        sim_sample = np.random.normal(true_mean, 15, n)
        _, p_val = stats.ttest_1samp(sim_sample, 100)
        p_vals_n.append(p_val)
    p_values_by_n.append(np.median(p_vals_n))

plt.plot(sample_sizes, p_values_by_n, 'o-', linewidth=2)
plt.axhline(0.05, color='r', linestyle='--', label='α=0.05')
plt.xlabel('Sample Size')
plt.ylabel('Median P-value')
plt.title('P-value Decreases with Sample Size (True μ=105)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: P-value vs effect size
plt.subplot(2, 3, 3)
effect_sizes = np.linspace(0, 1.5, 20)  # Cohen's d
n = 50
p_values_by_effect = []

for effect in effect_sizes:
    # effect = (μ - μ₀) / σ
    true_mean = 100 + effect * 15
    p_vals_effect = []
    for _ in range(500):
        sim_sample = np.random.normal(true_mean, 15, n)
        _, p_val = stats.ttest_1samp(sim_sample, 100)
        p_vals_effect.append(p_val)
    p_values_by_effect.append(np.median(p_vals_effect))

plt.plot(effect_sizes, p_values_by_effect, 'o-', linewidth=2)
plt.axhline(0.05, color='r', linestyle='--', label='α=0.05')
plt.xlabel("Cohen's d (Effect Size)")
plt.ylabel('Median P-value')
plt.title('P-value vs Effect Size (n=50)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Common misinterpretation
plt.subplot(2, 3, 4)
# Show p=0.05 does NOT mean 5% chance H₀ is true
prior_prob_H0 = np.linspace(0.01, 0.99, 100)
p_value = 0.05
# Using Bayes: P(H₀|data) depends on prior
# Simplified: P(H₀|p=0.05) ≈ P(H₀) * P(p=0.05|H₀) / P(p=0.05)
# This is illustrative, not exact
posterior_H0 = prior_prob_H0 * 1 / (prior_prob_H0 + (1-prior_prob_H0)*20)

plt.plot(prior_prob_H0, posterior_H0, linewidth=2)
plt.xlabel('Prior P(H₀ true)')
plt.ylabel('Posterior P(H₀ true | p=0.05)')
plt.title('P(H₀|data) ≠ p-value\n(Depends on Prior)')
plt.axhline(0.05, color='r', linestyle='--', label='p-value=0.05')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Multiple testing inflation
plt.subplot(2, 3, 5)
n_tests_range = range(1, 51)
family_wise_error = [1 - (1 - 0.05)**n for n in n_tests_range]

plt.plot(n_tests_range, family_wise_error, linewidth=2)
plt.axhline(0.05, color='r', linestyle='--', label='Intended α=0.05')
plt.xlabel('Number of Independent Tests')
plt.ylabel('Family-Wise Error Rate')
plt.title('Multiple Testing Inflates False Positives')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: P-value dance (replication variability)
plt.subplot(2, 3, 6)
true_effect = 0.5  # Cohen's d
n = 30
replication_p_values = []

for _ in range(100):
    sim_sample = np.random.normal(100 + true_effect*15, 15, n)
    _, p_val = stats.ttest_1samp(sim_sample, 100)
    replication_p_values.append(p_val)

plt.hist(replication_p_values, bins=20, alpha=0.7, edgecolor='black')
plt.axvline(0.05, color='r', linestyle='--', label='α=0.05')
plt.xlabel('P-value in Replication')
plt.ylabel('Frequency')
plt.title('P-value Variability Across Studies\n(True effect d=0.5, n=30)')
significant = np.sum(np.array(replication_p_values) < 0.05)
plt.text(0.5, plt.ylim()[1]*0.8, f'{significant}% significant', fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()

# Demonstration 3: Common misconceptions
print("\n=== Common P-value Misconceptions ===")
print("❌ p=0.03 means 3% chance result is due to chance")
print("✓  p=0.03 means: if H₀ true, 3% chance of seeing data this extreme\n")

print("❌ p=0.05 means 5% chance H₀ is true")
print("✓  Cannot calculate P(H₀|data) without prior probability\n")

print("❌ p=0.001 is 'more significant' than p=0.04")
print("✓  Smaller p indicates stronger evidence, but don't overinterpret\n")

print("❌ p>0.05 means no effect exists")
print("✓  Absence of evidence ≠ evidence of absence; may lack power\n")

print("❌ p=0.05 is magic threshold")
print("✓  Arbitrary convention; consider effect size, context, consequences")
```

## 2.6 Challenge Round
When are p-values misleading or inappropriate?
- **Multiple testing**: Running 20 tests → expect 1 false positive at α=0.05; use Bonferroni/FDR correction
- **p-hacking**: Trying analyses until p<0.05; cherry-picking inflates false positives
- **Large samples**: Trivial effects become "significant"; report effect size and CI
- **Stopping rules**: Checking p-value repeatedly, stopping when significant → inflated Type I error
- **Exploratory analysis**: No specific hypothesis; p-values less meaningful; better for confirmatory tests

## 2.7 Key References
- [ASA Statement on P-values](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf) - Official guidance: 6 principles, common misinterpretations
- [P-value Misconceptions](https://fivethirtyeight.com/features/statisticians-found-one-thing-they-can-agree-on-its-time-to-stop-misusing-p-values/) - Accessible explanation of widespread errors
- [Dance of P-values](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108) - Replication variability visualization

---
**Status:** Most misunderstood statistic | **Complements:** Hypothesis Testing, Effect Size, Confidence Intervals
