# Multiple Comparisons Problem

## 1.1 Concept Skeleton
**Definition:** Inflated false positive rate when conducting multiple hypothesis tests simultaneously; more tests increase chance of Type I errors  
**Purpose:** Control family-wise error rate, adjust p-values, prevent spurious discoveries  
**Prerequisites:** Hypothesis testing, p-values, Type I error, probability multiplication

## 1.2 Comparative Framing
| Method | No Adjustment | Bonferroni | False Discovery Rate (FDR) |
|--------|--------------|-----------|---------------------------|
| **Control** | Per-test error rate | Family-wise error rate (FWER) | Expected false discovery proportion |
| **Threshold** | α (e.g., 0.05) | α/m (m tests) | Adaptive based on p-value ranks |
| **Power** | Highest | Lowest (conservative) | Moderate (balanced) |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Test 20 independent hypotheses at α=0.05: Expected 1 false positive even if all null true (0.05×20=1)

**Failure Case:**  
No correction with 100 tests: ~99.4% chance of ≥1 false positive (1-(0.95)^100), making results unreliable

**Edge Case:**  
Single planned hypothesis: No adjustment needed; multiple comparisons only for exploratory analyses

## 1.4 Layer Breakdown
```
Multiple Comparisons Framework:
├─ Problem:
│   ├─ Per-comparison error rate: α (e.g., 0.05)
│   ├─ Family-wise error rate: P(≥1 false positive)
│   ├─ With m tests: FWER ≈ 1 - (1-α)^m
│   └─ Example: 20 tests → FWER ≈ 0.64 (64% chance of error)
├─ Bonferroni Correction:
│   ├─ Method: Test each at α/m instead of α
│   ├─ Controls: FWER ≤ α
│   ├─ Pros: Simple, conservative, guarantees control
│   ├─ Cons: Very conservative (low power), assumes independence
│   └─ Use: Small number of tests, want strong FWER control
├─ Holm-Bonferroni (Sequential):
│   ├─ Method: Order p-values, reject until p > α/(m-i+1)
│   ├─ Controls: FWER ≤ α
│   ├─ Pros: More powerful than Bonferroni, still simple
│   └─ Use: Step-up procedure, uniformly better than Bonferroni
├─ False Discovery Rate (FDR):
│   ├─ Method: Benjamini-Hochberg procedure
│   ├─ Controls: E[FDR] ≤ α (expected proportion of false discoveries)
│   ├─ Calculation: Find largest i where p(i) ≤ (i/m)α
│   ├─ Pros: Much more powerful, appropriate for exploratory work
│   ├─ Cons: Allows some false positives
│   └─ Use: High-dimensional data (genomics, neuroimaging)
├─ Šidák Correction:
│   ├─ Method: Test each at 1-(1-α)^(1/m)
│   ├─ Assumes: Independence between tests
│   └─ Less conservative than Bonferroni when independent
├─ Permutation/Resampling Methods:
│   ├─ Method: Generate null distribution via permutation
│   ├─ Pros: Exact control, no independence assumption
│   └─ Cons: Computationally intensive
└─ When to Apply:
    ├─ Multiple endpoints in clinical trial
    ├─ Subgroup analyses
    ├─ Post-hoc comparisons (ANOVA)
    ├─ Feature selection in machine learning
    └─ High-throughput screening (genomics, proteomics)
```

## 1.5 Mini-Project
Demonstrate multiple comparisons problem and corrections:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

np.random.seed(42)

# Demonstrate the problem
def simulate_multiple_tests(n_tests, alpha=0.05, n_sims=1000, effect_size=0):
    """Simulate multiple hypothesis tests"""
    false_positives = []
    
    for _ in range(n_sims):
        # Generate data under null (no effect) or alternative
        p_values = []
        for _ in range(n_tests):
            group1 = np.random.normal(0, 1, 30)
            group2 = np.random.normal(effect_size, 1, 30)
            _, p = stats.ttest_ind(group1, group2)
            p_values.append(p)
        
        # Count false positives (when effect_size=0, all rejections are false)
        false_positives.append(np.sum(np.array(p_values) < alpha))
    
    return np.array(false_positives)

# 1. Show inflated error rate
print("=== Multiple Comparisons Problem ===")
n_tests_range = [1, 5, 10, 20, 50, 100]

for n_tests in n_tests_range:
    fp = simulate_multiple_tests(n_tests, alpha=0.05, n_sims=1000)
    at_least_one_fp = np.mean(fp > 0)
    theoretical = 1 - (1 - 0.05)**n_tests
    
    print(f"\n{n_tests} tests:")
    print(f"  Theoretical FWER: {theoretical:.3f}")
    print(f"  Observed FWER: {at_least_one_fp:.3f}")
    print(f"  Average false positives: {np.mean(fp):.2f}")

# 2. Compare correction methods
print("\n\n=== Correction Methods Comparison ===")

# Generate p-values: 90% null, 10% alternative
n_total = 100
n_alternative = 10
n_null = n_total - n_alternative

np.random.seed(42)
# Null hypotheses (no effect)
p_null = np.random.uniform(0, 1, n_null)
# Alternative hypotheses (with effect) - beta distribution skewed toward 0
p_alternative = np.random.beta(0.5, 5, n_alternative)
p_values = np.concatenate([p_null, p_alternative])
true_nulls = np.concatenate([np.ones(n_null), np.zeros(n_alternative)])

# Apply different corrections
alpha = 0.05

# No correction
reject_none = p_values < alpha

# Bonferroni
bonf_threshold = alpha / n_total
reject_bonf = p_values < bonf_threshold

# Holm-Bonferroni
reject_holm, pvals_corrected_holm, _, _ = multipletests(p_values, alpha=alpha, 
                                                         method='holm')

# FDR (Benjamini-Hochberg)
reject_fdr, pvals_corrected_fdr, _, _ = multipletests(p_values, alpha=alpha, 
                                                       method='fdr_bh')

# Calculate performance metrics
def calc_metrics(rejections, true_nulls):
    true_positives = np.sum(rejections & (true_nulls == 0))
    false_positives = np.sum(rejections & (true_nulls == 1))
    true_negatives = np.sum(~rejections & (true_nulls == 1))
    false_negatives = np.sum(~rejections & (true_nulls == 0))
    
    power = true_positives / np.sum(true_nulls == 0) if np.sum(true_nulls == 0) > 0 else 0
    fdr = false_positives / np.sum(rejections) if np.sum(rejections) > 0 else 0
    
    return {
        'rejections': np.sum(rejections),
        'true_pos': true_positives,
        'false_pos': false_positives,
        'power': power,
        'fdr': fdr
    }

methods = {
    'No correction': reject_none,
    'Bonferroni': reject_bonf,
    'Holm': reject_holm,
    'FDR (BH)': reject_fdr
}

print(f"\nTrue state: {n_null} null, {n_alternative} alternative hypotheses")
print(f"Target α = {alpha}\n")

for name, rejections in methods.items():
    metrics = calc_metrics(rejections, true_nulls)
    print(f"{name}:")
    print(f"  Total rejections: {metrics['rejections']}")
    print(f"  True positives: {metrics['true_pos']}")
    print(f"  False positives: {metrics['false_pos']}")
    print(f"  Power: {metrics['power']:.3f}")
    print(f"  FDR: {metrics['fdr']:.3f}\n")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. FWER vs number of tests
n_tests_plot = np.arange(1, 101)
fwer_theoretical = 1 - (1 - 0.05)**n_tests_plot

axes[0, 0].plot(n_tests_plot, fwer_theoretical, 'b-', linewidth=2)
axes[0, 0].axhline(0.05, color='r', linestyle='--', label='Nominal α = 0.05')
axes[0, 0].set_xlabel('Number of Tests')
axes[0, 0].set_ylabel('Family-Wise Error Rate')
axes[0, 0].set_title('Multiple Testing Problem: FWER Inflation')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. P-value distributions
axes[0, 1].hist(p_null, bins=20, alpha=0.6, label='Null (uniform)', edgecolor='black')
axes[0, 1].hist(p_alternative, bins=20, alpha=0.6, label='Alternative (skewed)', 
                edgecolor='black')
axes[0, 1].axvline(alpha, color='r', linestyle='--', label='α = 0.05')
axes[0, 1].axvline(bonf_threshold, color='orange', linestyle='--', 
                   label=f'Bonferroni = {bonf_threshold:.4f}')
axes[0, 1].set_xlabel('P-value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('P-value Distributions')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Rejection comparison
sorted_idx = np.argsort(p_values)
sorted_p = p_values[sorted_idx]
sorted_nulls = true_nulls[sorted_idx]

axes[1, 0].scatter(range(n_total), sorted_p, c=sorted_nulls, cmap='RdYlGn', 
                   alpha=0.7, s=50, edgecolors='black')
axes[1, 0].axhline(alpha, color='b', linestyle='--', label='No correction')
axes[1, 0].axhline(bonf_threshold, color='orange', linestyle='--', label='Bonferroni')

# Add FDR line (Benjamini-Hochberg)
fdr_line = alpha * np.arange(1, n_total+1) / n_total
axes[1, 0].plot(range(n_total), fdr_line, 'purple', linestyle='--', 
                linewidth=2, label='FDR threshold')

axes[1, 0].set_xlabel('Test Index (sorted by p-value)')
axes[1, 0].set_ylabel('P-value')
axes[1, 0].set_title('Sorted P-values with Thresholds\n(Green=Null, Red=Alternative)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# 4. Power vs FDR tradeoff
effect_sizes = np.linspace(0, 1, 20)
power_none, power_bonf, power_fdr = [], [], []
fdr_none, fdr_bonf, fdr_fdr = [], [], []

for es in effect_sizes:
    # Simulate with varying effect sizes
    n_sim = 200
    results_none, results_bonf, results_fdr = [], [], []
    
    for _ in range(n_sim):
        # Generate p-values
        p_null_sim = np.random.uniform(0, 1, 90)
        p_alt_sim = []
        for _ in range(10):
            g1 = np.random.normal(0, 1, 30)
            g2 = np.random.normal(es, 1, 30)
            _, p = stats.ttest_ind(g1, g2)
            p_alt_sim.append(p)
        
        p_all = np.concatenate([p_null_sim, p_alt_sim])
        true_state = np.concatenate([np.ones(90), np.zeros(10)])
        
        # Apply methods
        reject_n = p_all < 0.05
        reject_b = p_all < (0.05 / 100)
        reject_f, _, _, _ = multipletests(p_all, alpha=0.05, method='fdr_bh')
        
        results_none.append(calc_metrics(reject_n, true_state))
        results_bonf.append(calc_metrics(reject_b, true_state))
        results_fdr.append(calc_metrics(reject_f, true_state))
    
    power_none.append(np.mean([r['power'] for r in results_none]))
    power_bonf.append(np.mean([r['power'] for r in results_bonf]))
    power_fdr.append(np.mean([r['power'] for r in results_fdr]))
    
    fdr_none.append(np.mean([r['fdr'] for r in results_none]))
    fdr_bonf.append(np.mean([r['fdr'] for r in results_bonf]))
    fdr_fdr.append(np.mean([r['fdr'] for r in results_fdr]))

axes[1, 1].plot(effect_sizes, power_none, 'b-', linewidth=2, label='No correction')
axes[1, 1].plot(effect_sizes, power_bonf, 'orange', linewidth=2, label='Bonferroni')
axes[1, 1].plot(effect_sizes, power_fdr, 'purple', linewidth=2, label='FDR')
axes[1, 1].set_xlabel('Effect Size')
axes[1, 1].set_ylabel('Power')
axes[1, 1].set_title('Power Comparison Across Methods')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Practical example: ANOVA post-hoc comparisons
print("\n=== Practical Example: ANOVA Post-Hoc ===")
# Simulate 5 groups
groups = [np.random.normal(i*0.3, 1, 20) for i in range(5)]

# Pairwise comparisons
from itertools import combinations
comparisons = list(combinations(range(5), 2))
p_values_pairwise = []

for i, j in comparisons:
    _, p = stats.ttest_ind(groups[i], groups[j])
    p_values_pairwise.append(p)

print(f"\nNumber of pairwise comparisons: {len(comparisons)}")
print(f"Without correction (α=0.05): {np.sum(np.array(p_values_pairwise) < 0.05)} significant")
print(f"With Bonferroni (α={0.05/len(comparisons):.4f}): {np.sum(np.array(p_values_pairwise) < 0.05/len(comparisons))} significant")

reject_fdr_pair, _, _, _ = multipletests(p_values_pairwise, alpha=0.05, method='fdr_bh')
print(f"With FDR (α=0.05): {np.sum(reject_fdr_pair)} significant")
```

## 1.6 Challenge Round
When is multiple comparison correction NOT needed?
- **Single pre-specified hypothesis**: Primary endpoint in confirmatory trial
- **Independent studies**: Each with own research question (not exploratory)
- **Descriptive analysis**: Presenting data without inference claims
- **Exploratory phase**: Hypothesis generation (but note as exploratory)
- **Strong prior evidence**: Replication of well-established effects

## 1.7 Key References
- [Wikipedia - Multiple Comparisons Problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem)
- [Benjamini & Hochberg (1995) - Controlling FDR](https://www.jstor.org/stable/2346101)
- [Statistics Done Wrong - Multiple Comparisons](https://www.statisticsdonewrong.com/multiple-comparisons.html)
- Thinking: More tests = more chances for false positives; Bonferroni conservative (controls FWER); FDR allows controlled false discoveries; Match correction to context (exploratory vs confirmatory)

---
**Status:** Critical for valid inference | **Complements:** Hypothesis Testing, P-values, Type I Error, High-Dimensional Data Analysis
