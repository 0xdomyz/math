# Sampling Theory

## 1. Concept Skeleton
**Definition:** Statistical framework for selecting subsets from populations to make inferences about entire population with quantifiable uncertainty  
**Purpose:** Obtain representative data efficiently, enable generalization, estimate population parameters with confidence  
**Prerequisites:** Probability concepts, population vs sample distinction, bias understanding, basic statistics

## 2. Comparative Framing
| Method | Simple Random | Stratified | Cluster | Systematic |
|--------|--------------|------------|---------|------------|
| **Selection** | Every unit equal chance | Random within subgroups | Random groups, all units | Every kth unit |
| **Efficiency** | Baseline | Higher (reduces variance) | Lower (within-cluster similarity) | Moderate |
| **Complexity** | Low | Moderate (need strata info) | Low (natural groupings) | Very low |
| **Use Case** | Homogeneous populations | Heterogeneous with known groups | Geographically dispersed | Systematic ordering |

## 3. Examples + Counterexamples

**Simple Example:**  
Survey 1000 voters from 10 million: Simple random ensures each has 1/10000 selection chance, results generalize to population

**Failure Case:**  
Convenience sampling (mall intercepts): Oversamples shoppers, misses working professionals → biased political estimates

**Edge Case:**  
Stratified sampling with 99% in one stratum: Gains minimal efficiency, complexity cost outweighs benefit

## 4. Layer Breakdown
```
Sampling Framework:
├─ Population:
│   ├─ Target Population: Group of interest
│   ├─ Sampling Frame: List of units available for selection
│   └─ Coverage Error: Frame ≠ population (undercoverage, overcoverage)
├─ Probability Sampling Methods:
│   ├─ Simple Random Sampling (SRS):
│   │   └─ Each unit has equal probability n/N
│   ├─ Stratified Sampling:
│   │   ├─ Divide into homogeneous strata
│   │   ├─ Sample independently within each
│   │   └─ Weights: Proportional or optimal allocation
│   ├─ Cluster Sampling:
│   │   ├─ Select random clusters (PSU - Primary Sampling Units)
│   │   ├─ All units within selected clusters
│   │   └─ Multistage: Sample within clusters (SSU)
│   ├─ Systematic Sampling:
│   │   ├─ Select every kth unit (k = N/n)
│   │   └─ Random start in [1, k]
│   └─ Complex Designs: Combinations of above
├─ Non-Probability Sampling:
│   ├─ Convenience: Easiest to access
│   ├─ Quota: Match population proportions
│   ├─ Purposive: Expert judgment selection
│   └─ Snowball: Referrals from participants
├─ Sample Size Determination:
│   └─ n = (Z²σ²) / E² (for mean estimation)
│   └─ Balances precision, cost, variance
└─ Weighting & Post-Stratification:
    └─ Adjust for unequal selection probabilities
    └─ Correct for nonresponse bias
```

**Interaction:** Define population → Choose method → Determine size → Select units → Adjust weights → Generalize

## 5. Mini-Project
Implement and compare sampling methods:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Create simulated population
np.random.seed(42)
N = 10000

# Population with 3 strata: young, middle, old age groups
age_group = np.random.choice(['Young', 'Middle', 'Old'], size=N, p=[0.3, 0.5, 0.2])
income = np.where(age_group == 'Young', 
                  np.random.normal(40000, 12000, N),
                  np.where(age_group == 'Middle',
                          np.random.normal(70000, 20000, N),
                          np.random.normal(50000, 15000, N)))

population = pd.DataFrame({
    'id': range(N),
    'age_group': age_group,
    'income': income
})

true_mean = population['income'].mean()
true_std = population['income'].std()

print("Population Parameters:")
print(f"Mean income: ${true_mean:,.0f}")
print(f"Std deviation: ${true_std:,.0f}")
print(f"Size: {N}")
print("\nAge group distribution:")
print(population['age_group'].value_counts(normalize=True))

# Sample size
n = 500

# Method 1: Simple Random Sampling
def simple_random_sample(df, n):
    return df.sample(n=n, random_state=42)

# Method 2: Stratified Sampling (Proportional)
def stratified_sample(df, n, strata_col):
    return df.groupby(strata_col, group_keys=False).apply(
        lambda x: x.sample(frac=n/len(df), random_state=42)
    )

# Method 3: Cluster Sampling (simulate geographic clusters)
def cluster_sample(df, n_clusters=50, clusters_to_select=5):
    # Assign random clusters
    df = df.copy()
    df['cluster'] = np.random.randint(0, n_clusters, len(df))
    
    # Select random clusters
    selected_clusters = np.random.choice(n_clusters, clusters_to_select, replace=False)
    return df[df['cluster'].isin(selected_clusters)]

# Method 4: Systematic Sampling
def systematic_sample(df, n):
    k = len(df) // n
    start = np.random.randint(0, k)
    indices = np.arange(start, len(df), k)[:n]
    return df.iloc[indices]

# Perform sampling
srs_sample = simple_random_sample(population, n)
stratified_sample_df = stratified_sample(population, n, 'age_group')
cluster_sample_df = cluster_sample(population)
systematic_sample_df = systematic_sample(population, n)

# Calculate estimates
methods = {
    'Simple Random': srs_sample,
    'Stratified': stratified_sample_df,
    'Cluster': cluster_sample_df,
    'Systematic': systematic_sample_df
}

results = []
for method_name, sample_df in methods.items():
    sample_mean = sample_df['income'].mean()
    sample_std = sample_df['income'].std()
    sample_se = sample_std / np.sqrt(len(sample_df))
    bias = sample_mean - true_mean
    
    # 95% CI
    ci_lower = sample_mean - 1.96 * sample_se
    ci_upper = sample_mean + 1.96 * sample_se
    
    results.append({
        'Method': method_name,
        'Sample Size': len(sample_df),
        'Estimate': sample_mean,
        'Bias': bias,
        'SE': sample_se,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'Covers_True': ci_lower <= true_mean <= ci_upper
    })

results_df = pd.DataFrame(results)
print("\nSampling Method Comparison:")
print(results_df.to_string(index=False))

# Simulation: Repeated sampling to assess variability
n_simulations = 1000
srs_means = []
stratified_means = []

for i in range(n_simulations):
    srs = population.sample(n=n)
    srs_means.append(srs['income'].mean())
    
    strat = population.groupby('age_group', group_keys=False).apply(
        lambda x: x.sample(frac=n/len(population))
    )
    stratified_means.append(strat['income'].mean())

srs_means = np.array(srs_means)
stratified_means = np.array(stratified_means)

print(f"\nSimulation Results ({n_simulations} iterations):")
print(f"SRS - Mean: ${srs_means.mean():,.0f}, SD: ${srs_means.std():,.0f}")
print(f"Stratified - Mean: ${stratified_means.mean():,.0f}, SD: ${stratified_means.std():,.0f}")
print(f"Efficiency gain: {(1 - stratified_means.std()/srs_means.std())*100:.1f}%")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Population distribution
axes[0, 0].hist(population['income'], bins=50, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(true_mean, color='r', linewidth=2, label=f'True Mean: ${true_mean:,.0f}')
axes[0, 0].set_title('Population Income Distribution')
axes[0, 0].set_xlabel('Income ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Plot 2: Sample distributions comparison
for method_name, sample_df in methods.items():
    axes[0, 1].hist(sample_df['income'], bins=30, alpha=0.5, label=method_name)

axes[0, 1].axvline(true_mean, color='r', linewidth=2, linestyle='--', label='True Mean')
axes[0, 1].set_title('Sample Distributions by Method')
axes[0, 1].set_xlabel('Income ($)')
axes[0, 1].legend()

# Plot 3: Estimates with confidence intervals
methods_list = results_df['Method'].values
estimates = results_df['Estimate'].values
ci_lower = results_df['CI_Lower'].values
ci_upper = results_df['CI_Upper'].values

y_pos = np.arange(len(methods_list))
axes[0, 2].errorbar(estimates, y_pos, 
                    xerr=[(estimates - ci_lower), (ci_upper - estimates)],
                    fmt='o', markersize=8, capsize=5)
axes[0, 2].axvline(true_mean, color='r', linestyle='--', linewidth=2, label='True Mean')
axes[0, 2].set_yticks(y_pos)
axes[0, 2].set_yticklabels(methods_list)
axes[0, 2].set_xlabel('Income Estimate ($)')
axes[0, 2].set_title('Point Estimates with 95% CI')
axes[0, 2].legend()
axes[0, 2].grid(axis='x', alpha=0.3)

# Plot 4: Stratified vs population by group
axes[1, 0].bar(['Young', 'Middle', 'Old'], 
               population.groupby('age_group')['income'].mean(),
               alpha=0.6, label='Population')
axes[1, 0].bar(['Young', 'Middle', 'Old'],
               stratified_sample_df.groupby('age_group')['income'].mean(),
               alpha=0.6, label='Stratified Sample')
axes[1, 0].set_ylabel('Mean Income ($)')
axes[1, 0].set_title('Mean Income by Age Group')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 5: Simulation - Sampling distribution of SRS
axes[1, 1].hist(srs_means, bins=50, alpha=0.7, edgecolor='black', density=True)
axes[1, 1].axvline(true_mean, color='r', linewidth=2, label='True Mean')
axes[1, 1].axvline(srs_means.mean(), color='g', linestyle='--', linewidth=2, label='Mean of Estimates')
axes[1, 1].set_title(f'SRS Sampling Distribution\n({n_simulations} samples, n={n})')
axes[1, 1].set_xlabel('Sample Mean Income ($)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()

# Plot 6: Comparison of sampling distributions
axes[1, 2].hist(srs_means, bins=50, alpha=0.5, label='SRS', density=True)
axes[1, 2].hist(stratified_means, bins=50, alpha=0.5, label='Stratified', density=True)
axes[1, 2].axvline(true_mean, color='r', linewidth=2, linestyle='--', label='True Mean')
axes[1, 2].set_title('SRS vs Stratified: Sampling Distributions')
axes[1, 2].set_xlabel('Sample Mean Income ($)')
axes[1, 2].set_ylabel('Density')
axes[1, 2].legend()

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is sampling theory the wrong tool?
- Census feasible: Sample entire population when small and accessible
- Non-generalizable goals: Case study research, exploratory qualitative studies
- Population unknown: Cannot define sampling frame (hidden populations)
- Real-time streaming data: Use online/sequential methods instead
- Perfect data required: Safety-critical applications where errors unacceptable

## 7. Key References
- [Sampling Methods Overview (Wikipedia)](https://en.wikipedia.org/wiki/Sampling_(statistics))
- [Cochran, Sampling Techniques (Book)](https://www.wiley.com/en-us/Sampling+Techniques%2C+3rd+Edition-p-9780471162407)
- [Survey Sampling Explained (Pew Research)](https://www.pewresearch.org/methods/u-s-survey-research/sampling/)

---
**Status:** Foundational for inference | **Complements:** Sampling Bias, Survey Design, Central Limit Theorem
