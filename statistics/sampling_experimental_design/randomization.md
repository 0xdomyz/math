# Randomization

## 1. Concept Skeleton
**Definition:** Process of randomly assigning experimental units to treatments to eliminate systematic bias and enable causal inference  
**Purpose:** Control confounding, balance known/unknown variables, validate statistical tests, establish causality  
**Prerequisites:** Probability concepts, confounding understanding, experimental design basics, independence concepts

## 2. Comparative Framing
| Method | Simple Randomization | Block Randomization | Stratified Randomization | Minimization |
|--------|---------------------|--------------------|-----------------------|--------------|
| **Mechanism** | Pure chance (coin flip) | Random within blocks | Random within strata | Algorithmic balance |
| **Balance** | Expected, not guaranteed | Guaranteed within blocks | Guaranteed within strata | Actively optimized |
| **Complexity** | Very low | Low-moderate | Moderate | High |
| **Use Case** | Large samples, homogeneous | Sequential enrollment | Multiple prognostic factors | Small trials, covariates |

## 3. Examples + Counterexamples

**Simple Example:**  
Drug trial: Flip coin for each patient → Treatment (heads) or Placebo (tails) ensures unbiased assignment

**Failure Case:**  
Alternating assignment (patient 1→treatment, 2→control): Predictable pattern allows selection bias, investigators may influence enrollment timing

**Edge Case:**  
Simple randomization in small trial (n=20): May yield 15 treatment, 5 control by chance → use block randomization

## 4. Layer Breakdown
```
Randomization Framework:
├─ Core Principles:
│   ├─ Eliminates Selection Bias: Investigator cannot influence assignment
│   ├─ Balances Confounders: Known and unknown factors distributed equally
│   ├─ Justifies Inference: Validates significance tests, causal claims
│   └─ Breaks Temporal Confounding: Order effects neutralized
├─ Randomization Methods:
│   ├─ Simple (Unrestricted) Randomization:
│   │   ├─ Each unit independent probability (e.g., 0.5)
│   │   ├─ Pros: Unpredictable, eliminates bias
│   │   └─ Cons: May produce imbalance in small samples
│   ├─ Block Randomization:
│   │   ├─ Divide into blocks of fixed size (e.g., 4 or 6)
│   │   ├─ Randomize within each block
│   │   ├─ Ensures balance after each block completes
│   │   └─ Use varying block sizes to reduce predictability
│   ├─ Stratified Randomization:
│   │   ├─ Separate randomization within subgroups (strata)
│   │   ├─ Balance important prognostic factors (age, severity)
│   │   └─ Example: Randomize separately for men/women
│   ├─ Cluster Randomization:
│   │   ├─ Randomize groups (schools, clinics) not individuals
│   │   ├─ Use when individual assignment infeasible
│   │   └─ Requires adjustment for intra-cluster correlation
│   ├─ Adaptive Randomization:
│   │   ├─ Covariate-Adaptive: Balance on covariates
│   │   ├─ Response-Adaptive: Favor better-performing treatment
│   │   └─ Minimization: Minimize imbalance across factors
│   └─ Crossover Randomization:
│       ├─ Random sequence of treatments
│       └─ Each subject receives all treatments
├─ Implementation:
│   ├─ Random Number Generators: Computer algorithms (seed-based)
│   ├─ Allocation Concealment: Hide sequence until assignment
│   ├─ Permuted Block Lists: Pre-generated sequences
│   └─ Central Randomization: Third-party service reduces bias
├─ Verification:
│   ├─ Balance Check: Compare baseline characteristics
│   ├─ Consort Diagram: Report randomization flow
│   └─ Sensitivity Analysis: Test robustness to imbalance
└─ Causal Inference:
    ├─ Counterfactual Framework: What if different assignment?
    ├─ Intention-to-Treat: Analyze as randomized
    └─ Per-Protocol: Analyze compliers only (caution: bias)
```

**Interaction:** Generate sequence → Conceal allocation → Assign sequentially → Verify balance → Analyze causally

## 5. Mini-Project
Implement and compare randomization methods:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Simulate patient enrollment with prognostic factors
np.random.seed(42)
n_patients = 100

# Patient characteristics (confounders)
patients = pd.DataFrame({
    'patient_id': range(1, n_patients + 1),
    'age': np.random.normal(55, 12, n_patients),
    'disease_severity': np.random.choice(['Mild', 'Moderate', 'Severe'], 
                                         size=n_patients, p=[0.3, 0.5, 0.2]),
    'sex': np.random.choice(['M', 'F'], size=n_patients, p=[0.6, 0.4])
})

print("Patient Population Summary:")
print(patients.describe())
print("\nDisease Severity Distribution:")
print(patients['disease_severity'].value_counts())
print("\nSex Distribution:")
print(patients['sex'].value_counts())

# Method 1: Simple Randomization
def simple_randomization(n, p=0.5):
    """Assign treatment with probability p"""
    return np.random.binomial(1, p, n)

# Method 2: Block Randomization
def block_randomization(n, block_size=4):
    """Randomize within fixed blocks"""
    n_blocks = int(np.ceil(n / block_size))
    assignments = []
    
    for _ in range(n_blocks):
        block = [1] * (block_size // 2) + [0] * (block_size // 2)
        np.random.shuffle(block)
        assignments.extend(block)
    
    return np.array(assignments[:n])

# Method 3: Stratified Randomization
def stratified_randomization(df, strata_cols):
    """Randomize separately within strata"""
    assignments = []
    
    for _, group in df.groupby(strata_cols):
        n_group = len(group)
        group_assignments = simple_randomization(n_group)
        assignments.extend(group_assignments)
    
    return np.array(assignments)

# Apply randomization methods
patients['simple_rand'] = simple_randomization(n_patients)
patients['block_rand'] = block_randomization(n_patients, block_size=4)
patients['stratified_rand'] = stratified_randomization(patients, ['disease_severity', 'sex'])

# Function to assess balance
def assess_balance(df, assignment_col):
    """Compare baseline characteristics between groups"""
    treatment = df[df[assignment_col] == 1]
    control = df[df[assignment_col] == 0]
    
    balance = {
        'n_treatment': len(treatment),
        'n_control': len(control),
        'age_treatment': treatment['age'].mean(),
        'age_control': control['age'].mean(),
        'age_diff': abs(treatment['age'].mean() - control['age'].mean()),
        'age_pvalue': stats.ttest_ind(treatment['age'], control['age'])[1]
    }
    
    # Check categorical balance
    for severity in ['Mild', 'Moderate', 'Severe']:
        t_prop = (treatment['disease_severity'] == severity).sum() / len(treatment)
        c_prop = (control['disease_severity'] == severity).sum() / len(control)
        balance[f'{severity}_treatment'] = t_prop
        balance[f'{severity}_control'] = c_prop
        balance[f'{severity}_diff'] = abs(t_prop - c_prop)
    
    return balance

# Assess balance for each method
methods = ['simple_rand', 'block_rand', 'stratified_rand']
balance_results = []

for method in methods:
    balance = assess_balance(patients, method)
    balance['method'] = method
    balance_results.append(balance)

balance_df = pd.DataFrame(balance_results)

print("\n" + "="*70)
print("Balance Assessment Across Randomization Methods")
print("="*70)
print("\nSample Size Balance:")
print(balance_df[['method', 'n_treatment', 'n_control']])

print("\nAge Balance:")
print(balance_df[['method', 'age_treatment', 'age_control', 'age_diff', 'age_pvalue']].round(3))

print("\nDisease Severity Balance:")
severity_cols = [col for col in balance_df.columns if 'Mild' in col or 'Moderate' in col or 'Severe' in col]
print(balance_df[['method'] + severity_cols].round(3))

# Simulation: Repeated randomization to assess variability
n_simulations = 1000
imbalance_simple = []
imbalance_block = []

for _ in range(n_simulations):
    # Simple randomization
    simple_assign = simple_randomization(n_patients)
    imbalance_simple.append(abs(simple_assign.sum() - (n_patients / 2)))
    
    # Block randomization
    block_assign = block_randomization(n_patients, block_size=4)
    imbalance_block.append(abs(block_assign.sum() - (n_patients / 2)))

imbalance_simple = np.array(imbalance_simple)
imbalance_block = np.array(imbalance_block)

print(f"\n{n_simulations} Simulation Summary:")
print(f"Simple Randomization - Mean imbalance: {imbalance_simple.mean():.2f} ± {imbalance_simple.std():.2f}")
print(f"Block Randomization - Mean imbalance: {imbalance_block.mean():.2f} ± {imbalance_block.std():.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Sample size balance
method_labels = ['Simple', 'Block', 'Stratified']
x = np.arange(len(method_labels))
width = 0.35

axes[0, 0].bar(x - width/2, balance_df['n_treatment'], width, label='Treatment', alpha=0.8)
axes[0, 0].bar(x + width/2, balance_df['n_control'], width, label='Control', alpha=0.8)
axes[0, 0].axhline(n_patients/2, color='r', linestyle='--', label='Perfect Balance')
axes[0, 0].set_ylabel('Sample Size')
axes[0, 0].set_title('Sample Size Balance by Method')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(method_labels)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Age distribution by group (stratified method)
treatment_ages = patients[patients['stratified_rand'] == 1]['age']
control_ages = patients[patients['stratified_rand'] == 0]['age']

axes[0, 1].hist(treatment_ages, bins=20, alpha=0.6, label='Treatment', edgecolor='black')
axes[0, 1].hist(control_ages, bins=20, alpha=0.6, label='Control', edgecolor='black')
axes[0, 1].axvline(treatment_ages.mean(), color='blue', linestyle='--', linewidth=2)
axes[0, 1].axvline(control_ages.mean(), color='orange', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Age Distribution: Stratified Randomization')
axes[0, 1].legend()

# Plot 3: Disease severity balance
severity_treatment = [balance_df[balance_df['method'] == 'stratified_rand'][f'{s}_treatment'].values[0] 
                     for s in ['Mild', 'Moderate', 'Severe']]
severity_control = [balance_df[balance_df['method'] == 'stratified_rand'][f'{s}_control'].values[0] 
                   for s in ['Mild', 'Moderate', 'Severe']]

x_sev = np.arange(3)
axes[0, 2].bar(x_sev - width/2, severity_treatment, width, label='Treatment', alpha=0.8)
axes[0, 2].bar(x_sev + width/2, severity_control, width, label='Control', alpha=0.8)
axes[0, 2].set_ylabel('Proportion')
axes[0, 2].set_title('Disease Severity Balance\n(Stratified Method)')
axes[0, 2].set_xticks(x_sev)
axes[0, 2].set_xticklabels(['Mild', 'Moderate', 'Severe'])
axes[0, 2].legend()
axes[0, 2].grid(axis='y', alpha=0.3)

# Plot 4: Sequential allocation (block randomization)
block_assignments = block_randomization(100, block_size=4)
cumulative_treatment = np.cumsum(block_assignments)
cumulative_ideal = np.arange(1, 101) * 0.5

axes[1, 0].plot(range(1, 101), cumulative_treatment, label='Actual', linewidth=2)
axes[1, 0].plot(range(1, 101), cumulative_ideal, 'r--', label='Perfect Balance', linewidth=2)
axes[1, 0].fill_between(range(1, 101), cumulative_ideal - 2, cumulative_ideal + 2, 
                        alpha=0.2, color='red', label='±2 patients')
axes[1, 0].set_xlabel('Patient Number')
axes[1, 0].set_ylabel('Cumulative Treatment Assignments')
axes[1, 0].set_title('Sequential Balance: Block Randomization')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Imbalance distribution comparison
axes[1, 1].hist(imbalance_simple, bins=30, alpha=0.6, label='Simple', density=True, edgecolor='black')
axes[1, 1].hist(imbalance_block, bins=30, alpha=0.6, label='Block', density=True, edgecolor='black')
axes[1, 1].axvline(imbalance_simple.mean(), color='blue', linestyle='--', linewidth=2)
axes[1, 1].axvline(imbalance_block.mean(), color='orange', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Absolute Imbalance (# patients)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title(f'Imbalance Distribution\n({n_simulations} simulations)')
axes[1, 1].legend()

# Plot 6: P-value distribution under null (should be uniform)
p_values = []
for _ in range(1000):
    # Generate data under null (no treatment effect)
    temp_patients = patients.copy()
    temp_patients['outcome'] = np.random.normal(100, 15, n_patients)
    temp_patients['assignment'] = simple_randomization(n_patients)
    
    # T-test
    t_group = temp_patients[temp_patients['assignment'] == 1]['outcome']
    c_group = temp_patients[temp_patients['assignment'] == 0]['outcome']
    _, p_val = stats.ttest_ind(t_group, c_group)
    p_values.append(p_val)

axes[1, 2].hist(p_values, bins=20, edgecolor='black', alpha=0.7)
axes[1, 2].axhline(50, color='r', linestyle='--', linewidth=2, label='Uniform expectation')
axes[1, 2].set_xlabel('P-value')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('P-value Distribution Under Null\n(Should be uniform)')
axes[1, 2].legend()

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is randomization the wrong tool?
- Observational studies: Cannot randomize, use propensity scores or matching
- Ethical prohibitions: Randomizing harmful exposures unethical
- Logistical impossibility: Policy interventions, large-scale programs
- Time-ordering required: Historical controls in before-after studies
- Extreme heterogeneity: Stratified or minimization needed, not simple randomization

## 7. Key References
- [Randomized Controlled Trial (Wikipedia)](https://en.wikipedia.org/wiki/Randomized_controlled_trial)
- [CONSORT Statement (RCT Reporting)](http://www.consort-statement.org/)
- [Altman & Bland, Treatment Allocation by Minimisation](https://www.bmj.com/content/330/7495/843)

---
**Status:** Gold standard for causal inference | **Complements:** Control Groups, Design of Experiments, Confounding Control
