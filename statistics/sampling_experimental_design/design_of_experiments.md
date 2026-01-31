# Design of Experiments (DOE)

## 1. Concept Skeleton
**Definition:** Systematic planning and structure of experiments to maximize information while controlling variability and confounding  
**Purpose:** Establish causality, test multiple factors efficiently, quantify interactions, optimize processes with minimal trials  
**Prerequisites:** Hypothesis testing, ANOVA, randomization concepts, factorial notation, blocking principles

## 2. Comparative Framing
| Design | Completely Randomized | Randomized Block | Factorial | Latin Square |
|--------|----------------------|------------------|-----------|--------------|
| **Control Method** | Randomization only | Blocking + randomization | Systematic factor combinations | Row & column blocking |
| **Factors Tested** | One primary | One primary + nuisance | Multiple simultaneously | Two factors + two blocks |
| **Efficiency** | Baseline | Higher (reduces error) | Very high (interactions) | High (two-way blocking) |
| **Complexity** | Low | Moderate | High (exponential trials) | Moderate |

## 3. Examples + Counterexamples

**Simple Example:**  
Test 3 fertilizers on crop yield: Randomized block design groups plots by soil type, tests all fertilizers within each block

**Failure Case:**  
One-factor-at-a-time (OFAT): Test fertilizer amounts separately from watering → miss interaction where high fertilizer needs more water

**Edge Case:**  
Full factorial with 10 factors, 3 levels each: 3¹⁰ = 59,049 runs infeasible → use fractional factorial or screening design

## 4. Layer Breakdown
```
DOE Framework:
├─ Basic Principles:
│   ├─ Replication: Repeat treatments to estimate error
│   ├─ Randomization: Eliminate bias, validate inference
│   └─ Blocking: Group similar units, reduce variability
├─ Design Types:
│   ├─ Completely Randomized Design (CRD):
│   │   ├─ Random assignment to treatments
│   │   ├─ Analysis: One-way ANOVA
│   │   └─ Use: Homogeneous experimental units
│   ├─ Randomized Complete Block Design (RCBD):
│   │   ├─ Block by nuisance factor (soil, time, batch)
│   │   ├─ Randomize treatments within blocks
│   │   ├─ Analysis: Two-way ANOVA without interaction
│   │   └─ Increases precision vs CRD
│   ├─ Factorial Designs:
│   │   ├─ Test all factor combinations
│   │   ├─ 2^k design: k factors, 2 levels each
│   │   ├─ 3^k design: k factors, 3 levels each
│   │   ├─ Main effects: Average effect of factor
│   │   ├─ Interactions: Effect depends on other factors
│   │   └─ Analysis: Multi-way ANOVA
│   ├─ Fractional Factorial:
│   │   ├─ Test subset of combinations
│   │   ├─ Confound high-order interactions
│   │   └─ Efficient screening (2^(k-p) designs)
│   ├─ Latin Square:
│   │   ├─ Block on two nuisance factors
│   │   ├─ Each treatment once per row/column
│   │   └─ Example: Agricultural field trials
│   ├─ Crossover Design:
│   │   ├─ Each subject receives all treatments
│   │   ├─ Sequence randomized
│   │   ├─ Washout period between treatments
│   │   └─ Analyzes within-subject differences
│   └─ Response Surface Methodology:
│       ├─ Optimize factor levels
│       └─ Central composite designs
├─ Sample Size:
│   └─ Power analysis determines replicates needed
├─ Covariate Adjustment:
│   └─ ANCOVA: Control for continuous confounders
└─ Analysis:
    ├─ Check assumptions (normality, homogeneity)
    ├─ ANOVA F-tests for main effects and interactions
    ├─ Post-hoc comparisons (Tukey, Bonferroni)
    └─ Effect size estimation
```

**Interaction:** Plan factors → Choose design → Randomize → Collect data → Analyze → Interpret effects

## 5. Mini-Project
Factorial experiment with interaction visualization:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from itertools import product
import seaborn as sns

# 2x3 Factorial Design: Temperature (2 levels) x Catalyst (3 types)
# Measuring chemical reaction yield

np.random.seed(42)

# Design factors
temperatures = [50, 80]  # Celsius
catalysts = ['A', 'B', 'C']
replicates = 5

# Generate experimental data with interaction
data = []
for temp, cat, rep in product(temperatures, catalysts, range(replicates)):
    # Baseline yield
    yield_base = 60
    
    # Main effects
    temp_effect = 10 if temp == 80 else 0
    cat_effect = {'A': 0, 'B': 5, 'C': 3}[cat]
    
    # Interaction: Catalyst C works better at high temp
    interaction = 8 if (temp == 80 and cat == 'C') else 0
    
    # Random error
    error = np.random.normal(0, 3)
    
    yield_value = yield_base + temp_effect + cat_effect + interaction + error
    
    data.append({
        'temperature': temp,
        'catalyst': cat,
        'replicate': rep,
        'yield': yield_value
    })

df = pd.DataFrame(data)

print("Experimental Design: 2×3 Factorial")
print(f"Factors: Temperature (2 levels), Catalyst (3 types)")
print(f"Replicates per combination: {replicates}")
print(f"Total runs: {len(df)}")

# Summary statistics by factor combination
summary = df.groupby(['temperature', 'catalyst'])['yield'].agg(['mean', 'std', 'count'])
print("\nMean Yield by Treatment Combination:")
print(summary.round(2))

# Two-way ANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Fit model with interaction
model = ols('yield ~ C(temperature) + C(catalyst) + C(temperature):C(catalyst)', data=df).fit()
anova_table = anova_lm(model, typ=2)

print("\nTwo-Way ANOVA Table:")
print(anova_table.round(4))

# Effect sizes (eta-squared)
anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
print("\nEffect Sizes (η²):")
print(anova_table['eta_sq'].round(3))

# Post-hoc tests for catalyst
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_catalyst = pairwise_tukeyhsd(df['yield'], df['catalyst'], alpha=0.05)
print("\nTukey HSD Post-hoc Test (Catalyst):")
print(tukey_catalyst)

# Interaction means
interaction_means = df.pivot_table(values='yield', 
                                   index='temperature', 
                                   columns='catalyst', 
                                   aggfunc='mean')
print("\nInteraction Means:")
print(interaction_means.round(2))

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Main effect - Temperature
temp_means = df.groupby('temperature')['yield'].mean()
axes[0, 0].bar([str(t) for t in temperatures], temp_means, alpha=0.7, color='steelblue')
axes[0, 0].set_xlabel('Temperature (°C)')
axes[0, 0].set_ylabel('Mean Yield')
axes[0, 0].set_title('Main Effect: Temperature')
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Main effect - Catalyst
cat_means = df.groupby('catalyst')['yield'].mean()
axes[0, 1].bar(catalysts, cat_means, alpha=0.7, color='coral')
axes[0, 1].set_xlabel('Catalyst Type')
axes[0, 1].set_ylabel('Mean Yield')
axes[0, 1].set_title('Main Effect: Catalyst')
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Interaction plot
for cat in catalysts:
    cat_data = df[df['catalyst'] == cat].groupby('temperature')['yield'].mean()
    axes[0, 2].plot(cat_data.index, cat_data.values, 'o-', 
                   linewidth=2, markersize=8, label=f'Catalyst {cat}')

axes[0, 2].set_xlabel('Temperature (°C)')
axes[0, 2].set_ylabel('Mean Yield')
axes[0, 2].set_title('Interaction Plot\n(Non-parallel = interaction)')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Box plots by combination
df['treatment'] = df['temperature'].astype(str) + '°C-' + df['catalyst']
treatment_order = [f'{t}°C-{c}' for t in temperatures for c in catalysts]
sns.boxplot(data=df, x='treatment', y='yield', order=treatment_order, ax=axes[1, 0])
axes[1, 0].set_xlabel('Treatment Combination')
axes[1, 0].set_ylabel('Yield')
axes[1, 0].set_title('Yield Distribution by Treatment')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 5: Residual plot
residuals = model.resid
fitted = model.fittedvalues
axes[1, 1].scatter(fitted, residuals, alpha=0.6)
axes[1, 1].axhline(0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Fitted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residual Plot (Check Assumptions)')
axes[1, 1].grid(alpha=0.3)

# Plot 6: Q-Q plot for normality
stats.probplot(residuals, dist="norm", plot=axes[1, 2])
axes[1, 2].set_title('Q-Q Plot (Normality Check)')

plt.tight_layout()
plt.show()

# Example: Randomized Block Design
print("\n" + "="*60)
print("Example 2: Randomized Complete Block Design (RCBD)")
print("="*60)

# Simulate agricultural trial: 4 fertilizers tested in 6 fields (blocks)
np.random.seed(43)
fertilizers = ['Control', 'NPK-10', 'NPK-20', 'Organic']
fields = ['Field_1', 'Field_2', 'Field_3', 'Field_4', 'Field_5', 'Field_6']

rcbd_data = []
for field in fields:
    # Field-specific baseline (blocking factor)
    field_baseline = np.random.normal(100, 15)
    
    for fert in fertilizers:
        # Treatment effect
        fert_effect = {'Control': 0, 'NPK-10': 12, 'NPK-20': 18, 'Organic': 10}[fert]
        
        # Random error
        error = np.random.normal(0, 5)
        
        yield_val = field_baseline + fert_effect + error
        
        rcbd_data.append({
            'field': field,
            'fertilizer': fert,
            'yield': yield_val
        })

rcbd_df = pd.DataFrame(rcbd_data)

# RCBD Analysis
rcbd_model = ols('yield ~ C(field) + C(fertilizer)', data=rcbd_df).fit()
rcbd_anova = anova_lm(rcbd_model, typ=2)

print("\nRCBD ANOVA Table:")
print(rcbd_anova.round(4))

# Efficiency: Compare to CRD
crd_model = ols('yield ~ C(fertilizer)', data=rcbd_df).fit()
mse_crd = np.sum(crd_model.resid**2) / crd_model.df_resid
mse_rcbd = np.sum(rcbd_model.resid**2) / rcbd_model.df_resid

efficiency = (mse_crd / mse_rcbd) * 100
print(f"\nBlocking Efficiency:")
print(f"MSE (CRD): {mse_crd:.2f}")
print(f"MSE (RCBD): {mse_rcbd:.2f}")
print(f"Relative Efficiency: {efficiency:.1f}% (RCBD reduces error variance)")
```

## 6. Challenge Round
When is DOE the wrong tool?
- Observational data only: Cannot manipulate factors, use regression or matching
- Single factor of interest: Simple t-test or one-way ANOVA sufficient
- Exploratory research: Use qualitative methods before structured experiments
- Ethical constraints: Cannot randomize treatment (medical contexts)
- Real-world complexity: Too many uncontrolled factors, consider adaptive designs

## 7. Key References
- [Design of Experiments Overview (Wikipedia)](https://en.wikipedia.org/wiki/Design_of_experiments)
- [Montgomery, Design and Analysis of Experiments (Book)](https://www.wiley.com/en-us/Design+and+Analysis+of+Experiments%2C+10th+Edition-p-9781119492443)
- [NIST Engineering Statistics Handbook - DOE](https://www.itl.nist.gov/div898/handbook/pri/pri.htm)

---
**Status:** Core experimental methodology | **Complements:** ANOVA, Randomization, Hypothesis Testing
