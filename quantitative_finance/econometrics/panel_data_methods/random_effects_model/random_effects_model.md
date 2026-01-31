# Random Effects Model

## 1. Concept Skeleton
**Definition:** Panel data estimator treating unobserved heterogeneity as random component uncorrelated with regressors; GLS estimation  
**Purpose:** Efficient estimation when individual effects uncorrelated with X; estimate time-invariant variables; between and within variation  
**Prerequisites:** Panel data structure, random effects assumption, GLS, variance components, Hausman test

## 2. Comparative Framing
| Method | Random Effects (RE) | Fixed Effects (FE) | Between Effects | Pooled OLS |
|--------|---------------------|--------------------|-----------------|-----------| 
| **αᵢ Assumption** | Uncorrelated with X | Correlated with X allowed | Between only | αᵢ = 0 |
| **Consistency** | If Cov(αᵢ,X)=0 | Always (strict exog.) | If Cov(αᵢ,X)=0 | If αᵢ = 0 |
| **Efficiency** | Most efficient if valid | Less efficient | Least efficient | Most if no αᵢ |
| **Time-Invariant X** | Can estimate | Cannot estimate | Can estimate | Can estimate |

## 3. Examples + Counterexamples

**Classic Example:**  
Hospital quality study: Random sample of hospitals, quality (αᵢ) plausibly uncorrelated with observed inputs. RE efficient, can estimate hospital size (time-invariant).

**Failure Case:**  
Wage regression: Ability (αᵢ) correlated with education. RE inconsistent, Hausman test rejects. Use FE instead.

**Edge Case:**  
Clustered data structure: Students within schools. Two-level random effects (student + school) needed, not single-level RE.

## 4. Layer Breakdown
```
Random Effects Model Framework:
├─ Model Specification:
│   Yᵢₜ = Xᵢₜ'β + αᵢ + εᵢₜ
│   ├─ αᵢ ~ IID(0, σ²_α): Random individual effect
│   ├─ εᵢₜ ~ IID(0, σ²_ε): Idiosyncratic error
│   ├─ Composite Error: uᵢₜ = αᵢ + εᵢₜ
│   └─ Key Assumption: Cov(αᵢ, Xᵢₜ) = 0  (orthogonality)
├─ Variance Structure:
│   ├─ Var(uᵢₜ) = σ²_α + σ²_ε
│   ├─ Cov(uᵢₜ, uᵢₛ) = σ²_α  (t ≠ s, same unit)
│   │   └─ Equicorrelation within units
│   ├─ Corr(uᵢₜ, uᵢₛ) = ρ = σ²_α / (σ²_α + σ²_ε)
│   │   └─ Intraclass correlation coefficient
│   └─ Cov(uᵢₜ, uⱼₛ) = 0  (i ≠ j, different units)
├─ GLS Estimation:
│   ├─ Quasi-Demeaning Transformation:
│   │   Ỹᵢₜ = Yᵢₜ - θȲᵢ,  X̃ᵢₜ = Xᵢₜ - θX̄ᵢ
│   │   └─ θ = 1 - √[σ²_ε / (σ²_ε + Tσ²_α)]
│   ├─ θ Interpretation:
│   │   ├─ θ = 0 → RE = Between estimator (all between variation)
│   │   ├─ θ = 1 → RE = FE estimator (all within variation)
│   │   └─ 0 < θ < 1 → Weighted average of within and between
│   ├─ Variance Components:
│   │   ├─ σ²_ε: From FE residuals or within variation
│   │   └─ σ²_α: From between variation minus σ²_ε/T
│   └─ Efficiency: RE more efficient than FE when assumption valid
├─ Estimation Methods:
│   ├─ Feasible GLS (FGLS): Estimate variance components, then GLS
│   ├─ Maximum Likelihood: Joint estimation of β, σ²_α, σ²_ε
│   └─ Swamy-Arora: Common variance component estimator
├─ Standard Errors:
│   ├─ Default: GLS standard errors (assumes correct specification)
│   ├─ Robust: Heteroskedasticity-robust
│   └─ Clustered: By entity for additional robustness
├─ Hausman Test:
│   ├─ H₀: Cov(αᵢ, Xᵢₜ) = 0 (RE consistent and efficient)
│   ├─ H₁: Cov(αᵢ, Xᵢₜ) ≠ 0 (RE inconsistent, use FE)
│   ├─ Test Statistic: H = (β̂_FE - β̂_RE)'[Var(β̂_FE) - Var(β̂_RE)]⁻¹(β̂_FE - β̂_RE)
│   │   └─ H ~ χ²(k) under H₀
│   └─ Reject H₀ → Use FE; Fail to reject → Use RE
├─ Between Effects Estimator:
│   ├─ Regression: Ȳᵢ = X̄ᵢ'β + αᵢ + ε̄ᵢ
│   ├─ Uses only cross-sectional variation
│   └─ Inefficient but simple alternative
├─ Extensions:
│   ├─ Correlated RE: Include X̄ᵢ as regressors (Mundlak)
│   ├─ Two-Way RE: Random unit + time effects
│   ├─ Hierarchical/Multilevel: Nested random effects
│   └─ Random Coefficients: βᵢ ~ Distribution
└─ Diagnostics:
    ├─ Breusch-Pagan LM Test: H₀: σ²_α = 0 (no RE needed)
    ├─ Intraclass Correlation: ρ̂ magnitude and interpretation
    ├─ Residual Patterns: Normality, heteroskedasticity
    └─ Hausman Specification: RE vs FE comparison
```

**Interaction:** Estimate variance components → Quasi-demean with θ → GLS → Hausman test for validity

## 5. Mini-Project
Estimate random effects, compare with fixed effects, and perform Hausman test:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, BetweenOLS
from linearmodels.panel import compare
from scipy import stats

np.random.seed(123)

# ===== Simulate Panel Data with Random Effects =====
N = 150  # Number of units
T = 8    # Time periods
n_obs = N * T

# Random individual effects (UNCORRELATED with X)
alpha_i = np.random.normal(0, 3, N)

# Generate data where RE assumption holds
data_list = []
for i in range(N):
    for t in range(T):
        # X1, X2 exogenous (uncorrelated with alpha_i)
        x1 = 10 + np.random.normal(0, 2)
        x2 = 5 + 0.2 * t + np.random.normal(0, 1.5)
        
        # X3: Time-invariant variable (school size, gender, etc.)
        x3 = 7 + np.random.normal(0, 1)  # Same for all t within unit i
        
        # Outcome
        # True effects: β1=2.0, β2=1.5, β3=0.8
        epsilon = np.random.normal(0, 2)
        y = 10 + 2.0*x1 + 1.5*x2 + 0.8*x3 + alpha_i[i] + epsilon
        
        data_list.append({
            'id': i,
            'time': t,
            'y': y,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'alpha_true': alpha_i[i]
        })

df = pd.DataFrame(data_list)

# Make x3 truly time-invariant per unit
df['x3'] = df.groupby('id')['x3'].transform('first')

print("="*70)
print("RANDOM EFFECTS MODEL: PANEL DATA ANALYSIS")
print("="*70)
print(f"\nPanel Dimensions:")
print(f"  Units (N): {N}")
print(f"  Time Periods (T): {T}")
print(f"  Total Observations: {n_obs}")
print(f"\nTrue Coefficients:")
print(f"  β₁ (X1): 2.0")
print(f"  β₂ (X2): 1.5")
print(f"  β₃ (X3, time-invariant): 0.8")
print(f"\nDescriptive Statistics:")
print(df[['y', 'x1', 'x2', 'x3']].describe().round(3))

# Set multi-index
df_panel = df.set_index(['id', 'time'])

# ===== Pooled OLS =====
print("\n" + "="*70)
print("POOLED OLS")
print("="*70)

exog_vars = ['x1', 'x2', 'x3']
pooled_model = PooledOLS(df_panel['y'], df_panel[exog_vars]).fit()
print(pooled_model.summary)

# ===== Fixed Effects =====
print("\n" + "="*70)
print("FIXED EFFECTS MODEL")
print("="*70)

fe_model = PanelOLS(df_panel['y'], df_panel[['x1', 'x2']],  # Cannot include x3
                    entity_effects=True).fit(cov_type='clustered',
                                             cluster_entity=True)
print(fe_model.summary)

print("\nNote: X3 (time-invariant) dropped from FE model")

# ===== Random Effects =====
print("\n" + "="*70)
print("RANDOM EFFECTS MODEL")
print("="*70)

re_model = RandomEffects(df_panel['y'], df_panel[exog_vars]).fit(
    cov_type='clustered', cluster_entity=True)
print(re_model.summary)

# ===== Between Effects =====
print("\n" + "="*70)
print("BETWEEN EFFECTS ESTIMATOR")
print("="*70)

be_model = BetweenOLS(df_panel['y'], df_panel[exog_vars]).fit()
print(be_model.summary)

# ===== Variance Components =====
print("\n" + "="*70)
print("VARIANCE COMPONENTS")
print("="*70)

# Extract variance components from RE model
sigma_eps = np.sqrt(re_model.variance_decomposition.loc['Effects']['Var'])
sigma_alpha = np.sqrt(re_model.variance_decomposition.loc['Residual']['Var'])

# Calculate rho (intraclass correlation)
total_var = sigma_alpha**2 + sigma_eps**2
rho = sigma_alpha**2 / total_var

# Calculate theta (quasi-demeaning parameter)
theta = 1 - np.sqrt(sigma_eps**2 / (sigma_eps**2 + T * sigma_alpha**2))

print(f"\nVariance Decomposition:")
print(f"  σ²_ε (idiosyncratic):    {sigma_eps**2:.4f}")
print(f"  σ²_α (individual effect): {sigma_alpha**2:.4f}")
print(f"  Total variance:           {total_var:.4f}")

print(f"\nIntraclass Correlation (ρ):")
print(f"  ρ = σ²_α / (σ²_α + σ²_ε) = {rho:.4f}")
print(f"  Interpretation: {rho*100:.1f}% of total variance due to individual effects")

print(f"\nQuasi-Demeaning Parameter (θ):")
print(f"  θ = {theta:.4f}")
print(f"  0 → Between estimator, 1 → FE estimator")
print(f"  Current: {'Closer to FE' if theta > 0.5 else 'Closer to Between'}")

# ===== Hausman Test =====
print("\n" + "="*70)
print("HAUSMAN SPECIFICATION TEST")
print("="*70)

# Manual Hausman test (for variables in both models)
fe_coefs = fe_model.params[['x1', 'x2']].values
re_coefs = re_model.params[['x1', 'x2']].values

coef_diff = fe_coefs - re_coefs

# Variance difference
var_fe = fe_model.cov[['x1', 'x2']].loc[['x1', 'x2']].values
var_re = re_model.cov[['x1', 'x2']].loc[['x1', 'x2']].values
var_diff = var_fe - var_re

# Hausman statistic
try:
    hausman_stat = coef_diff.T @ np.linalg.inv(var_diff) @ coef_diff
    hausman_pval = 1 - stats.chi2.cdf(hausman_stat, df=2)
    
    print(f"H₀: Cov(αᵢ, Xᵢₜ) = 0 (RE is consistent and efficient)")
    print(f"H₁: Cov(αᵢ, Xᵢₜ) ≠ 0 (RE is inconsistent, use FE)")
    print(f"\nHausman Test Statistic: χ²(2) = {hausman_stat:.4f}")
    print(f"p-value: {hausman_pval:.4f}")
    
    if hausman_pval < 0.05:
        print("\n✗ Reject H₀: Use FIXED EFFECTS")
        print("   Individual effects correlated with regressors")
    else:
        print("\n✓ Fail to reject H₀: Use RANDOM EFFECTS")
        print("   RE is consistent and more efficient")
except:
    print("Hausman test computation issue (variance matrix not positive definite)")

# ===== Breusch-Pagan LM Test =====
print("\n" + "="*70)
print("BREUSCH-PAGAN LM TEST FOR RANDOM EFFECTS")
print("="*70)

# Test H0: sigma_alpha^2 = 0 (no random effects needed)
pooled_resid = pooled_model.resid
n = len(df_panel)

# Group residuals by individual
resid_by_id = []
for i in range(N):
    resid_i = pooled_resid[df_panel.index.get_level_values('id') == i]
    resid_by_id.append(resid_i.values)

# LM statistic
sum_squares = sum([np.sum(r)**2 for r in resid_by_id])
total_ss = np.sum(pooled_resid**2)

lm_stat = (n / (2 * (T - 1))) * ((sum_squares / total_ss) - 1)**2
lm_pval = 1 - stats.chi2.cdf(lm_stat, df=1)

print(f"H₀: σ²_α = 0 (no random effects, use Pooled OLS)")
print(f"H₁: σ²_α > 0 (random effects present)")
print(f"\nLM Statistic: χ²(1) = {lm_stat:.4f}")
print(f"p-value: {lm_pval:.4f}")

if lm_pval < 0.05:
    print("\n✓ Reject H₀: Random effects are present")
    print("   Use RE or FE, not Pooled OLS")
else:
    print("\n  Fail to reject: No evidence of random effects")

# ===== Model Comparison =====
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Pooled OLS': pooled_model.params,
    'Between': be_model.params,
    'Random Effects': re_model.params,
    'Fixed Effects': [fe_model.params.get('x1', np.nan),
                     fe_model.params.get('x2', np.nan),
                     np.nan]  # FE can't estimate x3
}, index=['x1', 'x2', 'x3'])

print("\nCoefficient Estimates:")
print(comparison_df.round(4))

print("\nTrue Coefficients: β₁=2.0, β₂=1.5, β₃=0.8")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Coefficient Comparison
models = ['True', 'Pooled', 'Between', 'RE', 'FE']
x1_coefs = [2.0, pooled_model.params['x1'], be_model.params['x1'], 
            re_model.params['x1'], fe_model.params['x1']]
x2_coefs = [1.5, pooled_model.params['x2'], be_model.params['x2'],
            re_model.params['x2'], fe_model.params['x2']]

x_pos = np.arange(len(models))
width = 0.35

axes[0, 0].bar(x_pos - width/2, x1_coefs, width, label='X1', alpha=0.8)
axes[0, 0].bar(x_pos + width/2, x2_coefs, width, label='X2', alpha=0.8)
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(models, rotation=45)
axes[0, 0].set_ylabel('Coefficient')
axes[0, 0].set_title('Coefficient Estimates Across Models')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Standard Errors Comparison
se_comparison = pd.DataFrame({
    'RE': re_model.std_errors[['x1', 'x2']].values,
    'FE': fe_model.std_errors[['x1', 'x2']].values,
    'Pooled': pooled_model.std_errors[['x1', 'x2']].values
}, index=['x1', 'x2'])

se_comparison.T.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
axes[0, 1].set_ylabel('Standard Error')
axes[0, 1].set_title('Standard Errors: RE vs FE vs Pooled')
axes[0, 1].legend(title='Variable')
axes[0, 1].grid(alpha=0.3, axis='y')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

# Plot 3: Within vs Between Variation
df_within = df.copy()
df_within['y_within'] = df_within.groupby('id')['y'].transform(lambda x: x - x.mean())
df_within['x1_within'] = df_within.groupby('id')['x1'].transform(lambda x: x - x.mean())

df_between = df.groupby('id')[['y', 'x1']].mean()

axes[0, 2].scatter(df_within['x1_within'], df_within['y_within'],
                   alpha=0.2, s=5, label='Within', color='blue')
axes[0, 2].scatter(df_between['x1'], df_between['y'],
                   alpha=0.6, s=40, label='Between', color='red')
axes[0, 2].set_xlabel('X1')
axes[0, 2].set_ylabel('Y')
axes[0, 2].set_title(f'Within vs Between Variation (ρ={rho:.3f})')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Residual Comparison
axes[1, 0].hist(pooled_model.resid, bins=40, alpha=0.5, 
               label='Pooled', density=True)
axes[1, 0].hist(re_model.resid, bins=40, alpha=0.5,
               label='RE', density=True)
axes[1, 0].hist(fe_model.resid, bins=40, alpha=0.5,
               label='FE', density=True)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Residual Distributions')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 5: Time-Invariant Variable (X3)
# Only Pooled, Between, and RE can estimate this
x3_estimates = {
    'True': 0.8,
    'Pooled': pooled_model.params['x3'],
    'Between': be_model.params['x3'],
    'RE': re_model.params['x3']
}

x3_se = {
    'Pooled': pooled_model.std_errors['x3'],
    'Between': be_model.std_errors['x3'],
    'RE': re_model.std_errors['x3']
}

models_x3 = list(x3_estimates.keys())
x3_vals = list(x3_estimates.values())
x3_errors = [0, x3_se['Pooled'], x3_se['Between'], x3_se['RE']]

axes[1, 1].bar(models_x3, x3_vals, alpha=0.8, color=['gray', 'orange', 'green', 'blue'])
axes[1, 1].errorbar(models_x3[1:], x3_vals[1:], yerr=[1.96*e for e in x3_errors[1:]],
                   fmt='none', color='black', capsize=5)
axes[1, 1].axhline(0.8, color='red', linestyle='--', linewidth=2, label='True')
axes[1, 1].set_ylabel('X3 Coefficient')
axes[1, 1].set_title('Time-Invariant Variable (X3)\n(FE Cannot Estimate)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

# Plot 6: Variance Decomposition
var_decomp_data = {
    'Within (idiosyncratic)': sigma_eps**2,
    'Between (individual)': sigma_alpha**2
}

axes[1, 2].pie(var_decomp_data.values(), labels=var_decomp_data.keys(),
              autopct='%1.1f%%', startangle=90)
axes[1, 2].set_title(f'Variance Decomposition\n(ρ = {rho:.3f})')

plt.tight_layout()
plt.show()

# ===== Summary =====
print("\n" + "="*70)
print("INTERPRETATION AND RECOMMENDATIONS")
print("="*70)

print("\n1. Variance Components:")
print(f"   - {rho*100:.1f}% of variance is between individuals")
print(f"   - {(1-rho)*100:.1f}% of variance is within individuals over time")

print("\n2. Model Choice:")
if hausman_pval >= 0.05:
    print("   ✓ Hausman test: Use RANDOM EFFECTS")
    print("     • RE is consistent and more efficient than FE")
    print("     • Can estimate time-invariant variables")
    print(f"     • X3 coefficient: {re_model.params['x3']:.4f} (True: 0.8)")
else:
    print("   ✗ Hausman test: Use FIXED EFFECTS")
    print("     • RE is inconsistent (correlated with X)")

print("\n3. Advantages of RE (when valid):")
print("   • More efficient (smaller standard errors)")
print("   • Can estimate time-invariant variables")
print("   • Uses both within and between variation")

print("\n4. RE Assumptions:")
print("   • Cov(αᵢ, Xᵢₜ) = 0 (crucial!)")
print("   • Random sample of units")
print("   • Homoskedasticity and no serial correlation (or use robust SE)")

print(f"\n5. Quasi-Demeaning (θ = {theta:.3f}):")
if theta < 0.3:
    print("   • RE close to between estimator")
elif theta > 0.7:
    print("   • RE close to fixed effects")
else:
    print("   • RE balanced between within and between")
```

## 6. Challenge Round
When does random effects fail or need reconsideration?
- **Correlation with regressors**: If Cov(αᵢ, X) ≠ 0, RE inconsistent → Use FE or test with Hausman
- **Non-random sample**: Selected units (e.g., top firms) violate random sampling → FE more appropriate
- **Hierarchical structure**: Nested levels (students in schools) → Use multilevel/hierarchical models
- **Time-varying effects**: Random slopes, not just intercepts → Random coefficients model
- **Small N, large T**: RE gains diminish as T increases → FE and RE converge
- **Unbalanced panels**: Complicated variance structure → Use ML estimation, not FGLS

## 7. Key References
- [Wooldridge - Econometric Analysis of Cross Section and Panel Data (Ch 10)](https://mitpress.mit.edu/9780262232586/)
- [Baltagi - Econometric Analysis of Panel Data (6th ed)](https://www.springer.com/gp/book/9783030533557)
- [Hausman (1978) - Specification Tests in Econometrics](https://doi.org/10.2307/1913827)

---
**Status:** Core panel data method | **Complements:** Fixed Effects, Hausman Test, GLS, Multilevel Models
