# Instrumental Variables (IV)

## 1. Concept Skeleton
**Definition:** Estimation technique using external variable (instrument) to isolate exogenous variation when explanatory variable is endogenous  
**Purpose:** Obtain consistent estimates when Cov(X, ε) ≠ 0; address omitted variables, measurement error, simultaneity  
**Prerequisites:** OLS assumptions, endogeneity sources, covariance algebra, two-stage least squares

## 2. Comparative Framing
| Method | OLS | IV/2SLS | Fixed Effects | Matching |
|--------|-----|---------|---------------|----------|
| **Endogeneity** | Assumes none | Handles via instrument | Controls time-invariant unobs. | Balances observables |
| **Consistency** | Biased if endogenous | Consistent if valid IV | Consistent if strict exog. | Biased if unobs. confounders |
| **Efficiency** | Most efficient | Less efficient (larger SE) | Moderate | Non-parametric |
| **Requirement** | E[X'ε]=0 | Valid instrument Z | Panel data | Overlap assumption |

## 3. Examples + Counterexamples

**Classic Example:**  
Returns to education: Schooling endogenous (ability bias). Instrument: Quarter of birth (Angrist & Krueger) affects schooling via compulsory laws, uncorrelated with ability.

**Failure Case:**  
Instrument: Father's education for own education in wage equation. Fails exclusion if father's education directly affects child's wages through social networks.

**Edge Case:**  
Weak instrument: F-stat < 10 in first stage. IV estimates inconsistent, biased toward OLS, inflated standard errors.

## 4. Layer Breakdown
```
Instrumental Variables Framework:
├─ Structural Equation:
│   Y = β₀ + β₁X + ε,  Cov(X, ε) ≠ 0
├─ Instrument Requirements:
│   ├─ Relevance: Cov(Z, X) ≠ 0  (first stage)
│   │   └─ Test: F-statistic > 10 in first stage regression
│   └─ Exogeneity: Cov(Z, ε) = 0  (exclusion restriction)
│       └─ Not testable; requires economic reasoning
├─ Two-Stage Least Squares (2SLS):
│   ├─ First Stage: X = π₀ + π₁Z + ν
│   │   └─ Predicted: X̂ = π̂₀ + π̂₁Z
│   ├─ Second Stage: Y = β₀ + β₁X̂ + residual
│   │   └─ β̂₁ᴵⱽ = Cov(Z,Y) / Cov(Z,X)
│   └─ Standard Errors: Adjust for two-stage procedure
├─ Overidentification:
│   ├─ Just-identified: # instruments = # endogenous (unique estimate)
│   ├─ Over-identified: # instruments > # endogenous
│   │   └─ Test: Sargan/Hansen J-test for instrument validity
│   └─ Under-identified: # instruments < # endogenous (not estimable)
└─ Diagnostics:
    ├─ Weak Instruments: Stock-Yogo critical values
    ├─ Endogeneity Test: Hausman test (IV vs OLS)
    └─ Anderson-Rubin: Robust to weak instruments
```

**Interaction:** Instrument Z → Endogenous X → Outcome Y, with Z ⊥ ε ensuring identification

## 5. Mini-Project
Estimate returns to education with simulated endogeneity and IV:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.sandbox.regression.gmm import IV2SLS
from linearmodels.iv import IV2SLS as IV2SLS_robust
import statsmodels.api as sm

np.random.seed(42)
n = 1000

# Generate data with endogeneity
# True model: log(wage) = β₀ + β₁(education) + β₂(ability) + ε
# Problem: Ability unobserved, correlated with education

ability = np.random.normal(0, 1, n)  # Unobserved
instrument = np.random.normal(0, 1, n)  # Quarter of birth proxy
epsilon = np.random.normal(0, 0.5, n)

# Education depends on ability (endogeneity) and instrument
education = 12 + 0.5*ability + 0.8*instrument + np.random.normal(0, 1, n)

# Wage depends on education and ability (omitted variable bias)
true_return = 0.10  # True return to education
log_wage = 1.5 + true_return*education + 0.3*ability + epsilon

# Create DataFrame
df = pd.DataFrame({
    'log_wage': log_wage,
    'education': education,
    'instrument': instrument,
    'ability': ability  # In reality, unobserved
})

# ===== Naive OLS (biased due to omitted ability) =====
X_ols = sm.add_constant(df['education'])
ols_model = sm.OLS(df['log_wage'], X_ols).fit()

print("="*60)
print("NAIVE OLS (Biased - Ability Omitted)")
print("="*60)
print(f"Education Coefficient: {ols_model.params['education']:.4f}")
print(f"Standard Error: {ols_model.bse['education']:.4f}")
print(f"True Return: {true_return:.4f}")
print(f"Bias: {ols_model.params['education'] - true_return:.4f}")
print(f"R²: {ols_model.rsquared:.4f}\n")

# ===== First Stage: Check instrument relevance =====
first_stage = sm.OLS(df['education'], 
                     sm.add_constant(df['instrument'])).fit()

print("="*60)
print("FIRST STAGE (Instrument Relevance)")
print("="*60)
print(f"Instrument Coefficient: {first_stage.params['instrument']:.4f}")
print(f"F-statistic: {first_stage.fvalue:.2f}")
print(f"Rule of Thumb: F > 10 (Weak if below)")

if first_stage.fvalue < 10:
    print("⚠️  WARNING: Weak instrument detected!")
else:
    print("✓ Instrument appears strong")
print(f"R² (First Stage): {first_stage.rsquared:.4f}\n")

# ===== IV/2SLS Estimation =====
# Using linearmodels for robust standard errors
iv_formula = 'log_wage ~ 1 + [education ~ instrument]'
iv_model = IV2SLS_robust.from_formula(iv_formula, df).fit(cov_type='robust')

print("="*60)
print("IV/2SLS ESTIMATION")
print("="*60)
print(f"Education Coefficient (IV): {iv_model.params['education']:.4f}")
print(f"Robust Standard Error: {iv_model.std_errors['education']:.4f}")
print(f"95% CI: [{iv_model.params['education'] - 1.96*iv_model.std_errors['education']:.4f}, "
      f"{iv_model.params['education'] + 1.96*iv_model.std_errors['education']:.4f}]")
print(f"True Return: {true_return:.4f}")
print(f"Bias: {iv_model.params['education'] - true_return:.4f}")

# Compare standard errors
se_ratio = iv_model.std_errors['education'] / ols_model.bse['education']
print(f"\nIV SE / OLS SE: {se_ratio:.2f}x (IV less efficient)")

# ===== Oracle Estimation (if ability observed) =====
X_oracle = sm.add_constant(df[['education', 'ability']])
oracle_model = sm.OLS(df['log_wage'], X_oracle).fit()

print("\n" + "="*60)
print("ORACLE OLS (Ability Included - Unbiased)")
print("="*60)
print(f"Education Coefficient: {oracle_model.params['education']:.4f}")
print(f"Standard Error: {oracle_model.bse['education']:.4f}")
print(f"True Return: {true_return:.4f}\n")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: First Stage - Instrument vs Education
axes[0, 0].scatter(df['instrument'], df['education'], alpha=0.3, s=20)
x_range = np.array([df['instrument'].min(), df['instrument'].max()])
y_pred = first_stage.params['const'] + first_stage.params['instrument']*x_range
axes[0, 0].plot(x_range, y_pred, 'r-', linewidth=2, 
                label=f"Slope: {first_stage.params['instrument']:.3f}")
axes[0, 0].set_xlabel('Instrument (Z)')
axes[0, 0].set_ylabel('Education (X)')
axes[0, 0].set_title(f'First Stage: F-stat = {first_stage.fvalue:.1f}')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Reduced Form - Instrument vs Outcome
reduced_form = sm.OLS(df['log_wage'], 
                      sm.add_constant(df['instrument'])).fit()
axes[0, 1].scatter(df['instrument'], df['log_wage'], alpha=0.3, s=20)
y_pred_rf = reduced_form.params['const'] + reduced_form.params['instrument']*x_range
axes[0, 1].plot(x_range, y_pred_rf, 'g-', linewidth=2,
                label=f"Reduced Form")
axes[0, 1].set_xlabel('Instrument (Z)')
axes[0, 1].set_ylabel('Log Wage (Y)')
axes[0, 1].set_title('Reduced Form (Intent-to-Treat)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Compare Estimates
estimates = pd.DataFrame({
    'Method': ['OLS\n(Biased)', 'IV/2SLS', 'Oracle\n(Truth)', 'True Value'],
    'Coefficient': [ols_model.params['education'], 
                   iv_model.params['education'],
                   oracle_model.params['education'],
                   true_return],
    'SE': [ols_model.bse['education'], 
           iv_model.std_errors['education'],
           oracle_model.bse['education'],
           0]
})

x_pos = np.arange(len(estimates))
axes[1, 0].bar(x_pos[:3], estimates['Coefficient'][:3], 
               yerr=1.96*estimates['SE'][:3], capsize=5,
               color=['red', 'blue', 'green'], alpha=0.7)
axes[1, 0].axhline(true_return, color='black', linestyle='--', 
                   linewidth=2, label='True Value')
axes[1, 0].set_xticks(x_pos[:3])
axes[1, 0].set_xticklabels(estimates['Method'][:3])
axes[1, 0].set_ylabel('Education Coefficient')
axes[1, 0].set_title('Comparison of Estimates (95% CI)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Bias-Variance Tradeoff
axes[1, 1].text(0.5, 0.85, 'BIAS-VARIANCE TRADEOFF', 
                ha='center', fontsize=14, weight='bold', 
                transform=axes[1, 1].transAxes)

summary_text = f"""
OLS (Naive):
  • Bias: {ols_model.params['education'] - true_return:+.4f}
  • SE: {ols_model.bse['education']:.4f}
  • Consistent: No (omitted variable)

IV/2SLS:
  • Bias: {iv_model.params['education'] - true_return:+.4f}
  • SE: {iv_model.std_errors['education']:.4f}
  • Consistent: Yes (if valid instrument)
  • Efficiency Loss: {se_ratio:.1f}x

Oracle (ability included):
  • Bias: {oracle_model.params['education'] - true_return:+.4f}
  • SE: {oracle_model.bse['education']:.4f}
  • Best case (unobservables known)
"""

axes[1, 1].text(0.05, 0.7, summary_text, 
                transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# ===== Endogeneity Test (Hausman) =====
print("="*60)
print("ENDOGENEITY TEST")
print("="*60)

# Manual Hausman test
# Add first-stage residuals to OLS
first_stage_resid = df['education'] - first_stage.predict(sm.add_constant(df['instrument']))
X_hausman = sm.add_constant(df[['education']].assign(resid=first_stage_resid))
hausman_model = sm.OLS(df['log_wage'], X_hausman).fit()

print(f"First-stage residual coefficient: {hausman_model.params['resid']:.4f}")
print(f"p-value: {hausman_model.pvalues['resid']:.4f}")
if hausman_model.pvalues['resid'] < 0.05:
    print("✓ Reject H₀: Education is endogenous (use IV)")
else:
    print("  Fail to reject: Education exogenous (OLS valid)")
```

## 6. Challenge Round
When does IV fail or mislead?
- **Weak instruments** (F < 10): Finite sample bias toward OLS, unreliable inference
- **Exclusion violation**: Instrument affects outcome directly, not just through endogenous variable
- **LATE interpretation**: IV estimates Local Average Treatment Effect (compliers only), not ATE
- **Heterogeneous effects**: IV captures effect for subpopulation moved by instrument, may differ from population
- **Monotonicity violation**: Instrument has opposite effect on some units (defiers exist)

## 7. Key References
- [Angrist & Pischke - Mostly Harmless Econometrics (Ch 4)](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)
- [Stock & Watson - Introduction to Econometrics (Ch 12)](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000005522)
- [Angrist & Krueger (1991) - Does Compulsory School Attendance Affect Schooling and Earnings?](https://www.jstor.org/stable/2937954)

---
**Status:** Core causal inference method | **Complements:** 2SLS, GMM, Difference-in-Differences, RDD
