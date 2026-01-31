# Regression Discontinuity Design (RDD)

## 1. Concept Skeleton
**Definition:** Quasi-experimental method exploiting discontinuous treatment assignment rule at threshold to estimate causal effects  
**Purpose:** Identify local treatment effect for units near cutoff where treatment assignment is as-good-as-random  
**Prerequisites:** Running variable, assignment threshold, local randomization assumption, continuity of potential outcomes

## 2. Comparative Framing
| Method | Sharp RDD | Fuzzy RDD | DiD | IV |
|--------|-----------|-----------|-----|-----|
| **Assignment** | Deterministic at cutoff | Probabilistic jump | Before/after comparison | Instrument-based |
| **Estimand** | ATE at cutoff | LATE at cutoff (compliers) | ATT (parallel trends) | LATE (movers) |
| **External Validity** | Local (cutoff only) | Local (cutoff only) | Context-specific | Subpopulation |
| **Key Threat** | Manipulation of running variable | Weak first stage | Differential trends | Weak/invalid instrument |

## 3. Examples + Counterexamples

**Classic Example:**  
Maimonides' Rule: Class size capped at 40 students. Schools with 41 students split into 2 classes (~20 each). Compare outcomes just above/below multiples of 40.

**Failure Case:**  
Test score threshold for scholarship. Students retake tests to cross cutoff (manipulation), invalidating as-good-as-random assumption.

**Edge Case:**  
Threshold at age 65 for Medicare eligibility. Bandwidth choice critical: Too narrow → imprecise; too wide → confounding from age trends.

## 4. Layer Breakdown
```
Regression Discontinuity Design:
├─ Components:
│   ├─ Running Variable: X (continuous, determines treatment assignment)
│   ├─ Cutoff: c (threshold value)
│   ├─ Treatment: D = 1[X ≥ c] (sharp) or P(D=1) jumps at c (fuzzy)
│   └─ Outcome: Y (measured pre/post cutoff)
├─ Sharp RDD:
│   ├─ Assignment: D_i = 1 if X_i ≥ c, else 0 (deterministic)
│   ├─ Estimand: τ = lim[x↓c] E[Y|X=x] - lim[x↑c] E[Y|X=x]
│   │   └─ Jump in outcome at cutoff = causal effect
│   ├─ Regression: Y = α + τ·D + f(X-c) + ε
│   │   ├─ f(·): Flexible function of running variable (polynomial, local linear)
│   │   └─ Separate slopes: Allow f_left(X-c) ≠ f_right(X-c)
│   └─ Bandwidth Selection: Optimal h balances bias-variance tradeoff
├─ Fuzzy RDD:
│   ├─ Assignment: P(D=1|X) jumps at c but not 0→1
│   ├─ First Stage: E[D|X=x] discontinuous at c
│   ├─ Reduced Form: E[Y|X=x] discontinuous at c
│   └─ Wald Estimator: τ^fuzzy = (lim↓ Y - lim↑ Y) / (lim↓ D - lim↑ D)
│       └─ Ratio of outcome jump to treatment jump (IV-like)
├─ Assumptions:
│   ├─ Continuity: E[Y₀|X=x] and E[Y₁|X=x] continuous at c
│   │   └─ No confounders jump discontinuously at cutoff
│   ├─ No Manipulation: Units cannot precisely control X to cross cutoff
│   │   └─ Test: McCrary density test (smooth density of X at c)
│   ├─ Local Randomization: Units near c effectively randomized
│   └─ Exclusion: Treatment assignment rule only affects Y through D
├─ Implementation:
│   ├─ Bandwidth Choice:
│   │   ├─ Imbens-Kalyanaraman (2012): Data-driven optimal h
│   │   ├─ Calonico-Cattaneo-Titiunik (2014): MSE-optimal with robust inference
│   │   └─ Cross-validation: Minimize out-of-sample prediction error
│   ├─ Polynomial Order: Local linear (p=1) preferred over higher orders
│   ├─ Kernel Function: Triangular, Epanechnikov weight observations by distance
│   └─ Robust Inference: Clustered SE if running variable discrete
└─ Diagnostics:
    ├─ Placebo Cutoffs: Test for jumps at fake thresholds (should be zero)
    ├─ Covariate Balance: Pre-determined X should be smooth at c
    ├─ Density Test: Check for bunching/gaps in running variable
    └─ Donut RDD: Exclude observations in (c-δ, c+δ) to test sensitivity
```

**Interaction:** Discontinuous treatment rule + Continuous potential outcomes → Identify local causal effect

## 5. Mini-Project
Simulate and estimate sharp and fuzzy RDD:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.kde import KDEUnivariate

np.random.seed(42)
n = 1000

# ===== Data Generating Process: Sharp RDD =====
# Running variable (e.g., test score, income, age)
cutoff = 0
X = np.random.normal(0, 1, n)

# Treatment assignment (sharp): D = 1 if X >= cutoff
D = (X >= cutoff).astype(int)

# Outcome: Smooth function of X + treatment jump at cutoff
true_effect = 2.5
Y = 10 + 0.5*X - 0.2*X**2 + true_effect*D + np.random.normal(0, 1, n)

df = pd.DataFrame({'X': X, 'D': D, 'Y': Y})
df = df.sort_values('X').reset_index(drop=True)

# ===== Visualization of Sharp RDD =====
print("="*70)
print("SHARP REGRESSION DISCONTINUITY DESIGN")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Raw data with cutoff
axes[0, 0].scatter(df[df['D']==0]['X'], df[df['D']==0]['Y'], 
                   alpha=0.4, s=20, color='red', label='Control (D=0)')
axes[0, 0].scatter(df[df['D']==1]['X'], df[df['D']==1]['Y'], 
                   alpha=0.4, s=20, color='blue', label='Treated (D=1)')
axes[0, 0].axvline(cutoff, color='black', linestyle='--', linewidth=2,
                   label=f'Cutoff ({cutoff})')
axes[0, 0].set_xlabel('Running Variable (X)')
axes[0, 0].set_ylabel('Outcome (Y)')
axes[0, 0].set_title('Sharp RDD: Treatment Assignment')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# ===== Local Linear Regression (Optimal Bandwidth) =====
# Estimate using observations within bandwidth h on each side
def estimate_rdd(data, bandwidth, polynomial_order=1):
    """Estimate RDD effect using local polynomial regression"""
    subset = data[np.abs(data['X'] - cutoff) <= bandwidth].copy()
    
    if len(subset) < 20:
        return np.nan, np.nan
    
    # Create centered running variable
    subset['X_centered'] = subset['X'] - cutoff
    
    # Separate regressions on each side
    left_data = subset[subset['X'] < cutoff]
    right_data = subset[subset['X'] >= cutoff]
    
    # Polynomial terms
    formula_parts = ['Y ~ 1']
    for p in range(1, polynomial_order + 1):
        formula_parts.append(f'I(X_centered**{p})')
    formula = ' + '.join(formula_parts)
    
    if len(left_data) > polynomial_order + 1:
        model_left = smf.ols(formula, data=left_data).fit()
        y_left = model_left.predict(pd.DataFrame({'X_centered': [0]})).values[0]
    else:
        y_left = left_data['Y'].mean()
    
    if len(right_data) > polynomial_order + 1:
        model_right = smf.ols(formula, data=right_data).fit()
        y_right = model_right.predict(pd.DataFrame({'X_centered': [0]})).values[0]
    else:
        y_right = right_data['Y'].mean()
    
    tau = y_right - y_left
    
    # Standard error (simplified)
    se_left = left_data['Y'].std() / np.sqrt(len(left_data))
    se_right = right_data['Y'].std() / np.sqrt(len(right_data))
    se_tau = np.sqrt(se_left**2 + se_right**2)
    
    return tau, se_tau

# Try different bandwidths
bandwidths = np.linspace(0.2, 2, 20)
estimates = []
ses = []

for h in bandwidths:
    tau, se = estimate_rdd(df, h, polynomial_order=1)
    estimates.append(tau)
    ses.append(se)

estimates = np.array(estimates)
ses = np.array(ses)

# Plot 2: Estimates by bandwidth
axes[0, 1].plot(bandwidths, estimates, 'o-', linewidth=2, markersize=5,
                color='darkgreen', label='RDD Estimate')
axes[0, 1].fill_between(bandwidths, estimates - 1.96*ses, estimates + 1.96*ses,
                        alpha=0.3, color='green')
axes[0, 1].axhline(true_effect, color='red', linestyle='--', linewidth=2,
                   label=f'True Effect ({true_effect})')
axes[0, 1].set_xlabel('Bandwidth (h)')
axes[0, 1].set_ylabel('Treatment Effect Estimate')
axes[0, 1].set_title('Sensitivity to Bandwidth Choice')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Select optimal bandwidth (minimize MSE approximation)
optimal_h = 0.8  # Simplified; use IK or CCT in practice
tau_opt, se_opt = estimate_rdd(df, optimal_h, polynomial_order=1)

print(f"\nOptimal Bandwidth: {optimal_h:.3f}")
print(f"RDD Estimate: {tau_opt:.4f}")
print(f"Standard Error: {se_opt:.4f}")
print(f"95% CI: [{tau_opt - 1.96*se_opt:.4f}, {tau_opt + 1.96*se_opt:.4f}]")
print(f"True Effect: {true_effect:.4f}")
print(f"Bias: {tau_opt - true_effect:.4f}")

# Plot 3: Local linear fits around cutoff
subset = df[np.abs(df['X'] - cutoff) <= optimal_h].copy()
subset['X_centered'] = subset['X'] - cutoff

left_subset = subset[subset['X'] < cutoff]
right_subset = subset[subset['X'] >= cutoff]

# Fit local linear models
model_left = smf.ols('Y ~ X_centered', data=left_subset).fit()
model_right = smf.ols('Y ~ X_centered', data=right_subset).fit()

x_left_plot = np.linspace(left_subset['X_centered'].min(), 0, 100)
x_right_plot = np.linspace(0, right_subset['X_centered'].max(), 100)

y_left_plot = model_left.predict(pd.DataFrame({'X_centered': x_left_plot}))
y_right_plot = model_right.predict(pd.DataFrame({'X_centered': x_right_plot}))

axes[0, 2].scatter(subset[subset['D']==0]['X'], subset[subset['D']==0]['Y'],
                   alpha=0.5, s=30, color='red', label='Control')
axes[0, 2].scatter(subset[subset['D']==1]['X'], subset[subset['D']==1]['Y'],
                   alpha=0.5, s=30, color='blue', label='Treated')
axes[0, 2].plot(x_left_plot + cutoff, y_left_plot, 'r-', linewidth=3,
                label='Left fit')
axes[0, 2].plot(x_right_plot + cutoff, y_right_plot, 'b-', linewidth=3,
                label='Right fit')

# Highlight discontinuity
y_left_at_c = model_left.predict(pd.DataFrame({'X_centered': [0]})).values[0]
y_right_at_c = model_right.predict(pd.DataFrame({'X_centered': [0]})).values[0]
axes[0, 2].plot([cutoff, cutoff], [y_left_at_c, y_right_at_c], 'go-',
                linewidth=4, markersize=10, label=f'Jump = {tau_opt:.2f}')

axes[0, 2].axvline(cutoff, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
axes[0, 2].set_xlabel('Running Variable (X)')
axes[0, 2].set_ylabel('Outcome (Y)')
axes[0, 2].set_title(f'Local Linear Fit (h={optimal_h:.2f})')
axes[0, 2].legend(loc='upper left', fontsize=8)
axes[0, 2].grid(alpha=0.3)
axes[0, 2].set_xlim(cutoff - optimal_h, cutoff + optimal_h)

# ===== Fuzzy RDD =====
# Imperfect compliance: Treatment uptake jumps at cutoff but not 0→1
np.random.seed(43)
X_fuzzy = np.random.normal(0, 1, n)
eligibility = (X_fuzzy >= cutoff).astype(int)

# Treatment uptake depends on eligibility but imperfectly
compliance_prob = 0.4 + 0.5*eligibility  # 40% control, 90% treated
D_fuzzy = (np.random.rand(n) < compliance_prob).astype(int)

# Outcome depends on actual treatment
Y_fuzzy = 10 + 0.5*X_fuzzy - 0.2*X_fuzzy**2 + true_effect*D_fuzzy + np.random.normal(0, 1, n)

df_fuzzy = pd.DataFrame({'X': X_fuzzy, 'Eligible': eligibility, 'D': D_fuzzy, 'Y': Y_fuzzy})
df_fuzzy = df_fuzzy.sort_values('X').reset_index(drop=True)

print("\n" + "="*70)
print("FUZZY REGRESSION DISCONTINUITY DESIGN")
print("="*70)

# First stage: Treatment uptake
def estimate_first_stage(data, bandwidth):
    subset = data[np.abs(data['X'] - cutoff) <= bandwidth].copy()
    subset['X_centered'] = subset['X'] - cutoff
    
    left = subset[subset['X'] < cutoff]
    right = subset[subset['X'] >= cutoff]
    
    if len(left) > 2 and len(right) > 2:
        d_left = left['D'].mean()
        d_right = right['D'].mean()
        jump_first = d_right - d_left
    else:
        jump_first = np.nan
    
    return jump_first

# Reduced form: Outcome
def estimate_reduced_form(data, bandwidth):
    subset = data[np.abs(data['X'] - cutoff) <= bandwidth].copy()
    subset['X_centered'] = subset['X'] - cutoff
    
    left = subset[subset['X'] < cutoff]
    right = subset[subset['X'] >= cutoff]
    
    if len(left) > 2 and len(right) > 2:
        y_left = left['Y'].mean()
        y_right = right['Y'].mean()
        jump_reduced = y_right - y_left
    else:
        jump_reduced = np.nan
    
    return jump_reduced

first_stage_jump = estimate_first_stage(df_fuzzy, optimal_h)
reduced_form_jump = estimate_reduced_form(df_fuzzy, optimal_h)

# Fuzzy RDD estimate (Wald estimator)
tau_fuzzy = reduced_form_jump / first_stage_jump

print(f"\nFirst Stage Jump (Treatment Uptake): {first_stage_jump:.4f}")
print(f"Reduced Form Jump (Intent-to-Treat): {reduced_form_jump:.4f}")
print(f"Fuzzy RDD Estimate (Wald): {tau_fuzzy:.4f}")
print(f"True Effect: {true_effect:.4f}")
print(f"Interpretation: LATE for compliers at cutoff")

# Plot 4: First stage (treatment uptake)
bins = np.linspace(-2, 2, 20)
bin_centers = (bins[:-1] + bins[1:]) / 2
uptake_by_bin = []

for i in range(len(bins)-1):
    mask = (df_fuzzy['X'] >= bins[i]) & (df_fuzzy['X'] < bins[i+1])
    uptake_by_bin.append(df_fuzzy[mask]['D'].mean())

axes[1, 0].scatter(bin_centers, uptake_by_bin, s=80, alpha=0.7,
                   c=['red' if x < cutoff else 'blue' for x in bin_centers])
axes[1, 0].axvline(cutoff, color='black', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Running Variable (X)')
axes[1, 0].set_ylabel('Treatment Uptake Probability')
axes[1, 0].set_title(f'First Stage: Jump = {first_stage_jump:.3f}')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_ylim(0, 1)

# Plot 5: Reduced form (intent-to-treat)
outcome_by_bin = []
for i in range(len(bins)-1):
    mask = (df_fuzzy['X'] >= bins[i]) & (df_fuzzy['X'] < bins[i+1])
    outcome_by_bin.append(df_fuzzy[mask]['Y'].mean())

axes[1, 1].scatter(bin_centers, outcome_by_bin, s=80, alpha=0.7,
                   c=['red' if x < cutoff else 'blue' for x in bin_centers])
axes[1, 1].axvline(cutoff, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Running Variable (X)')
axes[1, 1].set_ylabel('Mean Outcome (Y)')
axes[1, 1].set_title(f'Reduced Form: Jump = {reduced_form_jump:.3f}')
axes[1, 1].grid(alpha=0.3)

# Plot 6: McCrary density test (manipulation check)
kde = KDEUnivariate(df['X'])
kde.fit()

axes[1, 2].plot(kde.support, kde.density, linewidth=2, color='purple')
axes[1, 2].axvline(cutoff, color='black', linestyle='--', linewidth=2,
                   label='Cutoff')
axes[1, 2].fill_between(kde.support, kde.density, alpha=0.3, color='purple')
axes[1, 2].set_xlabel('Running Variable (X)')
axes[1, 2].set_ylabel('Density')
axes[1, 2].set_title('McCrary Density Test (No Manipulation)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

# Formal McCrary test
left_density = len(df[df['X'] < cutoff]) / (df[df['X'] < cutoff]['X'].max() - 
                                             df[df['X'] < cutoff]['X'].min())
right_density = len(df[df['X'] >= cutoff]) / (df[df['X'] >= cutoff]['X'].max() - 
                                               df[df['X'] >= cutoff]['X'].min())
log_diff = np.log(right_density) - np.log(left_density)

print(f"\nMcCrary Density Test:")
print(f"Log density difference: {log_diff:.4f}")
print(f"(Close to zero suggests no manipulation)")

plt.tight_layout()
plt.show()

# ===== Placebo Tests =====
print("\n" + "="*70)
print("PLACEBO TESTS (Fake Cutoffs)")
print("="*70)

fake_cutoffs = [-0.5, 0.5]
for fake_c in fake_cutoffs:
    df_temp = df.copy()
    df_temp['D_fake'] = (df_temp['X'] >= fake_c).astype(int)
    
    # Estimate at fake cutoff
    subset = df_temp[np.abs(df_temp['X'] - fake_c) <= optimal_h].copy()
    if len(subset) > 20:
        subset['X_centered'] = subset['X'] - fake_c
        left = subset[subset['X'] < fake_c]
        right = subset[subset['X'] >= fake_c]
        
        y_left_fake = left['Y'].mean()
        y_right_fake = right['Y'].mean()
        tau_fake = y_right_fake - y_left_fake
        
        print(f"Fake Cutoff at {fake_c}: τ = {tau_fake:.4f} (expect ~0)")
```

## 6. Challenge Round
When does RDD fail or mislead?
- **Manipulation**: Units sort across cutoff (McCrary test detects bunching/gaps in density)
- **Confounding discontinuity**: Other policies/characteristics jump at same threshold
- **Functional form**: Misspecified polynomial creates artificial jumps
- **Bandwidth sensitivity**: Results change dramatically with different h choices
- **Extrapolation error**: Estimand is local (at cutoff only), not generalizable
- **Discrete running variable**: Bunching at integer values complicates inference

## 7. Key References
- [Lee & Lemieux (2010) - Regression Discontinuity Designs in Economics](https://www.aeaweb.org/articles?id=10.1257/jel.48.2.281)
- [Imbens & Lemieux (2008) - RDD: A Guide to Practice](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001091)
- [Calonico, Cattaneo, Titiunik (2014) - Robust Data-Driven Inference in RDD](https://www.stata-journal.com/article.html?article=st0366)

---
**Status:** Core quasi-experimental method | **Complements:** IV, Fuzzy RDD, Kink Design, Bunching Estimators
