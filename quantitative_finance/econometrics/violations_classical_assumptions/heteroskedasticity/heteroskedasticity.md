# Heteroskedasticity

## 1. Concept Skeleton
**Definition:** Non-constant error variance across observations; Var(εᵢ|Xᵢ) = σᵢ² varies with i; violates homoskedasticity assumption  
**Purpose:** Detect inefficiency in OLS standard errors; use robust/GLS methods; model variance structure for efficient estimation  
**Prerequisites:** OLS regression, variance properties, hypothesis testing, weighted least squares, robust standard errors

## 2. Comparative Framing
| Method | OLS (Homoskedastic) | OLS + Robust SE | Weighted Least Squares (WLS) | Feasible GLS (FGLS) |
|--------|---------------------|-----------------|------------------------------|---------------------|
| **Assumption** | Var(ε)=σ² constant | Allows heteroskedasticity | Known variance weights | Estimated variance function |
| **Efficiency** | BLUE if homoskedastic | Inefficient but consistent | Efficient if weights correct | Efficient asymptotically |
| **Standard Errors** | Biased under hetero | Consistent (robust) | Correct if model right | Correct asymptotically |
| **Inference** | Invalid if hetero | Valid | Valid | Valid for large N |

## 3. Examples + Counterexamples

**Classic Example:**  
Income-consumption regression: High-income households have more variance in spending. Var(εᵢ|incomeᵢ) increases with income. White/robust SE corrects inference.

**Failure Case:**  
Small sample (N<50) with severe heteroskedasticity: Robust SE have poor finite-sample properties. Bootstrap or WLS preferred.

**Edge Case:**  
Grouped data with known group sizes: Variance proportional to 1/nᵢ. WLS with weights wᵢ=nᵢ is optimal, better than robust SE.

## 4. Layer Breakdown
```
Heteroskedasticity Framework:
├─ Definition and Consequences:
│   ├─ Homoskedasticity: Var(εᵢ|Xᵢ) = σ² (constant)
│   ├─ Heteroskedasticity: Var(εᵢ|Xᵢ) = σᵢ² (varies)
│   ├─ Consequences for OLS:
│   │   ├─ β̂ still unbiased and consistent
│   │   ├─ β̂ no longer BLUE (inefficient)
│   │   ├─ Standard errors biased (usually downward)
│   │   └─ t-tests, F-tests invalid
│   └─ Common Patterns:
│       ├─ Increasing variance: σᵢ² = σ²Xᵢᵏ (k>0)
│       ├─ Grouped heteroskedasticity: Different σ² by group
│       └─ ARCH effects: Variance depends on past errors
├─ Detection Tests:
│   ├─ Graphical Methods:
│   │   ├─ Residual plot: |ε̂ᵢ| or ε̂ᵢ² vs X̂ᵢ
│   │   ├─ Scale-location plot: √|ε̂ᵢ| vs Ŷᵢ
│   │   └─ Patterns: Fan shape, clusters indicate hetero
│   ├─ Breusch-Pagan Test:
│   │   ├─ H₀: Homoskedasticity
│   │   ├─ Auxiliary regression: ε̂ᵢ² on Xᵢ
│   │   ├─ Test statistic: LM = n·R² ~ χ²(k)
│   │   └─ Reject H₀ → Heteroskedasticity present
│   ├─ White Test:
│   │   ├─ More general than BP (includes squares, cross-products)
│   │   ├─ Auxiliary regression: ε̂ᵢ² on Xᵢ, Xᵢ², XᵢXⱼ
│   │   └─ LM = n·R² ~ χ²(df)
│   ├─ Goldfeld-Quandt Test:
│   │   ├─ Order data by suspected X variable
│   │   ├─ Split sample, drop middle observations
│   │   ├─ Compare SSR: F = SSR₂/SSR₁
│   │   └─ Reject if F significantly > 1
│   └─ Park Test:
│       ├─ Log transformation: ln(ε̂ᵢ²) = α + β ln(Xᵢ) + vᵢ
│       └─ Test H₀: β = 0
├─ Robust Standard Errors (Heteroskedasticity-Consistent):
│   ├─ White (HC0): Var(β̂) = (X'X)⁻¹(Σε̂ᵢ²xᵢxᵢ')(X'X)⁻¹
│   │   └─ Original sandwich estimator
│   ├─ HC1: Correction factor n/(n-k)
│   ├─ HC2: ε̂ᵢ²/(1-hᵢᵢ) where hᵢᵢ = leverage
│   ├─ HC3: ε̂ᵢ²/(1-hᵢᵢ)² (recommended for small samples)
│   ├─ Clustered SE: For grouped heteroskedasticity
│   └─ Properties:
│       ├─ β̂ unchanged (still OLS estimates)
│       ├─ SE consistent as n→∞
│       └─ Poor finite-sample properties if N small
├─ Weighted Least Squares (WLS):
│   ├─ Idea: Give less weight to high-variance observations
│   ├─ Transformation: Multiply by wᵢ = 1/σᵢ
│   │   └─ wᵢYᵢ = wᵢXᵢ'β + wᵢεᵢ, now Var(wᵢεᵢ) = σ²
│   ├─ Optimal Weights: wᵢ = 1/σᵢ (need to know σᵢ)
│   ├─ Known Heteroskedasticity:
│   │   ├─ Grouped data: wᵢ = √nᵢ
│   │   ├─ Replicated observations: wᵢ = √mᵢ
│   │   └─ Known variance function: σᵢ² = h(Xᵢ)
│   └─ Properties:
│       ├─ BLUE if weights correct
│       ├─ More efficient than OLS
│       └─ Wrong weights → inconsistent
├─ Feasible GLS (FGLS):
│   ├─ Step 1: OLS regression, obtain ε̂ᵢ
│   ├─ Step 2: Model variance function
│   │   ├─ ln(ε̂ᵢ²) = α + Zᵢ'γ + vᵢ
│   │   ├─ σ̂ᵢ² = exp(α̂ + Zᵢ'γ̂)
│   │   └─ Z can be X, X², functions of X
│   ├─ Step 3: WLS with ŵᵢ = 1/σ̂ᵢ
│   ├─ Properties:
│   │   ├─ Consistent and asymptotically efficient
│   │   ├─ SE require adjustment (estimated weights)
│   │   └─ May not improve over robust SE in finite samples
│   └─ Alternatives:
│       ├─ Multiplicative heteroskedasticity
│       └─ Two-step or iterative procedures
├─ Model Selection:
│   ├─ Variance Function Forms:
│   │   ├─ Linear: σᵢ² = α + β·Xᵢ
│   │   ├─ Exponential: σᵢ² = exp(α + β·Xᵢ)
│   │   └─ Power: σᵢ² = σ²·Xᵢᵝ
│   ├─ Specification Tests: Auxiliary regression diagnostics
│   └─ Information Criteria: AIC, BIC for variance model
└─ Practical Recommendations:
    ├─ Always Check: Residual plots before formal tests
    ├─ Large N (>200): Use robust SE (HC3)
    ├─ Small N (<50): WLS if variance structure known
    ├─ Unknown Form: Robust SE more reliable than FGLS
    └─ Panel Data: Use clustered SE at entity level
```

**Interaction:** Detect heteroskedasticity (BP/White test) → Use robust SE for inference OR estimate variance function → WLS/FGLS for efficiency

## 5. Mini-Project
Simulate heteroskedasticity, detect it, and compare correction methods:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.regression.linear_model import WLS
from scipy import stats
import seaborn as sns

np.random.seed(456)

# ===== Simulate Data with Heteroskedasticity =====
n = 500  # Sample size

# Generate X variables
X1 = np.random.uniform(1, 10, n)
X2 = np.random.normal(5, 2, n)

# Heteroskedastic error variance increases with X1
# σᵢ² = σ² * X1ᵢ
sigma_i = 0.5 * X1  # Standard deviation proportional to X1

# Generate heteroskedastic errors
epsilon = np.random.normal(0, sigma_i)

# True model: Y = 2 + 3*X1 + 1.5*X2 + ε
Y = 2 + 3*X1 + 1.5*X2 + epsilon

# Create DataFrame
df = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'sigma_true': sigma_i})

print("="*80)
print("HETEROSKEDASTICITY: DETECTION AND CORRECTION")
print("="*80)
print(f"\nSimulation Setup:")
print(f"  Sample Size: {n}")
print(f"  True Model: Y = 2 + 3*X1 + 1.5*X2 + ε")
print(f"  Heteroskedasticity: σᵢ = 0.5*X1ᵢ (variance increases with X1)")
print(f"\nDescriptive Statistics:")
print(df[['Y', 'X1', 'X2']].describe().round(3))

# ===== OLS Estimation (Ignoring Heteroskedasticity) =====
print("\n" + "="*80)
print("OLS REGRESSION (Standard Errors Incorrect)")
print("="*80)

X = sm.add_constant(df[['X1', 'X2']])
ols_model = sm.OLS(df['Y'], X).fit()
print(ols_model.summary())

# Save residuals
df['residuals_ols'] = ols_model.resid
df['residuals_squared'] = df['residuals_ols']**2
df['fitted_values'] = ols_model.fittedvalues

print("\nOLS Estimates vs True:")
print(f"  Intercept: {ols_model.params['const']:.4f} (True: 2.0)")
print(f"  X1 coeff:  {ols_model.params['X1']:.4f} (True: 3.0)")
print(f"  X2 coeff:  {ols_model.params['X2']:.4f} (True: 1.5)")

# ===== Visual Diagnosis =====
print("\n" + "="*80)
print("GRAPHICAL DIAGNOSIS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Residuals vs Fitted Values
axes[0, 0].scatter(df['fitted_values'], df['residuals_ols'], alpha=0.5, s=20)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted (Fan Pattern)')
axes[0, 0].grid(alpha=0.3)

# Add smoothed line
from scipy.signal import savgol_filter
sorted_idx = np.argsort(df['fitted_values'])
smoothed = savgol_filter(df['residuals_ols'].iloc[sorted_idx], 51, 3)
axes[0, 0].plot(df['fitted_values'].iloc[sorted_idx], smoothed, 
               'b-', linewidth=2, label='Smoothed')
axes[0, 0].legend()

# Plot 2: Residuals vs X1 (Source of Heteroskedasticity)
axes[0, 1].scatter(df['X1'], df['residuals_ols'], alpha=0.5, s=20)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('X1')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals vs X1 (Increasing Spread)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Squared Residuals vs X1
axes[0, 2].scatter(df['X1'], df['residuals_squared'], alpha=0.5, s=20)
axes[0, 2].set_xlabel('X1')
axes[0, 2].set_ylabel('Squared Residuals')
axes[0, 2].set_title('ε² vs X1 (Positive Relationship)')
axes[0, 2].grid(alpha=0.3)

# Fit line to show relationship
z = np.polyfit(df['X1'], df['residuals_squared'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['X1'].min(), df['X1'].max(), 100)
axes[0, 2].plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear Fit')
axes[0, 2].legend()

# Plot 4: Scale-Location Plot
sqrt_abs_resid = np.sqrt(np.abs(df['residuals_ols']))
axes[1, 0].scatter(df['fitted_values'], sqrt_abs_resid, alpha=0.5, s=20)
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Residuals|')
axes[1, 0].set_title('Scale-Location Plot')
axes[1, 0].grid(alpha=0.3)

# Plot 5: Q-Q Plot
stats.probplot(df['residuals_ols'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Check)')
axes[1, 1].grid(alpha=0.3)

# Plot 6: Histogram of Residuals
axes[1, 2].hist(df['residuals_ols'], bins=40, alpha=0.7, edgecolor='black')
axes[1, 2].set_xlabel('Residuals')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Residual Distribution')
axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('heteroskedasticity_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visual inspection suggests heteroskedasticity (fan-shaped pattern)")

# ===== Formal Tests =====
print("\n" + "="*80)
print("FORMAL TESTS FOR HETEROSKEDASTICITY")
print("="*80)

# Breusch-Pagan Test
bp_stat, bp_pval, _, _ = het_breuschpagan(df['residuals_ols'], X)
print("\nBreusch-Pagan Test:")
print(f"  H₀: Homoskedasticity (constant variance)")
print(f"  LM Statistic: {bp_stat:.4f}")
print(f"  p-value: {bp_pval:.6f}")
if bp_pval < 0.05:
    print("  ✓ Reject H₀: Heteroskedasticity detected")
else:
    print("  Fail to reject: No evidence of heteroskedasticity")

# White Test
white_stat, white_pval, _, _ = het_white(df['residuals_ols'], X)
print("\nWhite Test (General):")
print(f"  H₀: Homoskedasticity")
print(f"  LM Statistic: {white_stat:.4f}")
print(f"  p-value: {white_pval:.6f}")
if white_pval < 0.05:
    print("  ✓ Reject H₀: Heteroskedasticity detected")
else:
    print("  Fail to reject: No evidence of heteroskedasticity")

# Goldfeld-Quandt Test (Manual)
print("\nGoldfeld-Quandt Test:")
# Sort by X1, split sample
df_sorted = df.sort_values('X1').reset_index(drop=True)
n_drop = int(0.2 * n)  # Drop middle 20%
n1 = int((n - n_drop) / 2)

df_low = df_sorted.iloc[:n1]
df_high = df_sorted.iloc[-n1:]

# Regressions on subsamples
X_low = sm.add_constant(df_low[['X1', 'X2']])
X_high = sm.add_constant(df_high[['X1', 'X2']])

ols_low = sm.OLS(df_low['Y'], X_low).fit()
ols_high = sm.OLS(df_high['Y'], X_high).fit()

# F-statistic
gq_stat = ols_high.ssr / ols_low.ssr
gq_pval = 1 - stats.f.cdf(gq_stat, ols_high.df_resid, ols_low.df_resid)

print(f"  Order by X1, drop middle {n_drop} observations")
print(f"  SSR (low X1):  {ols_low.ssr:.2f}")
print(f"  SSR (high X1): {ols_high.ssr:.2f}")
print(f"  F-statistic: {gq_stat:.4f}")
print(f"  p-value: {gq_pval:.6f}")
if gq_pval < 0.05:
    print("  ✓ Reject H₀: Variance differs across X1 range")
else:
    print("  Fail to reject: No evidence of heteroskedasticity")

# ===== OLS with Robust Standard Errors =====
print("\n" + "="*80)
print("OLS WITH ROBUST STANDARD ERRORS (WHITE/HC)")
print("="*80)

# HC0 (White)
ols_hc0 = sm.OLS(df['Y'], X).fit(cov_type='HC0')
print("\nHC0 (White) - Original:")
print(ols_hc0.summary())

# HC3 (recommended for small samples)
ols_hc3 = sm.OLS(df['Y'], X).fit(cov_type='HC3')
print("\nHC3 (Recommended):")
print(ols_hc3.summary())

# Compare standard errors
se_comparison = pd.DataFrame({
    'OLS (incorrect)': ols_model.bse,
    'HC0 (White)': ols_hc0.bse,
    'HC3': ols_hc3.bse
})

print("\n" + "="*80)
print("STANDARD ERROR COMPARISON")
print("="*80)
print(se_comparison.round(4))

se_increase = ((ols_hc3.bse - ols_model.bse) / ols_model.bse * 100)
print("\nPercentage Increase (HC3 vs OLS):")
print(se_increase.round(2))

# ===== Weighted Least Squares =====
print("\n" + "="*80)
print("WEIGHTED LEAST SQUARES (WLS)")
print("="*80)

# Since we know the true variance function: σᵢ² ∝ X1ᵢ
# Optimal weights: wᵢ = 1/σᵢ = 1/√X1ᵢ (assuming we know the form)
weights = 1 / df['X1']

wls_model = WLS(df['Y'], X, weights=weights).fit()
print(wls_model.summary())

print("\nWLS Estimates (with correct weights):")
print(f"  Intercept: {wls_model.params['const']:.4f} (True: 2.0)")
print(f"  X1 coeff:  {wls_model.params['X1']:.4f} (True: 3.0)")
print(f"  X2 coeff:  {wls_model.params['X2']:.4f} (True: 1.5)")

# ===== Feasible GLS =====
print("\n" + "="*80)
print("FEASIBLE GLS (FGLS)")
print("="*80)

# Step 1: OLS to get residuals (already done)
# Step 2: Model log(ε²) on X
log_resid_sq = np.log(df['residuals_squared'] + 0.01)  # Add small constant
X_variance = sm.add_constant(df[['X1', 'X2']])
variance_model = sm.OLS(log_resid_sq, X_variance).fit()

print("Variance Function Model:")
print(variance_model.summary())

# Predicted variances
df['predicted_log_var'] = variance_model.fittedvalues
df['predicted_var'] = np.exp(df['predicted_log_var'])
df['weights_fgls'] = 1 / df['predicted_var']

# Step 3: WLS with estimated weights
fgls_model = WLS(df['Y'], X, weights=df['weights_fgls']).fit()
print("\nFGLS Results:")
print(fgls_model.summary())

# ===== Comparison of All Methods =====
print("\n" + "="*80)
print("COEFFICIENT COMPARISON: ALL METHODS")
print("="*80)

coef_comparison = pd.DataFrame({
    'True': [2.0, 3.0, 1.5],
    'OLS': [ols_model.params['const'], ols_model.params['X1'], ols_model.params['X2']],
    'OLS+HC3': [ols_hc3.params['const'], ols_hc3.params['X1'], ols_hc3.params['X2']],
    'WLS': [wls_model.params['const'], wls_model.params['X1'], wls_model.params['X2']],
    'FGLS': [fgls_model.params['const'], fgls_model.params['X1'], fgls_model.params['X2']]
}, index=['Intercept', 'X1', 'X2'])

print("\nCoefficients:")
print(coef_comparison.round(4))

# Standard errors
se_all = pd.DataFrame({
    'OLS': ols_model.bse,
    'OLS+HC3': ols_hc3.bse,
    'WLS': wls_model.bse,
    'FGLS': fgls_model.bse
})

print("\nStandard Errors:")
print(se_all.round(4))

# ===== Additional Visualizations =====
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Coefficient Estimates
methods = ['True', 'OLS', 'OLS+HC3', 'WLS', 'FGLS']
x1_coefs = coef_comparison.loc['X1'].values
x2_coefs = coef_comparison.loc['X2'].values

x_pos = np.arange(len(methods))
width = 0.35

axes2[0, 0].bar(x_pos - width/2, x1_coefs, width, label='X1', alpha=0.8)
axes2[0, 0].bar(x_pos + width/2, x2_coefs, width, label='X2', alpha=0.8)
axes2[0, 0].set_xticks(x_pos)
axes2[0, 0].set_xticklabels(methods, rotation=45)
axes2[0, 0].set_ylabel('Coefficient')
axes2[0, 0].set_title('Coefficient Estimates Across Methods')
axes2[0, 0].legend()
axes2[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Standard Errors Comparison
se_plot = se_all.loc[['X1', 'X2']].T
se_plot.plot(kind='bar', ax=axes2[0, 1], alpha=0.8)
axes2[0, 1].set_ylabel('Standard Error')
axes2[0, 1].set_title('Standard Errors: OLS vs Robust vs WLS')
axes2[0, 1].set_xticklabels(axes2[0, 1].get_xticklabels(), rotation=45)
axes2[0, 1].legend(title='Variable')
axes2[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: True vs Estimated Variance
axes2[1, 0].scatter(df['sigma_true']**2, df['predicted_var'], alpha=0.5, s=20)
axes2[1, 0].plot([0, df['sigma_true'].max()**2], 
                [0, df['sigma_true'].max()**2],
                'r--', linewidth=2, label='45° line')
axes2[1, 0].set_xlabel('True Variance (σᵢ²)')
axes2[1, 0].set_ylabel('FGLS Estimated Variance')
axes2[1, 0].set_title('Variance Estimation Quality')
axes2[1, 0].legend()
axes2[1, 0].grid(alpha=0.3)

# Plot 4: Efficiency Gains (SE ratio)
se_ratio = pd.DataFrame({
    'X1': [ols_model.bse['X1'], ols_hc3.bse['X1'], 
           wls_model.bse['X1'], fgls_model.bse['X1']],
    'X2': [ols_model.bse['X2'], ols_hc3.bse['X2'],
           wls_model.bse['X2'], fgls_model.bse['X2']]
}, index=['OLS', 'OLS+HC3', 'WLS', 'FGLS'])

se_ratio.plot(kind='bar', ax=axes2[1, 1], alpha=0.8)
axes2[1, 1].set_ylabel('Standard Error')
axes2[1, 1].set_title('Efficiency Comparison (Lower is Better)')
axes2[1, 1].set_xticklabels(axes2[1, 1].get_xticklabels(), rotation=45)
axes2[1, 1].legend(title='Variable')
axes2[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('heteroskedasticity_methods_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Monte Carlo Simulation: Coverage Rates =====
print("\n" + "="*80)
print("MONTE CARLO: CONFIDENCE INTERVAL COVERAGE")
print("="*80)

n_sim = 1000
coverage_ols = 0
coverage_hc3 = 0
coverage_wls = 0

true_beta1 = 3.0

for sim in range(n_sim):
    # Generate data
    X1_sim = np.random.uniform(1, 10, n)
    X2_sim = np.random.normal(5, 2, n)
    sigma_i_sim = 0.5 * X1_sim
    epsilon_sim = np.random.normal(0, sigma_i_sim)
    Y_sim = 2 + 3*X1_sim + 1.5*X2_sim + epsilon_sim
    
    X_sim = np.column_stack([np.ones(n), X1_sim, X2_sim])
    
    # OLS
    ols_sim = sm.OLS(Y_sim, X_sim).fit()
    ci_ols = ols_sim.conf_int().iloc[1]
    if ci_ols[0] <= true_beta1 <= ci_ols[1]:
        coverage_ols += 1
    
    # OLS + HC3
    ols_hc3_sim = sm.OLS(Y_sim, X_sim).fit(cov_type='HC3')
    ci_hc3 = ols_hc3_sim.conf_int().iloc[1]
    if ci_hc3[0] <= true_beta1 <= ci_hc3[1]:
        coverage_hc3 += 1
    
    # WLS (with known weights)
    weights_sim = 1 / X1_sim
    wls_sim = WLS(Y_sim, X_sim, weights=weights_sim).fit()
    ci_wls = wls_sim.conf_int().iloc[1]
    if ci_wls[0] <= true_beta1 <= ci_wls[1]:
        coverage_wls += 1

coverage_ols /= n_sim
coverage_hc3 /= n_sim
coverage_wls /= n_sim

print(f"\n95% Confidence Interval Coverage (should be 0.95):")
print(f"  OLS (incorrect SE):  {coverage_ols:.3f}  {'✗' if coverage_ols < 0.90 else '✓'}")
print(f"  OLS + HC3:           {coverage_hc3:.3f}  {'✓' if 0.94 <= coverage_hc3 <= 0.96 else '⚠'}")
print(f"  WLS (known weights): {coverage_wls:.3f}  {'✓' if 0.94 <= coverage_wls <= 0.96 else '⚠'}")

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n1. Detection:")
print("   • Visual: Fan-shaped residual plot indicates heteroskedasticity")
print("   • Formal: BP test, White test both reject homoskedasticity")

print("\n2. Consequences of Ignoring:")
print(f"   • Coefficients: Unbiased but inefficient")
print(f"   • Standard errors: Biased (OLS underestimates by ~{(ols_hc3.bse['X1'] - ols_model.bse['X1'])/ols_model.bse['X1']*100:.1f}% for X1)")
print(f"   • Coverage rate: {coverage_ols:.1%} instead of 95%")

print("\n3. Correction Methods:")
print("   • Robust SE (HC3): Always safe, valid inference, no efficiency gain")
print("   • WLS: More efficient if variance function known")
print("   • FGLS: Asymptotically efficient, may not beat robust SE in finite samples")

print("\n4. Practical Recommendations:")
print("   • Default: Use robust SE (HC3) for valid inference")
print("   • Known structure (grouped data): Use WLS")
print("   • Large N with clear pattern: Consider FGLS")
print("   • Always plot residuals before formal tests")
```

## 6. Challenge Round
When do heteroskedasticity corrections fail or mislead?
- **Small samples (N<50)**: Robust SE have poor finite-sample properties → Bootstrap or exact methods
- **Wrong WLS weights**: Incorrect variance specification makes WLS inconsistent → Robust SE safer
- **Clustered heteroskedasticity**: Individual robust SE insufficient → Use clustered standard errors
- **Time series**: ARCH effects require specialized models (GARCH) → Standard methods inadequate
- **Extreme heteroskedasticity**: Log transformation may be needed → Retransformation bias
- **Influential observations**: High-variance outliers get low weight in WLS → May hide problems

## 7. Key References
- [White (1980) - Heteroskedasticity-Consistent Covariance Matrix Estimator](https://doi.org/10.2307/1912934)
- [Wooldridge - Introductory Econometrics (Ch 8)](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge)
- [Long & Ervin (2000) - Using HC Standard Errors in Small Samples](https://doi.org/10.1080/00031305.2000.10474549)

---
**Status:** Fundamental diagnostic | **Complements:** Robust Standard Errors, GLS, Diagnostic Tests, Residual Analysis
