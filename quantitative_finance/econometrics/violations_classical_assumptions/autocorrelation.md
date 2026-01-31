# Autocorrelation (Serial Correlation)

## 1. Concept Skeleton
**Definition:** Correlation between error terms across time; Cov(εₜ, εₛ) ≠ 0 for t≠s; violates independence assumption  
**Purpose:** Detect inefficiency in OLS; use HAC standard errors; model dynamic structure with AR/MA errors  
**Prerequisites:** Time series data, OLS assumptions, ACF/PACF plots, Durbin-Watson test, Newey-West standard errors

## 2. Comparative Framing
| Method | OLS (No Autocorr) | OLS + HAC SE | Cochrane-Orcutt | Prais-Winsten | FGLS with AR Errors |
|--------|-------------------|--------------|-----------------|---------------|---------------------|
| **Assumption** | Cov(εₜ,εₛ)=0 | Allows autocorrelation | AR(1) errors | AR(1) errors | General AR(p) |
| **Efficiency** | BLUE if valid | Inefficient | Efficient if AR(1) | Efficient + first obs | Efficient |
| **Standard Errors** | Biased if autocorr | Consistent (HAC) | Correct if AR(1) | Correct if AR(1) | Correct |
| **First Observation** | Used | Used | Dropped | Transformed | Depends |

## 3. Examples + Counterexamples

**Classic Example:**  
GDP regression with quarterly data: Economic shocks persist across quarters. Positive autocorrelation inflates t-statistics. Newey-West SE corrects inference.

**Failure Case:**  
Structural break mistaken for autocorrelation: DW test rejects, but actually model misspecified. Adding time trend fixes issue.

**Edge Case:**  
Spatial correlation misdiagnosed as autocorrelation: Cross-sectional data with spatial dependence. Need spatial econometrics, not time series methods.

## 4. Layer Breakdown
```
Autocorrelation Framework:
├─ Definition and Patterns:
│   ├─ No Autocorrelation: Cov(εₜ, εₛ) = 0 for all t ≠ s
│   ├─ Positive Autocorr: Cov(εₜ, εₜ₋₁) > 0 (common)
│   │   └─ Errors persist; shocks carry over periods
│   ├─ Negative Autocorr: Cov(εₜ, εₜ₋₁) < 0 (less common)
│   │   └─ Errors alternate; mean reversion
│   └─ Common Structures:
│       ├─ AR(1): εₜ = ρεₜ₋₁ + uₜ, |ρ| < 1
│       ├─ MA(1): εₜ = uₜ + θuₜ₋₁
│       └─ ARMA(p,q): General autoregressive moving average
├─ Consequences for OLS:
│   ├─ Coefficients: β̂ unbiased but inefficient
│   ├─ Standard Errors: Biased (usually downward if ρ > 0)
│   ├─ t-statistics: Inflated (Type I error)
│   ├─ R²: Spuriously high
│   └─ Prediction intervals: Too narrow
├─ Sources of Autocorrelation:
│   ├─ Omitted Variables: Missing time-varying factors
│   ├─ Model Misspecification: Wrong functional form
│   ├─ Data Issues: Smoothing, interpolation, temporal aggregation
│   ├─ Economic Behavior: Adjustment lags, habit persistence
│   └─ Measurement Error: Persistent measurement issues
├─ Detection Methods:
│   ├─ Graphical:
│   │   ├─ Time series plot of residuals: εₜ vs t
│   │   ├─ ACF plot: Correlogram of residuals
│   │   └─ PACF plot: Partial autocorrelations
│   ├─ Durbin-Watson Test:
│   │   ├─ DW = Σ(εₜ - εₜ₋₁)² / Σεₜ²
│   │   ├─ Range: 0 to 4, value of 2 indicates no autocorr
│   │   ├─ DW < 2: Positive autocorrelation
│   │   ├─ DW > 2: Negative autocorrelation
│   │   ├─ Limitations: Only AR(1), no lagged Y, inconclusive region
│   │   └─ Critical values: d_L, d_U from tables
│   ├─ Breusch-Godfrey (LM) Test:
│   │   ├─ More general than DW (tests AR(p))
│   │   ├─ Auxiliary regression: εₜ on Xₜ, εₜ₋₁,...,εₜ₋ₚ
│   │   ├─ LM = n·R² ~ χ²(p)
│   │   └─ Valid with lagged Y as regressor
│   ├─ Ljung-Box Q Test:
│   │   ├─ Tests joint significance of multiple lags
│   │   ├─ Q = n(n+2)Σ(ρ̂ₖ²/(n-k))
│   │   └─ Q ~ χ²(m) where m = # lags tested
│   └─ Runs Test:
│       └─ Non-parametric test for randomness
├─ HAC Standard Errors (Heteroskedasticity and Autocorrelation Consistent):
│   ├─ Newey-West (1987):
│   │   ├─ Extends White SE to time series
│   │   ├─ Var(β̂) = (X'X)⁻¹Ω(X'X)⁻¹
│   │   ├─ Ω = Γ₀ + Σ[w(l)·(Γₗ + Γₗ')] 
│   │   │   └─ Γₗ = Σxₜxₜ₋ₗ'εₜεₜ₋ₗ (autocovariances)
│   │   ├─ Kernel weight: w(l) = 1 - l/(L+1) (Bartlett)
│   │   └─ Bandwidth L: Choose based on n (e.g., L = 0.75·n^(1/3))
│   ├─ Properties:
│   │   ├─ Consistent as n→∞
│   │   ├─ β̂ unchanged (still OLS estimates)
│   │   ├─ SE increases to account for autocorrelation
│   │   └─ Bandwidth choice matters (too small: under-correct; too large: inefficient)
│   └─ Alternative Kernels: Parzen, Quadratic Spectral
├─ Cochrane-Orcutt Procedure:
│   ├─ Assumes AR(1) errors: εₜ = ρεₜ₋₁ + uₜ
│   ├─ Step 1: OLS, get residuals ε̂ₜ
│   ├─ Step 2: Estimate ρ: ε̂ₜ = ρε̂ₜ₋₁ + vₜ
│   ├─ Step 3: Quasi-difference transformation
│   │   └─ Yₜ* = Yₜ - ρ̂Yₜ₋₁, Xₜ* = Xₜ - ρ̂Xₜ₋₁
│   ├─ Step 4: OLS on transformed data
│   ├─ Step 5: Iterate until convergence
│   ├─ Drawback: Loses first observation
│   └─ Use: Efficiency gain if ρ known/well-estimated
├─ Prais-Winsten Transformation:
│   ├─ Similar to Cochrane-Orcutt but retains first observation
│   ├─ First obs: Y₁* = √(1-ρ²)Y₁, X₁* = √(1-ρ²)X₁
│   ├─ Other obs: Same quasi-differencing as C-O
│   └─ More efficient than C-O (uses all data)
├─ Feasible GLS with AR Errors:
│   ├─ General AR(p) specification: εₜ = ρ₁εₜ₋₁ + ... + ρₚεₜ₋ₚ + uₜ
│   ├─ Estimate AR parameters from residuals
│   ├─ Transform data to remove autocorrelation
│   └─ GLS on transformed model
├─ Dynamic Models:
│   ├─ Include lagged Y: Yₜ = βXₜ + γYₜ₋₁ + εₜ
│   │   └─ May eliminate autocorrelation if omitted dynamics
│   ├─ Lagged X: Distributed lag models
│   └─ Caution: Autocorrelation tests change with lagged Y
├─ Model Selection:
│   ├─ AIC/BIC: Choose AR order p
│   ├─ Information criteria for lag length
│   └─ Diagnostic checks on transformed residuals
└─ Practical Recommendations:
    ├─ Time Series: Default to Newey-West SE
    ├─ Strong Pattern: Model dynamics explicitly
    ├─ Panel Data: Cluster SE by entity
    ├─ Short Series (T<30): Be cautious with FGLS
    └─ Always Plot: Visual inspection critical
```

**Interaction:** Detect autocorrelation (DW, BG test) → Use HAC SE for inference OR model AR structure → FGLS

## 5. Mini-Project
Simulate autocorrelated errors, detect them, and apply corrections:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS, GLSAR
from scipy import stats

np.random.seed(321)

# ===== Simulate Time Series with AR(1) Errors =====
T = 200  # Time periods
rho = 0.7  # AR(1) coefficient (positive autocorrelation)

# Generate X variable (with trend)
t = np.arange(T)
X = 5 + 0.1*t + np.random.normal(0, 1, T)

# Generate AR(1) errors
epsilon = np.zeros(T)
u = np.random.normal(0, 1, T)  # White noise innovations
epsilon[0] = u[0] / np.sqrt(1 - rho**2)  # Stationary initialization

for i in range(1, T):
    epsilon[i] = rho * epsilon[i-1] + u[i]

# True model: Y = 10 + 2*X + ε (with AR(1) errors)
Y = 10 + 2*X + epsilon

# Create DataFrame
df = pd.DataFrame({'t': t, 'Y': Y, 'X': X, 'epsilon_true': epsilon})

print("="*80)
print("AUTOCORRELATION: DETECTION AND CORRECTION")
print("="*80)
print(f"\nSimulation Setup:")
print(f"  Time Periods: {T}")
print(f"  True Model: Y = 10 + 2*X + ε")
print(f"  Error Structure: εₜ = {rho}*εₜ₋₁ + uₜ (AR(1))")
print(f"\nDescriptive Statistics:")
print(df[['Y', 'X']].describe().round(3))

# ===== OLS Estimation (Ignoring Autocorrelation) =====
print("\n" + "="*80)
print("OLS REGRESSION (Ignoring Autocorrelation)")
print("="*80)

X_reg = sm.add_constant(df['X'])
ols_model = sm.OLS(df['Y'], X_reg).fit()
print(ols_model.summary())

df['residuals'] = ols_model.resid
df['fitted'] = ols_model.fittedvalues

print("\nOLS Estimates vs True:")
print(f"  Intercept: {ols_model.params['const']:.4f} (True: 10.0)")
print(f"  X coeff:   {ols_model.params['X']:.4f} (True: 2.0)")

# ===== Visual Diagnosis =====
print("\n" + "="*80)
print("GRAPHICAL DIAGNOSIS")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Time Series of Residuals
axes[0, 0].plot(df['t'], df['residuals'], linewidth=1)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals Over Time (Persistence Visible)')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Residuals vs Lagged Residuals
df['residuals_lag1'] = df['residuals'].shift(1)
axes[0, 1].scatter(df['residuals_lag1'], df['residuals'], alpha=0.6, s=20)
axes[0, 1].set_xlabel('εₜ₋₁')
axes[0, 1].set_ylabel('εₜ')
axes[0, 1].set_title(f'εₜ vs εₜ₋₁ (Positive Correlation)')
axes[0, 1].grid(alpha=0.3)

# Add regression line
valid_data = df.dropna(subset=['residuals', 'residuals_lag1'])
z = np.polyfit(valid_data['residuals_lag1'], valid_data['residuals'], 1)
p = np.poly1d(z)
x_line = np.linspace(valid_data['residuals_lag1'].min(), 
                     valid_data['residuals_lag1'].max(), 100)
axes[0, 1].plot(x_line, p(x_line), 'r-', linewidth=2, 
               label=f'ρ̂ = {z[0]:.3f}')
axes[0, 1].legend()

# Plot 3: ACF
plot_acf(df['residuals'].dropna(), lags=20, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title('Autocorrelation Function (ACF)')
axes[1, 0].set_xlabel('Lag')

# Plot 4: PACF
plot_pacf(df['residuals'].dropna(), lags=20, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('Partial Autocorrelation Function (PACF)')
axes[1, 1].set_xlabel('Lag')

# Plot 5: True vs Estimated Errors
axes[2, 0].plot(df['t'], df['epsilon_true'], label='True ε', alpha=0.7)
axes[2, 0].plot(df['t'], df['residuals'], label='OLS Residuals', alpha=0.7)
axes[2, 0].set_xlabel('Time')
axes[2, 0].set_ylabel('Error/Residual')
axes[2, 0].set_title('True Errors vs OLS Residuals')
axes[2, 0].legend()
axes[2, 0].grid(alpha=0.3)

# Plot 6: Residual Distribution
axes[2, 1].hist(df['residuals'], bins=30, alpha=0.7, edgecolor='black')
axes[2, 1].set_xlabel('Residuals')
axes[2, 1].set_ylabel('Frequency')
axes[2, 1].set_title('Residual Distribution')
axes[2, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('autocorrelation_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Formal Tests =====
print("\n" + "="*80)
print("FORMAL TESTS FOR AUTOCORRELATION")
print("="*80)

# Durbin-Watson Test
dw_stat = durbin_watson(df['residuals'])
print(f"\nDurbin-Watson Test:")
print(f"  DW Statistic: {dw_stat:.4f}")
print(f"  Interpretation: 2 = no autocorr, 0 = positive, 4 = negative")
if dw_stat < 1.5:
    print("  ✓ Evidence of positive autocorrelation")
elif dw_stat > 2.5:
    print("  Evidence of negative autocorrelation")
else:
    print("  No strong evidence of autocorrelation")

# Estimate AR(1) coefficient from DW
rho_dw = 1 - dw_stat / 2
print(f"  Implied ρ̂ ≈ 1 - DW/2 = {rho_dw:.4f} (True ρ: {rho})")

# Breusch-Godfrey Test
bg_test = acorr_breusch_godfrey(ols_model, nlags=4)
print(f"\nBreusch-Godfrey LM Test (4 lags):")
print(f"  H₀: No autocorrelation up to lag 4")
print(f"  LM Statistic: {bg_test[0]:.4f}")
print(f"  p-value: {bg_test[1]:.6f}")
print(f"  F-statistic: {bg_test[2]:.4f}")
print(f"  F p-value: {bg_test[3]:.6f}")
if bg_test[1] < 0.05:
    print("  ✓ Reject H₀: Autocorrelation detected")
else:
    print("  Fail to reject: No evidence of autocorrelation")

# Ljung-Box Test
lb_test = acorr_ljungbox(df['residuals'].dropna(), lags=10, return_df=True)
print(f"\nLjung-Box Q Test:")
print(f"  Tests joint significance of ACF up to lag 10")
print(lb_test[['lb_stat', 'lb_pvalue']].head(10).round(4))

significant_lags = lb_test[lb_test['lb_pvalue'] < 0.05].index.tolist()
if significant_lags:
    print(f"  ✓ Significant autocorrelation at lags: {significant_lags}")
else:
    print("  No significant autocorrelation detected")

# ===== OLS with HAC Standard Errors (Newey-West) =====
print("\n" + "="*80)
print("OLS WITH NEWEY-WEST HAC STANDARD ERRORS")
print("="*80)

# Choose bandwidth (lag length) for Newey-West
# Rule of thumb: L = floor(0.75 * T^(1/3))
bandwidth = int(0.75 * T**(1/3))
print(f"Bandwidth (lag length): {bandwidth}")

ols_hac = sm.OLS(df['Y'], X_reg).fit(cov_type='HAC', cov_kwds={'maxlags': bandwidth})
print(ols_hac.summary())

# Compare standard errors
se_comparison = pd.DataFrame({
    'OLS (incorrect)': ols_model.bse,
    'Newey-West HAC': ols_hac.bse
})

print("\n" + "="*80)
print("STANDARD ERROR COMPARISON")
print("="*80)
print(se_comparison.round(4))

se_increase = ((ols_hac.bse - ols_model.bse) / ols_model.bse * 100)
print("\nPercentage Increase (HAC vs OLS):")
print(se_increase.round(2))

# ===== Cochrane-Orcutt Procedure =====
print("\n" + "="*80)
print("COCHRANE-ORCUTT ITERATIVE PROCEDURE")
print("="*80)

# Step 1: Initial OLS (already done)
# Step 2: Estimate ρ from residuals
df_lag = df.dropna(subset=['residuals', 'residuals_lag1'])
rho_hat = (df_lag['residuals'] * df_lag['residuals_lag1']).sum() / \
          (df_lag['residuals_lag1'] ** 2).sum()

print(f"Estimated ρ from OLS residuals: {rho_hat:.4f} (True: {rho})")

# Step 3: Quasi-difference transformation (loses first obs)
df['Y_diff'] = df['Y'] - rho_hat * df['Y'].shift(1)
df['X_diff'] = df['X'] - rho_hat * df['X'].shift(1)
df['const_diff'] = 1 - rho_hat

df_co = df.dropna(subset=['Y_diff', 'X_diff'])

# OLS on transformed data
X_co = df_co[['const_diff', 'X_diff']]
co_model = sm.OLS(df_co['Y_diff'], X_co).fit()

print("\nCochrane-Orcutt Results (1 iteration):")
print(co_model.summary())

# Extract original scale coefficients
beta_0_co = co_model.params['const_diff'] / (1 - rho_hat)
beta_1_co = co_model.params['X_diff']

print(f"\nCoefficients (original scale):")
print(f"  Intercept: {beta_0_co:.4f} (True: 10.0)")
print(f"  X coeff:   {beta_1_co:.4f} (True: 2.0)")

# ===== Prais-Winsten Transformation =====
print("\n" + "="*80)
print("PRAIS-WINSTEN TRANSFORMATION (Retains First Observation)")
print("="*80)

# Use statsmodels GLSAR (Generalized Least Squares with AR errors)
glsar_model = GLSAR(df['Y'].values, X_reg.values, rho=1).iterative_fit(maxiter=5)

print("\nPrais-Winsten (via GLSAR) Results:")
print(f"  Estimated ρ: {glsar_model.rho:.4f}")
print(f"\nCoefficients:")
print(f"  Intercept: {glsar_model.params[0]:.4f} (True: 10.0)")
print(f"  X coeff:   {glsar_model.params[1]:.4f} (True: 2.0)")

# Get standard errors
pw_se = np.sqrt(np.diag(glsar_model.cov_params()))
print(f"\nStandard Errors:")
print(f"  Intercept: {pw_se[0]:.4f}")
print(f"  X:         {pw_se[1]:.4f}")

# ===== Comparison of All Methods =====
print("\n" + "="*80)
print("COEFFICIENT COMPARISON: ALL METHODS")
print("="*80)

coef_comparison = pd.DataFrame({
    'True': [10.0, 2.0],
    'OLS': [ols_model.params['const'], ols_model.params['X']],
    'OLS+HAC': [ols_hac.params['const'], ols_hac.params['X']],
    'Cochrane-Orcutt': [beta_0_co, beta_1_co],
    'Prais-Winsten': [glsar_model.params[0], glsar_model.params[1]]
}, index=['Intercept', 'X'])

print("\nCoefficients:")
print(coef_comparison.round(4))

# Standard errors
se_all = pd.DataFrame({
    'OLS': ols_model.bse.values,
    'OLS+HAC': ols_hac.bse.values,
    'Cochrane-Orcutt': co_model.bse.values,
    'Prais-Winsten': pw_se
}, index=['Intercept', 'X'])

print("\nStandard Errors:")
print(se_all.round(4))

# ===== Check for Remaining Autocorrelation =====
print("\n" + "="*80)
print("DIAGNOSTIC CHECK: REMAINING AUTOCORRELATION")
print("="*80)

# Cochrane-Orcutt residuals
df_co['residuals_co'] = df_co['Y_diff'] - co_model.fittedvalues
dw_co = durbin_watson(df_co['residuals_co'])

print(f"\nDurbin-Watson on Transformed Residuals:")
print(f"  Original OLS: {dw_stat:.4f}")
print(f"  Cochrane-Orcutt: {dw_co:.4f}")
print(f"  {'✓ Improved (closer to 2)' if abs(dw_co - 2) < abs(dw_stat - 2) else '⚠ Not improved'}")

# ===== Additional Visualizations =====
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Coefficient Estimates
methods = ['True', 'OLS', 'OLS+HAC', 'C-O', 'P-W']
intercepts = coef_comparison.loc['Intercept'].values
x_coefs = coef_comparison.loc['X'].values

x_pos = np.arange(len(methods))
width = 0.35

axes2[0, 0].bar(x_pos - width/2, intercepts, width, label='Intercept', alpha=0.8)
axes2[0, 0].bar(x_pos + width/2, x_coefs, width, label='X', alpha=0.8)
axes2[0, 0].set_xticks(x_pos)
axes2[0, 0].set_xticklabels(methods)
axes2[0, 0].set_ylabel('Coefficient')
axes2[0, 0].set_title('Coefficient Estimates Across Methods')
axes2[0, 0].legend()
axes2[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Standard Errors Comparison
se_all.T.plot(kind='bar', ax=axes2[0, 1], alpha=0.8)
axes2[0, 1].set_ylabel('Standard Error')
axes2[0, 1].set_title('Standard Errors: OLS vs HAC vs FGLS')
axes2[0, 1].set_xticklabels(axes2[0, 1].get_xticklabels(), rotation=45)
axes2[0, 1].legend(title='Variable')
axes2[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: ACF Comparison (OLS vs Corrected Residuals)
axes2[1, 0].acorr(df['residuals'].dropna(), maxlags=20, 
                 label='OLS Residuals', alpha=0.7)
axes2[1, 0].acorr(df_co['residuals_co'].dropna(), maxlags=20,
                 label='C-O Residuals', alpha=0.7)
axes2[1, 0].set_xlabel('Lag')
axes2[1, 0].set_ylabel('ACF')
axes2[1, 0].set_title('Autocorrelation: Before vs After Correction')
axes2[1, 0].legend()
axes2[1, 0].grid(alpha=0.3)

# Plot 4: DW Statistics
dw_values = [dw_stat, dw_co]
dw_labels = ['OLS', 'Cochrane-Orcutt']

axes2[1, 1].bar(dw_labels, dw_values, alpha=0.8, color=['red', 'green'])
axes2[1, 1].axhline(2, color='blue', linestyle='--', linewidth=2, label='DW=2 (no autocorr)')
axes2[1, 1].set_ylabel('Durbin-Watson Statistic')
axes2[1, 1].set_title('DW Statistics: Before vs After')
axes2[1, 1].legend()
axes2[1, 1].grid(alpha=0.3, axis='y')
axes2[1, 1].set_ylim([0, 4])

plt.tight_layout()
plt.savefig('autocorrelation_methods_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Monte Carlo Simulation: Coverage Rates =====
print("\n" + "="*80)
print("MONTE CARLO: CONFIDENCE INTERVAL COVERAGE")
print("="*80)

n_sim = 1000
coverage_ols = 0
coverage_hac = 0
coverage_co = 0

true_beta = 2.0

for sim in range(n_sim):
    # Generate data
    X_sim = 5 + 0.1*np.arange(T) + np.random.normal(0, 1, T)
    
    # AR(1) errors
    eps_sim = np.zeros(T)
    u_sim = np.random.normal(0, 1, T)
    eps_sim[0] = u_sim[0] / np.sqrt(1 - rho**2)
    for i in range(1, T):
        eps_sim[i] = rho * eps_sim[i-1] + u_sim[i]
    
    Y_sim = 10 + 2*X_sim + eps_sim
    
    X_sim_reg = np.column_stack([np.ones(T), X_sim])
    
    # OLS
    ols_sim = sm.OLS(Y_sim, X_sim_reg).fit()
    ci_ols = ols_sim.conf_int()[1]
    if ci_ols[0] <= true_beta <= ci_ols[1]:
        coverage_ols += 1
    
    # OLS + HAC
    ols_hac_sim = sm.OLS(Y_sim, X_sim_reg).fit(cov_type='HAC', 
                                               cov_kwds={'maxlags': bandwidth})
    ci_hac = ols_hac_sim.conf_int()[1]
    if ci_hac[0] <= true_beta <= ci_hac[1]:
        coverage_hac += 1
    
    # Cochrane-Orcutt (simplified)
    rho_sim = (ols_sim.resid[1:] * ols_sim.resid[:-1]).sum() / \
              (ols_sim.resid[:-1] ** 2).sum()
    Y_diff_sim = Y_sim[1:] - rho_sim * Y_sim[:-1]
    X_diff_sim = X_sim[1:] - rho_sim * X_sim[:-1]
    const_diff_sim = np.ones(T-1) * (1 - rho_sim)
    
    X_co_sim = np.column_stack([const_diff_sim, X_diff_sim])
    co_sim = sm.OLS(Y_diff_sim, X_co_sim).fit()
    
    # Confidence interval for original scale coefficient
    ci_co = co_sim.conf_int()[1]
    if ci_co[0] <= true_beta <= ci_co[1]:
        coverage_co += 1

coverage_ols /= n_sim
coverage_hac /= n_sim
coverage_co /= n_sim

print(f"\n95% Confidence Interval Coverage (should be 0.95):")
print(f"  OLS (incorrect SE):  {coverage_ols:.3f}  {'✗' if coverage_ols < 0.90 else '✓'}")
print(f"  OLS + HAC:           {coverage_hac:.3f}  {'✓' if 0.94 <= coverage_hac <= 0.96 else '⚠'}")
print(f"  Cochrane-Orcutt:     {coverage_co:.3f}  {'✓' if 0.94 <= coverage_co <= 0.96 else '⚠'}")

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n1. Detection:")
print(f"   • Visual: Residual time plot shows persistence")
print(f"   • DW: {dw_stat:.2f} < 2 indicates positive autocorrelation")
print(f"   • BG test: p < 0.001, strong evidence")

print("\n2. Consequences of Ignoring:")
print(f"   • Coefficients: Unbiased but inefficient")
print(f"   • Standard errors: Underestimated by ~{(ols_hac.bse['X'] - ols_model.bse['X'])/ols_model.bse['X']*100:.1f}%")
print(f"   • Coverage rate: {coverage_ols:.1%} instead of 95%")

print("\n3. Correction Methods:")
print("   • HAC (Newey-West): Always safe, valid inference")
print("   • Cochrane-Orcutt: Efficiency gain if AR(1) structure correct")
print("   • Prais-Winsten: Better than C-O (retains first observation)")

print("\n4. Practical Recommendations:")
print("   • Time series data: Always check for autocorrelation")
print("   • Default: Use Newey-West HAC standard errors")
print("   • Strong AR pattern: Consider dynamic specification (lagged Y)")
print("   • Panel data: Use clustered SE by entity")
print("   • Model diagnostics: Check ACF of final residuals")
```

## 6. Challenge Round
When do autocorrelation corrections fail or mislead?
- **Lagged dependent variable**: Standard HAC/Cochrane-Orcutt inconsistent → Use GMM (Arellano-Bond)
- **Structural breaks**: Time-varying parameters mimic autocorrelation → Test for breaks first
- **Omitted variables**: Persistent omitted factors cause autocorrelation → Specify model correctly
- **Small samples (T<50)**: HAC SE have poor finite-sample properties → Use bootstrap or exact methods
- **Nonstationary series**: Unit roots not solved by AR correction → Need differencing or cointegration
- **Spatial correlation**: Cross-sectional dependence mistaken for time series → Use spatial methods

## 7. Key References
- [Newey & West (1987) - HAC Covariance Matrix Estimator](https://doi.org/10.2307/1913610)
- [Wooldridge - Introductory Econometrics (Ch 12)](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge)
- [Cochrane & Orcutt (1949) - Application of Least Squares Regression](https://doi.org/10.2307/2280400)

---
**Status:** Fundamental time series diagnostic | **Complements:** HAC Standard Errors, ARMA Models, Dynamic Specifications, Panel Data
