# Dynamic Panel Data Models

## 1. Concept Skeleton
**Definition:** Panel models with lagged dependent variable as regressor; FE/RE inconsistent; GMM estimation addresses Nickell bias  
**Purpose:** Model persistence, dynamics, adjustment processes; identify short-run vs long-run effects; handle endogenous feedback  
**Prerequisites:** Panel data, GMM estimation, instrumental variables, moment conditions, weak instruments

## 2. Comparative Framing
| Method | Arellano-Bond (Diff GMM) | Arellano-Bover/Blundell-Bond (Sys GMM) | Fixed Effects | Anderson-Hsiao |
|--------|--------------------------|----------------------------------------|---------------|----------------|
| **Transformation** | First differences | Levels + differences | Within demeaning | First differences |
| **Instruments** | Lagged levels (t-2, t-3,...) | Lagged levels + lagged diffs | None | Lagged levels (t-2) |
| **Efficiency** | Good for persistent series | Most efficient | Biased (Nickell) | Inefficient |
| **Weak Instruments** | Problem if T large or ρ→1 | More robust | N/A | Severe |

## 3. Examples + Counterexamples

**Classic Example:**  
Employment dynamics: Nᵢₜ = ρNᵢₜ₋₁ + βWageᵢₜ + αᵢ + εᵢₜ. Persistence ρ captures adjustment costs. System GMM with weak instrument robust test.

**Failure Case:**  
Short panel (T=3): Insufficient lags for valid instruments. Difference GMM has weak instruments. System GMM performs better.

**Edge Case:**  
Near unit root (ρ≈1): Difference GMM instruments very weak. Level equation in System GMM provides identifying power.

## 4. Layer Breakdown
```
Dynamic Panel Data Framework:
├─ Model Specification:
│   Yᵢₜ = ρYᵢₜ₋₁ + Xᵢₜ'β + αᵢ + εᵢₜ
│   ├─ i = 1,...,N units; t = 2,...,T periods (lose t=1 for lag)
│   ├─ |ρ| < 1: Stationarity condition
│   ├─ ρ: Persistence parameter (key interest)
│   ├─ Long-run effect: β/(1-ρ) if ρ < 1
│   └─ Problem: Yᵢₜ₋₁ correlated with αᵢ → OLS/FE/RE inconsistent
├─ Nickell Bias:
│   ├─ FE with Yᵢₜ₋₁: Bias of order 1/T
│   ├─ Direction: Downward bias on ρ
│   ├─ plim(ρ̂_FE) ≈ ρ - (1+ρ)/T  (large N, fixed T)
│   └─ Does not vanish as N→∞ unless T→∞
├─ Difference GMM (Arellano-Bond, 1991):
│   ├─ Transformation: ΔYᵢₜ = ρΔYᵢₜ₋₁ + ΔXᵢₜ'β + Δεᵢₜ
│   │   └─ First differences eliminate αᵢ
│   ├─ Problem: ΔYᵢₜ₋₁ correlated with Δεᵢₜ
│   │   └─ Cov(ΔYᵢₜ₋₁, Δεᵢₜ) = Cov(Yᵢₜ₋₁ - Yᵢₜ₋₂, εᵢₜ - εᵢₜ₋₁) ≠ 0
│   ├─ Instruments: Yᵢₜ₋₂, Yᵢₜ₋₃,... (in levels)
│   │   ├─ Valid if E[Yᵢₜ₋ₛ Δεᵢₜ] = 0 for s ≥ 2
│   │   └─ Requires no serial correlation in εᵢₜ
│   ├─ Moment Conditions:
│   │   E[Yᵢₜ₋ₛ(ΔYᵢₜ - ρΔYᵢₜ₋₁ - ΔXᵢₜ'β)] = 0, s ≥ 2, t = 3,...,T
│   └─ Two-Step GMM: Efficient weighting matrix
├─ System GMM (Arellano-Bover, Blundell-Bond, 1998):
│   ├─ Combines: Differenced equations + level equations
│   ├─ Level Equation: Yᵢₜ = ρYᵢₜ₋₁ + Xᵢₜ'β + αᵢ + εᵢₜ
│   │   └─ Instruments: ΔYᵢₜ₋₁, ΔYᵢₜ₋₂,... (lagged differences)
│   ├─ Additional Moment Conditions:
│   │   E[ΔYᵢₜ₋ₛ(Yᵢₜ - ρYᵢₜ₋₁ - Xᵢₜ'β - αᵢ)] = 0, s ≥ 1
│   ├─ Assumption: E[Δαᵢ ΔYᵢₜ] = 0
│   │   └─ Changes in αᵢ uncorrelated with changes in Y
│   └─ Advantage: More instruments, more efficient, robust to weak IVs
├─ Strict vs Predetermined Variables:
│   ├─ Strictly Exogenous: E[Xᵢₜ εᵢₛ] = 0 for all s,t
│   │   └─ Can use all lags and leads as instruments
│   ├─ Predetermined: E[Xᵢₜ εᵢₛ] = 0 for s ≥ t only
│   │   └─ Use lags t-2, t-3,... as instruments
│   └─ Endogenous: Contemporaneous correlation
│       └─ Require lags t-2, t-3,... similar to Yᵢₜ₋₁
├─ Estimation Steps:
│   ├─ One-Step GMM: Use initial weight matrix (robust to heteroskedasticity)
│   ├─ Two-Step GMM: Update weight matrix with residuals from one-step
│   │   └─ More efficient but downward bias in SE (use finite-sample correction)
│   └─ Standard Errors: Robust, Windmeijer correction for two-step
├─ Specification Tests:
│   ├─ Hansen/Sargan J-Test: Overidentifying restrictions
│   │   ├─ H₀: Instruments valid (moment conditions hold)
│   │   ├─ J ~ χ²(# moments - # parameters)
│   │   └─ Warning: Weak if many instruments (instrument proliferation)
│   ├─ Arellano-Bond AR(1) Test: First-order serial correlation in Δεᵢₜ
│   │   └─ Expected to reject (Δεᵢₜ = εᵢₜ - εᵢₜ₋₁ creates AR(1))
│   ├─ Arellano-Bond AR(2) Test: Second-order serial correlation in Δεᵢₜ
│   │   ├─ Should NOT reject if εᵢₜ no serial correlation
│   │   └─ Rejection invalidates instruments Yᵢₜ₋₂
│   └─ Difference-in-Sargan Test: Validity of level instruments (System GMM)
├─ Instrument Proliferation:
│   ├─ Problem: Many instruments (increases with T) → overfitting, weak Hansen test
│   ├─ Solutions:
│   │   ├─ Collapse instruments: Use ΣₛYᵢₜ₋ₛ instead of separate instruments
│   │   ├─ Limit lags: Use only t-2, t-3 lags, not all available
│   │   └─ Rule of thumb: # instruments ≤ N (preferably much less)
│   └─ Roodman (2009): Xtabond2 implementation in Stata
├─ Parameter Bounds:
│   ├─ Consistency Check: ρ̂_OLS < ρ̂_GMM < ρ̂_FE
│   │   └─ OLS upward bias, FE downward bias, GMM should be between
│   ├─ Bond (2002): Use pooled OLS and FE as bounds
│   └─ If GMM outside bounds: Misspecification or weak instruments
└─ Extensions:
    ├─ Nonlinear GMM: Logit, Probit with lagged dependent variable
    ├─ Spatial Dynamic Panels: Spatial lag + time lag
    ├─ Bias-Corrected FE: Analytical corrections for small T
    └─ LSDVC (Least Squares Dummy Variable Corrected): Bruno (2005)
```

**Interaction:** First difference → Use lagged levels as IV → GMM estimation → AR(2) test for validity

## 5. Mini-Project
Estimate dynamic panel model with Arellano-Bond and System GMM:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
from scipy import stats

np.random.seed(789)

# ===== Simulate Dynamic Panel Data =====
N = 200  # Number of firms
T = 8    # Time periods
rho_true = 0.6  # Persistence parameter
beta_true = 0.8  # Effect of X

# Individual fixed effects
alpha_i = np.random.normal(0, 2, N)

# Initialize
data_dict = {'id': [], 'time': [], 'y': [], 'x': [], 'y_lag': [], 'alpha_true': []}

for i in range(N):
    # Initial condition
    y_initial = alpha_i[i] / (1 - rho_true) + np.random.normal(0, 1)
    
    y_values = [y_initial]
    
    for t in range(T):
        # X is predetermined (uncorrelated with current epsilon)
        x_t = 3 + 0.1 * t + np.random.normal(0, 1)
        
        # Generate Y with dynamics
        epsilon = np.random.normal(0, 1)
        y_t = rho_true * y_values[-1] + beta_true * x_t + alpha_i[i] + epsilon
        
        y_values.append(y_t)
        
        data_dict['id'].append(i)
        data_dict['time'].append(t)
        data_dict['y'].append(y_t)
        data_dict['x'].append(x_t)
        data_dict['y_lag'].append(y_values[-2])  # Lagged Y
        data_dict['alpha_true'].append(alpha_i[i])

df = pd.DataFrame(data_dict)

print("="*80)
print("DYNAMIC PANEL DATA MODELS: GMM ESTIMATION")
print("="*80)
print(f"\nSimulation Parameters:")
print(f"  Units (N): {N}")
print(f"  Time Periods (T): {T}")
print(f"  True ρ (persistence): {rho_true}")
print(f"  True β (X effect): {beta_true}")
print(f"  Long-run effect: β/(1-ρ) = {beta_true/(1-rho_true):.3f}")

print(f"\nDescriptive Statistics:")
print(df[['y', 'y_lag', 'x']].describe().round(3))

# ===== Pooled OLS (Upward Bias) =====
print("\n" + "="*80)
print("POOLED OLS (Upward Biased)")
print("="*80)

X_ols = sm.add_constant(df[['y_lag', 'x']])
ols_model = sm.OLS(df['y'], X_ols).fit()
print(ols_model.summary())

rho_ols = ols_model.params['y_lag']
print(f"\nPooled OLS ρ: {rho_ols:.4f} (True: {rho_true})")
print(f"Bias: {rho_ols - rho_true:.4f} (expected positive)")

# ===== Fixed Effects (Downward Bias - Nickell Bias) =====
print("\n" + "="*80)
print("FIXED EFFECTS (Nickell Bias - Downward)")
print("="*80)

df_panel = df.set_index(['id', 'time'])
fe_model = PanelOLS(df_panel['y'], df_panel[['y_lag', 'x']], 
                    entity_effects=True).fit()
print(fe_model.summary)

rho_fe = fe_model.params['y_lag']
print(f"\nFixed Effects ρ: {rho_fe:.4f} (True: {rho_true})")
print(f"Bias: {rho_fe - rho_true:.4f} (expected negative)")
print(f"Approximate Nickell bias: -{(1 + rho_true)/T:.4f}")

# ===== Anderson-Hsiao Estimator (First Differences, IV) =====
print("\n" + "="*80)
print("ANDERSON-HSIAO IV ESTIMATOR")
print("="*80)

# First differences
df['dy'] = df.groupby('id')['y'].diff()
df['dy_lag'] = df.groupby('id')['y_lag'].diff()
df['dx'] = df.groupby('id')['x'].diff()
df['y_lag2'] = df.groupby('id')['y_lag'].shift(1)  # Instrument: Y_{t-2}

# Drop missing values
df_ah = df.dropna(subset=['dy', 'dy_lag', 'dx', 'y_lag2'])

print(f"Observations after differencing: {len(df_ah)}")

# Manual IV regression
X_ah = df_ah[['dy_lag', 'dx']].values
Z_ah = df_ah[['y_lag2', 'dx']].values  # Instruments
y_ah = df_ah['dy'].values

# Two-stage least squares
# Stage 1: Regress endogenous on instruments
X1_stage1 = np.column_stack([Z_ah])
beta_stage1 = np.linalg.lstsq(Z_ah, X_ah[:, 0], rcond=None)[0]
dy_lag_hat = Z_ah @ beta_stage1.reshape(-1, 1)

# Stage 2: Regress Y on predicted X
X_stage2 = np.column_stack([dy_lag_hat.flatten(), X_ah[:, 1]])
beta_ah = np.linalg.lstsq(X_stage2, y_ah, rcond=None)[0]

rho_ah = beta_ah[0]
beta_ah_x = beta_ah[1]

print(f"Anderson-Hsiao IV Estimates:")
print(f"  ρ: {rho_ah:.4f} (True: {rho_true})")
print(f"  β: {beta_ah_x:.4f} (True: {beta_true})")

# ===== Difference GMM (Arellano-Bond) - Conceptual =====
print("\n" + "="*80)
print("DIFFERENCE GMM (ARELLANO-BOND)")
print("="*80)
print("Note: Full implementation requires specialized libraries")
print("      (e.g., pydynpd, or use Stata/R)")

# Simplified two-step approach with multiple instruments
# Instruments: Y_{t-2}, Y_{t-3} for t >= 3

# Create instrument matrix (simplified)
df['y_lag2'] = df.groupby('id')['y'].shift(2)
df['y_lag3'] = df.groupby('id')['y'].shift(3)

df_gmm = df.dropna(subset=['dy', 'dy_lag', 'dx', 'y_lag2', 'y_lag3'])

print(f"Observations with instruments (t >= 4): {len(df_gmm)}")

# Use multiple instruments
Z_gmm = df_gmm[['y_lag2', 'y_lag3', 'dx']].values
X_gmm = df_gmm[['dy_lag', 'dx']].values
y_gmm = df_gmm['dy'].values

# One-step GMM (simplified - should use proper GMM weighting)
# Stage 1
beta_stage1_gmm = np.linalg.lstsq(Z_gmm, X_gmm[:, 0], rcond=None)[0]
dy_lag_hat_gmm = Z_gmm @ beta_stage1_gmm

# Stage 2
X_stage2_gmm = np.column_stack([dy_lag_hat_gmm, X_gmm[:, 1]])
beta_diff_gmm = np.linalg.lstsq(X_stage2_gmm, y_gmm, rcond=None)[0]

rho_diff_gmm = beta_diff_gmm[0]
beta_diff_gmm_x = beta_diff_gmm[1]

print(f"Difference GMM Estimates (simplified):")
print(f"  ρ: {rho_diff_gmm:.4f} (True: {rho_true})")
print(f"  β: {beta_diff_gmm_x:.4f} (True: {beta_true})")

# ===== Parameter Bounds Check =====
print("\n" + "="*80)
print("CONSISTENCY CHECK: PARAMETER BOUNDS")
print("="*80)

print(f"Expected ordering: ρ̂_OLS > ρ̂_TRUE > ρ̂_FE")
print(f"\nActual estimates:")
print(f"  Pooled OLS:       {rho_ols:.4f}  {'✓' if rho_ols > rho_true else '✗'}")
print(f"  True:             {rho_true:.4f}")
print(f"  Fixed Effects:    {rho_fe:.4f}  {'✓' if rho_fe < rho_true else '✗'}")
print(f"  Anderson-Hsiao:   {rho_ah:.4f}  {'✓' if rho_fe < rho_ah < rho_ols else '⚠'}")
print(f"  Diff GMM (simpl): {rho_diff_gmm:.4f}  {'✓' if rho_fe < rho_diff_gmm < rho_ols else '⚠'}")

# ===== Arellano-Bond AR Tests (Conceptual) =====
print("\n" + "="*80)
print("ARELLANO-BOND SERIAL CORRELATION TESTS")
print("="*80)

# Compute differenced residuals (from simplified GMM)
resid_diff = y_gmm - X_stage2_gmm @ beta_diff_gmm

# Group residuals by id
df_gmm_temp = df_gmm.copy()
df_gmm_temp['resid_diff'] = resid_diff

# AR(1) test in differences (expected to reject)
df_gmm_temp['resid_diff_lag'] = df_gmm_temp.groupby('id')['resid_diff'].shift(1)
ar1_data = df_gmm_temp.dropna(subset=['resid_diff', 'resid_diff_lag'])

if len(ar1_data) > 0:
    ar1_corr = ar1_data['resid_diff'].corr(ar1_data['resid_diff_lag'])
    n_ar1 = len(ar1_data)
    ar1_stat = ar1_corr * np.sqrt(n_ar1)
    ar1_pval = 2 * (1 - stats.norm.cdf(np.abs(ar1_stat)))
    
    print(f"AR(1) Test in First Differences:")
    print(f"  Correlation: {ar1_corr:.4f}")
    print(f"  z-statistic: {ar1_stat:.4f}")
    print(f"  p-value: {ar1_pval:.4f}")
    print(f"  {'✓ Reject (expected)' if ar1_pval < 0.05 else '⚠ Fail to reject'}")

# AR(2) test (should NOT reject)
df_gmm_temp['resid_diff_lag2'] = df_gmm_temp.groupby('id')['resid_diff'].shift(2)
ar2_data = df_gmm_temp.dropna(subset=['resid_diff', 'resid_diff_lag2'])

if len(ar2_data) > 0:
    ar2_corr = ar2_data['resid_diff'].corr(ar2_data['resid_diff_lag2'])
    n_ar2 = len(ar2_data)
    ar2_stat = ar2_corr * np.sqrt(n_ar2)
    ar2_pval = 2 * (1 - stats.norm.cdf(np.abs(ar2_stat)))
    
    print(f"\nAR(2) Test in First Differences:")
    print(f"  Correlation: {ar2_corr:.4f}")
    print(f"  z-statistic: {ar2_stat:.4f}")
    print(f"  p-value: {ar2_pval:.4f}")
    print(f"  {'✓ Fail to reject (valid instruments)' if ar2_pval >= 0.05 else '✗ Reject (invalid)'}")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Coefficient Comparison
methods = ['True', 'OLS', 'FE', 'AH-IV', 'Diff GMM']
rho_estimates = [rho_true, rho_ols, rho_fe, rho_ah, rho_diff_gmm]

axes[0, 0].bar(methods, rho_estimates, alpha=0.8, 
              color=['red', 'blue', 'green', 'orange', 'purple'])
axes[0, 0].axhline(rho_true, color='red', linestyle='--', linewidth=2, label='True ρ')
axes[0, 0].set_ylabel('ρ (Persistence)')
axes[0, 0].set_title('Persistence Parameter Estimates')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')
axes[0, 0].set_xticklabels(methods, rotation=45)

# Plot 2: Bias Comparison
biases = [0, rho_ols - rho_true, rho_fe - rho_true, 
          rho_ah - rho_true, rho_diff_gmm - rho_true]

colors_bias = ['red' if b > 0 else 'blue' for b in biases]
axes[0, 1].bar(methods, biases, alpha=0.8, color=colors_bias)
axes[0, 1].axhline(0, color='black', linewidth=1)
axes[0, 1].set_ylabel('Bias (Estimate - True)')
axes[0, 1].set_title('Estimation Bias by Method')
axes[0, 1].grid(alpha=0.3, axis='y')
axes[0, 1].set_xticklabels(methods, rotation=45)

# Plot 3: Time Series for Selected Units
selected_ids = [0, 5, 10, 15]
for unit_id in selected_ids:
    unit_data = df[df['id'] == unit_id].sort_values('time')
    axes[0, 2].plot(unit_data['time'], unit_data['y'], 
                   marker='o', linewidth=1.5, markersize=5,
                   alpha=0.7, label=f'Unit {unit_id}')

axes[0, 2].set_xlabel('Time')
axes[0, 2].set_ylabel('Y')
axes[0, 2].set_title('Dynamic Paths: Selected Units')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(alpha=0.3)

# Plot 4: Y vs Y_lag Scatter
axes[1, 0].scatter(df['y_lag'], df['y'], alpha=0.3, s=10)

# Add OLS and FE fit lines
y_lag_range = np.linspace(df['y_lag'].min(), df['y_lag'].max(), 100)
axes[1, 0].plot(y_lag_range, rho_ols * y_lag_range + ols_model.params['const'],
               'b-', linewidth=2, label=f'OLS: {rho_ols:.3f}')
axes[1, 0].plot(y_lag_range, rho_true * y_lag_range,
               'r--', linewidth=2, label=f'True: {rho_true:.3f}')

axes[1, 0].set_xlabel('Y_{t-1}')
axes[1, 0].set_ylabel('Y_t')
axes[1, 0].set_title('Persistence: Y vs Lagged Y')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Distribution of Individual Effects
axes[1, 1].hist(alpha_i, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Individual Effect (αᵢ)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Distribution of Fixed Effects\n(Mean: {alpha_i.mean():.2f}, SD: {alpha_i.std():.2f})')
axes[1, 1].grid(alpha=0.3, axis='y')

# Plot 6: Residual Autocorrelation
if len(ar1_data) > 0:
    lag_range = range(1, 6)
    autocorrs = []
    
    for lag in lag_range:
        df_gmm_temp[f'resid_lag{lag}'] = df_gmm_temp.groupby('id')['resid_diff'].shift(lag)
        temp_data = df_gmm_temp.dropna(subset=['resid_diff', f'resid_lag{lag}'])
        if len(temp_data) > 0:
            autocorrs.append(temp_data['resid_diff'].corr(temp_data[f'resid_lag{lag}']))
        else:
            autocorrs.append(0)
    
    axes[1, 2].bar(lag_range, autocorrs, alpha=0.8)
    axes[1, 2].axhline(0, color='black', linewidth=1)
    axes[1, 2].axhline(1.96/np.sqrt(len(ar1_data)), color='red', 
                      linestyle='--', linewidth=1, label='95% CI')
    axes[1, 2].axhline(-1.96/np.sqrt(len(ar1_data)), color='red', 
                      linestyle='--', linewidth=1)
    axes[1, 2].set_xlabel('Lag')
    axes[1, 2].set_ylabel('Autocorrelation')
    axes[1, 2].set_title('Residual Autocorrelation (Diff GMM)')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND PRACTICAL RECOMMENDATIONS")
print("="*80)

print("\n1. Bias Patterns:")
print(f"   • Pooled OLS: {rho_ols - rho_true:+.4f} (upward bias from omitting αᵢ)")
print(f"   • Fixed Effects: {rho_fe - rho_true:+.4f} (Nickell bias, order 1/T={1/T:.3f})")
print(f"   • GMM methods closer to true value")

print("\n2. When to Use Each Method:")
print("   • N small, T large (T > 20): FE bias negligible")
print("   • N large, T small (T < 10): Use GMM (Diff or System)")
print("   • Persistent series (ρ > 0.8): System GMM preferred (Diff GMM weak IV)")
print("   • T = 2,3: Very limited, System GMM or bias-corrected FE")

print("\n3. GMM Implementation:")
print("   • Use specialized software: Stata (xtabond2), R (plm), Python (pydynpd)")
print("   • Two-step GMM with Windmeijer correction")
print("   • Limit instruments to avoid proliferation (# instruments < N)")
print("   • Collapse instrument matrix if T large")

print("\n4. Diagnostic Checks:")
print("   • AR(2) test: Should NOT reject (validates instruments)")
print("   • Hansen J-test: Should NOT reject (overidentifying restrictions)")
print("   • Parameter bounds: ρ̂_FE < ρ̂_GMM < ρ̂_OLS")
print("   • Number of instruments: Should be less than N")

print(f"\n5. Long-Run Effects:")
print(f"   • Short-run β: {beta_true:.3f}")
print(f"   • Long-run β/(1-ρ): {beta_true/(1-rho_true):.3f}")
print(f"   • Persistence half-life: ln(0.5)/ln({rho_true}) = {np.log(0.5)/np.log(rho_true):.2f} periods")

print("\n6. Practical Tools:")
print("   • Python: pydynpd package")
print("   • Stata: xtabond2 command (Roodman, 2009)")
print("   • R: plm package with pgmm() function")
```

## 6. Challenge Round
When do dynamic panel GMM estimators struggle?
- **Short panels (T=2,3)**: Too few lags for valid instruments → Use bias-corrected LSDV or System GMM with caution
- **Near unit root (ρ≈1)**: Difference GMM instruments very weak → System GMM with additional assumptions
- **Instrument proliferation**: T large → Many instruments → Overfitting, weak Hansen test → Collapse/limit instruments
- **Serial correlation in εᵢₜ**: AR(2) test rejects → Instruments invalid → Need longer lags, model specification issue
- **Unbalanced panels**: Complicated instrument structure → Use specialized software
- **Measurement error**: Exacerbated in differences → Consider System GMM or alternative estimators

## 7. Key References
- [Arellano & Bond (1991) - Some Tests of Specification for Panel Data](https://doi.org/10.2307/2297968)
- [Blundell & Bond (1998) - Initial Conditions and Moment Restrictions](https://doi.org/10.1016/S0304-4076(98)00009-8)
- [Roodman (2009) - How to Do xtabond2](https://doi.org/10.1177/1536867X0900900106)

---
**Status:** Advanced panel data method | **Complements:** Fixed Effects, GMM Estimation, Weak Instruments, Serial Correlation
