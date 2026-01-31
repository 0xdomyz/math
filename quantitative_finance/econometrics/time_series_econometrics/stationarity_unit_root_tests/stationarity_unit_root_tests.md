# Stationarity and Unit Root Tests

## 1. Concept Skeleton
**Definition:** Stationarity requires constant mean, variance, and autocovariances over time; unit root tests detect non-stationary processes  
**Purpose:** Ensure valid inference in time series models; avoid spurious regression; determine differencing requirements  
**Prerequisites:** Time series properties, autocovariance, random walk, asymptotic theory, trend vs stochastic components

## 2. Comparative Framing
| Method | ADF Test | Phillips-Perron | KPSS Test | DF-GLS |
|--------|----------|-----------------|-----------|--------|
| **Null Hypothesis** | Unit root (non-stationary) | Unit root | Stationary | Unit root |
| **Alternative** | Stationary | Stationary | Unit root (reversed!) | Stationary |
| **Autocorrelation** | Parametric (lags) | Non-parametric (HAC) | Non-parametric | GLS detrending |
| **Power** | Moderate | Low if MA errors | Complement to ADF | Higher than ADF |

## 3. Examples + Counterexamples

**Classic Example:**  
Stock prices: Random walk with drift (unit root). Returns: Stationary differences. ADF fails to reject unit root for prices, rejects for returns.

**Failure Case:**  
Structural break mistaken for unit root: GDP appears non-stationary, but break at recession creates false unit root. Use Zivot-Andrews test.

**Edge Case:**  
Near unit root (ρ=0.98): Large sample needed to distinguish from ρ=1. ADF has low power, may fail to reject even if stationary.

## 4. Layer Breakdown
```
Stationarity Framework:
├─ Types of Stationarity:
│   ├─ Strict Stationarity: Joint distribution invariant to time shift
│   ├─ Weak (Covariance) Stationarity:
│   │   ├─ E[Yₜ] = μ (constant mean)
│   │   ├─ Var(Yₜ) = σ² (constant variance)
│   │   └─ Cov(Yₜ, Yₜ₋ₖ) = γₖ (depends only on lag k)
│   └─ Trend Stationary: Yₜ = α + βt + εₜ, εₜ stationary
├─ Unit Root Process:
│   ├─ Random Walk: Yₜ = Yₜ₋₁ + εₜ
│   ├─ With Drift: Yₜ = α + Yₜ₋₁ + εₜ
│   ├─ General AR(1): Yₜ = ρYₜ₋₁ + εₜ
│   │   └─ Unit root if ρ = 1 (non-stationary)
│   └─ Variance: Var(Yₜ) = tσ² → ∞ as t → ∞
├─ Augmented Dickey-Fuller (ADF) Test:
│   ├─ Regression: ΔYₜ = α + βt + γYₜ₋₁ + Σφᵢ ΔYₜ₋ᵢ + εₜ
│   │   └─ H₀: γ = 0 (unit root), H₁: γ < 0 (stationary)
│   ├─ Test Statistic: t_γ = γ̂ / SE(γ̂)
│   │   └─ Non-standard distribution (Dickey-Fuller critical values)
│   ├─ Model Selection:
│   │   ├─ No constant/trend: Pure random walk
│   │   ├─ Constant only: Random walk with drift
│   │   └─ Constant + trend: Trend-stationary alternative
│   └─ Lag Selection: AIC, BIC, or t-test sequential procedure
├─ Phillips-Perron (PP) Test:
│   ├─ Non-parametric correction for serial correlation
│   ├─ Same regression as DF but adjust t-statistic
│   └─ Robust to heteroskedasticity and autocorrelation (HAC)
├─ KPSS Test:
│   ├─ H₀: Stationary (reversed null!)
│   ├─ Decomposition: Yₜ = μₜ + βt + εₜ, test if σ²_μ = 0
│   └─ Use with ADF: Both reject → ambiguous, both fail → near unit root
├─ Differencing:
│   ├─ First Difference: ΔYₜ = Yₜ - Yₜ₋₁ (I(1) → I(0))
│   ├─ Seasonal Difference: Yₜ - Yₜ₋ₛ (seasonal unit root)
│   └─ Over-differencing: Introduces MA component, reduces efficiency
└─ Consequences of Non-Stationarity:
    ├─ Spurious Regression: R² high, t-stats significant but meaningless
    ├─ Invalid Inference: Standard errors, t/F tests unreliable
    ├─ Trending Behavior: Mean/variance change over time
    └─ Solution: Difference, detrend, or use cointegration framework
```

**Interaction:** Test stationarity → If unit root, difference → Re-test → Model stationary series

## 5. Mini-Project
Test stationarity with ADF, PP, KPSS and visualize non-stationary behavior:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 500

# ===== Generate Time Series with Different Properties =====

# 1. Stationary AR(1) process (ρ = 0.7)
ar_stationary = np.zeros(n)
for t in range(1, n):
    ar_stationary[t] = 0.7 * ar_stationary[t-1] + np.random.normal(0, 1)

# 2. Random Walk (unit root, ρ = 1)
random_walk = np.cumsum(np.random.normal(0, 1, n))

# 3. Random Walk with Drift
drift = 0.1
rw_drift = drift * np.arange(n) + np.cumsum(np.random.normal(0, 1, n))

# 4. Trend Stationary
trend_stationary = 0.5 * np.arange(n) + np.random.normal(0, 5, n)

# 5. Near Unit Root (ρ = 0.98)
near_unit_root = np.zeros(n)
for t in range(1, n):
    near_unit_root[t] = 0.98 * near_unit_root[t-1] + np.random.normal(0, 1)

# Create DataFrame
df = pd.DataFrame({
    'Stationary_AR1': ar_stationary,
    'Random_Walk': random_walk,
    'RW_Drift': rw_drift,
    'Trend_Stationary': trend_stationary,
    'Near_Unit_Root': near_unit_root
})

print("="*70)
print("STATIONARITY AND UNIT ROOT TESTS")
print("="*70)

# ===== Function to Perform and Display Tests =====
def test_stationarity(series, name):
    """Perform ADF, PP-equivalent, and KPSS tests"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print('='*70)
    
    # Augmented Dickey-Fuller Test
    adf_result = adfuller(series, autolag='AIC')
    print("\nAugmented Dickey-Fuller Test:")
    print(f"  ADF Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    print(f"  Critical Values:")
    for key, value in adf_result[4].items():
        print(f"    {key}: {value:.4f}")
    
    if adf_result[1] < 0.05:
        print(f"  ✓ Reject H₀ (unit root): Series is STATIONARY")
    else:
        print(f"  ✗ Fail to reject H₀: Series has UNIT ROOT (non-stationary)")
    
    # KPSS Test (reversed null)
    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
        print("\nKPSS Test (H₀: Stationary):")
        print(f"  KPSS Statistic: {kpss_result[0]:.4f}")
        print(f"  p-value: {kpss_result[1]:.4f}")
        print(f"  Critical Values:")
        for key, value in kpss_result[3].items():
            print(f"    {key}: {value:.4f}")
        
        if kpss_result[1] < 0.05:
            print(f"  ✗ Reject H₀: Series is NON-STATIONARY")
        else:
            print(f"  ✓ Fail to reject H₀: Series is STATIONARY")
    except:
        print("\nKPSS Test: Failed (possible perfect fit)")
    
    # Summary
    print("\nConclusion:")
    if adf_result[1] < 0.05:
        print(f"  {name}: STATIONARY (ADF rejects unit root)")
    else:
        print(f"  {name}: NON-STATIONARY (ADF fails to reject unit root)")

# Test each series
for col in df.columns:
    test_stationarity(df[col], col)

# ===== Test on Differenced Series =====
print(f"\n{'='*70}")
print("TESTING FIRST DIFFERENCES")
print('='*70)

for col in ['Random_Walk', 'RW_Drift']:
    diff_series = df[col].diff().dropna()
    test_stationarity(diff_series, f"{col} (First Difference)")

# ===== Visualizations =====
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

# Plot 1-5: Time Series
series_list = [
    ('Stationary_AR1', 'Stationary AR(1), ρ=0.7'),
    ('Random_Walk', 'Random Walk (Unit Root)'),
    ('RW_Drift', 'Random Walk with Drift'),
    ('Trend_Stationary', 'Trend Stationary'),
    ('Near_Unit_Root', 'Near Unit Root (ρ=0.98)')
]

for idx, (col, title) in enumerate(series_list):
    row = idx // 3
    col_idx = idx % 3
    
    axes[row, col_idx].plot(df[col], linewidth=1)
    axes[row, col_idx].set_title(title, fontsize=10)
    axes[row, col_idx].set_xlabel('Time')
    axes[row, col_idx].set_ylabel('Value')
    axes[row, col_idx].grid(alpha=0.3)
    
    # Add horizontal line for stationary processes
    if col == 'Stationary_AR1':
        axes[row, col_idx].axhline(0, color='r', linestyle='--', 
                                   linewidth=1, alpha=0.5)

# Plot 6: First Difference of Random Walk
axes[1, 2].plot(df['Random_Walk'].diff(), linewidth=1, color='green')
axes[1, 2].set_title('Random Walk: First Difference (Stationary)', fontsize=10)
axes[1, 2].set_xlabel('Time')
axes[1, 2].set_ylabel('ΔY')
axes[1, 2].grid(alpha=0.3)
axes[1, 2].axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.5)

# Plot 7: ACF Comparison (Stationary vs Non-Stationary)
from statsmodels.tsa.stattools import acf

lags = 40
acf_stationary = acf(df['Stationary_AR1'], nlags=lags)
acf_rw = acf(df['Random_Walk'], nlags=lags)

axes[2, 0].bar(range(len(acf_stationary)), acf_stationary, 
               alpha=0.7, label='Stationary AR(1)')
axes[2, 0].set_title('ACF: Stationary Process', fontsize=10)
axes[2, 0].set_xlabel('Lag')
axes[2, 0].set_ylabel('Autocorrelation')
axes[2, 0].axhline(0, color='black', linewidth=0.5)
axes[2, 0].axhline(1.96/np.sqrt(n), color='r', linestyle='--', 
                   linewidth=1, label='95% CI')
axes[2, 0].axhline(-1.96/np.sqrt(n), color='r', linestyle='--', linewidth=1)
axes[2, 0].legend(fontsize=8)
axes[2, 0].grid(alpha=0.3, axis='y')

axes[2, 1].bar(range(len(acf_rw)), acf_rw, 
               alpha=0.7, color='orange', label='Random Walk')
axes[2, 1].set_title('ACF: Non-Stationary Process', fontsize=10)
axes[2, 1].set_xlabel('Lag')
axes[2, 1].set_ylabel('Autocorrelation')
axes[2, 1].axhline(0, color='black', linewidth=0.5)
axes[2, 1].axhline(1.96/np.sqrt(n), color='r', linestyle='--', linewidth=1)
axes[2, 1].axhline(-1.96/np.sqrt(n), color='r', linestyle='--', linewidth=1)
axes[2, 1].legend(fontsize=8)
axes[2, 1].grid(alpha=0.3, axis='y')

# Plot 8: Variance over Time (Rolling)
window = 50
rolling_var_stationary = df['Stationary_AR1'].rolling(window).var()
rolling_var_rw = df['Random_Walk'].rolling(window).var()

axes[2, 2].plot(rolling_var_stationary, label='Stationary AR(1)', linewidth=2)
axes[2, 2].plot(rolling_var_rw, label='Random Walk', linewidth=2)
axes[2, 2].set_title(f'Rolling Variance (window={window})', fontsize=10)
axes[2, 2].set_xlabel('Time')
axes[2, 2].set_ylabel('Variance')
axes[2, 2].legend(fontsize=8)
axes[2, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ===== Simulate Spurious Regression =====
print(f"\n{'='*70}")
print("SPURIOUS REGRESSION DEMONSTRATION")
print('='*70)

# Generate two independent random walks
rw1 = np.cumsum(np.random.normal(0, 1, n))
rw2 = np.cumsum(np.random.normal(0, 1, n))

# Regress one on the other (should find no relationship, but...)
import statsmodels.api as sm

X_spurious = sm.add_constant(rw1)
spurious_model = sm.OLS(rw2, X_spurious).fit()

print("\nRegression of RW2 on RW1 (independent random walks):")
print(f"  Coefficient: {spurious_model.params[1]:.4f}")
print(f"  t-statistic: {spurious_model.tvalues[1]:.2f}")
print(f"  p-value: {spurious_model.pvalues[1]:.4f}")
print(f"  R²: {spurious_model.rsquared:.4f}")
print("\n⚠️  WARNING: High R² and significant coefficient despite")
print("    no true relationship! This is SPURIOUS REGRESSION.")

# Now difference both series and re-run
diff_rw1 = np.diff(rw1)
diff_rw2 = np.diff(rw2)

X_diff = sm.add_constant(diff_rw1)
diff_model = sm.OLS(diff_rw2, X_diff).fit()

print("\nRegression of ΔRW2 on ΔRW1 (stationary differences):")
print(f"  Coefficient: {diff_model.params[1]:.4f}")
print(f"  t-statistic: {diff_model.tvalues[1]:.2f}")
print(f"  p-value: {diff_model.pvalues[1]:.4f}")
print(f"  R²: {diff_model.rsquared:.4f}")
print("\n✓ After differencing: No spurious relationship found")

# ===== Power Analysis: Near Unit Root =====
print(f"\n{'='*70}")
print("POWER ANALYSIS: NEAR UNIT ROOT")
print('='*70)

rho_values = [0.9, 0.95, 0.98, 0.99, 1.0]
n_sims = 100
rejection_rates = []

for rho in rho_values:
    rejections = 0
    for _ in range(n_sims):
        # Generate AR(1) with coefficient rho
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = rho * y[t-1] + np.random.normal(0, 1)
        
        # ADF test
        try:
            adf_result = adfuller(y, autolag='AIC')
            if adf_result[1] < 0.05:
                rejections += 1
        except:
            pass
    
    rejection_rates.append(rejections / n_sims)

print("\nADF Test Rejection Rates (H₀: Unit Root):")
print(f"{'ρ':>6s}  {'Rejection Rate':>15s}  {'Interpretation':>20s}")
print("-" * 50)
for rho, rate in zip(rho_values, rejection_rates):
    interp = "Stationary" if rho < 1 else "Unit Root (H₀ true)"
    print(f"{rho:>6.2f}  {rate:>15.1%}  {interp:>20s}")

print("\nNote: Lower rejection rates for ρ near 1.0 show")
print("      reduced power to detect stationarity.")

# ===== Lag Selection for ADF =====
print(f"\n{'='*70}")
print("LAG SELECTION FOR ADF TEST")
print('='*70)

# Test with different lag specifications
test_series = df['Near_Unit_Root']
max_lags = 10

print(f"\nTesting '{test_series.name}' with different lag lengths:")
print(f"{'Lags':>6s}  {'ADF Stat':>10s}  {'p-value':>10s}  {'Decision':>15s}")
print("-" * 50)

for lag in range(0, max_lags + 1):
    try:
        adf_result = adfuller(test_series, maxlag=lag, autolag=None)
        decision = "Reject H₀" if adf_result[1] < 0.05 else "Fail to Reject"
        print(f"{lag:>6d}  {adf_result[0]:>10.4f}  {adf_result[1]:>10.4f}  {decision:>15s}")
    except:
        print(f"{lag:>6d}  {'N/A':>10s}  {'N/A':>10s}  {'Error':>15s}")

print("\nRecommendation: Use AIC/BIC for automatic lag selection")

# ===== Summary Table =====
print(f"\n{'='*70}")
print("SUMMARY: STATIONARITY PROPERTIES")
print('='*70)

summary_data = []
for col in df.columns:
    adf = adfuller(df[col], autolag='AIC')
    try:
        kpss_test = kpss(df[col], regression='c', nlags='auto')
        kpss_stat = kpss_test[0]
        kpss_pval = kpss_test[1]
    except:
        kpss_stat = np.nan
        kpss_pval = np.nan
    
    summary_data.append({
        'Series': col,
        'ADF Stat': adf[0],
        'ADF p-val': adf[1],
        'KPSS Stat': kpss_stat,
        'KPSS p-val': kpss_pval,
        'Conclusion': 'Stationary' if adf[1] < 0.05 else 'Unit Root'
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False, float_format='%.4f'))
```

## 6. Challenge Round
When do unit root tests mislead or fail?
- **Structural breaks**: Trend break mimics unit root → Use Zivot-Andrews, Perron tests with break
- **Low power near unit root**: ρ=0.98 vs ρ=1.0 indistinguishable in small samples → Need very large n
- **Over-differencing**: Differencing stationary series introduces MA component, reduces efficiency
- **Seasonal unit roots**: Standard ADF misses seasonal non-stationarity → Use seasonal unit root tests
- **Nonlinear trends**: Exponential/logistic growth not captured by linear trend → Transform data first
- **KPSS vs ADF conflict**: Both reject or both fail to reject → Ambiguous, near-integrated process

## 7. Key References
- [Dickey & Fuller (1979) - Distribution of Estimators for AR Time Series with Unit Root](https://doi.org/10.1080/01621459.1979.10482531)
- [Phillips & Perron (1988) - Testing for Unit Roots in Time Series Regression](https://doi.org/10.1093/biomet/75.2.335)
- [Kwiatkowski et al. (1992) - Testing the Null Hypothesis of Stationarity (KPSS)](https://doi.org/10.1016/0304-4076(92)90104-Y)

---
**Status:** Foundational time series diagnostic | **Complements:** ARIMA, Cointegration, Detrending, Differencing
