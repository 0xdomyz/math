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

# 1. Stationary AR(1) process (Ï = 0.7)
ar_stationary = np.zeros(n)
for t in range(1, n):
    ar_stationary[t] = 0.7 * ar_stationary[t-1] + np.random.normal(0, 1)

# 2. Random Walk (unit root, Ï = 1)
random_walk = np.cumsum(np.random.normal(0, 1, n))

# 3. Random Walk with Drift
drift = 0.1
rw_drift = drift * np.arange(n) + np.cumsum(np.random.normal(0, 1, n))

# 4. Trend Stationary
trend_stationary = 0.5 * np.arange(n) + np.random.normal(0, 5, n)

# 5. Near Unit Root (Ï = 0.98)
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
        print(f"  âœ“ Reject Hâ‚€ (unit root): Series is STATIONARY")
    else:
        print(f"  âœ— Fail to reject Hâ‚€: Series has UNIT ROOT (non-stationary)")
    
    # KPSS Test (reversed null)
    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
        print("\nKPSS Test (Hâ‚€: Stationary):")
        print(f"  KPSS Statistic: {kpss_result[0]:.4f}")
        print(f"  p-value: {kpss_result[1]:.4f}")
        print(f"  Critical Values:")
        for key, value in kpss_result[3].items():
            print(f"    {key}: {value:.4f}")
        
        if kpss_result[1] < 0.05:
            print(f"  âœ— Reject Hâ‚€: Series is NON-STATIONARY")
        else:
            print(f"  âœ“ Fail to reject Hâ‚€: Series is STATIONARY")
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
    ('Stationary_AR1', 'Stationary AR(1), Ï=0.7'),
    ('Random_Walk', 'Random Walk (Unit Root)'),
    ('RW_Drift', 'Random Walk with Drift'),
    ('Trend_Stationary', 'Trend Stationary'),
    ('Near_Unit_Root', 'Near Unit Root (Ï=0.98)')
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
axes[1, 2].set_ylabel('Î”Y')
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
print(f"  RÂ²: {spurious_model.rsquared:.4f}")
print("\nâš ï¸  WARNING: High RÂ² and significant coefficient despite")
print("    no true relationship! This is SPURIOUS REGRESSION.")

# Now difference both series and re-run
diff_rw1 = np.diff(rw1)
diff_rw2 = np.diff(rw2)

X_diff = sm.add_constant(diff_rw1)
diff_model = sm.OLS(diff_rw2, X_diff).fit()

print("\nRegression of Î”RW2 on Î”RW1 (stationary differences):")
print(f"  Coefficient: {diff_model.params[1]:.4f}")
print(f"  t-statistic: {diff_model.tvalues[1]:.2f}")
print(f"  p-value: {diff_model.pvalues[1]:.4f}")
print(f"  RÂ²: {diff_model.rsquared:.4f}")
print("\nâœ“ After differencing: No spurious relationship found")

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

print("\nADF Test Rejection Rates (Hâ‚€: Unit Root):")
print(f"{'Ï':>6s}  {'Rejection Rate':>15s}  {'Interpretation':>20s}")
print("-" * 50)
for rho, rate in zip(rho_values, rejection_rates):
    interp = "Stationary" if rho < 1 else "Unit Root (Hâ‚€ true)"
    print(f"{rho:>6.2f}  {rate:>15.1%}  {interp:>20s}")

print("\nNote: Lower rejection rates for Ï near 1.0 show")
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
        decision = "Reject Hâ‚€" if adf_result[1] < 0.05 else "Fail to Reject"
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
