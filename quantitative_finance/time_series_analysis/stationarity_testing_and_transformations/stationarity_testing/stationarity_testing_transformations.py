from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

class Transformations:
    """Transformations to achieve stationarity"""
    
    def __init__(self):
        pass
    
    def difference(self, y, order=1, seasonal=False, period=12):
        """
        Differencing transformation
        order: Number of times to difference
        seasonal: Apply seasonal differencing
        """
        result = y.copy()
        
        for _ in range(order):
            result = np.diff(result)
        
        if seasonal:
            if len(result) > period:
                result = result[period:] - result[:-period]
        
        return result
    
    def log_transform(self, y, shift=0):
        """
        Log transformation
        shift: Add constant if y has zeros or negatives
        """
        y_shifted = y + shift
        
        if np.any(y_shifted <= 0):
            raise ValueError("Cannot take log of non-positive values")
        
        return np.log(y_shifted)
    
    def box_cox(self, y, lambda_=None):
        """
        Box-Cox transformation
        If lambda_ is None, estimate it via MLE
        """
        if np.any(y <= 0):
            raise ValueError("Box-Cox requires positive data")
        
        if lambda_ is None:
            # Estimate lambda via profile likelihood
            lambda_ = self.estimate_box_cox_lambda(y)
        
        if abs(lambda_) < 1e-10:
            return np.log(y), lambda_
        else:
            return (y**lambda_ - 1) / lambda_, lambda_
    
    def estimate_box_cox_lambda(self, y, lambda_range=(-2, 2)):
        """Estimate Box-Cox lambda via maximum likelihood"""
        
        def neg_log_likelihood(lambda_, y):
            """Negative log-likelihood for Box-Cox"""
            n = len(y)
            
            if abs(lambda_) < 1e-10:
                y_trans = np.log(y)
            else:
                y_trans = (y**lambda_ - 1) / lambda_
            
            # Add Jacobian term
            jacobian = (lambda_ - 1) * np.sum(np.log(y))
            
            # Variance of transformed series
            sigma2 = np.var(y_trans, ddof=1)
            
            # Log-likelihood
            ll = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sigma2) - 0.5 * n + jacobian
            
            return -ll
        
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=lambda_range,
            args=(y,),
            method='bounded'
        )
        
        return result.x
    
    def detrend_linear(self, y):
        """Remove linear trend via OLS"""
        t = np.arange(len(y))
        X = np.column_stack([np.ones(len(y)), t])
        model = OLS(y, X).fit()
        
        trend = model.predict(X)
        detrended = y - trend
        
        return detrended, model.params
    
    def hp_filter(self, y, lamb=1600):
        """
        Hodrick-Prescott filter
        lamb: Smoothing parameter (1600 for quarterly, 100 for annual)
        """
        n = len(y)
        
        # Construct second-difference matrix K
        # (1, -2, 1) pattern for second differences
        K = np.zeros((n-2, n))
        for i in range(n-2):
            K[i, i:i+3] = [1, -2, 1]
        
        # Solve (I + λK'K)τ = y
        I = np.eye(n)
        A = I + lamb * K.T @ K
        trend = np.linalg.solve(A, y)
        cycle = y - trend
        
        return trend, cycle
    
    def hp_filter_sparse(self, y, lamb=1600):
        """HP filter using sparse matrices (faster for large n)"""
        n = len(y)
        
        # Second difference matrix as sparse
        data = np.array([
            np.ones(n-2),
            -2 * np.ones(n-2),
            np.ones(n-2)
        ])
        offsets = [0, 1, 2]
        K = diags(data, offsets, shape=(n-2, n), format='csr')
        
        # Solve sparse system
        I = diags([1], [0], shape=(n, n), format='csr')
        A = I + lamb * (K.T @ K)
        trend = spsolve(A, y)
        cycle = y - trend
        
        return trend, cycle

# Generate test series
n = 500

# 1. White noise (stationary)
white_noise = np.random.normal(0, 1, n)

# 2. Random walk (unit root)
random_walk = np.cumsum(white_noise)

# 3. Random walk with drift
drift = 0.05
rw_drift = np.cumsum(white_noise + drift)

# 4. Trend-stationary
t = np.arange(n)
trend_stationary = 0.05 * t + white_noise

# 5. AR(1) with high persistence
ar1_high = np.zeros(n)
phi = 0.95
for i in range(1, n):
    ar1_high[i] = phi * ar1_high[i-1] + white_noise[i]

# Test all series
tester = StationarityTester()

print("\n" + "="*80)
print("SCENARIO 1: WHITE NOISE (Stationary Benchmark)")
print("="*80)
results_wn = tester.comprehensive_test(white_noise, "White Noise")

print("\n" + "="*80)
print("SCENARIO 2: RANDOM WALK (Unit Root)")
print("="*80)
results_rw = tester.comprehensive_test(random_walk, "Random Walk")

# Test differenced random walk
print("\n" + "="*80)
print("SCENARIO 2b: RANDOM WALK - FIRST DIFFERENCE")
print("="*80)
transformer = Transformations()
rw_diff = transformer.difference(random_walk, order=1)
results_rw_diff = tester.comprehensive_test(rw_diff, "Differenced Random Walk")

print("\n" + "="*80)
print("SCENARIO 3: TREND-STATIONARY PROCESS")
print("="*80)
results_ts = tester.comprehensive_test(trend_stationary, "Trend-Stationary")

# Detrend
trend_detrended, trend_params = transformer.detrend_linear(trend_stationary)
print(f"\nLinear detrending:")
print(f"  Slope: {trend_params[1]:.6f} (True: 0.05)")
print(f"  Intercept: {trend_params[0]:.4f}")

print("\n" + "="*80)
print("SCENARIO 3b: AFTER DETRENDING")
print("="*80)
results_detrended = tester.comprehensive_test(trend_detrended, "Detrended Series")

print("\n" + "="*80)
print("SCENARIO 4: AR(1) WITH HIGH PERSISTENCE (Near Unit Root)")
print("="*80)
results_ar1 = tester.comprehensive_test(ar1_high, f"AR(1) φ={phi}")

print("\n" + "="*80)
print("SCENARIO 5: VARIANCE RATIO TESTS")
print("="*80)

processes = {
    'White Noise': white_noise,
    'Random Walk': random_walk,
    'AR(1) φ=0.95': ar1_high
}

for name, series in processes.items():
    print(f"\n{name}:")
    vr_results = tester.variance_ratio_test(series, lags=[2, 4, 8, 16])
    print(f"  {'Lag':<8} {'VR':<10} {'Z-stat':<10} {'p-value':<10} {'Reject RW':<12}")
    print("  " + "-"*50)
    for vr in vr_results:
        print(f"  {vr['lag']:<8} {vr['VR']:<10.3f} {vr['z_stat']:<10.3f} {vr['p_value']:<10.4f} {str(vr['reject_rw']):<12}")

# Box-Cox transformation example
print("\n" + "="*80)
print("SCENARIO 6: BOX-COX TRANSFORMATION")
print("="*80)

# Generate exponentially growing series (needs variance stabilization)
exp_series = np.exp(0.01 * t + 0.1 * white_noise)

print(f"Original series statistics:")
print(f"  Mean: {np.mean(exp_series):.2f}")
print(f"  Std: {np.std(exp_series):.2f}")
print(f"  CV: {np.std(exp_series)/np.mean(exp_series):.4f}")

# Estimate Box-Cox lambda
lambda_hat = transformer.estimate_box_cox_lambda(exp_series)
print(f"\nEstimated λ: {lambda_hat:.4f}")
print(f"  λ=0 suggests log transformation")
print(f"  λ=0.5 suggests square root")
print(f"  λ=1 suggests no transformation")

# Apply transformation
exp_transformed, _ = transformer.box_cox(exp_series, lambda_=lambda_hat)
print(f"\nTransformed series statistics:")
print(f"  Mean: {np.mean(exp_transformed):.2f}")
print(f"  Std: {np.std(exp_transformed):.2f}")
print(f"  CV: {np.std(exp_transformed)/np.mean(exp_transformed):.4f} (more stable)")

# Hodrick-Prescott filter
print("\n" + "="*80)
print("SCENARIO 7: HODRICK-PRESCOTT FILTER")
print("="*80)

# Generate series with trend and cycle
cycle_component = 5 * np.sin(2 * np.pi * t / 40)
hp_series = 0.05 * t + cycle_component + 0.5 * white_noise

trend_hp, cycle_hp = transformer.hp_filter(hp_series, lamb=1600)

print(f"HP Filter (λ=1600):")
print(f"  Original variance: {np.var(hp_series):.2f}")
print(f"  Trend variance: {np.var(trend_hp):.2f}")
print(f"  Cycle variance: {np.var(cycle_hp):.2f}")
print(f"  Decomposition: Var(y) ≈ Var(trend) + Var(cycle) + 2·Cov")

# Structural break test (Zivot-Andrews)
print("\n" + "="*80)
print("SCENARIO 8: STRUCTURAL BREAK (Zivot-Andrews)")
print("="*80)

# Generate series with level shift at t=250
break_series = white_noise.copy()
break_series[250:] += 3  # Level shift
break_series_cumsum = np.cumsum(break_series)

print("Series with structural break:")
za_result = tester.zivot_andrews(break_series_cumsum, model='C', trim=0.15)
print(f"  ZA statistic: {za_result['za_stat']:.4f}")
print(f"  Critical value (5%): {za_result['critical_value']:.4f}")
print(f"  Break point detected: {za_result['break_point']}")
print(f"  True break point: 250")
print(f"  Stationary with break: {za_result['stationary']}")

# Standard ADF without break
print("\nStandard ADF (ignoring break):")
adf_no_break = tester.adf_test(break_series_cumsum)
print(f"  ADF p-value: {adf_no_break['p_value']:.4f}")
print(f"  Incorrectly concludes: {'Stationary' if adf_no_break['stationary'] else 'Non-stationary'}")

# Visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot 1: Random walk levels
ax = axes[0, 0]
ax.plot(random_walk, linewidth=1, alpha=0.7, label='Levels')
ax.set_title('Random Walk (Non-Stationary)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Random walk first difference
ax = axes[0, 1]
ax.plot(rw_diff, linewidth=0.8, alpha=0.7, label='First Difference')
ax.axhline(0, color='r', linestyle='--', alpha=0.5)
ax.set_title('Random Walk - First Difference (Stationary)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: ACF of levels vs differences
ax = axes[0, 2]
plot_acf(random_walk, lags=40, ax=ax, alpha=0.05, label='Levels')
ax.set_title('ACF: Random Walk Levels (Slow Decay)')
ax.grid(alpha=0.3)

# Plot 4: Trend-stationary
ax = axes[1, 0]
ax.plot(t, trend_stationary, 'b-', linewidth=1, alpha=0.7, label='Observed')
ax.plot(t, 0.05*t, 'r--', linewidth=2, label='True Trend')
ax.set_title('Trend-Stationary Process')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Detrended
ax = axes[1, 1]
ax.plot(trend_detrended, linewidth=0.8, alpha=0.7)
ax.axhline(0, color='r', linestyle='--', alpha=0.5)
ax.set_title('After Detrending (Stationary)')
ax.set_xlabel('Time')
ax.set_ylabel('Detrended Value')
ax.grid(alpha=0.3)

# Plot 6: AR(1) high persistence
ax = axes[1, 2]
ax.plot(ar1_high, linewidth=1, alpha=0.7)
ax.set_title(f'AR(1) φ={phi} (Near Unit Root)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.grid(alpha=0.3)

# Plot 7: Box-Cox original vs transformed
ax = axes[2, 0]
ax2 = ax.twinx()
ax.plot(exp_series, 'b-', linewidth=1, alpha=0.7, label='Original')
ax2.plot(exp_transformed, 'r-', linewidth=1, alpha=0.7, label='Transformed')
ax.set_xlabel('Time')
ax.set_ylabel('Original', color='b')
ax2.set_ylabel('Transformed', color='r')
ax.set_title(f'Box-Cox Transform (λ={lambda_hat:.3f})')
ax.grid(alpha=0.3)

# Plot 8: HP filter decomposition
ax = axes[2, 1]
ax.plot(hp_series, 'k-', linewidth=1, alpha=0.5, label='Original')
ax.plot(trend_hp, 'r-', linewidth=2, label='Trend')
ax.plot(cycle_hp, 'b-', linewidth=1, alpha=0.7, label='Cycle')
ax.set_title('HP Filter Decomposition')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 9: Structural break
ax = axes[2, 2]
ax.plot(break_series_cumsum, linewidth=1, alpha=0.7)
ax.axvline(250, color='r', linestyle='--', linewidth=2, label='True Break')
if za_result['break_point']:
    ax.axvline(za_result['break_point'], color='g', linestyle=':', linewidth=2, label='Detected Break')
ax.set_title('Structural Break Detection')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary table
print("\n" + "="*80)
print("SUMMARY: STATIONARITY TEST RESULTS")
print("="*80)

summary_data = {
    'White Noise': results_wn,
    'Random Walk': results_rw,
    'RW Differenced': results_rw_diff,
    'Trend-Stationary': results_ts,
    'TS Detrended': results_detrended,
    f'AR(1) φ={phi}': results_ar1
}

print(f"\n{'Process':<20} {'ADF p-val':<12} {'KPSS p-val':<12} {'Consensus':<20}")
print("-" * 64)

for name, results in summary_data.items():
    adf_p = results['adf']['p_value']
    kpss_p = results['kpss']['p_value']
    
    if results['adf']['stationary'] and results['kpss']['stationary']:
        consensus = "Stationary"
    elif not results['adf']['stationary'] and not results['kpss']['stationary']:
        consensus = "Non-stationary"
    else:
        consensus = "Ambiguous"
    
    print(f"{name:<20} {adf_p:<12.4f} {kpss_p:<12.4f} {consensus:<20}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("1. Random walk: Both ADF and KPSS detect non-stationarity")
print("2. First difference of RW: Restores stationarity")
print("3. Trend-stationary: ADF may fail (trend confounds test)")
print("4. After detrending: Clear stationarity")
print("5. High AR(1): Near unit root → low power, ambiguous")
print("6. Box-Cox: Stabilizes variance in exponential growth")
print("7. HP filter: Separates trend and cycle components")
print("8. Structural breaks: Standard tests misleading, use ZA")
