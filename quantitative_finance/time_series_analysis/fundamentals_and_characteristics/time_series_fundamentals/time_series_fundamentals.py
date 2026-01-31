from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("TIME SERIES FUNDAMENTALS AND CHARACTERISTICS")
print("="*80)

class TimeSeriesDiagnostics:
    """Comprehensive time series diagnostics"""
    
    def __init__(self):
        pass
    
    def summary_statistics(self, y):
        """Basic descriptive statistics"""
        return {
            'mean': np.mean(y),
            'std': np.std(y),
            'min': np.min(y),
            'max': np.max(y),
            'skewness': stats.skew(y),
            'kurtosis': stats.kurtosis(y)
        }
    
    def autocorrelation_analysis(self, y, nlags=40):
        """Compute ACF and PACF"""
        acf_vals = acf(y, nlags=nlags, fft=False)
        pacf_vals = pacf(y, nlags=nlags)
        
        # Standard error for testing significance
        se = 1.96 / np.sqrt(len(y))
        
        return {
            'acf': acf_vals,
            'pacf': pacf_vals,
            'se': se
        }
    
    def ljung_box_test(self, y, lags=20):
        """
        Ljung-Box test for white noise
        H0: No autocorrelation up to lag h
        """
        n = len(y)
        acf_vals = acf(y, nlags=lags, fft=False)[1:]  # Exclude lag 0
        
        # Ljung-Box statistic
        Q = n * (n + 2) * np.sum(acf_vals**2 / (n - np.arange(1, lags+1)))
        
        # Chi-squared test
        p_value = 1 - stats.chi2.cdf(Q, lags)
        
        return {
            'Q_statistic': Q,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def adf_test(self, y, regression='c'):
        """
        Augmented Dickey-Fuller test
        H0: Unit root (non-stationary)
        
        regression: 'c' (constant), 'ct' (constant+trend), 'n' (none)
        """
        result = adfuller(y, regression=regression, autolag='AIC')
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'critical_values': result[4],
            'stationary': result[1] < 0.05
        }
    
    def kpss_test(self, y, regression='c'):
        """
        KPSS test
        H0: Trend stationary (opposite of ADF!)
        
        regression: 'c' (level stationary), 'ct' (trend stationary)
        """
        result = kpss(y, regression=regression, nlags='auto')
        
        return {
            'kpss_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'critical_values': result[3],
            'stationary': result[1] > 0.05
        }
    
    def variance_ratio_test(self, y, lags=[2, 4, 8, 16]):
        """
        Variance ratio test for random walk
        VR(k) should be 1 under random walk
        """
        n = len(y)
        returns = np.diff(y)
        
        var1 = np.var(returns, ddof=1)
        
        vr_stats = []
        for k in lags:
            # k-period returns
            returns_k = np.sum([returns[i::k] for i in range(k)], axis=0)
            vark = np.var(returns_k, ddof=1) / k
            
            vr = vark / var1
            
            # Under H0 (random walk), VR(k) = 1
            # Standard error
            se = np.sqrt(2 * (k - 1) / (n - k))
            z_stat = (vr - 1) / se
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
            
            vr_stats.append({
                'lag': k,
                'VR': vr,
                'z_stat': z_stat,
                'p_value': p_value
            })
        
        return vr_stats
    
    def detect_trend(self, y):
        """Simple linear trend detection"""
        t = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significant_trend': p_value < 0.05
        }
    
    def detect_seasonality(self, y, period=12):
        """Detect seasonality via ACF peaks"""
        acf_vals = acf(y, nlags=min(3*period, len(y)//2), fft=False)
        
        # Check for significant correlation at seasonal lags
        se = 1.96 / np.sqrt(len(y))
        seasonal_lags = [i*period for i in range(1, 4) if i*period < len(acf_vals)]
        seasonal_acf = [acf_vals[lag] for lag in seasonal_lags if lag < len(acf_vals)]
        
        has_seasonality = any(abs(a) > se for a in seasonal_acf)
        
        return {
            'seasonal_lags': seasonal_lags,
            'seasonal_acf': seasonal_acf,
            'has_seasonality': has_seasonality
        }

# Scenario 1: White Noise (Stationary Benchmark)
print("\n" + "="*80)
print("SCENARIO 1: White Noise - Ideal Stationary Process")
print("="*80)

# Generate white noise
n = 500
white_noise = np.random.normal(0, 1, n)

diagnostics = TimeSeriesDiagnostics()

# Summary statistics
stats_wn = diagnostics.summary_statistics(white_noise)
print(f"\nSummary Statistics:")
print(f"  Mean: {stats_wn['mean']:.4f} (Expected: 0)")
print(f"  Std Dev: {stats_wn['std']:.4f} (Expected: 1)")
print(f"  Skewness: {stats_wn['skewness']:.4f}")
print(f"  Kurtosis: {stats_wn['kurtosis']:.4f}")

# Autocorrelation
acf_results = diagnostics.autocorrelation_analysis(white_noise, nlags=40)
print(f"\nAutocorrelation:")
print(f"  ACF at lag 1: {acf_results['acf'][1]:.4f}")
print(f"  95% significance bound: ±{acf_results['se']:.4f}")
print(f"  All lags insignificant: {all(abs(acf_results['acf'][1:]) < acf_results['se'])}")

# Ljung-Box test
lb_test = diagnostics.ljung_box_test(white_noise, lags=20)
print(f"\nLjung-Box Test (White Noise):")
print(f"  Q-statistic: {lb_test['Q_statistic']:.2f}")
print(f"  p-value: {lb_test['p_value']:.4f}")
print(f"  Reject H0 (white noise): {lb_test['significant']}")

# ADF test
adf_result = diagnostics.adf_test(white_noise)
print(f"\nAugmented Dickey-Fuller Test:")
print(f"  ADF statistic: {adf_result['adf_statistic']:.4f}")
print(f"  p-value: {adf_result['p_value']:.4f}")
print(f"  Stationary: {adf_result['stationary']}")

# Scenario 2: Random Walk (Non-Stationary)
print("\n" + "="*80)
print("SCENARIO 2: Random Walk - Difference Stationary")
print("="*80)

# Generate random walk
random_walk = np.cumsum(white_noise)

stats_rw = diagnostics.summary_statistics(random_walk)
print(f"\nSummary Statistics:")
print(f"  Mean: {stats_rw['mean']:.4f}")
print(f"  Std Dev: {stats_rw['std']:.4f} (Growing with time)")

# ADF test on levels
adf_rw = diagnostics.adf_test(random_walk)
print(f"\nADF Test (Levels):")
print(f"  ADF statistic: {adf_rw['adf_statistic']:.4f}")
print(f"  p-value: {adf_rw['p_value']:.4f}")
print(f"  Stationary: {adf_rw['stationary']}")

# ADF test on first differences
diff_rw = np.diff(random_walk)
adf_diff = diagnostics.adf_test(diff_rw)
print(f"\nADF Test (First Differences):")
print(f"  ADF statistic: {adf_diff['adf_statistic']:.4f}")
print(f"  p-value: {adf_diff['p_value']:.4f}")
print(f"  Stationary: {adf_diff['stationary']}")

# Variance ratio test
vr_results = diagnostics.variance_ratio_test(random_walk, lags=[2, 4, 8, 16])
print(f"\nVariance Ratio Test (Random Walk):")
print(f"{'Lag':<8} {'VR':<10} {'Z-stat':<10} {'p-value':<10}")
print("-" * 38)
for vr in vr_results:
    print(f"{vr['lag']:<8} {vr['VR']:<10.3f} {vr['z_stat']:<10.3f} {vr['p_value']:<10.4f}")
print(f"VR ≈ 1 consistent with random walk")

# Scenario 3: AR(1) Process (Stationary with Memory)
print("\n" + "="*80)
print("SCENARIO 3: AR(1) Process - Stationary with Autocorrelation")
print("="*80)

# Generate AR(1) with φ=0.8
phi = 0.8
ar1 = np.zeros(n)
ar1[0] = white_noise[0]
for t in range(1, n):
    ar1[t] = phi * ar1[t-1] + white_noise[t]

# Theoretical vs empirical ACF
acf_ar1 = diagnostics.autocorrelation_analysis(ar1, nlags=20)
theoretical_acf = phi**np.arange(21)

print(f"\nAR(1) with φ={phi}:")
print(f"  Theoretical variance: {1/(1-phi**2):.3f}")
print(f"  Empirical variance: {np.var(ar1):.3f}")

print(f"\n{'Lag':<8} {'Empirical ACF':<18} {'Theoretical ACF':<18}")
print("-" * 44)
for k in range(1, 6):
    print(f"{k:<8} {acf_ar1['acf'][k]:<18.4f} {theoretical_acf[k]:<18.4f}")

# Stationarity tests
adf_ar1 = diagnostics.adf_test(ar1)
print(f"\nADF Test:")
print(f"  Stationary: {adf_ar1['stationary']}")

# Ljung-Box
lb_ar1 = diagnostics.ljung_box_test(ar1, lags=20)
print(f"\nLjung-Box Test:")
print(f"  p-value: {lb_ar1['p_value']:.4f}")
print(f"  Significant autocorrelation detected: {lb_ar1['significant']}")

# Scenario 4: Trend + Seasonality
print("\n" + "="*80)
print("SCENARIO 4: Trend + Seasonality - Decomposing Components")
print("="*80)

# Generate series with trend and seasonality
t = np.arange(n)
trend = 0.05 * t
seasonal = 5 * np.sin(2 * np.pi * t / 12)
y_complex = trend + seasonal + white_noise

# Detect trend
trend_test = diagnostics.detect_trend(y_complex)
print(f"\nLinear Trend Detection:")
print(f"  Slope: {trend_test['slope']:.6f} (True: 0.05)")
print(f"  R²: {trend_test['r_squared']:.4f}")
print(f"  Significant trend: {trend_test['significant_trend']}")

# Detect seasonality
seasonality_test = diagnostics.detect_seasonality(y_complex, period=12)
print(f"\nSeasonality Detection:")
print(f"  Has seasonality: {seasonality_test['has_seasonality']}")
print(f"  ACF at seasonal lags: {seasonality_test['seasonal_acf'][:3]}")

# Stationarity before and after differencing/detrending
adf_complex = diagnostics.adf_test(y_complex, regression='ct')
print(f"\nStationarity Tests (Original):")
print(f"  ADF p-value: {adf_complex['p_value']:.4f}")
print(f"  Stationary: {adf_complex['stationary']}")

# Detrend
y_detrended = y_complex - (trend_test['intercept'] + trend_test['slope'] * t)
adf_detrended = diagnostics.adf_test(y_detrended)
print(f"\nAfter Detrending:")
print(f"  ADF p-value: {adf_detrended['p_value']:.4f}")
print(f"  Stationary: {adf_detrended['stationary']}")

# Scenario 5: KPSS vs ADF (Complementary Tests)
print("\n" + "="*80)
print("SCENARIO 5: KPSS vs ADF - Complementary Stationarity Tests")
print("="*80)

# Test different processes
processes = {
    'White Noise': white_noise,
    'Random Walk': random_walk,
    'AR(1)': ar1,
    'Trend + Seasonal': y_complex
}

print(f"\n{'Process':<20} {'ADF (Unit Root)':<20} {'KPSS (Stationary)':<20} {'Conclusion':<20}")
print("-" * 80)

for name, series in processes.items():
    adf_res = diagnostics.adf_test(series)
    kpss_res = diagnostics.kpss_test(series)
    
    # Interpret both tests
    if adf_res['stationary'] and kpss_res['stationary']:
        conclusion = "Stationary"
    elif not adf_res['stationary'] and not kpss_res['stationary']:
        conclusion = "Non-stationary"
    else:
        conclusion = "Inconclusive"
    
    print(f"{name:<20} p={adf_res['p_value']:<17.4f} p={kpss_res['p_value']:<17.4f} {conclusion:<20}")

# Visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot 1: White noise time series
ax = axes[0, 0]
ax.plot(white_noise, linewidth=0.8, alpha=0.7)
ax.set_title('White Noise (Stationary)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.axhline(0, color='r', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)

# Plot 2: White noise ACF
ax = axes[0, 1]
plot_acf(white_noise, lags=40, ax=ax, alpha=0.05)
ax.set_title('White Noise ACF')
ax.grid(alpha=0.3)

# Plot 3: White noise PACF
ax = axes[0, 2]
plot_pacf(white_noise, lags=40, ax=ax, alpha=0.05)
ax.set_title('White Noise PACF')
ax.grid(alpha=0.3)

# Plot 4: Random walk time series
ax = axes[1, 0]
ax.plot(random_walk, linewidth=1.5, alpha=0.7)
ax.set_title('Random Walk (Non-Stationary)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.grid(alpha=0.3)

# Plot 5: Random walk ACF
ax = axes[1, 1]
plot_acf(random_walk, lags=40, ax=ax, alpha=0.05)
ax.set_title('Random Walk ACF (Slow Decay)')
ax.grid(alpha=0.3)

# Plot 6: Random walk first difference ACF
ax = axes[1, 2]
plot_acf(diff_rw, lags=40, ax=ax, alpha=0.05)
ax.set_title('First Difference ACF (White Noise)')
ax.grid(alpha=0.3)

# Plot 7: AR(1) time series
ax = axes[2, 0]
ax.plot(ar1, linewidth=1, alpha=0.7)
ax.set_title(f'AR(1) Process (φ={phi})')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.axhline(0, color='r', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)

# Plot 8: AR(1) ACF with theoretical
ax = axes[2, 1]
lags_plot = np.arange(21)
ax.stem(lags_plot, acf_ar1['acf'], linefmt='b-', markerfmt='bo', basefmt=' ', label='Empirical')
ax.plot(lags_plot, theoretical_acf, 'r--', linewidth=2, label='Theoretical')
ax.axhline(acf_ar1['se'], color='gray', linestyle='--', alpha=0.5)
ax.axhline(-acf_ar1['se'], color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_title(f'AR(1) ACF vs Theoretical')
ax.legend()
ax.grid(alpha=0.3)

# Plot 9: AR(1) PACF
ax = axes[2, 2]
plot_pacf(ar1, lags=20, ax=ax, alpha=0.05)
ax.set_title('AR(1) PACF (Cutoff at lag 1)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Additional visualization: Trend + Seasonality
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Original series with components
ax = axes2[0, 0]
ax.plot(t, y_complex, 'b-', linewidth=1, alpha=0.7, label='Observed')
ax.plot(t, trend, 'r--', linewidth=2, label='Trend')
ax.plot(t, trend + seasonal, 'g:', linewidth=2, label='Trend + Seasonal')
ax.set_title('Series with Trend and Seasonality')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: ACF showing seasonality
ax = axes2[0, 1]
plot_acf(y_complex, lags=60, ax=ax, alpha=0.05)
ax.set_title('ACF: Seasonal Peaks Visible')
# Mark seasonal lags
for s in [12, 24, 36, 48]:
    ax.axvline(s, color='r', linestyle=':', alpha=0.5)
ax.grid(alpha=0.3)

# Plot 3: Detrended series
ax = axes2[1, 0]
ax.plot(t, y_detrended, linewidth=1, alpha=0.7)
ax.set_title('After Detrending (Seasonal Pattern Remains)')
ax.set_xlabel('Time')
ax.set_ylabel('Detrended Value')
ax.grid(alpha=0.3)

# Plot 4: Comparison of processes (variance over time)
ax = axes2[1, 1]
window = 50
variance_wn = [np.var(white_noise[max(0,i-window):i+1]) for i in range(len(white_noise))]
variance_rw = [np.var(random_walk[max(0,i-window):i+1]) for i in range(len(random_walk))]
ax.plot(variance_wn, label='White Noise (Constant)', linewidth=1.5)
ax.plot(variance_rw, label='Random Walk (Growing)', linewidth=1.5)
ax.set_title('Rolling Variance: Stationary vs Non-Stationary')
ax.set_xlabel('Time')
ax.set_ylabel('Rolling Variance')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)
print(f"\n1. White Noise: Perfect stationarity, ACF=0 for all lags>0")
print(f"2. Random Walk: Non-stationary (ADF p={adf_rw['p_value']:.4f}), stationary after differencing")
print(f"3. AR(1): Stationary with memory, ACF decays exponentially (ρ_k = {phi}^k)")
print(f"4. Trend+Seasonal: Non-stationary, requires detrending and seasonal adjustment")
print(f"5. KPSS vs ADF: Complementary tests (different null hypotheses)")
