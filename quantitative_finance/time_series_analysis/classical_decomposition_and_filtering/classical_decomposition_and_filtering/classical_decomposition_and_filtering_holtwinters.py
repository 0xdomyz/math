from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

class HoltWinters:
    """Holt-Winters exponential smoothing with seasonality"""
    
    def __init__(self, alpha=0.3, beta=0.1, gamma=0.1, seasonal='additive', period=12):
        self.alpha = alpha  # Level
        self.beta = beta    # Trend
        self.gamma = gamma  # Seasonal
        self.seasonal = seasonal
        self.period = period
    
    def fit(self, y):
        """Estimate components"""
        n = len(y)
        
        # Initialize
        self.level = np.zeros(n)
        self.trend = np.zeros(n)
        self.season = np.zeros(n + self.period)
        self.fitted = np.zeros(n)
        
        # Initial values
        self.level[0] = y[0]
        self.trend[0] = (y[self.period] - y[0]) / self.period if n > self.period else 0
        
        # Initial seasonal (first year average)
        if n >= self.period:
            for s in range(self.period):
                season_vals = y[s::self.period][:int(n/self.period)]
                if self.seasonal == 'additive':
                    self.season[s] = np.mean(season_vals) - np.mean(y[:self.period])
                else:
                    self.season[s] = np.mean(season_vals) / np.mean(y[:self.period])
        
        # Recursion
        for t in range(1, n):
            s_idx = (t - self.period) % self.period
            
            if self.seasonal == 'additive':
                # Additive
                self.level[t] = (self.alpha * (y[t] - self.season[s_idx]) +
                                (1 - self.alpha) * (self.level[t-1] + self.trend[t-1]))
                self.trend[t] = (self.beta * (self.level[t] - self.level[t-1]) +
                                (1 - self.beta) * self.trend[t-1])
                self.season[t] = (self.gamma * (y[t] - self.level[t]) +
                                 (1 - self.gamma) * self.season[s_idx])
                self.fitted[t] = self.level[t] + self.trend[t] + self.season[s_idx]
            else:
                # Multiplicative
                self.level[t] = (self.alpha * (y[t] / self.season[s_idx]) +
                                (1 - self.alpha) * (self.level[t-1] + self.trend[t-1]))
                self.trend[t] = (self.beta * (self.level[t] - self.level[t-1]) +
                                (1 - self.beta) * self.trend[t-1])
                self.season[t] = (self.gamma * (y[t] / self.level[t]) +
                                 (1 - self.gamma) * self.season[s_idx])
                self.fitted[t] = (self.level[t] + self.trend[t]) * self.season[s_idx]
        
        return self.fitted
    
    def forecast(self, h):
        """Forecast h steps ahead"""
        forecasts = np.zeros(h)
        
        for i in range(h):
            s_idx = (len(self.level) - self.period + i) % self.period
            
            if self.seasonal == 'additive':
                forecasts[i] = self.level[-1] + (i+1) * self.trend[-1] + self.season[s_idx]
            else:
                forecasts[i] = (self.level[-1] + (i+1) * self.trend[-1]) * self.season[s_idx]
        
        return forecasts

# Scenario 1: Generate synthetic time series with known components
print("\n" + "="*80)
print("SCENARIO 1: Synthetic Data - Known Components")
print("="*80)

# Generate data
t = np.arange(120)  # 10 years monthly
trend = 100 + 0.5 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
irregular = np.random.normal(0, 3, 120)
y_additive = trend + seasonal + irregular

print(f"\nGenerated Series: T={len(y_additive)} (monthly, 10 years)")
print(f"True Components:")
print(f"  Trend: Linear (slope=0.5)")
print(f"  Seasonal: Sine wave (period=12, amplitude=10)")
print(f"  Irregular: Normal noise (σ=3)")

# Decompose
decomposer = TimeSeriesDecomposer()
decomp = decomposer.additive_decomposition(y_additive, period=12)

# Compare with true components
trend_rmse = np.sqrt(np.nanmean((decomp['trend'] - trend)**2))
seasonal_rmse = np.sqrt(np.nanmean((decomp['seasonal'] - seasonal)**2))

print(f"\nClassical Additive Decomposition:")
print(f"  Trend RMSE: {trend_rmse:.3f}")
print(f"  Seasonal RMSE: {seasonal_rmse:.3f}")
print(f"  Irregular Std: {np.nanstd(decomp['irregular']):.3f} (True: 3.0)")

# STL decomposition
stl_decomp = decomposer.stl_decomposition(y_additive, period=12)

trend_rmse_stl = np.sqrt(np.mean((stl_decomp['trend'] - trend)**2))
seasonal_rmse_stl = np.sqrt(np.mean((stl_decomp['seasonal'] - seasonal)**2))

print(f"\nSTL Decomposition:")
print(f"  Trend RMSE: {trend_rmse_stl:.3f}")
print(f"  Seasonal RMSE: {seasonal_rmse_stl:.3f}")
print(f"  Irregular Std: {np.std(stl_decomp['irregular']):.3f}")

# Scenario 2: Multiplicative decomposition
print("\n" + "="*80)
print("SCENARIO 2: Multiplicative Seasonality")
print("="*80)

# Generate multiplicative series
trend_mult = 50 * np.exp(0.01 * t)
seasonal_mult = 1 + 0.3 * np.sin(2 * np.pi * t / 12)
irregular_mult = 1 + np.random.normal(0, 0.05, 120)
y_mult = trend_mult * seasonal_mult * irregular_mult

print(f"\nMultiplicative Series: Exponential trend × seasonal pattern")

# Additive (wrong model)
decomp_add = decomposer.additive_decomposition(y_mult, period=12)
residual_add = y_mult - decomp_add['trend'] - decomp_add['seasonal']
residual_std_add = np.nanstd(residual_add)

# Multiplicative (correct model)
decomp_mult = decomposer.multiplicative_decomposition(y_mult, period=12)
residual_mult = y_mult / (decomp_mult['trend'] * decomp_mult['seasonal'])
residual_std_mult = np.nanstd(residual_mult)

print(f"\nAdditive Decomposition (wrong):")
print(f"  Residual Std: {residual_std_add:.3f}")

print(f"\nMultiplicative Decomposition (correct):")
print(f"  Residual Std: {residual_std_mult:.3f}")
print(f"  → {(residual_std_add/residual_std_mult):.1f}× better fit")

# Scenario 3: Filtering comparison
print("\n" + "="*80)
print("SCENARIO 3: Filter Comparison - Trend Extraction")
print("="*80)

# Use additive series
filter_obj = TimeSeriesFilter()

# Moving average
ma_trend = filter_obj.moving_average(y_additive, window=12)

# Exponential smoothing
es_trend = filter_obj.exponential_smoothing(y_additive, alpha=0.1)

# Hodrick-Prescott
hp_trend, hp_cycle = filter_obj.hodrick_prescott(y_additive, lam=1600)

# Butterworth low-pass
butter_trend = filter_obj.butterworth_filter(y_additive, cutoff_freq=0.05, order=5)

# Compare to true trend
ma_rmse = np.sqrt(np.mean((ma_trend - trend)**2))
es_rmse = np.sqrt(np.mean((es_trend - trend)**2))
hp_rmse = np.sqrt(np.mean((hp_trend - trend)**2))
butter_rmse = np.sqrt(np.mean((butter_trend - trend)**2))

print(f"\nTrend Extraction Performance (RMSE vs True Trend):")
print(f"  Moving Average (12-month): {ma_rmse:.3f}")
print(f"  Exponential Smoothing (α=0.1): {es_rmse:.3f}")
print(f"  Hodrick-Prescott (λ=1600): {hp_rmse:.3f}")
print(f"  Butterworth Low-Pass: {butter_rmse:.3f}")

# Scenario 4: Holt-Winters forecasting
print("\n" + "="*80)
print("SCENARIO 4: Holt-Winters Exponential Smoothing")
print("="*80)

# Fit Holt-Winters
hw = HoltWinters(alpha=0.2, beta=0.1, gamma=0.1, seasonal='additive', period=12)
fitted = hw.fit(y_additive)

# In-sample fit
fit_rmse = np.sqrt(np.mean((fitted - y_additive)**2))

print(f"\nHolt-Winters (α=0.2, β=0.1, γ=0.1):")
print(f"  In-sample RMSE: {fit_rmse:.3f}")

# Forecast
forecast_horizon = 24
forecasts = hw.forecast(forecast_horizon)

# Generate true future values
t_future = np.arange(120, 120 + forecast_horizon)
trend_future = 100 + 0.5 * t_future
seasonal_future = 10 * np.sin(2 * np.pi * t_future / 12)
y_future = trend_future + seasonal_future

# Forecast accuracy
forecast_rmse = np.sqrt(np.mean((forecasts - y_future)**2))

print(f"  Forecast RMSE (24 months): {forecast_rmse:.3f}")

# Scenario 5: Band-pass filtering (business cycle)
print("\n" + "="*80)
print("SCENARIO 5: Band-Pass Filter - Business Cycle Extraction")
print("="*80)

# Generate quarterly GDP-like series
t_q = np.arange(200)  # 50 years quarterly
trend_gdp = 1000 + 5 * t_q
cycle_gdp = 50 * np.sin(2 * np.pi * t_q / 32)  # 8-year cycle
short_gdp = 20 * np.sin(2 * np.pi * t_q / 4)   # 1-year seasonal
noise_gdp = np.random.normal(0, 10, 200)
y_gdp = trend_gdp + cycle_gdp + short_gdp + noise_gdp

print(f"\nSimulated GDP: Trend + 8-year cycle + 1-year seasonal + noise")

# Extract business cycle (2-8 years = 8-32 quarters)
cycle_extracted = filter_obj.band_pass_filter(y_gdp, low_freq=1/32, high_freq=1/8, fs=1.0)

# Correlation with true cycle
corr = np.corrcoef(cycle_extracted, cycle_gdp)[0, 1]

print(f"  Band-Pass Filter [8-32 quarters]:")
print(f"    Correlation with true cycle: {corr:.3f}")

# Compare HP filter
hp_trend_gdp, hp_cycle_gdp = filter_obj.hodrick_prescott(y_gdp, lam=1600)
hp_corr = np.corrcoef(hp_cycle_gdp, cycle_gdp)[0, 1]

print(f"  HP Filter (λ=1600):")
print(f"    Correlation with true cycle: {hp_corr:.3f}")

# Scenario 6: Real-time vs two-sided filtering
print("\n" + "="*80)
print("SCENARIO 6: Real-Time (Causal) vs Two-Sided Filtering")
print("="*80)

# Real-time MA (trailing)
rt_ma = filter_obj.moving_average(y_additive, window=12, center=False)

# Two-sided MA (centered)
ts_ma = filter_obj.moving_average(y_additive, window=12, center=True)

# Lag measurement
lag_rt = np.argmax(np.correlate(trend - np.mean(trend), rt_ma - np.mean(rt_ma), mode='full')) - len(trend) + 1
lag_ts = np.argmax(np.correlate(trend - np.mean(trend), ts_ma - np.mean(ts_ma), mode='full')) - len(trend) + 1

print(f"\n12-Month Moving Average:")
print(f"  Real-Time (trailing): Lag ≈ {abs(lag_rt)} periods")
print(f"  Two-Sided (centered): Lag ≈ {abs(lag_ts)} periods")
print(f"\nReal-time suitable for forecasting, two-sided for historical analysis")

# Visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot 1: Original series and decomposition (additive)
ax = axes[0, 0]
ax.plot(t, y_additive, 'gray', alpha=0.5, label='Original')
ax.plot(t, decomp['trend'], 'b-', linewidth=2, label='Trend')
ax.plot(t, trend, 'r--', linewidth=1.5, label='True Trend')
ax.set_title('Additive Decomposition: Trend')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Seasonal component
ax = axes[0, 1]
ax.plot(t[:36], decomp['seasonal'][:36], 'g-', linewidth=2, label='Estimated')
ax.plot(t[:36], seasonal[:36], 'r--', linewidth=1.5, label='True')
ax.set_title('Seasonal Component (First 3 Years)')
ax.set_xlabel('Time')
ax.set_ylabel('Seasonal Effect')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Irregular component
ax = axes[0, 2]
ax.plot(t, decomp['irregular'], 'k.', markersize=3, alpha=0.6)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_title('Irregular Component (Residuals)')
ax.set_xlabel('Time')
ax.set_ylabel('Residual')
ax.grid(alpha=0.3)

# Plot 4: STL vs Classical
ax = axes[1, 0]
valid_idx = ~np.isnan(decomp['trend'])
ax.plot(t[valid_idx], decomp['trend'][valid_idx], 'b-', linewidth=2, label='Classical', alpha=0.7)
ax.plot(t, stl_decomp['trend'], 'g-', linewidth=2, label='STL', alpha=0.7)
ax.plot(t, trend, 'r--', linewidth=1.5, label='True')
ax.set_title('Classical vs STL: Trend Comparison')
ax.set_xlabel('Time')
ax.set_ylabel('Trend')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Multiplicative decomposition
ax = axes[1, 1]
ax.plot(t, y_mult, 'gray', alpha=0.5, label='Original')
ax.plot(t, decomp_mult['trend'], 'b-', linewidth=2, label='Trend')
ax.set_title('Multiplicative Decomposition')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Filter comparison
ax = axes[1, 2]
ax.plot(t, trend, 'r-', linewidth=2.5, label='True Trend', alpha=0.8)
ax.plot(t, ma_trend, 'b-', linewidth=1.5, label='MA', alpha=0.7)
ax.plot(t, hp_trend, 'g-', linewidth=1.5, label='HP Filter', alpha=0.7)
ax.plot(t, butter_trend, 'm-', linewidth=1.5, label='Butterworth', alpha=0.7)
ax.set_title('Filter Comparison: Trend Extraction')
ax.set_xlabel('Time')
ax.set_ylabel('Trend')
ax.legend()
ax.grid(alpha=0.3)

# Plot 7: Holt-Winters fit and forecast
ax = axes[2, 0]
ax.plot(t, y_additive, 'gray', alpha=0.5, label='Observed', linewidth=1)
ax.plot(t, fitted, 'b-', linewidth=2, label='Fitted')
t_forecast = np.arange(120, 120 + forecast_horizon)
ax.plot(t_forecast, forecasts, 'r--', linewidth=2, label='Forecast')
ax.plot(t_forecast, y_future, 'g:', linewidth=1.5, label='True Future')
ax.axvline(x=120, color='k', linestyle='--', alpha=0.5)
ax.set_title('Holt-Winters: Fit and Forecast')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 8: Business cycle extraction
ax = axes[2, 1]
ax.plot(t_q, cycle_gdp, 'r-', linewidth=2, label='True Cycle', alpha=0.7)
ax.plot(t_q, cycle_extracted, 'b-', linewidth=1.5, label='Band-Pass', alpha=0.7)
ax.plot(t_q, hp_cycle_gdp, 'g-', linewidth=1.5, label='HP Filter', alpha=0.7)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_title('Business Cycle Extraction')
ax.set_xlabel('Time (Quarters)')
ax.set_ylabel('Cyclical Component')
ax.legend()
ax.grid(alpha=0.3)

# Plot 9: Real-time vs two-sided lag
ax = axes[2, 2]
ax.plot(t, trend, 'r-', linewidth=2.5, label='True Trend', alpha=0.8)
ax.plot(t, rt_ma, 'b-', linewidth=1.5, label='Real-Time MA', alpha=0.7)
ax.plot(t, ts_ma, 'g-', linewidth=1.5, label='Two-Sided MA', alpha=0.7)
ax.set_title('Real-Time vs Two-Sided Filtering')
ax.set_xlabel('Time')
ax.set_ylabel('Trend')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Additional analysis: Frequency domain
print("\n" + "="*80)
print("SCENARIO 7: Frequency Domain Analysis")
print("="*80)

# Compute periodogram
Y_fft = fft(y_additive - np.mean(y_additive))
freqs = fftfreq(len(y_additive), 1.0)
power = np.abs(Y_fft)**2 / len(y_additive)

# Find dominant frequencies (positive half)
pos_mask = freqs > 0
freqs_pos = freqs[pos_mask]
power_pos = power[pos_mask]

# Peak detection
peak_idx = np.argmax(power_pos)
dominant_freq = freqs_pos[peak_idx]
dominant_period = 1 / dominant_freq

print(f"\nPeriodogram Analysis:")
print(f"  Dominant frequency: {dominant_freq:.4f} cycles/period")
print(f"  Corresponding period: {dominant_period:.1f} (expected: 12 months)")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Periodogram
ax = axes2[0]
ax.plot(freqs_pos, power_pos, 'b-', linewidth=1.5)
ax.axvline(x=1/12, color='r', linestyle='--', label='Expected (12-month)')
ax.set_xlabel('Frequency')
ax.set_ylabel('Power')
ax.set_title('Periodogram: Power Spectral Density')
ax.legend()
ax.grid(alpha=0.3)

# Seasonal subseries plot
ax = axes2[1]
for year in range(10):
    start = year * 12
    end = start + 12
    if end <= len(y_additive):
        ax.plot(range(1, 13), y_additive[start:end], 'o-', alpha=0.5)

ax.set_xlabel('Month')
ax.set_ylabel('Value')
ax.set_title('Seasonal Subseries Plot (Each Year)')
ax.set_xticks(range(1, 13))
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
