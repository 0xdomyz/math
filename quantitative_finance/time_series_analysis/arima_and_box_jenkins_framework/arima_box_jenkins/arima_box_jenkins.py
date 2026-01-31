from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("ARIMA AND BOX-JENKINS FRAMEWORK")
print("="*80)

# Generate synthetic data: AR(2) process with trend
np.random.seed(42)
n = 300
t = np.arange(n)

# Non-stationary: Random walk with drift + AR(2) component
trend = 0.5 * t
ar2_component = np.zeros(n)
ar2_component[0] = np.random.normal(0, 1)
ar2_component[1] = 0.6 * ar2_component[0] + np.random.normal(0, 1)

for i in range(2, n):
    ar2_component[i] = 0.6 * ar2_component[i-1] - 0.2 * ar2_component[i-2] + np.random.normal(0, 1)

y = trend + ar2_component

# Create time series
dates = pd.date_range('2020-01-01', periods=n, freq='D')
ts = pd.Series(y, index=dates)

print("\n" + "="*80)
print("STEP 1: MODEL IDENTIFICATION")
print("="*80)

# 1.1 Stationarity Tests
print("\n1.1 STATIONARITY ASSESSMENT")
print("-" * 40)

def adf_test(series, name=''):
    """Augmented Dickey-Fuller test"""
    result = adfuller(series, autolag='AIC')
    print(f"\nADF Test - {name}:")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  P-value: {result[1]:.4f}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.3f}")
    print(f"  Result: {'Stationary' if result[1] < 0.05 else 'Non-Stationary'}")
    return result[1] < 0.05

def kpss_test(series, name=''):
    """KPSS test"""
    result = kpss(series, regression='c', nlags='auto')
    print(f"\nKPSS Test - {name}:")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  P-value: {result[1]:.4f}")
    print(f"  Critical Values:")
    for key, value in result[3].items():
        print(f"    {key}: {value:.3f}")
    print(f"  Result: {'Non-Stationary' if result[1] < 0.05 else 'Stationary'}")
    return result[1] >= 0.05

# Test original series
is_stationary_adf = adf_test(ts, 'Original Series')
is_stationary_kpss = kpss_test(ts, 'Original Series')

print(f"\nConclusion: Series is {'stationary' if is_stationary_adf and is_stationary_kpss else 'non-stationary'}")

# 1.2 Differencing
print("\n1.2 DIFFERENCING TO ACHIEVE STATIONARITY")
print("-" * 40)

ts_diff = ts.diff().dropna()

print("\nFirst Difference (d=1):")
is_stationary_adf_diff = adf_test(ts_diff, 'First Difference')
is_stationary_kpss_diff = kpss_test(ts_diff, 'First Difference')

d = 1 if not (is_stationary_adf and is_stationary_kpss) else 0
print(f"\n→ Differencing order d = {d}")

# 1.3 ACF/PACF Analysis
print("\n1.3 ACF/PACF ANALYSIS")
print("-" * 40)

# Use differenced series if d=1
analysis_series = ts_diff if d == 1 else ts

# Calculate ACF/PACF

acf_values = acf(analysis_series, nlags=20)
pacf_values = pacf(analysis_series, nlags=20)

# Identify cutoffs (simplified heuristic)
conf_interval = 1.96 / np.sqrt(len(analysis_series))

acf_significant = np.where(np.abs(acf_values[1:]) > conf_interval)[0] + 1
pacf_significant = np.where(np.abs(pacf_values[1:]) > conf_interval)[0] + 1

print(f"\nSignificant ACF lags: {acf_significant[:5] if len(acf_significant) > 0 else 'None'}")
print(f"Significant PACF lags: {pacf_significant[:5] if len(pacf_significant) > 0 else 'None'}")

# Suggest initial orders
if len(pacf_significant) > 0:
    p_suggest = min(pacf_significant[0], 3)
else:
    p_suggest = 0

if len(acf_significant) > 0:
    q_suggest = min(acf_significant[0], 3)
else:
    q_suggest = 0

print(f"\nSuggested initial model: ARIMA({p_suggest},{d},{q_suggest})")

# STEP 2: PARAMETER ESTIMATION
print("\n" + "="*80)
print("STEP 2: PARAMETER ESTIMATION AND MODEL SELECTION")
print("="*80)

# Grid search over candidate models
candidate_models = []
max_p, max_q = 3, 3

print("\nGrid Search Results:")
print(f"{'Model':<15} {'AIC':<12} {'BIC':<12} {'Log-Likelihood':<15}")
print("-" * 60)

for p in range(max_p + 1):
    for q in range(max_q + 1):
        if p == 0 and q == 0:
            continue
        try:
            model = ARIMA(ts, order=(p, d, q))
            fitted_model = model.fit()
            
            candidate_models.append({
                'order': (p, d, q),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'loglik': fitted_model.llf,
                'model': fitted_model
            })
            
            print(f"ARIMA({p},{d},{q}){'':<5} {fitted_model.aic:<12.2f} "
                  f"{fitted_model.bic:<12.2f} {fitted_model.llf:<15.2f}")
        except:
            pass

# Select best by AIC
best_model_aic = min(candidate_models, key=lambda x: x['aic'])
best_model_bic = min(candidate_models, key=lambda x: x['bic'])

print(f"\nBest by AIC: ARIMA{best_model_aic['order']} (AIC={best_model_aic['aic']:.2f})")
print(f"Best by BIC: ARIMA{best_model_bic['order']} (BIC={best_model_bic['bic']:.2f})")

# Use BIC selection (more parsimonious)
final_model = best_model_bic['model']
p, d, q = best_model_bic['order']

print(f"\n→ Selected Model: ARIMA({p},{d},{q})")
print(f"\nParameter Estimates:")
print(final_model.summary())

# STEP 3: DIAGNOSTIC CHECKING
print("\n" + "="*80)
print("STEP 3: DIAGNOSTIC CHECKING")
print("="*80)

residuals = final_model.resid

# 3.1 Ljung-Box Test
print("\n3.1 LJUNG-BOX TEST (Residual Autocorrelation)")
print("-" * 40)

lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
print(lb_test)
print(f"\nInterpretation: P-values > 0.05 indicate no significant autocorrelation")
print(f"  Result: {'PASS (White noise residuals)' if (lb_test['lb_pvalue'] > 0.05).all() else 'FAIL (Autocorrelation remains)'}")

# 3.2 Normality Test
print("\n3.2 NORMALITY TEST")
print("-" * 40)

jb_stat, jb_pvalue = stats.jarque_bera(residuals)
print(f"Jarque-Bera Test:")
print(f"  Statistic: {jb_stat:.4f}")
print(f"  P-value: {jb_pvalue:.4f}")
print(f"  Result: {'Normally distributed' if jb_pvalue > 0.05 else 'Non-normal (outliers or fat tails)'}")

# 3.3 Heteroscedasticity (ARCH effects)
print("\n3.3 HETEROSCEDASTICITY CHECK")
print("-" * 40)

residuals_squared = residuals ** 2
arch_test = acorr_ljungbox(residuals_squared, lags=10, return_df=True)
print(f"Ljung-Box test on squared residuals:")
print(f"  Minimum p-value: {arch_test['lb_pvalue'].min():.4f}")
print(f"  Result: {'No ARCH effects' if arch_test['lb_pvalue'].min() > 0.05 else 'ARCH effects present (consider GARCH)'}")

# STEP 4: FORECASTING
print("\n" + "="*80)
print("STEP 4: FORECASTING")
print("="*80)

# Out-of-sample forecast
forecast_horizon = 30
train_size = len(ts) - forecast_horizon
ts_train = ts[:train_size]
ts_test = ts[train_size:]

# Refit on training data
train_model = ARIMA(ts_train, order=(p, d, q))
train_fitted = train_model.fit()

# Forecast
forecast_result = train_fitted.get_forecast(steps=forecast_horizon)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95% CI

# Evaluation metrics
mae = np.mean(np.abs(ts_test - forecast_mean))
rmse = np.sqrt(np.mean((ts_test - forecast_mean) ** 2))
mape = np.mean(np.abs((ts_test - forecast_mean) / ts_test)) * 100

print(f"\nOut-of-Sample Forecast Evaluation:")
print(f"  Forecast Horizon: {forecast_horizon} periods")
print(f"  Training Size: {train_size}")
print(f"  Test Size: {forecast_horizon}")
print(f"\nAccuracy Metrics:")
print(f"  MAE (Mean Absolute Error): {mae:.3f}")
print(f"  RMSE (Root Mean Squared Error): {rmse:.3f}")
print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

# Seasonal ARIMA Example
print("\n" + "="*80)
print("BONUS: SEASONAL ARIMA (SARIMA) EXAMPLE")
print("="*80)

# Generate seasonal data
np.random.seed(123)
n_seasonal = 200
seasonal_period = 12

# Monthly data with trend and seasonality
t_seasonal = np.arange(n_seasonal)
trend_seasonal = 0.1 * t_seasonal
seasonal_component = 10 * np.sin(2 * np.pi * t_seasonal / seasonal_period)
noise = np.random.normal(0, 2, n_seasonal)
y_seasonal = 50 + trend_seasonal + seasonal_component + noise

dates_seasonal = pd.date_range('2005-01-01', periods=n_seasonal, freq='M')
ts_seasonal = pd.Series(y_seasonal, index=dates_seasonal)

# Fit SARIMA
print("\nFitting SARIMA(1,1,1)(1,1,1)_12...")
sarima_model = SARIMAX(ts_seasonal, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fitted = sarima_model.fit(disp=False)

print(f"\nSARIMA Model Summary:")
print(f"  AIC: {sarima_fitted.aic:.2f}")
print(f"  BIC: {sarima_fitted.bic:.2f}")

# Forecast 24 months ahead
sarima_forecast = sarima_fitted.get_forecast(steps=24)
sarima_mean = sarima_forecast.predicted_mean
sarima_ci = sarima_forecast.conf_int()

# Visualizations
fig = plt.figure(figsize=(18, 14))

# Plot 1: Original Series and Differenced
ax1 = plt.subplot(3, 3, 1)
ax1.plot(ts, label='Original', linewidth=1.5)
ax1.set_title('Original Time Series')
ax1.set_xlabel('Date')
ax1.set_ylabel('Value')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
ax2.plot(ts_diff, label='First Difference', color='orange', linewidth=1.5)
ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
ax2.set_title('First Difference (d=1)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Δy')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: ACF
ax3 = plt.subplot(3, 3, 3)
plot_acf(analysis_series, lags=20, ax=ax3, alpha=0.05)
ax3.set_title('ACF (Differenced Series)')
ax3.grid(alpha=0.3)

# Plot 4: PACF
ax4 = plt.subplot(3, 3, 4)
plot_pacf(analysis_series, lags=20, ax=ax4, alpha=0.05)
ax4.set_title('PACF (Differenced Series)')
ax4.grid(alpha=0.3)

# Plot 5: Residuals
ax5 = plt.subplot(3, 3, 5)
ax5.plot(residuals, linewidth=1, alpha=0.7)
ax5.axhline(0, color='red', linestyle='--')
ax5.set_title(f'Residuals: ARIMA({p},{d},{q})')
ax5.set_xlabel('Date')
ax5.set_ylabel('Residual')
ax5.grid(alpha=0.3)

# Plot 6: ACF of Residuals
ax6 = plt.subplot(3, 3, 6)
plot_acf(residuals, lags=20, ax=ax6, alpha=0.05)
ax6.set_title('ACF of Residuals (Should be White Noise)')
ax6.grid(alpha=0.3)

# Plot 7: Q-Q Plot
ax7 = plt.subplot(3, 3, 7)
stats.probplot(residuals, dist="norm", plot=ax7)
ax7.set_title('Q-Q Plot (Normality Check)')
ax7.grid(alpha=0.3)

# Plot 8: Forecast vs Actual
ax8 = plt.subplot(3, 3, 8)
ax8.plot(ts_train.index, ts_train, label='Training', linewidth=1.5)
ax8.plot(ts_test.index, ts_test, label='Actual', color='green', linewidth=1.5)
ax8.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red', linewidth=2, linestyle='--')
ax8.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                 color='red', alpha=0.2, label='95% CI')
ax8.set_title(f'Forecast: ARIMA({p},{d},{q})')
ax8.set_xlabel('Date')
ax8.set_ylabel('Value')
ax8.legend()
ax8.grid(alpha=0.3)

# Plot 9: SARIMA Forecast
ax9 = plt.subplot(3, 3, 9)
ax9.plot(ts_seasonal, label='Historical', linewidth=1.5)
ax9.plot(sarima_mean.index, sarima_mean, label='Forecast', color='red', linewidth=2, linestyle='--')
ax9.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1],
                 color='red', alpha=0.2, label='95% CI')
ax9.set_title('SARIMA(1,1,1)(1,1,1)₁₂ Forecast')
ax9.set_xlabel('Date')
ax9.set_ylabel('Value')
ax9.legend()
ax9.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. Original series non-stationary → d={d} differencing required")
print(f"2. ACF/PACF analysis suggested ARIMA({p_suggest},{d},{q_suggest}), BIC selected ({p},{d},{q})")
print(f"3. Ljung-Box test: Residuals {'pass' if (lb_test['lb_pvalue'] > 0.05).all() else 'fail'} white noise check")
print(f"4. Out-of-sample RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")
print(f"5. Box-Jenkins iterative: Identify → Estimate → Diagnose → Forecast")
print(f"6. SARIMA captures seasonality via (P,D,Q)_s seasonal terms")
print(f"7. Model selection: Lower AIC/BIC preferred, balance fit vs. parsimony")
