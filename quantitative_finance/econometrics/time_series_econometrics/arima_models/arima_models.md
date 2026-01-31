# ARIMA Models

## 1. Concept Skeleton
**Definition:** AutoRegressive Integrated Moving Average (ARIMA) combines autoregressive lags, differencing, and moving average terms to model non-stationary time series  
**Purpose:** Forecast future values, decompose trend/seasonality, handle integrated (unit root) processes  
**Prerequisites:** Stationarity tests, AR/MA processes, differencing, Box-Jenkins methodology

## 2. Comparative Framing
| Model | AR(p) | MA(q) | ARIMA(p,d,q) | SARIMA(p,d,q)(P,D,Q)s |
|-------|-------|-------|--------------|----------------------|
| **Components** | p lags of Y | q lags of ε | Autoregressive + Differencing + MA | + Seasonal AR/MA |
| **Non-stationary** | No | No | Yes (via d) | Yes (seasonal differencing) |
| **Estimation** | OLS/MLE | MLE | MLE | MLE |
| **Forecasting** | Short-medium | Short-medium | Medium-long | Medium-long |
| **Complexity** | Low | Low | Medium | High |

## 3. Examples + Counterexamples

**Simple Example:**  
Monthly retail sales: ARIMA(1,1,1) - one AR lag, first differencing, one MA lag captures trend and short-term dependency

**Failure Case:**  
Structural break in 2008 financial crisis: ARIMA fits pre-break pattern, forecasts fail post-break → use regime-switching or intervention

**Edge Case:**  
Seasonal data (quarterly with annual cycle): ARIMA(1,1,1) misses seasonality → use SARIMA(1,1,1)(1,1,1)₄

## 4. Layer Breakdown
```
ARIMA(p,d,q) Structure:
├─ Differencing (I - Integrated):
│   ├─ d=0: Already stationary (use ARMA)
│   ├─ d=1: First differences ΔYₜ = Yₜ - Yₜ₋₁
│   ├─ d=2: Second differences Δ²Yₜ = ΔYₜ - ΔYₜ₋₁
│   └─ Higher d: Rare; overdifferencing removes signal
├─ Autoregressive (AR - p):
│   ├─ Coefficient φ₁, φ₂, ..., φₚ for Yₜ₋₁, Yₜ₋₂, ..., Yₜ₋ₚ
│   ├─ Captures momentum, mean reversion
│   └─ Stationarity requires |roots| > 1 (characteristic polynomial)
├─ Moving Average (MA - q):
│   ├─ Coefficient θ₁, θ₂, ..., θₑ for εₜ₋₁, εₜ₋₂, ..., εₜ₋ₑ
│   ├─ Captures short-run shocks
│   └─ Invertibility requires |roots| > 1
├─ Model Equation:
│   ├─ ARIMA(p,d,q): Δᵈ(Yₜ) = φ₁Δᵈ(Yₜ₋₁) + ... + φₚΔᵈ(Yₜ₋ₚ) + εₜ + θ₁εₜ₋₁ + ... + θₑεₜ₋ₑ
│   └─ Operator form: φ(L)(1-L)ᵈYₜ = θ(L)εₜ
├─ Parameter Selection (Box-Jenkins):
│   ├─ ACF/PACF plots: Identify p, d, q
│   ├─ AIC/BIC: Compare competing models
│   ├─ d: Difference until stationary (ADF test)
│   └─ Grid search: Try all combinations within reasonable bounds
└─ Forecasting:
    ├─ Point forecast: Plug in fitted parameters and lagged values
    ├─ Prediction intervals: Widen with forecast horizon
    └─ Seasonal: SARIMA accounts for repeating patterns
```

**Interaction:** Differencing removes trend → AR models pattern in residuals → MA smooths shocks

## 5. Mini-Project
Fit ARIMA to monthly airline passengers and compare with SARIMA:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic monthly airline data (trend + seasonality)
np.random.seed(42)
months = 144  # 12 years
t = np.arange(months)
trend = 100 + 0.5*t
seasonal = 20*np.sin(2*np.pi*t/12)
noise = np.random.normal(0, 5, months)
y = trend + seasonal + noise

dates = pd.date_range('2010-01', periods=months, freq='MS')
ts_data = pd.Series(y, index=dates)

# 1. Check Stationarity
def test_stationarity(timeseries, name=''):
    result = adfuller(timeseries, autolag='AIC')
    print(f'\nADF Test for {name}:')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    is_stationary = result[1] < 0.05
    print(f'Stationary (α=0.05): {is_stationary}')
    return is_stationary

print("Original Series:")
test_stationarity(ts_data, 'Original')

# First difference
ts_diff1 = ts_data.diff().dropna()
print("\nFirst Difference:")
is_stat_d1 = test_stationarity(ts_diff1, 'First Difference')

# Seasonal difference (12-month lag)
ts_diff12 = ts_data.diff(12).dropna()
print("\nSeasonal Difference (lag=12):")
test_stationarity(ts_diff12, 'Seasonal Difference')

# 2. ACF/PACF Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# ACF/PACF of original
plot_acf(ts_data, lags=40, ax=axes[0, 0])
axes[0, 0].set_title('ACF - Original Series')

plot_pacf(ts_data, lags=40, ax=axes[0, 1])
axes[0, 1].set_title('PACF - Original Series')

# ACF/PACF of differenced
plot_acf(ts_diff1, lags=40, ax=axes[1, 0])
axes[1, 0].set_title('ACF - First Differenced')

plot_pacf(ts_diff1, lags=40, ax=axes[1, 1])
axes[1, 1].set_title('PACF - First Differenced')

plt.tight_layout()
plt.savefig('arima_acf_pacf.png', dpi=100, bbox_inches='tight')
plt.show()

# 3. Fit ARIMA models
print("\n" + "="*60)
print("ARIMA Model Selection")
print("="*60)

# Grid search over reasonable parameter ranges
p_range = range(0, 4)
d_range = range(0, 2)
q_range = range(0, 4)

best_aic = np.inf
best_params = None
results_summary = []

for p in p_range:
    for d in d_range:
        for q in q_range:
            try:
                model = ARIMA(ts_data, order=(p, d, q))
                results = model.fit()
                results_summary.append({
                    'order': (p, d, q),
                    'AIC': results.aic,
                    'BIC': results.bic
                })
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (p, d, q)
            except:
                pass

# Top 5 models by AIC
results_df = pd.DataFrame(results_summary).sort_values('AIC').head(5)
print("\nTop 5 Models by AIC:")
print(results_df.to_string())

# 4. Fit best ARIMA and check diagnostics
print(f"\nBest ARIMA{best_params}: AIC={best_aic:.2f}")
model_arima = ARIMA(ts_data, order=best_params)
results_arima = model_arima.fit()
print(results_arima.summary())

# 5. Fit SARIMA(1,1,1)(0,1,1)₁₂
print("\n" + "="*60)
print("SARIMA(1,1,1)(0,1,1)_12 Model")
print("="*60)

from statsmodels.tsa.statespace.sarimax import SARIMAX

model_sarima = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
results_sarima = model_sarima.fit(disp=False)
print(results_sarima.summary())

# 6. Forecast comparison
forecast_steps = 24
fcst_arima = results_arima.get_forecast(steps=forecast_steps)
fcst_arima_df = fcst_arima.conf_int()
fcst_arima_df['forecast'] = fcst_arima.predicted_mean

fcst_sarima = results_sarima.get_forecast(steps=forecast_steps)
fcst_sarima_df = fcst_sarima.conf_int()
fcst_sarima_df['forecast'] = fcst_sarima.predicted_mean

# 7. Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Time series with forecast
ax = axes[0, 0]
ts_data.plot(ax=ax, label='Observed', linewidth=2)
fcst_arima_df['forecast'].plot(ax=ax, label=f'ARIMA{best_params}', 
                                color='r', linestyle='--', linewidth=2)
ax.fill_between(fcst_arima_df.index,
                fcst_arima_df.iloc[:, 0],
                fcst_arima_df.iloc[:, 1],
                alpha=0.2, color='r')
ax.set_title('ARIMA Forecast (24 months ahead)')
ax.set_ylabel('Passengers')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: SARIMA forecast
ax = axes[0, 1]
ts_data.plot(ax=ax, label='Observed', linewidth=2)
fcst_sarima_df['forecast'].plot(ax=ax, label='SARIMA(1,1,1)(0,1,1)₁₂', 
                                 color='g', linestyle='--', linewidth=2)
ax.fill_between(fcst_sarima_df.index,
                fcst_sarima_df.iloc[:, 0],
                fcst_sarima_df.iloc[:, 1],
                alpha=0.2, color='g')
ax.set_title('SARIMA Forecast (24 months ahead)')
ax.set_ylabel('Passengers')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Residual diagnostics (ARIMA)
ax = axes[1, 0]
residuals_arima = results_arima.resid
ax.plot(residuals_arima)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_title(f'ARIMA{best_params} Residuals')
ax.set_ylabel('Residual')
ax.grid(alpha=0.3)

# Plot 4: Residual diagnostics (SARIMA)
ax = axes[1, 1]
residuals_sarima = results_sarima.resid
ax.plot(residuals_sarima)
ax.axhline(y=0, color='g', linestyle='--')
ax.set_title('SARIMA(1,1,1)(0,1,1)₁₂ Residuals')
ax.set_ylabel('Residual')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('arima_forecast.png', dpi=100, bbox_inches='tight')
plt.show()

# 8. Diagnostics summary
print("\n" + "="*60)
print("Diagnostic Statistics")
print("="*60)

from scipy import stats

print(f"\nARIMA{best_params} Residuals:")
print(f"Mean: {residuals_arima.mean():.6f}")
print(f"Std Dev: {residuals_arima.std():.6f}")
print(f"Skewness: {stats.skew(residuals_arima):.6f}")
print(f"Kurtosis: {stats.kurtosis(residuals_arima):.6f}")

# Ljung-Box test for autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test_arima = acorr_ljungbox(residuals_arima, lags=10, return_df=True)
print(f"\nLjung-Box Test (ARIMA) - p-values for autocorrelation:")
print(lb_test_arima)

print(f"\n\nSARIMA Residuals:")
print(f"Mean: {residuals_sarima.mean():.6f}")
print(f"Std Dev: {residuals_sarima.std():.6f}")
print(f"Skewness: {stats.skew(residuals_sarima):.6f}")
print(f"Kurtosis: {stats.kurtosis(residuals_sarima):.6f}")

lb_test_sarima = acorr_ljungbox(residuals_sarima, lags=10, return_df=True)
print(f"\nLjung-Box Test (SARIMA) - p-values for autocorrelation:")
print(lb_test_sarima)

# 9. Model comparison
print("\n" + "="*60)
print("Model Comparison Metrics")
print("="*60)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Use in-sample predictions for comparison
pred_arima = results_arima.fittedvalues
pred_sarima = results_sarima.fittedvalues

# Align with actual data (handle different lengths)
common_idx = pred_arima.index.intersection(pred_sarima.index).intersection(ts_data.index)

y_true = ts_data.loc[common_idx]
y_pred_arima = pred_arima.loc[common_idx]
y_pred_sarima = pred_sarima.loc[common_idx]

print(f"\nARIMA{best_params}:")
print(f"  MSE: {mean_squared_error(y_true, y_pred_arima):.2f}")
print(f"  MAE: {mean_absolute_error(y_true, y_pred_arima):.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_arima)):.2f}")

print(f"\nSARIMA(1,1,1)(0,1,1)₁₂:")
print(f"  MSE: {mean_squared_error(y_true, y_pred_sarima):.2f}")
print(f"  MAE: {mean_absolute_error(y_true, y_pred_sarima):.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_sarima)):.2f}")

print(f"\nForecast Performance (24-step ahead):")
print(f"  ARIMA AIC: {results_arima.aic:.2f}")
print(f"  SARIMA AIC: {results_sarima.aic:.2f}")
print(f"  Preferred: {'ARIMA' if results_arima.aic < results_sarima.aic else 'SARIMA'} (lower AIC)")
```

## 6. Challenge Round
When ARIMA fails or needs modification:
- Multiple seasonality (weekly + yearly): Use TBATS or Prophet instead
- Exogenous variables (weather, holidays): Use ARIMAX or SARIMAX
- Regime changes: ARIMA assumes constant parameters; use Markov-switching models
- Missing data: ARIMA requires complete series, use interpolation or state-space methods
- Heteroskedastic residuals: Use GARCH-ARIMA hybrid model

## 7. Key References
- [Box & Jenkins, Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675778)
- [statsmodels ARIMA/SARIMAX Documentation](https://www.statsmodels.org/stable/tsa.html)
- [Hyndman & Athanasopoulos, Forecasting: Principles and Practice (Chapter 8-9)](https://otexts.com/fpp3/arima.html)

---
**Status:** Core forecasting model | **Complements:** Stationarity Testing, VAR Models, GARCH Models
