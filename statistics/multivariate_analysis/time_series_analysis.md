# Time Series Analysis

## 1. Concept Skeleton
**Definition:** Statistical methods analyzing sequential temporal data to identify patterns, trends, seasonality, and forecast future values  
**Purpose:** Model temporal dependence, predict future observations, decompose into components, detect anomalies  
**Prerequisites:** Autocorrelation, stationarity concepts, moving averages, differencing, spectral analysis basics

## 2. Comparative Framing
| Method | ARIMA | Exponential Smoothing | LSTM Neural Networks |
|--------|-------|----------------------|---------------------|
| **Model Type** | Parametric linear | Weighted averages | Non-linear deep learning |
| **Interpretability** | High (coefficients) | Moderate (smoothing params) | Low (black box) |
| **Data Required** | Moderate (50-100+) | Small-moderate | Large (1000s) |
| **Seasonality** | SARIMA extension | Holt-Winters | Learns automatically |

## 3. Examples + Counterexamples

**Simple Example:**  
Monthly sales data with trend and yearly seasonality: SARIMA(1,1,1)(1,1,1)₁₂ captures both patterns for forecasting

**Failure Case:**  
Structural break (e.g., COVID-19 lockdown): Past patterns invalid, model forecasts poorly → need intervention analysis or regime switching

**Edge Case:**  
Perfectly deterministic data (y_t = 2t + 5): ARIMA overfits, simple linear regression more appropriate

## 4. Layer Breakdown
```
Time Series Components:
├─ Decomposition:
│   ├─ Trend: Long-term increase/decrease
│   ├─ Seasonality: Regular periodic patterns
│   ├─ Cyclical: Non-fixed frequency oscillations
│   └─ Residual: Random fluctuations
├─ Stationarity:
│   ├─ Weak: Constant mean, variance, autocorrelation
│   ├─ Tests: ADF (Augmented Dickey-Fuller), KPSS
│   └─ Transformation: Differencing, log, Box-Cox
├─ ARIMA(p,d,q) Model:
│   ├─ AR(p): Autoregressive, yₜ depends on past p values
│   ├─ I(d): Integrated, differencing d times for stationarity
│   ├─ MA(q): Moving average, yₜ depends on past q errors
│   └─ Equation: (1-Σφᵢ Lⁱ)(1-L)ᵈ yₜ = (1+Σθⱼ Lʲ)εₜ
├─ Seasonal ARIMA: SARIMA(p,d,q)(P,D,Q)ₛ
│   └─ Additional terms for seasonal lags (s=period)
├─ Model Selection:
│   ├─ ACF/PACF plots: Identify p, q orders
│   ├─ Information Criteria: AIC, BIC (lower better)
│   └─ Grid search: Try multiple combinations
├─ Diagnostics:
│   ├─ Residual ACF: Should be white noise
│   ├─ Ljung-Box test: Test for autocorrelation
│   └─ Forecast accuracy: RMSE, MAE, MAPE
└─ Applications:
    ├─ Economic indicators (GDP, unemployment)
    ├─ Stock prices and volatility
    ├─ Weather forecasting
    └─ Demand forecasting
```

**Interaction:** Check stationarity → Transform if needed → Identify model → Estimate parameters → Diagnose → Forecast

## 5. Mini-Project
ARIMA modeling with decomposition and forecasting:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic time series with trend and seasonality
np.random.seed(42)
n = 200
time = np.arange(n)

# Components
trend = 0.5 * time
seasonality = 10 * np.sin(2 * np.pi * time / 12)
noise = np.random.normal(0, 3, n)

# Combine
ts_data = trend + seasonality + noise + 50

# Create DatetimeIndex
dates = pd.date_range(start='2010-01', periods=n, freq='M')
ts = pd.Series(ts_data, index=dates)

print("Time Series Summary:")
print(f"Length: {len(ts)}")
print(f"Start: {ts.index[0].strftime('%Y-%m')}")
print(f"End: {ts.index[-1].strftime('%Y-%m')}")
print(f"Mean: {ts.mean():.2f}")
print(f"Std: {ts.std():.2f}")

# Stationarity test (ADF)
def adf_test(series, name=''):
    result = adfuller(series, autolag='AIC')
    print(f'\n{name}Augmented Dickey-Fuller Test:')
    print(f'  ADF Statistic: {result[0]:.4f}')
    print(f'  P-value: {result[1]:.4f}')
    print(f'  Critical Values: {result[4]}')
    print(f'  Conclusion: {"Stationary" if result[1] < 0.05 else "Non-Stationary"}')
    return result[1] < 0.05

is_stationary = adf_test(ts, 'Original Series - ')

# If non-stationary, difference
if not is_stationary:
    ts_diff = ts.diff().dropna()
    adf_test(ts_diff, 'First Difference - ')

# Decomposition
decomposition = seasonal_decompose(ts, model='additive', period=12)

# ACF and PACF for model identification
acf_values = acf(ts_diff, nlags=40)
pacf_values = pacf(ts_diff, nlags=40)

# Fit ARIMA model
# Based on ACF/PACF, try ARIMA(1,1,1)
arima_model = ARIMA(ts, order=(1, 1, 1))
arima_fit = arima_model.fit()

print("\nARIMA(1,1,1) Model Summary:")
print(arima_fit.summary().tables[1])

# Fit SARIMA model (with seasonality)
sarima_model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit()

print("\nSARIMA(1,1,1)(1,1,1,12) Model AIC:")
print(f"  AIC: {sarima_fit.aic:.2f}")
print(f"  BIC: {sarima_fit.bic:.2f}")

# Forecasting
n_forecast = 24
forecast_arima = arima_fit.forecast(steps=n_forecast)
forecast_sarima = sarima_fit.forecast(steps=n_forecast)

# Create forecast index
forecast_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), 
                               periods=n_forecast, freq='M')

# Model diagnostics
residuals_arima = arima_fit.resid
residuals_sarima = sarima_fit.resid

# Ljung-Box test for residual autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals_sarima, lags=[10], return_df=True)
print(f"\nLjung-Box Test (lag 10):")
print(f"  Statistic: {lb_test['lb_stat'].values[0]:.3f}")
print(f"  P-value: {lb_test['lb_pvalue'].values[0]:.4f}")
print(f"  Conclusion: {'No autocorrelation' if lb_test['lb_pvalue'].values[0] > 0.05 else 'Autocorrelation present'}")

# Visualization
fig = plt.figure(figsize=(16, 14))

# Plot 1: Original time series
ax1 = plt.subplot(4, 2, 1)
ax1.plot(ts, linewidth=1.5)
ax1.set_title('Original Time Series')
ax1.set_xlabel('Date')
ax1.set_ylabel('Value')
ax1.grid(alpha=0.3)

# Plot 2: Decomposition - Trend
ax2 = plt.subplot(4, 2, 2)
ax2.plot(decomposition.trend, linewidth=1.5, color='orange')
ax2.set_title('Trend Component')
ax2.set_xlabel('Date')
ax2.set_ylabel('Trend')
ax2.grid(alpha=0.3)

# Plot 3: Decomposition - Seasonal
ax3 = plt.subplot(4, 2, 3)
ax3.plot(decomposition.seasonal, linewidth=1.5, color='green')
ax3.set_title('Seasonal Component (12-month period)')
ax3.set_xlabel('Date')
ax3.set_ylabel('Seasonal')
ax3.grid(alpha=0.3)

# Plot 4: Decomposition - Residual
ax4 = plt.subplot(4, 2, 4)
ax4.plot(decomposition.resid, linewidth=1, color='red', alpha=0.7)
ax4.set_title('Residual Component')
ax4.set_xlabel('Date')
ax4.set_ylabel('Residual')
ax4.grid(alpha=0.3)

# Plot 5: ACF
ax5 = plt.subplot(4, 2, 5)
plot_acf(ts_diff, lags=40, ax=ax5)
ax5.set_title('Autocorrelation Function (ACF) - First Difference')

# Plot 6: PACF
ax6 = plt.subplot(4, 2, 6)
plot_pacf(ts_diff, lags=40, ax=ax6)
ax6.set_title('Partial Autocorrelation Function (PACF)')

# Plot 7: Fitted values and forecast
ax7 = plt.subplot(4, 2, 7)
ax7.plot(ts, label='Observed', linewidth=1.5)
ax7.plot(sarima_fit.fittedvalues, label='SARIMA Fitted', linewidth=1.5, alpha=0.7)
ax7.plot(forecast_dates, forecast_sarima, label='Forecast', linewidth=2, 
         linestyle='--', color='red')
ax7.axvline(ts.index[-1], color='black', linestyle=':', alpha=0.5)
ax7.set_title('SARIMA: Fitted Values and Forecast')
ax7.set_xlabel('Date')
ax7.set_ylabel('Value')
ax7.legend()
ax7.grid(alpha=0.3)

# Plot 8: Residual diagnostics
ax8 = plt.subplot(4, 2, 8)
ax8.scatter(sarima_fit.fittedvalues, residuals_sarima, alpha=0.5)
ax8.axhline(0, color='r', linestyle='--')
ax8.set_title('Residuals vs Fitted Values')
ax8.set_xlabel('Fitted Values')
ax8.set_ylabel('Residuals')
ax8.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Additional diagnostic plots
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residual ACF
plot_acf(residuals_sarima, lags=40, ax=axes[0, 0])
axes[0, 0].set_title('Residual ACF (should be white noise)')

# Residual histogram
axes[0, 1].hist(residuals_sarima, bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Residual Distribution')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')

# Q-Q plot
from scipy import stats
stats.probplot(residuals_sarima, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality Check)')

# Residuals over time
axes[1, 1].plot(residuals_sarima, linewidth=1)
axes[1, 1].axhline(0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals Over Time')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Residual')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Model comparison
print("\nModel Comparison:")
print(f"ARIMA(1,1,1) AIC: {arima_fit.aic:.2f}")
print(f"SARIMA(1,1,1)(1,1,1,12) AIC: {sarima_fit.aic:.2f}")
print(f"Better model: {'SARIMA' if sarima_fit.aic < arima_fit.aic else 'ARIMA'} (lower AIC)")

# Forecast accuracy metrics (on training data)
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(ts, sarima_fit.fittedvalues))
mae = mean_absolute_error(ts, sarima_fit.fittedvalues)
mape = np.mean(np.abs((ts - sarima_fit.fittedvalues) / ts)) * 100

print(f"\nIn-Sample Fit Metrics (SARIMA):")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  MAPE: {mape:.2f}%")
```

## 6. Challenge Round
When is time series analysis the wrong tool?
- No temporal ordering: Use standard regression or cross-sectional methods
- Very short series (n<20): Insufficient data for reliable model identification
- Strong non-linearity: Consider machine learning (LSTM, Prophet, XGBoost)
- Multiple interacting series: Use VAR (Vector Autoregression) or dynamic factor models
- Real-time constraints: ARIMA retraining slow, consider exponential smoothing

## 7. Key References
- [Time Series Analysis (statsmodels)](https://www.statsmodels.org/stable/tsa.html)
- [ARIMA Model Explained](https://otexts.com/fpp2/arima.html)
- [Facebook Prophet (Alternative)](https://facebook.github.io/prophet/)

---
**Status:** Core temporal data analysis | **Complements:** Regression, Forecasting, Spectral Analysis
