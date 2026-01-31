# Time Series Analysis Topics Guide

**Complete reference of foundational and advanced time series concepts with categories, brief descriptions, and sources.**

---

## I. Fundamentals & Characteristics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Time Series Definition** | N/A | Sequential observations ordered by time; temporal dependence critical | [Wiki - Time Series](https://en.wikipedia.org/wiki/Time_series) |
| **Stationarity** | N/A | Mean, variance constant over time; no trend; ACF decays quickly | [Wiki - Stationary Process](https://en.wikipedia.org/wiki/Stationary_process) |
| **Non-Stationarity** | N/A | Mean/variance change; trend present; requires differencing to stabilize | [Wiki - Non-stationary](https://en.wikipedia.org/wiki/Unit_root) |
| **Trend** | N/A | Long-term directional movement; deterministic or stochastic | [Wiki - Trend](https://en.wikipedia.org/wiki/Trend_analysis) |
| **Seasonality** | N/A | Regular periodic patterns; daily/weekly/yearly cycles; multiplicative/additive | [Wiki - Seasonality](https://en.wikipedia.org/wiki/Seasonality) |
| **Autocorrelation (ACF)** | N/A | Correlation between observations at different lags; diagnostic tool | [Wiki - Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) |
| **Partial Autocorrelation (PACF)** | N/A | Correlation at lag k controlling for intermediate lags | [Wiki - PACF](https://en.wikipedia.org/wiki/Partial_autocorrelation_function) |
| **Heteroscedasticity** | N/A | Variance changes over time; ARCH/GARCH models address | [Wiki - Heteroscedasticity](https://en.wikipedia.org/wiki/Heteroscedasticity) |

---

## II. Classical Decomposition & Filtering

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Decomposition** | N/A | Separate trend, seasonality, residual; additive/multiplicative | [Wiki - Decomposition](https://en.wikipedia.org/wiki/Decomposition_of_time_series) |
| **Moving Averages** | N/A | Smooth data; lag structure; SMA, EMA, WMA | [Wiki - Moving Average](https://en.wikipedia.org/wiki/Moving_average) |
| **Exponential Smoothing** | N/A | Weighted average favoring recent data; simple, Holt, Holt-Winters | [Wiki - Exponential Smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing) |
| **Hodrick-Prescott Filter** | N/A | Extract trend minimizing deviation from actual + smoothness penalty | [Wiki - HP Filter](https://en.wikipedia.org/wiki/Hodrick%E2%80%93Prescott_filter) |
| **Kalman Filter** | N/A | Optimal state estimation; recursive Bayesian filtering; handles noise | [Wiki - Kalman](https://en.wikipedia.org/wiki/Kalman_filter) |

---

## III. Stationarity Testing & Transformations

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Augmented Dickey-Fuller (ADF) Test** | N/A | Test for unit root; H₀: non-stationary; rejects if p < 0.05 | [Wiki - ADF](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test) |
| **KPSS Test** | N/A | H₀: stationary; reverses ADF hypothesis; complementary diagnostic | [Wiki - KPSS](https://en.wikipedia.org/wiki/KPSS_test) |
| **Phillips-Perron Test** | N/A | Alternative to ADF; non-parametric unit root test | [Wiki - Phillips-Perron](https://en.wikipedia.org/wiki/Phillips%E2%80%93Perron_test) |
| **Differencing** | N/A | First difference: Δy_t = y_t - y_{t-1}; achieves stationarity | [Wiki - Differencing](https://en.wikipedia.org/wiki/Finite_difference) |
| **Log Transformations** | N/A | Stabilize variance; convert multiplicative to additive | [Wiki - Log Transform](https://en.wikipedia.org/wiki/Data_transformation_(statistics)) |
| **Detrending** | N/A | Remove trend component; linear detrending or polynomial regression | [Wiki - Detrending](https://en.wikipedia.org/wiki/Detrending) |
| **Deseasonalization** | N/A | Remove seasonal component; seasonal difference or fitted seasonal indices | [Wiki - Deseasonalization](https://en.wikipedia.org/wiki/Seasonal_adjustment) |

---

## IV. ARIMA & Box-Jenkins Framework

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Autoregressive (AR)** | N/A | AR(p): y_t = c + Σ(φ_i * y_{t-i}) + ε_t; lags of itself | [Wiki - Autoregressive](https://en.wikipedia.org/wiki/Autoregressive_model) |
| **Moving Average (MA)** | N/A | MA(q): y_t = c + ε_t + Σ(θ_j * ε_{t-j}); lags of errors | [Wiki - Moving Average Model](https://en.wikipedia.org/wiki/Moving-average_model) |
| **ARMA(p,q)** | N/A | Combines AR and MA; stationary univariate model | [Wiki - ARMA](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model) |
| **ARIMA(p,d,q)** | N/A | Integrated (d differencing); handles non-stationarity + AR/MA | [Wiki - ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) |
| **Seasonal ARIMA (SARIMA)** | N/A | ARIMA with seasonal terms: (p,d,q)×(P,D,Q,s) | [Wiki - SARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average#Seasonal_ARIMA) |
| **Box-Jenkins Methodology** | N/A | ACF/PACF → identify (p,d,q) → estimate → diagnostic tests | [Wiki - Box-Jenkins](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins) |
| **Information Criteria (AIC/BIC)** | N/A | Model selection; penalize complexity; lower values preferred | [Wiki - AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion) |

---

## V. Univariate Conditional Heteroscedasticity

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **ARCH (Autoregressive Conditional Heteroscedasticity)** | N/A | Variance depends on past squared errors; captures volatility clustering | [Wiki - ARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity) |
| **GARCH (Generalized ARCH)** | N/A | GARCH(p,q): adds lagged variance terms; more flexible volatility | [Wiki - GARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity#GARCH) |
| **Exponential GARCH (EGARCH)** | N/A | Asymmetric volatility response; leverage effect modeling | [Wiki - EGARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity#EGARCH) |
| **Volatility Clustering** | N/A | High/low volatility periods cluster together; empirical feature | [Wiki - Volatility](https://en.wikipedia.org/wiki/Volatility_(finance)) |

---

## VI. Vector Autoregression & Multivariate

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Vector Autoregression (VAR)** | N/A | Multiple series; each variable regressed on own & other lags | [Wiki - VAR](https://en.wikipedia.org/wiki/Vector_autoregression) |
| **Granger Causality** | N/A | Past values of X improve Y prediction; testing causality direction | [Wiki - Granger](https://en.wikipedia.org/wiki/Granger_causality) |
| **Cointegration** | N/A | Non-stationary series combination is stationary; long-run equilibrium | [Wiki - Cointegration](https://en.wikipedia.org/wiki/Cointegration) |
| **Vector Error Correction (VECM)** | N/A | Captures cointegrating relationships; equilibrium correction mechanism | [Wiki - VECM](https://en.wikipedia.org/wiki/Error_correction_model) |
| **Impulse Response Functions (IRF)** | N/A | How shock to one series affects others over time | [Wiki - IRF](https://en.wikipedia.org/wiki/Impulse_response) |
| **Forecast Error Variance Decomposition** | N/A | Proportion of forecast error variance explained by each shock | [Wiki - FEVD](https://en.wikipedia.org/wiki/Variance_decomposition) |

---

## VII. Advanced Time Series Models

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **State Space Models** | N/A | Hidden state + observation equations; Kalman filter framework | [Wiki - State Space](https://en.wikipedia.org/wiki/State_space) |
| **Structural Time Series** | N/A | Explicit trend, seasonal, irregular components; Bayesian approach | [Wiki - Structural](https://en.wikipedia.org/wiki/Trend_estimation) |
| **ARIMAX** | N/A | ARIMA with exogenous variables; external regressors | [Wiki - ARIMAX](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) |
| **Dynamic Regression** | N/A | Lagged effects of exogenous variables; transfer functions | [Wiki - Dynamic Regression](https://en.wikipedia.org/wiki/Regression_analysis) |
| **Threshold Models** | N/A | Different dynamics above/below threshold; nonlinear switching | [Wiki - Threshold](https://en.wikipedia.org/wiki/Threshold_model) |
| **Markov Regime-Switching** | N/A | Multiple regimes with probabilistic switching; hidden Markov models | [Wiki - Regime Switching](https://en.wikipedia.org/wiki/Markov_chain) |

---

## VIII. Forecasting & Evaluation

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **In-Sample vs Out-of-Sample Fit** | N/A | Training error vs test error; cross-validation guards overfitting | [Wiki - Overfitting](https://en.wikipedia.org/wiki/Overfitting) |
| **Rolling Window Validation** | N/A | Successive train/test splits preserving temporal order | [Wiki - Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) |
| **Forecast Horizons** | N/A | 1-step ahead, multi-step, long-term; accuracy degrades with horizon | [Wiki - Forecasting](https://en.wikipedia.org/wiki/Forecasting) |
| **Mean Absolute Error (MAE)** | N/A | Average absolute deviations; robust to outliers | [Wiki - MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) |
| **Root Mean Squared Error (RMSE)** | N/A | Penalizes large errors more; standard in finance | [Wiki - RMSE](https://en.wikipedia.org/wiki/Root_mean_square_deviation) |
| **Mean Absolute Percentage Error (MAPE)** | N/A | Percentage errors; scale-independent; undefined at y=0 | [Wiki - MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) |
| **Diebold-Mariano Test** | N/A | Compare forecast accuracy of two models; statistical test | [Wiki - DM Test](https://en.wikipedia.org/wiki/Forecast_error) |

---

## IX. Frequency Domain & Spectral Analysis

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Fourier Transform** | N/A | Convert time domain to frequency domain; identify dominant cycles | [Wiki - Fourier](https://en.wikipedia.org/wiki/Fourier_transform) |
| **Power Spectral Density (PSD)** | N/A | Frequency content of signal; peak frequency indicates dominant cycle | [Wiki - PSD](https://en.wikipedia.org/wiki/Power_spectral_density) |
| **Periodogram** | N/A | Estimate spectral density from data; frequency resolution trade-off | [Wiki - Periodogram](https://en.wikipedia.org/wiki/Periodogram) |
| **Spectral Analysis** | N/A | Identify periodicities in time series; decompose by frequency | [Wiki - Spectral](https://en.wikipedia.org/wiki/Spectral_analysis) |

---

## X. Nonlinear & Machine Learning Approaches

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Recurrent Neural Networks (RNN)** | N/A | LSTM, GRU; handle sequential dependencies; capture long-range | [Wiki - RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) |
| **Attention Mechanisms** | N/A | Transformer-based; self-attention for sequence modeling | [Wiki - Attention](https://en.wikipedia.org/wiki/Attention_mechanism) |
| **Nonlinear Autoregressive Models** | N/A | Neural network AR models; flexible feature extraction | [Wiki - NNAR](https://en.wikipedia.org/wiki/Nonlinear_regression) |
| **Genetic Algorithms** | N/A | Evolutionary optimization for model parameters/hyperparameters | [Wiki - GA](https://en.wikipedia.org/wiki/Genetic_algorithm) |
| **Prophet (Facebook)** | N/A | Additive model with trend, seasonality, holidays; interpretable | [Wiki - Prophet](https://en.wikipedia.org/wiki/Time_series#Forecasting) |

---

## XI. Practical Applications & Domain Specifics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Financial Time Series** | N/A | Asset prices, returns, volatility; stylized facts: heavy tails, clustering | [Wiki - Financial Time Series](https://en.wikipedia.org/wiki/Financial_time_series) |
| **High-Frequency Trading Data** | N/A | Tick data, microstructure noise, bid-ask spreads; regularization needed | [Wiki - Microstructure](https://en.wikipedia.org/wiki/Market_microstructure) |
| **Intraday Seasonality** | N/A | U-shaped volatility patterns; opening/closing effects | [Wiki - Intraday](https://en.wikipedia.org/wiki/Intraday) |
| **Day-of-Week Effects** | N/A | Monday effect, day-of-week patterns in returns/volatility | [Wiki - Day-of-Week](https://en.wikipedia.org/wiki/Calendar_anomaly) |
| **Jump Detection** | N/A | Identify discontinuous price movements; diffusion vs jumps | [Wiki - Jump Process](https://en.wikipedia.org/wiki/Jump_diffusion) |

---

## XII. Relationships & Integration

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **ACF & PACF Diagnostics** | N/A | Interpret correlation patterns for ARIMA order selection | [Wiki - ACF/PACF](https://en.wikipedia.org/wiki/Autocorrelation) |
| **Residual Analysis** | N/A | Check normality, independence, homoscedasticity post-model fit | [Wiki - Diagnostics](https://en.wikipedia.org/wiki/Diagnostic_plot) |
| **Ljung-Box Test** | N/A | Test residual autocorrelation; H₀: residuals independent | [Wiki - Ljung-Box](https://en.wikipedia.org/wiki/Ljung%E2%80%93Box_test) |
| **Model Comparison** | N/A | AIC/BIC trade-off; Bayes factors; choose simplest adequate model | [Wiki - Model Selection](https://en.wikipedia.org/wiki/Model_selection) |

---

## XIII. Meta-Topics & Foundations

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Time Series Core Concepts** | N/A | Integrated overview: stationarity, ACF/PACF, ARIMA foundations | [Wiki - Core](https://en.wikipedia.org/wiki/Time_series) |
| **Wiener Process (Brownian Motion)** | N/A | Continuous-time model; increments independent, normally distributed | [Wiki - Wiener](https://en.wikipedia.org/wiki/Wiener_process) |
| **Itô Calculus** | N/A | Stochastic integration; handles path dependence in diffusion models | [Wiki - Itô](https://en.wikipedia.org/wiki/It%C3%B4_calculus) |
| **Stochastic Differential Equations (SDE)** | N/A | dy_t = μ(y,t)dt + σ(y,t)dW_t; continuous-time dynamics | [Wiki - SDE](https://en.wikipedia.org/wiki/Stochastic_differential_equation) |
| **Ergodicity** | N/A | Time average equals ensemble average; long-run behavior stability | [Wiki - Ergodic](https://en.wikipedia.org/wiki/Ergodic_theory) |

---

## Reference Sources

| Source | URL | Coverage |
|--------|-----|----------|
| **Wikipedia Time Series** | https://en.wikipedia.org/wiki/Time_series | Comprehensive overview; 50+ related topics |
| **Forecasting: Principles & Practice** | https://otexts.com/fpp2/ | ARIMA, exponential smoothing, practical examples |
| **Box, Jenkins, Reinsel & Ljung** | https://onlinelibrary.wiley.com/doi/book/10.1002/9781118674912 | Canonical Box-Jenkins reference; ARIMA theory |
| **Brockwell & Davis** | https://link.springer.com/book/10.1007/978-1-4419-0320-4 | Time Series Theory & Methods; rigorous foundation |
| **Hamilton** | https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis | Advanced macroeconomic/financial applications |

---

## Quick Stats

- **Total Topics Documented**: 70+
- **Major Categories**: 13
- **Classical Methods (ARIMA/GARCH)**: 20+ topics
- **Advanced Methods (VAR/VECM/ML)**: 15+ topics
- **Application/Testing**: 20+ topics
- **Coverage**: Fundamentals → Classical → Multivariate → Modern ML → Applications

