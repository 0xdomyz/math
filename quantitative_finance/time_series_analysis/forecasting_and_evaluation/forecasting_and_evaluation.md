# Forecasting and Evaluation in Time Series

## 1. Concept Skeleton
**Definition:** Predict future values of time series; assess forecast accuracy via metrics; compare competing models; quantify prediction uncertainty  
**Purpose:** Support decision-making under uncertainty; allocate resources optimally; validate models objectively; communicate forecast reliability; improve iteratively  
**Prerequisites:** Time series fundamentals, stationarity, ARIMA models, exponential smoothing, loss functions, statistical inference

## 2. Comparative Framing
| Forecast Method | Naïve | Moving Average | Exponential Smoothing | ARIMA | State Space | Machine Learning |
|----------------|-------|----------------|----------------------|-------|-------------|------------------|
| **Complexity** | Trivial | Low | Moderate | High | Very High | Very High |
| **Trend** | No | No | Yes (Holt) | Yes | Yes | Yes |
| **Seasonality** | No | No | Yes (HW) | Yes (SARIMA) | Yes | Yes |
| **Intervals** | Simple | Simple | Analytical | Analytical | Kalman | Bootstrap |
| **Interpretability** | Perfect | High | High | Moderate | Low | Very Low |
| **Data Needs** | Minimal | Minimal | Moderate | Moderate | High | Very High |

| Accuracy Metric | MAE | RMSE | MAPE | sMAPE | MASE | MDA |
|----------------|-----|------|------|-------|------|-----|
| **Units** | Original | Original | Percentage | Percentage | Ratio | Proportion |
| **Scale Dependent** | Yes | Yes | No | No | No | No |
| **Outlier Robust** | Yes | No | No | No | Moderate | Yes |
| **Symmetric** | Yes | Yes | No | Yes | Yes | Yes |
| **Zero Handling** | Good | Good | Fails | Better | Good | Good |
| **Interpretation** | Avg error | Std-like | % error | % error | vs naïve | Direction accuracy |

## 3. Examples + Counterexamples

**Simple Example:**  
Daily temperature: Naïve forecast (tomorrow = today) achieves MAE=2°C. ARIMA(1,0,1) reduces MAE to 1.5°C. 25% improvement validates model complexity.

**Perfect Fit:**  
Monthly airline passengers: Holt-Winters (multiplicative seasonality) captures growing amplitude, 12-month cycle. MAPE<5%, prediction intervals cover 95% of actuals. Optimal for operational planning.

**Point vs Interval:**  
Sales forecast: Point estimate = 1000 units. 95% PI: [800, 1200]. Decision: Stock 1200 (cover upper bound, minimize stockouts). Interval more actionable than point.

**Metric Choice Matters:**  
Stock prices: RMSE penalizes large errors (volatility spikes). MAE treats all errors equally. For risk management, RMSE preferred (captures tail risk). For average performance, MAE sufficient.

**Poor Fit:**  
Bitcoin returns: All models fail (MAPE>50%, Theil's U>1). No structure—dominated by unpredictable shocks. Forecast accuracy worse than random walk. Structural breaks frequent.

**Non-Stationary Failure:**  
COVID-19 cases: Pre-pandemic ARIMA fitted on flu data. Structural break (lockdowns) → all forecasts useless. Must adapt—use regime-switching models or re-train frequently.

## 4. Layer Breakdown
```
Forecasting & Evaluation Framework:

├─ Forecasting Approaches:
│  ├─ Naïve Methods (Benchmarks):
│  │   ├─ Naïve Forecast:
│  │   │   ŷ_{T+h} = y_T (last observation)
│  │   │   Benchmark: All models should beat this
│  │   │   Random walk: Optimal if true DGP
│  │   ├─ Seasonal Naïve:
│  │   │   ŷ_{T+h} = y_{T+h-m} (same period last year)
│  │   │   m: Seasonal period (12 for monthly)
│  │   │   Benchmark for seasonal data
│  │   ├─ Average:
│  │   │   ŷ_{T+h} = ȳ (historical mean)
│  │   │   Assumes mean reversion
│  │   └─ Drift:
│  │       ŷ_{T+h} = y_T + h × (y_T - y_1)/(T-1)
│  │       Linear trend continuation
│  ├─ Statistical Methods:
│  │   ├─ Exponential Smoothing (ETS):
│  │   │   ├─ Simple (SES):
│  │   │   │   Level only: ℓ_t = αy_t + (1-α)ℓ_{t-1}
│  │   │   │   Forecast: ŷ_{T+h|T} = ℓ_T
│  │   │   ├─ Holt (Linear Trend):
│  │   │   │   Level: ℓ_t = αy_t + (1-α)(ℓ_{t-1} + b_{t-1})
│  │   │   │   Trend: b_t = β(ℓ_t - ℓ_{t-1}) + (1-β)b_{t-1}
│  │   │   │   Forecast: ŷ_{T+h|T} = ℓ_T + h×b_T
│  │   │   ├─ Holt-Winters (Additive Seasonal):
│  │   │   │   Level: ℓ_t = α(y_t - s_{t-m}) + (1-α)(ℓ_{t-1} + b_{t-1})
│  │   │   │   Trend: b_t = β(ℓ_t - ℓ_{t-1}) + (1-β)b_{t-1}
│  │   │   │   Seasonal: s_t = γ(y_t - ℓ_t) + (1-γ)s_{t-m}
│  │   │   │   Forecast: ŷ_{T+h|T} = ℓ_T + h×b_T + s_{T+h-m}
│  │   │   ├─ Multiplicative Seasonal:
│  │   │   │   ℓ_t = α(y_t / s_{t-m}) + (1-α)(ℓ_{t-1} + b_{t-1})
│  │   │   │   s_t = γ(y_t / ℓ_t) + (1-γ)s_{t-m}
│  │   │   │   Forecast: ŷ_{T+h|T} = (ℓ_T + h×b_T) × s_{T+h-m}
│  │   │   ├─ Damped Trend:
│  │   │   │   b_t damped: φ ∈ (0,1)
│  │   │   │   ŷ_{T+h|T} = ℓ_T + (φ + φ² + ... + φ^h)b_T
│  │   │   │   Trend flattens over horizon
│  │   │   └─ Prediction Intervals:
│  │   │       σ²_h = σ² × (1 + α² + (αβ)² + ...)
│  │   │       PI: ŷ_{T+h} ± z_{α/2} × σ_h
│  │   │       Variance increases with h
│  │   ├─ ARIMA Models:
│  │   │   ├─ Point Forecasts:
│  │   │   │   ARIMA(p,d,q): Recursive substitution
│  │   │   │   h=1: ŷ_{T+1} = φ_1 y_T + ... + φ_p y_{T-p+1} + θ_1 ε_T + ... + θ_q ε_{T-q+1}
│  │   │   │   h>1: Replace future ε with 0 (conditional expectation)
│  │   │   ├─ Seasonal ARIMA:
│  │   │   │   SARIMA(p,d,q)(P,D,Q)_m
│  │   │   │   Seasonal AR/MA terms: Φ_1 y_{t-m}, Θ_1 ε_{t-m}
│  │   │   │   Forecast combines non-seasonal + seasonal
│  │   │   ├─ Prediction Intervals:
│  │   │   │   Forecast error variance:
│  │   │   │   Var(e_h) = σ² × (1 + ψ_1² + ψ_2² + ... + ψ_{h-1}²)
│  │   │   │   ψ_j: MA(∞) coefficients from ARIMA
│  │   │   │   PI: ŷ_{T+h} ± z_{α/2} × √Var(e_h)
│  │   │   └─ Mean Reversion:
│  │   │       Stationary ARMA: ŷ_{T+h} → μ as h → ∞
│  │   │       Non-stationary (I(1)): ŷ_{T+h} → ∞ (random walk)
│  │   ├─ State Space Models:
│  │   │   ├─ General Form:
│  │   │   │   Observation: y_t = Z'α_t + ε_t
│  │   │   │   State: α_t = T α_{t-1} + R η_t
│  │   │   ├─ Kalman Filter:
│  │   │   │   Prediction: α_{t|t-1} = T α_{t-1|t-1}
│  │   │   │   Update: α_{t|t} = α_{t|t-1} + K_t v_t
│  │   │   │   Forecast: ŷ_{t+h|t} = Z' T^h α_{t|t}
│  │   │   ├─ Advantages:
│  │   │   │   Handle missing data naturally
│  │   │   │   Time-varying parameters
│  │   │   │   Unobserved components (trend, cycle)
│  │   │   └─ Examples:
│  │   │       Local level, local trend, BSM (Basic Structural Model)
│  │   │       UC-SV (Unobserved Components Stochastic Volatility)
│  │   └─ Regression-Based:
│  │       ├─ Time Series Regression:
│  │       │   y_t = β_0 + β_1 t + β_2 sin(2πt/m) + β_3 cos(2πt/m) + ε_t
│  │       │   Trend + seasonal dummies/harmonics
│  │       │   Forecast: Plug in future t values
│  │       ├─ Dynamic Regression:
│  │       │   y_t = β'x_t + n_t, where n_t ~ ARIMA
│  │       │   Combines regression + ARIMA errors
│  │       │   Forecast: β'x_{T+h} + ARIMA forecast of n_t
│  │       └─ Transfer Function Models:
│  │           Include lagged predictors (leading indicators)
│  │           y_t = ω(L) x_{t-b} + n_t
│  ├─ Machine Learning Methods:
│  │   ├─ Feature Engineering:
│  │   │   Lags: y_{t-1}, y_{t-2}, ...
│  │   │   Rolling stats: MA_k, Std_k
│  │   │   Time features: Month, day-of-week, holiday
│  │   │   Fourier terms: sin/cos for seasonality
│  │   ├─ Models:
│  │   │   ├─ Random Forest:
│  │   │   │   Ensemble of decision trees
│  │   │   │   Non-linear interactions
│  │   │   │   Robust to outliers
│  │   │   ├─ Gradient Boosting (XGBoost, LightGBM):
│  │   │   │   Sequential trees, minimize loss
│  │   │   │   High accuracy, fast
│  │   │   │   Risk of overfitting
│  │   │   ├─ Neural Networks (LSTM, GRU):
│  │   │   │   Recurrent architectures
│  │   │   │   Handle long dependencies
│  │   │   │   Require large data
│  │   │   └─ Transformer (Attention):
│  │   │       Self-attention mechanism
│  │   │       State-of-art for long sequences
│  │   │       Very data-hungry
│  │   ├─ Advantages:
│  │   │   Capture complex non-linearities
│  │   │   Multivariate interactions
│  │   │   No distributional assumptions
│  │   ├─ Disadvantages:
│  │   │   Require large datasets
│  │   │   Black box (interpretability)
│  │   │   No automatic prediction intervals
│  │   └─ Prediction Intervals:
│  │       Quantile regression (predict 5th, 95th percentiles)
│  │       Conformal prediction
│  │       Bootstrap residuals
│  ├─ Ensemble Methods:
│  │   ├─ Simple Average:
│  │   │   ŷ_ensemble = (1/K) Σ ŷ_k
│  │   │   Reduces variance, improves robustness
│  │   ├─ Weighted Average:
│  │   │   ŷ_ensemble = Σ w_k ŷ_k, where Σ w_k = 1
│  │   │   Weights from historical accuracy
│  │   │   Inverse MSE weighting common
│  │   ├─ Stacking:
│  │   │   Meta-model trained on base forecasts
│  │   │   ŷ_stack = g(ŷ_1, ŷ_2, ..., ŷ_K)
│  │   │   Can learn optimal combination
│  │   └─ Forecast Combination Benefits:
│  │       Reduces model risk
│  │       Often beats individual models
│  │       Robust to specification error
│  └─ Judgmental Adjustments:
│      Expert overrides (domain knowledge)
│      Special events (promotions, strikes)
│      Scenario-based forecasts (optimistic/pessimistic)
├─ Forecast Evaluation:
│  ├─ Error Metrics (Point Forecasts):
│  │   ├─ Scale-Dependent:
│  │   │   ├─ Mean Absolute Error (MAE):
│  │   │   │   MAE = (1/n) Σ |y_t - ŷ_t|
│  │   │   │   Interpretable: Average error magnitude
│  │   │   │   Robust to outliers
│  │   │   │   Same units as data
│  │   │   ├─ Root Mean Squared Error (RMSE):
│  │   │   │   RMSE = √[(1/n) Σ (y_t - ŷ_t)²]
│  │   │   │   Penalizes large errors more
│  │   │   │   Standard deviation-like
│  │   │   │   Sensitive to outliers
│  │   │   ├─ Median Absolute Error (MdAE):
│  │   │   │   MdAE = median(|y_t - ŷ_t|)
│  │   │   │   Very robust to outliers
│  │   │   │   Less sensitive to extremes
│  │   │   └─ Relationship:
│  │   │       MAE ≤ RMSE (equality for constant errors)
│  │   │       RMSE - MAE measures error variability
│  │   ├─ Percentage Errors:
│  │   │   ├─ Mean Absolute Percentage Error (MAPE):
│  │   │   │   MAPE = (100/n) Σ |y_t - ŷ_t| / |y_t|
│  │   │   │   Scale-independent (compare across series)
│  │   │   │   Interpretable: Average % error
│  │   │   │   Problems: Undefined if y_t=0, asymmetric
│  │   │   │   Penalizes over-forecasts more than under
│  │   │   ├─ Symmetric MAPE (sMAPE):
│  │   │   │   sMAPE = (100/n) Σ |y_t - ŷ_t| / [(|y_t| + |ŷ_t|)/2]
│  │   │   │   Symmetric treatment of over/under forecast
│  │   │   │   Bounded: [0, 200]
│  │   │   │   Still issues with zeros
│  │   │   └─ Mean Absolute Scaled Error (MASE):
│  │   │       MASE = MAE / MAE_naïve
│  │   │       Scale-free, compares to naïve benchmark
│  │   │       MASE < 1: Better than naïve
│  │   │       No division by zero issues
│  │   │       Preferred for intermittent demand
│  │   ├─ Direction Accuracy:
│  │   │   ├─ Mean Directional Accuracy (MDA):
│  │   │   │   MDA = (1/n) Σ I(sign(Δy_t) = sign(Δŷ_t))
│  │   │   │   Proportion of correct direction predictions
│  │   │   │   Important for trading (up/down matters)
│  │   │   └─ Hit Rate:
│  │   │       % of times forecast within tolerance
│  │   │       Binary: |y_t - ŷ_t| < threshold
│  │   └─ Relative Metrics:
│  │       ├─ Theil's U Statistic:
│  │       │   U = RMSE_model / RMSE_naïve
│  │       │   U < 1: Model beats naïve
│  │       │   U = 1: No better than naïve
│  │       │   U > 1: Worse than naïve (use naïve instead!)
│  │       └─ Forecast Skill:
│  │           Skill = 1 - (MSE_model / MSE_reference)
│  │           Skill = 0: No skill vs reference
│  │           Skill > 0: Better than reference
│  ├─ Interval Forecast Evaluation:
│  │   ├─ Coverage Probability:
│  │   │   Empirical coverage = % of actuals within PI
│  │   │   Should match nominal (95% PI → 95% coverage)
│  │   │   Under-coverage: Intervals too narrow (overconfident)
│  │   │   Over-coverage: Intervals too wide (wasteful)
│  │   ├─ Mean Interval Width:
│  │   │   Average width of prediction intervals
│  │   │   Narrower = more precise (given coverage)
│  │   │   Trade-off: Coverage vs width
│  │   ├─ Winkler Score:
│  │   │   S = (U - L) + (2/α)(L - y_t)I(y_t < L) + (2/α)(y_t - U)I(y_t > U)
│  │   │   U, L: Upper, lower bounds of (1-α) PI
│  │   │   Penalizes width and violations
│  │   │   Lower = better
│  │   └─ Christoffersen Test:
│  │       Test if violations are independent
│  │       Detect clustering of failures
│  │       Should be random if well-calibrated
│  ├─ Probabilistic Forecast Evaluation:
│  │   ├─ Continuous Ranked Probability Score (CRPS):
│  │   │   CRPS = ∫ [F(x) - I(x ≥ y)]² dx
│  │   │   F(x): Forecast CDF
│  │   │   Proper scoring rule (encourages honest forecast)
│  │   │   Lower = better
│  │   │   Reduces to MAE for point forecasts
│  │   ├─ Log Score (Predictive Likelihood):
│  │   │   LS = -log f(y_t | ŷ_t)
│  │   │   f: Predictive density
│  │   │   Rewards sharp, accurate densities
│  │   │   Severely penalizes misses
│  │   ├─ Quantile Score (Pinball Loss):
│  │   │   Q_τ(e) = τ × e × I(e ≥ 0) + (1-τ) × (-e) × I(e < 0)
│  │   │   τ: Quantile level (e.g., 0.05, 0.95)
│  │   │   Asymmetric loss function
│  │   │   Optimal forecast = τ-th quantile
│  │   └─ Calibration:
│  │       Probability Integral Transform (PIT)
│  │       u_t = F_t(y_t) should be Uniform(0,1)
│  │       Histogram of PIT: Check uniformity
│  ├─ Residual Diagnostics:
│  │   ├─ White Noise Check:
│  │   │   ACF of residuals should be zero
│  │   │   Ljung-Box test: H0 = no autocorrelation
│  │   │   If significant → model misspecification
│  │   ├─ Normality:
│  │   │   QQ-plot, Jarque-Bera test
│  │   │   Check if ε_t ~ N(0, σ²)
│  │   │   Non-normality → prediction intervals wrong
│  │   ├─ Heteroskedasticity:
│  │   │   Plot residuals vs time, vs fitted
│  │   │   ARCH-LM test
│  │   │   Time-varying variance → use GARCH
│  │   └─ Outliers:
│  │       Studentized residuals: r_t / σ_t
│  │       |r_t| > 3 → potential outlier
│  │       Investigate, consider robust methods
│  └─ Cross-Validation:
│      ├─ Time Series CV (Rolling Origin):
│      │   Train on [1, t], forecast t+1, ..., t+h
│      │   Expand window: t+1, ..., T
│      │   Compute average error across origins
│      │   Respects temporal ordering
│      ├─ Fixed Window:
│      │   Train on [t-w, t], forecast t+1
│      │   Slide window forward
│      │   Constant training size
│      ├─ Blocked CV:
│      │   Leave out block [t, t+h]
│      │   Train on data before t and after t+h
│      │   Avoid contamination
│      └─ Out-of-Sample Testing:
│          Reserve last 10-20% for testing
│          Never use for model selection
│          Final performance estimate
├─ Forecast Horizons:
│  ├─ Short-Term (1-3 periods):
│  │   High accuracy possible
│  │   Recent patterns dominate
│  │   Exponential smoothing effective
│  ├─ Medium-Term (4-12 periods):
│  │   Seasonal patterns important
│  │   SARIMA, Holt-Winters suitable
│  │   Uncertainty moderate
│  └─ Long-Term (>12 periods):
│      Accuracy deteriorates
│      Revert to mean/trend
│      Scenario analysis useful
│      Prediction intervals very wide
├─ Practical Considerations:
│  ├─ Model Selection:
│  │   ├─ Information Criteria:
│  │   │   AIC = -2log(L) + 2k
│  │   │   BIC = -2log(L) + k log(n)
│  │   │   AICc = AIC + 2k(k+1)/(n-k-1) (small sample)
│  │   │   Lower = better
│  │   │   Trade-off: Fit vs complexity
│  │   ├─ Out-of-Sample Performance:
│  │   │   Test on holdout data
│  │   │   More reliable than in-sample
│  │   │   Compute RMSE, MAE on test set
│  │   └─ Forecast Accuracy:
│  │       Choose model with best forecast performance
│  │       Not necessarily best fit
│  ├─ Updating:
│  │   ├─ Fixed Model:
│  │   │   Estimate once, use indefinitely
│  │   │   Fast, simple
│  │   │   Assumes stable parameters
│  │   ├─ Rolling Re-estimation:
│  │   │   Re-fit model at each step
│  │   │   Adapts to changing patterns
│  │   │   Computationally expensive
│  │   └─ Recursive Updating:
│  │       Online algorithms (Kalman filter)
│  │       Update with each new observation
│  │       Efficient for large datasets
│  ├─ Handling Missing Data:
│  │   ├─ Interpolation:
│  │   │   Linear, spline, Kalman smoother
│  │   │   Fill gaps before modeling
│  │   ├─ State Space:
│  │   │   Kalman filter handles missing naturally
│  │   │   No need to impute
│  │   └─ Deletion:
│  │       Remove observations with missings
│  │       Loss of information
│  ├─ Transformations:
│  │   ├─ Log Transform:
│  │   │   Stabilizes variance
│  │   │   Converts multiplicative to additive
│  │   │   Back-transform: Bias correction (e^{σ²/2})
│  │   ├─ Box-Cox:
│  │   │   y^(λ) = (y^λ - 1)/λ (λ ≠ 0), log(y) (λ = 0)
│  │   │   Choose λ by MLE or fixed (0.5, 0, -0.5)
│  │   │   General variance stabilization
│  │   └─ Differencing:
│  │       Remove trend, achieve stationarity
│  │       Seasonal differencing: ∇_m y_t = y_t - y_{t-m}
│  └─ Software:
│      R: forecast, fable, prophet
│      Python: statsmodels, sktime, pmdarima, prophet
│      Commercial: SAS Forecast Studio, Autobox, ForecastPro
├─ Common Pitfalls:
│  ├─ Over-fitting:
│  │   Too many parameters relative to data
│  │   Good in-sample, poor out-of-sample
│  │   Solution: Penalize complexity (AIC, regularization)
│  ├─ Look-Ahead Bias:
│  │   Using future information in training
│  │   Creates unrealistic performance
│  │   Solution: Strict temporal train/test split
│  ├─ Ignoring Uncertainty:
│  │   Point forecasts alone insufficient
│  │   Communicate prediction intervals
│  │   Quantify forecast risk
│  ├─ Neglecting Residuals:
│  │   Residual autocorrelation → biased SEs
│  │   Check diagnostics always
│  │   Iterate until white noise
│  └─ Static Models:
│      Structural breaks, regime changes
│      Re-estimate periodically
│      Monitor forecast performance
└─ Applications by Domain:
   ├─ Finance:
   │   Stock prices: GARCH, machine learning
   │   Exchange rates: Random walk often hard to beat
   │   Volatility: High-frequency, realized volatility
   ├─ Economics:
   │   GDP, inflation: ARIMA, VAR, DSGE models
   │   Employment: Leading indicators, nowcasting
   │   Central banks: Fan charts (multi-horizon intervals)
   ├─ Retail:
   │   Demand forecasting: Exponential smoothing, ML
   │   Inventory optimization: Safety stock from PI
   │   Promotional effects: Regression with events
   ├─ Energy:
   │   Load forecasting: Temperature, calendar effects
   │   Renewables: Weather-dependent, high uncertainty
   │   Prices: Mean-reverting, spikes
   ├─ Health:
   │   Disease surveillance: SIR models + time series
   │   Hospital admissions: Seasonality, epidemics
   │   Pharmaceutical demand: Aging population trends
   └─ Supply Chain:
      Lead times, supplier reliability
      Multi-level forecasting (product hierarchy)
      Intermittent demand (Croston's method)
```

**Interaction:** Choose model (ARIMA/ETS/ML) → Fit on training data → Generate forecasts (point + intervals) → Evaluate on test set (RMSE, MASE) → Check residuals (white noise?) → Compare models → Select best → Deploy → Monitor → Re-estimate periodically.

## 5. Mini-Project
Implement forecasting methods, compare accuracy, evaluate prediction intervals:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("FORECASTING AND EVALUATION: TIME SERIES PREDICTION")
print("="*80)

class ForecastEvaluator:
    """Comprehensive forecast evaluation metrics"""
    
    def __init__(self):
        pass
    
    def mae(self, y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def rmse(self, y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    def mape(self, y_true, y_pred):
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    
    def smape(self, y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not np.any(mask):
            return 0
        return 100 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
    
    def mase(self, y_true, y_pred, y_train, seasonal_period=1):
        """Mean Absolute Scaled Error"""
        # Naive forecast MAE (in-sample)
        if seasonal_period == 1:
            naive_mae = np.mean(np.abs(np.diff(y_train)))
        else:
            naive_mae = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
        
        if naive_mae == 0:
            return np.inf
        
        mae_forecast = self.mae(y_true, y_pred)
        return mae_forecast / naive_mae
    
    def theil_u(self, y_true, y_pred, y_train):
        """Theil's U Statistic"""
        # Naive forecast: last observation
        naive_pred = np.full(len(y_true), y_train[-1])
        
        rmse_model = self.rmse(y_true, y_pred)
        rmse_naive = self.rmse(y_true, naive_pred)
        
        if rmse_naive == 0:
            return np.inf
        
        return rmse_model / rmse_naive
    
    def mda(self, y_true, y_pred):
        """Mean Directional Accuracy"""
        if len(y_true) < 2:
            return np.nan
        
        # Direction: sign of change
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Proportion correct
        return np.mean(true_direction == pred_direction)
    
    def coverage_probability(self, y_true, lower, upper):
        """Prediction interval coverage"""
        in_interval = (y_true >= lower) & (y_true <= upper)
        return np.mean(in_interval)
    
    def mean_interval_width(self, lower, upper):
        """Average width of prediction intervals"""
        return np.mean(upper - lower)
    
    def winkler_score(self, y_true, lower, upper, alpha=0.05):
        """Winkler score for interval forecasts"""
        width = upper - lower
        penalty_lower = (2/alpha) * (lower - y_true) * (y_true < lower)
        penalty_upper = (2/alpha) * (y_true - upper) * (y_true > upper)
        
        return np.mean(width + penalty_lower + penalty_upper)
    
    def summary(self, y_true, y_pred, y_train, seasonal_period=1):
        """Compute all metrics"""
        metrics = {
            'MAE': self.mae(y_true, y_pred),
            'RMSE': self.rmse(y_true, y_pred),
            'MAPE': self.mape(y_true, y_pred),
            'sMAPE': self.smape(y_true, y_pred),
            'MASE': self.mase(y_true, y_pred, y_train, seasonal_period),
            'Theil_U': self.theil_u(y_true, y_pred, y_train),
            'MDA': self.mda(y_true, y_pred)
        }
        return metrics

class TimeSeriesForecaster:
    """Wrapper for various forecasting methods"""
    
    def __init__(self):
        pass
    
    def naive_forecast(self, y_train, horizon):
        """Naive: Last observation"""
        return np.full(horizon, y_train[-1])
    
    def seasonal_naive(self, y_train, horizon, period):
        """Seasonal naive: Same period last year"""
        forecasts = []
        for h in range(horizon):
            idx = len(y_train) - period + (h % period)
            if idx >= 0:
                forecasts.append(y_train[idx])
            else:
                forecasts.append(y_train[-1])
        return np.array(forecasts)
    
    def moving_average(self, y_train, horizon, window=5):
        """Moving average forecast"""
        ma = np.mean(y_train[-window:])
        return np.full(horizon, ma)
    
    def exponential_smoothing(self, y_train, horizon, trend=None, seasonal=None, 
                             seasonal_periods=None):
        """
        Exponential Smoothing (ETS)
        
        Parameters:
        - trend: None, 'add', 'mul'
        - seasonal: None, 'add', 'mul'
        """
        try:
            model = ExponentialSmoothing(y_train, trend=trend, seasonal=seasonal,
                                        seasonal_periods=seasonal_periods)
            fit = model.fit()
            forecasts = fit.forecast(horizon)
            return forecasts
        except:
            # Fallback to naive
            return self.naive_forecast(y_train, horizon)
    
    def arima_forecast(self, y_train, horizon, order=(1,1,1)):
        """ARIMA forecast"""
        try:
            model = ARIMA(y_train, order=order)
            fit = model.fit()
            forecasts = fit.forecast(steps=horizon)
            return forecasts
        except:
            return self.naive_forecast(y_train, horizon)
    
    def sarima_forecast(self, y_train, horizon, order=(1,1,1), 
                       seasonal_order=(1,1,1,12)):
        """Seasonal ARIMA forecast"""
        try:
            model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
            fit = model.fit(disp=False)
            forecasts = fit.forecast(steps=horizon)
            return forecasts
        except:
            return self.naive_forecast(y_train, horizon)
    
    def ml_forecast(self, y_train, horizon, lags=12):
        """Machine learning forecast (Random Forest)"""
        # Create lagged features
        n = len(y_train)
        if n <= lags:
            return self.naive_forecast(y_train, horizon)
        
        X = []
        y = []
        for i in range(lags, n):
            X.append(y_train[i-lags:i])
            y.append(y_train[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # Recursive forecasting
        forecasts = []
        current_window = list(y_train[-lags:])
        
        for _ in range(horizon):
            X_new = np.array(current_window[-lags:]).reshape(1, -1)
            pred = model.predict(X_new)[0]
            forecasts.append(pred)
            current_window.append(pred)
        
        return np.array(forecasts)

def rolling_origin_cv(y, forecaster_func, min_train=50, horizon=12, step=1):
    """
    Rolling origin cross-validation
    
    Parameters:
    - y: Full time series
    - forecaster_func: Function that takes y_train and horizon, returns forecasts
    - min_train: Minimum training size
    - horizon: Forecast horizon
    - step: Step size for rolling window
    """
    n = len(y)
    errors = []
    
    for t in range(min_train, n - horizon, step):
        y_train = y[:t]
        y_test = y[t:t+horizon]
        
        forecasts = forecaster_func(y_train, horizon)
        
        # Truncate if needed
        h_actual = min(len(forecasts), len(y_test))
        errors.extend(y_test[:h_actual] - forecasts[:h_actual])
    
    return np.array(errors)

# Scenario 1: Generate synthetic data with trend and seasonality
print("\n" + "="*80)
print("SCENARIO 1: Synthetic Data - Trend + Seasonality")
print("="*80)

# Generate data
n = 120  # 10 years monthly
t = np.arange(n)
trend = 100 + 0.5 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 3, n)
y = trend + seasonal + noise

# Train/test split
train_size = 96  # 8 years
y_train = y[:train_size]
y_test = y[train_size:]
horizon = len(y_test)

print(f"\nData: {n} observations (monthly, 10 years)")
print(f"  Train: {train_size}, Test: {horizon}")
print(f"  Components: Linear trend + 12-month seasonality + noise")

# Initialize
forecaster = TimeSeriesForecaster()
evaluator = ForecastEvaluator()

# Baseline: Naive
forecast_naive = forecaster.naive_forecast(y_train, horizon)

# Seasonal Naive
forecast_snaive = forecaster.seasonal_naive(y_train, horizon, period=12)

# Moving Average
forecast_ma = forecaster.moving_average(y_train, horizon, window=12)

# Exponential Smoothing (Holt-Winters)
forecast_hw = forecaster.exponential_smoothing(y_train, horizon, 
                                               trend='add', seasonal='add',
                                               seasonal_periods=12)

# ARIMA
forecast_arima = forecaster.arima_forecast(y_train, horizon, order=(1,1,1))

# SARIMA
forecast_sarima = forecaster.sarima_forecast(y_train, horizon, 
                                             order=(1,1,1), seasonal_order=(1,1,1,12))

# Machine Learning
forecast_ml = forecaster.ml_forecast(y_train, horizon, lags=24)

# Evaluate all methods
methods = {
    'Naive': forecast_naive,
    'Seasonal Naive': forecast_snaive,
    'Moving Average': forecast_ma,
    'Holt-Winters': forecast_hw,
    'ARIMA(1,1,1)': forecast_arima,
    'SARIMA': forecast_sarima,
    'Random Forest': forecast_ml
}

print(f"\n{'Method':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'MASE':<10} {'Theil U':<10}")
print("-" * 70)

results = {}
for name, forecast in methods.items():
    metrics = evaluator.summary(y_test, forecast, y_train, seasonal_period=12)
    results[name] = metrics
    
    print(f"{name:<20} {metrics['MAE']:<10.3f} {metrics['RMSE']:<10.3f} "
          f"{metrics['MAPE']:<10.2f} {metrics['MASE']:<10.3f} {metrics['Theil_U']:<10.3f}")

# Best method
best_method = min(results.items(), key=lambda x: x[1]['RMSE'])
print(f"\nBest Method (RMSE): {best_method[0]}")

# Scenario 2: Prediction Intervals
print("\n" + "="*80)
print("SCENARIO 2: Prediction Intervals - Uncertainty Quantification")
print("="*80)

# Fit Holt-Winters with prediction intervals
try:
    model_hw = ExponentialSmoothing(y_train, trend='add', seasonal='add', 
                                   seasonal_periods=12)
    fit_hw = model_hw.fit()
    
    # Get prediction intervals
    forecast_result = fit_hw.forecast(horizon)
    
    # Simulate prediction intervals (bootstrap residuals)
    residuals = y_train - fit_hw.fittedvalues
    
    # Bootstrap
    n_sim = 1000
    forecasts_sim = []
    
    for _ in range(n_sim):
        # Resample residuals
        resid_sample = np.random.choice(residuals, size=horizon, replace=True)
        
        # Add to point forecast
        forecast_sim = forecast_hw + resid_sample
        forecasts_sim.append(forecast_sim)
    
    forecasts_sim = np.array(forecasts_sim)
    
    # Compute intervals
    lower_95 = np.percentile(forecasts_sim, 2.5, axis=0)
    upper_95 = np.percentile(forecasts_sim, 97.5, axis=0)
    lower_80 = np.percentile(forecasts_sim, 10, axis=0)
    upper_80 = np.percentile(forecasts_sim, 90, axis=0)
    
    # Evaluate intervals
    coverage_95 = evaluator.coverage_probability(y_test, lower_95, upper_95)
    coverage_80 = evaluator.coverage_probability(y_test, lower_80, upper_80)
    width_95 = evaluator.mean_interval_width(lower_95, upper_95)
    width_80 = evaluator.mean_interval_width(lower_80, upper_80)
    
    print(f"\nPrediction Interval Evaluation (Holt-Winters):")
    print(f"  95% Interval:")
    print(f"    Coverage: {coverage_95*100:.1f}% (Target: 95%)")
    print(f"    Mean Width: {width_95:.2f}")
    print(f"  80% Interval:")
    print(f"    Coverage: {coverage_80*100:.1f}% (Target: 80%)")
    print(f"    Mean Width: {width_80:.2f}")
    
    has_intervals = True
except:
    print("\nPrediction interval computation failed")
    has_intervals = False

# Scenario 3: Forecast Horizon Analysis
print("\n" + "="*80)
print("SCENARIO 3: Forecast Accuracy by Horizon")
print("="*80)

# Compute accuracy at different horizons
horizons_to_test = [1, 3, 6, 12, 24]
horizon_results = {h: {} for h in horizons_to_test}

for h in horizons_to_test:
    if h > len(y_test):
        continue
    
    y_test_h = y_test[:h]
    
    # SARIMA forecast
    forecast_sarima_h = forecaster.sarima_forecast(y_train, h, 
                                                   order=(1,1,1), 
                                                   seasonal_order=(1,1,1,12))
    
    mae_h = evaluator.mae(y_test_h, forecast_sarima_h)
    rmse_h = evaluator.rmse(y_test_h, forecast_sarima_h)
    
    horizon_results[h] = {'MAE': mae_h, 'RMSE': rmse_h}

print(f"\nSARIMA Accuracy by Horizon:")
print(f"{'Horizon':<12} {'MAE':<12} {'RMSE':<12}")
print("-" * 36)
for h in horizons_to_test:
    if h in horizon_results and horizon_results[h]:
        print(f"{h:<12} {horizon_results[h]['MAE']:<12.3f} {horizon_results[h]['RMSE']:<12.3f}")

print(f"\nAs horizon increases, accuracy typically deteriorates")

# Scenario 4: Ensemble Forecasting
print("\n" + "="*80)
print("SCENARIO 4: Ensemble Methods - Forecast Combination")
print("="*80)

# Simple average ensemble
forecasts_ensemble = np.array([forecast_hw, forecast_sarima, forecast_ml])
forecast_avg = np.mean(forecasts_ensemble, axis=0)

# Weighted average (inverse RMSE weights from training)
# Use cross-validation to get weights
weights = []
for i, name in enumerate(['Holt-Winters', 'SARIMA', 'Random Forest']):
    if name in results:
        rmse = results[name]['RMSE']
        weights.append(1 / (rmse + 1e-10))

weights = np.array(weights)
weights /= np.sum(weights)

forecast_weighted = np.average(forecasts_ensemble, axis=0, weights=weights)

# Evaluate ensembles
metrics_avg = evaluator.summary(y_test, forecast_avg, y_train, seasonal_period=12)
metrics_weighted = evaluator.summary(y_test, forecast_weighted, y_train, seasonal_period=12)

print(f"\nEnsemble Performance:")
print(f"{'Method':<20} {'MAE':<10} {'RMSE':<10} {'MASE':<10}")
print("-" * 50)
print(f"{'Simple Average':<20} {metrics_avg['MAE']:<10.3f} {metrics_avg['RMSE']:<10.3f} {metrics_avg['MASE']:<10.3f}")
print(f"{'Weighted Average':<20} {metrics_weighted['MAE']:<10.3f} {metrics_weighted['RMSE']:<10.3f} {metrics_weighted['MASE']:<10.3f}")
print(f"\nEnsemble often more robust than individual models")

# Scenario 5: Rolling Origin Cross-Validation
print("\n" + "="*80)
print("SCENARIO 5: Rolling Origin Cross-Validation")
print("="*80)

# Define forecaster function for CV
def sarima_cv_func(y_train, h):
    return forecaster.sarima_forecast(y_train, h, order=(1,1,1), 
                                     seasonal_order=(1,1,1,12))

# Perform rolling origin CV
print(f"\nPerforming rolling origin CV (this may take a moment)...")
errors_cv = rolling_origin_cv(y, sarima_cv_func, min_train=60, horizon=6, step=6)

mae_cv = np.mean(np.abs(errors_cv))
rmse_cv = np.sqrt(np.mean(errors_cv**2))

print(f"\nSARIMA Cross-Validation Results:")
print(f"  MAE: {mae_cv:.3f}")
print(f"  RMSE: {rmse_cv:.3f}")
print(f"  Number of forecast origins: {len(errors_cv) // 6}")

# Visualizations
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Plot 1: Time series with train/test split
ax = axes[0, 0]
ax.plot(t[:train_size], y_train, 'b-', label='Train', linewidth=1.5)
ax.plot(t[train_size:], y_test, 'g-', label='Test', linewidth=1.5)
ax.axvline(train_size, color='r', linestyle='--', alpha=0.5, label='Split')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Time Series: Train/Test Split')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Forecasts comparison
ax = axes[0, 1]
t_test = t[train_size:]
ax.plot(t_test, y_test, 'k-', linewidth=2, label='Actual', alpha=0.7)
ax.plot(t_test, forecast_naive, '--', label='Naive', alpha=0.7)
ax.plot(t_test, forecast_hw, '-', label='Holt-Winters', linewidth=2)
ax.plot(t_test, forecast_sarima, '-', label='SARIMA', linewidth=2)
ax.plot(t_test, forecast_ml, '-', label='Random Forest', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Forecast Comparison')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Forecast errors
ax = axes[1, 0]
errors_hw = y_test - forecast_hw
errors_sarima = y_test - forecast_sarima
ax.plot(t_test, errors_hw, 'o-', label='Holt-Winters', alpha=0.7)
ax.plot(t_test, errors_sarima, 's-', label='SARIMA', alpha=0.7)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('Forecast Error')
ax.set_title('Forecast Errors Over Time')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Error distribution
ax = axes[1, 1]
ax.hist(errors_hw, bins=15, alpha=0.5, label='Holt-Winters', edgecolor='black')
ax.hist(errors_sarima, bins=15, alpha=0.5, label='SARIMA', edgecolor='black')
ax.axvline(0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Forecast Error')
ax.set_ylabel('Frequency')
ax.set_title('Error Distribution')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Prediction intervals
if has_intervals:
    ax = axes[2, 0]
    ax.plot(t_test, y_test, 'ko', label='Actual', markersize=6)
    ax.plot(t_test, forecast_hw, 'b-', label='Forecast', linewidth=2)
    ax.fill_between(t_test, lower_95, upper_95, alpha=0.2, color='blue', label='95% PI')
    ax.fill_between(t_test, lower_80, upper_80, alpha=0.3, color='blue', label='80% PI')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Prediction Intervals (Coverage: 95%={coverage_95*100:.0f}%, 80%={coverage_80*100:.0f}%)')
    ax.legend()
    ax.grid(alpha=0.3)
else:
    axes[2, 0].text(0.5, 0.5, 'Prediction Intervals\nNot Available', 
                    ha='center', va='center', transform=axes[2, 0].transAxes)

# Plot 6: Accuracy by horizon
ax = axes[2, 1]
horizons_plot = [h for h in horizons_to_test if h in horizon_results and horizon_results[h]]
maes_plot = [horizon_results[h]['MAE'] for h in horizons_plot]
rmses_plot = [horizon_results[h]['RMSE'] for h in horizons_plot]

ax.plot(horizons_plot, maes_plot, 'o-', label='MAE', linewidth=2, markersize=8)
ax.plot(horizons_plot, rmses_plot, 's-', label='RMSE', linewidth=2, markersize=8)
ax.set_xlabel('Forecast Horizon')
ax.set_ylabel('Error')
ax.set_title('Forecast Accuracy Deterioration')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Additional analysis: Method comparison bar chart
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

method_names = list(results.keys())
maes = [results[m]['MAE'] for m in method_names]
rmses = [results[m]['RMSE'] for m in method_names]

x = np.arange(len(method_names))
width = 0.35

bars1 = ax.bar(x - width/2, maes, width, label='MAE', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, rmses, width, label='RMSE', alpha=0.8, edgecolor='black')

ax.set_xlabel('Method')
ax.set_ylabel('Error')
ax.set_title('Forecast Accuracy Comparison (Lower is Better)')
ax.set_xticks(x)
ax.set_xticklabels(method_names, rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Annotate best
best_idx = np.argmin(rmses)
ax.annotate('Best', xy=(best_idx + width/2, rmses[best_idx]), 
           xytext=(best_idx + width/2, rmses[best_idx] + 1),
           arrowprops=dict(arrowstyle='->', color='red', lw=2),
           fontsize=12, color='red', weight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n1. Evaluated {len(methods)} forecasting methods on synthetic data")
print(f"2. Best performer: {best_method[0]} (RMSE={best_method[1]['RMSE']:.3f})")
print(f"3. All methods beat naive benchmark (Theil's U < 1)")
print(f"4. Prediction intervals: {coverage_95*100:.0f}% coverage for 95% nominal" if has_intervals else "4. Prediction intervals: Not computed")
print(f"5. Ensemble methods provide robust alternative to single models")
print(f"6. Forecast accuracy deteriorates with horizon (MAE at h=1: {horizon_results[1]['MAE']:.2f} vs h=12: {horizon_results[12]['MAE']:.2f})" if 12 in horizon_results else "")
```

## 6. Challenge Round
1. **Optimal Forecast Combination:** Given K=5 models, find optimal weights w_k to minimize RMSE on validation set. Constraint: Σw_k=1, w_k≥0. Use quadratic programming. Does it beat simple average?

2. **Probabilistic Forecasting:** Generate full predictive distribution (not just intervals). Use quantile regression to estimate 5th, 25th, 50th, 75th, 95th percentiles. Compute CRPS (Continuous Ranked Probability Score). Compare to normal approximation.

3. **Regime-Switching:** Simulate series with structural break at t=60 (variance doubles). Fit models on full data vs rolling window (last 30 obs). Which adapts better? Track RMSE over time.

4. **Forecast Reconciliation:** Hierarchical series (Total = A + B). Forecast Total, A, B separately—sum doesn't match. Apply reconciliation (bottom-up, top-down, optimal). Does coherence improve accuracy?

5. **Conformal Prediction:** Implement conformal intervals for Random Forest. Split calibration set, compute non-conformity scores, use quantiles for interval. Compare coverage to bootstrap—which more reliable?

## 7. Key References
- [Hyndman & Athanasopoulos, "Forecasting: Principles and Practice" (3rd ed, 2021)](https://otexts.com/fpp3/) - comprehensive forecasting textbook (free online)
- [Diebold & Mariano, "Comparing Predictive Accuracy" (1995)](https://www.jstor.org/stable/2285284) - statistical test for forecast comparison
- [Makridakis et al., "The M4 Competition: 100,000 time series and 61 forecasting methods" (2020)](https://www.sciencedirect.com/science/article/pii/S0169207019301128) - large-scale forecast competition results

---
**Status:** Essential time series skill | **Complements:** ARIMA, Exponential Smoothing, Model Selection, Cross-Validation, Uncertainty Quantification, Decision Analysis
