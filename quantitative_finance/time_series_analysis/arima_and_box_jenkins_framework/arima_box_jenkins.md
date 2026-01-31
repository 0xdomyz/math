# ARIMA and Box-Jenkins Framework

## 1. Concept Skeleton
**Definition:** Autoregressive Integrated Moving Average (ARIMA) models for univariate time series forecasting via differencing, autoregression, and moving average components  
**Purpose:** Model non-stationary series by differencing to achieve stationarity, then fit AR and MA components to capture temporal dependencies and forecast future values  
**Prerequisites:** Stationarity concepts, ACF/PACF interpretation, lag operators, backshift notation, maximum likelihood estimation, model diagnostics

## 2. Comparative Framing
| Model Type | AR(p) | MA(q) | ARMA(p,q) | ARIMA(p,d,q) | SARIMA(p,d,q)(P,D,Q)_s |
|------------|-------|-------|-----------|--------------|------------------------|
| **Stationarity** | Requires stationary | Requires stationary | Requires stationary | Differencing (d) achieves | Seasonal differencing (D) |
| **Components** | Past values only | Past errors only | Both past values & errors | AR, I (differencing), MA | + Seasonal AR, MA |
| **Parameters** | p lags | q lags | p AR + q MA | p AR + d diff + q MA | + P, D, Q seasonal |
| **ACF Pattern** | Exponential decay | Cutoff after q lags | Mixed decay | After differencing | Seasonal spikes |
| **PACF Pattern** | Cutoff after p lags | Exponential decay | Mixed decay | After differencing | Seasonal spikes |
| **Use Case** | Slowly decaying ACF | Sharp cutoff ACF | Complex patterns | Trending/non-stationary | Quarterly/monthly data |

| Identification Tool | Purpose | Interpretation | When to Use |
|--------------------|---------|----------------|-------------|
| **ACF (Autocorrelation)** | MA order detection | Cutoff at lag q → MA(q) | Always (first step) |
| **PACF (Partial Autocorrelation)** | AR order detection | Cutoff at lag p → AR(p) | Always (first step) |
| **ADF/KPSS Tests** | Stationarity check | Reject H0 in ADF → stationary | Before modeling |
| **Ljung-Box Test** | Residual autocorrelation | Fail to reject → white noise residuals | Model validation |
| **AIC/BIC** | Model selection | Lower is better (penalizes complexity) | Compare competing models |
| **Out-of-Sample RMSE** | Forecast accuracy | Lower is better | Final model selection |

## 3. Examples + Counterexamples

**Simple Example:**  
Stock returns (stationary): ACF decays slowly, PACF cuts off at lag 1 → AR(1) model. y_t = φ₁y_{t-1} + ε_t. Estimate φ₁ = 0.15, forecast next day.

**Perfect Fit:**  
Daily temperature: Strong seasonality (365-day cycle), trend (climate change). SARIMA(1,1,1)(1,1,1)_{365} captures trend via d=1, seasonality via D=1, and short-term AR/MA patterns.

**Failure Case:**  
Cryptocurrency prices: Extreme volatility, regime changes, non-Gaussian errors. ARIMA assumes constant variance, linear dynamics → poor forecasts during crashes. Use GARCH or regime-switching models instead.

**Edge Case:**  
Economic data with structural break (policy change in Year 5): ARIMA fit to full series averages pre/post regimes → poor performance. Either split series, use exogenous variables, or intervention analysis.

**Common Mistake:**  
Overdifferencing: Series already stationary but apply d=1 differencing → introduces negative autocorrelation, degrades forecast. Check stationarity first (ADF test) before differencing.

**Counterexample:**  
High-frequency financial data (tick-by-tick): Microstructure noise, bid-ask bounce, non-Gaussian errors. ARIMA inappropriate; use realized volatility models, microstructure-aware filters.

## 4. Layer Breakdown
```
Box-Jenkins Methodology (Iterative Process):

├─ Step 1: Model Identification
│   ├─ Stationarity Assessment:
│   │   ├─ Visual Inspection:
│   │   │   │ Plot time series: Look for trend (non-constant mean), seasonality (periodic)
│   │   │   │ Rolling mean/variance: Calculate over windows (e.g., 12 months)
│   │   │   │ If changing → non-stationary
│   │   │   └─ Example: Stock prices (trending up) → non-stationary
│   │   ├─ Augmented Dickey-Fuller (ADF) Test:
│   │   │   │ H0: Unit root exists (non-stationary)
│   │   │   │ H1: No unit root (stationary)
│   │   │   │ If p-value < 0.05 → reject H0 → stationary
│   │   │   └─ Caveats: Assumes no structural breaks, linear trend
│   │   ├─ KPSS Test (complement to ADF):
│   │   │   │ H0: Stationary
│   │   │   │ H1: Non-stationary
│   │   │   │ If p-value < 0.05 → reject H0 → non-stationary
│   │   │   └─ Use both: ADF fails to reject + KPSS rejects → differencing needed
│   │   └─ Ljung-Box Test:
│   │       Q-statistic tests if residuals are white noise
│   │       For raw series: Tests randomness (if random, no ARIMA needed)
│   ├─ Differencing to Achieve Stationarity:
│   │   ├─ First Difference (d=1):
│   │   │   │ Δy_t = y_t - y_{t-1}
│   │   │   │ Removes linear trend, converts I(1) to I(0)
│   │   │   │ Example: Stock prices (I(1)) → Returns (I(0))
│   │   │   └─ Warning: Don't overdifference (introduces spurious MA structure)
│   │   ├─ Second Difference (d=2):
│   │   │   │ Δ²y_t = Δy_t - Δy_{t-1} = y_t - 2y_{t-1} + y_{t-2}
│   │   │   │ Removes quadratic trend
│   │   │   │ Rarely needed (usually d=1 sufficient)
│   │   │   └─ Example: Accelerating growth processes
│   │   ├─ Seasonal Differencing (D=1):
│   │   │   │ Δ_s y_t = y_t - y_{t-s}
│   │   │   │ s = seasonal period (12 for monthly, 4 for quarterly)
│   │   │   │ Removes seasonal pattern
│   │   │   └─ Can combine: (1-B)(1-B^s)y_t = Δ Δ_s y_t
│   │   └─ How Many Differences?:
│   │       Start with d=0, test stationarity
│   │       If non-stationary: d=1, retest
│   │       Rarely need d>2 (economic/financial data)
│   │       Monitor ACF: Overdifferencing shows negative lag-1 autocorrelation
│   ├─ ACF/PACF Analysis:
│   │   ├─ Autoregressive AR(p) Patterns:
│   │   │   │ ACF: Exponential decay or damped sine wave
│   │   │   │ PACF: Sharp cutoff after lag p
│   │   │   │ Example: PACF significant at lags 1, 2; zero after → AR(2)
│   │   │   └─ Mechanism: Direct effect at lag p; indirect via intermediate lags
│   │   ├─ Moving Average MA(q) Patterns:
│   │   │   │ ACF: Sharp cutoff after lag q
│   │   │   │ PACF: Exponential decay
│   │   │   │ Example: ACF significant at lags 1, 2, 3; zero after → MA(3)
│   │   │   └─ Mechanism: Error shocks have finite memory (q periods)
│   │   ├─ ARMA(p,q) Mixed Patterns:
│   │   │   │ ACF: Exponential decay after lag q-p
│   │   │   │ PACF: Exponential decay after lag p-q
│   │   │   │ Both taper off gradually (no clear cutoff)
│   │   │   └─ Challenging: Try multiple (p,q) combinations, use AIC/BIC
│   │   └─ Confidence Intervals:
│   │       95% CI: ±1.96/√n (Bartlett's formula)
│       │       If ACF/PACF outside bands → significant
│   │       Example: n=100 → bands at ±0.196
│   ├─ Initial Model Specification:
│   │   │ Based on ACF/PACF, propose ARIMA(p,d,q)
│   │   │ Start simple: ARIMA(1,1,1) often good baseline
│   │   │ Consider seasonal SARIMA(p,d,q)(P,D,Q)_s if seasonal pattern
│   │   └─ Grid search: Try p,q ∈ {0,1,2,3}, select best via AIC/BIC
│   │
├─ Step 2: Parameter Estimation
│   ├─ Maximum Likelihood Estimation (MLE):
│   │   ├─ Likelihood Function:
│   │   │   │ Assume errors ε_t ~ N(0, σ²) (Gaussian white noise)
│   │   │   │ Joint likelihood: L(θ|y) = product of individual densities
│   │   │   │ Log-likelihood: ℓ(θ) = -n/2 log(2πσ²) - (1/2σ²) Σ ε_t²
│   │   │   └─ Maximize ℓ w.r.t. parameters θ = (φ, θ, σ²)
│   │   ├─ Conditional Sum of Squares (CSS):
│   │   │   │ Condition on initial observations
│   │   │   │ Minimize Σ ε_t² (residual sum of squares)
│   │   │   │ Computationally simpler than MLE
│   │   │   └─ Used as starting values for MLE
│   │   ├─ Optimization:
│   │   │   │ Nonlinear optimization (Newton-Raphson, BFGS)
│   │   │   │ Iterate until convergence (gradient near zero)
│   │   │   └─ Software handles automatically (statsmodels, forecast package)
│   │   └─ Parameter Constraints:
│   │       AR: Stationarity requires roots of φ(B) outside unit circle
│   │       MA: Invertibility requires roots of θ(B) outside unit circle
│   │       Software enforces during optimization
│   ├─ Standard Errors and Confidence Intervals:
│   │   │ Asymptotic standard errors from Hessian matrix
│   │   │ 95% CI: θ̂ ± 1.96 × SE(θ̂)
│   │   │ Test significance: |θ̂/SE| > 1.96 → reject H0: θ=0
│   │   └─ Example: φ₁ = 0.65 ± 0.12 → significant (t=5.4)
│   ├─ Model Selection Criteria:
│   │   ├─ Akaike Information Criterion (AIC):
│   │   │   │ AIC = -2ℓ + 2k
│   │   │   │ ℓ = log-likelihood, k = # parameters
│   │   │   │ Penalizes complexity, but less than BIC
│   │   │   └─ Lower AIC preferred
│   │   ├─ Bayesian Information Criterion (BIC):
│   │   │   │ BIC = -2ℓ + k log(n)
│   │   │   │ Stronger penalty for complexity (log(n) > 2 when n > 7)
│   │   │   │ Favors parsimonious models
│   │   │   └─ Lower BIC preferred
│   │   └─ AICc (corrected AIC):
│   │       AICc = AIC + 2k(k+1)/(n-k-1)
│   │       Corrects for small sample bias
│   │       Use when n/k < 40
│   └─ Software Implementation:
│       Python: statsmodels.tsa.arima.model.ARIMA
│       R: forecast::auto.arima(), stats::arima()
│       Automated search: auto.arima() grid searches (p,d,q)
│
├─ Step 3: Diagnostic Checking
│   ├─ Residual Analysis:
│   │   ├─ White Noise Test (Ljung-Box):
│   │   │   │ H0: Residuals are white noise (no autocorrelation)
│   │   │   │ Q-statistic: Q = n(n+2) Σ_{k=1}^h ρ_k²/(n-k)
│   │   │   │ h = max lag tested (typically 10-20)
│   │   │   │ If p-value > 0.05 → fail to reject → good (white noise)
│   │   │   └─ If p-value < 0.05 → model inadequate (remaining patterns)
│   │   ├─ ACF of Residuals:
│   │   │   │ Plot ACF of ε̂_t
│   │   │   │ Should be within confidence bands (all lags insignificant)
│   │   │   │ If spikes → missing AR/MA terms
│   │   │   └─ Example: Spike at lag 12 → seasonal component missed
│   │   ├─ Normality Tests:
│   │   │   │ Shapiro-Wilk, Jarque-Bera tests
│   │   │   │ H0: Residuals normally distributed
│   │   │   │ If rejected: Outliers, fat tails → consider robust methods
│   │   │   └─ ARIMA assumes normality, but robust to mild departures
│   │   └─ Heteroscedasticity (ARCH Effects):
│   │       Plot squared residuals, test for autocorrelation
│   │       If present: Variance not constant → use GARCH
│   ├─ Parameter Stability:
│   │   │ Check if parameters change over time (rolling window estimation)
│   │   │ If unstable: Structural break, regime change → split sample
│   │   └─ Example: Pre/post-2008 financial crisis
│   ├─ Overfitting Check:
│   │   │ High in-sample fit but poor out-of-sample → overfitting
│   │   │ Penalize with AIC/BIC
│   │   └─ Cross-validation: Hold-out test set for validation
│   └─ If Diagnostics Fail:
│       Modify model: Add AR/MA terms, seasonal components, exogenous variables
│       Re-estimate and re-check diagnostics
│       Iterate until residuals pass tests
│
├─ Step 4: Forecasting
│   ├─ Point Forecasts:
│   │   ├─ h-Step Ahead Forecast:
│   │   │   │ ŷ_{T+h|T} = E[y_{T+h} | y_T, y_{T-1}, ...]
│   │   │   │ For AR(1): ŷ_{T+1} = φ₁y_T
│   │   │   │            ŷ_{T+2} = φ₁ŷ_{T+1} = φ₁²y_T
│   │   │   │            ŷ_{T+h} = φ₁^h y_T → decays to mean
│   │   │   └─ For ARMA: Recursive calculation using Kalman filter
│   │   ├─ Forecast Horizon Effects:
│   │   │   │ Short-term (h=1-5): Accurate, captures patterns
│   │   │   │ Medium-term (h=6-20): Increasing uncertainty
│   │   │   │ Long-term (h>20): Forecasts revert to unconditional mean
│   │   │   └─ ARIMA not suitable for long horizons (trend extrapolation)
│   │   └─ Differencing Reversal:
│   │       If fitted on differences, integrate forecasts back
│   │       Example: Forecast Δy_{T+1}, then y_{T+1} = y_T + Δy_{T+1}
│   ├─ Forecast Intervals (Uncertainty Quantification):
│   │   ├─ Standard Error of Forecast:
│   │   │   │ SE(ŷ_{T+h}) increases with h
│   │   │   │ For AR(1): SE²(ŷ_{T+h}) = σ² [1 + φ₁² + φ₁⁴ + ... + φ₁^{2(h-1)}]
│   │   │   │            = σ² (1 - φ₁^{2h}) / (1 - φ₁²)
│   │   │   └─ As h → ∞: SE → σ / √(1-φ₁²) (unconditional SD)
│   │   ├─ Confidence Intervals:
│   │   │   │ 95% CI: ŷ_{T+h} ± 1.96 × SE(ŷ_{T+h})
│   │   │   │ Assumes normality (Gaussian errors)
│   │   │   └─ Fan chart: Plot multiple CIs (50%, 80%, 95%)
│   │   └─ Simulation-Based Intervals:
│   │       Monte Carlo: Simulate future paths from fitted model
│   │       Calculate empirical quantiles (percentiles)
│   │       Robust to non-normality
│   ├─ Forecast Evaluation:
│   │   ├─ Out-of-Sample Metrics:
│   │   │   │ RMSE = √(1/h Σ (y_{T+i} - ŷ_{T+i})²)
│   │   │   │ MAE = 1/h Σ |y_{T+i} - ŷ_{T+i}|
│   │   │   │ MAPE = 100/h Σ |y_{T+i} - ŷ_{T+i}| / |y_{T+i}|
│   │   │   └─ Lower is better
│   │   ├─ Diebold-Mariano Test:
│   │   │   Test if two forecasting methods differ significantly
│   │   │   H0: Equal forecast accuracy
│   │   └─ Forecast Encompassing:
│   │       Does one forecast contain all info of another?
│   ├─ Rolling vs Expanding Window:
│   │   │ Rolling: Fixed window size, drop oldest as new data arrives
│   │   │         Adapts to recent changes
│   │   │ Expanding: Add new data, keep all historical
│   │   │           Stable parameters, less reactive
│   │   └─ Choice depends on structural stability
│   └─ Re-estimation Frequency:
│       How often to update model parameters?
│       High-frequency data: Daily/weekly re-estimation
│       Low-frequency: Monthly/quarterly
│       Computational cost vs. adaptation trade-off
│
├─ ARIMA Model Variants and Extensions:
│   ├─ Seasonal ARIMA (SARIMA):
│   │   ├─ Notation: ARIMA(p,d,q)(P,D,Q)_s
│   │   │   │ (p,d,q): Non-seasonal AR, differencing, MA
│   │   │   │ (P,D,Q): Seasonal AR, differencing, MA at lag s
│   │   │   │ s: Seasonal period (12 monthly, 4 quarterly, 7 daily)
│   │   │   └─ Example: ARIMA(1,1,1)(1,1,1)₁₂ for monthly data
│   │   ├─ Model Structure:
│   │   │   │ Φ_P(B^s) φ_p(B) (1-B)^d (1-B^s)^D y_t = Θ_Q(B^s) θ_q(B) ε_t
│   │   │   │ Φ_P: Seasonal AR polynomial
│   │   │   │ Θ_Q: Seasonal MA polynomial
│   │   │   └─ Captures both short-term and seasonal dynamics
│   │   └─ Identification:
│   │       ACF/PACF at seasonal lags (12, 24, 36 for monthly)
│   │       Spikes at multiples of s indicate seasonal AR/MA
│   ├─ ARIMAX (with Exogenous Variables):
│   │   │ Add external regressors: y_t = β'x_t + η_t, where η_t ~ ARIMA
│   │   │ Example: Sales as function of advertising (x_t)
│   │   │ η_t captures residual autocorrelation after regression
│   │   └─ Requires forecasts of x_t for future predictions
│   ├─ Intervention Analysis:
│   │   │ Model impact of known events (policy change, strike, crisis)
│   │   │ Add dummy variables or pulse/step functions
│   │   │ Example: Outlier at t=50 → add I_{t=50}
│   │   └─ Avoids contaminating parameter estimates
│   ├─ Transfer Function Models:
│   │   │ Dynamic regression: y_t depends on current and lagged x_t
│   │   │ y_t = ω(B)/δ(B) x_t + η_t, η_t ~ ARIMA
│   │   │ Captures lead-lag relationships
│   │   └─ Example: Output as function of input with delays
│   └─ Fractionally Integrated ARIMA (ARFIMA):
│       Long memory: d ∈ (0, 1) instead of integer
│       Captures slow hyperbolic decay in ACF
│       Rare in practice (computational complexity)
│
└─ Practical Considerations:
    ├─ Sample Size Requirements:
    │   │ Minimum: 50-100 observations for reliable estimation
    │   │ Seasonal models: Need multiple cycles (3-5 years monthly data)
    │   │ Small samples: Use AICc, avoid overparameterization
    │   └─ High-frequency: More data, but microstructure noise complicates
    ├─ Missing Values:
    │   │ ARIMA requires equally spaced, complete data
    │   │ Imputation: Linear interpolation, Kalman smoothing
    │   │ Or: Use state-space models (handle missing naturally)
    │   └─ Irregular spacing: Convert to state-space representation
    ├─ Outliers and Anomalies:
    │   │ Outliers distort parameter estimates
    │   │ Detection: Standardized residuals > 3σ
    │   │ Treatment: Winsorize, use robust estimation, or intervention analysis
    │   └─ Example: COVID-19 shock in economic data
    ├─ Non-Gaussian Errors:
    │   │ ARIMA assumes normality, but estimation robust
    │   │ Forecasting intervals affected (use bootstrap)
    │   │ Fat tails: Consider t-distribution errors
    │   └─ Skewness: Log-transform data if multiplicative
    ├─ Model Parsimony:
    │   │ Principle: Simplest model that fits adequately
    │   │ Avoid overfitting (high p, q)
    │   │ Use AIC/BIC, prefer lower order
    │   └─ Example: ARIMA(1,1,1) often sufficient vs. ARIMA(5,1,5)
    └─ Computational Tools:
        Python: statsmodels.tsa.arima.model.ARIMA, pmdarima.auto_arima
        R: forecast::auto.arima(), stats::arima()
        Automated: Grid search over (p,d,q), select by AIC/BIC
```

**Interaction:** Check stationarity (ADF, KPSS) → Difference if needed (d=1 or 2) → Plot ACF/PACF on stationary series → Identify tentative (p,q) → Estimate ARIMA(p,d,q) via MLE → Check diagnostics (Ljung-Box, ACF of residuals, normality) → If fail, modify (p,q) and re-estimate → Forecast h-steps ahead with confidence intervals → Evaluate on hold-out set (RMSE, MAE)

## 5. Mini-Project
Comprehensive ARIMA modeling with Box-Jenkins methodology, seasonal extension, and forecast evaluation:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
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
from statsmodels.tsa.stattools import acf, pacf

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
```

## 6. Challenge Round
Advanced ARIMA applications and extensions:

1. **Automatic Model Selection:** Implement your own `auto_arima()` function with grid search over (p,d,q), pruning non-stationary/non-invertible parameters, selecting by AICc. Compare to `pmdarima.auto_arima()`. When does automated selection fail?

2. **Intervention Analysis:** Time series with known structural break (e.g., policy change at t=150). Fit ARIMA pre-break, post-break, and with dummy variable. Compare forecast performance. How do you detect unknown break points?

3. **Long-Horizon Forecasting:** ARIMA(1,1,1) on quarterly GDP. Forecast 20 quarters ahead. Compare to naive (random walk) and mean reversion models. At what horizon does ARIMA lose advantage?

4. **Seasonal Decomposition + ARIMA:** Decompose seasonal series into trend, seasonal, residual. Fit ARIMA to residual component, then reconstruct forecasts. Compare to direct SARIMA. When is decomposition superior?

5. **Bootstrap Forecast Intervals:** Standard ARIMA CIs assume normality. Implement residual bootstrap: Sample residuals with replacement, generate synthetic future paths, calculate empirical quantiles. Compare to parametric CIs for fat-tailed data.

6. **Transfer Function Models:** Model sales (y_t) as dynamic function of advertising spend (x_t). Estimate transfer function y_t = ω(B)/δ(B) x_{t-b} + η_t where η_t ~ ARIMA. Determine lag b and decay structure. How do you forecast when x_t future values unknown?

7. **Model Averaging:** Instead of selecting single "best" model, average forecasts from top 5 models (by AIC) weighted by AIC weights: w_i ∝ exp(-0.5 × ΔAIC_i). Does this improve out-of-sample performance vs. single-model selection?

## 7. Key References
- [Box, Jenkins, Reinsel & Ljung, "Time Series Analysis: Forecasting and Control" (5th Edition, 2015)](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) - foundational text on Box-Jenkins methodology, comprehensive treatment
- [Hyndman & Athanasopoulos, "Forecasting: Principles and Practice" (3rd Edition, 2021)](https://otexts.com/fpp3/) - modern practical guide, R-based, freely available online
- [Brockwell & Davis, "Introduction to Time Series and Forecasting" (3rd Edition, 2016)](https://link.springer.com/book/10.1007/978-3-319-29854-2) - rigorous mathematical treatment, includes seasonal models and diagnostics

---
**Status:** Core univariate time series methodology | **Complements:** Stationarity Testing, ACF/PACF Analysis, GARCH Models, State-Space Models, Vector Autoregression (VAR), Forecasting Evaluation
