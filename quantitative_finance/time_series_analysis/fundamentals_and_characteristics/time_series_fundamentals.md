# Time Series Fundamentals and Characteristics

## 1. Concept Skeleton
**Definition:** Core properties distinguishing time series from cross-sectional data; temporal dependence structure; autocorrelation; trend, seasonality, cyclical patterns; stationarity concepts  
**Purpose:** Understand data-generating process; identify appropriate models; diagnose violations; choose transformations; interpret empirical patterns correctly  
**Prerequisites:** Probability theory, random processes, expectation, variance, covariance, conditional distributions, asymptotic theory

## 2. Comparative Framing
| Property | Cross-Sectional Data | Time Series Data | Panel Data |
|----------|---------------------|------------------|------------|
| **Observations** | Independent | Serially dependent | Both dimensions |
| **Ordering** | Arbitrary | Natural (time) | Time + cross-section |
| **Key Feature** | i.i.d. assumption | Autocorrelation | Complex dependence |
| **Main Issue** | Heterogeneity | Non-stationarity | Both |
| **Inference** | Standard CLT | Modified CLT | Clustered SEs |

| Stationarity Type | Strict Stationarity | Weak Stationarity | Trend Stationarity | Difference Stationarity |
|------------------|--------------------|--------------------|-------------------|------------------------|
| **Definition** | Joint dist. invariant | Mean/var/autocov constant | Stationary around trend | Stationary after differencing |
| **Condition** | All moments constant | First two moments | Remove deterministic trend | I(1) process |
| **Example** | White noise | ARMA | Linear trend + ARMA | Random walk + drift |
| **Test** | Not testable | Variance ratio | Regression residuals | Unit root (ADF, PP) |
| **Transformation** | None needed | None needed | Detrend | Difference |

## 3. Examples + Counterexamples

**Simple Example:**  
Daily temperature: Mean varies by season (not stationary). Seasonal adjustment → stationary residuals. Autocorrelation: Today's temp predicts tomorrow's (ρ₁≈0.9).

**Perfect Fit:**  
White noise: ε_t ~ i.i.d. N(0,σ²). Strictly stationary—all moments constant. ACF=0 for all lags>0. Benchmark for uncorrelated series.

**Stationary Process:**  
AR(1) with |φ|<1: x_t = φx_{t-1} + ε_t. Mean=0, variance=σ²/(1-φ²), autocorrelation ρ_k=φ^k decays exponentially. Weak stationarity satisfied.

**Non-Stationary:**  
Random walk: x_t = x_{t-1} + ε_t. Variance grows linearly with t (Var=tσ²). Not stationary—unit root. First difference stationary: Δx_t=ε_t ~ white noise.

**Spurious Regression:**  
Two independent random walks: Regress y_t on x_t → R²=0.8, significant t-stat! Both non-stationary → spurious correlation. Cointegration test needed.

**Poor Fit:**  
Stock prices assumed AR(1): Ignores stochastic volatility, fat tails, structural breaks. Model misspecification → forecasts fail, intervals too narrow.

## 4. Layer Breakdown
```
Time Series Fundamentals Framework:

├─ Defining Characteristics:
│  ├─ Temporal Ordering:
│  │   ├─ Observations indexed by time: y_t, t=1,2,...,T
│  │   ├─ Natural ordering cannot be permuted
│  │   ├─ Past influences present/future
│  │   └─ Information sets: I_t = {y_t, y_{t-1}, ...}
│  ├─ Serial Dependence:
│  │   ├─ Observations NOT independent
│  │   ├─ Cov(y_t, y_{t-k}) ≠ 0 typically
│  │   ├─ Memory: Past affects future
│  │   └─ Key difference from cross-section
│  ├─ Sampling Frequency:
│  │   ├─ High-frequency: Tick, second, minute
│  │   ├─ Daily: Stock prices, weather
│  │   ├─ Monthly: Economic indicators, sales
│  │   ├─ Quarterly: GDP, earnings
│  │   ├─ Annual: Population, mortality
│  │   └─ Irregular: Event studies, transactions
│  └─ Data-Generating Process (DGP):
│      Unknown stochastic mechanism
│      Model is approximation
│      Goal: Capture key features
├─ Stochastic Processes:
│  ├─ Definition:
│  │   {Y_t : t ∈ T} where T is index set
│  │   Collection of random variables
│  │   Realization: y_1, y_2, ..., y_T (observed path)
│  ├─ Discrete Time vs Continuous:
│  │   Discrete: t = 0, 1, 2, ... (most econometrics)
│  │   Continuous: t ∈ [0,∞) (finance, physics)
│  ├─ Finite Dimensional Distributions:
│  │   F(y_1,...,y_n; t_1,...,t_n) for any n, t_i
│  │   Complete probabilistic description
│  │   Usually focus on low-order moments
│  └─ Sample Path:
│      One realization of process
│      What we observe
│      Ergodicity: Sample → population
├─ Moments and Autocorrelation:
│  ├─ Mean Function:
│  │   μ_t = E[Y_t]
│  │   Time-varying or constant
│  │   Unconditional mean
│  ├─ Variance Function:
│  │   σ²_t = Var(Y_t) = E[(Y_t - μ_t)²]
│  │   Measures dispersion at time t
│  │   Homoskedastic: σ²_t = σ² (constant)
│  │   Heteroskedastic: σ²_t varies
│  ├─ Autocovariance Function (ACVF):
│  │   ├─ Definition:
│  │   │   γ(s,t) = Cov(Y_s, Y_t) = E[(Y_s - μ_s)(Y_t - μ_t)]
│  │   │   Measures linear dependence between times s, t
│  │   ├─ Properties:
│  │   │   γ(t,t) = Var(Y_t)
│  │   │   γ(s,t) = γ(t,s) (symmetric)
│  │   │   |γ(s,t)| ≤ √[γ(s,s)γ(t,t)] (Cauchy-Schwarz)
│  │   └─ Lag-k Autocovariance:
│  │       γ_k = Cov(Y_t, Y_{t-k})
│  │       Depends only on lag k (if stationary)
│  ├─ Autocorrelation Function (ACF):
│  │   ├─ Definition:
│  │   │   ρ_k = Corr(Y_t, Y_{t-k}) = γ_k / γ_0
│  │   │   Normalized autocovariance
│  │   ├─ Properties:
│  │   │   ρ_0 = 1 (correlation with self)
│  │   │   |ρ_k| ≤ 1
│  │   │   ρ_{-k} = ρ_k (symmetric)
│  │   ├─ Interpretation:
│  │   │   ρ_k > 0: Positive persistence
│  │   │   ρ_k < 0: Mean reversion
│  │   │   ρ_k ≈ 0: No linear dependence at lag k
│  │   ├─ Sample ACF:
│  │   │   r_k = Σ(y_t - ȳ)(y_{t-k} - ȳ) / Σ(y_t - ȳ)²
│  │   │   Estimator of ρ_k
│  │   │   Under H0 (white noise): r_k ~ N(0, 1/T) approx
│  │   └─ Significance Bounds:
│  │       ±1.96/√T (95% CI)
│  │       If |r_k| > 1.96/√T → significant
│  └─ Partial Autocorrelation Function (PACF):
│      ├─ Definition:
│      │   φ_kk = Corr(Y_t, Y_{t-k} | Y_{t-1}, ..., Y_{t-k+1})
│      │   Correlation controlling for intermediate lags
│      ├─ Interpretation:
│      │   Direct effect at lag k
│      │   Removes indirect effects
│      ├─ AR(p) Identification:
│      │   PACF cuts off after lag p
│      │   φ_kk = 0 for k > p
│      └─ Sample PACF:
│          Estimate via Yule-Walker equations
│          Regression of y_t on y_{t-1}, ..., y_{t-k}
├─ Stationarity:
│  ├─ Strict Stationarity (Strong):
│  │   ├─ Definition:
│  │   │   Joint distribution of (Y_{t1},...,Y_{tn}) same as
│  │   │   (Y_{t1+h},...,Y_{tn+h}) for all n, t_i, h
│  │   │   Distribution invariant to time shifts
│  │   ├─ Implication:
│  │   │   All moments constant over time
│  │   │   F(y_t) = F(y_{t+h}) (marginal distributions identical)
│  │   ├─ Example:
│  │   │   i.i.d. sequence (white noise)
│  │   │   Gaussian AR(1) with |φ|<1
│  │   └─ Problem:
│  │       Not testable (requires all moments)
│  │       Too strong for practical use
│  ├─ Weak Stationarity (Covariance Stationarity):
│  │   ├─ Definition:
│  │   │   1. E[Y_t] = μ (constant mean)
│  │   │   2. Var(Y_t) = σ² (constant variance)
│  │   │   3. Cov(Y_t, Y_{t-k}) = γ_k (depends only on lag k)
│  │   ├─ Autocovariance:
│  │   │   γ(s,t) = γ(|t-s|) = γ_k
│  │   │   Function of lag only, not absolute time
│  │   ├─ Sufficient for:
│  │   │   ARMA models
│  │   │   Most time series analysis
│  │   │   Forecasting, inference
│  │   ├─ Testing:
│  │   │   Augmented Dickey-Fuller (ADF)
│  │   │   Phillips-Perron (PP)
│  │   │   KPSS test
│  │   └─ Violations:
│  │       Trends (mean non-constant)
│  │       Seasonality (periodic mean/variance)
│  │       Structural breaks
│  │       Heteroskedasticity (time-varying variance)
│  ├─ Trend Stationarity:
│  │   ├─ Definition:
│  │   │   Y_t = T_t + X_t
│  │   │   T_t: Deterministic trend (e.g., α + βt)
│  │   │   X_t: Stationary process
│  │   ├─ Characteristics:
│  │   │   Non-stationary due to trend
│  │   │   Detrend → stationary
│  │   │   Shocks have temporary effect
│  │   ├─ Detrending:
│  │   │   Linear: Y_t - (α̂ + β̂t)
│  │   │   Polynomial: Higher-order trends
│  │   │   Regression residuals are stationary
│  │   └─ Example:
│  │       GDP with linear growth
│  │       + business cycle fluctuations
│  ├─ Difference Stationarity (Unit Root):
│  │   ├─ Definition:
│  │   │   Y_t = Y_{t-1} + ε_t (+ drift)
│  │   │   Non-stationary in levels
│  │   │   Stationary in differences: ΔY_t = ε_t
│  │   ├─ I(1) Process:
│  │   │   Integrated of order 1
│  │   │   Must difference once to achieve stationarity
│  │   │   I(d): Difference d times
│  │   ├─ Characteristics:
│  │   │   Shocks have permanent effect
│  │   │   Variance grows with time
│  │   │   Mean reversion only in differences
│  │   ├─ Random Walk:
│  │   │   Y_t = Y_{t-1} + ε_t
│  │   │   ε_t ~ i.i.d.(0, σ²)
│  │   │   Var(Y_t) = tσ² (unbounded)
│  │   │   E[Y_t] = Y_0 (depends on initial condition)
│  │   └─ With Drift:
│  │       Y_t = δ + Y_{t-1} + ε_t
│  │       Stochastic trend (random + deterministic)
│  └─ Distinguishing TS vs DS:
│      ├─ Unit Root Tests:
│      │   Null: Unit root (DS)
│      │   Alternative: Stationary (TS or none)
│      ├─ Visual Inspection:
│      │   TS: Fluctuates around trend line
│      │   DS: Wanders, no clear reversion
│      ├─ ACF Pattern:
│      │   TS: Decays exponentially
│      │   DS: Decays very slowly (near 1)
│      └─ Economic Interpretation:
│          TS: Shocks temporary (business cycles)
│          DS: Shocks permanent (productivity changes)
├─ Common Patterns:
│  ├─ Trend:
│  │   ├─ Deterministic Trend:
│  │   │   T_t = α + βt + γt² + ...
│  │   │   Predictable, smooth
│  │   │   Remove via regression
│  │   ├─ Stochastic Trend:
│  │   │   Random walk component
│  │   │   Unpredictable direction
│  │   │   Remove via differencing
│  │   └─ Identification:
│  │       Plot series over time
│  │       Unit root tests
│  ├─ Seasonality:
│  │   ├─ Deterministic Seasonal:
│  │   │   S_t = Σ γ_i D_it (seasonal dummies)
│  │   │   Fixed pattern each period
│  │   │   Remove via regression or seasonal dummies
│  │   ├─ Stochastic Seasonal:
│  │   │   (1 - L^s)Y_t = ε_t (seasonal unit root)
│  │   │   s: Seasonal period (12 for monthly)
│  │   │   Remove via seasonal differencing
│  │   ├─ Characteristics:
│  │   │   Regular periodic fluctuations
│  │   │   Predictable timing
│  │   │   Correlation at lags s, 2s, 3s, ...
│  │   └─ Detection:
│  │       ACF: Peaks at seasonal lags
│  │       Seasonal subseries plots
│  │       Periodogram (spectral analysis)
│  ├─ Cycles:
│  │   ├─ Definition:
│  │   │   Oscillations without fixed period
│  │   │   Business cycles: 2-10 years
│  │   │   Not as regular as seasonality
│  │   ├─ Characteristics:
│  │   │   Variable duration and amplitude
│  │   │   Often economic phenomena
│  │   │   Hard to model deterministically
│  │   └─ Modeling:
│  │       ARMA processes (stochastic cycles)
│  │       Unobserved components (state space)
│  │       Band-pass filters
│  └─ Irregular (Random) Component:
│      Unpredictable noise
│      White noise ideally: ε_t ~ i.i.d.(0, σ²)
│      Residual after removing systematic patterns
├─ Ergodicity and Mixing:
│  ├─ Ergodicity:
│  │   ├─ Definition:
│  │   │   Time averages → population moments
│  │   │   (1/T)Σy_t → E[Y] as T → ∞
│  │   ├─ Importance:
│  │   │   Justifies statistical inference
│  │   │   Sample estimates consistent
│  │   │   Law of Large Numbers applies
│  │   └─ Stationarity + Mixing → Ergodicity
│  ├─ Mixing:
│  │   ├─ Strong Mixing (α-mixing):
│  │   │   Distant past and future become independent
│  │   │   Correlation decays over time
│  │   ├─ Definition:
│  │   │   α(m) → 0 as m → ∞
│  │   │   Measures dependence between far-apart events
│  │   └─ Implication:
│  │       Central Limit Theorem applies
│  │       Standard inference valid
│  └─ Non-Ergodic Examples:
│      Random walk (non-stationary)
│      Regime-switching without ergodic switching
│      Structural breaks
├─ White Noise:
│  ├─ Definition:
│  │   {ε_t} is white noise if:
│  │   1. E[ε_t] = 0
│  │   2. Var(ε_t) = σ²
│  │   3. Cov(ε_t, ε_s) = 0 for s ≠ t
│  ├─ Strong White Noise:
│  │   ε_t ~ i.i.d.(0, σ²)
│  │   Independence (not just uncorrelated)
│  ├─ Gaussian White Noise:
│  │   ε_t ~ i.i.d. N(0, σ²)
│  │   Strictly stationary
│  │   Uncorrelated = Independent (Gaussianity)
│  ├─ Properties:
│  │   ACF(k) = 0 for all k > 0
│  │   Unpredictable (best forecast = mean = 0)
│  │   Building block for ARMA models
│  └─ Tests:
│      Ljung-Box Q-statistic
│      Portmanteau tests
│      ACF/PACF plots
├─ Linear vs Non-Linear Processes:
│  ├─ Linear Process:
│  │   ├─ Definition:
│  │   │   Y_t = μ + Σ_{j=0}^∞ ψ_j ε_{t-j}
│  │   │   Weighted sum of white noise
│  │   │   ψ_j: MA(∞) coefficients
│  │   ├─ Examples:
│  │   │   AR, MA, ARMA processes
│  │   │   Wold decomposition (any stationary process)
│  │   ├─ Properties:
│  │   │   Gaussian noise → Gaussian process
│  │   │   Linear forecasting optimal
│  │   │   ACF fully characterizes dependence
│  │   └─ Limitations:
│  │       Cannot capture volatility clustering
│  │       Symmetric response (up/down same)
│  │       No threshold effects
│  ├─ Non-Linear Process:
│  │   ├─ Examples:
│  │   │   ARCH/GARCH (volatility)
│  │   │   Bilinear models
│  │   │   Threshold AR (TAR)
│  │   │   Smooth transition AR (STAR)
│  │   ├─ Characteristics:
│  │   │   ACF may be zero (no linear dependence)
│  │   │   Non-linear dependence in squares, abs values
│  │   │   Asymmetric response to shocks
│  │   └─ Tests:
│  │       BDS test (independence)
│  │       McLeod-Li test (ARCH)
│  │       Brock-Dechert-Scheinkman
│  └─ Conditional Moments:
│      E[Y_t | I_{t-1}]: Conditional mean
│      Var(Y_t | I_{t-1}): Conditional variance
│      Time-varying in non-linear models
├─ Filtering vs Smoothing:
│  ├─ Filtering:
│  │   Estimate state at time t using data up to t
│  │   Real-time estimation
│  │   Kalman filter
│  ├─ Smoothing:
│  │   Estimate state at time t using all data (past + future)
│  │   Backward pass
│  │   More accurate, not real-time
│  └─ Forecasting:
│      Predict future beyond observed data
│      Only past information available
├─ Asymptotic Theory:
│  ├─ Central Limit Theorem:
│  │   √T(ȳ_T - μ) →^d N(0, Ω)
│  │   Ω: Long-run variance (depends on autocorrelation)
│  │   Ω = σ² × [1 + 2Σρ_k]
│  ├─ HAC Standard Errors:
│  │   Heteroskedasticity and Autocorrelation Consistent
│  │   Newey-West estimator
│  │   Adjusts for serial correlation
│  └─ Consistency:
│      Estimators converge to true value
│      Requires stationarity + ergodicity
└─ Practical Considerations:
   ├─ Sample Size Requirements:
   │   Need T >> p (parameters)
   │   Seasonal data: Multiple cycles (T ≥ 4×s recommended)
   │   Unit root tests: T ≥ 50 preferable
   ├─ Missing Data:
   │   Interpolation (linear, spline)
   │   State space models (Kalman handles naturally)
   │   Avoid deletion (breaks time structure)
   ├─ Outliers:
   │   Additive outlier: AO at time t affects y_t only
   │   Innovative outlier: IO affects y_t and future
   │   Level shift: LS permanent change in mean
   │   Detection: Intervention analysis, robust methods
   ├─ Sampling Rate:
   │   Nyquist frequency: Sample at 2× max frequency
   │   Aliasing: High freq appears as low freq (under-sample)
   │   Aggregation: Temporal → changes properties (e.g., MA→ARMA)
   └─ Software:
      R: stats, tseries, forecast
      Python: statsmodels, arch, pmdarima
      MATLAB: Econometrics Toolbox
      Stata: Time series commands (tsset, etc.)
```

**Interaction:** Observe series → Check stationarity (plot, ACF, unit root tests) → Identify patterns (trend, seasonal, cycles) → Transform if needed (difference, detrend, log) → Model stationary series → Validate (residuals = white noise?) → Forecast.

## 5. Mini-Project
Implement fundamental diagnostics, test stationarity, identify patterns:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
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
```

## 6. Challenge Round
1. **Spurious Regression:** Generate two independent random walks. Regress one on the other—get high R² and significant t-stat! Why? Test residuals for unit root. Demonstrate cointegration test solves this.

2. **Long Memory:** Simulate fractionally integrated process (ARFIMA d=0.3). Compare ACF decay to AR(1)—which slower? Estimate d via GPH estimator (log periodogram regression). Does it recover true value?

3. **Structural Break:** Generate series with mean shift at t=250. Apply ADF test—does it reject stationarity incorrectly? Implement Perron test (allow break). Does it correctly identify?

4. **Seasonal Unit Root:** Generate process with seasonal differencing needed: (1-L^12)y_t = ε_t. Apply regular ADF—fails. Apply seasonal unit root test (Hylleberg et al.). Does it detect?

5. **Ergodicity Violation:** Simulate regime-switching process that never ergodic (absorbing state). Show sample mean doesn't converge to population mean. Plot rolling average—does it stabilize?

## 7. Key References
- [Hamilton, "Time Series Analysis" (1994)](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis) - comprehensive graduate-level textbook covering fundamentals and advanced topics
- [Box, Jenkins & Reinsel, "Time Series Analysis: Forecasting and Control" (5th ed, 2015)](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) - classic reference for ARIMA methodology
- [Tsay, "Analysis of Financial Time Series" (3rd ed, 2010)](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) - financial applications focus, GARCH models

---
**Status:** Foundation for all time series analysis | **Complements:** Stationarity Testing, ARIMA Models, Unit Root Tests, Forecasting, Spectral Analysis
