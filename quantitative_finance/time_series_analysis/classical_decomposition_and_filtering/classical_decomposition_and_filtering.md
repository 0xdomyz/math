# Classical Decomposition and Filtering in Time Series

## 1. Concept Skeleton
**Definition:** Decompose time series into trend, seasonal, and irregular components; apply filters to extract or remove specific frequency components  
**Purpose:** Separate systematic patterns from noise; understand data structure; forecast by component; remove seasonality; smooth data for analysis  
**Prerequisites:** Time series basics, moving averages, differencing, Fourier analysis, frequency domain concepts

## 2. Comparative Framing
| Decomposition | Additive Model | Multiplicative Model | STL Decomposition | X-13ARIMA-SEATS |
|---------------|----------------|---------------------|-------------------|-----------------|
| **Formula** | Y = T + S + I | Y = T × S × I | Robust to outliers | Census Bureau official |
| **Seasonality** | Constant amplitude | Proportional to trend | Changing over time | Time-varying |
| **Use Case** | Stable seasonal pattern | Growing/shrinking series | Non-stationary seasonality | Official statistics |
| **Flexibility** | Low | Low | High (LOESS-based) | Very high |
| **Outlier Robust** | No | No | Yes | Yes |

| Filter Type | Moving Average | Exponential Smoothing | Hodrick-Prescott (HP) | Butterworth | Kalman Filter |
|-------------|---------------|----------------------|---------------------|-------------|---------------|
| **Mechanism** | Sliding window average | Weighted decay | Penalized regression | Frequency cutoff | State-space optimal |
| **Lag** | (k-1)/2 periods | Minimal | Symmetric (two-sided) | None (acausal) | None (causal) |
| **Weights** | Equal | Exponential decline | Data-driven | Smooth frequency response | Optimal (Bayesian) |
| **Trend Extraction** | Simple | Adaptive | Smooth | Smooth | Adaptive |
| **Real-Time** | Yes (one-sided) | Yes | No (needs full data) | No | Yes |

## 3. Examples + Counterexamples

**Simple Example:**  
Monthly retail sales: Clear upward trend + December spikes (Christmas). Additive decomposition: Trend=smooth growth line, Seasonal=+20% Dec/-5% Feb, Irregular=random noise.

**Perfect Fit:**  
Airline passenger data (classic): Multiplicative decomposition ideal. Seasonality grows with trend (1950s: summer +50 passengers, 1960s: +200 passengers). Log-transform → additive.

**Hodrick-Prescott Filter:**  
Quarterly GDP: HP filter (λ=1600) separates business cycle (smooth trend) from fluctuations. Economists extract "output gap" for monetary policy.

**Moving Average Smoothing:**  
Daily stock prices: 50-day MA removes noise, reveals trend. Crossover strategies (50-day vs 200-day) popular in technical analysis.

**Poor Fit:**  
Bitcoin prices: No stable seasonality, extreme volatility, structural breaks (2017 bubble). Classical decomposition fails—irregular component dominates, seasonal extraction meaningless.

**STL Advantage:**  
Electricity demand: Weekly + annual seasonality. STL handles multiple seasonal periods, robust to outliers (heatwave spikes don't distort pattern). Classical methods struggle with overlapping cycles.

## 4. Layer Breakdown
```
Classical Decomposition & Filtering Framework:

├─ Time Series Components:
│  ├─ Trend (T):
│  │   Long-term direction, smooth movement
│  │   Growth, decline, or stagnation over time
│  │   Example: Population growth, inflation trajectory
│  ├─ Seasonal (S):
│  │   Periodic, predictable fluctuations
│  │   Fixed period (monthly, quarterly, yearly)
│  │   Example: Retail sales (holiday spikes), temperature
│  ├─ Cyclic (C):
│  │   Non-periodic fluctuations (business cycles)
│  │   Variable duration (2-10 years)
│  │   Often merged with trend in practice
│  └─ Irregular / Random (I):
│      Unpredictable noise
│      Residual after removing trend + seasonal
│      White noise assumption (forecasting)
├─ Decomposition Models:
│  ├─ Additive Model:
│  │   ├─ Formula: Y_t = T_t + S_t + I_t
│  │   ├─ Assumption:
│  │   │   Seasonal variation independent of trend level
│  │   │   Constant amplitude of seasonal swings
│  │   ├─ Use Cases:
│  │   │   Stationary seasonality
│  │   │   Economic indicators (unemployment rate)
│  │   │   Temperature data
│  │   ├─ Estimation:
│  │   │   1. Extract trend (moving average)
│  │   │   2. Detrend: Y_t - T_t
│  │   │   3. Average by season → S_t
│  │   │   4. Residual: I_t = Y_t - T_t - S_t
│  │   └─ Limitations:
│  │       Doesn't handle growing seasonality
│  │       Sensitive to outliers
│  ├─ Multiplicative Model:
│  │   ├─ Formula: Y_t = T_t × S_t × I_t
│  │   ├─ Assumption:
│  │   │   Seasonal variation proportional to trend
│  │   │   Percentage changes constant, not absolute
│  │   ├─ Use Cases:
│  │   │   Economic time series (sales, production)
│  │   │   Exponential growth with seasonality
│  │   │   Airline passengers, retail sales
│  │   ├─ Log Transformation:
│  │   │   log(Y_t) = log(T_t) + log(S_t) + log(I_t)
│  │   │   Converts to additive model
│  │   │   Back-transform: Y_t = exp(log(T_t) + log(S_t) + log(I_t))
│  │   ├─ Estimation:
│  │   │   1. Extract trend (centered MA)
│  │   │   2. Detrend: Y_t / T_t
│  │   │   3. Average by season → S_t (ratios)
│  │   │   4. Normalize: Σ S_t = periods (or = 1 in logs)
│  │   │   5. Residual: I_t = Y_t / (T_t × S_t)
│  │   └─ Advantages:
│  │       Natural for percentage data (returns, growth rates)
│  │       Variance stabilization (via logs)
│  └─ Pseudo-Additive:
│      Y_t = T_t × (S_t + I_t)
│      Hybrid: Multiplicative trend, additive seasonal+irregular
├─ Classical Decomposition Algorithms:
│  ├─ Simple Moving Average Method:
│  │   ├─ Step 1: Estimate Trend
│  │   │   ├─ Centered Moving Average:
│  │   │   │   For period m (odd): T_t = (1/m) Σ Y_{t-k} to Y_{t+k}
│  │   │   │   For period m (even): Use 2×m weights (centered)
│  │   │   │   Example (m=12 monthly): 2×12 MA
│  │   │   │   T_t = (0.5×Y_{t-6} + Y_{t-5} + ... + Y_{t+5} + 0.5×Y_{t+6}) / 12
│  │   │   ├─ Loses m observations at ends
│  │   │   └─ Smooths out seasonal + irregular
│  │   ├─ Step 2: Detrend
│  │   │   Additive: D_t = Y_t - T_t
│  │   │   Multiplicative: D_t = Y_t / T_t
│  │   ├─ Step 3: Estimate Seasonal Component
│  │   │   ├─ Average detrended values by season:
│  │   │   │   S_j = average of all D_t where t is in season j
│  │   │   │   (e.g., all Januaries, all Februaries, ...)
│  │   │   ├─ Normalize (additive):
│  │   │   │   Adjust so Σ S_j = 0 over full cycle
│  │   │   ├─ Normalize (multiplicative):
│  │   │   │   Adjust so Σ S_j = m (or product = 1)
│  │   │   └─ Replicate pattern for all years
│  │   ├─ Step 4: Irregular Component
│  │   │   Additive: I_t = Y_t - T_t - S_t
│  │   │   Multiplicative: I_t = Y_t / (T_t × S_t)
│  │   └─ Issues:
│  │       Assumes constant seasonal pattern
│  │       Outliers contaminate seasonal estimates
│  │       No handling of missing data
│  ├─ X-11/X-12/X-13ARIMA-SEATS:
│  │   ├─ Census Bureau Method:
│  │   │   Industry standard for official statistics
│  │   │   Iterative refinement of components
│  │   ├─ X-13 Features:
│  │   │   ├─ Pre-adjustment:
│  │   │   │   Detect outliers (AO, LS, TC)
│  │   │   │   Calendar effects (trading days, holidays)
│  │   │   │   Prior adjustment for known events
│  │   │   ├─ ARIMA Modeling:
│  │   │   │   RegARIMA: Regression + ARIMA
│  │   │   │   Forecast/backcast to extend series
│  │   │   │   Improves end-point estimates
│  │   │   ├─ Seasonal Adjustment:
│  │   │   │   Moving averages (Henderson filter for trend)
│  │   │   │   Composite seasonal filters
│  │   │   │   Iterative: Re-estimate with adjusted data
│  │   │   ├─ Diagnostics:
│  │   │   │   M-statistics (quality metrics)
│  │   │   │   Spectral diagnostics
│  │   │   │   Residual autocorrelation tests
│  │   │   └─ Final Output:
│  │   │       Seasonally adjusted series
│  │   │       Trend-cycle component
│  │   │       Irregular component
│  │   └─ SEATS (Signal Extraction in ARIMA Time Series):
│  │       Model-based decomposition
│  │       ARIMA representation of each component
│  │       Optimal (minimum MSE) extraction
│  └─ STL (Seasonal and Trend decomposition using LOESS):
│      ├─ Cleveland et al. (1990):
│      │   Versatile, robust decomposition
│      │   Handles any seasonal period
│      ├─ Algorithm:
│      │   ├─ Outer Loop (robustness):
│      │   │   Iterate with robust weights (downweight outliers)
│      │   ├─ Inner Loop:
│      │   │   1. Detrending:
│      │   │      Compute seasonal component
│      │   │      Y_t - S_t = T_t + I_t
│      │   │   2. Trend Smoothing:
│      │   │      LOESS on deseasonalized series
│      │   │      Low-pass filter (large window)
│      │   │   3. Seasonal Smoothing:
│      │   │      Detrend: Y_t - T_t
│      │   │      Cycle sub-series (all Jan, all Feb, ...)
│      │   │      LOESS on each sub-series
│      │   │      Low-pass filter on seasonal component
│      │   │   4. Iterate until convergence
│      │   └─ Parameters:
│      │       n_p: Period of seasonal component
│      │       n_s: Seasonal smoothing parameter (odd)
│      │       n_t: Trend smoothing parameter (odd)
│      │       n_l: Low-pass filter length
│      ├─ Advantages:
│      │   ├─ Changing seasonality over time
│      │   ├─ Multiple seasonal periods (MSTL extension)
│      │   ├─ Robust to outliers (via weights)
│      │   ├─ No need for log transform
│      │   └─ Flexible smoothing parameters
│      └─ Use Cases:
│          Electricity demand (multiple seasonality)
│          Irregular seasonal patterns
│          Outlier-contaminated data
├─ Filtering Techniques:
│  ├─ Moving Average Filters:
│  │   ├─ Simple Moving Average (SMA):
│  │   │   ├─ Formula: MA_t = (1/k) Σ_{j=0}^{k-1} Y_{t-j}
│  │   │   ├─ Weights: Equal (1/k each)
│  │   │   ├─ Lag: (k-1)/2 periods behind
│  │   │   ├─ Frequency Response:
│  │   │   │   Low-pass filter (smooths high freq)
│  │   │   │   Cutoff at π/k radians
│  │   │   └─ Trade-off:
│  │   │       Larger k → smoother, more lag
│  │   │       Smaller k → responsive, noisy
│  │   ├─ Weighted Moving Average:
│  │   │   WMA_t = Σ w_j Y_{t-j}, where Σ w_j = 1
│  │   │   Linear weights: w_j = 2(k-j+1)/(k(k+1))
│  │   │   More weight on recent observations
│  │   ├─ Centered Moving Average:
│  │   │   CMA_t = (1/k) Σ_{j=-(k-1)/2}^{(k-1)/2} Y_{t+j}
│  │   │   Symmetric (no phase shift)
│  │   │   Cannot compute in real-time (needs future data)
│  │   │   Used in decomposition
│  │   └─ Spencer's 15-Point Filter:
│  │       Weighted 15-term MA for graduation
│  │       Smooth curve fitting
│  ├─ Exponential Smoothing:
│  │   ├─ Simple Exponential Smoothing (SES):
│  │   │   ├─ Formula: S_t = α Y_t + (1-α) S_{t-1}
│  │   │   ├─ Equivalent: S_t = S_{t-1} + α(Y_t - S_{t-1})
│  │   │   ├─ Parameter: α ∈ (0, 1) (smoothing constant)
│  │   │   │   High α → responsive (less smoothing)
│  │   │   │   Low α → smooth (more inertia)
│  │   │   ├─ Weights: Exponential decay
│  │   │   │   w_j = α(1-α)^j for lag j
│  │   │   │   Recent observations weighted more
│  │   │   ├─ Advantages:
│  │   │   │   Only need current smoothed value + new observation
│  │   │   │   Minimal storage (recursive)
│  │   │   │   Minimal lag compared to SMA
│  │   │   └─ Use: Level estimation (no trend/season)
│  │   ├─ Double Exponential Smoothing (Holt):
│  │   │   ├─ Level: L_t = α Y_t + (1-α)(L_{t-1} + T_{t-1})
│  │   │   ├─ Trend: T_t = β(L_t - L_{t-1}) + (1-β) T_{t-1}
│  │   │   ├─ Parameters: α (level), β (trend)
│  │   │   └─ Forecast: F_{t+h} = L_t + h × T_t
│  │   └─ Triple Exponential Smoothing (Holt-Winters):
│  │       ├─ Additive Seasonality:
│  │       │   L_t = α(Y_t - S_{t-m}) + (1-α)(L_{t-1} + T_{t-1})
│  │       │   T_t = β(L_t - L_{t-1}) + (1-β) T_{t-1}
│  │       │   S_t = γ(Y_t - L_t) + (1-γ) S_{t-m}
│  │       ├─ Multiplicative Seasonality:
│  │       │   L_t = α(Y_t / S_{t-m}) + (1-α)(L_{t-1} + T_{t-1})
│  │       │   S_t = γ(Y_t / L_t) + (1-γ) S_{t-m}
│  │       └─ Parameters: α (level), β (trend), γ (seasonal)
│  ├─ Hodrick-Prescott (HP) Filter:
│  │   ├─ Objective:
│  │   │   Decompose into trend (τ) and cycle (c)
│  │   │   Y_t = τ_t + c_t
│  │   ├─ Optimization Problem:
│  │   │   min_τ [ Σ(Y_t - τ_t)² + λ Σ((τ_{t+1} - τ_t) - (τ_t - τ_{t-1}))² ]
│  │   │   First term: Fit to data
│  │   │   Second term: Penalize curvature (second differences)
│  │   ├─ Smoothing Parameter λ:
│  │   │   λ → 0: Trend follows data closely (no smoothing)
│  │   │   λ → ∞: Trend becomes linear
│  │   │   Standard values:
│  │   │     Annual data: λ = 100
│  │   │     Quarterly data: λ = 1600 (most common)
│  │   │     Monthly data: λ = 14400
│  │   ├─ Solution:
│  │   │   (I + λK'K)τ = Y
│  │   │   K: Second difference operator matrix
│  │   │   Sparse band matrix (efficient solve)
│  │   ├─ Properties:
│  │   │   Symmetric (two-sided filter)
│  │   │   Linear operator
│  │   │   Minimizes penalized sum of squares
│  │   ├─ Criticisms:
│  │   │   End-point problem (trend less reliable at boundaries)
│  │   │   Spurious cycles (can create artificial oscillations)
│  │   │   Not real-time (needs full data)
│  │   │   Choice of λ arbitrary
│  │   └─ Applications:
│  │       Business cycle extraction (GDP, output gap)
│  │       Macro policy analysis
│  │       Trend-following in finance
│  ├─ Butterworth Filter:
│  │   ├─ Frequency Domain Filter:
│  │   │   Maximally flat passband
│  │   │   Smooth frequency response (no ripples)
│  │   ├─ Transfer Function:
│  │   │   |H(ω)|² = 1 / (1 + (ω/ω_c)^(2n))
│  │   │   ω_c: Cutoff frequency
│  │   │   n: Filter order (higher → sharper cutoff)
│  │   ├─ Types:
│  │   │   Low-pass: Keep frequencies below ω_c (trend extraction)
│  │   │   High-pass: Keep frequencies above ω_c (detrending)
│  │   │   Band-pass: Keep frequencies in range [ω_1, ω_2] (cycle isolation)
│  │   ├─ Implementation:
│  │   │   Design in frequency domain
│  │   │   Apply via FFT or recursive IIR filter
│  │   ├─ Advantages:
│  │   │   Precise frequency control
│  │   │   No phase distortion (zero-phase filtering)
│  │   │   Clean frequency separation
│  │   └─ Disadvantages:
│  │       Requires full data (acausal)
│  │       Not real-time
│  │       Gibbs phenomenon at discontinuities
│  ├─ Baxter-King Filter:
│  │   ├─ Band-Pass Filter:
│  │   │   Extract business cycle frequencies (2-8 years)
│  │   │   Remove trend and high-frequency noise
│  │   ├─ Formula:
│  │   │   Symmetric moving average
│  │   │   Weights chosen to pass [ω_L, ω_H]
│  │   ├─ Implementation:
│  │   │   Truncated ideal filter (finite approximation)
│  │   │   Loses 2K observations (K on each end)
│  │   └─ Use:
│  │       Business cycle analysis
│  │       Macro co-movement studies
│  └─ Christiano-Fitzgerald Filter:
│      ├─ Asymmetric Band-Pass:
│      │   Optimal approximation to ideal filter
│      │   Uses full sample (no data loss)
│      ├─ Weights:
│      │   Data-dependent (ARIMA model assumed)
│      │   Different weights at each time point
│      └─ Better End-Point Behavior:
│          Compared to Baxter-King
│          Useful for real-time analysis
├─ Kalman Filter (State-Space Approach):
│  ├─ State-Space Model:
│  │   ├─ State Equation:
│  │   │   x_t = F x_{t-1} + w_t    (w_t ~ N(0, Q))
│  │   │   x_t: Unobserved state (trend, seasonal)
│  │   ├─ Observation Equation:
│  │   │   Y_t = H x_t + v_t        (v_t ~ N(0, R))
│  │   └─ Components:
│  │       F: State transition matrix
│  │       H: Observation matrix
│  │       Q: Process noise covariance
│  │       R: Observation noise variance
│  ├─ Kalman Recursion:
│  │   ├─ Prediction:
│  │   │   x_{t|t-1} = F x_{t-1|t-1}
│  │   │   P_{t|t-1} = F P_{t-1|t-1} F' + Q
│  │   ├─ Update:
│  │   │   K_t = P_{t|t-1} H' (H P_{t|t-1} H' + R)^(-1)  (Kalman gain)
│  │   │   x_{t|t} = x_{t|t-1} + K_t(Y_t - H x_{t|t-1})
│  │   │   P_{t|t} = (I - K_t H) P_{t|t-1}
│  │   └─ Properties:
│  │       Optimal (minimum MSE) estimator
│  │       Real-time (recursive)
│  │       Handles missing data naturally
│  ├─ Structural Time Series Models:
│  │   ├─ Local Level Model:
│  │   │   T_t = T_{t-1} + w_t^(T)
│  │   │   Y_t = T_t + v_t
│  │   ├─ Local Linear Trend:
│  │   │   T_t = T_{t-1} + β_{t-1} + w_t^(T)
│  │   │   β_t = β_{t-1} + w_t^(β)
│  │   │   Y_t = T_t + v_t
│  │   └─ With Seasonality:
│  │       Add seasonal state component
│  │       Trigonometric or dummy variable form
│  ├─ Advantages:
│  │   Flexible (any component structure)
│  │   Probabilistic (confidence intervals)
│  │   Real-time updates
│  │   Optimal filtering
│  └─ Applications:
│      Nowcasting (real-time estimates)
│      Unobserved components models
│      Adaptive forecasting
├─ Frequency Domain Methods:
│  ├─ Fourier Transform:
│  │   ├─ Decompose into sinusoids:
│  │   │   Y_t = Σ [A_k cos(ω_k t) + B_k sin(ω_k t)]
│  │   │   ω_k = 2πk/T (fundamental frequencies)
│  │   ├─ Discrete Fourier Transform (DFT):
│  │   │   X_k = Σ_{t=0}^{T-1} Y_t e^(-i 2πkt/T)
│  │   ├─ Fast Fourier Transform (FFT):
│  │   │   O(T log T) algorithm
│  │   │   Efficient computation
│  │   └─ Inverse FFT:
│  │       Reconstruct time series from frequencies
│  ├─ Periodogram:
│  │   ├─ Power Spectral Density:
│  │   │   I(ω_k) = (1/T) |X_k|²
│  │   │   Shows power at each frequency
│  │   ├─ Interpretation:
│  │   │   Peaks indicate dominant cycles
│  │   │   Flat → white noise
│  │   │   Decay → autocorrelation
│  │   └─ Smoothing:
│  │       Raw periodogram inconsistent
│  │       Smooth via window (Welch, Bartlett)
│  ├─ Spectral Filtering:
│  │   ├─ Ideal Band-Pass:
│  │   │   Multiply DFT by frequency mask
│  │   │   Keep [ω_L, ω_H], zero elsewhere
│  │   ├─ Inverse Transform:
│  │   │   Reconstruct filtered series
│  │   └─ Advantages:
│  │       Clean frequency separation
│  │       No phase shift (zero-phase filtering)
│  └─ Wavelet Transform:
│      Time-frequency decomposition
│      Localized in time and frequency
│      Good for non-stationary signals
└─ Practical Considerations:
   ├─ Choice of Method:
   │   ├─ Simple patterns → Classical decomposition
   │   ├─ Changing seasonality → STL
   │   ├─ Official statistics → X-13ARIMA-SEATS
   │   ├─ Business cycles → HP filter, Baxter-King
   │   ├─ Real-time → Exponential smoothing, Kalman
   │   └─ Precise frequency control → Butterworth, FFT
   ├─ Parameter Selection:
   │   ├─ Moving average window: k = period length (seasonality)
   │   ├─ HP filter λ: 1600 (quarterly), 14400 (monthly)
   │   ├─ STL smoothing: n_s odd, larger for more smoothing
   │   ├─ Exponential α: Cross-validation, minimize MSE
   │   └─ Kalman Q, R: Maximum likelihood estimation
   ├─ Validation:
   │   ├─ Residual diagnostics:
   │   │   Check irregular component for white noise
   │   │   ACF should show no pattern
   │   │   Ljung-Box test for autocorrelation
   │   ├─ Seasonal stability:
   │   │   Plot seasonal component over time
   │   │   Should be consistent (classical methods)
   │   ├─ Forecasting:
   │   │   Extrapolate trend + seasonal
   │   │   Out-of-sample validation
   │   └─ Visual inspection:
   │       Plot original vs decomposed components
   │       Ensure sensible separation
   ├─ Common Pitfalls:
   │   ├─ Over-smoothing:
   │   │   Removes signal, not just noise
   │   │   Loss of valuable information
   │   ├─ End-point bias:
   │   │   HP filter, moving averages unreliable at ends
   │   │   Use forecasts to extend series
   │   ├─ Spurious seasonality:
   │   │   Outliers create false seasonal pattern
   │   │   Use robust methods (STL)
   │   └─ Wrong model type:
   │       Additive vs multiplicative
   │       Test both, compare residuals
   └─ Software:
      R: decompose(), stl(), mFilter, stats
      Python: statsmodels.tsa.seasonal_decompose, scipy.signal
      SAS: PROC X13, PROC UCM (Kalman)
      EViews: X-13, HP filter built-in
```

**Interaction:** Observe time series → Choose decomposition model (additive/multiplicative/STL) → Extract trend (MA, HP, LOESS) → Calculate seasonal component → Compute irregular residual → Validate (ACF on residuals) → Use for forecasting or analysis.

## 5. Mini-Project
Implement decomposition methods and filters, compare performance:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.fft import fft, ifft, fftfreq
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("CLASSICAL DECOMPOSITION AND FILTERING: TIME SERIES ANALYSIS")
print("="*80)

class TimeSeriesDecomposer:
    """Classical decomposition methods"""
    
    def __init__(self):
        pass
    
    def additive_decomposition(self, y, period=12):
        """
        Classical additive decomposition: Y = T + S + I
        
        Parameters:
        - y: Time series (array)
        - period: Seasonal period (e.g., 12 for monthly)
        """
        n = len(y)
        
        # Step 1: Estimate trend using centered moving average
        if period % 2 == 0:
            # Even period: 2×m MA
            weights = np.ones(period) / period
            weights[0] = weights[-1] = 0.5 / period
            trend = np.convolve(y, weights, mode='same')
            # Fix endpoints
            half_window = period // 2
            trend[:half_window] = np.nan
            trend[-half_window:] = np.nan
        else:
            # Odd period: simple MA
            half_window = period // 2
            trend = np.full(n, np.nan)
            for t in range(half_window, n - half_window):
                trend[t] = np.mean(y[t-half_window:t+half_window+1])
        
        # Step 2: Detrend
        detrended = y - trend
        
        # Step 3: Estimate seasonal component
        seasonal = np.full(n, np.nan)
        seasonal_avg = np.zeros(period)
        
        for s in range(period):
            # Average all observations in season s
            season_vals = detrended[s::period]
            season_vals = season_vals[~np.isnan(season_vals)]
            if len(season_vals) > 0:
                seasonal_avg[s] = np.mean(season_vals)
        
        # Normalize (sum to zero)
        seasonal_avg -= np.mean(seasonal_avg)
        
        # Replicate pattern
        for t in range(n):
            seasonal[t] = seasonal_avg[t % period]
        
        # Step 4: Irregular component
        irregular = y - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'irregular': irregular,
            'seasonal_avg': seasonal_avg
        }
    
    def multiplicative_decomposition(self, y, period=12):
        """
        Classical multiplicative decomposition: Y = T × S × I
        Convert to additive via log transform
        """
        # Take logs (ensure positive)
        y_log = np.log(np.maximum(y, 1e-10))
        
        # Additive decomposition on logs
        decomp_log = self.additive_decomposition(y_log, period)
        
        # Back-transform
        trend = np.exp(decomp_log['trend'])
        seasonal = np.exp(decomp_log['seasonal'])
        irregular = np.exp(decomp_log['irregular'])
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'irregular': irregular,
            'seasonal_avg': np.exp(decomp_log['seasonal_avg'])
        }
    
    def stl_decomposition(self, y, period=12, n_s=7, n_t=None, n_l=13, n_i=2, n_o=0):
        """
        STL: Seasonal and Trend decomposition using LOESS
        Simplified implementation
        
        Parameters:
        - n_s: Seasonal smoothing parameter (odd, ≥3)
        - n_t: Trend smoothing parameter (odd)
        - n_l: Low-pass filter length
        - n_i: Inner loop iterations
        - n_o: Outer loop iterations (robustness)
        """
        if n_t is None:
            n_t = int(np.ceil((1.5 * period) / (1 - 1.5 / n_s)))
            if n_t % 2 == 0:
                n_t += 1
        
        n = len(y)
        seasonal = np.zeros(n)
        trend = np.zeros(n)
        weights = np.ones(n)
        
        for outer in range(max(1, n_o)):
            for inner in range(n_i):
                # Step 1: Detrend
                detrended = y - trend
                
                # Step 2: Cycle-subseries smoothing
                seasonal_temp = np.zeros(n)
                for s in range(period):
                    # Extract subseries for season s
                    indices = np.arange(s, n, period)
                    sub_series = detrended[indices]
                    
                    # LOESS smooth (simplified: moving average)
                    smoothed = self._moving_average_smooth(sub_series, n_s, weights[indices])
                    seasonal_temp[indices] = smoothed
                
                # Step 3: Low-pass filter on seasonal
                seasonal = self._moving_average_smooth(seasonal_temp, n_l, weights)
                
                # Step 4: Deseasonalize and smooth for trend
                deseasonalized = y - seasonal
                trend = self._moving_average_smooth(deseasonalized, n_t, weights)
            
            # Outer loop: Compute robustness weights
            if n_o > 0 and outer < n_o - 1:
                remainder = y - trend - seasonal
                weights = self._bisquare_weights(remainder)
        
        irregular = y - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'irregular': irregular
        }
    
    def _moving_average_smooth(self, y, window, weights=None):
        """Weighted moving average"""
        if weights is None:
            weights = np.ones(len(y))
        
        n = len(y)
        smoothed = np.zeros(n)
        half_window = window // 2
        
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            window_vals = y[start:end]
            window_weights = weights[start:end]
            
            if np.sum(window_weights) > 0:
                smoothed[i] = np.sum(window_vals * window_weights) / np.sum(window_weights)
            else:
                smoothed[i] = y[i]
        
        return smoothed
    
    def _bisquare_weights(self, residuals):
        """Compute bisquare robustness weights"""
        abs_res = np.abs(residuals)
        median_abs = np.median(abs_res)
        
        if median_abs == 0:
            return np.ones(len(residuals))
        
        standardized = abs_res / (6 * median_abs)
        weights = np.where(standardized < 1, (1 - standardized**2)**2, 0)
        
        return weights

class TimeSeriesFilter:
    """Various filtering methods"""
    
    def moving_average(self, y, window=5, center=True):
        """Simple moving average filter"""
        if center:
            # Centered MA (symmetric)
            smoothed = np.convolve(y, np.ones(window)/window, mode='same')
        else:
            # Trailing MA (causal, real-time)
            smoothed = np.zeros(len(y))
            for t in range(len(y)):
                start = max(0, t - window + 1)
                smoothed[t] = np.mean(y[start:t+1])
        
        return smoothed
    
    def exponential_smoothing(self, y, alpha=0.3):
        """Simple exponential smoothing"""
        n = len(y)
        smoothed = np.zeros(n)
        smoothed[0] = y[0]
        
        for t in range(1, n):
            smoothed[t] = alpha * y[t] + (1 - alpha) * smoothed[t-1]
        
        return smoothed
    
    def hodrick_prescott(self, y, lam=1600):
        """
        Hodrick-Prescott filter
        Minimize: Σ(y_t - τ_t)² + λ Σ((τ_{t+1} - τ_t) - (τ_t - τ_{t-1}))²
        """
        n = len(y)
        
        # Build second difference matrix K
        # K @ τ computes second differences
        diag_vals = np.array([1, -2, 1])
        offsets = np.array([0, 1, 2])
        K = diags(diag_vals, offsets, shape=(n-2, n)).tocsr()
        
        # Solve: (I + λ K'K) τ = y
        I = diags([1], [0], shape=(n, n))
        A = I + lam * K.T @ K
        
        trend = spsolve(A, y)
        cycle = y - trend
        
        return trend, cycle
    
    def butterworth_filter(self, y, cutoff_freq, fs=1.0, order=5, btype='low'):
        """
        Butterworth filter
        
        Parameters:
        - cutoff_freq: Cutoff frequency (normalized, 0-0.5)
        - fs: Sampling frequency
        - order: Filter order (higher = sharper)
        - btype: 'low', 'high', or 'band'
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
        
        # Apply zero-phase filter (forward-backward)
        filtered = signal.filtfilt(b, a, y)
        
        return filtered
    
    def band_pass_filter(self, y, low_freq, high_freq, fs=1.0):
        """
        Band-pass filter (extract specific frequency range)
        Uses FFT
        """
        n = len(y)
        
        # FFT
        Y = fft(y)
        freqs = fftfreq(n, 1/fs)
        
        # Create mask
        mask = np.zeros(n)
        mask[(np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)] = 1
        
        # Apply mask and inverse FFT
        Y_filtered = Y * mask
        filtered = np.real(ifft(Y_filtered))
        
        return filtered

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
```

## 6. Challenge Round
1. **Changing Seasonality:** Simulate series where seasonal pattern evolves over time (amplitude increases). Compare Classical vs STL decomposition. Which handles better? Quantify via RMSE.

2. **End-Point Problem:** HP filter unreliable at boundaries. Implement: (1) extend series with ARIMA forecast/backcast, (2) apply HP filter, (3) compare end-point estimates to true trend. Does extension help?

3. **Multiple Seasonality:** Create data with daily + weekly + annual patterns (e.g., electricity demand). Implement MSTL (multiple STL). Extract all seasonal components simultaneously.

4. **Outlier Robustness:** Add 5% outliers (spikes) to series. Compare Classical vs STL decomposition. Does STL correctly downweight outliers? Plot robustness weights.

5. **Real-Time Forecast:** Implement expanding window: fit Holt-Winters on data up to time t, forecast t+1, observe, refit. Track forecast errors. Compare to static model (fit once, never update). Which better?

## 7. Key References
- [Cleveland et al., "STL: A Seasonal-Trend Decomposition Procedure Based on Loess" (1990)](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf) - original STL paper
- [Hodrick & Prescott, "Postwar U.S. Business Cycles: An Empirical Investigation" (1997)](https://www.jstor.org/stable/2953682) - HP filter foundation
- [Hamilton, "Why You Should Never Use the Hodrick-Prescott Filter" (2018)](https://www.nber.org/papers/w23429) - critical analysis of HP filter pitfalls

---
**Status:** Fundamental time series technique | **Complements:** Trend Analysis, Seasonality Modeling, ARIMA, Forecasting, Signal Processing
