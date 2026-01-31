# Stationarity Testing and Transformations

## 1. Concept Skeleton
**Definition:** Statistical tests for weak stationarity; unit root detection; transformations achieving stationarity; differencing operators; detrending; variance stabilization; Box-Cox transforms  
**Purpose:** Diagnose non-stationarity; validate modeling assumptions; prevent spurious regression; ensure consistent estimators; enable valid inference; prepare data for ARMA/GARCH  
**Prerequisites:** Time series fundamentals, autocorrelation, asymptotic theory, hypothesis testing, OLS regression, maximum likelihood

## 2. Comparative Framing
| Test | Null Hypothesis | Alternative | Power Against | Limitations |
|------|----------------|-------------|---------------|-------------|
| **ADF** | Unit root (I(1)) | Stationary (I(0)) | AR alternatives | Low power, size distortions |
| **PP** | Unit root | Stationary | MA errors, heteroskedasticity | Spurious rejections with breaks |
| **KPSS** | Stationary | Unit root | Complements ADF | Restrictive under alternative |
| **DF-GLS** | Unit root | Stationary | Near-unit root | Better power than ADF |
| **ZA** | Unit root | Stationary with break | Structural breaks | Must specify break location |
| **Variance Ratio** | Random walk | Mean reversion | Predictability | Assumes i.i.d. increments |

| Transformation | Purpose | Formula | When to Use | Side Effects |
|----------------|---------|---------|-------------|--------------|
| **First Difference** | Remove trend/unit root | Δy_t = y_t - y_{t-1} | I(1) processes | Lose one obs, induces MA(1) |
| **Seasonal Difference** | Remove seasonality | Δ_s y_t = y_t - y_{t-s} | Seasonal unit roots | Lose s observations |
| **Log Transform** | Stabilize variance, multiplicative → additive | ln(y_t) | Exponential growth, heteroskedasticity | Only for y_t > 0 |
| **Box-Cox** | Flexible variance stabilization | (y_t^λ - 1)/λ if λ≠0; ln(y_t) if λ=0 | Unknown transformation | Requires positive data |
| **Detrending** | Remove deterministic trend | y_t - (α̂ + β̂t) | Trend stationarity | Model misspecification risk |
| **HP Filter** | Extract cycle | Minimize trend + λ(second diff)² | Business cycles | Spurious cycles, end-point bias |

## 3. Examples + Counterexamples

**Simple Example:**  
Stock price: ADF p=0.72 (unit root). First difference (returns): ADF p=0.001 (stationary). I(1) process—model returns, not prices.

**Perfect Fit:**  
GDP with linear trend: Detrend via regression. Residuals pass ADF (p<0.01). Trend stationary—shocks temporary, mean reversion to trend.

**Unit Root:**  
Exchange rate: Random walk with drift. ADF fails to reject (p=0.45). PP confirms (p=0.38). KPSS rejects stationarity (p=0.01). All three agree: I(1).

**Structural Break:**  
Interest rates pre/post-2008 crisis. Standard ADF p=0.15 (appears non-stationary). Zivot-Andrews test p=0.02, break detected at 2008 Q4. Actually stationary around break.

**Over-Differencing:**  
Stationary AR(1): φ=0.5. Difference anyway → MA(1) with θ=-1 (non-invertible root). Forecasts degrade, inefficient estimation.

**Poor Fit:**  
Apply log transform to series with negative values → NaN. Box-Cox requires y_t > 0. Must shift: y_t + constant, or use different method.

## 4. Layer Breakdown
```
Stationarity Testing and Transformations Framework:

├─ Stationarity Concepts:
│  ├─ Strict Stationarity:
│  │   Joint distribution invariant to time shifts
│  │   F(y_t1,...,y_tn) = F(y_t1+h,...,y_tn+h)
│  │   Not testable—requires all moments
│  ├─ Weak Stationarity (Covariance):
│  │   ├─ Constant mean: E[Y_t] = μ
│  │   ├─ Constant variance: Var(Y_t) = σ²
│  │   └─ Autocovariance depends only on lag: Cov(Y_t, Y_{t-k}) = γ_k
│  ├─ Trend Stationarity (TS):
│  │   Y_t = T_t + X_t where T_t deterministic, X_t stationary
│  │   Remove trend → stationary
│  │   Shocks temporary (mean reversion to trend)
│  ├─ Difference Stationarity (DS):
│  │   Y_t non-stationary, ΔY_t stationary
│  │   Integrated of order d: I(d)
│  │   Shocks permanent (no mean reversion in levels)
│  └─ Consequences of Non-Stationarity:
│      Spurious regression (high R² between unrelated series)
│      Invalid t-statistics (standard distributions don't apply)
│      Inconsistent estimators
│      Non-ergodic (sample moments ≠ population)
├─ Unit Root Tests:
│  ├─ Dickey-Fuller (DF) Test:
│  │   ├─ Model:
│  │   │   Δy_t = α + βt + γy_{t-1} + ε_t
│  │   │   H0: γ = 0 (unit root)
│  │   │   H1: γ < 0 (stationary)
│  │   ├─ Test Statistic:
│  │   │   t_γ = γ̂ / SE(γ̂)
│  │   │   NOT t-distributed under H0!
│  │   │   Uses Dickey-Fuller critical values (more negative)
│  │   ├─ Three Specifications:
│  │   │   1. No constant, no trend: Δy_t = γy_{t-1} + ε_t
│  │   │   2. Constant: Δy_t = α + γy_{t-1} + ε_t
│  │   │   3. Constant + trend: Δy_t = α + βt + γy_{t-1} + ε_t
│  │   └─ Decision Rule:
│  │       If t_γ < critical value → Reject H0 (stationary)
│  │       If t_γ ≥ critical value → Fail to reject (unit root)
│  ├─ Augmented Dickey-Fuller (ADF):
│  │   ├─ Model:
│  │   │   Δy_t = α + βt + γy_{t-1} + Σ δ_i Δy_{t-i} + ε_t
│  │   │   Add lagged differences to capture serial correlation
│  │   ├─ Lag Selection:
│  │   │   AIC: 2k - 2ln(L)
│  │   │   BIC: k·ln(T) - 2ln(L)
│  │   │   Sequential testing (start high, drop insignificant)
│  │   ├─ Null Hypothesis:
│  │   │   γ = 0 (unit root)
│  │   │   Under H0: y_t ~ I(1)
│  │   ├─ Advantages:
│  │   │   Handles serial correlation
│  │   │   Most widely used
│  │   │   Software readily available
│  │   └─ Disadvantages:
│  │       Low power (fails to reject false H0)
│  │       Size distortions (MA errors)
│  │       Sensitive to structural breaks
│  ├─ Phillips-Perron (PP):
│  │   ├─ Model:
│  │   │   Δy_t = α + βt + γy_{t-1} + ε_t
│  │   │   No lagged differences
│  │   ├─ Correction:
│  │   │   Non-parametric adjustment for serial correlation
│  │   │   Newey-West HAC standard errors
│  │   │   Modified t-statistic: t_γ* = t_γ · √(σ²/σ_LR²) + adjustment
│  │   ├─ Bandwidth Selection:
│  │   │   Newey-West automatic (4(T/100)^(2/9))
│  │   │   Andrews (data-dependent)
│  │   ├─ Advantages:
│  │   │   Robust to heteroskedasticity
│  │   │   No lag selection needed
│  │   │   Robust to MA errors
│  │   └─ Disadvantages:
│  │       Spurious rejections with structural breaks
│  │       Choice of bandwidth affects results
│  ├─ KPSS Test:
│  │   ├─ Key Difference:
│  │   │   H0: Stationary (opposite of ADF/PP!)
│  │   │   H1: Unit root
│  │   │   Complement to ADF
│  │   ├─ Model:
│  │   │   y_t = βt + r_t + ε_t
│  │   │   r_t: Random walk (r_t = r_{t-1} + u_t)
│  │   │   Test if Var(u_t) = 0 (no random walk → stationary)
│  │   ├─ Test Statistic:
│  │   │   KPSS = (1/T²) Σ S_t² / σ̂_LR²
│  │   │   S_t: Partial sum of residuals
│  │   │   Large KPSS → Reject stationarity
│  │   ├─ Two Specifications:
│  │   │   Level: Test around constant mean
│  │   │   Trend: Test around linear trend
│  │   └─ Interpretation with ADF:
│  │       Both fail to reject → Inconclusive
│  │       ADF rejects, KPSS fails → Stationary
│  │       ADF fails, KPSS rejects → Unit root
│  │       Both reject → Possible break or other issue
│  ├─ DF-GLS (Elliott-Rothenberg-Stock):
│  │   ├─ Modification:
│  │   │   GLS detrending before running DF test
│  │   │   Better power near unit root
│  │   ├─ Procedure:
│  │   │   1. GLS detrend: y_t^d = y_t - β̂_GLS·z_t
│  │   │   2. Run ADF on y_t^d
│  │   ├─ Local-to-Unity Asymptotics:
│  │   │   Optimal when γ = 1 - c/T (near unit root)
│  │   │   More powerful than ADF in finite samples
│  │   └─ Recommendation:
│  │       Preferred over ADF when power important
│  │       Especially for quarterly/annual data
│  ├─ Zivot-Andrews (ZA):
│  │   ├─ Purpose:
│  │   │   Test unit root allowing one structural break
│  │   │   Standard tests have low power with breaks
│  │   ├─ Model:
│  │   │   Δy_t = α + βt + θDU_t + δDT_t + γy_{t-1} + Σδ_iΔy_{t-i} + ε_t
│  │   │   DU_t: Dummy for level shift (1 if t > TB)
│  │   │   DT_t: Dummy for trend change (t - TB if t > TB, else 0)
│  │   ├─ Break Point Selection:
│  │   │   Endogenous: Test all possible breaks
│  │   │   Choose TB that minimizes ADF t-statistic
│  │   │   Trimming: Exclude first/last 15% of sample
│  │   ├─ Three Models:
│  │   │   A: Break in level only
│  │   │   B: Break in trend only
│  │   │   C: Break in both
│  │   └─ Critical Values:
│  │       More negative than standard ADF
│  │       Account for search over break points
│  ├─ Seasonal Unit Roots:
│  │   ├─ HEGY Test (Hylleberg et al.):
│  │   │   Tests for unit roots at seasonal frequencies
│  │   │   Example (quarterly): Test at 0, π, π/2, 3π/2
│  │   │   Separate tests for each frequency
│  │   ├─ Model (quarterly):
│  │   │   (1-L⁴)y_t = α + βt + π₁y₁,t-1 + π₂y₂,t-1 + π₃y₃,t-2 + π₄y₃,t-1 + ε_t
│  │   │   Test π₁=0 (zero freq), π₂=0 (annual), π₃=π₄=0 (semiannual)
│  │   └─ Implication:
│  │       May need seasonal differencing: (1-L^s)
│  │       Or deterministic seasonals sufficient
│  └─ Variance Ratio Tests:
│      ├─ Lo-MacKinlay:
│      │   VR(q) = Var(q-period return) / [q × Var(1-period return)]
│      │   Under random walk: VR(q) = 1
│      │   Mean reversion: VR(q) < 1
│      │   Momentum: VR(q) > 1
│      ├─ Test Statistic:
│      │   z = [VR(q) - 1] / SE
│      │   Asymptotically N(0,1) under H0
│      └─ Multiple Variance Ratios:
│          Joint test across q = 2, 4, 8, 16
│          Chow-Denning max statistic
├─ Transformations to Achieve Stationarity:
│  ├─ Differencing:
│  │   ├─ First Difference:
│  │   │   Δy_t = y_t - y_{t-1} = (1-L)y_t
│  │   │   Removes stochastic trend (unit root)
│  │   │   I(1) → I(0)
│  │   ├─ Properties:
│  │   │   Introduces MA(1) correlation
│  │   │   Loses one observation
│  │   │   Variance may increase
│  │   ├─ Second Difference:
│  │   │   Δ²y_t = Δy_t - Δy_{t-1} = (1-L)²y_t
│  │   │   For I(2) processes (rare)
│  │   ├─ Seasonal Difference:
│  │   │   Δ_s y_t = y_t - y_{t-s}
│  │   │   s: Seasonal period (12 monthly, 4 quarterly)
│  │   │   Removes seasonal unit root
│  │   ├─ Combined Differencing:
│  │   │   (1-L)(1-L^s)y_t
│  │   │   Both non-seasonal and seasonal unit roots
│  │   │   SARIMA(p,1,q)(P,1,Q)_s
│  │   └─ Risks:
│  │       Over-differencing → Non-invertible MA root
│  │       Loss of long-run information
│  │       ADF on differences not standard (Dickey-Pantula)
│  ├─ Logarithmic Transform:
│  │   ├─ Purpose:
│  │   │   Stabilize variance (multiplicative → additive)
│  │   │   Interpret differences as growth rates
│  │   │   Ln(y_t) - ln(y_{t-1}) ≈ (y_t - y_{t-1})/y_{t-1}
│  │   ├─ When Effective:
│  │   │   Variance proportional to level
│  │   │   Exponential growth
│  │   │   Financial prices (convert to log-returns)
│  │   ├─ Requirements:
│  │   │   y_t > 0 for all t
│  │   │   If y_t ≤ 0: Shift or use alternatives
│  │   └─ Properties:
│  │       Makes multiplicative seasonality additive
│  │       Limits extreme values
│  │       Normal approximation for returns
│  ├─ Box-Cox Transformation:
│  │   ├─ Definition:
│  │   │   y_t^(λ) = (y_t^λ - 1)/λ  if λ ≠ 0
│  │   │            = ln(y_t)        if λ = 0
│  │   ├─ Special Cases:
│  │   │   λ = 1: No transformation (y_t - 1)
│  │   │   λ = 0.5: Square root
│  │   │   λ = 0: Log
│  │   │   λ = -1: Inverse
│  │   ├─ Estimation:
│  │   │   Maximum likelihood: Maximize L(λ)
│  │   │   Profile likelihood over grid of λ
│  │   │   95% CI: {λ : L(λ) > L(λ̂) - χ²_0.05(1)/2}
│  │   ├─ Interpretation:
│  │   │   λ̂ indicates optimal transformation
│  │   │   Test λ=1 (no transform needed)
│  │   │   Test λ=0 (log appropriate)
│  │   └─ Limitations:
│  │       Requires y_t > 0
│  │       Changes interpretation of coefficients
│  │       Forecasts need back-transformation (bias)
│  ├─ Detrending:
│  │   ├─ Linear Detrending:
│  │   │   ├─ Regression: y_t = α + βt + ε_t
│  │   │   ├─ Residuals: ε̂_t = y_t - (α̂ + β̂t)
│  │   │   └─ Use ε̂_t for analysis (assumed stationary)
│  │   ├─ Polynomial Detrending:
│  │   │   y_t = α + β₁t + β₂t² + ... + ε_t
│  │   │   Higher order for nonlinear trends
│  │   │   Risk: Overfitting, spurious cycles
│  │   ├─ Hodrick-Prescott (HP) Filter:
│  │   │   ├─ Objective: Decompose y_t = τ_t + c_t
│  │   │   │   τ_t: Trend, c_t: Cycle
│  │   │   ├─ Minimization:
│  │   │   │   min Σ(y_t - τ_t)² + λΣ[(τ_t+1 - τ_t) - (τ_t - τ_t-1)]²
│  │   │   │   First term: Fit, Second: Smoothness penalty
│  │   │   ├─ Smoothing Parameter λ:
│  │   │   │   λ=100 (annual), λ=1600 (quarterly), λ=14400 (monthly)
│  │   │   │   Larger λ → smoother trend
│  │   │   ├─ Solution:
│  │   │   │   τ = (I + λK'K)⁻¹ y
│  │   │   │   K: Second difference matrix
│  │   │   └─ Issues:
│  │   │       End-point bias (poor at boundaries)
│  │   │       Spurious cycles (Ravn-Uhlig critique)
│  │   │       Assumes smooth deterministic trend
│  │   ├─ Baxter-King Filter:
│  │   │   Band-pass filter (isolate business cycle frequencies)
│  │   │   Symmetric MA filter
│  │   │   Stationarity by construction
│  │   └─ Christiano-Fitzgerald Filter:
│  │       Asymmetric band-pass (better at boundaries)
│  │       Less loss of observations
│  ├─ Variance Stabilization:
│  │   ├─ Goal:
│  │   │   Achieve constant variance (homoskedasticity)
│  │   │   Required for OLS efficiency, valid inference
│  │   ├─ Diagnostics:
│  │   │   Plot residuals vs fitted
│  │   │   Breusch-Pagan test
│  │   │   White test
│  │   ├─ Transformations:
│  │   │   Square root: For Poisson-like (variance ~ mean)
│  │   │   Log: Variance ~ mean²
│  │   │   Inverse: Strong right skew
│  │   └─ Alternative:
│  │       Model heteroskedasticity directly (GARCH)
│  │       Weighted least squares
│  │       Robust standard errors
│  └─ Practical Workflow:
│      1. Plot series, check for trends/seasonality
│      2. Run ADF and KPSS (complementary)
│      3. If non-stationary:
│         a. Deterministic trend → Detrend
│         b. Stochastic trend → Difference
│         c. Both → Difference + include trend in ADF
│      4. Check variance stability (plot, test)
│      5. Transform if needed (log, Box-Cox)
│      6. Re-test stationarity on transformed series
│      7. Validate: Residuals from final model ~ white noise
├─ Power and Size of Tests:
│  ├─ Power Issues:
│  │   ADF low power vs near-unit-root alternatives
│  │   Power decreases with:
│  │     - Number of lags included
│  │     - Inclusion of trend (reduces df)
│  │     - Structural breaks (appear non-stationary)
│  │   DF-GLS improves power significantly
│  ├─ Size Distortions:
│  │   Tests over-reject H0 (too many false positives) when:
│  │     - MA errors present (ADF)
│  │     - Negative MA roots (PP worse)
│  │     - Small samples (T < 50)
│  │   Tests under-reject (miss unit roots) when:
│  │     - Structural breaks present
│  │     - High-frequency noise
│  ├─ Sample Size Requirements:
│  │   ADF: T ≥ 50 recommended, T ≥ 100 preferred
│  │   PP: Sensitive to bandwidth choice in small samples
│  │   KPSS: Better in small samples (different asymptotics)
│  │   ZA: Need T ≥ 100 (search over break points)
│  └─ Multiple Testing:
│      Running many tests inflates Type I error
│      Sequential testing preferred (cointegration)
│      Bonferroni adjustment conservative
├─ Interpretation and Decision-Making:
│  ├─ Ambiguous Results:
│  │   ADF fails to reject, KPSS fails to reject → Inconclusive
│  │   Possible near-unit-root: γ = 1 - c/T
│  │   Try DF-GLS, longer sample, structural break tests
│  ├─ Structural Breaks:
│  │   Standard tests confuse breaks with unit roots
│  │   Perron (1989): Most macro series trend-stationary with breaks
│  │   Use ZA, Perron, Lumsdaine-Papell (two breaks)
│  ├─ Economic vs Statistical:
│  │   Stationarity over sample ≠ true DGP
│  │   Long-run: Many processes non-stationary
│  │   Short-run: May approximate as stationary
│  │   Forecasting: Differences often better even if stationary
│  └─ Reporting:
│      Report test statistic, p-value, lags used
│      Report multiple tests (ADF + KPSS)
│      Discuss economic rationale (shocks permanent?)
│      Show plots (visual inspection critical)
├─ Special Cases:
│  ├─ Near-Unit-Root Processes:
│  │   γ = 1 - c/T (local-to-unity)
│  │   Behaves non-stationary in finite samples
│  │   Asymptotically stationary
│  │   Standard tests low power
│  ├─ Explosive Processes:
│  │   γ > 1 (explosive root)
│  │   Right-tail unit root test (Phillips et al. 2011)
│  │   Asset bubbles, diverging series
│  ├─ Panel Unit Root Tests:
│  │   Multiple time series (cross-section + time)
│  │   Im-Pesaran-Shin (IPS)
│  │   Levin-Lin-Chu
│  │   Greater power (pool information)
│  └─ Nonlinear Unit Root Tests:
│  │   Standard tests assume linear adjustment
│  │   Threshold cointegration (TAR, ESTAR)
│  │   Enders-Granger (nonlinear ADF)
│  │   Kapetanios-Shin-Snell (KSS)
└─ Software Implementation:
   ├─ Python:
   │   statsmodels.tsa.stattools: adfuller, kpss
   │   arch.unitroot: ADF, PP, KPSS, DF-GLS, variance ratio, ZA
   │   scipy.signal: detrend (linear)
   │   statsmodels.tsa.filters: hp_filter, cf_filter, bk_filter
   ├─ R:
   │   tseries: adf.test, kpss.test, pp.test
   │   urca: ur.df, ur.pp, ur.kpss, ur.ers (DF-GLS), ur.za
   │   fUnitRoots: unitrootTest (comprehensive)
   │   mFilter: hpfilter
   └─ Interpretation Notes:
       Always check documentation for exact implementation
       Critical values may differ slightly across packages
       Default lag selection varies (BIC vs AIC)
       Some tests require data preprocessing
```

**Interaction:** Plot series → Run ADF (H0: unit root) + KPSS (H0: stationary) → If both indicate non-stationary, apply transformation (difference, detrend, log) → Re-test → Validate residuals → Proceed to modeling.

## 5. Mini-Project
Comprehensive stationarity diagnostics and transformations:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("STATIONARITY TESTING AND TRANSFORMATIONS")
print("="*80)

class StationarityTester:
    """Comprehensive stationarity testing suite"""
    
    def __init__(self):
        self.critical_values_adf = {
            0.01: -3.43, 0.05: -2.86, 0.10: -2.57
        }
    
    def adf_test(self, y, regression='c', maxlag=None):
        """
        Augmented Dickey-Fuller test
        regression: 'n' (none), 'c' (constant), 'ct' (constant+trend)
        """
        result = adfuller(y, regression=regression, maxlag=maxlag, autolag='AIC')
        
        return {
            'adf_stat': result[0],
            'p_value': result[1],
            'lags': result[2],
            'nobs': result[3],
            'critical_values': result[4],
            'stationary': result[1] < 0.05
        }
    
    def kpss_test(self, y, regression='c', nlags='auto'):
        """
        KPSS test: H0 is stationarity (opposite of ADF!)
        regression: 'c' (level), 'ct' (trend)
        """
        result = kpss(y, regression=regression, nlags=nlags)
        
        return {
            'kpss_stat': result[0],
            'p_value': result[1],
            'lags': result[2],
            'critical_values': result[3],
            'stationary': result[1] > 0.05  # Fail to reject H0
        }
    
    def phillips_perron(self, y, regression='c', lags=None):
        """
        Phillips-Perron test
        Non-parametric correction for serial correlation
        """
        n = len(y)
        
        # OLS regression
        if regression == 'c':
            X = np.ones((n, 1))
        elif regression == 'ct':
            X = np.column_stack([np.ones(n), np.arange(n)])
        else:
            X = None
        
        y_lag = y[:-1]
        dy = np.diff(y)
        
        if X is not None:
            X_reg = np.column_stack([X[1:], y_lag])
        else:
            X_reg = y_lag.reshape(-1, 1)
        
        model = OLS(dy, X_reg).fit()
        
        if regression == 'c':
            rho_idx = 1
        elif regression == 'ct':
            rho_idx = 2
        else:
            rho_idx = 0
        
        rho = model.params[rho_idx]
        se_rho = model.bse[rho_idx]
        
        # Long-run variance (Newey-West)
        if lags is None:
            lags = int(4 * (n/100)**(2/9))
        
        residuals = model.resid
        gamma0 = np.var(residuals)
        gamma_sum = 0
        
        for lag in range(1, lags+1):
            weight = 1 - lag / (lags + 1)  # Bartlett kernel
            gamma_lag = np.cov(residuals[:-lag], residuals[lag:])[0, 1]
            gamma_sum += 2 * weight * gamma_lag
        
        sigma2_lr = gamma0 + gamma_sum
        
        # PP adjustment
        lambda_hat = 0.5 * (sigma2_lr - gamma0)
        
        # Corrected test statistic
        se_correction = np.sqrt(sigma2_lr / gamma0)
        pp_stat = rho / (se_rho * se_correction) - (n * se_rho * lambda_hat) / (gamma0 * se_correction)
        
        # Use ADF critical values (approximately)
        p_value = None  # Would need to interpolate from tables
        stationary = pp_stat < -2.86  # Approximate 5% critical value
        
        return {
            'pp_stat': pp_stat,
            'p_value': p_value,
            'lags': lags,
            'stationary': stationary
        }
    
    def variance_ratio_test(self, y, lags=[2, 4, 8, 16]):
        """
        Variance ratio test (Lo-MacKinlay)
        H0: Random walk (VR=1)
        """
        n = len(y)
        returns = np.diff(y)
        
        mu = np.mean(returns)
        var1 = np.sum((returns - mu)**2) / (n - 1)
        
        results = []
        for q in lags:
            # q-period returns
            returns_q = np.array([np.sum(returns[i:i+q]) for i in range(0, n-q, q)])
            m = len(returns_q)
            
            varq = np.sum((returns_q - q*mu)**2) / m
            
            # Variance ratio
            vr = varq / (q * var1)
            
            # Asymptotic variance under H0 (i.i.d.)
            theta = 2 * (2*q - 1) * (q - 1) / (3 * q)
            asy_var_iid = theta / (n - q + 1)
            
            # Test statistic
            z_stat = (vr - 1) / np.sqrt(asy_var_iid)
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
            
            results.append({
                'lag': q,
                'VR': vr,
                'z_stat': z_stat,
                'p_value': p_value,
                'reject_rw': p_value < 0.05
            })
        
        return results
    
    def zivot_andrews(self, y, model='C', trim=0.15):
        """
        Zivot-Andrews unit root test with one structural break
        model: 'A' (level), 'B' (trend), 'C' (both)
        """
        n = len(y)
        trim_n = int(n * trim)
        
        break_points = range(trim_n, n - trim_n)
        min_adf = np.inf
        opt_break = None
        
        for tb in break_points:
            # Create break dummies
            DU = np.zeros(n)
            DU[tb:] = 1
            
            DT = np.zeros(n)
            DT[tb:] = np.arange(n - tb)
            
            # Construct regression
            t_trend = np.arange(n)
            y_lag = y[:-1]
            dy = np.diff(y)
            
            if model == 'A':
                X = np.column_stack([np.ones(n-1), t_trend[1:], DU[1:], y_lag])
            elif model == 'B':
                X = np.column_stack([np.ones(n-1), t_trend[1:], DT[1:], y_lag])
            else:  # 'C'
                X = np.column_stack([np.ones(n-1), t_trend[1:], DU[1:], DT[1:], y_lag])
            
            # Add lagged differences (simplified: use 1 lag)
            dy_lag = np.diff(y, n=1)[:-1]
            X_full = np.column_stack([X[:-1], dy_lag])
            y_reg = dy[1:]
            
            try:
                model_fit = OLS(y_reg, X_full).fit()
                adf_stat = model_fit.tvalues[-2]  # Coefficient on y_lag
                
                if adf_stat < min_adf:
                    min_adf = adf_stat
                    opt_break = tb
            except:
                continue
        
        # ZA critical values more negative than standard ADF
        # 5% critical value approximately -4.8 for model C
        critical_value = -4.8 if model == 'C' else -4.5
        
        return {
            'za_stat': min_adf,
            'break_point': opt_break,
            'critical_value': critical_value,
            'stationary': min_adf < critical_value
        }
    
    def comprehensive_test(self, y, name='Series'):
        """Run all tests and summarize"""
        print(f"\n{'='*60}")
        print(f"STATIONARITY TESTS: {name}")
        print(f"{'='*60}")
        
        # ADF
        adf = self.adf_test(y, regression='c')
        print(f"\nAugmented Dickey-Fuller (H0: Unit Root):")
        print(f"  Statistic: {adf['adf_stat']:.4f}")
        print(f"  p-value: {adf['p_value']:.4f}")
        print(f"  Lags: {adf['lags']}")
        print(f"  Conclusion: {'Stationary' if adf['stationary'] else 'Non-stationary (unit root)'}")
        
        # KPSS
        kpss_result = self.kpss_test(y, regression='c')
        print(f"\nKPSS (H0: Stationary):")
        print(f"  Statistic: {kpss_result['kpss_stat']:.4f}")
        print(f"  p-value: {kpss_result['p_value']:.4f}")
        print(f"  Conclusion: {'Stationary' if kpss_result['stationary'] else 'Non-stationary'}")
        
        # Phillips-Perron
        pp = self.phillips_perron(y, regression='c')
        print(f"\nPhillips-Perron (H0: Unit Root):")
        print(f"  Statistic: {pp['pp_stat']:.4f}")
        print(f"  Lags: {pp['lags']}")
        print(f"  Conclusion: {'Stationary' if pp['stationary'] else 'Non-stationary'}")
        
        # Combined interpretation
        print(f"\n{'='*60}")
        if adf['stationary'] and kpss_result['stationary']:
            print("CONSENSUS: Series is STATIONARY")
        elif not adf['stationary'] and not kpss_result['stationary']:
            print("CONSENSUS: Series is NON-STATIONARY (unit root)")
        else:
            print("AMBIGUOUS: Tests disagree - possible near-unit-root or structural break")
        
        return {
            'adf': adf,
            'kpss': kpss_result,
            'pp': pp
        }

class Transformations:
    """Transformations to achieve stationarity"""
    
    def __init__(self):
        pass
    
    def difference(self, y, order=1, seasonal=False, period=12):
        """
        Differencing transformation
        order: Number of times to difference
        seasonal: Apply seasonal differencing
        """
        result = y.copy()
        
        for _ in range(order):
            result = np.diff(result)
        
        if seasonal:
            if len(result) > period:
                result = result[period:] - result[:-period]
        
        return result
    
    def log_transform(self, y, shift=0):
        """
        Log transformation
        shift: Add constant if y has zeros or negatives
        """
        y_shifted = y + shift
        
        if np.any(y_shifted <= 0):
            raise ValueError("Cannot take log of non-positive values")
        
        return np.log(y_shifted)
    
    def box_cox(self, y, lambda_=None):
        """
        Box-Cox transformation
        If lambda_ is None, estimate it via MLE
        """
        if np.any(y <= 0):
            raise ValueError("Box-Cox requires positive data")
        
        if lambda_ is None:
            # Estimate lambda via profile likelihood
            lambda_ = self.estimate_box_cox_lambda(y)
        
        if abs(lambda_) < 1e-10:
            return np.log(y), lambda_
        else:
            return (y**lambda_ - 1) / lambda_, lambda_
    
    def estimate_box_cox_lambda(self, y, lambda_range=(-2, 2)):
        """Estimate Box-Cox lambda via maximum likelihood"""
        
        def neg_log_likelihood(lambda_, y):
            """Negative log-likelihood for Box-Cox"""
            n = len(y)
            
            if abs(lambda_) < 1e-10:
                y_trans = np.log(y)
            else:
                y_trans = (y**lambda_ - 1) / lambda_
            
            # Add Jacobian term
            jacobian = (lambda_ - 1) * np.sum(np.log(y))
            
            # Variance of transformed series
            sigma2 = np.var(y_trans, ddof=1)
            
            # Log-likelihood
            ll = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sigma2) - 0.5 * n + jacobian
            
            return -ll
        
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=lambda_range,
            args=(y,),
            method='bounded'
        )
        
        return result.x
    
    def detrend_linear(self, y):
        """Remove linear trend via OLS"""
        t = np.arange(len(y))
        X = np.column_stack([np.ones(len(y)), t])
        model = OLS(y, X).fit()
        
        trend = model.predict(X)
        detrended = y - trend
        
        return detrended, model.params
    
    def hp_filter(self, y, lamb=1600):
        """
        Hodrick-Prescott filter
        lamb: Smoothing parameter (1600 for quarterly, 100 for annual)
        """
        n = len(y)
        
        # Construct second-difference matrix K
        # (1, -2, 1) pattern for second differences
        K = np.zeros((n-2, n))
        for i in range(n-2):
            K[i, i:i+3] = [1, -2, 1]
        
        # Solve (I + λK'K)τ = y
        I = np.eye(n)
        A = I + lamb * K.T @ K
        trend = np.linalg.solve(A, y)
        cycle = y - trend
        
        return trend, cycle
    
    def hp_filter_sparse(self, y, lamb=1600):
        """HP filter using sparse matrices (faster for large n)"""
        n = len(y)
        
        # Second difference matrix as sparse
        data = np.array([
            np.ones(n-2),
            -2 * np.ones(n-2),
            np.ones(n-2)
        ])
        offsets = [0, 1, 2]
        K = diags(data, offsets, shape=(n-2, n), format='csr')
        
        # Solve sparse system
        I = diags([1], [0], shape=(n, n), format='csr')
        A = I + lamb * (K.T @ K)
        trend = spsolve(A, y)
        cycle = y - trend
        
        return trend, cycle

# Generate test series
n = 500

# 1. White noise (stationary)
white_noise = np.random.normal(0, 1, n)

# 2. Random walk (unit root)
random_walk = np.cumsum(white_noise)

# 3. Random walk with drift
drift = 0.05
rw_drift = np.cumsum(white_noise + drift)

# 4. Trend-stationary
t = np.arange(n)
trend_stationary = 0.05 * t + white_noise

# 5. AR(1) with high persistence
ar1_high = np.zeros(n)
phi = 0.95
for i in range(1, n):
    ar1_high[i] = phi * ar1_high[i-1] + white_noise[i]

# Test all series
tester = StationarityTester()

print("\n" + "="*80)
print("SCENARIO 1: WHITE NOISE (Stationary Benchmark)")
print("="*80)
results_wn = tester.comprehensive_test(white_noise, "White Noise")

print("\n" + "="*80)
print("SCENARIO 2: RANDOM WALK (Unit Root)")
print("="*80)
results_rw = tester.comprehensive_test(random_walk, "Random Walk")

# Test differenced random walk
print("\n" + "="*80)
print("SCENARIO 2b: RANDOM WALK - FIRST DIFFERENCE")
print("="*80)
transformer = Transformations()
rw_diff = transformer.difference(random_walk, order=1)
results_rw_diff = tester.comprehensive_test(rw_diff, "Differenced Random Walk")

print("\n" + "="*80)
print("SCENARIO 3: TREND-STATIONARY PROCESS")
print("="*80)
results_ts = tester.comprehensive_test(trend_stationary, "Trend-Stationary")

# Detrend
trend_detrended, trend_params = transformer.detrend_linear(trend_stationary)
print(f"\nLinear detrending:")
print(f"  Slope: {trend_params[1]:.6f} (True: 0.05)")
print(f"  Intercept: {trend_params[0]:.4f}")

print("\n" + "="*80)
print("SCENARIO 3b: AFTER DETRENDING")
print("="*80)
results_detrended = tester.comprehensive_test(trend_detrended, "Detrended Series")

print("\n" + "="*80)
print("SCENARIO 4: AR(1) WITH HIGH PERSISTENCE (Near Unit Root)")
print("="*80)
results_ar1 = tester.comprehensive_test(ar1_high, f"AR(1) φ={phi}")

print("\n" + "="*80)
print("SCENARIO 5: VARIANCE RATIO TESTS")
print("="*80)

processes = {
    'White Noise': white_noise,
    'Random Walk': random_walk,
    'AR(1) φ=0.95': ar1_high
}

for name, series in processes.items():
    print(f"\n{name}:")
    vr_results = tester.variance_ratio_test(series, lags=[2, 4, 8, 16])
    print(f"  {'Lag':<8} {'VR':<10} {'Z-stat':<10} {'p-value':<10} {'Reject RW':<12}")
    print("  " + "-"*50)
    for vr in vr_results:
        print(f"  {vr['lag']:<8} {vr['VR']:<10.3f} {vr['z_stat']:<10.3f} {vr['p_value']:<10.4f} {str(vr['reject_rw']):<12}")

# Box-Cox transformation example
print("\n" + "="*80)
print("SCENARIO 6: BOX-COX TRANSFORMATION")
print("="*80)

# Generate exponentially growing series (needs variance stabilization)
exp_series = np.exp(0.01 * t + 0.1 * white_noise)

print(f"Original series statistics:")
print(f"  Mean: {np.mean(exp_series):.2f}")
print(f"  Std: {np.std(exp_series):.2f}")
print(f"  CV: {np.std(exp_series)/np.mean(exp_series):.4f}")

# Estimate Box-Cox lambda
lambda_hat = transformer.estimate_box_cox_lambda(exp_series)
print(f"\nEstimated λ: {lambda_hat:.4f}")
print(f"  λ=0 suggests log transformation")
print(f"  λ=0.5 suggests square root")
print(f"  λ=1 suggests no transformation")

# Apply transformation
exp_transformed, _ = transformer.box_cox(exp_series, lambda_=lambda_hat)
print(f"\nTransformed series statistics:")
print(f"  Mean: {np.mean(exp_transformed):.2f}")
print(f"  Std: {np.std(exp_transformed):.2f}")
print(f"  CV: {np.std(exp_transformed)/np.mean(exp_transformed):.4f} (more stable)")

# Hodrick-Prescott filter
print("\n" + "="*80)
print("SCENARIO 7: HODRICK-PRESCOTT FILTER")
print("="*80)

# Generate series with trend and cycle
cycle_component = 5 * np.sin(2 * np.pi * t / 40)
hp_series = 0.05 * t + cycle_component + 0.5 * white_noise

trend_hp, cycle_hp = transformer.hp_filter(hp_series, lamb=1600)

print(f"HP Filter (λ=1600):")
print(f"  Original variance: {np.var(hp_series):.2f}")
print(f"  Trend variance: {np.var(trend_hp):.2f}")
print(f"  Cycle variance: {np.var(cycle_hp):.2f}")
print(f"  Decomposition: Var(y) ≈ Var(trend) + Var(cycle) + 2·Cov")

# Structural break test (Zivot-Andrews)
print("\n" + "="*80)
print("SCENARIO 8: STRUCTURAL BREAK (Zivot-Andrews)")
print("="*80)

# Generate series with level shift at t=250
break_series = white_noise.copy()
break_series[250:] += 3  # Level shift
break_series_cumsum = np.cumsum(break_series)

print("Series with structural break:")
za_result = tester.zivot_andrews(break_series_cumsum, model='C', trim=0.15)
print(f"  ZA statistic: {za_result['za_stat']:.4f}")
print(f"  Critical value (5%): {za_result['critical_value']:.4f}")
print(f"  Break point detected: {za_result['break_point']}")
print(f"  True break point: 250")
print(f"  Stationary with break: {za_result['stationary']}")

# Standard ADF without break
print("\nStandard ADF (ignoring break):")
adf_no_break = tester.adf_test(break_series_cumsum)
print(f"  ADF p-value: {adf_no_break['p_value']:.4f}")
print(f"  Incorrectly concludes: {'Stationary' if adf_no_break['stationary'] else 'Non-stationary'}")

# Visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot 1: Random walk levels
ax = axes[0, 0]
ax.plot(random_walk, linewidth=1, alpha=0.7, label='Levels')
ax.set_title('Random Walk (Non-Stationary)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Random walk first difference
ax = axes[0, 1]
ax.plot(rw_diff, linewidth=0.8, alpha=0.7, label='First Difference')
ax.axhline(0, color='r', linestyle='--', alpha=0.5)
ax.set_title('Random Walk - First Difference (Stationary)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: ACF of levels vs differences
ax = axes[0, 2]
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(random_walk, lags=40, ax=ax, alpha=0.05, label='Levels')
ax.set_title('ACF: Random Walk Levels (Slow Decay)')
ax.grid(alpha=0.3)

# Plot 4: Trend-stationary
ax = axes[1, 0]
ax.plot(t, trend_stationary, 'b-', linewidth=1, alpha=0.7, label='Observed')
ax.plot(t, 0.05*t, 'r--', linewidth=2, label='True Trend')
ax.set_title('Trend-Stationary Process')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Detrended
ax = axes[1, 1]
ax.plot(trend_detrended, linewidth=0.8, alpha=0.7)
ax.axhline(0, color='r', linestyle='--', alpha=0.5)
ax.set_title('After Detrending (Stationary)')
ax.set_xlabel('Time')
ax.set_ylabel('Detrended Value')
ax.grid(alpha=0.3)

# Plot 6: AR(1) high persistence
ax = axes[1, 2]
ax.plot(ar1_high, linewidth=1, alpha=0.7)
ax.set_title(f'AR(1) φ={phi} (Near Unit Root)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.grid(alpha=0.3)

# Plot 7: Box-Cox original vs transformed
ax = axes[2, 0]
ax2 = ax.twinx()
ax.plot(exp_series, 'b-', linewidth=1, alpha=0.7, label='Original')
ax2.plot(exp_transformed, 'r-', linewidth=1, alpha=0.7, label='Transformed')
ax.set_xlabel('Time')
ax.set_ylabel('Original', color='b')
ax2.set_ylabel('Transformed', color='r')
ax.set_title(f'Box-Cox Transform (λ={lambda_hat:.3f})')
ax.grid(alpha=0.3)

# Plot 8: HP filter decomposition
ax = axes[2, 1]
ax.plot(hp_series, 'k-', linewidth=1, alpha=0.5, label='Original')
ax.plot(trend_hp, 'r-', linewidth=2, label='Trend')
ax.plot(cycle_hp, 'b-', linewidth=1, alpha=0.7, label='Cycle')
ax.set_title('HP Filter Decomposition')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 9: Structural break
ax = axes[2, 2]
ax.plot(break_series_cumsum, linewidth=1, alpha=0.7)
ax.axvline(250, color='r', linestyle='--', linewidth=2, label='True Break')
if za_result['break_point']:
    ax.axvline(za_result['break_point'], color='g', linestyle=':', linewidth=2, label='Detected Break')
ax.set_title('Structural Break Detection')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary table
print("\n" + "="*80)
print("SUMMARY: STATIONARITY TEST RESULTS")
print("="*80)

summary_data = {
    'White Noise': results_wn,
    'Random Walk': results_rw,
    'RW Differenced': results_rw_diff,
    'Trend-Stationary': results_ts,
    'TS Detrended': results_detrended,
    f'AR(1) φ={phi}': results_ar1
}

print(f"\n{'Process':<20} {'ADF p-val':<12} {'KPSS p-val':<12} {'Consensus':<20}")
print("-" * 64)

for name, results in summary_data.items():
    adf_p = results['adf']['p_value']
    kpss_p = results['kpss']['p_value']
    
    if results['adf']['stationary'] and results['kpss']['stationary']:
        consensus = "Stationary"
    elif not results['adf']['stationary'] and not results['kpss']['stationary']:
        consensus = "Non-stationary"
    else:
        consensus = "Ambiguous"
    
    print(f"{name:<20} {adf_p:<12.4f} {kpss_p:<12.4f} {consensus:<20}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("1. Random walk: Both ADF and KPSS detect non-stationarity")
print("2. First difference of RW: Restores stationarity")
print("3. Trend-stationary: ADF may fail (trend confounds test)")
print("4. After detrending: Clear stationarity")
print("5. High AR(1): Near unit root → low power, ambiguous")
print("6. Box-Cox: Stabilizes variance in exponential growth")
print("7. HP filter: Separates trend and cycle components")
print("8. Structural breaks: Standard tests misleading, use ZA")
```

## 6. Challenge Round
1. **Power Analysis:** Simulate AR(1) with φ varying from 0.8 to 1.0 in steps of 0.02. For each φ, generate 1000 series (T=100). Compute ADF rejection rate at 5% level. Plot power curve. How close to 1 before power drops below 50%?

2. **Over-Differencing:** Generate stationary AR(1) φ=0.6. Difference it. Estimate ARMA model on differenced series—what order selected? Compare AIC of AR(1) on levels vs ARMA(p,q) on differences. Which better?

3. **Seasonal Unit Root:** Simulate quarterly series: (1-L^4)y_t = ε_t. Apply standard ADF—does it reject? Apply seasonal differencing, then ADF on (1-L^4)y_t. Now stationary? Implement simplified HEGY test.

4. **Break vs Unit Root:** Generate trend-stationary series with break at t=T/2 (shift mean by 2σ). Run ADF without break—p-value? Run Zivot-Andrews allowing break—does it find it? Compare critical values.

5. **Box-Cox Sensitivity:** Generate GARCH(1,1) series (time-varying volatility). Estimate λ via MLE. Apply transformation. Does it fully stabilize variance? Test residuals for ARCH effects (McLeod-Li test). Still present?

## 7. Key References
- [Dickey & Fuller, "Distribution of Estimators for AR Time Series with Unit Root" (1979)](https://www.jstor.org/stable/1912517) - foundational unit root test methodology
- [Phillips & Perron, "Testing for Unit Root in Time Series Regression" (1988)](https://academic.oup.com/biomet/article/75/2/335/323343) - non-parametric unit root test robust to heteroskedasticity
- [Kwiatkowski et al., "Testing Null of Stationarity Against Alternative of Unit Root" (1992)](https://www.sciencedirect.com/science/article/abs/pii/030440769290104Y) - KPSS test complementing ADF/PP

---
**Status:** Essential pre-modeling diagnostic | **Complements:** Time Series Fundamentals, ARIMA Models, Cointegration, Spurious Regression, Forecasting
