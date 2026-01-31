# GARCH Models and Conditional Heteroscedasticity

## 1. Concept Skeleton
**Definition:** Generalized Autoregressive Conditional Heteroscedasticity (GARCH) models time-varying volatility where variance depends on past squared residuals and past conditional variances  
**Purpose:** Capture volatility clustering (large changes followed by large changes), model fat tails in financial returns, generate dynamic volatility forecasts for risk management  
**Prerequisites:** ARMA models, maximum likelihood estimation, stationarity conditions, kurtosis/skewness, volatility concepts, financial time series properties

## 2. Comparative Framing
| Model | ARCH(q) | GARCH(p,q) | EGARCH(p,q) | GJR-GARCH(p,q) | IGARCH(1,1) |
|-------|---------|------------|-------------|----------------|-------------|
| **Variance Equation** | σ²ₜ = ω + Σα_i ε²ₜ₋ᵢ | σ²ₜ = ω + Σα_i ε²ₜ₋ᵢ + Σβ_j σ²ₜ₋ⱼ | log(σ²ₜ) = ω + α|εₜ₋₁/σₜ₋₁| + γεₜ₋₁/σₜ₋₁ + β log(σ²ₜ₋₁) | σ²ₜ = ω + α ε²ₜ₋₁ + γI_{εₜ₋₁<0} ε²ₜ₋₁ + β σ²ₜ₋₁ | σ²ₜ = ω + α ε²ₜ₋₁ + (1-α) σ²ₜ₋₁ |
| **Asymmetry** | No (symmetric) | No | Yes (leverage effect) | Yes (threshold) | No |
| **Stationarity** | Requires Σα_i < 1 | Requires Σα_i + Σβ_j < 1 | Always stationary (log form) | Requires α + γ/2 + β < 1 | Unit root (non-stationary) |
| **Parameters** | q + 1 (many if q large) | p + q + 1 (parsimonious) | 3 (efficient) | 4 (leverage + persistence) | 2 (persistence = 1) |
| **Persistence** | Low (short memory) | High (long memory) | Flexible | High (asymmetric) | Infinite (shocks permanent) |
| **Use Case** | Rarely used (GARCH better) | Standard volatility model | Leverage effects (equities) | Downside risk focus | Volatility trending |

| Volatility Phenomenon | Description | GARCH Property | Real-World Example |
|-----------------------|-------------|----------------|-------------------|
| **Volatility Clustering** | High vol follows high vol, low follows low | ARCH/GARCH effects (α, β > 0) | 2008 crisis: sustained high volatility |
| **Leverage Effect** | Negative returns → higher volatility | Asymmetric GARCH (γ ≠ 0) | Stock market crashes → vol spikes |
| **Fat Tails (Leptokurtosis)** | More extreme events than normal | Conditional t-distribution | Black Monday 1987, COVID crash |
| **Volatility Mean Reversion** | Vol returns to long-run average | Stationarity (α + β < 1) | VIX spikes, then decays |
| **Volatility Persistence** | Shocks decay slowly | High β (close to 1) | Post-crisis elevated vol for months |

## 3. Examples + Counterexamples

**Simple Example:**  
S&P 500 daily returns: GARCH(1,1) with α=0.08, β=0.90. Persistence = 0.98 (near unit root). Shock increases volatility, decays slowly over ~50 days. Captures 2008 crisis clustering.

**Perfect Fit:**  
FX rates (EUR/USD): Strong volatility clustering, no asymmetry. GARCH(1,1) captures dynamics well. Out-of-sample VaR forecasts accurate at 95% and 99% confidence levels.

**Failure Case:**  
Intraday tick data: Market microstructure noise, bid-ask bounce, ultra-high frequency. GARCH assumes regular spacing, misspecified. Use realized volatility or high-frequency GARCH variants instead.

**Edge Case:**  
Cryptocurrency (Bitcoin): Extreme volatility, frequent regime changes (bull/bear), non-Gaussian even conditionally. GARCH(1,1) with t-distribution helps, but regime-switching GARCH better captures structural breaks.

**Common Mistake:**  
Estimating GARCH on non-stationary mean (trending series). GARCH models conditional variance of stationary mean-adjusted series. Must first model mean (ARMA) or ensure returns stationary.

**Counterexample:**  
Bond yields: Low volatility, occasional jumps (policy announcements). GARCH overpredicts volatility during calm periods, underpredicts during jumps. Jump-diffusion or regime-switching models preferred.

## 4. Layer Breakdown
```
GARCH Framework and Extensions:

├─ ARCH(q) - Foundation (Engle 1982):
│   ├─ Model Structure:
│   │   │ Mean Equation: rₜ = μ + εₜ
│   │   │ Variance Equation: σ²ₜ = ω + α₁ε²ₜ₋₁ + α₂ε²ₜ₋₂ + ... + αqε²ₜ₋q
│   │   │ Error: εₜ = σₜ zₜ, where zₜ ~ N(0,1) or t_ν
│   │   │
│   │   │ Interpretation:
│   │   │   - ω: Baseline volatility (constant term)
│   │   │   - αᵢ: Impact of past squared shocks on current volatility
│   │   │   - Large past shock (|εₜ₋₁|) → high current volatility σ²ₜ
│   │   │
│   │   └─ Example: ARCH(1) with α₁ = 0.5
│   │       If εₜ₋₁ = 2σ → σ²ₜ = ω + 0.5×(4σ²) = ω + 2σ²
│   │       Volatility doubles from shock
│   ├─ Stationarity Condition:
│   │   │ Requires: α₁ + α₂ + ... + αq < 1
│   │   │ Ensures variance doesn't explode
│   │   │ Unconditional variance: E[ε²ₜ] = ω / (1 - Σαᵢ)
│   │   └─ If Σαᵢ = 1: Integrated ARCH (non-stationary, unit root)
│   ├─ Drawbacks:
│   │   │ Requires many lags (large q) to capture persistence
│   │   │ Example: q = 20-30 common for daily financial data
│   │   │ Many parameters → estimation uncertainty
│   │   └─ GARCH solves with parsimony (fewer parameters)
│   └─ Identification:
│       Plot squared residuals from ARMA model
│       ACF of ε²ₜ shows significant lags → ARCH effects present
│       Ljung-Box test on ε²ₜ: Reject H0 → heteroscedasticity
│
├─ GARCH(p,q) - Generalized ARCH (Bollerslev 1986):
│   ├─ Model Structure:
│   │   │ Mean: rₜ = μ + εₜ (or ARMA specification)
│   │   │ Variance: σ²ₜ = ω + Σᵢ₌₁ᵖ αᵢε²ₜ₋ᵢ + Σⱼ₌₁ᵍ βⱼσ²ₜ₋ⱼ
│   │   │
│   │   │ Components:
│   │   │   - ω: Long-run baseline volatility
│   │   │   - αᵢ: ARCH terms (news/shock impact)
│   │   │   - βⱼ: GARCH terms (past variance persistence)
│   │   │
│   │   │ Typical: GARCH(1,1) sufficient (p=1, q=1)
│   │   │   σ²ₜ = ω + α ε²ₜ₋₁ + β σ²ₜ₋₁
│   │   │
│   │   └─ Intuition:
│   │       α: "News" (recent shock) impact
│   │       β: "Old volatility" persistence
│   │       α + β: Total persistence (near 1 → long memory)
│   ├─ GARCH(1,1) Properties:
│   │   ├─ Unconditional Variance:
│   │   │   E[ε²ₜ] = ω / (1 - α - β)
│   │   │   Requires α + β < 1 for stationarity
│   │   ├─ Volatility Persistence:
│   │   │   Half-life: log(0.5) / log(α + β)
│   │   │   Example: α + β = 0.98 → half-life ≈ 35 days
│   │   ├─ Kurtosis (Fat Tails):
│   │   │   Excess kurtosis = 6α² / [1 - (α+β)² - 2α²]
│   │   │   GARCH generates fat tails even with normal innovations
│   │   │   Example: α=0.1, β=0.85 → kurtosis ≈ 6 (vs 0 for normal)
│   │   └─ Volatility Clustering:
│   │       Large |εₜ| → large σ²ₜ₊₁ → likely large |εₜ₊₁|
│   │       Autocorrelation in ε²ₜ captured by α, β
│   ├─ Estimation (Maximum Likelihood):
│   │   ├─ Likelihood Function:
│   │   │   │ Assume εₜ|Fₜ₋₁ ~ N(0, σ²ₜ)
│   │   │   │ Log-likelihood: ℓ(θ) = -1/2 Σₜ [log(2π) + log(σ²ₜ) + ε²ₜ/σ²ₜ]
│   │   │   │ Maximize w.r.t. θ = (ω, α, β)
│   │   │   └─ Numerically optimize (BFGS, Newton-Raphson)
│   │   ├─ Constraints:
│   │   │   ω > 0, α ≥ 0, β ≥ 0 (non-negativity)
│   │   │   α + β < 1 (stationarity)
│   │   │   Software (arch, rugarch) enforces during estimation
│   │   ├─ Standard Errors:
│   │   │   Asymptotic SE from Hessian matrix
│   │   │   Bollerslev-Wooldridge robust SE for misspecification
│   │   └─ Convergence:
│   │       Sensitive to starting values
│   │       Use ARCH(q) estimates as initial guess
│   ├─ Model Selection:
│   │   │ AIC, BIC: Lower preferred
│   │   │ Ljung-Box on standardized residuals: z²ₜ = ε²ₜ/σ²ₜ
│   │   │ Should be i.i.d. (no remaining autocorrelation)
│   │   └─ Typically GARCH(1,1) sufficient; rarely need p,q > 2
│   └─ Forecasting:
│       ├─ h-Step Ahead Volatility:
│       │   │ GARCH(1,1): σ²ₜ₊ₕ|ₜ = unconditional variance + (α+β)ʰ [σ²ₜ - unconditional]
│       │   │ Exponential decay toward long-run average
│       │   │ Example: σ²ₜ = 2%, unconditional = 1%, α+β=0.95
│       │   │          σ²ₜ₊₁₀ = 1% + 0.95¹⁰ × 1% ≈ 1.6%
│       │   └─ Long horizon: Converges to √(ω/(1-α-β))
│       └─ VaR (Value at Risk):
│           1-day 95% VaR = -1.645 × σₜ₊₁ (for long position)
│           Update daily as volatility forecast changes
│
├─ Asymmetric GARCH Models (Leverage Effect):
│   ├─ GJR-GARCH (Glosten-Jagannathan-Runkle 1993):
│   │   │ σ²ₜ = ω + α ε²ₜ₋₁ + γ Iₜ₋₁ ε²ₜ₋₁ + β σ²ₜ₋₁
│   │   │ Iₜ₋₁ = 1 if εₜ₋₁ < 0 (negative return), else 0
│   │   │
│   │   │ Total impact of negative shock: (α + γ)
│   │   │ Total impact of positive shock: α
│   │   │ Leverage effect: γ > 0 → bad news increases vol more
│   │   │
│   │   │ Example: α=0.05, γ=0.08, β=0.88
│   │   │   Negative shock: (0.05 + 0.08) = 0.13 impact
│   │   │   Positive shock: 0.05 impact
│   │   │   Asymmetry ratio: 0.13/0.05 = 2.6×
│   │   │
│   │   ├─ Stationarity:
│   │   │   α + β + γ/2 < 1 (expected value of Iₜ = 0.5)
│   │   ├─ Use Cases:
│   │   │   Equity markets: Crashes → vol spikes asymmetrically
│   │   │   VIX modeling: Negative S&P returns → VIX surge
│   │   └─ Estimation:
│   │       MLE with asymmetric term
│   │       Test: H0: γ = 0 (symmetric GARCH)
│   │       Typically γ > 0 and significant for equities
│   ├─ EGARCH (Exponential GARCH, Nelson 1991):
│   │   │ log(σ²ₜ) = ω + α |zₜ₋₁| + γ zₜ₋₁ + β log(σ²ₜ₋₁)
│   │   │ zₜ = εₜ/σₜ (standardized residual)
│   │   │
│   │   │ Advantages:
│   │   │   - Log form: σ²ₜ always positive (no constraints on α, γ, β)
│   │   │   - Asymmetry via γ: γ < 0 → leverage effect
│   │   │   - |zₜ₋₁| captures magnitude, zₜ₋₁ captures sign
│   │   │
│   │   │ Interpretation:
│   │   │   If zₜ₋₁ = -2 (large negative shock):
│   │   │     Contribution = α×2 + γ×(-2) = 2α - 2γ
│   │   │   If zₜ₋₁ = +2 (large positive shock):
│   │   │     Contribution = α×2 + γ×2 = 2α + 2γ
│   │   │   Asymmetry: Negative has extra -2γ impact (if γ<0)
│   │   │
│   │   ├─ Stationarity:
│   │   │   Always stationary (log form doesn't explode)
│   │   │   No constraints on parameters (beyond sign of γ for asymmetry)
│   │   └─ Use Cases:
│   │       Strong asymmetry (emerging markets)
│   │       When non-negativity constraints binding in GARCH
│   └─ TGARCH (Threshold GARCH):
│       Similar to GJR but uses absolute value threshold
│       Less common than GJR-GARCH in practice
│
├─ IGARCH (Integrated GARCH):
│   │ Unit root in variance: α + β = 1
│   │ σ²ₜ = ω + α ε²ₜ₋₁ + (1-α) σ²ₜ₋₁
│   │ Persistence: Shocks have permanent effect on volatility level
│   │
│   ├─ Properties:
│   │   Non-stationary (unconditional variance infinite)
│   │   But conditionally well-defined
│   │   Forecasts don't revert to long-run mean
│   ├─ When to Use:
│   │   If α̂ + β̂ very close to 1 (0.98-0.99)
│   │   High-frequency financial data
│   │   Volatility trending (no clear mean reversion)
│   └─ Drawbacks:
│       Unrealistic long-term (volatility doesn't explode in reality)
│       Often artifact of structural breaks
│       Component GARCH better if long memory
│
├─ Extensions and Variants:
│   ├─ GARCH-in-Mean (GARCH-M):
│   │   │ Mean equation includes conditional variance:
│   │   │ rₜ = μ + λ σ²ₜ + εₜ
│   │   │ λ: Risk premium (compensation for volatility)
│   │   │
│   │   │ Use: Asset pricing (risk-return tradeoff)
│   │   │ Example: Higher vol periods → higher expected returns
│   │   └─ Interpretation: λ > 0 → positive risk premium
│   ├─ Component GARCH:
│   │   │ Decompose volatility: σ²ₜ = qₜ + (σ²ₜ - qₜ)
│   │   │ qₜ: Permanent component (long-run trend)
│   │   │ σ²ₜ - qₜ: Transitory component (short-run fluctuations)
│   │   │ Captures long memory without unit root
│   │   └─ Use: Volatility forecasting (separates trends from noise)
│   ├─ Multivariate GARCH:
│   │   BEKK, DCC, VECH models
│   │   Covariance matrices evolve over time
│   │   Portfolio risk, correlation dynamics
│   │   Complex estimation (many parameters)
│   ├─ Student-t GARCH:
│   │   εₜ|Fₜ₋₁ ~ t_ν (Student-t with ν degrees of freedom)
│   │   Fatter tails than normal even conditionally
│   │   Better for extreme events (crashes)
│   │   ν estimated alongside GARCH parameters
│   └─ Fractionally Integrated GARCH (FIGARCH):
│       Long memory: Fractional differencing (0 < d < 1)
│       Hyperbolic decay in ACF of squared residuals
│       Computationally intensive
│
└─ Practical Implementation:
    ├─ Diagnostic Checks:
    │   ├─ Standardized Residuals:
    │   │   zₜ = εₜ/σₜ should be i.i.d.
    │   │   ACF of zₜ and z²ₜ: No significant autocorrelation
    │   │   Ljung-Box test: p-value > 0.05 (pass)
    │   ├─ Normality:
    │   │   Q-Q plot, Jarque-Bera test
    │   │   If non-normal: Use t-distribution
    │   ├─ Sign Bias Test:
    │   │   Test if positive/negative shocks have different impact
    │   │   If significant: Use GJR or EGARCH
    │   └─ Goodness of Fit:
    │       VaR backtesting: Count violations (actual < VaR)
    │       Should match nominal level (5% for 95% VaR)
    │       Kupiec POF test: H0: Correct coverage
    ├─ Software:
    │   Python: arch package (arch.univariate)
    │   R: rugarch, fGarch packages
    │   MATLAB: Econometrics Toolbox
    │   Automated: Model selection via AIC/BIC
    ├─ Computational Considerations:
    │   Optimization sensitive to starting values
    │   Use two-stage: OLS then MLE
    │   Variance targeting: Fix unconditional variance
    │   Reduces parameters, improves convergence
    └─ Out-of-Sample Validation:
        Rolling window: Re-estimate every k days
        Forecast accuracy: MAE, RMSE of σ²ₜ vs realized vol
        VaR coverage: Actual exceedances vs theoretical
        Compare to benchmarks: RiskMetrics (EWMA), HAR-RV
```

**Interaction:** Estimate mean model (ARMA) on returns → Extract residuals εₜ → Test for ARCH effects (Ljung-Box on ε²ₜ) → If present, specify GARCH(p,q) → Estimate via MLE with constraints → Diagnose standardized residuals (ACF, normality, sign bias) → If asymmetry, use GJR/EGARCH → Forecast σₜ₊ₕ → Calculate VaR → Backtest on hold-out sample

## 5. Mini-Project
Comprehensive GARCH modeling with symmetric/asymmetric variants, diagnostics, and volatility forecasting:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from arch.univariate import GARCH, EGARCH, ConstantMean, ZeroMean
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GARCH MODELS AND CONDITIONAL HETEROSCEDASTICITY")
print("="*80)

# Generate synthetic GARCH(1,1) data
np.random.seed(42)
n = 2000

# Parameters for GARCH(1,1)
omega = 0.01
alpha = 0.10
beta = 0.85
persistence = alpha + beta

print(f"\nTrue GARCH(1,1) Parameters:")
print(f"  ω (omega): {omega:.3f}")
print(f"  α (alpha): {alpha:.3f}")
print(f"  β (beta): {beta:.3f}")
print(f"  Persistence (α+β): {persistence:.3f}")
print(f"  Unconditional Variance: {omega/(1-persistence):.3f}")
print(f"  Half-life: {np.log(0.5)/np.log(persistence):.1f} periods")

# Simulate GARCH(1,1)
returns = np.zeros(n)
sigma2 = np.zeros(n)
sigma2[0] = omega / (1 - persistence)  # Start at unconditional variance

for t in range(1, n):
    sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    returns[t] = np.sqrt(sigma2[t]) * np.random.normal()

# Create time series
dates = pd.date_range('2010-01-01', periods=n, freq='D')
returns_series = pd.Series(returns, index=dates, name='Returns')
true_vol = pd.Series(np.sqrt(sigma2), index=dates, name='True_Volatility')

print(f"\nSimulated Series Statistics:")
print(f"  Mean: {returns.mean():.4f}")
print(f"  Std Dev: {returns.std():.4f}")
print(f"  Skewness: {stats.skew(returns):.4f}")
print(f"  Kurtosis (excess): {stats.kurtosis(returns):.4f}")
print(f"  Min: {returns.min():.4f}")
print(f"  Max: {returns.max():.4f}")

# Test for ARCH effects
print("\n" + "="*80)
print("ARCH EFFECTS DETECTION")
print("="*80)

squared_returns = returns_series ** 2
lb_test = acorr_ljungbox(squared_returns, lags=10, return_df=True)

print("\nLjung-Box Test on Squared Returns:")
print(lb_test)
print(f"\nInterpretation: P-values < 0.05 indicate ARCH effects present")
print(f"Result: {'ARCH effects detected' if (lb_test['lb_pvalue'] < 0.05).any() else 'No ARCH effects'}")

# GARCH(1,1) Estimation
print("\n" + "="*80)
print("GARCH(1,1) MODEL ESTIMATION")
print("="*80)

# Fit GARCH(1,1) with normal distribution
garch11 = arch_model(returns_series, vol='Garch', p=1, q=1, dist='normal')
garch11_fitted = garch11.fit(disp='off')

print(garch11_fitted.summary())

# Extract parameters
omega_est = garch11_fitted.params['omega']
alpha_est = garch11_fitted.params['alpha[1]']
beta_est = garch11_fitted.params['beta[1]']
persistence_est = alpha_est + beta_est

print(f"\nEstimated vs True Parameters:")
print(f"  ω: Estimated={omega_est:.4f}, True={omega:.4f}")
print(f"  α: Estimated={alpha_est:.4f}, True={alpha:.4f}")
print(f"  β: Estimated={beta_est:.4f}, True={beta:.4f}")
print(f"  Persistence: Estimated={persistence_est:.4f}, True={persistence:.4f}")

# GARCH(1,1) with Student-t distribution
print("\n" + "="*80)
print("GARCH(1,1) WITH STUDENT-T DISTRIBUTION")
print("="*80)

garch11_t = arch_model(returns_series, vol='Garch', p=1, q=1, dist='t')
garch11_t_fitted = garch11_t.fit(disp='off')

nu = garch11_t_fitted.params['nu']
print(f"\nStudent-t degrees of freedom: {nu:.2f}")
print(f"Model Comparison:")
print(f"  Normal GARCH AIC: {garch11_fitted.aic:.2f}")
print(f"  Student-t GARCH AIC: {garch11_t_fitted.aic:.2f}")
print(f"  Preferred: {'Student-t' if garch11_t_fitted.aic < garch11_fitted.aic else 'Normal'}")

# GJR-GARCH (Asymmetric)
print("\n" + "="*80)
print("GJR-GARCH (ASYMMETRIC LEVERAGE EFFECT)")
print("="*80)

# Generate asymmetric data
returns_asym = np.zeros(n)
sigma2_asym = np.zeros(n)
sigma2_asym[0] = 0.01
gamma = 0.08  # Asymmetry parameter

for t in range(1, n):
    indicator = 1 if returns_asym[t-1] < 0 else 0
    sigma2_asym[t] = omega + alpha * returns_asym[t-1]**2 + gamma * indicator * returns_asym[t-1]**2 + beta * sigma2_asym[t-1]
    returns_asym[t] = np.sqrt(sigma2_asym[t]) * np.random.normal()

returns_asym_series = pd.Series(returns_asym, index=dates, name='Asymmetric_Returns')

# Fit GJR-GARCH
gjr_garch = arch_model(returns_asym_series, vol='Garch', p=1, o=1, q=1, dist='normal')
gjr_fitted = gjr_garch.fit(disp='off')

print(gjr_fitted.summary())

gamma_est = gjr_fitted.params['gamma[1]']
print(f"\nAsymmetry Parameter (γ): {gamma_est:.4f}")
print(f"Interpretation: {'Leverage effect present' if gamma_est > 0 else 'No asymmetry'}")
print(f"  Negative shock impact: {alpha_est + gamma_est:.4f}")
print(f"  Positive shock impact: {alpha_est:.4f}")

# EGARCH
print("\n" + "="*80)
print("EGARCH (EXPONENTIAL GARCH)")
print("="*80)

egarch = arch_model(returns_asym_series, vol='EGARCH', p=1, q=1, dist='normal')
egarch_fitted = egarch.fit(disp='off')

print(egarch_fitted.summary())

# Diagnostics
print("\n" + "="*80)
print("MODEL DIAGNOSTICS")
print("="*80)

# Standardized residuals
std_resid = garch11_fitted.std_resid

print("\n1. STANDARDIZED RESIDUALS (Should be i.i.d.)")
print("-" * 40)
lb_std = acorr_ljungbox(std_resid, lags=10, return_df=True)
print(f"Ljung-Box test p-value (lag 10): {lb_std['lb_pvalue'].iloc[-1]:.4f}")
print(f"Result: {'PASS (No autocorrelation)' if lb_std['lb_pvalue'].iloc[-1] > 0.05 else 'FAIL'}")

print("\n2. STANDARDIZED SQUARED RESIDUALS")
print("-" * 40)
lb_std_sq = acorr_ljungbox(std_resid**2, lags=10, return_df=True)
print(f"Ljung-Box test p-value (lag 10): {lb_std_sq['lb_pvalue'].iloc[-1]:.4f}")
print(f"Result: {'PASS (No remaining ARCH)' if lb_std_sq['lb_pvalue'].iloc[-1] > 0.05 else 'FAIL'}")

print("\n3. NORMALITY TEST")
print("-" * 40)
jb_stat, jb_pvalue = stats.jarque_bera(std_resid)
print(f"Jarque-Bera statistic: {jb_stat:.4f}")
print(f"P-value: {jb_pvalue:.4f}")
print(f"Result: {'Normal' if jb_pvalue > 0.05 else 'Non-normal (consider t-distribution)'}")

# Forecasting
print("\n" + "="*80)
print("VOLATILITY FORECASTING")
print("="*80)

# Out-of-sample forecast
train_size = 1800
test_size = n - train_size
returns_train = returns_series[:train_size]
returns_test = returns_series[train_size:]

# Fit on training data
garch_train = arch_model(returns_train, vol='Garch', p=1, q=1, dist='normal')
garch_train_fitted = garch_train.fit(disp='off')

# Forecast volatility
forecast_horizon = test_size
forecasts = garch_train_fitted.forecast(horizon=forecast_horizon, reindex=False)
forecast_variance = forecasts.variance.values[-1, :]  # Last row = forecast from train end
forecast_volatility = np.sqrt(forecast_variance)

# Realized volatility (proxy: absolute returns)
realized_vol = np.abs(returns_test.values)

# Forecast evaluation
mae_vol = np.mean(np.abs(realized_vol - forecast_volatility))
rmse_vol = np.sqrt(np.mean((realized_vol - forecast_volatility)**2))

print(f"\nVolatility Forecast Evaluation:")
print(f"  Training period: {train_size} days")
print(f"  Test period: {test_size} days")
print(f"  Forecast Horizon: {forecast_horizon} days")
print(f"\nAccuracy Metrics:")
print(f"  MAE: {mae_vol:.6f}")
print(f"  RMSE: {rmse_vol:.6f}")

# VaR Calculation
print("\n" + "="*80)
print("VALUE-AT-RISK (VaR) CALCULATION")
print("="*80)

confidence_level = 0.95
quantile = stats.norm.ppf(1 - confidence_level)

# 1-day VaR
fitted_vol = garch11_fitted.conditional_volatility
var_95 = quantile * fitted_vol  # For long position (negative means loss)

# Backtesting
violations = returns_series < var_95
violation_rate = violations.sum() / len(returns_series)
expected_rate = 1 - confidence_level

print(f"\n95% VaR Backtesting:")
print(f"  Expected violation rate: {expected_rate:.2%}")
print(f"  Actual violation rate: {violation_rate:.2%}")
print(f"  Number of violations: {violations.sum()} out of {len(returns_series)}")
print(f"  Result: {'PASS' if abs(violation_rate - expected_rate) < 0.01 else 'CHECK (potential model issue)'}")

# Kupiec POF Test
from scipy.stats import chi2
n_obs = len(returns_series)
n_violations = violations.sum()
lr_stat = -2 * (np.log((1-expected_rate)**(n_obs-n_violations) * expected_rate**n_violations) -
                np.log((1-violation_rate)**(n_obs-n_violations) * violation_rate**n_violations))
lr_pvalue = 1 - chi2.cdf(lr_stat, df=1)

print(f"\nKupiec POF Test:")
print(f"  LR statistic: {lr_stat:.4f}")
print(f"  P-value: {lr_pvalue:.4f}")
print(f"  Result: {'PASS (Correct coverage)' if lr_pvalue > 0.05 else 'FAIL (Incorrect coverage)'}")

# Visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot 1: Returns with volatility clustering
ax = axes[0, 0]
ax.plot(returns_series, linewidth=0.8, alpha=0.7)
ax.set_title('Simulated Returns (Volatility Clustering)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Returns')
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)

# Plot 2: Squared returns (ARCH effects)
ax = axes[0, 1]
ax.plot(squared_returns, linewidth=0.8, alpha=0.7, color='orange')
ax.set_title('Squared Returns (ARCH Effects)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Squared Returns')
ax.grid(alpha=0.3)

# Plot 3: ACF of squared returns
ax = axes[0, 2]
plot_acf(squared_returns, lags=30, ax=ax, alpha=0.05)
ax.set_title('ACF of Squared Returns', fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: True vs Estimated Volatility
ax = axes[1, 0]
ax.plot(true_vol, label='True σ', linewidth=1.5, alpha=0.7)
ax.plot(garch11_fitted.conditional_volatility, label='Estimated σ (GARCH)', 
        linewidth=1.5, alpha=0.7, linestyle='--')
ax.set_title('True vs Estimated Conditional Volatility', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Standardized residuals
ax = axes[1, 1]
ax.plot(std_resid, linewidth=0.8, alpha=0.7, color='green')
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.axhline(2, color='orange', linestyle='--', alpha=0.5)
ax.axhline(-2, color='orange', linestyle='--', alpha=0.5)
ax.set_title('Standardized Residuals', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('z-score')
ax.grid(alpha=0.3)

# Plot 6: Q-Q plot
ax = axes[1, 2]
stats.probplot(std_resid.dropna(), dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
ax.grid(alpha=0.3)

# Plot 7: ACF of standardized residuals
ax = axes[2, 0]
plot_acf(std_resid.dropna(), lags=30, ax=ax, alpha=0.05)
ax.set_title('ACF of Standardized Residuals', fontweight='bold')
ax.grid(alpha=0.3)

# Plot 8: Volatility forecast
ax = axes[2, 1]
test_dates = returns_test.index
ax.plot(test_dates, realized_vol, label='Realized |r|', linewidth=1.5, alpha=0.7)
ax.plot(test_dates, forecast_volatility, label='GARCH Forecast', 
        linewidth=2, linestyle='--', color='red')
ax.set_title('Out-of-Sample Volatility Forecast', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility')
ax.legend()
ax.grid(alpha=0.3)

# Plot 9: VaR violations
ax = axes[2, 2]
ax.plot(returns_series, linewidth=0.8, alpha=0.6, label='Returns')
ax.plot(var_95, linewidth=1.5, color='red', linestyle='--', label='95% VaR', alpha=0.7)
ax.scatter(returns_series[violations].index, returns_series[violations], 
          color='red', s=20, marker='x', label=f'Violations ({violations.sum()})', zorder=5)
ax.set_title('VaR Backtesting', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Returns')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Model comparison table
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

comparison = pd.DataFrame({
    'Model': ['GARCH(1,1) Normal', 'GARCH(1,1) Student-t', 'GJR-GARCH', 'EGARCH'],
    'AIC': [garch11_fitted.aic, garch11_t_fitted.aic, gjr_fitted.aic, egarch_fitted.aic],
    'BIC': [garch11_fitted.bic, garch11_t_fitted.bic, gjr_fitted.bic, egarch_fitted.bic],
    'Log-Likelihood': [garch11_fitted.loglikelihood, garch11_t_fitted.loglikelihood, 
                      gjr_fitted.loglikelihood, egarch_fitted.loglikelihood]
})

print("\n" + comparison.to_string(index=False))

best_model = comparison.loc[comparison['AIC'].idxmin(), 'Model']
print(f"\nBest Model (by AIC): {best_model}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. GARCH(1,1) captures {persistence_est:.1%} volatility persistence (α+β)")
print(f"2. Student-t distribution fits better for fat tails (ν={nu:.1f})")
print(f"3. Leverage effect: GJR-GARCH γ={gamma_est:.3f} (negative shocks increase vol)")
print(f"4. Volatility clustering evident in ACF of squared returns")
print(f"5. VaR backtesting: {violation_rate:.2%} actual vs {expected_rate:.2%} expected")
print(f"6. Half-life of volatility shocks: ~{np.log(0.5)/np.log(persistence_est):.0f} days")
print(f"7. GARCH generates fat tails (kurtosis={stats.kurtosis(returns):.2f}) even with normal innovations")
```

## 6. Challenge Round
Advanced GARCH applications and extensions:

1. **Component GARCH:** Decompose volatility into permanent (qₜ) and transitory components. Estimate using two-equation system. Compare long-horizon forecasts to standard GARCH(1,1). When does component structure improve accuracy?

2. **Multivariate DCC-GARCH:** Model time-varying correlation between stock and bond returns. Estimate Dynamic Conditional Correlation. How does correlation change during crisis (2008, COVID)? Implications for portfolio diversification?

3. **GARCH Option Pricing:** Use GARCH(1,1) volatility forecast in Black-Scholes formula (replace constant σ with conditional σₜ). Compare to market option prices. Does GARCH improve implied volatility smile fit?

4. **Realized GARCH:** Incorporate realized volatility (computed from intraday data) into GARCH framework. Estimate on 5-minute returns. Compare forecast accuracy to standard GARCH. When is high-frequency data beneficial?

5. **Regime-Switching GARCH:** Allow GARCH parameters to switch between regimes (low/high volatility). Use Markov-Switching model. Estimate transition probabilities. How long do high-vol regimes persist?

6. **GARCH-MIDAS:** Mix high-frequency (daily) volatility with low-frequency (monthly) macro variables. How do GDP, inflation affect long-run volatility component? Forecast recession impacts.

7. **Portfolio Optimization with GARCH:** Estimate conditional covariance matrix for 10 stocks using DCC-GARCH. Solve dynamic mean-variance optimization (rebalance daily). Compare to static covariance vs. rolling window approaches.

## 7. Key References
- [Engle, "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation" (Econometrica, 1982)](https://www.jstor.org/stable/1912773) - original ARCH paper, Nobel Prize work
- [Bollerslev, "Generalized Autoregressive Conditional Heteroscedasticity" (Journal of Econometrics, 1986)](https://www.sciencedirect.com/science/article/abs/pii/0304407686900631) - GARCH extension, foundational reference
- [Hansen & Lunde, "A Forecast Comparison of Volatility Models" (Journal of Econometrics, 2005)](https://www.sciencedirect.com/science/article/abs/pii/S0304407603000025) - comprehensive comparison of 330 ARCH-type models, empirical guide

---
**Status:** Core financial volatility modeling | **Complements:** ARIMA Models, Risk Management (VaR/CVaR), Option Pricing, Portfolio Optimization, High-Frequency Econometrics, Realized Volatility
