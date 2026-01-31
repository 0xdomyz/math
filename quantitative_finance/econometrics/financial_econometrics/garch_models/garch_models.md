# GARCH Models (Generalized Autoregressive Conditional Heteroskedasticity)

## 1. Concept Skeleton
**Definition:** Time series models for conditional variance that evolves over time; captures volatility clustering; allows variance to depend on past squared errors and past variances  
**Purpose:** Model time-varying volatility; forecast variance; risk management; option pricing; portfolio optimization; capture stylized facts (clustering, asymmetry)  
**Prerequisites:** ARCH models, time series, maximum likelihood, stationarity, volatility clustering, leverage effects

## 2. Comparative Framing
| Model | Variance Equation | Parameters | Asymmetry | Persistence | Long Memory |
|-------|-------------------|------------|-----------|-------------|-------------|
| **ARCH(q)** | σₜ² = ω + Σαᵢεₜ₋ᵢ² | q+1 | No | Limited | No |
| **GARCH(p,q)** | σₜ² = ω + Σαᵢεₜ₋ᵢ² + Σβⱼσₜ₋ⱼ² | p+q+1 | No | High | No |
| **EGARCH** | log(σₜ²) = ω + Σαᵢg(zₜ₋ᵢ) + Σβⱼlog(σₜ₋ⱼ²) | p+q+2 | Yes | High | No |
| **GJR-GARCH** | σₜ² = ω + Σ(αᵢ+γᵢIₜ₋ᵢ)εₜ₋ᵢ² + Σβⱼσₜ₋ⱼ² | p+q+2 | Yes | High | No |
| **FIGARCH** | (1-L)^d(1-β(L))σₜ² = ω + α(L)εₜ² | p+q+2 | No | High | Yes |
| **APARCH** | σₜ^δ = ω + Σαᵢ(\|εₜ₋ᵢ\|-γᵢεₜ₋ᵢ)^δ + Σβⱼσₜ₋ⱼ^δ | p+q+3 | Yes | High | No |

## 3. Examples + Counterexamples

**Classic Example:**  
S&P 500 daily returns exhibit volatility clustering—high-vol days follow high-vol days. GARCH(1,1) with α+β≈0.99 captures persistence. Black Monday 1987 shows leverage effect—negative returns increase volatility more. GJR-GARCH improves fit.

**Failure Case:**  
Intraday 5-minute returns with market microstructure noise. GARCH assumes continuous sampling but bid-ask bounce creates spurious volatility. Realized GARCH or HAR models better for high-frequency data.

**Edge Case:**  
COVID-19 volatility spike (VIX 80+). Standard GARCH underestimates tail risk. Need heavy-tailed innovations (Student-t) or regime-switching GARCH. Single-regime GARCH forecasts revert too quickly.

## 4. Layer Breakdown
```
GARCH Models Framework:
├─ ARCH(q) Foundation:
│   ├─ Mean Equation: rₜ = μ + εₜ, εₜ = σₜzₜ, zₜ ~ N(0,1)
│   ├─ Variance Equation: σₜ² = ω + α₁εₜ₋₁² + ... + αqεₜ₋q²
│   ├─ Interpretation:
│   │   ├─ ω > 0: Baseline volatility
│   │   ├─ αᵢ ≥ 0: Impact of past shocks (ARCH effects)
│   │   └─ Σαᵢ < 1: Stationarity condition
│   ├─ Limitations:
│   │   ├─ Requires many lags (q large) for persistence
│   │   └─ No distinction between shock and volatility
│   └─ Historical Context:
│       └─ Engle (1982) Nobel Prize work on inflation volatility
├─ GARCH(p,q) Model:
│   ├─ Variance Equation: σₜ² = ω + Σαᵢεₜ₋ᵢ² + Σβⱼσₜ₋ⱼ²
│   │   ├─ ARCH terms (α): Past squared shocks
│   │   ├─ GARCH terms (β): Past conditional variances
│   │   └─ Parsimonious: GARCH(1,1) often sufficient
│   ├─ GARCH(1,1) Specification:
│   │   ├─ σₜ² = ω + α₁εₜ₋₁² + β₁σₜ₋₁²
│   │   ├─ Most popular in practice
│   │   └─ Only 3 parameters (ω, α₁, β₁)
│   ├─ Parameter Constraints:
│   │   ├─ ω > 0: Positive baseline
│   │   ├─ αᵢ ≥ 0, βⱼ ≥ 0: Non-negativity
│   │   └─ Σαᵢ + Σβⱼ < 1: Covariance stationarity
│   ├─ Unconditional Variance:
│   │   └─ Var(εₜ) = ω / (1 - Σαᵢ - Σβⱼ)
│   ├─ Persistence:
│   │   ├─ Measured by Σαᵢ + Σβⱼ
│   │   ├─ Close to 1: High persistence
│   │   └─ Typical value: 0.95-0.99 for daily returns
│   └─ Shock Decay:
│       └─ Half-life ≈ log(0.5)/log(α₁+β₁)
├─ Estimation:
│   ├─ Maximum Likelihood (Standard):
│   │   ├─ Log-likelihood: L = -Σ[log(σₜ²) + εₜ²/σₜ²]
│   │   ├─ Assumes Gaussian innovations
│   │   └─ Numerical optimization (BFGS, Newton-Raphson)
│   ├─ Quasi-Maximum Likelihood (QML):
│   │   ├─ Consistent even if zₜ not Gaussian
│   │   ├─ Robust standard errors (sandwich estimator)
│   │   └─ Preferred in practice
│   ├─ Algorithm:
│   │   ├─ Initialize: σ₀² = sample variance
│   │   ├─ Recursion: Compute σₜ² for t=1,...,T
│   │   ├─ Log-likelihood: Sum contributions
│   │   └─ Optimize: Find θ̂ = argmax L(θ)
│   └─ Numerical Issues:
│       ├─ Variance must stay positive
│       ├─ Reparameterize to enforce constraints
│       └─ Good starting values critical
├─ Asymmetric GARCH Models:
│   ├─ Leverage Effect:
│   │   ├─ Negative returns → Higher volatility increase
│   │   ├─ Black (1976), Christie (1982) empirical finding
│   │   └─ Standard GARCH symmetric (α same for ±shocks)
│   ├─ GJR-GARCH (Threshold GARCH):
│   │   ├─ σₜ² = ω + (α₁+γ₁Iₜ₋₁)εₜ₋₁² + β₁σₜ₋₁²
│   │   ├─ Iₜ₋₁ = 1 if εₜ₋₁ < 0, else 0
│   │   ├─ γ > 0: Leverage effect (bad news increases vol more)
│   │   └─ Glosten, Jagannathan, Runkle (1993)
│   ├─ EGARCH (Exponential GARCH):
│   │   ├─ log(σₜ²) = ω + α₁g(zₜ₋₁) + β₁log(σₜ₋₁²)
│   │   ├─ g(zₜ) = θzₜ + γ[|zₜ| - E|zₜ|]
│   │   │   ├─ θ ≠ 0: Asymmetry
│   │   │   └─ γ: Size effect
│   │   ├─ No constraints (log ensures σₜ² > 0)
│   │   └─ Nelson (1991)
│   ├─ APARCH (Asymmetric Power ARCH):
│   │   ├─ σₜ^δ = ω + α₁(|εₜ₋₁|-γ₁εₜ₋₁)^δ + β₁σₜ₋₁^δ
│   │   ├─ δ: Power parameter (estimated)
│   │   ├─ δ=2: Standard GARCH
│   │   └─ Ding, Granger, Engle (1993)
│   └─ News Impact Curve:
│       ├─ Plot: σₜ² vs εₜ₋₁ (holding σₜ₋₁² constant)
│       ├─ Symmetric: V-shape (GARCH)
│       └─ Asymmetric: Steeper left (negative shocks)
├─ Extensions:
│   ├─ GARCH-in-Mean (GARCH-M):
│   │   ├─ Mean equation: rₜ = μ + λσₜ² + εₜ
│   │   ├─ λ: Risk premium parameter
│   │   └─ Higher volatility → Higher expected return
│   ├─ Integrated GARCH (IGARCH):
│   │   ├─ Σαᵢ + Σβⱼ = 1 (unit root in variance)
│   │   ├─ Shocks have permanent effect
│   │   └─ No unconditional variance
│   ├─ Fractionally Integrated GARCH (FIGARCH):
│   │   ├─ Allows long memory (0 < d < 1)
│   │   ├─ Shock decay hyperbolic (not exponential)
│   │   └─ Baillie, Bollerslev, Mikkelsen (1996)
│   ├─ Component GARCH:
│   │   ├─ Separate short-run and long-run volatility
│   │   └─ Engle, Lee (1999)
│   └─ Realized GARCH:
│       ├─ Incorporates realized volatility measures
│       └─ Hansen, Huang, Shek (2012)
├─ Multivariate GARCH:
│   ├─ VEC-GARCH:
│   │   ├─ Full parametrization of covariance matrix
│   │   └─ O(n²) parameters (curse of dimensionality)
│   ├─ BEKK Model:
│   │   ├─ Ensures positive-definite covariance
│   │   └─ Still many parameters
│   ├─ DCC-GARCH (Dynamic Conditional Correlation):
│   │   ├─ Separate univariate GARCH + dynamic correlations
│   │   ├─ Rₜ = diag(σₜ)⁻¹Hₜdiag(σₜ)⁻¹
│   │   └─ Engle (2002), practical for large n
│   └─ CCC-GARCH (Constant Conditional Correlation):
│       └─ Assumes constant correlation (simpler)
├─ Diagnostics:
│   ├─ Standardized Residuals:
│   │   ├─ ẑₜ = εₜ / σ̂ₜ
│   │   └─ Should be i.i.d. with mean 0, variance 1
│   ├─ Tests on Standardized Residuals:
│   │   ├─ Ljung-Box: No autocorrelation
│   │   ├─ Ljung-Box on ẑₜ²: No remaining ARCH
│   │   ├─ Jarque-Bera: Normality (often rejected)
│   │   └─ ARCH-LM test: No remaining conditional heteroskedasticity
│   ├─ Sign Bias Test:
│   │   ├─ Engle-Ng (1993)
│   │   └─ Tests for remaining asymmetry
│   └─ Goodness-of-Fit:
│       ├─ AIC, BIC: Model comparison
│       └─ Log-likelihood ratio tests
├─ Forecasting:
│   ├─ 1-Step Ahead:
│   │   └─ σ̂ₜ₊₁² = ω̂ + α̂εₜ² + β̂σ̂ₜ²
│   ├─ h-Step Ahead (GARCH(1,1)):
│   │   ├─ σ̂ₜ₊ₕ² = Var_∞ + (α̂+β̂)^(h-1)(σ̂ₜ₊₁² - Var_∞)
│   │   ├─ Var_∞ = ω̂/(1-α̂-β̂): Unconditional variance
│   │   └─ Exponential decay to long-run variance
│   ├─ Forecast Evaluation:
│   │   ├─ MSE of volatility forecasts
│   │   ├─ QLIKE loss: log(σₜ²) + εₜ²/σₜ²
│   │   └─ Mincer-Zarnowitz regression
│   └─ Volatility Proxy:
│       ├─ Squared returns: εₜ² (noisy proxy)
│       └─ Realized volatility: Better proxy
├─ Alternative Distributions:
│   ├─ Student-t:
│   │   ├─ Heavier tails than Gaussian
│   │   ├─ Degrees of freedom ν estimated
│   │   └─ Better fit for financial returns
│   ├─ Generalized Error Distribution (GED):
│   │   ├─ Shape parameter κ
│   │   └─ Nests Gaussian (κ=2), Laplace (κ=1)
│   └─ Skewed Distributions:
│       └─ Capture return asymmetry
├─ Applications:
│   ├─ Risk Management:
│   │   ├─ VaR calculation: Use σ̂ₜ₊₁ and quantiles
│   │   └─ Expected shortfall (CVaR)
│   ├─ Option Pricing:
│   │   ├─ Duan (1995) GARCH option pricing
│   │   └─ Time-varying volatility improves fit
│   ├─ Portfolio Optimization:
│   │   ├─ Time-varying covariance matrix
│   │   └─ Dynamic hedging
│   └─ Volatility Trading:
│       ├─ VIX futures, variance swaps
│       └─ Statistical arbitrage
└─ Software:
    ├─ Python: arch package (Kevin Sheppard)
    ├─ R: rugarch, fGarch packages
    ├─ MATLAB: GARCH Toolbox
    └─ EViews, Stata: Built-in GARCH estimation
```

**Interaction:** Specify mean/variance equations → Estimate via MLE → Diagnose standardized residuals → Forecast volatility → Evaluate with realized measures

## 5. Mini-Project
Implement GARCH(1,1), GJR-GARCH, and EGARCH on S&P 500 returns with forecasting:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta

# ===== Download S&P 500 Data =====
print("="*80)
print("GARCH MODELS FOR VOLATILITY")
print("="*80)

# Download 5 years of S&P 500 data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print(f"\nDownloading S&P 500 data from {start_date.date()} to {end_date.date()}...")
data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)

# Calculate returns
returns = 100 * data['Adj Close'].pct_change().dropna()
returns = returns.values

print(f"✓ Downloaded {len(returns)} daily returns")

# Summary statistics
print(f"\nReturn Statistics:")
print(f"  Mean: {returns.mean():.4f}%")
print(f"  Std Dev: {returns.std():.4f}%")
print(f"  Skewness: {stats.skew(returns):.4f}")
print(f"  Kurtosis: {stats.kurtosis(returns):.4f}")
print(f"  Min: {returns.min():.4f}%")
print(f"  Max: {returns.max():.4f}%")

# ===== Test for ARCH Effects =====
print("\n" + "="*80)
print("ARCH EFFECTS TEST")
print("="*80)

# Ljung-Box test on squared returns
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(returns**2, lags=[10], return_df=True)
print(f"Ljung-Box test on squared returns (lag 10):")
print(f"  Test Statistic: {lb_test['lb_stat'].values[0]:.2f}")
print(f"  P-value: {lb_test['lb_pvalue'].values[0]:.4f}")

if lb_test['lb_pvalue'].values[0] < 0.05:
    print("  ✓ Reject null of no ARCH effects (GARCH appropriate)")
else:
    print("  ✗ No evidence of ARCH effects")

# ===== Split Data =====
train_size = int(0.8 * len(returns))
returns_train = returns[:train_size]
returns_test = returns[train_size:]

print(f"\nTrain/Test Split:")
print(f"  Training: {len(returns_train)} observations")
print(f"  Testing: {len(returns_test)} observations")

# ===== GARCH(1,1) Model =====
print("\n" + "="*80)
print("GARCH(1,1) MODEL")
print("="*80)

# Fit GARCH(1,1) with normal distribution
garch_model = arch_model(returns_train, vol='Garch', p=1, q=1, dist='normal')
garch_fit = garch_model.fit(disp='off')

print(garch_fit.summary())

# Extract parameters
omega = garch_fit.params['omega']
alpha = garch_fit.params['alpha[1]']
beta = garch_fit.params['beta[1]']

print(f"\nGARCH(1,1) Parameters:")
print(f"  ω (omega): {omega:.6f}")
print(f"  α (alpha): {alpha:.6f}")
print(f"  β (beta): {beta:.6f}")
print(f"  α + β: {alpha + beta:.6f} (persistence)")

# Unconditional variance
uncond_var = omega / (1 - alpha - beta)
print(f"  Unconditional variance: {uncond_var:.6f}")
print(f"  Unconditional vol: {np.sqrt(uncond_var):.4f}%")

# Half-life of shocks
half_life = np.log(0.5) / np.log(alpha + beta)
print(f"  Half-life of shocks: {half_life:.1f} days")

# ===== GJR-GARCH (Threshold GARCH) =====
print("\n" + "="*80)
print("GJR-GARCH MODEL (ASYMMETRIC)")
print("="*80)

gjr_model = arch_model(returns_train, vol='Garch', p=1, o=1, q=1, dist='normal')
gjr_fit = gjr_model.fit(disp='off')

print(gjr_fit.summary())

gamma = gjr_fit.params['gamma[1]']
print(f"\nLeverage Parameter:")
print(f"  γ (gamma): {gamma:.6f}")

if gamma > 0:
    print(f"  ✓ Negative shocks increase volatility more (leverage effect)")
else:
    print(f"  ✗ No significant leverage effect")

# ===== EGARCH Model =====
print("\n" + "="*80)
print("EGARCH MODEL (EXPONENTIAL)")
print("="*80)

egarch_model = arch_model(returns_train, vol='EGARCH', p=1, q=1, dist='normal')
egarch_fit = egarch_model.fit(disp='off')

print(egarch_fit.summary())

# ===== GARCH with Student-t Distribution =====
print("\n" + "="*80)
print("GARCH(1,1) WITH STUDENT-T DISTRIBUTION")
print("="*80)

garch_t_model = arch_model(returns_train, vol='Garch', p=1, q=1, dist='t')
garch_t_fit = garch_t_model.fit(disp='off')

print(garch_t_fit.summary())

df_param = garch_t_fit.params['nu']
print(f"\nDegrees of Freedom: {df_param:.2f}")
print(f"  ν < 5: Very heavy tails")
print(f"  ν ≈ 5-10: Heavy tails (typical for daily returns)")
print(f"  ν > 30: Close to normal")

# ===== Model Comparison =====
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

models_dict = {
    'GARCH(1,1)': garch_fit,
    'GJR-GARCH': gjr_fit,
    'EGARCH': egarch_fit,
    'GARCH-t': garch_t_fit
}

comparison_df = pd.DataFrame({
    'Model': list(models_dict.keys()),
    'LogLik': [m.loglikelihood for m in models_dict.values()],
    'AIC': [m.aic for m in models_dict.values()],
    'BIC': [m.bic for m in models_dict.values()],
    'Params': [len(m.params) for m in models_dict.values()]
})

comparison_df = comparison_df.sort_values('AIC')
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
print(f"\n✓ Best model by AIC: {best_model_name}")

# ===== Diagnostics =====
print("\n" + "="*80)
print("DIAGNOSTICS (GARCH(1,1))")
print("="*80)

# Standardized residuals
std_resid = garch_fit.std_resid

print(f"Standardized Residuals:")
print(f"  Mean: {std_resid.mean():.4f} (should be ≈0)")
print(f"  Std: {std_resid.std():.4f} (should be ≈1)")
print(f"  Skewness: {stats.skew(std_resid):.4f}")
print(f"  Kurtosis: {stats.kurtosis(std_resid):.4f}")

# Test on standardized residuals
lb_test_std = acorr_ljungbox(std_resid, lags=[10], return_df=True)
print(f"\nLjung-Box test on standardized residuals:")
print(f"  P-value: {lb_test_std['lb_pvalue'].values[0]:.4f}")
if lb_test_std['lb_pvalue'].values[0] > 0.05:
    print("  ✓ No autocorrelation remaining")

# Test on squared standardized residuals
lb_test_std_sq = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
print(f"\nLjung-Box test on squared standardized residuals:")
print(f"  P-value: {lb_test_std_sq['lb_pvalue'].values[0]:.4f}")
if lb_test_std_sq['lb_pvalue'].values[0] > 0.05:
    print("  ✓ No ARCH effects remaining")

# ===== Out-of-Sample Forecasting =====
print("\n" + "="*80)
print("OUT-OF-SAMPLE FORECASTING")
print("="*80)

# Rolling 1-step ahead forecasts
forecasts_garch = []
forecasts_gjr = []
realized_vol = []

print("Computing rolling forecasts...")

for t in range(len(returns_test)):
    # Expand training window
    returns_expanding = returns[:train_size + t]
    
    # Fit GARCH(1,1)
    model_garch = arch_model(returns_expanding, vol='Garch', p=1, q=1, dist='normal')
    fit_garch = model_garch.fit(disp='off')
    forecast_garch = fit_garch.forecast(horizon=1)
    forecasts_garch.append(np.sqrt(forecast_garch.variance.values[-1, 0]))
    
    # Fit GJR-GARCH
    model_gjr = arch_model(returns_expanding, vol='Garch', p=1, o=1, q=1, dist='normal')
    fit_gjr = model_gjr.fit(disp='off')
    forecast_gjr = fit_gjr.forecast(horizon=1)
    forecasts_gjr.append(np.sqrt(forecast_gjr.variance.values[-1, 0]))
    
    # Realized (proxy: absolute return)
    realized_vol.append(np.abs(returns_test[t]))
    
    if (t+1) % 50 == 0:
        print(f"  Processed {t+1}/{len(returns_test)} forecasts")

forecasts_garch = np.array(forecasts_garch)
forecasts_gjr = np.array(forecasts_gjr)
realized_vol = np.array(realized_vol)

# Forecast evaluation
mse_garch = np.mean((forecasts_garch - realized_vol)**2)
mse_gjr = np.mean((forecasts_gjr - realized_vol)**2)

mae_garch = np.mean(np.abs(forecasts_garch - realized_vol))
mae_gjr = np.mean(np.abs(forecasts_gjr - realized_vol))

# QLIKE loss
qlike_garch = np.mean(np.log(forecasts_garch**2) + (returns_test**2) / (forecasts_garch**2))
qlike_gjr = np.mean(np.log(forecasts_gjr**2) + (returns_test**2) / (forecasts_gjr**2))

print(f"\n✓ Forecast evaluation:")
print(f"GARCH(1,1):")
print(f"  MSE: {mse_garch:.6f}")
print(f"  MAE: {mae_garch:.4f}")
print(f"  QLIKE: {qlike_garch:.6f}")

print(f"\nGJR-GARCH:")
print(f"  MSE: {mse_gjr:.6f}")
print(f"  MAE: {mae_gjr:.4f}")
print(f"  QLIKE: {qlike_gjr:.6f}")

if qlike_gjr < qlike_garch:
    improvement = (qlike_garch - qlike_gjr) / qlike_garch * 100
    print(f"\n✓ GJR-GARCH improves over GARCH by {improvement:.2f}% (QLIKE)")

# ===== Visualizations =====
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: Returns with conditional volatility
cond_vol = garch_fit.conditional_volatility
dates_train = pd.date_range(end=end_date, periods=len(returns_train))

axes[0, 0].plot(dates_train, returns_train, linewidth=0.5, alpha=0.6, label='Returns')
axes[0, 0].fill_between(dates_train, -2*cond_vol, 2*cond_vol, 
                        alpha=0.3, color='red', label='±2σₜ')
axes[0, 0].set_ylabel('Returns (%)')
axes[0, 0].set_title('Returns with GARCH(1,1) Conditional Volatility')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Conditional volatility over time
axes[0, 1].plot(dates_train, cond_vol, linewidth=1.5, color='red')
axes[0, 1].axhline(np.sqrt(uncond_var), color='blue', linestyle='--', 
                  linewidth=2, label='Unconditional σ')
axes[0, 1].set_ylabel('Volatility (%)')
axes[0, 1].set_title('GARCH(1,1) Conditional Volatility')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: News Impact Curve
shock_range = np.linspace(-5, 5, 100)
nic_garch = omega + alpha * shock_range**2 + beta * uncond_var
nic_gjr = (omega + (alpha + gamma * (shock_range < 0)) * shock_range**2 
           + beta * uncond_var)

axes[1, 0].plot(shock_range, np.sqrt(nic_garch), linewidth=2, label='GARCH(1,1)')
axes[1, 0].plot(shock_range, np.sqrt(nic_gjr), linewidth=2, label='GJR-GARCH')
axes[1, 0].axvline(0, color='black', linestyle=':', linewidth=1)
axes[1, 0].set_xlabel('Shock (εₑ₋₁)')
axes[1, 0].set_ylabel('Next Period Volatility (σₜ)')
axes[1, 0].set_title('News Impact Curve')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Standardized Residuals
axes[1, 1].plot(std_resid, linewidth=0.5, alpha=0.7)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 1].axhline(2, color='orange', linestyle=':', linewidth=1)
axes[1, 1].axhline(-2, color='orange', linestyle=':', linewidth=1)
axes[1, 1].set_ylabel('Standardized Residuals')
axes[1, 1].set_title('Standardized Residuals (Should be i.i.d.)')
axes[1, 1].grid(alpha=0.3)

# Plot 5: QQ-Plot of Standardized Residuals
stats.probplot(std_resid, dist="norm", plot=axes[2, 0])
axes[2, 0].set_title('Q-Q Plot: Standardized Residuals vs Normal')
axes[2, 0].grid(alpha=0.3)

# Plot 6: Forecast vs Realized
dates_test = pd.date_range(start=dates_train[-1], periods=len(returns_test)+1)[1:]

axes[2, 1].plot(dates_test, realized_vol, linewidth=1, alpha=0.7, 
               label='Realized (|return|)', color='black')
axes[2, 1].plot(dates_test, forecasts_garch, linewidth=1.5, 
               label='GARCH(1,1)', alpha=0.8)
axes[2, 1].plot(dates_test, forecasts_gjr, linewidth=1.5,
               label='GJR-GARCH', alpha=0.8)
axes[2, 1].set_ylabel('Volatility (%)')
axes[2, 1].set_title('Out-of-Sample Volatility Forecasts')
axes[2, 1].legend(fontsize=8)
axes[2, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('garch_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== ACF of Squared Standardized Residuals =====
from statsmodels.graphics.tsaplots import plot_acf

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(std_resid, lags=20, ax=axes2[0])
axes2[0].set_title('ACF of Standardized Residuals')
axes2[0].grid(alpha=0.3)

plot_acf(std_resid**2, lags=20, ax=axes2[1])
axes2[1].set_title('ACF of Squared Standardized Residuals')
axes2[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('garch_acf.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== VaR Calculation =====
print("\n" + "="*80)
print("VALUE-AT-RISK (VaR) CALCULATION")
print("="*80)

# 1-day ahead forecast using full sample
model_final = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
fit_final = model_final.fit(disp='off')
forecast_final = fit_final.forecast(horizon=1)

vol_forecast = np.sqrt(forecast_final.variance.values[-1, 0])
mean_forecast = forecast_final.mean.values[-1, 0]

# VaR at 95% and 99% confidence
alpha_95 = 0.05
alpha_99 = 0.01

# Using Student-t quantiles
df_final = fit_final.params['nu']
var_95 = mean_forecast + vol_forecast * stats.t.ppf(alpha_95, df_final)
var_99 = mean_forecast + vol_forecast * stats.t.ppf(alpha_99, df_final)

print(f"1-Day Ahead Forecast:")
print(f"  Mean: {mean_forecast:.4f}%")
print(f"  Volatility: {vol_forecast:.4f}%")

print(f"\nValue-at-Risk (VaR):")
print(f"  95% VaR: {var_95:.4f}% (5% chance of loss exceeding this)")
print(f"  99% VaR: {var_99:.4f}% (1% chance of loss exceeding this)")

# For $1M portfolio
portfolio_value = 1_000_000
var_95_dollar = portfolio_value * abs(var_95) / 100
var_99_dollar = portfolio_value * abs(var_99) / 100

print(f"\nFor ${portfolio_value:,.0f} portfolio:")
print(f"  95% VaR: ${var_95_dollar:,.0f}")
print(f"  99% VaR: ${var_99_dollar:,.0f}")

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Volatility Characteristics:")
print(f"   Persistence (α+β): {alpha + beta:.4f}")
if alpha + beta > 0.95:
    print("   → High persistence: Shocks decay slowly")
print(f"   Half-life: {half_life:.1f} days")
print(f"   Unconditional volatility: {np.sqrt(uncond_var):.2f}%")

print("\n2. Asymmetry:")
if gamma > 0 and gamma / gjr_fit.std_err['gamma[1]'] > 2:
    print(f"   ✓ Significant leverage effect (γ={gamma:.4f})")
    print("   → Negative shocks increase volatility more")
else:
    print("   No significant asymmetry detected")

print("\n3. Distribution:")
if df_param < 10:
    print(f"   ✓ Heavy tails confirmed (ν={df_param:.2f} < 10)")
    print("   → Student-t better than Gaussian")
else:
    print("   Tails close to normal")

print("\n4. Forecasting:")
print(f"   Best model: {best_model_name} (lowest AIC)")
if qlike_gjr < qlike_garch:
    print(f"   GJR-GARCH outperforms GARCH(1,1) out-of-sample")
else:
    print(f"   Standard GARCH(1,1) adequate")

print("\n5. Practical Applications:")
print("   • Risk management: VaR, Expected Shortfall")
print("   • Trading: Volatility targeting, statistical arbitrage")
print("   • Option pricing: GARCH option models")
print("   • Portfolio: Dynamic hedging, time-varying weights")

print("\n6. Limitations:")
print("   • Assumes continuous sampling (issues with intraday)")
print("   • May underestimate tail risk during crises")
print("   • Symmetric GARCH misses leverage effect")
print("   • Consider regime-switching for structural breaks")
```

## 6. Challenge Round
When do GARCH models fail or mislead?
- **Structural breaks**: COVID crash, financial crisis → Single-regime GARCH poor fit; regime-switching GARCH or stochastic volatility better
- **Intraday data**: Market microstructure noise, bid-ask bounce → Realized GARCH or HAR models; standard GARCH assumes continuous sampling
- **Long memory misspecified**: GARCH imposes exponential decay → FIGARCH if true hyperbolic decay; affects long-horizon forecasts
- **Jump processes**: Large discrete jumps (earnings, news) → GARCH treats as extreme diffusion; explicit jump models (Merton, Bates) better
- **Contemporaneous feedback**: Volatility affects returns (risk premium) → GARCH-in-Mean but endogeneity issues; need structural model
- **Zero returns**: Limit orders, illiquidity → Variance undefined; duration-adjusted or realized measures better

## 7. Key References
- [Engle (1982) - Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of UK Inflation](https://doi.org/10.2307/1912773)
- [Bollerslev (1986) - Generalized Autoregressive Conditional Heteroskedasticity](https://doi.org/10.1016/0304-4076(86)90063-1)
- [Hansen & Lunde (2005) - A Forecast Comparison of Volatility Models](https://doi.org/10.1198/073500104000000602)

---
**Status:** Industry standard for volatility modeling | **Complements:** Realized Volatility, Stochastic Volatility, Jump Models, HAR
