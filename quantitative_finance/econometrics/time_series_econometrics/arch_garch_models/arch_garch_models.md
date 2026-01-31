# ARCH/GARCH Models

## 1. Concept Skeleton
**Definition:** Autoregressive Conditional Heteroskedasticity models; variance of errors depends on past squared errors (volatility clustering)  
**Purpose:** Model time-varying volatility, capture fat tails in financial returns, quantify risk over time  
**Prerequisites:** Conditional heteroskedasticity, squared residuals autocorrelation, volatility persistence

## 2. Comparative Framing
| Model | OLS | ARCH(q) | GARCH(p,q) | Exponential GARCH |
|-------|-----|---------|------------|-------------------|
| **Mean** | Constant | Constant | Constant | Constant |
| **Variance** | Constant | Lags of ε² | Lags of σ² + ε² | Lags + leverage |
| **Asymmetry** | No | No | No | Yes (good/bad news) |
| **Persistence** | N/A | Low | High (α+β→1) | Asymmetric |
| **Tail Risk** | Underestimated | Partial | Good | Very good |

## 3. Examples + Counterexamples

**Simple Example:**  
Stock returns GARCH(1,1): High volatility today → next day volatility higher due to persistence coefficient α+β=0.95. Shock half-life ≈ 13 days

**Failure Case:**  
Assume constant volatility in crisis: 2008 financial crisis showed vol jumps from 15% to 60% → VaR models using constant vol massively underestimated risk

**Edge Case:**  
Leverage effect in equity returns: Negative returns increase volatility more than positive returns → EGARCH captures asymmetry, standard GARCH misses

## 4. Layer Breakdown
```
GARCH Model Architecture:
├─ Conditional Mean (Unchanged):
│   └─ Yₜ = μ + εₜ
├─ Conditional Variance (Time-Varying):
│   ├─ ARCH(q): σ²ₜ = ω + α₁ε²ₜ₋₁ + ... + αₑε²ₜ₋ₑ
│   │   ├─ Direct dependence on past shocks
│   │   ├─ ω > 0: Base volatility
│   │   ├─ αᵢ > 0: Shock persistence (news impact)
│   │   └─ Σαᵢ = persistence measure
│   ├─ GARCH(p,q): σ²ₜ = ω + Σαᵢε²ₜ₋ᵢ + Σβⱼσ²ₜ₋ⱼ
│   │   ├─ βⱼ: Lagged variance terms (smoothness)
│   │   ├─ Persistence: α + β (typically 0.80-0.99)
│   │   ├─ Half-life = ln(0.5)/ln(α+β) periods
│   │   └─ GARCH(1,1) usually sufficient
│   └─ Exponential GARCH: σ²ₜ = exp(ω + γ·εₜ₋₁/σₜ₋₁ + βln(σ²ₜ₋₁))
│       └─ Captures leverage: negative shocks → larger vol increase
├─ Error Distribution:
│   ├─ Normal: Assumes εₜ ~ N(0, σ²ₜ)
│   ├─ Student-t: Heavier tails, df estimated
│   └─ Skewed-t: Asymmetric downside risk
├─ Estimation (Maximum Likelihood):
│   ├─ Log-likelihood: Σ[-(1/2)ln(σ²ₜ) - ε²ₜ/(2σ²ₜ)]
│   ├─ Numerical optimization (gradient-based)
│   └─ Standard errors from Hessian matrix
└─ Forecasting:
    ├─ 1-step ahead: σ²ₜ₊₁ = ω + αε²ₜ + βσ²ₜ
    ├─ Multi-step: Recursive using E[ε²ₜ₊ₖ] = σ²ₜ₊ₖ
    └─ Long-run variance: ω/(1-α-β) (unconditional)
```

**Interaction:** Shocks → vol spike → mean reversion → clustering pattern

## 5. Mini-Project
Model stock returns with GARCH, estimate VaR and volatility forecasts:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic stock returns with volatility clustering
np.random.seed(42)
periods = 500

# GARCH(1,1) process: σ²ₜ = 0.00001 + 0.08·ε²ₜ₋₁ + 0.90·σ²ₜ₋₁
omega = 0.00001
alpha = 0.08
beta = 0.90

volatility = np.zeros(periods)
returns = np.zeros(periods)
volatility[0] = np.sqrt(omega / (1 - alpha - beta))  # Unconditional vol

for t in range(1, periods):
    volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2)
    returns[t] = volatility[t] * np.random.normal(0, 1)

dates = pd.date_range('2018-01', periods=periods, freq='D')
returns_series = pd.Series(returns, index=dates)
volatility_series = pd.Series(volatility, index=dates)

print("="*70)
print("GARCH VOLATILITY MODELING: Stock Returns")
print("="*70)

# 1. Descriptive statistics
print("\n1. RETURN STATISTICS")
print("-"*70)
print(f"Mean return: {returns.mean():.6f} ({returns.mean()*252:.4f} annualized)")
print(f"Std Dev: {returns.std():.6f} ({returns.std()*np.sqrt(252):.4f} annualized)")
print(f"Skewness: {pd.Series(returns).skew():.4f}")
print(f"Excess Kurtosis: {pd.Series(returns).kurtosis():.4f}")
print(f"Min return: {returns.min():.6f}")
print(f"Max return: {returns.max():.6f}")

# 2. Test for ARCH effects (squared residuals correlation)
print("\n2. ARCH TEST (Squared Residuals)")
print("-"*70)

residuals = returns - returns.mean()
residuals_squared = residuals**2

# Ljung-Box test on squared residuals
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals_squared, lags=10, return_df=True)
print("\nLjung-Box test on squared residuals:")
print(lb_test)
print("Significant autocorrelation indicates ARCH/GARCH needed")

# 3. Simple GARCH(1,1) estimation (manual)
print("\n3. GARCH(1,1) ESTIMATION")
print("-"*70)

def garch_loglik(params, returns):
    omega, alpha, beta = params
    
    # Constraints
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    loglik = -np.sum(np.log(sigma2) + returns**2 / sigma2) / 2
    return -loglik

# Initial guess
x0 = [1e-5, 0.1, 0.8]

# Optimize
result = minimize(garch_loglik, x0, args=(returns,), method='Nelder-Mead',
                  options={'maxiter': 5000})

omega_est, alpha_est, beta_est = result.x

print(f"\nEstimated GARCH(1,1) parameters:")
print(f"  ω (omega): {omega_est:.8f}")
print(f"  α (alpha): {alpha_est:.6f}")
print(f"  β (beta):  {beta_est:.6f}")
print(f"  Persistence (α+β): {alpha_est + beta_est:.6f}")

# Half-life of volatility shock
if alpha_est + beta_est < 1:
    half_life = np.log(0.5) / np.log(alpha_est + beta_est)
    print(f"  Half-life of shock: {half_life:.2f} days")
    print(f"  Unconditional volatility: {np.sqrt(omega_est / (1 - alpha_est - beta_est)):.6f}")

# 4. Calculate conditional volatility
print("\n4. CONDITIONAL VOLATILITY DYNAMICS")
print("-"*70)

sigma2_est = np.zeros(len(returns))
sigma2_est[0] = np.var(returns)

for t in range(1, len(returns)):
    sigma2_est[t] = omega_est + alpha_est * returns[t-1]**2 + beta_est * sigma2_est[t-1]

sigma_est = np.sqrt(sigma2_est)

print(f"Average conditional volatility: {sigma_est.mean():.6f}")
print(f"Min conditional volatility: {sigma_est.min():.6f}")
print(f"Max conditional volatility: {sigma_est.max():.6f}")

# 5. Value-at-Risk (VaR)
print("\n5. VALUE-AT-RISK (VaR) ESTIMATES")
print("-"*70)

# Assuming normal distribution
confidence_levels = [0.95, 0.99]

var_estimates = {}
for conf in confidence_levels:
    z_score = norm.ppf(1 - conf)
    var_1day = z_score * sigma_est[-1]
    var_1day_pct = z_score * sigma_est[-1] * 100
    var_estimates[conf] = var_1day_pct
    
    print(f"\n{int(conf*100)}% VaR (1-day), portfolio value = $1M:")
    print(f"  VaR: {var_1day_pct:.3f}%")
    print(f"  Expected loss (1 in {int(1/(1-conf))} days): ${var_1day*1e6:,.0f}")

# 6. Volatility forecasting
print("\n6. VOLATILITY FORECASTING")
print("-"*70)

forecast_steps = 20
sigma2_forecast = np.zeros(forecast_steps)
sigma2_forecast[0] = sigma2_est[-1]

for t in range(1, forecast_steps):
    # Multi-step forecast: E[σ²ₜ₊ₖ] converges to unconditional variance
    sigma2_forecast[t] = (omega_est + 
                          (alpha_est + beta_est) * sigma2_forecast[t-1])

# Terminal value (unconditional)
sigma2_terminal = omega_est / (1 - alpha_est - beta_est)
sigma_forecast = np.sqrt(sigma2_forecast)

print(f"\nVolatility Forecast (next {forecast_steps} days):")
print(f"  Today's conditional vol: {sigma_est[-1]:.6f}")
print(f"  {forecast_steps}-day forecast: {sigma_forecast[-1]:.6f}")
print(f"  Long-run (unconditional): {np.sqrt(sigma2_terminal):.6f}")
print(f"  Mean reversion speed: {(sigma_est[-1] - sigma_forecast[-1]):.6f}")

# 7. Visualizations
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: Returns over time
ax = axes[0, 0]
ax.plot(dates, returns, linewidth=1, alpha=0.7)
ax.set_title('Daily Returns')
ax.set_ylabel('Return')
ax.grid(alpha=0.3)

# Plot 2: Actual vs Fitted Volatility
ax = axes[0, 1]
ax.plot(dates, np.abs(returns), label='|Returns|', linewidth=1, alpha=0.5)
ax.plot(dates, sigma_est, label='Conditional Vol', linewidth=2, color='r')
ax.set_title('Volatility Estimation')
ax.set_ylabel('Volatility')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Squared residuals
ax = axes[1, 0]
ax.bar(dates, residuals_squared, width=1, alpha=0.6)
ax.set_title('Squared Residuals (Volatility Clustering)')
ax.set_ylabel('ε²ₜ')
ax.grid(alpha=0.3)

# Plot 4: ACF of squared residuals
from statsmodels.graphics.tsaplots import plot_acf
ax = axes[1, 1]
plot_acf(residuals_squared, lags=40, ax=ax)
ax.set_title('ACF of Squared Residuals')

# Plot 5: Volatility forecast
ax = axes[2, 0]
forecast_dates = pd.date_range(dates[-1], periods=forecast_steps+1, freq='D')[1:]
ax.plot(dates[-100:], sigma_est[-100:], 'b-', linewidth=2, label='Historical')
ax.plot(forecast_dates, sigma_forecast, 'r--', linewidth=2, marker='o', label='Forecast')
ax.axhline(y=np.sqrt(sigma2_terminal), color='g', linestyle=':', linewidth=2, 
           label='Long-run vol')
ax.set_title('Volatility Forecast')
ax.set_ylabel('Conditional Volatility')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Distribution of returns with GARCH-based confidence intervals
ax = axes[2, 1]
ax.hist(returns, bins=50, density=True, alpha=0.7, label='Returns')
mean_vol = sigma_est.mean()
x_range = np.linspace(-4*mean_vol, 4*mean_vol, 100)
ax.plot(x_range, norm.pdf(x_range, 0, mean_vol), 'r-', linewidth=2, label='Normal fit')
ax.set_title('Return Distribution (Normal)')
ax.set_xlabel('Return')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('garch_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print(f"""
Key GARCH Results:
1. Persistence (α+β = {alpha_est + beta_est:.4f}):
   → Volatility very persistent (close to 1)
   → Shocks take {half_life:.0f} days to half-dissipate
   
2. VaR Estimates:
   → 95% confidence: Maximum daily loss = {var_estimates[0.95]:.3f}%
   → 99% confidence: Maximum daily loss = {var_estimates[0.99]:.3f}%
   
3. Conditional volatility changes over time:
   → Clustering: High volatility followed by high volatility
   → Mean reversion: Extreme moves revert toward average
   
4. Forecast implications:
   → Near-term volatility: {sigma_forecast[0]:.6f}
   → Term structure: Vol mean reverts to long-run {np.sqrt(sigma2_terminal):.6f}
""")
```

## 6. Challenge Round
When GARCH models fail:
- Leverage effect ignored: EGARCH captures asymmetric volatility response
- Structural breaks: Pre-crisis vs post-crisis volatility regimes differ; use Markov-switching
- Multiple assets: Multivariate GARCH (BEKK, DCC) needed for correlations
- Thick tails: Student-t GARCH better than normal for extreme risk
- Computation complexity: High-dimensional DCC-GARCH requires numerical approximation

## 7. Key References
- [Engle (1982), "Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of UK Inflation"](https://www.jstor.org/stable/1912773)
- [Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity"](https://www.jstor.org/stable/2336144)
- [Nelson (1991), "Conditional Heteroskedasticity in Asset Returns: A New Approach"](https://www.jstor.org/stable/2109358)

---
**Status:** Essential for volatility modeling | **Complements:** VaR, Risk Management, ARIMA
