# Asset Return Properties

## 1. Concept Skeleton
**Definition:** Statistical characteristics of financial returns including fat tails, volatility clustering, leverage effects, and serial dependence  
**Purpose:** Model return distributions accurately for risk management, option pricing, and portfolio optimization beyond Gaussian assumptions  
**Prerequisites:** Time series analysis, moment statistics, distributional testing, ARCH/GARCH models

## 2. Comparative Framing
| Property | Normal Distribution | Empirical Returns | Impact | Detection Method |
|----------|-------------------|-------------------|--------|------------------|
| **Tail behavior** | Thin tails (exponential decay) | Fat tails (power law) | Underestimate rare events | Excess kurtosis, QQ-plots |
| **Volatility** | Constant σ | Time-varying, clustered | Mispriced options | ARCH test, ACF of squared returns |
| **Skewness** | Symmetric (skew=0) | Negative skew (left tail) | Downside risk higher | Skewness statistic, moment tests |
| **Autocorrelation** | None (iid) | Small positive (momentum), mean reversion | Predictability | Ljung-Box test, ACF plots |
| **Leverage effect** | N/A | Negative returns → higher vol | Asymmetric risk | EGARCH, correlation(r_t, σ²_{t+1}) |

## 3. Examples + Counterexamples

**Simple Example:**  
S&P 500 daily returns: Mean 0.05%, StDev 1.2%, Kurtosis 7.5 (vs 3 for normal); Oct 1987 crash -20% = 16σ event (Gaussian probability ~10⁻⁵⁸); fat tails observed

**Failure Case:**  
Black-Scholes assumes constant volatility; Oct 2008 VIX spikes from 20 to 80 (4× increase); options mispriced by 50%+; GARCH models capture volatility clustering better

**Edge Case:**  
High-frequency returns (1-minute): Microstructure noise dominates (bid-ask bounce); appears mean-reverting but spurious; must aggregate to 5-15 minute intervals

## 4. Layer Breakdown
```
Asset Return Properties Structure:
├─ Distributional Characteristics:
│   ├─ Fat Tails (Leptokurtosis):
│   │   ├─ Definition: Excess kurtosis κ_excess = κ - 3 > 0 (normal has κ=3)
│   │   │   ├─ Interpretation: More probability mass in tails than normal
│   │   │   ├─ Typical values: S&P 500 daily κ ≈ 6-8; intraday κ ≈ 10-20
│   │   │   └─ Consequence: 3σ+ events occur 10-100× more than Gaussian predicts
│   │   ├─ Measurement:
│   │   │   ├─ Sample kurtosis: κ̂ = (1/n)Σ[(r_i - μ)⁴] / σ⁴
│   │   │   ├─ Jarque-Bera test: JB = (n/6)[S² + (κ-3)²/4] ~ χ²(2)
│   │   │   │   ├─ H₀: Normality (S=0, κ=3)
│   │   │   │   ├─ Example: S&P 500 (1990-2020) JB = 15,000 (p<0.001; reject normality)
│   │   │   │   └─ Critical value: χ²(2, 0.05) = 5.99
│   │   │   └─ QQ-plot: Quantile-quantile vs normal; fat tails show points above/below line
│   │   ├─ Modeling approaches:
│   │   │   ├─ Student's t-distribution: t(ν) with ν degrees of freedom
│   │   │   │   ├─ Lower ν → fatter tails; ν=5-10 typical for daily returns
│   │   │   │   ├─ Kurtosis: κ = 3 + 6/(ν-4) for ν>4
│   │   │   │   └─ Example: ν=6 → κ=6 (matches empirical)
│   │   │   ├─ Generalized Error Distribution (GED): exp(-|r/σ|^ν)
│   │   │   │   ├─ ν<2: Fat tails; ν=2: Normal; ν>2: Thin tails
│   │   │   │   └─ Empirical: ν ≈ 1.2-1.5 for stocks
│   │   │   └─ Mixture of normals: w·N(μ₁,σ₁) + (1-w)·N(μ₂,σ₂)
│   │   │       ├─ Regime-switching interpretation: normal times + crisis
│   │   │       └─ Example: 95% quiet (σ=1%) + 5% turmoil (σ=5%)
│   │   ├─ Implications:
│   │   │   ├─ VaR underestimation: Normal VaR₉₅ = μ - 1.65σ
│   │   │   │   ├─ Actual losses exceed VaR 5%+ of time (vs 5% expected)
│   │   │   │   └─ Fix: Use t-distribution or extreme value theory
│   │   │   ├─ Option pricing: Black-Scholes underprices OTM puts
│   │   │   │   ├─ Volatility smile: Implied vol higher for low strikes
│   │   │   │   └─ Jump-diffusion models (Merton 1976) add rare large moves
│   │   │   └─ Portfolio optimization: Mean-variance inadequate
│   │   │       ├─ Downside risk measures: semivariance, CVaR
│   │   │       └─ Higher moments: skewness preference, kurtosis aversion
│   │   └─ Historical examples:
│   │       ├─ Black Monday (Oct 19, 1987): S&P -20% = 16σ (Gaussian prob 10⁻⁵⁸)
│   │       ├─ LTCM (Aug 1998): 10σ+ moves multiple days (shouldn't happen in universe lifetime)
│   │       └─ Flash Crash (May 6, 2010): Intraday -9% in 5 min (massive tail event)
│   ├─ Negative Skewness:
│   │   ├─ Definition: Skewness S = E[(r-μ)³]/σ³; S<0 indicates left tail longer
│   │   │   ├─ Interpretation: Large negative returns more common than large positive
│   │   │   ├─ Typical: S ≈ -0.5 to -1.5 for equity indices
│   │   │   └─ Asymmetry: Crashes more severe than rallies
│   │   ├─ Measurement:
│   │   │   ├─ Sample skewness: Ŝ = (1/n)Σ[(r_i-μ̂)³] / σ̂³
│   │   │   ├─ Test: z = Ŝ / √(6/n) ~ N(0,1) under H₀: S=0
│   │   │   └─ Example: S&P 500 Ŝ = -0.8; z = -6.3 (p<0.001; significantly negative)
│   │   ├─ Sources:
│   │   │   ├─ Leverage effect: Negative returns → higher leverage → higher volatility
│   │   │   ├─ Volatility feedback: Vol increases → discount rate up → prices down
│   │   │   └─ Risk premium time-variation: Crashes occur when risk aversion spikes
│   │   ├─ Asset class differences:
│   │   │   ├─ Equity indices: S ≈ -0.5 to -1.0 (negative skew strong)
│   │   │   ├─ Individual stocks: S ≈ 0 to +0.5 (slightly positive; lottery effect)
│   │   │   ├─ Currencies: S ≈ 0 (symmetric; carry trades exception)
│   │   │   └─ Commodities: S ≈ +0.3 (supply shocks create positive skew)
│   │   └─ Trading implications:
│   │       ├─ Put options more valuable: Downside protection demand
│   │       ├─ Volatility smile: Implied vol higher for low-strike puts
│   │       └─ Portfolio insurance: Tail risk hedging via puts, VIX calls
│   └─ Higher Moment Evolution:
│       ├─ Time-varying skewness: EGARCH captures asymmetry
│       ├─ Time-varying kurtosis: Markov-switching models (regime shifts)
│       └─ Co-skewness/co-kurtosis: Cross-asset tail dependence
├─ Volatility Clustering:
│   ├─ Definition: Large returns tend to follow large returns (any sign)
│   │   ├─ Observation: |r_t| large → E[|r_{t+1}|] elevated
│   │   ├─ Persistence: Volatility shocks decay slowly (half-life days to weeks)
│   │   └─ Mandelbrot (1963): "Large changes tend to be followed by large changes"
│   ├─ Detection:
│   │   ├─ ACF of returns: ρ(r_t, r_{t-k}) ≈ 0 (no linear autocorrelation)
│   │   ├─ ACF of squared returns: ρ(r²_t, r²_{t-k}) > 0 (significant for k=1-20)
│   │   │   ├─ Example: S&P 500 ρ(r²,lag1) ≈ 0.15 (15% correlation)
│   │   │   └─ Decay: ρ(r²,lag k) ~ k^(-α) with α ≈ 0.3 (slow hyperbolic decay)
│   │   └─ ARCH test (Engle 1982):
│   │       ├─ Regression: r²_t = β₀ + β₁r²_{t-1} + ... + β_q r²_{t-q} + ε_t
│   │       ├─ Test: LM = n·R² ~ χ²(q) under H₀: β₁=...=β_q=0
│   │       └─ Reject H₀ → volatility clustering present
│   ├─ GARCH Modeling:
│   │   ├─ GARCH(1,1): σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
│   │   │   ├─ Interpretation: Today's vol depends on yesterday's shock + vol
│   │   │   ├─ Persistence: α+β ≈ 0.98-0.99 (near unit root; long memory)
│   │   │   └─ Unconditional: σ² = ω/(1-α-β) provided α+β<1
│   │   ├─ Parameter estimates (typical):
│   │   │   ├─ ω ≈ 0.01 (base volatility)
│   │   │   ├─ α ≈ 0.08 (shock impact; ARCH term)
│   │   │   ├─ β ≈ 0.90 (persistence; GARCH term)
│   │   │   └─ Half-life: log(0.5)/log(α+β) ≈ 70 days (slow mean reversion)
│   │   ├─ Extensions:
│   │   │   ├─ EGARCH: Captures leverage effect (negative returns → higher vol)
│   │   │   ├─ GJR-GARCH: Asymmetric response to positive/negative shocks
│   │   │   └─ FIGARCH: Fractional integration; very long memory
│   │   └─ Applications:
│   │       ├─ Option pricing: Stochastic volatility models
│   │       ├─ Risk management: Conditional VaR (vs unconditional)
│   │       └─ Portfolio allocation: Time-varying covariances
│   ├─ Economic mechanisms:
│   │   ├─ Information arrival: News clusters → return volatility clusters
│   │   ├─ Trading volume: High volume days → high volatility
│   │   └─ Market microstructure: Liquidity withdrawal amplifies vol
│   └─ Empirical patterns:
│       ├─ Intraday: U-shape volatility (high at open/close; low midday)
│       ├─ Weekly: Monday higher vol (weekend news accumulation)
│       └─ Seasonal: September-October higher vol (crash anniversary?)
├─ Leverage Effect (Volatility Asymmetry):
│   ├─ Definition: Negative returns → future volatility increase more than positive returns
│   │   ├─ Observation: Corr(r_t, σ²_{t+1}) < 0 (typically -0.5 to -0.7)
│   │   ├─ Asymmetry: -5% return → vol increases 2×; +5% return → vol unchanged
│   │   └─ Black (1976): Leverage explanation (debt/equity ratio rises when equity falls)
│   ├─ Measurement:
│   │   ├─ News impact curve: Plot σ²_{t+1} vs r_t
│   │   │   ├─ Symmetric (GARCH): Parabola centered at r_t=0
│   │   │   ├─ Asymmetric (EGARCH): Steeper for r_t<0
│   │   │   └─ Quantify: Slope(r_t<0) / Slope(r_t>0) ≈ 1.5-2.0
│   │   ├─ EGARCH specification:
│   │   │   ├─ log(σ²_t) = ω + α[|z_{t-1}| + γz_{t-1}] + β·log(σ²_{t-1})
│   │   │   ├─ z_t = r_t/σ_t (standardized return)
│   │   │   ├─ γ<0 → leverage effect (negative shocks increase vol more)
│   │   │   └─ Typical: γ ≈ -0.15 (significant asymmetry)
│   │   └─ Alternative: GJR-GARCH with indicator I(r_t<0)
│   ├─ Economic explanations:
│   │   ├─ Leverage hypothesis (Black 1976):
│   │   │   ├─ Stock price drops → debt/equity ratio rises
│   │   │   ├─ Higher leverage → higher equity risk
│   │   │   └─ Critique: Effect too large; immediate (should take time)
│   │   ├─ Volatility feedback (French et al. 1987):
│   │   │   ├─ Vol increases → expected return rises (risk premium)
│   │   │   ├─ Higher discount rate → stock price falls
│   │   │   └─ Creates negative correlation between returns and future vol
│   │   └─ Behavioral: Downside loss aversion → panic selling → vol spike
│   ├─ Empirical evidence:
│   │   ├─ Equity indices: Strong leverage effect (γ ≈ -0.15)
│   │   ├─ Individual stocks: Weaker but present (γ ≈ -0.08)
│   │   ├─ FX markets: Minimal leverage effect (γ ≈ 0)
│   │   └─ Commodities: Sometimes reversed (supply shocks positive)
│   └─ Implications:
│       ├─ Option pricing: Asymmetric volatility smile (skew)
│       ├─ Risk management: Downside VaR higher than upside
│       └─ Trading: Sell volatility after rallies; buy after crashes
├─ Serial Dependence:
│   ├─ Momentum (Short-term):
│   │   ├─ Definition: Positive autocorrelation at short lags (1-12 months)
│   │   │   ├─ Winners continue winning; losers continue losing
│   │   │   ├─ Magnitude: ρ(r_t, r_{t-1}) ≈ +0.05 to +0.15 (monthly)
│   │   │   └─ Jegadeesh & Titman (1993): 12-month momentum earns 1% monthly
│   │   ├─ Mechanisms:
│   │   │   ├─ Underreaction: Investors slow to incorporate news
│   │   │   ├─ Herding: Following others' trades amplifies trends
│   │   │   └─ Risk premium variation: Time-varying expected returns
│   │   ├─ Testing:
│   │   │   ├─ Ljung-Box: Q(m) = n(n+2)Σ[ρ̂²_k/(n-k)] ~ χ²(m)
│   │   │   ├─ Example: S&P 500 monthly Q(12) = 25.4 (p=0.01; reject iid)
│   │   │   └─ Runs test: Count sequences of same-sign returns
│   │   └─ Trading:
│   │       ├─ Momentum strategy: Long winners, short losers
│   │       ├─ Typical: 12-month formation, 1-month holding
│   │       └─ Caveat: High turnover costs; crashes (2009)
│   ├─ Mean Reversion (Long-term):
│   │   ├─ Definition: Negative autocorrelation at long lags (3-5 years)
│   │   │   ├─ Extreme moves tend to reverse over years
│   │   │   ├─ Magnitude: ρ(r_t, r_{t-36}) ≈ -0.3 (significant)
│   │   │   └─ Fama & French (1988): 25-40% of variance mean-reverting
│   │   ├─ Mechanisms:
│   │   │   ├─ Overreaction: Bubbles burst, panics recover
│   │   │   ├─ Time-varying discount rates: Valuation ratios revert
│   │   │   └─ Economic cycles: Profit margins mean-revert
│   │   ├─ Evidence:
│   │   │   ├─ Variance ratio: VR(k) = Var(r_t,...,r_{t-k+1})/[k·Var(r_t)]
│   │   │   ├─ Random walk: VR(k)=1; Mean reversion: VR(k)<1
│   │   │   └─ Empirical: VR(60 months) ≈ 0.5-0.7 (significant reversion)
│   │   └─ Trading:
│   │       ├─ Contrarian strategies: Buy losers, sell winners (3-5 yr horizon)
│       ├─ Value investing: Low P/E outperforms over long run
│       └─ Caveat: Long holding periods; survivorship bias
│   └─ Microstructure effects:
│       ├─ Bid-ask bounce: Negative autocorrelation at tick frequency
│       ├─ Nonsynchronous trading: Stale prices induce autocorrelation
│       └─ Price discreteness: Rounding creates spurious patterns
└─ Cross-Sectional Properties:
    ├─ Cross-Correlation Time Variation:
    │   ├─ Normal times: ρ(stock i, stock j) ≈ 0.3
    │   ├─ Crisis times: ρ increases to 0.7-0.9 (correlation breakdown)
    │   └─ Implication: Diversification fails when needed most
    ├─ Common Factors:
    │   ├─ Market factor: Explains 20-40% of individual stock variance
    │   ├─ Sector factors: Additional 10-20% explained
    │   └─ Idiosyncratic: 40-60% stock-specific
    └─ Tail Dependence:
        ├─ Copula models: Separate marginal distributions from dependence
        ├─ Lower tail dependence: Crashes more correlated than rallies
        └─ Measurement: Conditional correlation during extreme events
```

**Key Insight:** Asset returns deviate systematically from normality (fat tails, volatility clustering, negative skew, leverage effects); Gaussian models underestimate risk; GARCH and higher-moment frameworks essential

## 5. Mini-Project
Analyze S&P 500 return properties and test for non-normality:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model

# Simulate return data (substitute with real data in practice)
np.random.seed(42)
n = 2500  # ~10 years daily

# Generate GARCH(1,1) process with fat tails and leverage effect
from arch.univariate import GARCH, Normal, StudentsT

# Specify GARCH model
garch = arch_model(None, vol='GARCH', p=1, q=1, dist='t')
params = np.array([0.01, 0.08, 0.90])  # omega, alpha, beta (typical values)

# Simulate
sim_data = garch.simulate(params, n, x=None)
returns = sim_data['data'].values * 0.01  # Scale to ~1% daily vol

# Add negative skew (simulate leverage effect)
returns = returns - 0.3 * (returns < 0) * returns**2

# Calculate statistics
mean_ret = returns.mean() * 252 * 100  # Annualized %
std_ret = returns.std() * np.sqrt(252) * 100  # Annualized %
skew = stats.skew(returns)
kurt = stats.kurtosis(returns)  # Excess kurtosis
kurt_total = kurt + 3

print("="*70)
print("Asset Return Properties Analysis")
print("="*70)
print(f"Sample size: {n} observations (~{n/252:.1f} years daily)")
print(f"\nDistributional Statistics:")
print(f"  Mean (annualized): {mean_ret:>8.2f}%")
print(f"  Std Dev (annualized): {std_ret:>8.2f}%")
print(f"  Skewness: {skew:>8.3f} (Normal: 0)")
print(f"  Excess Kurtosis: {kurt:>8.3f} (Normal: 0)")
print(f"  Total Kurtosis: {kurt_total:>8.3f} (Normal: 3)")

# Jarque-Bera test
jb_stat, jb_pval = stats.jarque_bera(returns)
print(f"\nJarque-Bera Test:")
print(f"  JB statistic: {jb_stat:>8.2f}")
print(f"  p-value: {jb_pval:>8.6f}")
print(f"  Result: {'REJECT normality' if jb_pval < 0.05 else 'Cannot reject normality'}")

# Tail events
threshold_3sigma = 3 * returns.std()
threshold_5sigma = 5 * returns.std()

actual_3sigma = np.sum(np.abs(returns) > threshold_3sigma)
expected_3sigma = n * 2 * stats.norm.sf(3)  # Two tails

actual_5sigma = np.sum(np.abs(returns) > threshold_5sigma)
expected_5sigma = n * 2 * stats.norm.sf(5)

print(f"\nTail Event Analysis:")
print(f"  3-sigma events: Actual={actual_3sigma}, Expected(Normal)={expected_3sigma:.1f}")
print(f"  Ratio: {actual_3sigma/expected_3sigma:.1f}× (Fat tails indicator)")
print(f"  5-sigma events: Actual={actual_5sigma}, Expected(Normal)={expected_5sigma:.3f}")
if expected_5sigma > 0:
    print(f"  Ratio: {actual_5sigma/expected_5sigma:.0f}× (Extreme fat tails)")

# Volatility clustering test
returns_squared = returns**2
lb_stat, lb_pval = stats.diagnostic.acorr_ljungbox(returns_squared, lags=[10], return_df=False)
print(f"\nVolatility Clustering (ARCH Effects):")
print(f"  Ljung-Box Q(10) on r²: {lb_stat[0]:>8.2f}")
print(f"  p-value: {lb_pval[0]:>8.6f}")
print(f"  Result: {'Volatility clustering DETECTED' if lb_pval[0] < 0.05 else 'No clustering'}")

# Leverage effect
# Compute correlation between returns and future squared returns
leverage_corr = np.corrcoef(returns[:-1], returns_squared[1:])[0, 1]
print(f"\nLeverage Effect:")
print(f"  Corr(r_t, σ²_{{t+1}}): {leverage_corr:>8.3f}")
print(f"  Result: {'Negative leverage effect' if leverage_corr < -0.1 else 'No leverage effect'}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Return time series
axes[0, 0].plot(returns, linewidth=0.5, color='blue', alpha=0.7)
axes[0, 0].set_title('Return Time Series')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Return')
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[0, 0].grid(alpha=0.3)

# 2. Distribution vs Normal
axes[0, 1].hist(returns, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
x_range = np.linspace(returns.min(), returns.max(), 100)
normal_pdf = stats.norm.pdf(x_range, returns.mean(), returns.std())
t_pdf = stats.t.pdf(x_range, df=6, loc=returns.mean(), scale=returns.std())
axes[0, 1].plot(x_range, normal_pdf, 'r-', linewidth=2, label='Normal')
axes[0, 1].plot(x_range, t_pdf, 'g--', linewidth=2, label='Student-t (df=6)')
axes[0, 1].set_title('Distribution vs Normal')
axes[0, 1].set_xlabel('Return')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. QQ-plot
stats.probplot(returns, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('QQ-Plot vs Normal')
axes[0, 2].grid(alpha=0.3)

# 4. ACF of returns
plot_acf(returns, lags=20, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title('ACF of Returns (Linear Dependence)')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')

# 5. ACF of squared returns (volatility clustering)
plot_acf(returns_squared, lags=20, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('ACF of Squared Returns (Vol Clustering)')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('Autocorrelation')

# 6. Leverage effect scatter
axes[1, 2].scatter(returns[:-1], returns_squared[1:], alpha=0.3, s=10)
axes[1, 2].set_title(f'Leverage Effect (Corr={leverage_corr:.3f})')
axes[1, 2].set_xlabel('Return at t')
axes[1, 2].set_ylabel('Squared Return at t+1')
axes[1, 2].axvline(0, color='black', linestyle='--', linewidth=0.8)
axes[1, 2].grid(alpha=0.3)

# Add regression line
z = np.polyfit(returns[:-1], returns_squared[1:], 1)
p = np.poly1d(z)
x_line = np.linspace(returns.min(), returns.max(), 100)
axes[1, 2].plot(x_line, p(x_line), "r-", linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.5f}')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('asset_return_properties.png', dpi=300, bbox_inches='tight')
plt.show()

# Fit GARCH model
print(f"\n{'='*70}")
print("GARCH(1,1) Model Estimation")
print(f"{'='*70}")

model = arch_model(returns*100, vol='GARCH', p=1, q=1, dist='StudentsT')  # Scale up for numerical stability
results = model.fit(disp='off')
print(results.summary())

# Extract parameters
omega = results.params['omega']
alpha = results.params['alpha[1]']
beta = results.params['beta[1]']
persistence = alpha + beta
half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

print(f"\nGARCH Interpretation:")
print(f"  Persistence (α+β): {persistence:.4f}")
print(f"  Half-life: {half_life:.1f} days" if persistence < 1 else "  Half-life: Infinite (unit root)")
print(f"  Unconditional variance: {omega/(1-persistence):.4f}" if persistence < 1 else "  Unconditional variance: Undefined")
```

## 6. Challenge Round
When stylized facts mislead or fail:
- **Sample period bias**: 1990-2020 shows negative skew; 1950-1990 less clear; post-WWII data may not apply to 2030s (regime shifts)
- **Survivorship bias**: Indices exclude bankrupt firms; actual skewness more negative (missing left tail)
- **High-frequency data**: Bid-ask bounce creates spurious negative autocorrelation; microstructure noise dominates signal below 5-minute intervals
- **GARCH overfit**: Adding parameters improves in-sample fit; out-of-sample forecasts no better than constant volatility (overfitting vs parsimony)
- **Structural breaks**: COVID-19 March 2020 broke GARCH models calibrated on 2010-2019; single regime assumption fails
- **Cross-asset differences**: Equity properties don't apply to FX (symmetric), commodities (positive skew from supply shocks), or bonds (different volatility drivers)

## 7. Key References
- [Campbell, Lo & MacKinlay: Econometrics of Financial Markets (1997)](https://press.princeton.edu/books/hardcover/9780691043012/the-econometrics-of-financial-markets) - Comprehensive stylized facts
- [Cont: Empirical Properties of Asset Returns (2001)](http://www.cmap.polytechnique.fr/~rama/papers/empirical.pdf) - Quantitative Finance survey
- [Engle: Autoregressive Conditional Heteroskedasticity (1982)](https://www.jstor.org/stable/1912773) - Nobel Prize ARCH paper

---
**Status:** Core empirical finance | **Complements:** GARCH Models, Risk Management, Option Pricing, Portfolio Optimization
