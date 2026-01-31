# Risk Management Models

## 1. Concept Skeleton
**Definition:** Quantify downside risk exposure using Value-at-Risk (VaR), Expected Shortfall (CVaR), and stress testing to measure potential losses  
**Purpose:** Capital allocation, regulatory compliance (Basel III), risk limits, portfolio hedging decisions  
**Prerequisites:** Quantile estimation, tail risk, fat-tailed distributions, backtesting methodology

## 2. Comparative Framing
| Method | Definition | Tail Sensitivity | Subadditivity | Computational Cost | Basel Approved |
|--------|------------|------------------|---------------|-------------------|----------------|
| **Parametric VaR** | Assume Normal; α-quantile | Low (underestimates) | Yes | Very low | Yes (Pillar 1) |
| **Historical VaR** | Empirical quantile | Medium (data-dependent) | No | Low | Yes (Pillar 1) |
| **Monte Carlo VaR** | Simulate paths; quantile | High (flexible distributions) | Yes (if proper model) | High | Yes (Pillar 1) |
| **Expected Shortfall** | Mean loss beyond VaR | Very high (tail-aware) | Yes | Same as VaR | Yes (Basel III shift) |

## 3. Examples + Counterexamples

**Simple Example:**  
$10M portfolio; 1-day 99% VaR = $300K (1% chance loss exceeds $300K tomorrow); CVaR = $450K (expected loss if in worst 1%)

**Failure Case:**  
Oct 2008: 99% VaR (based on 2005-2007 data) = $500K; actual loss $2M (4× VaR); model assumed σ=1.5%, realized σ=8% (underestimated tail)

**Edge Case:**  
Heavy-tailed portfolio: VaR₉₉=100, VaR₉₉.₅=200 (huge jump); CVaR₉₉=180 (captures tail); VaR misses extreme loss concentration

## 4. Layer Breakdown
```
Risk Management Models Structure:
├─ Value-at-Risk (VaR):
│   ├─ Definition: Quantile of loss distribution
│   │   ├─ VaR_α = inf{x : P(Loss > x) ≤ 1-α}
│   │   ├─ α: Confidence level (typical 95%, 99%, 99.9%)
│   │   ├─ Interpretation: Maximum loss at α confidence over horizon h
│   │   └─ Example: VaR₉₉ = $1M → 1% chance loss exceeds $1M
│   ├─ Parametric VaR (Variance-Covariance):
│   │   ├─ Assumption: Portfolio returns ~ N(μ, σ²)
│   │   ├─ Formula: VaR_α = -μ + z_α·σ where z_α = Φ⁻¹(α)
│   │   │   ├─ 95% VaR: z₀.₉₅ = 1.645 → VaR = -μ + 1.645σ
│   │   │   ├─ 99% VaR: z₀.₉₉ = 2.326 → VaR = -μ + 2.326σ
│   │   │   └─ Example: μ=0.001, σ=0.02 (daily) → VaR₉₉ = -0.001 + 2.326×0.02 = 0.0455 (4.55%)
│   │   ├─ Portfolio extension:
│   │   │   ├─ w: Portfolio weights (n×1)
│   │   │   ├─ μ_p = w'μ, σ²_p = w'Σw
│   │   │   ├─ VaR_α = -(w'μ) + z_α·√(w'Σw)
│   │   │   └─ Scaling: h-day VaR ≈ 1-day VaR × √h (assumes iid)
│   │   ├─ Advantages:
│   │   │   ├─ Fast: Closed-form; no simulation
│   │   │   ├─ Stable: Parameters estimated from full sample
│   │   │   └─ Decomposable: Marginal VaR, component VaR
│   │   └─ Limitations:
│   │       ├─ Normality assumption: Underestimates fat tails (kurtosis>3)
│   │       ├─ Linear approximation: Ignores options, nonlinear instruments
│   │       └─ Example: True kurtosis=7 → 99% VaR underestimated by 30-50%
│   ├─ Historical Simulation VaR:
│   │   ├─ Method: Use empirical return distribution
│   │   │   ├─ Collect T historical returns: r₁, r₂, ..., r_T
│   │   │   ├─ Sort: r₍₁₎ ≤ r₍₂₎ ≤ ... ≤ r₍ₜ₎ (order statistics)
│   │   │   ├─ VaR_α = -r₍₍₁₋α₎T₎₎ (empirical quantile)
│   │   │   └─ Example: T=250, α=99% → VaR₉₉ = -r₍₂₎ (2nd worst day)
│   │   ├─ Portfolio application:
│   │   │   ├─ Current portfolio w
│   │   │   ├─ Historical returns matrix R (T×n)
│   │   │   ├─ Portfolio returns: p_t = Σ w_i·R_{t,i} for t=1..T
│   │   │   ├─ Sort p_t; take (1-α)×T quantile
│   │   │   └─ Example: $10M portfolio; 250 days; 99% → 3rd worst loss
│   │   ├─ Advantages:
│   │   │   ├─ No distributional assumptions
│   │   │   ├─ Captures fat tails, skewness (if present in data)
│   │   │   └─ Easy to implement
│   │   └─ Limitations:
│   │       ├─ Data-limited: (1-α)×T observations in tail (99%, T=250 → 2.5 obs)
│   │       ├─ Stale: Old data may not reflect current regime
│   │       ├─ Ghost effects: Large historical loss (e.g., 2008) dominates for years
│   │       └─ Non-stationarity: Volatility clustering not modeled
│   ├─ Monte Carlo VaR:
│   │   ├─ Procedure:
│   │   │   ├─ Specify model: r_t = μ + ε_t where ε_t ~ distribution (t, GED, etc.)
│   │   │   ├─ Estimate parameters: μ̂, σ̂, ν̂ (degrees of freedom for t)
│   │   │   ├─ Simulate N paths: r^(1), r^(2), ..., r^(N) ~ model
│   │   │   ├─ Calculate portfolio returns: p^(i) = w'r^(i)
│   │   │   ├─ Sort p^(i); VaR_α = -(1-α)×N quantile
│   │   │   └─ Example: N=10,000 simulations; α=99% → 100th worst draw
│   │   ├─ Advanced: GARCH + t-distribution
│   │   │   ├─ σ²_t = ω + αε²_{t-1} + βσ²_{t-1}
│   │   │   ├─ ε_t ~ t_ν (Student-t with ν≈5-10 df)
│   │   │   ├─ Captures: Vol clustering + fat tails
│   │   │   └─ Simulation: Bootstrap historical residuals or draw from fitted t
│   │   ├─ Advantages:
│   │   │   ├─ Flexible: Any distribution, dynamics (GARCH, jumps)
│   │   │   ├─ Tail modeling: Student-t, GED fit empirical tails better
│   │   │   └─ Complex portfolios: Options, path-dependent derivatives
│   │   └─ Limitations:
│   │       ├─ Model risk: Wrong distribution → biased VaR
│   │       ├─ Estimation error: Parameter uncertainty propagates
│   │       └─ Computational cost: Thousands of simulations (slow for large portfolios)
│   ├─ Properties of VaR:
│   │   ├─ Monotonicity: More risk → higher VaR ✓
│   │   ├─ Translation invariance: Add cash → VaR decreases by cash ✓
│   │   ├─ Homogeneity: Scale portfolio → VaR scales ✓
│   │   └─ Subadditivity: VaR(X+Y) ≤ VaR(X) + VaR(Y)? ✗ (FAILS)
│   │       ├─ Counterexample: Two independent binary assets
│   │       │   ├─ Asset A: Loss 10 with prob 1%, 0 otherwise → VaR₉₅(A) = 0
│   │       │   ├─ Asset B: Same → VaR₉₅(B) = 0
│   │       │   ├─ Portfolio A+B: Loss 10 with prob 2% → VaR₉₅(A+B) = 0
│   │       │   └─ BUT if threshold α=98%: VaR₉₈(A+B)=10 > VaR₉₈(A)+VaR₉₈(B)=0 (penalizes diversification!)
│   │       └─ Implication: Not a coherent risk measure (Artzner et al. 1999)
│   └─ Regulatory Use (Basel III):
│       ├─ Market risk capital: MRC = max(VaR_{t-1}, m_c × VaR_avg,60d) × multiplier
│       │   ├─ m_c: Multiplier (min 3.0; penalized if backtests fail)
│       │   ├─ VaR: 99%, 10-day horizon
│       │   └─ Stressed VaR: Additional charge using crisis period (2008)
│       ├─ Internal models approach:
│       │   ├─ Banks develop own VaR models
│       │   ├─ Backtesting: Must pass regulatory tests (exceptions <10/year)
│       │   └─ Approval: Requires 3+ years data, daily updates
│       └─ Standardized approach (alternative):
│           ├─ Risk-weighted assets (RWA) by asset class
│           └─ Simpler but more conservative (higher capital)
├─ Expected Shortfall (CVaR, Conditional VaR):
│   ├─ Definition: Expected loss given loss exceeds VaR
│   │   ├─ ES_α = E[Loss | Loss > VaR_α]
│   │   ├─ Tail conditional expectation: Average of worst (1-α) outcomes
│   │   └─ Example: VaR₉₉=$1M, ES₉₉=$1.5M → If exceed VaR, expect $1.5M loss
│   ├─ Parametric ES (Normal):
│   │   ├─ Formula: ES_α = μ + σ · φ(z_α)/(1-α)
│   │   │   ├─ φ: Standard normal PDF
│   │   │   ├─ z_α: α-quantile
│   │   │   └─ Example: α=99%, z=2.326 → ES = μ + 2.665σ (vs VaR = μ + 2.326σ)
│   │   └─ ES/VaR ratio: ~1.14 for Normal (increases with fat tails)
│   ├─ Historical/MC ES:
│   │   ├─ Sort returns: r₍₁₎ ≤ ... ≤ r₍ₜ₎
│   │   ├─ Identify tail: Observations ≤ VaR_α
│   │   ├─ Average: ES_α = (1/k)Σ r₍ᵢ₎ for i=1..k where k=(1-α)T
│   │   └─ Example: T=250, α=99% → Average worst 2-3 days
│   ├─ Properties:
│   │   ├─ Coherent risk measure ✓ (satisfies all 4 axioms):
│   │   │   ├─ Monotonicity ✓, Translation invariance ✓
│   │   │   ├─ Homogeneity ✓, Subadditivity ✓
│   │   │   └─ Implication: Encourages diversification
│   │   ├─ Tail-sensitive: Accounts for severity beyond VaR
│   │   └─ Elicitable: Cannot be backtested as easily as VaR (statistical issue)
│   ├─ Basel III shift (2016):
│   │   ├─ ES replaces VaR for market risk (internal models)
│   │   ├─ 97.5% ES, 10-day horizon (liquidity-adjusted)
│   │   └─ Rationale: ES more conservative; captures tail risk better
│   └─ Example comparison:
│       ├─ Normal: VaR₉₉=2.33σ, ES₉₉=2.67σ (ES 15% higher)
│       ├─ Student-t (ν=5): VaR₉₉=3.36σ, ES₉₉=4.73σ (ES 41% higher)
│       └─ Fat tails: ES/VaR ratio increases (ES more sensitive)
├─ Backtesting:
│   ├─ Purpose: Validate VaR model accuracy
│   │   ├─ Track exceptions: Days where Loss > VaR
│   │   ├─ Expected: (1-α) × T exceptions (99% VaR, T=250 → 2.5 exceptions/year)
│   │   ├─ Test: Is observed exceptions consistent with model?
│   │   └─ Regulatory: Basel traffic light (green <5, yellow 5-9, red ≥10 exceptions)
│   ├─ Kupiec Test (1995):
│   │   ├─ Null hypothesis: Exception rate = 1-α (model correct)
│   │   ├─ Test statistic: LR = -2ln[(1-p₀)^(T-N) · p₀^N / (1-p̂)^(T-N) · p̂^N]
│   │   │   ├─ p₀ = 1-α (theoretical exception rate)
│   │   │   ├─ p̂ = N/T (observed exception rate)
│   │   │   ├─ N: Number of exceptions observed
│   │   │   └─ T: Sample size
│   │   ├─ Distribution: LR ~ χ²(1) under H₀
│   │   ├─ Example: α=99%, T=250, N=7 exceptions
│   │   │   ├─ p₀=0.01, p̂=7/250=0.028
│   │   │   ├─ LR = -2ln[(0.99^243·0.01^7)/(0.972^243·0.028^7)] = 9.6
│   │   │   ├─ χ²(1, 0.05) = 3.84 (critical value)
│   │   │   └─ Reject H₀: Model underestimates risk (too many exceptions)
│   │   └─ Limitations:
│   │       ├─ Low power: Small T → hard to detect bad models
│   │       └─ Unconditional: Ignores clustering (multiple exceptions in short period)
│   ├─ Christoffersen Test (1998):
│   │   ├─ Tests independence + correct coverage
│   │   │   ├─ Correct unconditional coverage: p̂ = 1-α (Kupiec test)
│   │   │   ├─ Independence: Exceptions not clustered
│   │   │   └─ Joint test: LR_cc = LR_uc + LR_ind ~ χ²(2)
│   │   ├─ Independence test:
│   │   │   ├─ Transition matrix: P(exception_t | exception_{t-1})
│   │   │   ├─ H₀: P(E_t|E_{t-1}) = P(E_t|no E_{t-1}) = 1-α
│   │   │   └─ Clustering: Sequential exceptions → reject independence
│   │   ├─ Example: 7 exceptions, 3 consecutive pairs
│   │   │   ├─ Expected pairs: 0.01² × 249 ≈ 0.02 (very rare)
│   │   │   ├─ Observed: 3 pairs (clustering)
│   │   │   └─ Reject: Model fails to capture volatility clustering
│   │   └─ Advantage: Detects model deficiencies (GARCH needed if clustering)
│   ├─ Traffic Light Approach (Basel):
│   │   ├─ Green zone: 0-4 exceptions (250 days, 99% VaR) → No penalty
│   │   ├─ Yellow zone: 5-9 exceptions → Multiplier increases (3.0 → 3.4-3.85)
│   │   ├─ Red zone: ≥10 exceptions → Model rejected; use standardized approach
│   │   └─ Consequences: Higher capital requirements (up to 28% increase)
│   └─ ES backtesting challenges:
│       ├─ Non-elicitable: No scoring rule for conditional expectation
│       ├─ Approaches:
│       │   ├─ VaR-based: Backtest VaR at lower confidence (e.g., 97.5%)
│       │   ├─ Regression: Test E[L_t | L_t>VaR_t] ≈ ES_t
│       │   └─ Simulation: Compare realized tail mean to forecasted ES
│       └─ Research active: Acerbi & Szekely (2014), Fissler et al. (2016)
├─ Stress Testing & Scenario Analysis:
│   ├─ Purpose: Assess risk beyond VaR/ES (tail events, structural breaks)
│   │   ├─ Historical scenarios: Replay 2008 crisis, COVID March 2020
│   │   ├─ Hypothetical scenarios: Fed hikes 300 bps, recession, geopolitical shock
│   │   └─ Reverse stress test: What scenario causes $X loss?
│   ├─ Historical scenario:
│   │   ├─ Example: Lehman collapse (Sept 2008)
│   │   │   ├─ Equity: S&P 500 -20%
│   │   │   ├─ Credit: IG spreads +200 bps, HY +800 bps
│   │   │   ├─ Volatility: VIX 15 → 45
│   │   │   └─ FX: USD +10% (flight to quality)
│   │   ├─ Apply to current portfolio: Revalue all positions
│   │   └─ Loss: $5M (5% portfolio; VaR₉₉ was $1M; 5× exceedance)
│   ├─ Hypothetical scenario:
│   │   ├─ Design: Subject matter expertise + macroeconomic model
│   │   ├─ Example: "Emerging market crisis"
│   │   │   ├─ EM equity -30%, EM FX -15%
│   │   │   ├─ EM debt spreads +500 bps
│   │   │   ├─ US 10Y yield -50 bps (flight to safety)
│   │   │   └─ Correlation spike: 0.5 → 0.8
│   │   └─ Advantage: Covers plausible but not yet observed events
│   ├─ Reverse stress test:
│   │   ├─ Question: What scenario causes capital depletion?
│   │   ├─ Method: Optimization or enumeration
│   │   │   ├─ Find combination of risk factors → Loss = Capital
│   │   │   └─ Example: Equity -60%, credit spreads +1500 bps → Breach
│   │   └─ Regulatory: UK FSA requires; identifies vulnerabilities
│   ├─ Integration with VaR:
│   │   ├─ VaR: Day-to-day risk (95-99% quantile)
│   │   ├─ Stress: Tail risk (beyond 99.9%; structural breaks)
│   │   └─ Complementary: VaR for normal times; stress for crises
│   └─ Limitations:
│       ├─ Scenario selection: What if crisis differs from design?
│       ├─ Parameter uncertainty: Correlations in stress unclear
│       └─ Behavioral effects: Liquidity evaporation, margin calls (2nd order)
└─ Risk Decomposition & Attribution:
    ├─ Marginal VaR:
    │   ├─ Definition: Change in VaR from small position increase
    │   ├─ MVaR_i = ∂VaR/∂w_i (derivative wrt weight)
    │   ├─ Parametric: MVaR_i = -μ_i + (β_i,p)·σ_p·z_α
    │   │   ├─ β_i,p: Beta of asset i to portfolio
    │   │   └─ Interpretation: Asset's contribution to portfolio risk
    │   └─ Use: Risk budgeting (allocate MVaR across assets)
    ├─ Component VaR:
    │   ├─ Definition: Portion of VaR attributable to asset i
    │   ├─ CVaR_i = w_i × MVaR_i
    │   ├─ Additive: Σ CVaR_i = VaR (portfolio decomposition)
    │   └─ Example: Portfolio VaR=$1M; Asset A CVaR=$400K (40% contribution)
    ├─ Incremental VaR:
    │   ├─ Definition: VaR change from adding new position
    │   ├─ IVaR = VaR(P + Δ) - VaR(P)
    │   └─ Use: New trade approval; ensure IVaR < limit
    └─ Risk budgeting:
        ├─ Allocate risk (not capital) across strategies
        ├─ Example: 3 strategies with CVaR $300K, $500K, $200K → Total VaR $1M
        ├─ Limits: Strategy A max CVaR $400K (40%)
        └─ Advantage: Balances diversification + risk concentration
```

**Key Insight:** VaR simple but non-coherent (penalizes diversification in tail); ES preferred (tail-aware, coherent); backtesting critical for validation; stress tests complement statistical models

## 5. Mini-Project
VaR/ES calculation with backtesting:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# Set seed
np.random.seed(42)

# Generate simulated returns (fat-tailed: Student-t distribution)
T_history = 1000  # Historical data
T_backtest = 250  # Backtesting period
df_true = 5  # True degrees of freedom (fat tails)

# Generate returns using Student-t (fat-tailed)
returns_history = stats.t.rvs(df=df_true, loc=0.0005, scale=0.015, size=T_history)
returns_backtest = stats.t.rvs(df=df_true, loc=0.0005, scale=0.015, size=T_backtest)

# Portfolio value
portfolio_value = 10_000_000  # $10M

# VaR parameters
alpha = 0.99  # 99% confidence

print("="*70)
print("Risk Management: VaR & Expected Shortfall Analysis")
print("="*70)
print(f"Portfolio value: ${portfolio_value:,}")
print(f"Historical data: {T_history} days")
print(f"Backtesting period: {T_backtest} days")
print(f"Confidence level: {alpha*100}%")
print("")

# Method 1: Parametric VaR (Normal assumption)
mu_param = returns_history.mean()
sigma_param = returns_history.std()
z_alpha = stats.norm.ppf(alpha)

VaR_parametric_pct = -(mu_param - z_alpha * sigma_param)
VaR_parametric = VaR_parametric_pct * portfolio_value

ES_parametric_pct = -(mu_param - sigma_param * stats.norm.pdf(z_alpha) / (1 - alpha))
ES_parametric = ES_parametric_pct * portfolio_value

print("Method 1: Parametric VaR (Normal Assumption)")
print("-"*70)
print(f"  Mean: {mu_param*100:.4f}%")
print(f"  Std Dev: {sigma_param*100:.4f}%")
print(f"  VaR (99%): ${VaR_parametric:,.0f} ({VaR_parametric_pct*100:.3f}%)")
print(f"  ES (99%): ${ES_parametric:,.0f} ({ES_parametric_pct*100:.3f}%)")
print(f"  ES/VaR ratio: {ES_parametric/VaR_parametric:.3f}")
print("")

# Method 2: Historical Simulation VaR
returns_sorted = np.sort(returns_history)
var_index = int((1 - alpha) * T_history)

VaR_historical_pct = -returns_sorted[var_index]
VaR_historical = VaR_historical_pct * portfolio_value

# ES: Average of returns beyond VaR
ES_historical_pct = -returns_sorted[:var_index].mean()
ES_historical = ES_historical_pct * portfolio_value

print("Method 2: Historical Simulation")
print("-"*70)
print(f"  VaR (99%): ${VaR_historical:,.0f} ({VaR_historical_pct*100:.3f}%)")
print(f"  ES (99%): ${ES_historical:,.0f} ({ES_historical_pct*100:.3f}%)")
print(f"  ES/VaR ratio: {ES_historical/VaR_historical:.3f}")
print(f"  Worst historical loss: ${-returns_sorted[0]*portfolio_value:,.0f}")
print("")

# Method 3: Monte Carlo VaR (fit Student-t)
# Fit Student-t to historical data
params_t = stats.t.fit(returns_history)
df_fit, loc_fit, scale_fit = params_t

# Simulate
N_simulations = 10000
returns_mc = stats.t.rvs(df=df_fit, loc=loc_fit, scale=scale_fit, size=N_simulations)
returns_mc_sorted = np.sort(returns_mc)
var_index_mc = int((1 - alpha) * N_simulations)

VaR_mc_pct = -returns_mc_sorted[var_index_mc]
VaR_mc = VaR_mc_pct * portfolio_value

ES_mc_pct = -returns_mc_sorted[:var_index_mc].mean()
ES_mc = ES_mc_pct * portfolio_value

print("Method 3: Monte Carlo (Student-t Distribution)")
print("-"*70)
print(f"  Fitted df: {df_fit:.2f}")
print(f"  Fitted loc: {loc_fit*100:.4f}%")
print(f"  Fitted scale: {scale_fit*100:.4f}%")
print(f"  VaR (99%): ${VaR_mc:,.0f} ({VaR_mc_pct*100:.3f}%)")
print(f"  ES (99%): ${ES_mc:,.0f} ({ES_mc_pct*100:.3f}%)")
print(f"  ES/VaR ratio: {ES_mc/VaR_mc:.3f}")
print("")

# Backtesting
print("="*70)
print("Backtesting (Out-of-Sample)")
print("="*70)

methods = {
    'Parametric': VaR_parametric_pct,
    'Historical': VaR_historical_pct,
    'Monte Carlo': VaR_mc_pct
}

for method_name, var_pct in methods.items():
    # Count exceptions (losses exceeding VaR)
    losses = -returns_backtest  # Convert to losses
    exceptions = np.sum(losses > var_pct)
    exception_rate = exceptions / T_backtest
    expected_exceptions = (1 - alpha) * T_backtest
    
    # Kupiec Test
    p0 = 1 - alpha  # Theoretical exception rate
    p_hat = exception_rate  # Observed exception rate
    
    if exceptions > 0:
        LR = -2 * (np.log((1-p0)**(T_backtest - exceptions) * p0**exceptions) - 
                   np.log((1-p_hat)**(T_backtest - exceptions) * p_hat**exceptions))
    else:
        LR = -2 * np.log((1-p0)**(T_backtest))
    
    p_value = 1 - stats.chi2.cdf(LR, df=1)
    
    # Traffic light
    if exceptions <= 4:
        zone = "GREEN"
    elif exceptions <= 9:
        zone = "YELLOW"
    else:
        zone = "RED"
    
    print(f"\n{method_name} VaR:")
    print(f"  Expected exceptions: {expected_exceptions:.1f}")
    print(f"  Observed exceptions: {exceptions}")
    print(f"  Exception rate: {exception_rate*100:.2f}%")
    print(f"  Kupiec LR statistic: {LR:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Result: {'REJECT model' if p_value < 0.05 else 'Cannot reject model'}")
    print(f"  Basel Traffic Light: {zone}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Return distribution with VaR levels
axes[0, 0].hist(returns_history * 100, bins=50, color='blue', alpha=0.6, density=True, label='Historical')

# Overlay fitted distributions
x_range = np.linspace(returns_history.min(), returns_history.max(), 200)
axes[0, 0].plot(x_range * 100, stats.norm.pdf(x_range, mu_param, sigma_param) * 100, 
               linewidth=2, color='green', label='Normal fit', linestyle='--')
axes[0, 0].plot(x_range * 100, stats.t.pdf(x_range, df_fit, loc_fit, scale_fit) * 100,
               linewidth=2, color='red', label='Student-t fit')

# Mark VaR levels
axes[0, 0].axvline(-VaR_parametric_pct * 100, color='green', linestyle=':', linewidth=2, label='VaR Parametric')
axes[0, 0].axvline(-VaR_historical_pct * 100, color='orange', linestyle=':', linewidth=2, label='VaR Historical')
axes[0, 0].axvline(-VaR_mc_pct * 100, color='red', linestyle=':', linewidth=2, label='VaR MC')

axes[0, 0].set_xlabel('Daily Return (%)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Return Distribution & VaR Levels (99%)')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Backtesting - Exceptions over time
losses_backtest = -returns_backtest * 100
days = np.arange(T_backtest)

axes[0, 1].plot(days, losses_backtest, linewidth=0.8, color='blue', alpha=0.7)
axes[0, 1].axhline(VaR_parametric_pct * 100, color='green', linestyle='--', linewidth=2, label='VaR Parametric')
axes[0, 1].axhline(VaR_historical_pct * 100, color='orange', linestyle='--', linewidth=2, label='VaR Historical')
axes[0, 1].axhline(VaR_mc_pct * 100, color='red', linestyle='--', linewidth=2, label='VaR MC')

# Mark exceptions
for method_name, var_pct in methods.items():
    exceptions_idx = np.where(losses_backtest > var_pct * 100)[0]
    if method_name == 'Monte Carlo':  # Only show MC exceptions for clarity
        axes[0, 1].scatter(exceptions_idx, losses_backtest[exceptions_idx], 
                          s=80, marker='x', color='red', linewidth=2, label='Exceptions (MC)')

axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('Loss (%)')
axes[0, 1].set_title('Backtesting: Losses vs VaR Thresholds')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Plot 3: QQ plot (check normality assumption)
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns_history)))
sample_quantiles = np.sort(returns_history)

axes[1, 0].scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=10, color='blue')
axes[1, 0].plot(theoretical_quantiles, theoretical_quantiles, 'r--', linewidth=2, label='Perfect fit')
axes[1, 0].set_xlabel('Theoretical Quantiles (Normal)')
axes[1, 0].set_ylabel('Sample Quantiles')
axes[1, 0].set_title('QQ-Plot: Testing Normality')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: VaR & ES comparison
methods_labels = ['Parametric\n(Normal)', 'Historical', 'Monte Carlo\n(Student-t)']
var_values = [VaR_parametric, VaR_historical, VaR_mc]
es_values = [ES_parametric, ES_historical, ES_mc]

x_pos = np.arange(len(methods_labels))
width = 0.35

axes[1, 1].bar(x_pos - width/2, np.array(var_values)/1e6, width, label='VaR (99%)', color='blue', alpha=0.7)
axes[1, 1].bar(x_pos + width/2, np.array(es_values)/1e6, width, label='ES (99%)', color='red', alpha=0.7)

axes[1, 1].set_ylabel('Risk Measure ($M)')
axes[1, 1].set_title('VaR vs Expected Shortfall Comparison')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(methods_labels)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

# Add ES/VaR ratios as text
for i, (v, e) in enumerate(zip(var_values, es_values)):
    ratio = e/v
    axes[1, 1].text(i, e/1e6 + 0.05, f'{ratio:.2f}×', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('risk_management_var_es.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Parametric VaR underestimates risk (assumes Normal; ignores fat tails)")
print("   → Backtesting shows too many exceptions")
print("")
print("2. Monte Carlo with Student-t captures fat tails better")
print("   → Fewer backtest failures; higher VaR (more conservative)")
print("")
print("3. Expected Shortfall (ES) 15-40% higher than VaR")
print("   → Accounts for tail severity (not just probability)")
print("")
print("4. QQ-plot shows deviation in tails from Normal")
print("   → Justifies Student-t or other fat-tailed distributions")
```

## 6. Challenge Round
When risk models fail or mislead:
- **Model risk (wrong distribution)**: Assume Normal, true Student-t → VaR₉₉ underestimated 30%; 2008 crisis losses 5× VaR; use fat-tailed distributions or historical simulation
- **Non-stationarity**: 2019 σ=1%, 2020 σ=5% (COVID spike) → VaR from 2019 data useless; use rolling windows or GARCH; exponentially-weighted MA (EWMA)
- **Small sample tail**: T=250, α=99% → 2.5 observations in tail; high estimation error; lower α to 95% or use longer history (3-5 years)
- **Procyclicality**: VaR low in calm → high leverage; VaR spikes in crisis → forced deleveraging (fire sales); use through-the-cycle or stressed VaR
- **Neglected correlations**: Individual VaR₉₉(A)=$100K, VaR₉₉(B)=$100K; assume ρ=0; but crisis ρ→0.9 → Portfolio VaR=$190K >> $141K expected; stress correlation matrices
- **Liquidity risk ignored**: VaR assumes can liquidate at market price; but crisis bid-ask 1% → 10%; slippage adds 5-10% to losses; adjust for market impact

## 7. Key References
- [Jorion: Value at Risk (3rd ed, 2006)](https://www.mhprofessional.com/9780071464956-usa-value-at-risk-3rd-edition-group) - Comprehensive VaR handbook
- [Artzner et al: Coherent Risk Measures (1999)](https://www.jstor.org/stable/223483) - Axiomatic framework; ES coherent, VaR not
- [Christoffersen: Evaluating Interval Forecasts (1998)](https://www.jstor.org/stable/2527341) - Backtesting methodology; independence test

---
**Status:** Core risk management | **Complements:** Asset Return Properties, Portfolio Optimization, Stress Testing, Basel Regulation
