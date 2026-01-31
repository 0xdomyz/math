# Lower Partial Moments (LPM) and Downside Risk Measures

## 1. Concept Skeleton
**Definition:** Family of downside risk measures capturing nth-order moment of returns below threshold; generalizes semi-variance for flexible downside penalty weighting  
**Purpose:** Quantify tail risk with customizable severity focus (n parameter); align risk measurement with investor loss aversion; provide unified framework for downside risk  
**Prerequisites:** Moment theory, probability distributions, order statistics, downside risk concepts, portfolio optimization

## 2. Comparative Framing
| Aspect | Variance | Semi-Var (LPM-2) | LPM-1 | LPM-3 | CVaR | VaR |
|--------|----------|------------------|-------|-------|------|-----|
| **Definition** | All deviations² | Downside deviations² | Downside absolute | Downside³ | Tail average | Percentile |
| **Formula** | E[(R-μ)²] | E[max(0,τ-R)²] | E[max(0,τ-R)] | E[max(0,τ-R)³] | E[R\|R<VaR] | q-quantile |
| **Order (n)** | All directions | 2 | 1 | 3 | Tail | Tail |
| **Penalty Weight** | Quadratic | Quadratic down | Linear down | Cubic down | Average excess | Threshold |
| **Extreme Events** | Underweights | Squared penalty | Linear penalty | Cubed penalty | Averaged | Ignored |
| **Computation** | Easy | Easy | Easy | Easy | Moderate | Easy |
| **Tail Sensitivity** | Medium | Medium | Low | High | High | None |
| **Interpretability** | Familiar | Downside volatility | Expected loss | Tail severity focus | Catastrophic | Breakpoint |

## 3. Examples + Counterexamples

**Simple Calculation:**  
Returns: -5%, -3%, -1%, +2%, +4% (threshold τ = 0%).  
LPM(1,0) = [5% + 3% + 1% + 0 + 0]/5 = 1.8% (average loss).  
LPM(2,0) = √[(5²% + 3²% + 1²% + 0 + 0)/5] = 2.05% (semi-deviation).  
LPM(3,0): Cubed penalty → 5³ dominates → larger penalty for catastrophic loss.

**Differentiating Tail Profiles:**  
Strategy A: -10%, -5%, +10%, +15%, +20%.  
Strategy B: -2%, -1.5%, +10%, +15%, +20%.  
Both have same mean (+6%). Variance treats A/B differently. LPM(1,0): A=7.5%, B=1.75% (clear distinction). LPM(3,0): A>> B (extreme penalty for -10%).

**Investor Preference:**  
Risk-averse investor cares most about catastrophic losses (LPM-3 or higher n).  
Moderate investor: LPM-2 (semi-variance).  
Loss-frequency focused: LPM-1 (probability-weighted).  
Choose n aligned with investor loss aversion (risk tolerance).

**Portfolio with Rare Crash:**  
Mix of normal returns + rare -50% event. LPM-1: Only moderate impact (averaged). LPM-2: Larger impact (squared). LPM-3: Severe penalty (cubed -50³). Insurer portfolio: Choose LPM-3 to heavily penalize tail.

**Regime-Dependent Risk:**  
Normal market: LPM-1 ≈ LPM-2 (linear behavior). Crisis: Distinctions emerge. Leveraged hedge fund: LPM-3 >> LPM-2 (extreme losses compounded). Right choice of n matters during regime shift.

**Threshold Impact:**  
Same portfolio, different thresholds. τ=0%: Only negative returns count. τ=5%: Returns below 5% penalized (more losses). LPM changes with threshold choice (user responsibility).

## 4. Layer Breakdown
```
Lower Partial Moments Framework:

├─ Mathematical Definition:
│  ├─ Discrete Form:
│  │   LPM(n, τ) = (1/T) Σₜ max(0, τ - Rₜ)ⁿ
│  │   n = order (1, 2, 3, ...)
│  │   τ = threshold (often 0 or mean)
│  │   T = number of observations
│  ├─ Continuous Form:
│  │   LPM(n, τ) = ∫₋∞^τ (τ - x)ⁿ f(x) dx
│  │   f(x) = probability density function
│  │   Only tail below τ contributes
│  ├─ Alternative Notation:
│  │   Downside Deviation (n=2): √LPM(2, τ)
│  │   Downside Expected Value (n=1): LPM(1, τ)
│  │   Downside Skewness: LPM(3, τ) / LPM(2, τ)^(3/2)
│  ├─ Relationship to Other Measures:
│  │   LPM(1, 0) = expected loss (absolute value)
│  │   LPM(2, mean) = semi-variance
│  │   LPM(∞, VaR) ≈ CVaR × α
│  └─ Threshold Selection (Critical):
│      τ = 0%: Losses only
│      τ = mean: Symmetric downside
│      τ = target: Liability focus
│      τ = risk-free: Excess loss
├─ Interpretation by Order (n):
│  ├─ n = 0: Probability of Loss
│  │   LPM(0, τ) = P(R < τ) = fraction of returns below τ
│  │   Simple frequency measure
│  ├─ n = 1: Expected Downside (Dollar Loss)
│  │   LPM(1, τ) = E[max(0, τ - R)]
│  │   Average magnitude of shortfalls
│  │   Intuitive: "Expected loss if I fall short"
│  ├─ n = 2: Downside Deviation (Downside Volatility)
│  │   LPM(2, τ)^(1/2) = semi-deviation
│  │   Risk analog to variance
│  │   Quadratic penalty for losses
│  ├─ n = 3: Downside Skewness (Higher Moments)
│  │   LPM(3, τ) = E[max(0, τ - R)³]
│  │   Cubic penalty; extreme events heavily weighted
│  │   Rare but severe crashes dominate
│  ├─ n → ∞: Worst-Case (Maximum Loss)
│  │   LPM(∞, τ) = max(0, τ - min(R))
│  │   Single worst loss (not probabilistic)
│  │   Extreme case (see stress testing)
│  └─ Practical Order Selection:
│      Small n (1-2): General downside risk
│      Large n (3-4): Focus on tail severity
│      Depends on investor loss aversion
├─ Threshold Selection Strategies:
│  ├─ Zero Threshold (τ = 0):
│  │   LPM(n, 0) = E[max(0, -R)ⁿ]
│  │   Focuses only on negative returns
│  │   Common for crisis/stress analysis
│  ├─ Mean Threshold (τ = μ):
│  │   Splits portfolio into above/below average
│  │   Symmetric downside definition
│  │   Mathematical convenience
│  ├─ Target Return Threshold (τ = target):
│  │   Liability-driven investing
│  │   "Downside = falling below requirement"
│  │   Pension funds (4% needed): τ = 4%
│  ├─ Safety-First Criterion (τ = rf):
│  │   τ = risk-free rate
│  │   "Excess downside = falling below safe return"
│  │   Roy's safety-first framework
│  ├─ Historical Percentile (τ = quantile):
│  │   τ = median or other percentile
│  │   Regime-dependent definition
│  └─ Optimal τ (Advanced):
│      Minimize utility subject to LPM constraint
│      Varies by investor, time period
├─ Calculation Methods:
│  ├─ Historical Sample (Non-Parametric):
│  │   1. Collect returns R₁, ..., Rₜ
│  │   2. For each: Dᵢ = max(0, τ - Rᵢ)ⁿ
│  │   3. LPM = mean(D)
│  │   Advantages: No distribution assumption
│  │   Disadvantages: Sample errors, tail missing events
│  ├─ Parametric (Assume Distribution):
│  │   Normal: LPM(n,τ) = σⁿ × ∫ (φ scaled) gaussian terms
│  │   Student-t: Fatter tails, different integral
│  │   Analytical formulas available
│  ├─ Monte Carlo Simulation:
│  │   1. Generate many return paths
│  │   2. For each: Dᵢ = max(0, τ - Rᵢ)ⁿ
│  │   3. Average across simulations
│  │   Useful for complex distributions
│  ├─ Extreme Value Theory:
│  │   Model tail explicitly (GPD)
│  │   Extrapolate beyond sample
│  │   Better for rare events
│  └─ Approximations:
│      Taylor expansion around τ
│      Cornish-Fisher for non-normal
├─ Portfolio Application:
│  ├─ Portfolio LPM:
│  │   LPM(n, τ, Rₚ) = (1/T) Σₜ max(0, τ - (w·Rₜ))ⁿ
│  │   Where Rₚ = portfolio return = Σ wᵢRᵢ
│  ├─ Marginal Contribution:
│  │   How does adding asset j affect portfolio LPM?
│  │   ∂LPM(n, τ, Rₚ) / ∂wⱼ
│  │   Used for risk budgeting
│  ├─ LPM-Based Optimization:
│  │   min_w LPM(n, τ, Rₚ) s.t. E[Rₚ] = target return
│  │   Quadratic for n=2 (solvable)
│  │   Nonlinear for n≠2 (needs special solvers)
│  ├─ Efficient Frontier (LPM):
│  │   Curve of optimal portfolios by LPM order
│  │   Differs from mean-variance frontier
│  │   Different for each n choice
│  ├─ Multi-Stage LPM:
│  │   Multiple periods ahead (investment horizon)
│  │   LPM accumulates over time
│  │   Compounding effects
│  └─ Tail Dependence:
│      Correlations in downside critical
│      Copulas capture tail co-movement
├─ Roy's Safety-First Principle:
│  ├─ Framework:
│  │   Maximize expected return subject to:
│  │   P(Rₚ < disaster_level) ≤ acceptable_prob
│  ├─ Connection to LPM:
│  │   LPM(0, τ) = probability of loss below τ
│  │   Roy: Select portfolio minimizing prob(Rₚ < τ)
│  ├─ Interpretation:
│  │   "Don't let me go bankrupt" constraint
│  │   Worst-case focus
│  ├─ vs Mean-Variance:
│  │   MVE: Balance mean and variance
│  │   Safety-First: Avoid catastrophe then maximize return
│  └─ Practical Application:
│      Insurance companies (regulatory capital)
│      Pension funds (benefit coverage)
│      Hedge funds (avoid blow-up)
├─ Risk-Return Tradeoffs:
│  ├─ LPM-Based Sortino Ratio:
│  │   Sortino(n) = (E[Rₚ] - τ) / LPM(n, τ)^(1/n)
│  │   Risk-adjusted return using LPM
│  ├─ Alternative Performance Metrics:
│  │   Kappa: Return / LPM(n, τ)
│  │   Omega: Prob(R > τ) / Prob(R < τ)
│  ├─ Multi-Objective:
│  │   Maximize E[R] and LPM(1, τ) simultaneously
│  │   Pareto frontier (risk-return)
│  └─ Mean-LPM Frontier:
│      Like efficient frontier, but LPM on x-axis
│      Concave shape
├─ Empirical Considerations:
│  ├─ Data Quality:
│  │   Sufficient history for tail accuracy
│  │   High-frequency data (daily) for stability
│  │   Currency, survivorship bias risks
│  ├─ Parameter Stability:
│  │   LPM estimates noisy, especially high n
│  │   Rolling window vs fixed window trade-off
│  │   Regime switches affect LPM
│  ├─ Asset Class Dependence:
│  │   Equities: LPM(2) ≈ 0.3-0.5× variance
│  │   Bonds: LPM much smaller
│  │   Alternatives: LPM(3)>> LPM(2) (tail risk)
│  ├─ Backtesting:
│  │   Test if LPM-minimized portfolios outperform
│  │   Out-of-sample validation
│  │   Rolling optimization
│  └─ Model Risk:
│      LPM order choice arbitrary
│      Different n→ different portfolios
│      Sensitivity analysis essential
├─ Regulatory Use:
│  ├─ Insurance (Solvency II):
│  │   Use downside risk measures
│  │   LPM variants in capital calculations
│  ├─ Pension Funds:
│  │   LPM as downside protection measure
│  │   Aligned with liability coverage goal
│  ├─ Basel III (Banking):
│  │   CVaR preferred (tail-based)
│  │   LPM supplementary for some calculations
│  └─ Fund Manager Policies:
│      Risk limits via LPM(n, threshold)
│      Stress test thresholds set via LPM
└─ Advanced Topics:
   ├─ Conditional LPM:
   │   LPM given market state (crisis vs normal)
   │   Dynamic thresholds
   ├─ Stochastic Thresholds:
   │   τ random variable (not fixed)
   │   Adaptive targets
   ├─ Multivariate LPM:
   │   Joint LPM across assets
   │   Systemic risk
   └─ Machine Learning:
       Predict LPM from data-driven features
       Non-parametric estimation
```

**Interaction:** Choose order (n) and threshold (τ) → Calculate downside returns → Apply penalty → Average → Use in optimization.

## 5. Mini-Project
Calculate and compare LPM across different orders:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Generate return distributions with different tail profiles
np.random.seed(42)
n_sims = 10000

# Distribution 1: Normal (symmetric, light tails)
returns_normal = np.random.normal(0.05, 0.12, n_sims)

# Distribution 2: Student-t (fat tails, symmetric)
returns_fat = np.random.standard_t(5) * 0.12 * np.sqrt(5/3) + 0.05

# Distribution 3: Skewed (hedge fund-like: rare crashes)
normal_part = np.random.normal(0.06, 0.08, int(n_sims * 0.98))
crash_part = np.random.normal(-0.30, 0.05, int(n_sims * 0.02))
returns_skewed = np.concatenate([normal_part, crash_part])

distributions = {
    'Normal': returns_normal,
    'Fat-Tailed (t)': returns_fat,
    'Skewed (Crashes)': returns_skewed
}

# Calculate LPM for different orders and thresholds
lpm_orders = [0, 1, 2, 3, 4]
thresholds = [0, np.mean(returns_normal)]

def calculate_lpm(returns, tau, n):
    """Calculate LPM(n, tau)"""
    downside = np.maximum(0, tau - returns)
    if n == 0:
        return (downside > 0).mean()  # Probability of loss
    else:
        return np.mean(downside ** n) ** (1 / n)  # Root of n-th moment

results = {}
for dist_name, returns in distributions.items():
    results[dist_name] = {}
    for order in lpm_orders:
        for threshold_label, threshold in [('0%', 0), ('Mean', returns.mean())]:
            lpm_val = calculate_lpm(returns, threshold, order)
            key = f'LPM({order}, τ={threshold_label})'
            results[dist_name][key] = lpm_val

# Print results
print("="*100)
print("LOWER PARTIAL MOMENTS COMPARISON")
print("="*100)

for dist_name, dist_results in results.items():
    print(f"\n{dist_name} Distribution:")
    print(f"  Mean: {distributions[dist_name].mean():.4f}")
    print(f"  Std Dev: {distributions[dist_name].std():.4f}")
    print(f"  Skewness: {skew(distributions[dist_name]):.4f}")
    print(f"  Kurtosis: {kurtosis(distributions[dist_name]):.4f}")
    print(f"\n  {'LPM Metric':<30} {'Value':<12}")
    print(f"  {'-'*42}")
    for metric, value in dist_results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric:<30} {value:<12.4f}")

# Risk-adjusted returns (Sortino-like)
print("\n" + "="*100)
print("RISK-ADJUSTED RETURN (EXCESS RETURN / LPM)")
print("="*100)

for dist_name in distributions.keys():
    mean_ret = distributions[dist_name].mean()
    print(f"\n{dist_name}:")
    for order in [1, 2, 3, 4]:
        lpm_val = calculate_lpm(distributions[dist_name], 0, order)
        ratio = mean_ret / lpm_val if lpm_val > 0 else np.inf
        print(f"  Kappa({order}, τ=0): Return/LPM({order}) = {ratio:.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Distributions
for i, (dist_name, returns) in enumerate(distributions.items()):
    ax = axes[0, i]
    ax.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=returns.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero')
    ax.set_title(f'{dist_name}\n(Skew={skew(returns):.2f}, Kurt={kurtosis(returns):.2f})')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

# Plot 2: LPM by order
ax = axes[1, 0]
threshold = 0
lpm_values = {dist: [] for dist in distributions.keys()}
for order in lpm_orders:
    for dist in distributions.keys():
        lpm_val = calculate_lpm(distributions[dist], threshold, order)
        lpm_values[dist].append(lpm_val)

for dist in distributions.keys():
    ax.plot(lpm_orders, lpm_values[dist], marker='o', linewidth=2, label=dist)
ax.set_xlabel('LPM Order (n)')
ax.set_ylabel('LPM Value')
ax.set_title(f'LPM by Order (τ = {threshold:.1%})')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Tail focus comparison
ax = axes[1, 1]
tail_ratios = {}
for dist in distributions.keys():
    lpm1 = calculate_lpm(distributions[dist], 0, 1)
    lpm2 = calculate_lpm(distributions[dist], 0, 2)
    lpm3 = calculate_lpm(distributions[dist], 0, 3)
    tail_ratios[dist] = [lpm1, lpm2, lpm3]

x = np.arange(len(distributions))
width = 0.25
for i, order in enumerate([1, 2, 3]):
    values = [tail_ratios[dist][i] for dist in distributions.keys()]
    ax.bar(x + i*width, values, width, label=f'LPM({order})', alpha=0.8)

ax.set_ylabel('LPM Value')
ax.set_title('LPM by Distribution (Higher Orders = Tail Focus)')
ax.set_xticks(x + width)
ax.set_xticklabels(distributions.keys())
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Kappa ratios
ax = axes[1, 2]
kappas = {}
for dist_name in distributions.keys():
    mean_ret = distributions[dist_name].mean()
    kappas[dist_name] = []
    for order in [1, 2, 3, 4]:
        lpm_val = calculate_lpm(distributions[dist_name], 0, order)
        kappa = mean_ret / lpm_val if lpm_val > 0 else np.inf
        kappas[dist_name].append(kappa)

for dist in distributions.keys():
    ax.plot([1, 2, 3, 4], kappas[dist], marker='s', linewidth=2, label=dist)
ax.set_xlabel('LPM Order (n)')
ax.set_ylabel('Kappa(n) = Return / LPM(n)')
ax.set_title('Risk-Adjusted Return by LPM Order')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate portfolio selection impact
print("\n" + "="*100)
print("PORTFOLIO SELECTION: WHICH DISTRIBUTION TO PREFER?")
print("="*100)
print("\nChoice depends on LPM order (investor loss aversion):")
print("  n=1: Average downside → Fat-tailed worse (high avg loss)")
print("  n=2: Downside variance → Normal vs Fat-tail comparable")
print("  n=3: Tail severity → Skewed worse (crashes cubed)")
print("  n=4: Extreme tail → Skewed vastly worse")
print("\nInvestor risk tolerance determines order selection!")
```

## 6. Challenge Round
- Derive closed-form LPM for normal distribution (Cornish-Fisher expansion)
- Implement LPM-based portfolio optimization (CVX solver)
- Compare Roy's safety-first criterion with mean-variance optimization
- Design LPM risk budgeting framework for multi-asset portfolio
- Explain LPM relationship to stochastic dominance theory

## 7. Key References
- [Fishburn, "Stochastic Dominance and Mean-Variance Analysis" (1977)](https://www.jstor.org/stable/2630964) — LPM foundations
- [Bawa & Lindenberg, "Capital Market Equilibrium in a Mean-Lower Partial Moment Framework" (1977)](https://www.jstor.org/) — Portfolio theory extensions
- [Kaplan & Knowles, "Higher Moments and Optimal Portfolio Selection" (2013)](https://www.jstor.org/) — LPM optimization
- [Sortino & Satchell, "Managing Downside Risk in Financial Markets" (2007)](https://www.butterworth-heinemann.com/) — Comprehensive treatment

---
**Status:** Generalized downside risk family | **Complements:** Semi-Variance, CVaR, VaR, Sortino Ratio
