# Semi-Variance and Downside Risk

## 1. Concept Skeleton
**Definition:** Volatility measure capturing only returns below threshold (typically mean or target); ignores upside deviations  
**Purpose:** Reflect investor preference for downside protection; better align risk measure with loss aversion; improve portfolio optimization for asymmetric return distributions  
**Prerequisites:** Variance concept, portfolio theory, loss aversion, probability distributions, mean-variance analysis

## 2. Comparative Framing
| Aspect | Variance | Semi-Variance | CVaR | Downside Deviation | LPM |
|--------|----------|---------------|------|-------------------|-----|
| **Definition** | All deviations squared | Below-threshold squared | Tail average | Root of semi-variance | Power moments |
| **Upside Penalty** | Yes (wasted volatility) | No (beneficial) | No (tail only) | No | Customizable |
| **Mathematical** | E[(R - μ)²] | E[max(0, μ - R)²] | Tail conditional mean | √Semi-Var | E[max(τ - R, 0)ⁿ] |
| **Threshold** | Fixed at mean | Can be mean/target | At VaR quantile | At mean/target | User-defined |
| **Interpretation** | Total volatility | "Bad" volatility | Extreme loss | Downside std dev | Generalized downside |
| **Investor Intuition** | Symmetric pain | Asymmetric (losses hurt) | Catastrophic scenarios | Moderate downside | Flexible loss focus |
| **Computation** | Simple (quadratic) | Conditional expectation | Tail simulation | Moderate | Integral-based |

## 3. Examples + Counterexamples

**Simple Case:**  
Annual returns: -5%, -2%, +1%, +3%, +5%, +8%, +12%, +15% (mean = 4.625%).  
Variance: Averages all squared deviations from 4.625%.  
Semi-Variance: Only (-5% - 4.625%)² + (-2% - 4.625%)² / 8 = only bad outcomes contribute.  
Upside (+8%, +12%, +15%) ignored in semi-variance.

**Two Portfolios, Same Variance:**  
Portfolio A: -10%, +15% (returns symmetric around 2.5%).  
Portfolio B: +2%, +3% (returns clustered, no downside).  
Both have variance σ² = 56.25 (can verify). But B clearly less risky (no catastrophic loss). B's semi-variance < A's semi-variance. Semi-variance distinguishes them.

**Skewed Returns (Sell Puts Strategy):**  
Monthly returns: +1%, +1%, +1%, +1%, +1%, +1%, +1%, +1%, -50% (tail hedge ineffective).  
Mean = -4.2%, Variance = 226. Semi-variance (below mean) = huge (-50% loss).  
Variance treats upside +1% deviations same as downside -50% deviation (penalty symmetric).  
Semi-variance captures tail risk asymmetry.

**Insurance/Pension Use Case:**  
Pension fund: "We care about shortfalls below 4% target return, not outperformance above."  
Semi-variance with threshold = 4% targets exactly this worry.  
Variance would penalize beating 4% target (wasted optimization).

**Mean-Variance Optimization Flaw:**  
Min-variance portfolio often short-volatility, long-skew (like hedge fund: +small gains, -rare crashes).  
Mean-variance sees high vol, ranks low.  
Semi-variance recognizes skewed distribution, ranks higher (if average positive).  
Different efficient frontiers; semi-variance avoids tail-risk strategies.

**Stock vs Options:**  
Stock: Symmetric distribution, variance ≈ semi-variance.  
Long call option: Skewed distribution (limited downside, unlimited upside).  
Variance: Penalizes upside deviation.  
Semi-variance: Ignores upside, focuses on downside (none for long call).  
Semi-variance better for derivative portfolios.

## 4. Layer Breakdown
```
Semi-Variance and Downside Risk Framework:

├─ Definition & Notation:
│  ├─ Semi-Variance (Traditional):
│  │   SV(τ) = (1/n) Σᵢ max(0, τ - Rᵢ)²
│  │   τ = threshold (typically mean or target return)
│  │   Max: Returns below τ contribute to sum
│  ├─ Downside Deviation:
│  │   D(τ) = √SV(τ)
│  │   Square root of semi-variance (same scale as volatility)
│  ├─ Alternative: Semi-Variance (Below-Target):
│  │   SV(τ) = E[max(0, τ - R)²] = Prob(R < τ) × E[(τ - R)² | R < τ]
│  │   Decomposed into: frequency of shortfall × magnitude
│  ├─ Relationship to Variance:
│  │   If R ~ N(μ, σ²) symmetric:
│  │   SV(μ) = σ²/2 (variance split equally above/below)
│  ├─ Connection to LPM:
│  │   SV = LPM(2, τ) (2nd order lower partial moment)
│  │   Semi-Dev = √LPM(2, τ)
│  └─ Units:
│      Semi-Variance: Return units squared (%)²
│      Semi-Deviation: Return units (%), comparable to std dev
├─ Threshold Selection:
│  ├─ Mean as Threshold (Symmetric Benchmark):
│  │   τ = E[R] (mean return)
│  │   Standard choice, mathematical convenience
│  │   Interpretation: Half deviations below average
│  ├─ Zero as Threshold (Loss Definition):
│  │   τ = 0%
│  │   Focus on actual losses only (negative returns)
│  │   Common in insurance/pension contexts
│  ├─ Target Return (Liability-Driven):
│  │   τ = investor's required return
│  │   Pension: 4% annual needed for benefits
│  │   Below 4% = shortfall cost (valued asset losses)
│  ├─ Historical Mean (Backward-Looking):
│  │   τ = average of past returns
│  │   Assumes distribution stable (may not hold)
│  ├─ Risk-Free Rate:
│  │   τ = rf (e.g., T-bill rate)
│  │   Excess return focus; captures underperformance vs safe asset
│  └─ Optimal Choice:
│      Depends on investor objectives
│      Should align with actual loss definition
├─ Calculation Methods:
│  ├─ Historical Sample (Discrete):
│  │   1. Collect historical returns R₁, R₂, ..., Rₙ
│  │   2. For each return: Dᵢ = max(0, τ - Rᵢ)
│  │   3. Semi-Variance = (1/n) Σᵢ Dᵢ²
│  │   4. Semi-Dev = √(Semi-Var)
│  │   Advantages: Non-parametric, reflects actual tail
│  │   Disadvantages: Requires sufficient data for tail accuracy
│  ├─ Parametric (Normal Distribution):
│  │   Assume R ~ N(μ, σ²)
│  │   SV(τ) = σ² × [Φ((τ-μ)/σ) + ((τ-μ)/σ)×φ((τ-μ)/σ) - 0.5]
│  │   Φ = cumulative normal CDF, φ = standard normal PDF
│  │   For τ = μ: SV(μ) = σ²/2
│  ├─ Analytical (Truncated Normal):
│  │   Closed form for left truncation (R ≥ limit)
│  │   More complex algebra but exact for normal
│  ├─ Skewed Distribution:
│  │   Use Cornish-Fisher expansion for non-normal tails
│  │   Include skewness and kurtosis adjustments
│  ├─ Simulation:
│  │   Generate random returns from model
│  │   Compute semi-variance across simulated paths
│  │   Useful for complex, multi-asset portfolios
│  └─ Copula Methods:
│      For multivariate downside risk (joint tail dependence)
├─ Portfolio Application:
│  ├─ Portfolio Semi-Variance:
│  │   SV(Portfolio) = (1/n) Σᵢ max(0, τ - Rₚᵢ)²
│  │   Where Rₚᵢ = Σⱼ wⱼ Rⱼᵢ (portfolio return at time i)
│  ├─ Component Analysis:
│  │   Contribution of each asset to portfolio downside risk
│  │   Marginal semi-variance: ∂SV(Portfolio) / ∂wⱼ
│  │   Different from marginal variance (captures asymmetry)
│  ├─ Efficient Frontier (Semi-Variance):
│  │   min_w SV(Portfolio, τ) subject to E[Rₚ] = target return
│  │   Curves differ from mean-variance frontier
│  │   More conservative (emphasizes downside)
│  ├─ Correlation Impact:
│  │   Downside correlation ≠ unconditional correlation
│  │   Assets uncorrelated in mean, correlated in tail
│  │   Semi-variance portfolio captures tail co-movement
│  ├─ Optimization Algorithm:
│  │   Quadratic program (like mean-variance, but asymmetric)
│  │   Non-smooth at τ (abs value), need special solvers
│  │   Generally solvable via convex optimization
│  └─ Rebalancing:
│      Semi-variance optimal weights may differ from MV
│      Different rebalancing schedule optimal
├─ Sortino Ratio (Risk-Adjusted Performance):
│  ├─ Definition:
│  │   Sortino = (E[Rp] - τ) / Downside Deviation(τ)
│  │   Similar to Sharpe ratio but with semi-deviation denominator
│  ├─ Interpretation:
│  │   Excess return per unit of downside risk
│  │   Ranks strategies by downside-adjusted performance
│  ├─ Advantage over Sharpe:
│  │   High Sharpe ratio portfolio (volatile outperformer) may have low Sortino
│  │   If returns skewed: upside volatility wasted in Sharpe penalty
│  │   Sortino ignores upside, focuses on shortfall
│  ├─ Calculation:
│  │   1. Calculate downside deviation
│  │   2. Calculate excess return above threshold
│  │   3. Ratio = excess / downside_dev
│  └─ Use Case:
│      Hedge funds: Long skew premium strategies rank high on Sortino
│      Mean-variance poor (high vol), but downside-aware superior
├─ Empirical Considerations:
│  ├─ Data Requirements:
│  │   Sufficient observations to estimate tail accurately
│  │   For threshold at 5% tail: Need ≥ 200 observations
│  │   Longer look-back helps but introduces parameter instability
│  ├─ Estimation Noise:
│  │   Semi-variance standard error larger than variance
│  │   Concentration on tail (fewer observations)
│  │   Boostrapping confidence intervals recommended
│  ├─ Time Horizon:
│  │   Different thresholds for daily vs annual risk
│  │   Compounding effects matter for multi-period
│  │   Rolling window vs fixed window trade-offs
│  ├─ Regime Switching:
│  │   Downside risk varies across market regimes
│  │   Crisis periods: Semi-variance >> normal periods
│  │   Model need to adapt (dynamic semi-variance)
│  └─ Asset Class Dependence:
│      Stocks: Semi-variance ≈ 0.4-0.5 × variance
│      Bonds: Semi-variance much smaller (downside rare)
│      Alternatives (hedge funds): Semi-variance << variance (skewed positive)
├─ Practical Uses:
│  ├─ Pension Fund Management:
│  │   Liability = minimum return threshold
│  │   Minimize semi-variance around liability
│  │   Avoids excessive upside-only growth
│  ├─ Insurance Companies:
│  │   Solvency capital requirement (SCR) calculation
│  │   Focus on losses (downside) not gains
│  ├─ Risk Limits:
│  │   Traders: "Semi-variance not to exceed X%"
│  │   More intuitive: "Downside losses capped"
│  │   Captures risk trader actually cares about
│  ├─ Portfolio Rebalancing Triggers:
│  │   Rebalance if semi-variance drifts above limit
│  │   Rather than mechanical 1% drift on MV basis
│  └─ Client Communication:
│      Sortino ratio more intuitive than Sharpe
│      "We optimize for downside losses" resonates better
└─ Limitations & Extensions:
   ├─ Computational Complexity:
   │   Non-smooth optimization (abs value at threshold)
   │   Requires specialized solvers
   │   More complex than mean-variance
   ├─ Threshold Ambiguity:
   │   Multiple reasonable threshold choices
   │   Different thresholds → different portfolios
   │   Sensitivity to threshold selection critical
   ├─ Historical Bias:
   │   If past returns don't predict future distribution
   │   Tail estimates unstable
   │   Recent crises may be rare events
   ├─ Multiple Periods:
   │   Single-period semi-variance simpler
   │   Multi-period dynamic semi-variance complex
   │   Path dependency in tail events
   ├─ Extensions:
   │   Conditional Semi-Variance: Semi-Var | market state
   │   Stochastic Semi-Variance: Time-varying threshold
   │   Multi-target: Multiple shortfall thresholds
   └─ Integration with Other Metrics:
       Combine with skewness, kurtosis for full picture
       Use alongside VaR for comprehensive tail risk
       Complement with stress testing scenarios
```

**Interaction:** Return distribution → threshold selection → below-threshold measurement → semi-variance calculation → portfolio optimization.

## 5. Mini-Project
Compare portfolios using variance vs semi-variance:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate 3 assets with different skewness profiles
np.random.seed(42)
n = 1000

# Asset 1: Normal returns (symmetric)
asset1 = np.random.normal(0.05, 0.12, n)

# Asset 2: Positive skew (long call strategy)
asset2_normal = np.random.normal(0.04, 0.10, n)
asset2_crashes = np.random.normal(-0.30, 0.05, 20)
asset2 = np.concatenate([asset2_normal[:-20], asset2_crashes])

# Asset 3: Negative skew (short premium)
asset3_normal = np.random.normal(0.06, 0.08, n)
asset3_crashes = np.random.normal(-0.25, 0.08, 15)
asset3 = np.concatenate([asset3_normal[:-15], asset3_crashes])

returns = np.column_stack([asset1, asset2, asset3])
asset_names = ['Asset 1\n(Symmetric)', 'Asset 2\n(Positive Skew)', 'Asset 3\n(Negative Skew)']

# Calculate statistics
means = returns.mean(axis=0)
stds = returns.std(axis=0)
tau = means.mean()  # threshold = average of all means

# Calculate semi-variance (below threshold)
def semi_variance(returns_col, threshold):
    downside = np.maximum(0, threshold - returns_col)
    return np.mean(downside ** 2)

def downside_deviation(returns_col, threshold):
    return np.sqrt(semi_variance(returns_col, threshold))

semi_vars = [semi_variance(returns[:, i], tau) for i in range(3)]
semi_devs = [downside_deviation(returns[:, i], tau) for i in range(3)]

# Covariance and semi-covariance matrices
cov_matrix = np.cov(returns.T)

# Semi-covariance (downside co-movement)
def semi_cov_matrix(returns, threshold):
    n = returns.shape[1]
    semi_cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            downside_i = np.maximum(0, threshold - returns[:, i])
            downside_j = np.maximum(0, threshold - returns[:, j])
            semi_cov[i, j] = np.mean(downside_i * downside_j)
    return semi_cov

semi_cov = semi_cov_matrix(returns, tau)

# Portfolio optimization: Mean-Variance
def portfolio_variance(w, cov):
    return w @ cov @ w

def portfolio_return(w, means):
    return w @ means

target_return = 0.045
constraints_mv = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    {'type': 'eq', 'fun': lambda w: portfolio_return(w, means) - target_return}
]
bounds = [(0, 1) for _ in range(3)]
x0 = np.array([1/3, 1/3, 1/3])

result_mv = minimize(lambda w: portfolio_variance(w, cov_matrix),
                     x0, method='SLSQP', bounds=bounds, constraints=constraints_mv)
w_mv = result_mv.x

# Portfolio optimization: Semi-Variance
def portfolio_semi_variance(w, returns, threshold):
    p_returns = returns @ w
    downside = np.maximum(0, threshold - p_returns)
    return np.mean(downside ** 2)

result_sv = minimize(lambda w: portfolio_semi_variance(w, returns, tau),
                     x0, method='SLSQP', bounds=bounds, constraints=constraints_mv)
w_sv = result_sv.x

# Calculate results
port_ret_mv = portfolio_return(w_mv, means)
port_vol_mv = np.sqrt(portfolio_variance(w_mv, cov_matrix))
port_semi_dev_mv = downside_deviation(returns @ w_mv, tau)

port_ret_sv = portfolio_return(w_sv, means)
port_vol_sv = np.sqrt(portfolio_variance(w_sv, cov_matrix))
port_semi_dev_sv = downside_deviation(returns @ w_sv, tau)

# Print comparison
print("="*80)
print("VARIANCE vs SEMI-VARIANCE PORTFOLIO OPTIMIZATION")
print("="*80)
print(f"\nAsset Statistics:")
print(f"{'Asset':<20} {'Mean':<10} {'Std Dev':<12} {'Semi-Dev':<12} {'Skewness':<10}")
print("-"*80)
for i, name in enumerate(asset_names):
    skew = pd.Series(returns[:, i]).skew()
    print(f"{name:<20} {means[i]:<10.4f} {stds[i]:<12.4f} {semi_devs[i]:<12.4f} {skew:<10.3f}")

print(f"\nPortfolio Weights (Target Return = {target_return:.2%}):")
print(f"{'Portfolio':<20} {'Asset 1':<10} {'Asset 2':<10} {'Asset 3':<10}")
print("-"*80)
print(f"{'Mean-Variance':<20} {w_mv[0]:<10.3f} {w_mv[1]:<10.3f} {w_mv[2]:<10.3f}")
print(f"{'Semi-Variance':<20} {w_sv[0]:<10.3f} {w_sv[1]:<10.3f} {w_sv[2]:<10.3f}")

print(f"\nPortfolio Performance:")
print(f"{'Metric':<20} {'Mean-Var':<15} {'Semi-Var':<15}")
print("-"*80)
print(f"{'Expected Return':<20} {port_ret_mv:<15.4f} {port_ret_sv:<15.4f}")
print(f"{'Std Deviation':<20} {port_vol_mv:<15.4f} {port_vol_sv:<15.4f}")
print(f"{'Downside Dev':<20} {port_semi_dev_mv:<15.4f} {port_semi_dev_sv:<15.4f}")
print(f"{'Sortino Ratio':<20} {(port_ret_mv - tau) / port_semi_dev_mv:<15.4f} {(port_ret_sv - tau) / port_semi_dev_sv:<15.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Return distributions
for i, name in enumerate(asset_names):
    ax = axes[0, i]
    ax.hist(returns[:, i], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=means[i], color='red', linestyle='--', linewidth=2, label=f'Mean={means[i]:.3f}')
    ax.axvline(x=tau, color='green', linestyle='--', linewidth=2, label=f'Threshold={tau:.3f}')
    ax.set_title(name)
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

# Plot 2: Comparison bars (bottom)
ax = axes[1, 0]
metrics = ['Std Dev', 'Semi-Dev']
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, stds, width, label='Std Dev', alpha=0.8)
ax.bar(x + width/2, semi_devs, width, label='Semi-Dev', alpha=0.8)
ax.set_ylabel('Risk Measure')
ax.set_title('Variance vs Semi-Variance by Asset')
ax.set_xticks(x)
ax.set_xticklabels(asset_names)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Efficient frontier comparison
ax = axes[1, 1]
w_range = np.linspace(0, 1, 50)
frontier_mv_vol = []
frontier_mv_semi = []
frontier_sv_vol = []
frontier_sv_semi = []

for w1 in w_range:
    w = np.array([w1, (1-w1)/2, (1-w1)/2])
    vol = np.sqrt(portfolio_variance(w, cov_matrix))
    semi_dev = downside_deviation(returns @ w, tau)
    frontier_mv_vol.append(vol)
    frontier_mv_semi.append(semi_dev)

ax.scatter(frontier_mv_vol, [portfolio_return(np.array([w1, (1-w1)/2, (1-w1)/2]), means) 
           for w1 in w_range], alpha=0.5, s=30, label='Equal Weight Frontier', color='gray')
ax.scatter([port_vol_mv], [port_ret_mv], s=200, marker='*', color='blue', 
          label=f'MV Optimal', zorder=5)
ax.scatter([port_vol_sv], [port_ret_sv], s=200, marker='o', color='red',
          label=f'Semi-Var Optimal', zorder=5)
ax.set_xlabel('Volatility (Std Dev)')
ax.set_ylabel('Expected Return')
ax.set_title('Portfolio Optimization: MV vs Semi-Variance')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Weight comparison
ax = axes[1, 2]
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, w_mv, width, label='Mean-Variance', alpha=0.8)
ax.bar(x + width/2, w_sv, width, label='Semi-Variance', alpha=0.8)
ax.set_ylabel('Weight')
ax.set_title('Optimal Weights Comparison')
ax.set_xticks(x)
ax.set_xticklabels(asset_names)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
- Derive semi-variance for normal distribution closed-form
- Implement semi-variance portfolio optimization (convex solver)
- Compare efficient frontiers: mean-variance vs semi-variance
- Design Sortino ratio optimization vs Sharpe ratio optimization
- Explain tail co-movement and downside correlation vs Pearson correlation

## 7. Key References
- [Markowitz, "Portfolio Selection" (1959)](https://www.jstor.org/stable/2975974) — Foundational, mentions downside
- [Fishburn, "Mean-Risk Analysis with Risk Associated with Below-Target Returns" (1977)](https://www.jstor.org/stable/2630964) — Semi-variance formalization
- [Sortino & Price, "Performance Measurement in a Downside Risk Framework" (1994)](https://www.investopedia.com/terms/s/sortinoratio.asp) — Sortino ratio
- [Kaplan & Knowles, "Kurtosis and Semi-Variance Target Downside Risk" (2013)](https://www.jstor.org/) — Modern semi-variance applications

---
**Status:** Asymmetric downside risk measure | **Complements:** CVaR, Variance, Portfolio Optimization, Sortino Ratio
