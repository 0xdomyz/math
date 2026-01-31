# Conditional Value at Risk (CVaR) / Expected Shortfall

## 1. Concept Skeleton
**Definition:** Average of losses exceeding Value at Risk threshold; measures expected loss in worst-case scenarios beyond confidence level  
**Purpose:** Address VaR limitation (ignores tail severity); coherent risk measure; reflects true catastrophic loss potential; used in regulatory capital requirements  
**Prerequisites:** Value at Risk, quantile concepts, tail risk, probability distributions, coherent risk measures

## 2. Comparative Framing
| Aspect | VaR | CVaR/ES | Semi-Variance | Maximum Loss | LPM |
|--------|-----|---------|---------------|--------------|-----|
| **Definition** | Percentile loss threshold | Avg loss beyond VaR | Variance of downside | Worst historical loss | Moment-based downside |
| **Tail Information** | Boundary point | Full tail average | Partial downside | Single worst case | Generalized downside |
| **Coherent?** | No (fails subadditivity) | Yes (all axioms) | Semi (not subadditive) | No | Yes |
| **Computation** | Quantile estimation | Tail averaging | Below-zero variance | Scan history | Integral calculation |
| **Tail Sensitivity** | Binary (at/below) | Continuous (average) | Quadratic below target | Point estimate | Moment-dependent |
| **Regulatory Use** | Basel II (old) | Basel III, Solvency II | Specialist risk | Stress testing | Research |
| **Interpretation** | "95% won't lose more than X" | "If we lose, avg loss is Y" | "Downside volatility is Z" | "Worst was -W%" | "Power-weighted downside" |

## 3. Examples + Counterexamples

**Simple Case:**  
Portfolio returns: -5%, -3%, -2%, +1%, +4%, +5%, +6%, +8% over 8 scenarios. 95% VaR = -2% (5th worst). CVaR = avg of worst 5% = (-5% - 3%) / 2 = -4%. CVaR acknowledges tail severity.

**Same VaR, Different CVaR:**  
Portfolio A: -10%, -8%, +2%, +3%, +4%, +5%, +6%, +10% (95% VaR = -8%). Portfolio B: -8%, -7.5%, +2%, +3%, +4%, +5%, +6%, +10% (95% VaR = -7.5%). Same VaR threshold, but A has worse tail; A's CVaR = -9%, B's CVaR = -7.75%. CVaR distinguishes them.

**Coherent Risk Measure:**  
Portfolio = Long 50% Stock A + 50% Stock B. CVaR(Portfolio) ≤ 0.5×CVaR(A) + 0.5×CVaR(B). Subadditivity holds. VaR can violate this (diversification increases VaR in rare cases).

**Regulatory Advantage:**  
Basel III requires CVaR for market risk capital. Reflects true tail loss. Bank holding 100% junk bonds might have low VaR (rarely defaults) but huge CVaR (catastrophic when it happens). CVaR captures hidden risk.

**Insufficient Sample Tail:**  
Low-frequency events (1-in-100-year crisis). Fewer observations beyond VaR threshold → CVaR estimate noisy. Small changes in threshold assumptions → large CVaR swings. Model risk from extreme value extrapolation.

**Diversification Paradox:**  
Adding uncorrelated asset reduces total volatility but increases CVaR in rare joint-tail events (tail correlation). CVaR captures this hidden risk; Markowitz variance optimization misses it.

## 4. Layer Breakdown
```
Conditional Value at Risk (CVaR) Framework:

├─ Definition & Notation:
│  ├─ VaR at level α:
│  │   VaRα(X) = -inf{x : P(X ≤ x) ≥ α}
│  │   α typically 0.95 or 0.99 (1% or 5% left tail)
│  ├─ CVaR Definition:
│  │   CVaRα(X) = E[X | X ≤ VaRα(X)]
│  │   Expected value of returns in worst α% scenarios
│  ├─ Equivalence (Continuous):
│  │   CVaRα(X) = -1/α ∫₀^α VaRu(X) du
│  │   Average of all VaR levels from 0 to α
│  ├─ Alternative Name:
│  │   Expected Shortfall (ES), Tail Conditional Expectation
│  │   Accent Value at Risk, Incremental VaR
│  └─ Relationship to VaR:
│      CVaR ≥ |VaR| always (worse or equal)
│      In fat-tailed: CVaR >> VaR (significant tail severity)
├─ Calculation Methods:
│  ├─ Historical Simulation:
│  │   1. Calculate historical returns
│  │   2. Sort returns ascending (worst first)
│  │   3. Select α% worst returns (e.g., 5 worst of 100)
│  │   4. Average these returns → CVaR estimate
│  │   Advantages: Non-parametric, captures real tail
│  │   Disadvantages: Limited by sample size, rare events missed
│  ├─ Parametric (Normal Distribution):
│  │   Assume returns ~ N(μ, σ²)
│  │   CVaRα = -μ - σ × φ(z_α) / α
│  │   where z_α = Φ⁻¹(α) (inverse normal CDF)
│  │   φ(z) = standard normal PDF
│  │   Advantages: Smooth, requires only mean + std dev
│  │   Disadvantages: Misses fat tails if returns not normal
│  ├─ Cornish-Fisher Expansion:
│  │   Adjust normal CVaR for skewness and kurtosis
│  │   CVaRα,CF = -μ - σ × [φ(z_α) + ... skew/kurtosis terms]
│  │   Better for non-normal distributions
│  ├─ Extreme Value Theory (EVT):
│  │   Model tail beyond threshold using GPD
│  │   Generalized Pareto Distribution: P(X > u+y | X > u)
│  │   CVaR estimated from tail shape parameters
│  │   Advantages: Specialized for tail behavior
│  │   Disadvantages: Complex, requires tail calibration
│  ├─ Monte Carlo Simulation:
│  │   1. Simulate many portfolio return paths
│  │   2. For each path, calculate portfolio loss
│  │   3. Sort losses, take α% worst
│  │   4. Average worst α% → CVaR
│  │   Advantages: Handles complex dynamics, non-linear payoffs
│  │   Disadvantages: Computationally intensive, simulation error
│  └─ Kernel Density Estimation:
│      Non-parametric, smooth tail using kernel
│      More efficient than raw historical
├─ Coherence Properties (Artzner et al):
│  ├─ Monotonicity:
│  │   If X ≤ Y always, then CVaR(X) ≤ CVaR(Y)
│  │   Worse distribution has worse CVaR
│  ├─ Translation Invariance:
│  │   CVaR(X + c) = CVaR(X) + c
│  │   Adding constant shifts CVaR by same amount
│  ├─ Homogeneity:
│  │   CVaR(λX) = λ × CVaR(X) for λ > 0
│  │   Risk scales linearly with position size
│  ├─ Subadditivity (Key Advantage over VaR):
│  │   CVaR(X + Y) ≤ CVaR(X) + CVaR(Y)
│  │   Diversification always reduces risk (or keeps same)
│  │   Enforces coherent aggregation across positions
│  └─ Implication:
│      CVaR justified for portfolio aggregation
│      VaR violations: diversified portfolio can have higher VaR than sum!
├─ Relationship to VaR:
│  ├─ Lower Bound:
│  │   CVaRα ≥ VaRα (equal only if discrete atom at VaR)
│  ├─ Tail Severity:
│  │   Difference = (CVaR - VaR) quantifies tail heaviness
│  │   Normal dist: Small difference
│  │   Fat-tailed (t-dist): Large difference
│  ├─ Approximation (for continuous distributions):
│  │   CVaRα ≈ VaRα + E[|X - VaRα| | X ≤ VaRα]
│  │   VaR + expected excess beyond threshold
│  └─ Interpretation:
│      VaR: "95% won't lose MORE than $X"
│      CVaR: "If in worst 5%, expect to lose $Y on average"
├─ Portfolio Application:
│  ├─ Portfolio CVaR:
│  │   CVaR(Portfolio) ≠ weighted average of component CVaRs
│  │   Must account for joint tail dependence
│  │   Copula models essential for accurate calculation
│  ├─ CVaR Minimization:
│  │   min_w CVaRα(wₜR)
│  │   Subject to: Σ wᵢ = 1, wᵢ ≥ 0 (no short)
│  │   Convex optimization problem (unlike mean-variance)
│  ├─ Comparison to Mean-Variance:
│  │   MVE: Minimize variance for target return
│  │   CVaR-Based: Minimize CVaR for target return
│  │   CVaR frontier different, more conservative
│  │   Captures tail risk that variance ignores
│  └─ Rebalancing:
│      CVaR-based allocations more stable in crises
│      Less prone to concentration in tail events
├─ Empirical Considerations:
│  ├─ Confidence Level α Choice:
│  │   α = 0.01 (1%): Extreme tail, needs more data
│  │   α = 0.05 (5%): Standard choice, balance precision-data
│  │   Higher α: Smoother estimate, less tail sensitivity
│  ├─ Estimation Error:
│  │   CVaR standard error: σ(CVaR) ≈ √(Var(X) / (n × α²))
│  │   Depends on α (smaller α → larger error)
│  │   Requires large sample for reliable tail estimates
│  ├─ Backtesting:
│  │   Track: Do actual losses exceed CVaR?
│  │   If yes, happens more than α% → model underestimates risk
│  │   Kupiec POF test: Statistical test for CVaR violations
│  ├─ Stress Testing:
│  │   CVaR assumes past distributions continue
│  │   Extreme events (COVID, 2008) break assumptions
│  │   Supplement with scenario analysis
│  └─ Model Risk:
│      Choice of distribution (normal vs t vs other) matters
│      Parameters (volatility) estimates uncertain
│      CVaR sensitive to tail calibration
├─ Regulatory & Practical Use:
│  ├─ Basel III:
│  │   Replaces VaR with CVaR (Stressed CVaR)
│  │   More conservative capital requirements
│  │   Captured 2008 crisis lessons
│  ├─ Solvency II (Insurance):
│  │   CVaR at 99.5% confidence level
│  │   Ensures rare insurance events covered
│  ├─ CFTC/SEC Requirements:
│  │   Futures margin: CVaR-based calculations
│  │   Cleared derivatives: Initial margin via CVaR
│  ├─ Proprietary Trading:
│  │   Position limits via CVaR
│  │   Risk attribution by CVaR contribution
│  ├─ Asset Management:
│  │   CVaR-optimized portfolios marketed as "tail-safe"
│  │   Downside risk focus appeals to conservative investors
│  └─ Hedge Fund Benchmarking:
│      Performance measured by CVaR-adjusted returns
│      Penalizes tail losses more than Sharpe ratio
└─ Limitations & Alternatives:
   ├─ Estimation Risk:
   │   Tail estimates highly uncertain
   │   Rare events by definition have few observations
   ├─ Model Dependence:
   │   Normal model: CVaR far too low
   │   Extreme value model: Complex calibration
   ├─ Correlation Assumptions:
   │   Joint tail correlation critical
   │   Standard correlation matrices (Pearson) miss tail dependence
   │   Copula models essential but complex
   ├─ Static vs Dynamic:
   │   CVaR assumes stable distribution
   │   Regime shifts (vol clustering) require dynamic CVaR
   ├─ Alternatives:
   │   Expected Shortfall variations (component ES)
   │   Spectral risk measures (flexible weighting of tail)
   │   Risk parity / diversification approaches
   └─ Future Directions:
       Machine learning for tail prediction
       Causal models linking risk factors to tail events
       Real-time CVaR monitoring
```

**Interaction:** Distribution tail → VaR threshold → conditional expectation → CVaR calculation → portfolio application.

## 5. Mini-Project
Calculate and compare VaR vs CVaR under different distributions:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.optimize import minimize

# Generate sample returns from different distributions
np.random.seed(42)
n_samples = 10000
alpha = 0.05  # 5% confidence level

# Distribution 1: Normal (thin tails)
returns_normal = np.random.normal(0.0005, 0.01, n_samples)

# Distribution 2: Student's t (fat tails)
df = 5  # degrees of freedom
returns_fat = np.random.standard_t(df) * 0.01 / np.sqrt(df/(df-2)) + 0.0005

# Distribution 3: Mixture (normal + rare crashes)
normal_prob = 0.99
crash_prob = 0.01
returns_mixture = np.where(np.random.random(n_samples) < normal_prob,
                           np.random.normal(0.0005, 0.01, n_samples),
                           np.random.normal(-0.05, 0.02, n_samples))

distributions = {
    'Normal': returns_normal,
    'Fat-Tailed (t)': returns_fat,
    'Mixture (Crashes)': returns_mixture
}

# Calculate VaR and CVaR
results = {}

for name, returns in distributions.items():
    # VaR: α-th percentile
    var = np.percentile(returns, alpha * 100)
    
    # CVaR: average of returns ≤ VaR
    cvar = returns[returns <= var].mean()
    
    # Parametric CVaR (assuming normal)
    z_alpha = norm.ppf(alpha)
    mu, sigma = returns.mean(), returns.std()
    cvar_param = mu - sigma * norm.pdf(z_alpha) / alpha
    
    results[name] = {
        'VaR': var,
        'CVaR': cvar,
        'CVaR_Param': cvar_param,
        'Excess': cvar - var,
    }

# Print comparison
print("="*70)
print("VaR vs CVaR COMPARISON (95% confidence, α=0.05)")
print("="*70)
print(f"\n{'Distribution':<20} {'VaR':<12} {'CVaR':<12} {'Excess':<12} {'Excess %':<12}")
print("-"*70)

for name, stats in results.items():
    var = stats['VaR']
    cvar = stats['CVaR']
    excess = stats['Excess']
    excess_pct = (excess / var * 100) if var != 0 else 0
    print(f"{name:<20} {var:<12.4f} {cvar:<12.4f} {excess:<12.4f} {excess_pct:<12.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution comparison (histograms)
ax = axes[0, 0]
for name, returns in distributions.items():
    ax.hist(returns, bins=50, alpha=0.5, label=name, density=True)
ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax.set_title('Return Distributions')
ax.set_xlabel('Return')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Tail comparison (zoomed)
ax = axes[0, 1]
quantiles = np.linspace(0, 0.1, 50)
for name, returns in distributions.items():
    tail_values = [np.percentile(returns, q * 100) for q in quantiles]
    ax.plot(quantiles * 100, tail_values, linewidth=2, marker='o', markersize=3, label=name)
ax.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='α=5%')
ax.set_title('Left Tail Quantiles (Worst Returns)')
ax.set_xlabel('Percentile (%)')
ax.set_ylabel('Return Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: VaR vs CVaR bar chart
ax = axes[1, 0]
names_list = list(results.keys())
vars_vals = [results[name]['VaR'] for name in names_list]
cvars_vals = [results[name]['CVaR'] for name in names_list]

x = np.arange(len(names_list))
width = 0.35
ax.bar(x - width/2, vars_vals, width, label='VaR (5%)', color='steelblue')
ax.bar(x + width/2, cvars_vals, width, label='CVaR (5%)', color='darkred')
ax.set_ylabel('Loss (Return)')
ax.set_title('VaR vs CVaR Comparison')
ax.set_xticks(x)
ax.set_xticklabels(names_list, rotation=15, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Excess over VaR (tail severity)
ax = axes[1, 1]
excesses = [results[name]['Excess'] for name in names_list]
ax.bar(names_list, excesses, color=['green' if e > 0 else 'red' for e in excesses], alpha=0.7)
ax.set_ylabel('CVaR - VaR (Tail Severity)')
ax.set_title('Excess Loss Beyond VaR')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Additional: Coherence check (subadditivity)
print("\n" + "="*70)
print("COHERENCE CHECK: Diversification Reduces CVaR")
print("="*70)

# Create portfolio of Normal + Fat-tailed
weights = [0.5, 0.5]
portfolio_returns = weights[0] * (returns_normal / returns_normal.std()) + \
                   weights[1] * (returns_fat / returns_fat.std())

portfolio_var = np.percentile(portfolio_returns, alpha * 100)
portfolio_cvar = portfolio_returns[portfolio_returns <= portfolio_var].mean()

weighted_cvar = weights[0] * results['Normal']['CVaR'] + \
                weights[1] * results['Fat-Tailed (t)']['CVaR']

print(f"Normal Distribution CVaR: {results['Normal']['CVaR']:.4f}")
print(f"Fat-Tailed Distribution CVaR: {results['Fat-Tailed (t)']['CVaR']:.4f}")
print(f"Weighted Sum (should be ≥ Portfolio): {weighted_cvar:.4f}")
print(f"Portfolio CVaR (50/50 blend): {portfolio_cvar:.4f}")
print(f"Subadditivity holds: {portfolio_cvar <= weighted_cvar}")
```

## 6. Challenge Round
- Derive closed-form CVaR for normal distribution
- Implement CVaR-minimizing portfolio optimization (convex program)
- Compare CVaR estimates: parametric vs historical vs EVT methods
- Design backtesting procedure for CVaR model validation
- Explain why VaR violates subadditivity; construct counterexample

## 7. Key References
- [Rockafellar & Uryasev, "Optimization of CVaR" (2000)](https://www.jstor.org/stable/3646191) — Foundational convex formulation
- [Acerbi & Tasche, "On the Coherence of Expected Shortfall" (2002)](https://www.jstor.org/stable/2961178) — Coherence properties
- [Basel III Regulatory Capital Framework](https://www.bis.org/bcbs/) — CVaR in regulation
- [McNeil et al, "Quantitative Risk Management" (2015)](https://www.cambridge.org/core/books/) — Comprehensive treatment

---
**Status:** Coherent tail risk measure | **Complements:** VaR, Semi-Variance, Risk Parity, Portfolio Optimization
