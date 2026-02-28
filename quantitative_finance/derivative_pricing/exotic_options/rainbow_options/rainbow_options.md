# Rainbow Options

## Concept Skeleton
**Definition:** Multi-asset options with payoff on best/worst/spread of N underlyings; maximum/minimum selection  
**Purpose:** Portfolio diversification; correlation trading; best-of-best strategies; worst-case hedging  
**Prerequisites:** Multivariate simulation, Cholesky decomposition, order statistics, high-dimensional Monte Carlo

## Comparative Framing
| Feature | Rainbow (Best-of) | Rainbow (Worst-of) | Basket | Spread | Single European |
|---------|-------------------|-------------------|--------|--------|-----------------|
| **Payoff** | max(max(S_i) - K, 0) | max(min(S_i) - K, 0) | max(Σw_i S_i - K, 0) | max(S₁ - S₂ - K, 0) | max(S - K, 0) |
| **Correlation Impact** | Negative (diversification) | Positive (co-movement) | Moderate | High | N/A |
| **Value** | Most expensive | Cheap | Moderate | Correlation-dependent | Baseline |
| **Pricing** | Monte Carlo | Monte Carlo | Monte Carlo | Monte Carlo | Black-Scholes |
| **Complexity** | O(N log N) sorting | O(N log N) sorting | O(N) | O(2) | O(1) |

## Examples + Counterexamples
**Simple Example:**  
Best-of-3 call: Assets A, B, C → Payoff = max(max(S_A, S_B, S_C) - K, 0); investor gets upside of best performer

**Failure Case:**  
Perfect correlation (ρ=1): All assets move identically → rainbow = single-asset option → pays premium for no benefit

**Edge Case:**  
Worst-of put: Pays if ANY asset drops below K → diversification becomes risk (probability N× higher) → expensive hedge

## Layer Breakdown
```
Rainbow Option Pricing Pipeline:
├─ Rainbow Types:
│   ├─ Best-of Call: max(max(S₁, S₂, ..., Sₙ) - K, 0)
│   │   └─ Upside of best performer → most expensive
│   ├─ Worst-of Call: max(min(S₁, S₂, ..., Sₙ) - K, 0)
│   │   └─ Limited by weakest asset → cheaper
│   ├─ Best-of Put: max(K - min(S₁, S₂, ..., Sₙ), 0)
│   │   └─ Pays if ANY asset drops → expensive hedge
│   ├─ Worst-of Put: max(K - max(S₁, S₂, ..., Sₙ), 0)
│   │   └─ Requires ALL assets drop → cheap
│   ├─ Nth-to-Default: Triggered by nth worst performer
│   └─ Rainbow Spread: max(S_best - S_worst, 0)
├─ Multi-Asset Simulation:
│   ├─ Asset Dynamics (each i=1..N):
│   │   ├─ dS_i = r S_i dt + σ_i S_i dW_i
│   │   ├─ Discrete: S^i_{t+1} = S^i_t exp((r - σ_i²/2)dt + σ_i√dt Z_i)
│   │   └─ Different vols σ_i for each asset
│   ├─ Correlation Structure:
│   │   ├─ Correlation Matrix ρ: ρᵢⱼ = Corr(S_i, S_j)
│   │   ├─ Cholesky: ρ = LL^T → Correlated normals X = LZ
│   │   └─ Z ~ N(0, I) independent → X ~ N(0, ρ) correlated
│   ├─ Path Generation:
│   │   ├─ For each time step:
│   │   │   ├─ Generate Z = [Z₁, ..., Zₙ] ~ N(0, I)
│   │   │   ├─ X = LZ (apply Cholesky for correlation)
│   │   │   └─ S^i_{t+1} = S^i_t exp((r - σ_i²/2)dt + σ_i√dt X_i)
│   │   └─ Store terminal prices [S₁_T, ..., Sₙ_T] per path
│   └─ Terminal Selection:
│       ├─ Best: S_max = max(S₁_T, ..., Sₙ_T)
│       ├─ Worst: S_min = min(S₁_T, ..., Sₙ_T)
│       └─ Sort: S_(1) ≤ S_(2) ≤ ... ≤ S_(N) (order statistics)
├─ Payoff Calculation:
│   ├─ Best-of Call: max(S_max - K, 0)
│   ├─ Worst-of Call: max(S_min - K, 0)
│   ├─ Best-of Put: max(K - S_min, 0) (pays if ANY drops)
│   ├─ Worst-of Put: max(K - S_max, 0) (pays if ALL drop)
│   └─ Present Value: V = e^{-rT} E[Payoff]
├─ Correlation Impact:
│   ├─ Best-of Options:
│   │   ├─ Low Correlation (ρ → 0):
│   │   │   ├─ Diversification → high chance one asset performs well
│   │   │   ├─ P(at least one ITM) increases
│   │   │   └─ Option MORE expensive (value increases)
│   │   ├─ High Correlation (ρ → 1):
│   │   │   ├─ Assets move together → no diversification benefit
│   │   │   └─ Option LESS expensive (approaches single-asset)
│   │   └─ Value: V(ρ=0) > V(ρ=0.5) > V(ρ=1) ≈ Single Asset
│   ├─ Worst-of Options:
│   │   ├─ Low Correlation:
│   │   │   ├─ Independent movements → likely one lags
│   │   │   └─ Option LESS expensive (worst asset likely poor)
│   │   ├─ High Correlation:
│   │   │   ├─ Co-movement → if one up, all up
│   │   │   └─ Option MORE expensive (worst not so bad)
│   │   └─ Value: V(ρ=1) > V(ρ=0.5) > V(ρ=0)
│   └─ Opposite Effects: Best-of and worst-of have inverse correlation sensitivity
├─ Order Statistics:
│   ├─ Max Distribution (Best-of):
│   │   ├─ F_max(x) = P(max(S_i) ≤ x) = Π F_i(x) (product of CDFs)
│   │   ├─ Independent case: F_i identical → F_max = F^N
│   │   └─ Tail probability: P(S_max > K) = 1 - Π P(S_i ≤ K)
│   ├─ Min Distribution (Worst-of):
│   │   ├─ F_min(x) = P(min(S_i) ≤ x) = 1 - Π (1 - F_i(x))
│   │   └─ P(S_min > K) = Π P(S_i > K) (all must be above K)
│   └─ Correlation complicates: No closed-form with dependence
├─ Greeks:
│   ├─ Deltas: ∂V/∂S_i (vector of N deltas)
│   │   ├─ Best-of: Highest delta on currently leading asset
│   │   ├─ Worst-of: Highest delta on currently lagging asset
│   │   └─ Discontinuous at crossing points (S_i = S_j)
│   ├─ Cross-Gammas: ∂²V/∂S_i∂S_j
│   │   ├─ Positive for best-of (substitution effect)
│   │   └─ Negative for worst-of (competition effect)
│   ├─ Vega: ∂V/∂σ_i
│   │   ├─ Best-of: High vega (volatility increases upside)
│   │   ├─ Worst-of: Lower vega (volatility helps, but less than single)
│   │   └─ Per-asset vega depends on moneyness
│   ├─ Correlation Greeks (Cega): ∂V/∂ρᵢⱼ
│   │   ├─ Best-of: Negative (lower ρ → more value)
│   │   ├─ Worst-of: Positive (higher ρ → more value)
│   │   └─ Critical for correlation risk management
│   └─ Hedging Challenges:
│       ├─ Switching: Leading asset changes → delta jumps
│       ├─ High dimensionality: N×N Greeks matrix
│       └─ Correlation risk: Hard to hedge (no liquid instruments)
├─ Variance Reduction:
│   ├─ Antithetic Variates:
│   │   ├─ Z and -Z → Negatively correlated max/min
│   │   └─ Preserves Cholesky structure: LZ and L(-Z)
│   ├─ Control Variates:
│   │   ├─ Use basket option (has moment-matching approx)
│   │   ├─ Or use single-asset European with avg volatility
│   │   └─ Correlation 0.6-0.8 typical
│   ├─ Importance Sampling:
│   │   ├─ Shift drift toward region where max > K
│   │   └─ Effective for OTM rainbow options
│   └─ Stratification:
│       └─ Stratify on terminal maximum (best-of) or minimum (worst-of)
├─ Approximations:
│   ├─ Johnson's Bound:
│   │   ├─ Best-of Call ≤ Σ Call_i (sum of individual options)
│   │   ├─ Worst-of Call ≥ max(Basket - K, 0) (basket lower bound)
│   │   └─ Useful for quick checks, not tight
│   ├─ Moment Matching:
│   │   ├─ Approximate max distribution as lognormal
│   │   ├─ Match E[S_max] and Var[S_max]
│   │   └─ Accurate for high correlation
│   ├─ Copula Methods:
│   │   ├─ Model marginals separately from dependence structure
│   │   ├─ Use Gaussian copula (equivalent to Cholesky)
│   │   └─ Or Student-t copula (tail dependence)
│   └─ Kirk's Approximation:
│       └─ For spread options (S₁ - S₂): Approximate as single lognormal
└─ Practical Applications:
    ├─ Altiplano/Himalaya: Sequentially remove best performer (exotic structure)
    ├─ Best-of-Best: Two-level rainbow (e.g., best of 3 regions)
    ├─ Nth-to-Default CDS: Credit derivatives (rainbow on default times)
    ├─ Talent Hedge: Best employee among N candidates
    └─ Natural Resource: Best well/field production
```

**Interaction:** Generate correlated asset paths via Cholesky → Compute max/min at maturity → Apply payoff function → Discount to present

## Challenge Round
**Q1:** Why is best-of call MORE expensive with LOWER correlation?  
**A1:** Low correlation → assets move independently → higher chance at least one performs well → more diversification benefit. High correlation → assets co-move → no diversification → similar to single asset. Math: P(max > K) = 1 - Π P(S_i ≤ K); independent: probabilities multiply → higher probability of max being large.

**Q2:** Worst-of put as portfolio insurance: Why expensive?  
**A2:** Pays if ANY asset drops below K → N independent risks → probability of payout ≈ N × single-asset probability (low correlation). Diversification works AGAINST holder: More assets = more ways to lose. Used for worst-case hedging: Protects against scenario where at least one investment fails.

**Q3:** Derive upper bound: Best-of call ≤ Sum of individual calls. Tight?  
**A3:** E[max(max(S_i) - K, 0)] ≤ E[Σ max(S_i - K, 0)] by max ≤ sum. Not tight: RHS counts multiple payoffs, LHS only largest. Equality only if perfect correlation (all move identically). Gap largest for low correlation (diversification makes best-of much cheaper than sum).

**Q4:** Greeks discontinuity: Delta jumps when leading asset changes. Hedging implications?  
**A4:** At S₁ = S₂ crossing: Δ₁ suddenly drops, Δ₂ jumps up as leader changes. Creates hedging challenge: Frequent rebalancing needed near crossings. Cross-gamma ∂²V/∂S₁∂S₂ is negative spike (substitution effect). Practical: Use smooth approximation or dynamic hedging strategy.

**Q5:** Nth-to-default CDS: Triggered by 2nd default among 5 entities. Pricing vs correlation?  
**A5:** High correlation: Defaults cluster → if one defaults, others likely follow → 2nd default soon → expensive. Low correlation: Independent defaults → 2nd default rare → cheaper. Credit crisis: Correlation often underestimated (assumed 0.3, realized 0.8) → massive mispricing → 2008 losses.

**Q6:** Altiplano: Best-of N assets; each year remove best performer and reset. Complexity?  
**A6:** Path-dependent: Which asset removed depends on history → state space explodes. Need track removed set (2^N states). Pricing: Monte Carlo with careful bookkeeping of active assets. Early exercise: Best performers removed → remaining assets have lower expected growth → option value declines over time.

**Q7:** Compare rainbow to spread option (S₁ - S₂ - K). Which has higher correlation risk?  
**A7:** Spread: max(S₁ - S₂ - K, 0) → payoff directly depends on S₁ - S₂ → Cega very high (correlation dominates value). Rainbow (best-of): Correlation affects max distribution but less directly. Spread: Used explicitly for correlation trading; rainbow: Correlation is secondary to selection benefit. Spread has higher absolute Cega.

**Q8:** Dimension reduction for N=100 rainbow: PCA approach?  
**A8:** PCA: Assets = Σ w_k Factor_k; keep K≪N factors explaining 90% variance. Simulate K factors (uncorrelated), reconstruct N assets. Max selection: Only need factor loadings, not full covariance. Speeds up: O(K) vs O(N²) Cholesky. Loses accuracy if tail dependence important (non-Gaussian copulas).

## Key References
**Primary Sources:**
- [Rainbow Option Wikipedia](https://en.wikipedia.org/wiki/Rainbow_option) - Definitions and structures
- Johnson, N. & Kotz, S. "Continuous Multivariate Distributions" (1972) - Order statistics

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods* (2004) - Multi-asset simulation (pp. 101-125)
- Deelstra, G. et al. "Pricing of Basket Options" (2004) - Correlation impact on multi-asset options

**Thinking Steps:**
1. Define rainbow type (best/worst-of, call/put) and correlation structure
2. Cholesky decomposition for correlated asset paths
3. Simulate N correlated terminal prices per path
4. Select max or min depending on option type
5. Apply payoff function to selected price; discount to present
6. Analyze correlation sensitivity (Cega) and Greeks
