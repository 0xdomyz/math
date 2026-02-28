# Asian Options

## Concept Skeleton
**Definition:** Options whose payoff depends on the average price of the underlying asset over a specified period, not the final price; includes arithmetic and geometric averages  
**Purpose:** Reduce payoff volatility for hedging purposes; lower cost than vanilla (averaging dampens); match cash flow averaging (e.g., corporate expenses)  
**Prerequisites:** Path-dependent options, Monte Carlo methods, moment matching, convexity bias, stochastic integration

## Comparative Framing
| Type | Payoff | Mathematical | Closed-Form | Cost vs Vanilla |
|------|--------|--------------|-------------|-----------------|
| **Arithmetic-average call** | (A-K)⁺ | A = (1/n)∑Sᵢ | No (numerical) | 20-30% cheaper |
| **Geometric-average call** | (G-K)⁺ | G = (∏Sᵢ)^(1/n) | Yes (BS variant) | Similar to arith |
| **Average-strike call** | (S_T - A)⁺ | Payoff: Final vs avg | No | Highly variable |
| **Float-strike put** | (A - S_T)⁺ | Payoff: Avg vs final | No | Depends on path |
| **European vanilla call** | (S_T - K)⁺ | Single final price | Yes (BS) | Baseline | 

## Examples + Counterexamples
**Simple Asian Call Success:**  
Spot=$100, strike=$100, T=1yr, σ=30% (volatile). Vanilla call ~$18. Asian call (arithmetic) ~$12-14. Averaging dampens volatility effect, cheaper protection for commodity/oil users averaging consumption costs.

**Geometric Asian Closed-Form Win:**  
Exact closed-form solution: G ~ Lognormal with adjusted volatility σ_G = σ/√3. Allows instant pricing vs Monte Carlo.

**Arithmetic > Geometric (Inequality):**  
By AM-GM inequality: (A ≥ G always), so Asian call (arithmetic) ≥ Asian call (geometric) in expectation.

**Average-Strike Put Volatility Play:**  
S(0)=$100, ends S(T)=$110 (up 10%), but average A=$95. Put payoff = $95-$110 = 0 (ITM average, but OTM final). Betting volatility hurts buyer; extreme paths reduce average.

**Corporate Hedge Example:**  
Oil company monthly expenses $10/bbl. Oil spot=$60-$70 (volatile). Arithmetic average Asian puts protect average cost; cheaper than series of vanilla puts.

## Layer Breakdown
```
Asian Options Framework:

├─ Averaging Mechanics:
│  ├─ Arithmetic Average:
│  │   ├─ Definition: A = (1/n) ∑ᵢ₌₁ⁿ S_tᵢ
│  │   ├─ Observation dates: t₁, t₂, ..., tₙ
│  │   ├─ Averaging window: [t_start, t_end] ⊆ [0,T]
│  │   ├─ Frequency: Daily, weekly, monthly, or continuous
│  │   └─ Path-dependent: Full path history required
│  ├─ Geometric Average:
│  │   ├─ Definition: G = (∏ᵢ₌₁ⁿ S_tᵢ)^(1/n)
│  │   ├─ Log-additive: ln(G) = (1/n) ∑ln(S_tᵢ)
│  │   ├─ Advantage: Closed-form in BS model
│  │   ├─ G ≤ A always (Jensen's inequality)
│  │   └─ Interpretation: Continuous geometric ~ stock price lognormal
│  ├─ Averaging Horizons:
│  │   ├─ Full period: A based on [0,T]
│  │   ├─ Delayed start: A from t_start > 0
│  │   ├─ Multiple lookback windows: Complex structure
│  │   └─ Weighted average: ∑ wᵢ Sᵢ (favors recent prices)
│  └─ Frequency Effects:
│      ├─ Continuous: Mathematical ideal (limit of n→∞)
│      ├─ Daily: Practical standard
│      ├─ Weekly/Monthly: Coarser sampling, cheaper to compute
│      └─ Discretization bias: Arithmetic > daily > weekly prices
├─ Payoff Structures:
│  ├─ Fixed-Strike Asian Call:
│  │   ├─ Payoff: max(A - K, 0)
│  │   ├─ Buyer: Benefits from low average
│  │   ├─ Use: Oil/commodity importer averaging cost
│  │   ├─ vs vanilla: Average path matters, final less important
│  │   └─ Price: Lower than vanilla (volatility averaging effect)
│  ├─ Fixed-Strike Asian Put:
│  │   ├─ Payoff: max(K - A, 0)
│  │   ├─ Buyer: Protection against high average price
│  │   ├─ Use: Corporate buying (e.g., production line)
│  │   └─ Value: Depends on both average and final S
│  ├─ Average-Strike Call (Floating Strike):
│  │   ├─ Payoff: max(S_T - A, 0)
│  │   ├─ Buyer: Benefits if final S > average (rising trend)
│  │   ├─ Seller: Short; profits if trend down/flat
│  │   └─ Payoff: Can be 0 even if S_T > K (if A high)
│  ├─ Average-Strike Put (Floating Strike):
│  │   ├─ Payoff: max(A - S_T, 0)
│  │   ├─ Buyer: Protection against downtrend (falling S final)
│  │   ├─ Payoff: Compares average to final, asymmetric
│  │   └─ Often in equity warrants, commodity swaps
│  └─ Complex variants:
│      ├─ Partial averaging: A based on subset of dates
│      ├─ Reset Asian: Multiple reset dates, averaging resets
│      └─ Capped Asian: max(A - K, C) for cap C
├─ Valuation Approaches:
│  ├─ Exact Closed-Form (Geometric):
│  │   ├─ Continuous geometric average:
│  │   │   S̄_G = exp[(1/T) ∫₀ᵀ ln(S_t) dt]
│  │   ├─ Under GBM: S̄_G ~ Lognormal(μ', σ')
│  │   │   where σ'² = σ²/3 (variance reduction by factor 3)
│  │   ├─ Adjusted: Use BS with σ_adj = σ/√3
│  │   ├─ Advantage: Instant calculation
│  │   └─ Limitation: Only geometric, continuous
│  ├─ Monte Carlo (Arithmetic - Primary Method):
│  │   ├─ Simulate path S(t)
│  │   ├─ At observation dates: Record S_i
│  │   ├─ Compute arithmetic average: A = (1/n) ∑ Sᵢ
│  │   ├─ Payoff: (A - K)⁺ discounted
│  │   ├─ Strengths: Handles arbitrary averaging, simple
│  │   ├─ Weaknesses: Slow convergence (O(1/√M) error)
│  │   └─ Variance reduction: Control variate vs geometric
│  ├─ Moment-Matching (Turnbull & Wakeman):
│  │   ├─ Approximate A as lognormal
│  │   ├─ Match first two moments:
│  │   │   E[A] = E[S] (exact under martingale)
│  │   │   Var[A] = Var[∑ Sᵢ] (computable)
│  │   ├─ Adjust BS parameters to matched distribution
│  │   ├─ Fast: Closed-form with adjusted σ, μ
│  │   ├─ Accuracy: Good for ATM, weaker for extreme strikes
│  │   └─ Trade-off: Speed vs accuracy
│  ├─ Curran's Approximation:
│  │   ├─ Condition on average: E[C|A=a]
│  │   ├─ Decompose: E[C] = ∫ E[C|A=a] p(a) da
│  │   ├─ Inner conditional: Quasi-closed-form
│  │   ├─ Accuracy: Better than moment matching
│  │   └─ Complexity: Numerical integration needed
│  ├─ Finite Difference (PDE):
│  │   ├─ Extended state space: (S, A, t)
│  │   ├─ PDE: ∂V/∂t + rS(∂V/∂S) + 0.5σ²S²(∂²V/∂S²) = rV
│  │   ├─ with ∂V/∂A constraint (averaging rate)
│  │   ├─ Advantage: Greeks exact
│  │   ├─ Limitation: 3D grid slow; curse of dimensionality
│  │   └─ Use: High-accuracy benchmarks
│  └─ Numerical Integration (Asian Bounds):
│      ├─ Arithmetic Asian ≤ Vanilla (upper bound)
│      ├─ Geometric Asian ≤ Arithmetic Asian (lower bound)
│      └─ Use: Validation sanity checks
├─ Path-Dependent Complications:
│  ├─ State Space Expansion:
│  │   ├─ Vanilla: V(S, t) only depends on spot
│  │   ├─ Asian: V(S, A, t) must track average too
│  │   ├─ Lattice: Nodes multiplied by #averaging dates
│  │   └─ FD: Extra dimension makes solver 10-100x slower
│  ├─ Monte Carlo Advantage:
│  │   ├─ No expansion: Just track running sum A
│  │   ├─ Parallelizable paths independently
│  │   ├─ Scales better than FD for many dimensions
│  │   └─ Limitation: √M convergence slower than FD
│  ├─ Averaging Frequency Effects:
│  │   ├─ Continuous (n→∞):  Theoretical limit
│  │   ├─ Discrete (daily n=252): Standard market
│  │   ├─ Coarse (monthly n=12): Cheaper computation
│  │   └─ Bias: Coarser averaging → higher price (less averaging effect)
│  └─ Starting Average Conditions:
│      ├─ A_0 known if averaging started before trade
│      ├─ A_0 = starting price if averaging begins today
│      ├─ Affects option value; tracks realized-to-date
│      └─ Critical for corporate hedges mid-period
├─ Greeks & Risk Management:
│  ├─ Delta:
│  │   ├─ Lower than vanilla (averaging reduces sensitivity)
│  │   ├─ Depends on both S and A
│  │   ├─ Complex: Average history matters
│  │   ├─ Rebalancing: More frequent with path-dependence
│  │   └─ Numerical: Bump S and A separately
│  ├─ Gamma:
│  │   ├─ Lower than vanilla (volatility smoothing)
│  │   ├─ Benefits long average-strike (low gamma cost)
│  │   ├─ Gamma-dealer can run lower hedge cost
│  │   └─ Risk: Convexity nonlinearity
│  ├─ Vega:
│  │   ├─ Lower than vanilla (averaging reduces vol impact)
│  │   ├─ Volatility still matters, but diminished
│  │   ├─ Long average-strike: Long vega
│  │   └─ High vega regions: Near-ATM A and S
│  ├─ Theta:
│  │   ├─ Depends on average/current relationship
│  │   ├─ Complex decay pattern
│  │   ├─ Theta-Greeks Greeks of Greeks important
│  │   └─ Monitor: As averaging window completes
│  └─ Quanto-delta (A-dependent):
│      ├─ How changes in A affect option value
│      ├─ Important for path tracking
│      └─ Adjustment: Rebalance A hedge separately
└─ Practical Considerations:
   ├─ Corporate Applications:
   │   ├─ Commodity importers: Hedge average COGS
   │   ├─ Manufacturers: Protect margin averaging costs
   │   ├─ Energy traders: Price-averaging swaps
   │   └─ FX: Corporate average cost for conversions
   ├─ Market Conventions:
   │   ├─ Averaging start: Often fix date in past (realized)
   │   ├─ Frequency: Daily standard, weekly for less liquid
   │   ├─ Floor/cap: Minimum/maximum average bounds
   │   └─ Fixings: Official published rates (vs spot)
   ├─ Documentation Issues:
   │   ├─ Averaging method: Arithmetic vs geometric stated
   │   ├─ Starting average: Known A_0 critical
   │   ├─ Observation dates: Schedule must be precise
   │   └─ Settlement: Physical delivery of average-based contract
   └─ Variance Reduction (MC):
       ├─ Control variate: Use geometric Asian as control
       ├─ Antithetic: Pair up paths (Z, -Z)
       ├─ Stratified: Partition average values
       └─ Quasi-random: Deterministic low-discrepancy sequences
```

**Interaction:** Averaging mechanism → path-dependent state space → MC Monte Carlo natural choice → variance reduction essential → Greeks complex but lower than vanilla.

## Challenge Round
- Prove: E[A] = E[S] for arithmetic average (martingale property)
- Show variance reduction: Var[G] = Var[S]/3 for geometric
- Compare moment-matching vs MC accuracy at different moneyness levels
- Design Asian reset coupon (multiple averaging windows)
- Explain why Asian gamma lower than vanilla

## Key References
- [Turnbull & Wakeman, "Fast Algorithm for Pricing American Lookback" (1991)](https://www.jstor.org/stable/2352352) — Moment matching Asian
- [Curran, "Valuing Asian and Portfolio Options" (1994)](https://www.jstor.org/stable/2978589) — Curran approximation
- [Kemma & Vorst, "Numerical Procedure for Valuing Certain Exotic Options" (1990)](https://www.jstor.org/stable/2352569)

---
**Status:** Important commodity/corporate hedge | **Complements:** Geometric Average, Fixed-Strike, Average-Strike, MC Variance Reduction
