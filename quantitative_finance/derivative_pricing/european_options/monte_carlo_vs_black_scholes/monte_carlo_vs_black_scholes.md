# Monte Carlo vs Black-Scholes

## Concept Skeleton
**Definition:** Comparison of simulation-based (Monte Carlo) and analytical (Black-Scholes) methods for option pricing  
**Purpose:** Understand computational trade-offs, accuracy, flexibility; select appropriate method for payoff complexity  
**Prerequisites:** Black-Scholes formula, Monte Carlo convergence, law of large numbers, computational complexity

## Comparative Framing
| Aspect | Black-Scholes (Analytical) | Monte Carlo (Simulation) |
|--------|---------------------------|--------------------------|
| **Computation Time** | O(1) - instant | O(N paths) - scales linearly |
| **Accuracy** | Exact (under assumptions) | O(1/√N) convergence; statistical error |
| **Flexibility** | European vanilla only | Any payoff (path-dep, exotic) |
| **Greeks** | Closed-form derivatives | Finite diff, pathwise, LR method |
| **Assumptions** | Constant σ, r; lognormal | Arbitrary dynamics (jumps, stoch-vol) |
| **Multidimensional** | Infeasible (no closed-form) | Linear in dimensions (curse avoided) |

## Examples + Counterexamples
**BS Preferred:**  
European call/put on single stock; need instant quotes for thousands of options; Greeks required for hedging

**MC Preferred:**  
Asian option (path-dependent average); basket option (5+ assets with correlation); barrier option with monitoring

**Failure Case:**  
American option pricing: BS inapplicable (no early exercise); MC requires Longstaff-Schwartz (regression, not simple simulation)

## Layer Breakdown
```
Method Selection Decision Tree:
├─ Option Type:
│   ├─ European Vanilla (call/put):
│   │   ├─ Single Asset, Constant Vol → Black-Scholes (instant, exact)
│   │   └─ Complex Model (jumps, stoch-vol) → Monte Carlo (flexible)
│   ├─ Path-Dependent (Asian, lookback, barrier):
│   │   └─ Monte Carlo (only feasible method for many exotic payoffs)
│   ├─ American / Bermudan:
│   │   ├─ Binomial Tree (discrete time)
│   │   └─ Monte Carlo with LSM (Longstaff-Schwartz)
│   └─ Multi-Asset (basket, spread, rainbow):
│       ├─ 2-3 assets: Analytical approximations possible
│       └─ 4+ assets: Monte Carlo (dimension-independent complexity)
├─ Computational Requirements:
│   ├─ Real-Time Pricing (trading desk):
│   │   ├─ BS: Microseconds for Greeks + price
│   │   └─ MC: Milliseconds (1,000 paths) to seconds (1M paths)
│   ├─ Risk Management (overnight batch):
│   │   └─ MC acceptable: Compute 100k scenarios with variance reduction
│   └─ Model Calibration (iterative):
│       └─ BS preferred: Fast repeated evaluations for optimizer
├─ Accuracy Comparison:
│   ├─ BS Error Sources:
│   │   ├─ Model Misspecification: Real markets have vol smile/skew
│   │   ├─ Parameter Estimation: Historical σ ≠ implied σ
│   │   └─ Continuous Trading: Transaction costs, discrete hedging
│   ├─ MC Error Sources:
│   │   ├─ Statistical Error: SE = σ_payoff / √N → 95% CI = ±1.96 SE
│   │   ├─ Time Discretization: Euler scheme O(Δt) bias
│   │   └─ Random Seed: Different runs give different prices (reproducible with seed)
│   └─ Convergence Speed:
│       ├─ Standard MC: O(N^(-0.5)) - halve error → 4× paths
│       ├─ Quasi-MC (Sobol): O(N^(-1) log^d N) - faster for smooth payoffs
│       └─ Variance Reduction: 2-10× fewer paths for same accuracy
├─ Greeks Computation:
│   ├─ BS Greeks:
│   │   ├─ Closed-Form: Δ = N(d₁), Γ = n(d₁)/(Sσ√T), ν = S√T n(d₁)
│   │   └─ Instant Evaluation: No additional computation
│   ├─ MC Greeks:
│   │   ├─ Finite Difference: Δ ≈ (C(S+ε) - C(S-ε))/(2ε) - requires 2× simulations
│   │   ├─ Pathwise Derivative: Compute ∂Payoff/∂S along each path - efficient
│   │   └─ Likelihood Ratio: Multiply payoff by score function - works for discontinuous payoffs
│   └─ Accuracy: BS exact; MC Greeks have higher variance than price estimates
└─ When to Switch from BS to MC:
    ├─ Payoff Complexity: Path-dependence (Asian, barrier, lookback)
    ├─ Model Complexity: Jump-diffusion, stochastic volatility, local volatility
    ├─ High Dimensions: 5+ correlated assets (BS has no closed-form)
    └─ Custom Payoffs: Structured products, exotic derivatives
```

**Interaction:** Evaluate payoff type → Check model assumptions → Choose method → Implement with error bounds

## Challenge Round
**Q1:** Why does MC have O(1/√N) convergence while quasi-MC achieves O(N^(-1))? What's the catch?  
**A1:** Standard MC: CLT gives SE = σ/√N (random samples). Quasi-MC (Sobol, Halton): Low-discrepancy sequences cover space uniformly → deterministic error bound O(N^(-1)(log N)^d) for smooth integrands. Catch: Requires smooth payoffs (no discontinuities like digital options); high dimensions (d > 10) degrade performance.

**Q2:** For high-dimensional basket options (10 assets), why does MC outperform finite difference?  
**A2:** Finite difference (PDE): Grid size grows as K^d (d dimensions, K points per axis) → curse of dimensionality. MC: Sample paths in d-dimensional space; complexity O(N paths) independent of d. For d > 3, MC far superior.

**Q3:** Greeks via finite difference require multiple MC runs (bump-and-revalue). How does pathwise derivative method fix this?  
**A3:** Pathwise: Compute ∂Payoff/∂S analytically along each path (e.g., ∂max(S_T - K, 0)/∂S = 1_{S_T > K}). Average ∂Payoff/∂S over paths → delta from single simulation. Efficient but fails for discontinuous payoffs (barrier breach).

**Q4:** Compare MC accuracy for European call vs Asian call (same # paths). Which has lower error?  
**A4:** European call: Higher variance (payoff = max(S_T - K, 0) varies widely). Asian call: Lower variance (averaging reduces fluctuations; payoff smoother). Asian SE typically 30-50% lower for same N → faster convergence.

**Q5:** When is binomial tree preferred over both BS and MC?  
**A5:** American options with early exercise: Binomial backward induction optimal (exact as N → ∞). BS inapplicable (no early exercise). MC requires LSM (regression overhead). Tree also good for dividend adjustments, transparent for teaching.

**Q6:** Implement variance reduction for European call. Show antithetic variates halve variance empirically.  
**A6:** Standard MC: Var(Price) = σ²_payoff / N. Antithetic (Z, -Z pairs): Corr(Payoff(Z), Payoff(-Z)) < 0 → Var_AV ≈ Var/2 (if correlation ≈ -1). Empirically: σ²_AV / σ²_standard ≈ 0.45-0.55 for vanilla options.

**Q7:** Why does BS fail for barrier options even if payoff is European (exercise at T only)?  
**A7:** BS assumes terminal payoff depends only on S_T (Markovian). Barrier: Payoff depends on entire path (knocked out if S_t crosses barrier before T) → path-dependent. No closed-form under GBM (some approximations exist). MC handles naturally by monitoring path.

**Q8:** Calibrate BS volatility to market prices. Why does MC NOT replace BS for calibration?  
**A8:** Calibration requires repeated pricing (optimizer calls objective 100+ times). MC per call: 100ms (10k paths). Total: 10+ seconds. BS per call: 0.1ms. Total: 10ms. Speed difference: 1000×. Use BS for calibration; MC only for final pricing with calibrated parameters.

## Key References
**Primary Sources:**
- Glasserman, P. *Monte Carlo Methods in Financial Engineering* (2004) - Comprehensive MC techniques (Chapters 1-5)
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - BS vs numerical methods (Chapter 21)
- [Monte Carlo in Finance Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance) - Overview

**Technical Details:**
- Boyle, P. "Options: A Monte Carlo Approach" (1977) - Original MC option pricing paper
- L'Ecuyer, P. & Lemieux, C. "Variance Reduction via Lattice Rules" (2000) - Quasi-MC for finance

**Thinking Steps:**
1. Identify payoff type: European vanilla (BS) vs path-dependent (MC)
2. Check model assumptions: Constant vol (BS) vs jumps/stoch-vol (MC)
3. Evaluate dimensions: Single asset (BS fast) vs basket 5+ assets (MC scales)
4. Consider speed requirements: Real-time (BS) vs overnight (MC acceptable)
5. Assess accuracy needs: BS exact under assumptions; MC has statistical error O(1/√N)
6. For exotic payoffs with no closed-form: MC only viable method
