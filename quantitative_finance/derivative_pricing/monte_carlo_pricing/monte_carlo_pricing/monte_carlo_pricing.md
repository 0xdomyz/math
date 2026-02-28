# Monte Carlo Pricing

## Concept Skeleton
**Definition:** Stochastic simulation method generating random price paths under risk-neutral measure to estimate option value via discounted expected payoff  
**Purpose:** Price complex path-dependent and multi-dimensional derivatives where closed-forms don't exist; handle exotic features like barriers, lookbacks, Asians  
**Prerequisites:** Risk-neutral valuation, stochastic calculus, geometric Brownian motion, variance reduction techniques, law of large numbers

## Comparative Framing
| Method | Monte Carlo | Binomial Tree | Finite Difference | Closed-Form |
|--------|-------------|---------------|-------------------|-------------|
| **Dimensionality** | Excellent (100+ assets) | Poor (d>3 infeasible) | Poor (curse of dim.) | Limited |
| **Path-Dependent** | Natural | Difficult | Very difficult | Impossible |
| **Accuracy** | O(1/√n) | O(1/n) | O(h²) | Exact |
| **Speed** | Slow (parallel) | Fast (low-dim) | Fast (low-dim) | Instant |
| **American** | Difficult (LSM) | Natural | Natural | N/A |
| **Memory** | O(1) per path | O(n²) | O(n^d) | O(1) |

| Variance Reduction | Antithetic | Control Variate | Importance Sampling | Quasi-Random |
|-------------------|------------|-----------------|---------------------|--------------|
| **Complexity** | Trivial | Moderate | Hard | Easy |
| **Reduction** | 2x | 10-100x | 10-1000x | 10x |
| **Robustness** | High | Medium | Low | High |
| **Implementation** | Flip signs | Need benchmark | Tune distribution | Sobol sequence |

## Examples + Counterexamples
**Simple Example:**  
European call: Simulate 10,000 paths, calculate max(S_T - K, 0) each path, average and discount → converges to Black-Scholes price.

**Perfect Fit:**  
Asian option (average price): Simulate path, compute arithmetic average S̄, payoff = max(S̄ - K, 0). Tree methods require non-recombining tree (2^n nodes) → infeasible.

**Multi-Asset:**  
Rainbow option on 20 stocks (best-of): Generate 20 correlated GBM paths, payoff = max(S₁_T, ..., S₂₀_T) - K. Monte Carlo handles easily, other methods fail.

**Convergence:**  
10² paths: High standard error ~$2. 10⁴ paths: SE ~$0.20. 10⁶ paths: SE ~$0.02. Need 100x more samples for 10x precision.

**Poor Fit:**  
American options: Early exercise decision at each time step requires backward induction. Standard MC goes forward only → need Longstaff-Schwartz regression (complex).

**Bermudan Option:**  
Exercise on specific dates (not continuous). Can use MC with regression at each exercise date, but binomial tree much simpler for few dates.

## Layer Breakdown
```
Monte Carlo Pricing Framework:

├─ Basic Algorithm:
│  ├─ Step 1: Generate random paths under risk-neutral measure
│  │   For each simulation i = 1, ..., N:
│  │   ├─ Draw random numbers Z ~ N(0,1)
│  │   ├─ Evolve: S(t+Δt) = S(t) exp((r - 0.5σ²)Δt + σ√(Δt)Z)
│  │   └─ Store path: {S₀, S₁, ..., S_T}
│  ├─ Step 2: Calculate payoff for each path
│  │   V_i = Payoff(Path_i)
│  ├─ Step 3: Average discounted payoffs
│  │   V̂ = (1/N) Σᵢ e^(-rT) V_i
│  └─ Step 4: Calculate standard error
│      SE = σ̂/√N where σ̂ = sample std dev of payoffs
├─ Path Generation (GBM):
│  ├─ Continuous monitoring (discretized):
│  │   S_{t+Δt} = S_t exp((r - q - 0.5σ²)Δt + σ√Δt Z_t)
│  │   where Z_t ~ N(0,1) i.i.d.
│  ├─ Time steps: Choose Δt small enough
│  │   ├─ Barrier monitoring: Δt < 1/252 (daily or finer)
│  │   ├─ Asian averaging: Match averaging dates
│  │   └─ Trade-off: Accuracy vs computation time
│  ├─ Exact simulation (single time point):
│  │   S_T = S_0 exp((r - q - 0.5σ²)T + σ√T Z)
│  │   No discretization error
│  ├─ Milstein scheme (higher accuracy):
│  │   Includes second-order terms for better convergence
│  └─ Jump-diffusion:
│      Add Poisson jumps: dS = μS dt + σS dW + (J-1)S dN
├─ Multi-Asset Simulation:
│  ├─ Correlated paths:
│  │   Generate independent Z₁, ..., Z_d ~ N(0,1)
│  │   Apply Cholesky: Z_corr = L × Z
│  │   where L L^T = Correlation matrix Σ
│  ├─ Correlation matrix:
│  │   Must be positive semi-definite
│  │   Check eigenvalues ≥ 0
│  ├─ Copula approach:
│  │   Separate marginals from dependence structure
│  │   More flexible than Gaussian correlation
│  └─ Dimension: MC scales linearly in d
│      Unlike PDE/tree methods (exponential)
├─ Payoff Calculation:
│  ├─ European: Only terminal value matters
│  │   Call: max(S_T - K, 0)
│  │   Put: max(K - S_T, 0)
│  ├─ Path-Dependent:
│  │   ├─ Asian (arithmetic average):
│  │   │   Payoff = max((1/n)Σ S_tᵢ - K, 0)
│  │   ├─ Asian (geometric average):
│  │   │   Payoff = max(exp((1/n)Σ ln(S_tᵢ)) - K, 0)
│  │   ├─ Lookback (floating strike):
│  │   │   Payoff = S_T - min(S_t over [0,T])
│  │   ├─ Lookback (fixed strike):
│  │   │   Payoff = max(S_t over [0,T]) - K
│  │   ├─ Barrier (knock-out):
│  │   │   Payoff = max(S_T - K, 0) × 𝟙(S_t < H for all t)
│  │   ├─ Barrier (knock-in):
│  │   │   Payoff = max(S_T - K, 0) × 𝟙(S_t ≥ H for some t)
│  │   └─ Digital (binary):
│  │       Payoff = 𝟙(S_T > K) (0 or 1)
│  └─ Multi-asset:
│      ├─ Basket: max(Σ wᵢ Sᵢ_T - K, 0)
│      ├─ Best-of: max(max(S₁_T, ..., S_d_T) - K, 0)
│      └─ Worst-of: max(min(S₁_T, ..., S_d_T) - K, 0)
├─ Convergence & Error:
│  ├─ Central Limit Theorem:
│  │   V̂ → N(V_true, σ²/N) as N→∞
│  ├─ Standard error:
│  │   SE = σ̂/√N
│  │   95% CI: [V̂ - 1.96×SE, V̂ + 1.96×SE]
│  ├─ Convergence rate: O(1/√N)
│  │   Need 100x paths for 10x precision
│  │   Slow compared to deterministic methods
│  ├─ Bias vs variance:
│  │   ├─ Discretization bias: Δt too large
│  │   ├─ Statistical variance: N too small
│  │   └─ Optimal: Balance both errors
│  └─ Stopping criteria:
│      SE < desired tolerance or max iterations
├─ Variance Reduction Techniques:
│  ├─ Antithetic Variates:
│  │   ├─ For each Z, also simulate -Z
│  │   ├─ Paths are negatively correlated
│  │   ├─ Average reduces variance
│  │   ├─ Effective variance reduction: ~50%
│  │   └─ Cost: None (same # random numbers)
│  ├─ Control Variates:
│  │   ├─ Use correlated instrument with known price
│  │   │   V̂_CV = V̂ - β(Ĉ - C_true)
│  │   │   where C is control (e.g., European call)
│  │   ├─ Optimal β: Cov(V,C) / Var(C)
│  │   ├─ Variance reduction: Proportional to correlation²
│  │   ├─ Example: Use vanilla call to price exotic call
│  │   └─ Can combine multiple controls
│  ├─ Importance Sampling:
│  │   ├─ Change probability measure to focus on critical region
│  │   ├─ Example: OTM option → shift drift to make ITM more likely
│  │   ├─ Reweight: E[f(X)] = E_Q[f(X) × dP/dQ]
│  │   ├─ Radon-Nikodym derivative adjusts for measure change
│  │   ├─ Huge reduction for rare events (barriers, deep OTM)
│  │   └─ Difficult to tune (need domain knowledge)
│  ├─ Stratified Sampling:
│  │   ├─ Divide sample space into strata
│  │   ├─ Sample proportionally from each
│  │   ├─ Ensures coverage of entire range
│  │   └─ Reduces variance within strata
│  ├─ Quasi-Random (Low-Discrepancy) Sequences:
│  │   ├─ Sobol, Halton sequences: Fill space uniformly
│  │   ├─ Avoid clustering of random points
│  │   ├─ Convergence: O(log(N)^d / N) better than O(1/√N)
│  │   ├─ Effective for smooth payoffs
│  │   └─ Degrades for discontinuous payoffs (digitals)
│  └─ Moment Matching:
│      Force sample paths to match theoretical moments
│      Reduces bias from finite samples
├─ Greeks Calculation:
│  ├─ Finite Differences:
│  │   ├─ Delta: (V(S+ε) - V(S-ε)) / (2ε)
│  │   ├─ Gamma: (V(S+ε) - 2V(S) + V(S-ε)) / ε²
│  │   ├─ Requires multiple MC runs → expensive
│  │   ├─ High variance (ratio of noisy estimates)
│  │   └─ Use same random numbers (common random numbers)
│  ├─ Pathwise Method (Infinitesimal Perturbation):
│  │   ├─ Delta: E[∂Payoff/∂S₀] directly
│  │   ├─ Differentiate payoff along each path
│  │   ├─ Single MC run → efficient
│  │   ├─ Requires smooth payoff (fails for digitals)
│  │   └─ Example: Call delta = E[𝟙(S_T > K) × ∂S_T/∂S₀]
│  └─ Likelihood Ratio Method (Score Function):
│      ├─ Delta: E[Payoff × ∂ln(f)/∂S₀]
│      │   where f is path density
│      ├─ Works for discontinuous payoffs
│      ├─ Higher variance than pathwise
│      └─ Useful for digitals, barriers
├─ American Options (Longstaff-Schwartz):
│  ├─ Challenge: Need backward induction in MC
│  ├─ LSM Algorithm:
│  │   ├─ Step 1: Generate all paths forward
│  │   ├─ Step 2: Backward induction at exercise dates
│  │   │   At each date t and path i:
│  │   │   ├─ Intrinsic value: V_intrinsic = Payoff(S_t^i)
│  │   │   ├─ Continuation value: E[V_{t+1} | S_t] via regression
│  │   │   │   Regress future values on basis functions of S_t
│  │   │   │   φ(S) = [1, S, S², ..., polynomials, Laguerre, etc.]
│  │   │   └─ Optimal: Exercise if V_intrinsic > V_continuation
│  │   └─ Step 3: Average optimal exercise values
│  ├─ Regression basis:
│  │   ├─ Polynomials: 1, S, S², S³
│  │   ├─ Laguerre polynomials: Better conditioning
│  │   └─ Need enough terms but avoid overfitting
│  ├─ Only ITM paths:
│  │   Regression only on paths where intrinsic > 0
│  │   (OTM paths never exercise → no information)
│  └─ Convergence: Slower than European
│      Need more paths and time steps
├─ Advanced Topics:
│  ├─ Stochastic Volatility:
│  │   ├─ Heston model: dσ² = κ(θ - σ²)dt + ξσdW
│  │   ├─ Simulate both S and σ jointly
│  │   └─ Captures vol smile/skew
│  ├─ Jump-Diffusion:
│  │   ├─ Merton model: Add Poisson jumps to GBM
│  │   ├─ Simulate: Poisson arrivals + jump sizes
│  │   └─ Captures gap risk
│  ├─ Local Volatility:
│  │   σ(S,t) depends on spot and time
│  │   Calibrate to match implied vol surface
│  ├─ Multifactor Models:
│  │   Interest rate models (HJM, LMM)
│  │   Stochastic dividends, stochastic vol
│  └─ Parallel Computing:
│      ├─ Embarrassingly parallel: Each path independent
│      ├─ GPU acceleration: 100x+ speedup
│      └─ Distributed computing: Split paths across nodes
└─ Practical Considerations:
   ├─ Random Number Generation:
   │   ├─ Quality matters: Use Mersenne Twister or better
   │   ├─ Seed management: Reproducibility vs independence
   │   └─ Avoid periodicity: RNG period >> number of draws
   ├─ Performance Optimization:
   │   ├─ Vectorization: Batch path generation
   │   ├─ Memory: Stream paths (don't store all)
   │   ├─ Early termination: Adaptive # paths based on SE
   │   └─ Precomputation: Cholesky decomposition, constants
   ├─ Numerical Stability:
   │   ├─ Exponentiation: Use log-space for small probabilities
   │   ├─ Overflow: Cap extreme paths (rare in practice)
   │   └─ Underflow: Careful with barrier monitoring
   ├─ Validation:
   │   ├─ Compare to analytical (when available)
   │   ├─ Convergence plots: SE vs √N
   │   ├─ Greeks consistency: Put-call parity, bounds
   │   └─ Sensitivity to time steps, # paths
   └─ Production Implementation:
      ├─ Error handling: Invalid inputs, market data
      ├─ Diagnostics: Return SE, # paths used, convergence flag
      ├─ Caching: Reuse paths for Greeks
      └─ Monitoring: Track computation time, accuracy
```

**Interaction:** Random paths → Payoff calculation → Averaging → Discounting; variance reduction techniques dramatically improve efficiency without changing algorithm structure.

## Challenge Round
1. **Sobol Sequences:** Implement quasi-random Monte Carlo using Sobol sequence. Compare convergence to pseudo-random. When does it outperform?

2. **Longstaff-Schwartz:** Implement American put pricing using LSM regression. Use Laguerre polynomial basis. Compare to binomial tree pricing.

3. **Heston Model:** Add stochastic volatility (Heston dynamics). Price European call and compare to BS. How does vol-of-vol affect price?

4. **Importance Sampling:** Price deep OTM option (K=150, S=100) with/without importance sampling. Shift drift to focus on ITM region. Quantify variance reduction.

5. **Greeks via Pathwise:** Implement pathwise delta calculation. Compare variance to finite difference method. Why does pathwise fail for digital options?

## Key References
- [Glasserman, Monte Carlo Methods in Financial Engineering (Chapters 2-4)](https://www.springer.com/gp/book/9780387004518)
- [Longstaff & Schwartz (2001) - LSM for American Options](https://www.jstor.org/stable/222481)
- [Boyle (1977) - Options: A Monte Carlo Approach](https://www.jstor.org/stable/2978788)
- [Jäckel, Monte Carlo Methods in Finance (Chapters 5-7)](https://www.wiley.com/en-us/Monte+Carlo+Methods+in+Finance-p-9780471497417)

---
**Status:** Essential for exotic/multi-asset derivatives | **Complements:** Binomial Trees, Finite Difference, Risk-Neutral Valuation, Variance Reduction Techniques
