# Pricing Methodologies Comparison

## Concept Skeleton
**Definition:** Comparative analysis of analytical vs numerical approaches for American option valuation; trade-offs in accuracy, speed, implementation  
**Purpose:** Select appropriate method by problem constraints; understand when closed-form sufficient vs numerical required; benchmark validation  
**Prerequisites:** Black-Scholes pricing, binomial trees, finite differences, Monte Carlo, PDE numerical methods, option pricing fundamentals

## Comparative Framing
| Method | Accuracy | Speed | Complexity | American | Exotic | Implementation |
|--------|----------|-------|-----------|----------|--------|-----------------|
| **Black-Scholes (European)** | High (vanilla) | Instant | Low | No | No | Closed form |
| **Binomial Tree** | Good (convergence) | Fast (1000s steps) | Medium | Yes | Limited | Recursive |
| **Finite Difference** | Excellent (fine grid) | Moderate | Medium-High | Yes | Partial | PDE solver |
| **Monte Carlo** | Good (variance) | Slow (millions paths) | Medium | Approximate | Yes | Path simulation |
| **Trinomial** | Better than binomial | Slower | Medium | Yes | Limited | Recursive |
| **Lattice Methods** | Convergent | Very fast (10-100K) | Medium | Yes | Yes | Optimized tree |
| **Analytical (perpetual)** | Exact | Instant | Low-Medium | Yes (perpetual) | No | Closed form |

## Examples + Counterexamples
**Simple Benchmark:**  
American put S=100, K=100, r=5%, σ=20%, T=1yr. Binomial (100 steps): $10.45. Finite diff: $10.46. Monte Carlo (100K paths): $10.47±0.05. All converge to similar value.

**Excellent Fit - Binomial:**  
Short-dated, standard American option. Binomial efficient, intuitive for traders. Implementation standard.

**Poor Fit - Binomial:**  
Exotic with path dependency (lookback, barrier). Binomial recombining tree loses path history. Requires Monte Carlo.

**Lattice Efficiency:**  
Compensated tree (optimized branching) 1000x faster than naive binomial for high accuracy. Used in production.

**Exotic Options:**  
Basket call (min 3 assets): Monte Carlo most practical. Finite diff in ≥3 dimensions prohibitive (curse of dimensionality).

## Layer Breakdown
```
Pricing Methodologies Comparison:

├─ Black-Scholes Framework:
│  ├─ Applicability:
│  │   European options only
│  │   Non-dividend: Closed form
│  │   With dividends: Modification available
│  ├─ Formula:
│  │   C = S₀ N(d₁) - K e^(-rT) N(d₂)
│  │   Instant evaluation
│  ├─ Strengths:
│  │   Closed-form solution
│  │   Fast computation (microseconds)
│  │   Greeks explicit (derivatives available)
│  │   Benchmark for calibration
│  ├─ Limitations:
│  │   No early exercise (American)
│  │   Assumes constant volatility
│  │   No dividends without modification
│  │   Log-normal assumption (thin tails)
│  └─ Use Case: European options, Greeks computation, IV calibration
├─ Binomial Tree:
│  ├─ Framework:
│  │   Discrete time steps: 0, Δt, 2Δt, ..., T
│  │   Discrete price moves: S × u (up), S × d (down)
│  │   u = e^(σ√Δt), d = 1/u
│  ├─ Algorithm (Backward induction):
│  │   1. Initialize terminal payoffs at T
│  │   2. Work backward: V_i,j = max(Payoff, p×V_{i+1,j+1} + (1-p)×V_{i+1,j})
│  │   3. p = (e^(r×Δt) - d)/(u-d) [risk-neutral probability]
│  ├─ Accuracy:
│  │   Converges to Black-Scholes as steps→∞
│  │   Error ~ O(1/N) with N steps
│  │   Typically 50-100 steps sufficient for 1-2% error
│  ├─ Strengths:
│  │   Natural for American options (max at each node)
│  │   Easy to implement
│  │   Intuitive (visualizable)
│  │   Fast for standard options
│  ├─ Weaknesses:
│  │   Recombining property loses path memory
│  │   Inefficient for very short-dated (high N needed)
│  │   Exotic path-dependent options difficult
│  │   Slow convergence near strikes
│  ├─ Extensions:
│  │   Trinomial (3 outcomes per step): Better stability
│  │   Jump-adapted binomial: For jump-diffusion
│  │   Implied tree: Calibrated to market smiles
│  └─ Use Case: Standard American options, real options
├─ Finite Difference Methods:
│  ├─ Framework:
│  │   Discretize PDE: ∂V/∂t + rS(∂V/∂S) + 0.5σ²S²(∂²V/∂S²) = rV
│  │   Grid: (S_i, t_j) with spacing ΔS, Δt
│  │   Replace derivatives with finite differences
│  ├─ Schemes:
│  │   Explicit: V_{i,j} from V_{i,j+1} (stable if λ=r×Δt/ΔS² ≤ 0.5)
│  │   Implicit: Solve tridiagonal system (unconditionally stable)
│  │   Crank-Nicolson: Average explicit/implicit (2nd order, stable)
│  ├─ American handling:
│  │   At each (S,t): V = max(Intrinsic, [value from PDE])
│  │   Projected SOR or other optimization technique
│  ├─ Accuracy:
│  │   Convergent O(Δt + ΔS²) for explicit
│  │   O(Δt² + ΔS²) for Crank-Nicolson
│  │   Fine grid: High accuracy but slow
│  ├─ Strengths:
│  │   Excellent accuracy (can be very fine grid)
│  │   Handles American directly (constraint)
│  │   Greeks via finite differences at solution
│  │   Proven convergence theory
│  ├─ Weaknesses:
│  │   Requires careful boundary conditions
│  │   Curse of dimensionality (2D+ slow)
│  │   Grid-dependent accuracy (tuning needed)
│  │   Oscillations if parameters poor
│  ├─ Parameters:
│  │   ΔS choice: Small for accuracy, large for speed
│  │   Δt choice: Must satisfy CFL for explicit
│  │   Domain size: High S cutoff (S→∞ approximation)
│  └─ Use Case: High-accuracy American options, Greeks
├─ Monte Carlo Simulation:
│  ├─ Framework:
│  │   Generate M path realizations of S(t)
│  │   dS = μS dt + σS dW (Geometric Brownian Motion)
│  │   Euler discretization: S_{n+1} = S_n × exp((r-0.5σ²)Δt + σ√Δt Z_n)
│  ├─ American handling:
│  │   Standard MC can't directly value American (must know future)
│  │   Least-Squares MC (Longstaff-Schwartz):
│  │   ├─ Backward: At each time step, fit regression
│  │   ├─ Regression target: Continuation value (discounted future)
│  │   ├─ Decision: Compare intrinsic vs fitted continuation
│  │   ├─ Value: Take maximum
│  ├─ Accuracy:
│  │   Error ~ O(1/√M) [Monte Carlo error]
│  │   Biased (LSMC), but converges to true value
│  │   Variance: Antithetic variates, control variates reduce
│  ├─ Strengths:
│  │   Excellent for path-dependent exotics
│  │   Handles multiple assets naturally
│  │   Parallelizable (embarrassingly parallel)
│  │   Flexible: Any payoff structure
│  ├─ Weaknesses:
│  │   Slow: M paths × N steps × M_basis regressions
│  │   Biased for American (low bias, but biased)
│  │   Regression basis choice affects accuracy
│  │   High-dimensional: Slow convergence
│  ├─ Variance Reduction:
│  │   Antithetic: Z and -Z together → half variance
│  │   Control variate: Use European option as control
│  │   Importance sampling: Tilt distribution
│  └─ Use Case: Exotic American options, multi-asset, path-dependent
├─ Trinomial Tree:
│  ├─ Enhancement of binomial:
│  │   Three outcomes per step: u (up), m (middle), d (down)
│  │   u > 1, m = 1, d < 1 (typically)
│  ├─ Benefits:
│  │   Better stability properties
│  │   Faster convergence than binomial
│  │   Fewer steps needed for same accuracy
│  ├─ Drawback:
│  │   More complex implementation
│  │   Slightly slower per step (3 branches vs 2)
│  └─ Use: When binomial oscillation problematic
├─ Advanced Methods:
│  ├─ Implied tree (calibrated to vol smile):
│  │   Match market prices exactly
│  │   Exotic pricing respects smile
│  │   Complexity: Nonlinear optimization
│  ├─ Fourier methods:
│  │   Transform to frequency domain
│  │   Fast for certain payoffs
│  │   Complex implementation
│  ├─ Partial-Difference-Equation finite element:
│  │   FEM vs FD: Better handling of boundaries
│  │   Unstructured mesh
│  └─ Hermite spline interpolation:
│      Higher-order accuracy, smooth Greeks
└─ Selection Criteria:
   ├─ Vanilla American:
   │   Binomial (simple, standard)
   │   Finite Difference (high accuracy)
   │   Quick: 50-100 binomial steps
   ├─ American Exotic (path-dependent):
   │   Least-Squares MC (robust)
   │   Binomial + refinement (if low-dimensional)
   ├─ Greeks sensitivity:
   │   FD best (computed directly)
   │   Binomial: bump-and-recompute
   ├─ Very long dated (perpetual):
   │   Analytical (if available)
   │   Binomial with many steps
   ├─ Speed critical:
   │   Binomial (100 steps, milliseconds)
   │   Analytical if applicable
   └─ Accuracy critical:
       FD or MC with variance reduction
```

**Interaction:** Method choice = trade-off between accuracy, computation time, implementation effort, and problem structure.

## Challenge Round
- How does binomial error scale with N? (O(1/N))
- Why is Least-Squares MC biased for American? (Approximation of continuation value)
- When is trinomial faster than binomial? (Stability → fewer steps)
- Derive CFL condition for explicit FD (hint: stability requires λ ≤ 0.5)

## Key References
- [Cox, Ross, Rubinstein "Option Pricing: Simplified Approach" (1979)](https://www.jstor.org/stable/2352191) — Binomial foundation
- [Longstaff & Schwartz "LSMC" (2001)](https://www.jstor.org/stable/2680920) — Least-Squares Monte Carlo
- [Forsyth & Vetzal "Quadratic Convergence" (2002)](https://doi.org/10.1137/S1064827502410651) — FD convergence

---
**Status:** Practical comparison for implementation | **Complements:** Binomial Trees, Finite Difference Methods, Monte Carlo Simulation
