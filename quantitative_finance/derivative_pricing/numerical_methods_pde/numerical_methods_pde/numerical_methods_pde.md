# Numerical Methods for PDEs in Option Pricing

## Concept Skeleton
**Definition:** Finite difference approximations solving partial differential equations (PDEs) for derivative pricing, discretizing Black-Scholes PDE on spatial-temporal grid  
**Purpose:** Price European/American options, path-dependent derivatives when closed-form solutions unavailable, handle early exercise features  
**Prerequisites:** Black-Scholes PDE, Taylor series, numerical stability, boundary conditions

## Comparative Framing
| Method | Explicit Scheme | Implicit Scheme | Crank-Nicolson | ADI Method |
|--------|----------------|-----------------|----------------|------------|
| **Stability** | Conditionally stable (Δt ≤ Δx²/2) | Unconditionally stable | Unconditionally stable | Unconditionally stable |
| **Accuracy** | O(Δt, Δx²) | O(Δt, Δx²) | O(Δt², Δx²) | O(Δt², Δx²) |
| **Computation** | Explicit update (fast) | Matrix inversion (slow) | Tridiagonal solve | Dimensional splitting |
| **American Options** | No | Yes (backward) | Yes (backward) | Yes (backward) |

## Examples + Counterexamples
**Simple Example:**  
European put: Explicit scheme with S∈[0,200], Δx=2, Δt=0.01 → converges to Black-Scholes price within 0.1% (500 time steps)

**Failure Case:**  
Explicit scheme Δt=0.1, Δx=1 → violates stability condition Δt≤Δx²/(2σ²) → oscillations, divergence

**Edge Case:**  
American put near expiry (1 day): Implicit scheme required for early exercise boundary, explicit cannot handle optimal stopping

## Layer Breakdown
```
PDE Numerical Methods:
├─ Black-Scholes PDE (European Options):
│   ├─ PDE: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
│   ├─ Boundary Conditions:
│   │   ├─ V(0,t) = Ke^(-r(T-t)) (put) or 0 (call)
│   │   ├─ V(S→∞,t) = S (call) or 0 (put)
│   │   └─ V(S,T) = max(S-K,0) (call) or max(K-S,0) (put)
│   └─ Backward in Time: Solve from T → 0
├─ Finite Difference Discretization:
│   ├─ Grid Setup:
│   │   ├─ Space: S_min ≤ S ≤ S_max, uniform grid S_i = S_min + i·Δx, i=0,...,M
│   │   ├─ Time: 0 ≤ t ≤ T, uniform grid t_n = n·Δt, n=0,...,N
│   │   └─ Notation: V_i^n ≈ V(S_i, t_n)
│   ├─ Derivative Approximations:
│   │   ├─ ∂V/∂t ≈ (V_i^{n+1} - V_i^n)/Δt (forward difference)
│   │   ├─ ∂V/∂S ≈ (V_{i+1}^n - V_{i-1}^n)/(2Δx) (central difference)
│   │   └─ ∂²V/∂S² ≈ (V_{i+1}^n - 2V_i^n + V_{i-1}^n)/(Δx²)
│   └─ Dimensionless Form: Transform x = ln(S/K), τ = σ²(T-t)/2 for stability
├─ Explicit Finite Difference Method:
│   ├─ Update Formula: V_i^{n+1} = a_i V_{i-1}^n + b_i V_i^n + c_i V_{i+1}^n
│   │   where a_i, b_i, c_i are coefficients from PDE discretization
│   ├─ Stability Condition: Δt ≤ Δx²/(σ²S_max²) (CFL condition)
│   ├─ Advantages: Simple to implement, explicit update per node
│   └─ Disadvantages: Small time steps required, conditionally stable
├─ Implicit Finite Difference Method:
│   ├─ Update Formula: -a_i V_{i-1}^{n+1} + (1+b_i) V_i^{n+1} - c_i V_{i+1}^{n+1} = V_i^n
│   ├─ Matrix Form: A V^{n+1} = V^n (tridiagonal system)
│   ├─ Thomas Algorithm: O(M) solver for tridiagonal matrix
│   ├─ Unconditionally Stable: No restriction on Δt/Δx ratio
│   └─ American Options: Incorporate early exercise constraint max(V, payoff)
├─ Crank-Nicolson Method:
│   ├─ Scheme: θ-method with θ=0.5, average of explicit and implicit
│   ├─ Update: (I - ½ΔtL) V^{n+1} = (I + ½ΔtL) V^n
│   ├─ Accuracy: O(Δt², Δx²) second-order in time
│   ├─ Stability: Unconditionally stable, no spurious oscillations
│   └─ Gold Standard: Best accuracy-stability tradeoff for most problems
├─ American Options (Early Exercise):
│   ├─ Free Boundary Problem: Optimal exercise boundary S*(t)
│   ├─ Constraint: V(S,t) ≥ Payoff(S) (continuation value ≥ immediate exercise)
│   ├─ Projected SOR (Successive Over-Relaxation):
│   │   ├─ Step 1: Solve implicit system as if European
│   │   ├─ Step 2: Enforce V_i^{n+1} = max(V_i^{n+1}, Payoff(S_i))
│   │   └─ Iterate until convergence
│   ├─ Penalty Method: Add penalty term -λ max(Payoff - V, 0)
│   └─ Exercise Boundary: Identify S* where V(S*,t) = Payoff(S*)
└─ Advanced Techniques:
    ├─ Alternating Direction Implicit (ADI): For multi-dimensional PDEs (2+ factors)
    ├─ Non-Uniform Grids: Dense mesh near strike, payoff discontinuity
    ├─ Richardson Extrapolation: Combine solutions at different Δt for higher accuracy
    └─ Adaptive Mesh Refinement: Dynamic grid refinement near early exercise boundary
```

**Interaction:** Discretize PDE → Set boundary conditions → Backward iteration → Tridiagonal solve → Extract option value

## Challenge Round
When do PDE methods outperform Monte Carlo?
- **Low dimensions:** 1-2 factors (single underlying) → PDE O(M²N) efficient, MC needs 10⁶+ paths
- **American options:** Early exercise boundary naturally handled via backward iteration, MC requires regression (Longstaff-Schwartz)
- **Greeks calculation:** Delta/Gamma from finite differences on grid, MC requires multiple simulations or pathwise derivatives
- **Barrier options:** Discrete monitoring approximated well on grid, MC struggles with continuous barriers

When PDE methods struggle: High dimensions (>3 factors) → curse of dimensionality, grid size M^d explodes; path-dependent with complex history (Asian options with daily averaging) → state space too large.

## Key References
- [Wilmott (2006) Paul Wilmott on Quantitative Finance](https://www.wiley.com/en-us/Paul+Wilmott+on+Quantitative+Finance%2C+3+Volume+Set%2C+2nd+Edition-p-9780470018704) - Comprehensive PDE methods
- [Hull (2018) Options, Futures, and Other Derivatives, Ch. 21](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) - Numerical procedures
- [Duffy (2006) Finite Difference Methods in Financial Engineering](https://www.wiley.com/en-us/Finite+Difference+Methods+in+Financial+Engineering%3A+A+Partial+Differential+Equation+Approach-p-9780470858820) - Advanced FD techniques
- [Tavella & Randall (2000) Pricing Financial Instruments: The Finite Difference Method](https://www.wiley.com/en-us/Pricing+Financial+Instruments%3A+The+Finite+Difference+Method-p-9780471197607) - Implementation guide

---
**Status:** Core numerical technique | **Complements:** Black-Scholes Model, Monte Carlo Pricing, American Options
