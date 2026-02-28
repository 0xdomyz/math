# Finite Difference Methods for Option Pricing

## Concept Skeleton
**Definition:** Numerical PDE solvers discretizing space-time grid to approximate derivative prices governed by parabolic equations  
**Purpose:** Price options without closed-forms; handle early exercise, barriers, path-dependence; compute Greeks efficiently via grid  
**Prerequisites:** Black-Scholes PDE, numerical analysis, stability/convergence theory, linear algebra, boundary conditions

## Comparative Framing
| Method | Explicit | Implicit | Crank-Nicolson | Alternating Direction |
|--------|----------|----------|----------------|----------------------|
| **Stability** | Conditional (Δt ≤ Δx²/σ²) | Unconditional | Unconditional | Unconditional |
| **Accuracy** | O(Δt, Δx²) | O(Δt, Δx²) | O(Δt², Δx²) | O(Δt², Δx²) |
| **Complexity** | Trivial (forward) | Solve linear system | Solve linear system | Operator splitting |
| **Speed** | Fast per step | Moderate | Moderate | Fast (multi-dim) |
| **Suitability** | Small problems | General use | High accuracy | Multi-asset |

| Approach | Finite Difference | Monte Carlo | Binomial Tree | Closed-Form |
|----------|-------------------|-------------|---------------|-------------|
| **Dimensionality** | Low (1-3 assets) | High (any) | Low (1-2) | N/A |
| **Early Exercise** | Natural (grid) | LSM complex | Natural | N/A |
| **Greeks** | Excellent (grid) | Poor (noise) | Moderate | Analytical |
| **Speed** | Moderate | Slow (convergence) | Fast (low accuracy) | Instant |
| **Barriers** | Natural boundaries | Monitoring issues | Non-recombining | Limited cases |

## Examples + Counterexamples
**Simple Example:**  
European put on grid: Solve backward from terminal payoff max(K-S, 0). Explicit scheme updates each node from three neighbors at next time step.

**Perfect Fit:**  
American put on 1D grid: Check early exercise at each node, take max(intrinsic, continuation). PDE naturally incorporates optimal stopping.

**Implicit Euler:**  
Transform Black-Scholes PDE into heat equation via log transformation. Implicit scheme: θV_{i,j} + (1-θ)V_{i,j+1} = combination of neighbors. θ=1: fully implicit, unconditionally stable.

**Crank-Nicolson:**  
Average explicit and implicit (θ=0.5): Second-order in time. Oscillations possible for non-smooth payoffs (digitals). Rannacher smoothing (4 implicit steps) fixes.

**Barrier Option:**  
Down-and-out put: Set V(S_barrier, t) = 0 for all t. Grid naturally enforces boundary condition. No Brownian bridge needed like Monte Carlo.

**Poor Fit:**  
Multi-asset basket (5+ assets): Grid size explodes (curse of dimensionality). 100 nodes per dimension → 100^5 = 10 billion points. Monte Carlo better.

## Layer Breakdown
```
Finite Difference Framework:

├─ PDE Formulation:
│  ├─ Black-Scholes PDE:
│  │   ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
│  │   Parabolic PDE (backward in time)
│  ├─ Transformation to Heat Equation:
│  │   x = ln(S/K), τ = σ²(T-t)/2
│  │   u(x,τ) = V(S,t) / K
│  │   Yields: ∂u/∂τ = ∂²u/∂x² + (k-1)∂u/∂x - ku
│  │   where k = 2r/σ²
│  │   Removes variable coefficients
│  ├─ Terminal Condition:
│  │   V(S, T) = Payoff(S)
│  │   Call: max(S-K, 0), Put: max(K-S, 0)
│  │   American: also check intrinsic at each step
│  └─ Boundary Conditions:
│      ├─ S → 0: Call→0, Put→K×e^(-r(T-t))
│      ├─ S → ∞: Call→S-K×e^(-r(T-t)), Put→0
│      └─ Barriers: V(S_barrier, t) = rebate or 0
├─ Grid Construction:
│  ├─ Space Discretization:
│  │   ├─ Domain: [S_min, S_max] or [x_min, x_max]
│  │   ├─ Grid spacing: ΔS = (S_max - S_min) / M
│  │   │   Or uniform in log-space: Δx constant
│  │   ├─ Number of nodes: M+1 spatial points
│  │   └─ Considerations:
│  │       ├─ Finer grid near strike (payoff kink)
│  │       ├─ Wide enough to avoid boundary effects
│  │       └─ Typical: S_min = 0 or 0.1×K, S_max = 3×K
│  ├─ Time Discretization:
│  │   ├─ Domain: [0, T] (backward: T → 0)
│  │   ├─ Time step: Δt = T / N
│  │   ├─ Number of steps: N+1 time points
│  │   └─ Stability constraint (explicit):
│  │       Δt ≤ Δx² / σ² (or CFL condition)
│  ├─ Grid Indexing:
│  │   V[i,j] = Value at S_i, t_j
│  │   i = 0, 1, ..., M (space)
│  │   j = 0, 1, ..., N (time, backward)
│  └─ Non-Uniform Grids:
│      ├─ Clustered near strike: sinh transform
│      ├─ Adaptive mesh refinement (AMR)
│      └─ Better accuracy for non-smooth payoffs
├─ Explicit Finite Difference:
│  ├─ Discretization (Forward Time, Centered Space):
│  │   ∂V/∂t ≈ (V[i,j+1] - V[i,j]) / Δt
│  │   ∂V/∂S ≈ (V[i+1,j] - V[i-1,j]) / (2ΔS)
│  │   ∂²V/∂S² ≈ (V[i+1,j] - 2V[i,j] + V[i-1,j]) / ΔS²
│  ├─ Update Formula:
│  │   V[i,j] = α×V[i-1,j+1] + β×V[i,j+1] + γ×V[i+1,j+1]
│  │   α, β, γ: Coefficients from PDE discretization
│  ├─ Advantages:
│  │   ├─ Simple implementation (no linear solve)
│  │   ├─ Fast per time step
│  │   └─ Easy to parallelize
│  ├─ Disadvantages:
│  │   ├─ Stability constraint: Δt ≤ Δx²/σ²
│  │   │   Requires small time steps → slow convergence
│  │   ├─ Accuracy: O(Δt, Δx²)
│  │   └─ Can blow up if stability violated
│  └─ Use Cases:
│      European options, quick prototyping, benchmarking
├─ Implicit Finite Difference:
│  ├─ Discretization (Backward Time):
│  │   Use V[i±1,j] instead of V[i±1,j+1] for spatial terms
│  │   Implicit dependence on unknown values at time j
│  ├─ Linear System:
│  │   A × V_j = b (tridiagonal matrix)
│  │   A: M×M matrix from stencil coefficients
│  │   V_j: Vector of option values at time j
│  │   b: Right-hand side from V_{j+1} and boundaries
│  ├─ Matrix Structure (Tridiagonal):
│  │   ⎡ b₁  c₁   0   ...  ⎤
│  │   ⎢ a₂  b₂  c₂   0   ⎥
│  │   ⎢  0  a₃  b₃  c₃   ⎥
│  │   ⎢ ...           ...⎥
│  │   ⎣  0  ... aₘ  bₘ  ⎦
│  │   a, b, c: From discretized PDE coefficients
│  ├─ Solution Method:
│  │   ├─ Thomas algorithm (tridiagonal LU): O(M)
│  │   ├─ Gaussian elimination specialized
│  │   └─ Very efficient (linear complexity)
│  ├─ Advantages:
│  │   ├─ Unconditionally stable (any Δt)
│  │   ├─ Larger time steps → fewer steps
│  │   └─ Robust for stiff problems
│  ├─ Disadvantages:
│  │   ├─ Accuracy: O(Δt, Δx²) (first-order in time)
│  │   ├─ Linear solve each step (moderate overhead)
│  │   └─ Slightly more complex than explicit
│  └─ Use Cases:
│      General purpose, when stability critical
├─ Crank-Nicolson Method:
│  ├─ Theta Method (θ = 0.5):
│  │   Average of explicit (θ=0) and implicit (θ=1)
│  │   ∂V/∂t = θ×[spatial_terms at j] + (1-θ)×[spatial_terms at j+1]
│  ├─ Discretization:
│  │   (1 + ½A)V_j = (1 - ½A)V_{j+1}
│  │   A: Spatial difference operator
│  ├─ Accuracy:
│  │   O(Δt², Δx²): Second-order in time AND space
│  │   Best accuracy for smooth solutions
│  ├─ Stability:
│  │   Unconditionally stable (θ ≥ 0.5)
│  │   θ = 0.5: Optimal accuracy
│  ├─ Oscillations:
│  │   ├─ Non-smooth payoffs (digitals, barriers)
│  │   │   Can produce spurious oscillations near kinks
│  │   ├─ Rannacher Smoothing:
│  │   │   Use fully implicit (θ=1) for first 4 steps
│  │   │   Then switch to Crank-Nicolson
│  │   │   Eliminates oscillations
│  │   └─ Averaging: Local smoothing at kinks
│  ├─ Implementation:
│  │   Solve tridiagonal system at each step
│  │   Matrix slightly different from pure implicit
│  └─ Use Cases:
│      High accuracy requirements, smooth payoffs
│      Default choice for production pricing
├─ American Option Handling:
│  ├─ Optimal Stopping Condition:
│  │   V(S,t) = max(Intrinsic, Continuation)
│  │   Intrinsic: Immediate exercise payoff
│  │   Continuation: Hold value from PDE
│  ├─ Implementation:
│  │   ├─ After each PDE step (implicit/CN):
│  │   │   V[i,j] = max(V[i,j], Intrinsic[i])
│  │   │   Project onto constraint
│  │   ├─ Early Exercise Boundary:
│  │   │   S* where V = Intrinsic (optimal)
│  │   │   Moves through time (free boundary)
│  │   └─ Convergence:
│  │       PDE naturally finds optimal boundary
│  ├─ Linear Complementarity Problem (LCP):
│  │   min(∂V/∂t + LV - rV, V - g(S)) = 0
│  │   g(S): Intrinsic value function
│  │   Formal optimal stopping condition
│  ├─ Advantages over Trees:
│  │   ├─ No interpolation errors
│  │   ├─ Smooth convergence
│  │   └─ Easy to implement
│  └─ Penalty Methods:
│      Add large penalty for violating constraint
│      ρ × max(Intrinsic - V, 0)
│      Equivalent to LCP as ρ → ∞
├─ Multi-Dimensional Extensions:
│  ├─ 2D Problem (Two Assets):
│  │   ∂V/∂t + ½σ₁²S₁²∂²V/∂S₁² + ½σ₂²S₂²∂²V/∂S₂²
│  │        + ρσ₁σ₂S₁S₂∂²V/∂S₁∂S₂ + r(S₁∂V/∂S₁ + S₂∂V/∂S₂) - rV = 0
│  │   Cross-derivative term for correlation
│  ├─ 2D Grid:
│  │   V[i,j,k] = Value at (S₁ᵢ, S₂ⱼ, tₖ)
│  │   M×N spatial points → O(MN) per time step
│  ├─ Direct Approach (2D):
│  │   ├─ Vectorize 2D grid into 1D: V[i,j] → V[i×N+j]
│  │   ├─ Matrix: (M×N) × (M×N) sparse matrix
│  │   │   No longer tridiagonal (pentadiagonal pattern)
│  │   ├─ Solve: Ax = b via iterative methods
│  │   │   Conjugate gradient, BiCGStab, GMRES
│  │   └─ Cost: O((MN)²) or O(MN) with iterative
│  ├─ Alternating Direction Implicit (ADI):
│  │   ├─ Douglas-Rachford Splitting:
│  │   │   Split 2D problem into two 1D problems
│  │   │   Half-step in S₁, then half-step in S₂
│  │   ├─ Algorithm:
│  │   │   V* = (I + ½Δt×L₁)^(-1) × V_{j+1}
│  │   │   V_j = (I + ½Δt×L₂)^(-1) × V*
│  │   │   L₁, L₂: Operators for S₁, S₂ directions
│  │   ├─ Advantages:
│  │   │   ├─ Tridiagonal solves only (efficient)
│  │   │   ├─ Unconditionally stable
│  │   │   ├─ O(Δt², Δx²) accuracy
│  │   │   └─ Scales to 3D (three sweeps)
│  │   ├─ Cost: O(MN) per time step (linear!)
│  │   └─ Implementation:
│  │       Cycle through dimensions, solve 1D problems
│  ├─ Curse of Dimensionality:
│  │   ├─ Grid points: M^d (exponential in dimension d)
│  │   ├─ Practical limit: d ≤ 3 (maybe 4 with care)
│  │   └─ Monte Carlo better for d ≥ 4
│  └─ Use Cases:
│      Spread options, basket (2-3 assets), FX triangles
├─ Greeks Computation:
│  ├─ Delta:
│  │   ∂V/∂S ≈ (V[i+1,j] - V[i-1,j]) / (2ΔS)
│  │   Centered difference on grid
│  │   Direct from grid values (no re-pricing)
│  ├─ Gamma:
│  │   ∂²V/∂S² ≈ (V[i+1,j] - 2V[i,j] + V[i-1,j]) / ΔS²
│  │   Second derivative from stencil
│  ├─ Theta:
│  │   ∂V/∂t ≈ (V[i,j+1] - V[i,j]) / Δt
│  │   Time evolution on grid
│  │   Or extract from PDE residual
│  ├─ Vega:
│  │   Finite difference: Re-price with σ ± δσ
│  │   Or use adjoint/sensitivity equations
│  ├─ Rho:
│  │   Similar to Vega: Bump r and re-price
│  ├─ Advantages:
│  │   ├─ All Greeks from single grid solve
│  │   ├─ Smooth (no Monte Carlo noise)
│  │   ├─ Cross-Greeks (Vanna, Volga) easy
│  │   └─ Accurate near strike (fine grid)
│  └─ Higher-Order Greeks:
│      Speed, Color: Third/fourth derivatives
│      Directly from grid stencils
├─ Stability and Convergence:
│  ├─ Von Neumann Stability Analysis:
│  │   ├─ Fourier mode: V[i,j] = ξ^j e^(ikxᵢ)
│  │   ├─ Amplification factor: ξ(k)
│  │   ├─ Stability: |ξ| ≤ 1 for all k
│  │   └─ Explicit: Requires Δt ≤ Δx²/(σ²)
│  ├─ CFL Condition:
│  │   Courant-Friedrichs-Lewy: Domain of dependence
│  │   Numerical domain ⊇ PDE domain
│  │   Δt ≤ C × Δx² (explicit)
│  ├─ Consistency:
│  │   Truncation error → 0 as Δt, Δx → 0
│  │   Taylor expansion confirms order
│  ├─ Convergence:
│  │   Lax Equivalence Theorem:
│  │   Consistency + Stability ⇒ Convergence
│  │   Numerical solution → True solution
│  └─ Practical Checks:
│      ├─ Richardson extrapolation: Estimate error
│      ├─ Grid refinement: Compare multiple grids
│      └─ Known solutions: Benchmark accuracy
├─ Boundary Conditions:
│  ├─ Dirichlet:
│  │   Fix V(S_boundary, t) = known value
│  │   Used for barriers, extreme S limits
│  ├─ Neumann:
│  │   Fix ∂V/∂S at boundary
│  │   Far from strike: Linear extrapolation
│  ├─ Implementation:
│  │   ├─ Absorbing: V = 0 (knock-out barrier)
│  │   ├─ Reflecting: ∂V/∂S = 0 (symmetry)
│  │   └─ Far-field: Asymptotic expansions
│  ├─ Accuracy Impact:
│  │   Poor boundaries → global error propagation
│  │   Wide domain minimizes boundary effects
│  └─ Artificial Boundaries:
│      Transparent/radiation conditions
│      Minimize reflection artifacts
├─ Advanced Techniques:
│  ├─ Adaptive Mesh Refinement (AMR):
│  │   ├─ Concentrate nodes where needed
│  │   ├─ Detect error indicators (gradient, curvature)
│  │   ├─ Refine regions dynamically
│  │   └─ Efficient for localized features
│  ├─ High-Order Methods:
│  │   ├─ Fourth-order compact schemes
│  │   ├─ Spectral methods (global basis)
│  │   └─ Better accuracy, tighter stencils
│  ├─ Splitting Methods:
│  │   ├─ Strang splitting: Decompose operator
│  │   ├─ Solve drift and diffusion separately
│  │   └─ Efficient for complex operators
│  ├─ Exponential Time Differencing:
│  │   Exact integration of linear part
│  │   Handles stiffness better
│  ├─ Sparse Grid Methods:
│  │   ├─ Smolyak construction
│  │   ├─ Reduce curse of dimensionality
│  │   └─ d=4-6 feasible with sparsity
│  └─ Finite Element Methods (FEM):
│      Variational formulation, unstructured grids
│      Better for irregular domains
└─ Practical Implementation:
   ├─ Workflow:
   │   1. Transform PDE (log-space often better)
   │   2. Choose grid spacing (balance accuracy/cost)
   │   3. Select method (Crank-Nicolson default)
   │   4. Apply boundary conditions carefully
   │   5. Solve backward in time from payoff
   │   6. Extract price at S0, t=0
   │   7. Compute Greeks from grid
   ├─ Performance Optimization:
   │   ├─ Vectorization: NumPy arrays, avoid loops
   │   ├─ Sparse matrices: scipy.sparse for large grids
   │   ├─ Parallel: Multi-threading for scenarios
   │   └─ GPU: CUDA for massive grids (research)
   ├─ Error Control:
   │   ├─ Monitor convergence with refinement
   │   ├─ Compare to closed-forms when available
   │   ├─ Check put-call parity, arbitrage bounds
   │   └─ Validate Greeks vs finite difference bumps
   ├─ Common Pitfalls:
   │   ├─ Boundaries too close: Artificial effects
   │   ├─ Coarse grid: Missing sharp features
   │   ├─ Oscillations: Use Rannacher smoothing
   │   └─ Stability violation (explicit): Check CFL
   └─ Comparison to Alternatives:
      ├─ vs Monte Carlo: Better for 1-2D, Greeks, barriers
      ├─ vs Trees: More accurate, smooth convergence
      ├─ vs Closed-Form: Handles complexity, American
      └─ Hybrid: FD for Greeks, MC for high-dim scenarios
```

**Interaction:** Define PDE + boundaries → Discretize grid → Choose scheme (stability vs accuracy) → Solve backward → Extract Greeks from grid.

## Challenge Round
1. **Non-Uniform Grid:** Implement sinh-transform grid concentrating nodes near strike. Compare convergence to uniform grid for digital option.

2. **2D ADI for Spread:** Price spread option on two correlated assets using Alternating Direction Implicit. Handle cross-derivative term. Benchmark vs Monte Carlo.

3. **High-Order Compact:** Implement fourth-order compact scheme (nine-point stencil). Compare accuracy vs standard second-order for same computational cost.

4. **Adaptive Mesh Refinement:** Build AMR for American put. Detect exercise boundary, refine locally. Show speedup vs uniform fine grid.

5. **Jump-Diffusion PDE:** Extend to Merton jump-diffusion (PIDE with integral term). Use implicit-explicit (IMEX) splitting for jump component.

## Key References
- [Wilmott, Derivatives: The Theory and Practice of Financial Engineering (Chapter 31)](https://www.wiley.com/en-us/Derivatives%3A+The+Theory+and+Practice+of+Financial+Engineering-p-9780470013205) - comprehensive PDE methods
- [Duffy, Finite Difference Methods in Financial Engineering](https://www.wiley.com/en-us/Finite+Difference+Methods+in+Financial+Engineering%3A+A+Partial+Differential+Equation+Approach-p-9780470858820) - implementation details, stability analysis
- [Tavella & Randall, Pricing Financial Instruments: The Finite Difference Method](https://www.wiley.com/en-us/Pricing+Financial+Instruments%3A+The+Finite+Difference+Method-p-9780471197607) - ADI, multi-dimensional techniques

---
**Status:** Numerical PDE solvers | **Complements:** Black-Scholes, American Options, Barriers, Greeks, Monte Carlo Comparison
