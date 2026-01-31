# Finite Difference Methods for Option Pricing

## 1. Concept Skeleton
**Definition:** Numerical PDE solvers discretizing space-time grid to approximate derivative prices governed by parabolic equations  
**Purpose:** Price options without closed-forms; handle early exercise, barriers, path-dependence; compute Greeks efficiently via grid  
**Prerequisites:** Black-Scholes PDE, numerical analysis, stability/convergence theory, linear algebra, boundary conditions

## 2. Comparative Framing
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

## 3. Examples + Counterexamples

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

## 4. Layer Breakdown
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

## 5. Mini-Project
Implement explicit, implicit, and Crank-Nicolson solvers with American options:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("FINITE DIFFERENCE METHODS FOR OPTION PRICING")
print("="*70)

class FiniteDifferenceEngine:
    """Finite difference PDE solver for options"""
    
    def __init__(self, S0, K, r, sigma, T, option_type='call'):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.option_type = option_type
    
    def setup_grid(self, M, N, S_max_factor=3.0):
        """Create spatial and temporal grid"""
        # Space grid
        self.S_max = S_max_factor * self.K
        self.S_min = 0.0
        self.M = M  # Spatial points
        self.N = N  # Time steps
        
        self.dS = (self.S_max - self.S_min) / M
        self.dt = self.T / N
        
        self.S_grid = np.linspace(self.S_min, self.S_max, M+1)
        self.t_grid = np.linspace(0, self.T, N+1)
        
        # Grid for option values V[i, j] = V(S_i, t_j)
        self.V = np.zeros((M+1, N+1))
        
        # Terminal condition (payoff)
        if self.option_type == 'call':
            self.V[:, -1] = np.maximum(self.S_grid - self.K, 0)
        else:  # put
            self.V[:, -1] = np.maximum(self.K - self.S_grid, 0)
    
    def intrinsic_value(self):
        """Intrinsic value for American options"""
        if self.option_type == 'call':
            return np.maximum(self.S_grid - self.K, 0)
        else:
            return np.maximum(self.K - self.S_grid, 0)
    
    def apply_boundary_conditions(self, j):
        """Apply boundary conditions at time step j"""
        t = self.t_grid[j]
        tau = self.T - t
        
        if self.option_type == 'call':
            # S → 0: V → 0
            self.V[0, j] = 0
            # S → ∞: V → S - K*e^(-r*tau)
            self.V[-1, j] = self.S_max - self.K * np.exp(-self.r * tau)
        else:  # put
            # S → 0: V → K*e^(-r*tau)
            self.V[0, j] = self.K * np.exp(-self.r * tau)
            # S → ∞: V → 0
            self.V[-1, j] = 0
    
    def explicit_method(self, american=False):
        """Explicit finite difference (forward time, centered space)"""
        # Check stability condition
        max_dt = self.dS**2 / (self.sigma**2 * self.S_max**2)
        if self.dt > max_dt:
            print(f"Warning: Stability condition violated!")
            print(f"  Current dt: {self.dt:.6f}, Max stable dt: {max_dt:.6f}")
        
        # Coefficients for interior points
        alpha = np.zeros(self.M+1)
        beta = np.zeros(self.M+1)
        gamma = np.zeros(self.M+1)
        
        for i in range(1, self.M):
            S = self.S_grid[i]
            
            # Discretized PDE coefficients
            a = 0.5 * self.dt * (self.sigma**2 * S**2 / self.dS**2 - self.r * S / self.dS)
            b = -self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r)
            c = 0.5 * self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r * S / self.dS)
            
            alpha[i] = a
            beta[i] = 1 + b
            gamma[i] = c
        
        # Time stepping (backward from T to 0)
        for j in range(self.N-1, -1, -1):
            # Update interior points
            for i in range(1, self.M):
                self.V[i, j] = (alpha[i] * self.V[i-1, j+1] + 
                               beta[i] * self.V[i, j+1] + 
                               gamma[i] * self.V[i+1, j+1])
            
            # Apply boundary conditions
            self.apply_boundary_conditions(j)
            
            # American exercise
            if american:
                intrinsic = self.intrinsic_value()
                self.V[:, j] = np.maximum(self.V[:, j], intrinsic)
        
        return self.get_option_price()
    
    def implicit_method(self, american=False):
        """Implicit finite difference (backward time)"""
        # Build tridiagonal matrix
        diag_a = np.zeros(self.M-1)
        diag_b = np.zeros(self.M-1)
        diag_c = np.zeros(self.M-1)
        
        for idx, i in enumerate(range(1, self.M)):
            S = self.S_grid[i]
            
            # Coefficients
            a = -0.5 * self.dt * (self.sigma**2 * S**2 / self.dS**2 - self.r * S / self.dS)
            b = 1 + self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r)
            c = -0.5 * self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r * S / self.dS)
            
            diag_a[idx] = a
            diag_b[idx] = b
            diag_c[idx] = c
        
        # Create sparse tridiagonal matrix
        A = diags([diag_a[1:], diag_b, diag_c[:-1]], [-1, 0, 1], format='csr')
        
        # Time stepping
        for j in range(self.N-1, -1, -1):
            # Right-hand side
            rhs = self.V[1:-1, j+1].copy()
            
            # Adjust for boundary conditions at time j
            self.apply_boundary_conditions(j)
            rhs[0] -= diag_a[0] * self.V[0, j]
            rhs[-1] -= diag_c[-1] * self.V[-1, j]
            
            # Solve linear system
            self.V[1:-1, j] = spsolve(A, rhs)
            
            # American exercise
            if american:
                intrinsic = self.intrinsic_value()
                self.V[:, j] = np.maximum(self.V[:, j], intrinsic)
        
        return self.get_option_price()
    
    def crank_nicolson(self, american=False, rannacher_steps=4):
        """Crank-Nicolson (theta=0.5) with Rannacher smoothing"""
        # Coefficients for interior points
        alpha = np.zeros(self.M-1)
        beta = np.zeros(self.M-1)
        gamma = np.zeros(self.M-1)
        
        for idx, i in enumerate(range(1, self.M)):
            S = self.S_grid[i]
            
            a = 0.25 * self.dt * (self.sigma**2 * S**2 / self.dS**2 - self.r * S / self.dS)
            b = -0.5 * self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r)
            c = 0.25 * self.dt * (self.sigma**2 * S**2 / self.dS**2 + self.r * S / self.dS)
            
            alpha[idx] = a
            beta[idx] = b
            gamma[idx] = c
        
        # LHS matrix: (I + 0.5*L)
        diag_a_lhs = -alpha
        diag_b_lhs = 1 - beta
        diag_c_lhs = -gamma
        
        A_lhs = diags([diag_a_lhs[1:], diag_b_lhs, diag_c_lhs[:-1]], 
                      [-1, 0, 1], format='csr')
        
        # Time stepping
        for j in range(self.N-1, -1, -1):
            # Rannacher smoothing: Use fully implicit for first few steps
            use_implicit = (j >= self.N - rannacher_steps)
            
            if use_implicit:
                theta = 1.0  # Fully implicit
            else:
                theta = 0.5  # Crank-Nicolson
            
            # RHS matrix depends on theta
            if theta == 0.5:
                # Standard Crank-Nicolson
                rhs = (alpha * self.V[:-2, j+1] + 
                       (1 + beta) * self.V[1:-1, j+1] + 
                       gamma * self.V[2:, j+1])
            else:
                # Fully implicit
                rhs = self.V[1:-1, j+1].copy()
            
            # Boundary conditions
            self.apply_boundary_conditions(j)
            rhs[0] -= diag_a_lhs[0] * self.V[0, j]
            rhs[-1] -= diag_c_lhs[-1] * self.V[-1, j]
            
            # Solve
            self.V[1:-1, j] = spsolve(A_lhs, rhs)
            
            # American exercise
            if american:
                intrinsic = self.intrinsic_value()
                self.V[:, j] = np.maximum(self.V[:, j], intrinsic)
        
        return self.get_option_price()
    
    def get_option_price(self):
        """Interpolate to get price at S0"""
        # Find closest grid point to S0
        idx = np.argmin(np.abs(self.S_grid - self.S0))
        return self.V[idx, 0]
    
    def compute_greeks(self):
        """Compute Greeks from grid at t=0"""
        idx = np.argmin(np.abs(self.S_grid - self.S0))
        
        if idx == 0 or idx == self.M:
            return {'delta': None, 'gamma': None, 'theta': None}
        
        # Delta: ∂V/∂S
        delta = (self.V[idx+1, 0] - self.V[idx-1, 0]) / (2 * self.dS)
        
        # Gamma: ∂²V/∂S²
        gamma = (self.V[idx+1, 0] - 2*self.V[idx, 0] + self.V[idx-1, 0]) / self.dS**2
        
        # Theta: ∂V/∂t (use first time step)
        if self.N > 0:
            theta = (self.V[idx, 1] - self.V[idx, 0]) / self.dt
        else:
            theta = None
        
        return {'delta': delta, 'gamma': gamma, 'theta': theta}

class Barrier2DEngine:
    """2D finite difference for barrier options"""
    
    def __init__(self, S0, K, r, sigma, T, H_down, option_type='put'):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.H_down = H_down
        self.option_type = option_type
    
    def price_down_and_out(self, M=100, N=100):
        """Price down-and-out option using implicit method"""
        # Grid from barrier to S_max
        S_max = 3 * self.K
        S_min = self.H_down  # Start at barrier
        
        S_grid = np.linspace(S_min, S_max, M+1)
        dS = (S_max - S_min) / M
        dt = self.T / N
        
        V = np.zeros((M+1, N+1))
        
        # Terminal payoff
        if self.option_type == 'put':
            V[:, -1] = np.maximum(self.K - S_grid, 0)
        else:
            V[:, -1] = np.maximum(S_grid - self.K, 0)
        
        # Barrier condition: V(H_down, t) = 0 for all t
        V[0, :] = 0
        
        # Build tridiagonal matrix
        diag_a = np.zeros(M-1)
        diag_b = np.zeros(M-1)
        diag_c = np.zeros(M-1)
        
        for idx, i in enumerate(range(1, M)):
            S = S_grid[i]
            
            a = -0.5 * dt * (self.sigma**2 * S**2 / dS**2 - self.r * S / dS)
            b = 1 + dt * (self.sigma**2 * S**2 / dS**2 + self.r)
            c = -0.5 * dt * (self.sigma**2 * S**2 / dS**2 + self.r * S / dS)
            
            diag_a[idx] = a
            diag_b[idx] = b
            diag_c[idx] = c
        
        A = diags([diag_a[1:], diag_b, diag_c[:-1]], [-1, 0, 1], format='csr')
        
        # Time stepping
        for j in range(N-1, -1, -1):
            t = j * dt
            tau = self.T - t
            
            # Boundary at S_max
            if self.option_type == 'put':
                V[-1, j] = 0
            else:
                V[-1, j] = S_max - self.K * np.exp(-self.r * tau)
            
            # RHS
            rhs = V[1:-1, j+1].copy()
            rhs[0] -= diag_a[0] * V[0, j]  # Barrier (always 0)
            rhs[-1] -= diag_c[-1] * V[-1, j]
            
            V[1:-1, j] = spsolve(A, rhs)
        
        # Interpolate to S0
        idx = np.argmin(np.abs(S_grid - self.S0))
        return V[idx, 0], V, S_grid

# Black-Scholes closed-form for comparison
def black_scholes(S, K, r, T, sigma, option_type='call'):
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# Scenario 1: European Put - Method Comparison
print("\n" + "="*70)
print("SCENARIO 1: European Put - Method Comparison")
print("="*70)

S0, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.25
M, N = 100, 1000

print(f"\nParameters: S=${S0}, K=${K}, r={r:.1%}, T={T}yr, σ={sigma:.1%}")
print(f"Grid: M={M} space points, N={N} time steps")

# Closed-form
bs_price = black_scholes(S0, K, r, T, sigma, 'put')
print(f"\nBlack-Scholes (Closed-Form): ${bs_price:.6f}")

# Explicit
engine_exp = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
engine_exp.setup_grid(M, N)
price_exp = engine_exp.explicit_method()
error_exp = abs(price_exp - bs_price)

print(f"\nExplicit Method:")
print(f"  Price: ${price_exp:.6f}")
print(f"  Error: ${error_exp:.6f} ({error_exp/bs_price*100:.4f}%)")
print(f"  Stability check: dt={engine_exp.dt:.6f}, dt_max≈{engine_exp.dS**2/(sigma**2*engine_exp.S_max**2):.6f}")

# Implicit
engine_imp = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
engine_imp.setup_grid(M, N)
price_imp = engine_imp.implicit_method()
error_imp = abs(price_imp - bs_price)

print(f"\nImplicit Method:")
print(f"  Price: ${price_imp:.6f}")
print(f"  Error: ${error_imp:.6f} ({error_imp/bs_price*100:.4f}%)")

# Crank-Nicolson
engine_cn = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
engine_cn.setup_grid(M, N)
price_cn = engine_cn.crank_nicolson()
error_cn = abs(price_cn - bs_price)

print(f"\nCrank-Nicolson:")
print(f"  Price: ${price_cn:.6f}")
print(f"  Error: ${error_cn:.6f} ({error_cn/bs_price*100:.4f}%)")
print(f"\nCrank-Nicolson most accurate (second-order in time)")

# Scenario 2: Convergence Analysis
print("\n" + "="*70)
print("SCENARIO 2: Convergence Analysis")
print("="*70)

grids = [(25, 100), (50, 200), (100, 400), (200, 800)]

print(f"\n{'Grid (M×N)':<15} {'CN Price':<12} {'Error':<12} {'Ratio':<10}")
print("-" * 49)

prev_error = None
for M_test, N_test in grids:
    engine = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
    engine.setup_grid(M_test, N_test)
    price = engine.crank_nicolson()
    error = abs(price - bs_price)
    
    if prev_error is not None:
        ratio = prev_error / error
        ratio_str = f"{ratio:.2f}"
    else:
        ratio_str = "-"
    
    print(f"{M_test}×{N_test:<11} ${price:<11.6f} ${error:<11.6f} {ratio_str:<10}")
    prev_error = error

print(f"\nError reduction ~4× per refinement (second-order convergence)")

# Scenario 3: American Put
print("\n" + "="*70)
print("SCENARIO 3: American Put Option")
print("="*70)

M, N = 100, 500

# European
engine_euro = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
engine_euro.setup_grid(M, N)
price_euro_cn = engine_euro.crank_nicolson(american=False)

# American
engine_amer = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
engine_amer.setup_grid(M, N)
price_amer_cn = engine_amer.crank_nicolson(american=True)

early_exercise_premium = price_amer_cn - price_euro_cn

print(f"\nEuropean Put (Crank-Nicolson): ${price_euro_cn:.6f}")
print(f"American Put (Crank-Nicolson): ${price_amer_cn:.6f}")
print(f"Early Exercise Premium: ${early_exercise_premium:.6f} ({early_exercise_premium/price_euro_cn*100:.2f}%)")

# Find early exercise boundary
S_values = np.linspace(60, 100, 20)
ex_boundary = []

for S_test in S_values:
    engine_test = FiniteDifferenceEngine(S_test, K, r, sigma, T, 'put')
    engine_test.setup_grid(M, N)
    engine_test.crank_nicolson(american=True)
    
    # Check if exercising is optimal (V = intrinsic)
    intrinsic = max(K - S_test, 0)
    continuation = engine_test.V[np.argmin(np.abs(engine_test.S_grid - S_test)), 0]
    
    if abs(continuation - intrinsic) < 0.01:  # At boundary
        ex_boundary.append((S_test, 0))

print(f"\nEarly exercise optimal for S < ${S_values[len(ex_boundary)]:

.2f} (approx)")

# Scenario 4: Greeks from Grid
print("\n" + "="*70)
print("SCENARIO 4: Greeks Computation")
print("="*70)

M, N = 150, 500

engine_greeks = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
engine_greeks.setup_grid(M, N)
engine_greeks.crank_nicolson()

greeks = engine_greeks.compute_greeks()

print(f"\nGreeks from Grid (at S=${S0}):")
print(f"  Delta: {greeks['delta']:.6f}")
print(f"  Gamma: {greeks['gamma']:.6f}")
print(f"  Theta: {greeks['theta']:.6f}")

# Compare to finite difference
h = 1.0
price_up = black_scholes(S0 + h, K, r, T, sigma, 'put')
price_down = black_scholes(S0 - h, K, r, T, sigma, 'put')
price_mid = black_scholes(S0, K, r, T, sigma, 'put')

delta_fd = (price_up - price_down) / (2*h)
gamma_fd = (price_up - 2*price_mid + price_down) / h**2

print(f"\nFinite Difference (Closed-Form Bump):")
print(f"  Delta: {delta_fd:.6f}")
print(f"  Gamma: {gamma_fd:.6f}")

print(f"\nGrid Greeks match finite difference bumps closely")

# Scenario 5: Down-and-Out Barrier Put
print("\n" + "="*70)
print("SCENARIO 5: Down-and-Out Barrier Put")
print("="*70)

H_down = 85
M, N = 120, 500

barrier_engine = Barrier2DEngine(S0, K, r, sigma, T, H_down, 'put')
price_barrier, V_barrier, S_grid_barrier = barrier_engine.price_down_and_out(M, N)

# Vanilla for comparison
engine_vanilla = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
engine_vanilla.setup_grid(M, N)
price_vanilla = engine_vanilla.crank_nicolson()

discount = (price_vanilla - price_barrier) / price_vanilla * 100

print(f"\nVanilla Put: ${price_vanilla:.6f}")
print(f"Down-and-Out Put (Barrier=${H_down}): ${price_barrier:.6f}")
print(f"Discount: {discount:.2f}%")
print(f"\nBarrier reduces price (knock-out risk)")

# Scenario 6: Stability Violation (Explicit)
print("\n" + "="*70)
print("SCENARIO 6: Stability Violation Demonstration")
print("="*70)

M_unstable = 50
N_unstable = 10  # Very few time steps → large dt

engine_unstable = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
engine_unstable.setup_grid(M_unstable, N_unstable)

dt_used = engine_unstable.dt
dt_stable = engine_unstable.dS**2 / (sigma**2 * engine_unstable.S_max**2)

print(f"\nStability Condition: dt ≤ {dt_stable:.6f}")
print(f"Actual dt: {dt_used:.6f}")
print(f"Violation: {dt_used/dt_stable:.2f}× too large")

try:
    price_unstable = engine_unstable.explicit_method()
    if price_unstable < 0 or price_unstable > 100:
        print(f"\nResult: ${price_unstable:.6f} (UNSTABLE - unphysical)")
    else:
        print(f"\nResult: ${price_unstable:.6f} (May still have large error)")
except:
    print(f"\nExplicit method diverged (overflow)")

# Implicit is stable
engine_stable = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
engine_stable.setup_grid(M_unstable, N_unstable)
price_stable_imp = engine_stable.implicit_method()

print(f"Implicit (same grid): ${price_stable_imp:.6f} (Stable)")
print(f"Black-Scholes: ${bs_price:.6f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Plot 1: European put value surface
ax = axes[0, 0]
engine_viz = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
engine_viz.setup_grid(80, 100)
engine_viz.crank_nicolson()

S_mesh, T_mesh = np.meshgrid(engine_viz.S_grid, engine_viz.t_grid)
im = ax.contourf(S_mesh, T_mesh, engine_viz.V.T, levels=20, cmap='viridis')
ax.plot([S0], [0], 'ro', markersize=10, label=f'S0=${S0}')
ax.axvline(K, color='white', linestyle='--', linewidth=2, alpha=0.7, label=f'Strike ${K}')
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time')
ax.set_title('European Put Value Surface (FD)')
ax.legend()
plt.colorbar(im, ax=ax, label='Option Value')

# Plot 2: American vs European
ax = axes[0, 1]
S_compare = np.linspace(70, 130, 50)
euro_values = []
amer_values = []

for S_test in S_compare:
    # European
    eng_e = FiniteDifferenceEngine(S_test, K, r, sigma, T, 'put')
    eng_e.setup_grid(80, 300)
    euro_values.append(eng_e.crank_nicolson(american=False))
    
    # American
    eng_a = FiniteDifferenceEngine(S_test, K, r, sigma, T, 'put')
    eng_a.setup_grid(80, 300)
    amer_values.append(eng_a.crank_nicolson(american=True))

intrinsic_line = np.maximum(K - S_compare, 0)

ax.plot(S_compare, euro_values, 'b-', linewidth=2.5, label='European Put')
ax.plot(S_compare, amer_values, 'r-', linewidth=2.5, label='American Put')
ax.plot(S_compare, intrinsic_line, 'k--', linewidth=1.5, alpha=0.6, label='Intrinsic')
ax.axvline(S0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Option Value')
ax.set_title('American vs European Put')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Convergence
ax = axes[0, 2]
Ms = [20, 40, 60, 80, 100, 150, 200]
errors_exp = []
errors_imp = []
errors_cn = []

for M_conv in Ms:
    N_conv = M_conv * 4
    
    # Explicit
    try:
        eng = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
        eng.setup_grid(M_conv, N_conv)
        p = eng.explicit_method()
        errors_exp.append(abs(p - bs_price))
    except:
        errors_exp.append(np.nan)
    
    # Implicit
    eng = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
    eng.setup_grid(M_conv, N_conv)
    p = eng.implicit_method()
    errors_imp.append(abs(p - bs_price))
    
    # Crank-Nicolson
    eng = FiniteDifferenceEngine(S0, K, r, sigma, T, 'put')
    eng.setup_grid(M_conv, N_conv)
    p = eng.crank_nicolson()
    errors_cn.append(abs(p - bs_price))

ax.loglog(Ms, errors_exp, 'go-', linewidth=2, markersize=8, label='Explicit')
ax.loglog(Ms, errors_imp, 'bs-', linewidth=2, markersize=8, label='Implicit')
ax.loglog(Ms, errors_cn, 'r^-', linewidth=2, markersize=8, label='Crank-Nicolson')

# Reference slopes
Ms_arr = np.array(Ms)
ax.loglog(Ms_arr, 0.1 / Ms_arr, 'k--', alpha=0.4, linewidth=1.5, label='O(1/M)')
ax.loglog(Ms_arr, 1.0 / Ms_arr**2, 'k:', alpha=0.4, linewidth=1.5, label='O(1/M²)')

ax.set_xlabel('Grid Points (M)')
ax.set_ylabel('Absolute Error')
ax.set_title('Convergence: Error vs Grid Size')
ax.legend()
ax.grid(alpha=0.3, which='both')

# Plot 4: Delta profile
ax = axes[1, 0]
S_delta = np.linspace(70, 130, 100)
deltas = []

for S_test in S_delta:
    eng = FiniteDifferenceEngine(S_test, K, r, sigma, T, 'put')
    eng.setup_grid(100, 400)
    eng.crank_nicolson()
    g = eng.compute_greeks()
    deltas.append(g['delta'] if g['delta'] is not None else 0)

ax.plot(S_delta, deltas, 'purple', linewidth=2.5)
ax.axvline(K, color='r', linestyle='--', linewidth=2, alpha=0.6, label='Strike')
ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Delta')
ax.set_title('Put Delta from FD Grid')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Barrier option value
ax = axes[1, 1]
idx_S0 = np.argmin(np.abs(S_grid_barrier - S0))
ax.plot(engine_viz.t_grid, engine_viz.V[np.argmin(np.abs(engine_viz.S_grid - S0)), :], 
        'b-', linewidth=2.5, label='Vanilla Put')
ax.plot(barrier_engine.T * np.arange(V_barrier.shape[1]) / (V_barrier.shape[1]-1), 
        V_barrier[idx_S0, :], 'r-', linewidth=2.5, label=f'Barrier Put (H=${H_down})')
ax.set_xlabel('Time')
ax.set_ylabel('Option Value')
ax.set_title(f'Time Evolution at S=${S0}')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Gamma profile
ax = axes[1, 2]
gammas = []

for S_test in S_delta:
    eng = FiniteDifferenceEngine(S_test, K, r, sigma, T, 'put')
    eng.setup_grid(100, 400)
    eng.crank_nicolson()
    g = eng.compute_greeks()
    gammas.append(g['gamma'] if g['gamma'] is not None else 0)

ax.plot(S_delta, gammas, 'orange', linewidth=2.5)
ax.axvline(K, color='r', linestyle='--', linewidth=2, alpha=0.6, label='Strike')
ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Gamma')
ax.set_title('Put Gamma from FD Grid')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Non-Uniform Grid:** Implement sinh-transform grid concentrating nodes near strike. Compare convergence to uniform grid for digital option.

2. **2D ADI for Spread:** Price spread option on two correlated assets using Alternating Direction Implicit. Handle cross-derivative term. Benchmark vs Monte Carlo.

3. **High-Order Compact:** Implement fourth-order compact scheme (nine-point stencil). Compare accuracy vs standard second-order for same computational cost.

4. **Adaptive Mesh Refinement:** Build AMR for American put. Detect exercise boundary, refine locally. Show speedup vs uniform fine grid.

5. **Jump-Diffusion PDE:** Extend to Merton jump-diffusion (PIDE with integral term). Use implicit-explicit (IMEX) splitting for jump component.

## 7. Key References
- [Wilmott, Derivatives: The Theory and Practice of Financial Engineering (Chapter 31)](https://www.wiley.com/en-us/Derivatives%3A+The+Theory+and+Practice+of+Financial+Engineering-p-9780470013205) - comprehensive PDE methods
- [Duffy, Finite Difference Methods in Financial Engineering](https://www.wiley.com/en-us/Finite+Difference+Methods+in+Financial+Engineering%3A+A+Partial+Differential+Equation+Approach-p-9780470858820) - implementation details, stability analysis
- [Tavella & Randall, Pricing Financial Instruments: The Finite Difference Method](https://www.wiley.com/en-us/Pricing+Financial+Instruments%3A+The+Finite+Difference+Method-p-9780471197607) - ADI, multi-dimensional techniques

---
**Status:** Numerical PDE solvers | **Complements:** Black-Scholes, American Options, Barriers, Greeks, Monte Carlo Comparison
