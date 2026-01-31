# Numerical Methods for PDEs in Option Pricing

## 1. Concept Skeleton
**Definition:** Finite difference approximations solving partial differential equations (PDEs) for derivative pricing, discretizing Black-Scholes PDE on spatial-temporal grid  
**Purpose:** Price European/American options, path-dependent derivatives when closed-form solutions unavailable, handle early exercise features  
**Prerequisites:** Black-Scholes PDE, Taylor series, numerical stability, boundary conditions

## 2. Comparative Framing
| Method | Explicit Scheme | Implicit Scheme | Crank-Nicolson | ADI Method |
|--------|----------------|-----------------|----------------|------------|
| **Stability** | Conditionally stable (Δt ≤ Δx²/2) | Unconditionally stable | Unconditionally stable | Unconditionally stable |
| **Accuracy** | O(Δt, Δx²) | O(Δt, Δx²) | O(Δt², Δx²) | O(Δt², Δx²) |
| **Computation** | Explicit update (fast) | Matrix inversion (slow) | Tridiagonal solve | Dimensional splitting |
| **American Options** | No | Yes (backward) | Yes (backward) | Yes (backward) |

## 3. Examples + Counterexamples

**Simple Example:**  
European put: Explicit scheme with S∈[0,200], Δx=2, Δt=0.01 → converges to Black-Scholes price within 0.1% (500 time steps)

**Failure Case:**  
Explicit scheme Δt=0.1, Δx=1 → violates stability condition Δt≤Δx²/(2σ²) → oscillations, divergence

**Edge Case:**  
American put near expiry (1 day): Implicit scheme required for early exercise boundary, explicit cannot handle optimal stopping

## 4. Layer Breakdown
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

## 5. Mini-Project
Implement finite difference methods to price European and American options:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# =====================================
# FINITE DIFFERENCE PDE SOLVERS
# =====================================
print("="*70)
print("PDE METHODS FOR OPTION PRICING")
print("="*70)

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Analytical Black-Scholes price for comparison."""
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return price

def explicit_fd_european(S0, K, T, r, sigma, option_type='put', M=100, N=1000):
    """
    Explicit finite difference method for European options.
    
    Conditionally stable: Requires Δt ≤ Δx²/σ²S_max²
    """
    # Grid parameters
    S_max = 3 * K  # Maximum stock price
    S_min = 0
    
    dS = (S_max - S_min) / M
    dt = T / N
    
    # Stock price grid
    S = np.linspace(S_min, S_max, M+1)
    
    # Initialize option values at maturity
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:  # put
        V = np.maximum(K - S, 0)
    
    # Stability check
    max_dt = dS**2 / (sigma**2 * S_max**2)
    if dt > max_dt:
        print(f"   WARNING: Explicit scheme unstable! dt={dt:.6f} > max_dt={max_dt:.6f}")
    
    # Backward time-stepping
    for n in range(N):
        V_new = V.copy()
        
        # Interior points (i=1 to M-1)
        for i in range(1, M):
            # Coefficients from PDE discretization
            a = 0.5 * dt * (r * i - sigma**2 * i**2)
            b = 1 - dt * (sigma**2 * i**2 + r)
            c = 0.5 * dt * (r * i + sigma**2 * i**2)
            
            V_new[i] = a*V[i-1] + b*V[i] + c*V[i+1]
        
        # Boundary conditions
        if option_type == 'call':
            V_new[0] = 0  # V(0,t) = 0 for call
            V_new[M] = S_max - K*np.exp(-r*(T - (n+1)*dt))  # V(S_max,t) ≈ S_max
        else:  # put
            V_new[0] = K*np.exp(-r*(T - (n+1)*dt))  # V(0,t) = Ke^(-r(T-t))
            V_new[M] = 0  # V(S_max,t) = 0 for put
        
        V = V_new
    
    # Interpolate to S0
    option_price = np.interp(S0, S, V)
    
    return option_price, S, V

def implicit_fd_european(S0, K, T, r, sigma, option_type='put', M=100, N=500):
    """
    Implicit finite difference method for European options.
    
    Unconditionally stable, requires solving tridiagonal system.
    """
    S_max = 3 * K
    S_min = 0
    
    dS = (S_max - S_min) / M
    dt = T / N
    
    S = np.linspace(S_min, S_max, M+1)
    
    # Initialize at maturity
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)
    
    # Construct tridiagonal matrix
    alpha = np.zeros(M+1)
    beta = np.zeros(M+1)
    gamma = np.zeros(M+1)
    
    for i in range(1, M):
        alpha[i] = -0.5 * dt * (r * i - sigma**2 * i**2)
        beta[i] = 1 + dt * (sigma**2 * i**2 + r)
        gamma[i] = -0.5 * dt * (r * i + sigma**2 * i**2)
    
    # Create tridiagonal matrix
    diagonals = [alpha[1:M], beta[1:M], gamma[1:M-1]]
    A = diags(diagonals, [-1, 0, 1], shape=(M-1, M-1), format='csc')
    
    # Backward time-stepping
    for n in range(N):
        b = V[1:M].copy()
        
        # Adjust for boundary conditions
        if option_type == 'call':
            V[0] = 0
            V[M] = S_max - K*np.exp(-r*(T - (n+1)*dt))
        else:
            V[0] = K*np.exp(-r*(T - (n+1)*dt))
            V[M] = 0
        
        b[0] -= alpha[1] * V[0]
        b[-1] -= gamma[M-1] * V[M]
        
        # Solve tridiagonal system
        V[1:M] = spsolve(A, b)
    
    option_price = np.interp(S0, S, V)
    return option_price, S, V

def crank_nicolson_european(S0, K, T, r, sigma, option_type='put', M=100, N=500):
    """
    Crank-Nicolson method for European options (θ=0.5).
    
    Second-order accurate in time, unconditionally stable.
    """
    S_max = 3 * K
    S_min = 0
    
    dS = (S_max - S_min) / M
    dt = T / N
    
    S = np.linspace(S_min, S_max, M+1)
    
    # Initialize at maturity
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)
    
    # Coefficients
    alpha = np.zeros(M+1)
    beta = np.zeros(M+1)
    gamma = np.zeros(M+1)
    
    for i in range(1, M):
        alpha[i] = 0.25 * dt * (r * i - sigma**2 * i**2)
        beta[i] = -0.5 * dt * (sigma**2 * i**2 + r)
        gamma[i] = 0.25 * dt * (r * i + sigma**2 * i**2)
    
    # LHS matrix (implicit part)
    A_diags = [-alpha[1:M], 1-beta[1:M], -gamma[1:M-1]]
    A = diags(A_diags, [-1, 0, 1], shape=(M-1, M-1), format='csc')
    
    # Backward time-stepping
    for n in range(N):
        # RHS (explicit part)
        b = np.zeros(M-1)
        for i in range(1, M):
            b[i-1] = alpha[i]*V[i-1] + (1+beta[i])*V[i] + gamma[i]*V[i+1]
        
        # Boundary conditions
        if option_type == 'call':
            V[0] = 0
            V[M] = S_max - K*np.exp(-r*(T - (n+1)*dt))
        else:
            V[0] = K*np.exp(-r*(T - (n+1)*dt))
            V[M] = 0
        
        b[0] -= alpha[1] * V[0]
        b[-1] -= gamma[M-1] * V[M]
        
        # Solve system
        V[1:M] = spsolve(A, b)
    
    option_price = np.interp(S0, S, V)
    return option_price, S, V

def implicit_fd_american(S0, K, T, r, sigma, option_type='put', M=100, N=500):
    """
    Implicit finite difference with early exercise for American options.
    
    Uses projected SOR to enforce V ≥ Payoff constraint.
    """
    S_max = 3 * K
    S_min = 0
    
    dS = (S_max - S_min) / M
    dt = T / N
    
    S = np.linspace(S_min, S_max, M+1)
    
    # Payoff function
    if option_type == 'call':
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)
    
    V = payoff.copy()
    
    # Tridiagonal matrix coefficients
    alpha = np.zeros(M+1)
    beta = np.zeros(M+1)
    gamma = np.zeros(M+1)
    
    for i in range(1, M):
        alpha[i] = -0.5 * dt * (r * i - sigma**2 * i**2)
        beta[i] = 1 + dt * (sigma**2 * i**2 + r)
        gamma[i] = -0.5 * dt * (r * i + sigma**2 * i**2)
    
    diagonals = [alpha[1:M], beta[1:M], gamma[1:M-1]]
    A = diags(diagonals, [-1, 0, 1], shape=(M-1, M-1), format='csc')
    
    # Track early exercise boundary
    exercise_boundary = []
    
    # Backward time-stepping
    for n in range(N):
        b = V[1:M].copy()
        
        # Boundary conditions
        if option_type == 'call':
            V[0] = 0
            V[M] = S_max
        else:
            V[0] = K*np.exp(-r*(T - (n+1)*dt))
            V[M] = 0
        
        b[0] -= alpha[1] * V[0]
        b[-1] -= gamma[M-1] * V[M]
        
        # Solve as European
        V_european = spsolve(A, b)
        
        # Enforce early exercise constraint
        V[1:M] = np.maximum(V_european, payoff[1:M])
        
        # Find exercise boundary (first point where V = payoff)
        if option_type == 'put':
            exercise_idx = np.where(V[1:M] <= payoff[1:M] + 1e-6)[0]
            if len(exercise_idx) > 0:
                exercise_boundary.append(S[exercise_idx[-1]+1])
            else:
                exercise_boundary.append(S_min)
    
    option_price = np.interp(S0, S, V)
    return option_price, S, V, exercise_boundary

# =====================================
# TEST CASE: EUROPEAN PUT OPTION
# =====================================
print("\n" + "="*70)
print("EUROPEAN PUT OPTION")
print("="*70)

S0 = 100  # Current stock price
K = 100   # Strike price
T = 1.0   # Time to maturity (1 year)
r = 0.05  # Risk-free rate
sigma = 0.20  # Volatility

# Analytical Black-Scholes price
bs_price = black_scholes_price(S0, K, T, r, sigma, option_type='put')

# PDE methods
explicit_price, S_grid, V_explicit = explicit_fd_european(S0, K, T, r, sigma, 'put', M=100, N=2000)
implicit_price, _, V_implicit = implicit_fd_european(S0, K, T, r, sigma, 'put', M=100, N=500)
cn_price, _, V_cn = crank_nicolson_european(S0, K, T, r, sigma, 'put', M=100, N=500)

print(f"\nOption Parameters:")
print(f"   S0 = ${S0}, K = ${K}, T = {T} years")
print(f"   r = {r:.1%}, σ = {sigma:.1%}")

print(f"\nPricing Results:")
print(f"   Black-Scholes (Analytical): ${bs_price:.4f}")
print(f"   Explicit FD:                ${explicit_price:.4f} (Error: ${explicit_price-bs_price:.4f})")
print(f"   Implicit FD:                ${implicit_price:.4f} (Error: ${implicit_price-bs_price:.4f})")
print(f"   Crank-Nicolson:             ${cn_price:.4f} (Error: ${cn_price-bs_price:.4f})")

# =====================================
# AMERICAN PUT OPTION
# =====================================
print("\n" + "="*70)
print("AMERICAN PUT OPTION (EARLY EXERCISE)")
print("="*70)

american_price, S_grid_am, V_american, exercise_boundary = implicit_fd_american(
    S0, K, T, r, sigma, 'put', M=100, N=500
)

print(f"\nAmerican Put Price: ${american_price:.4f}")
print(f"European Put Price: ${bs_price:.4f}")
print(f"Early Exercise Premium: ${american_price - bs_price:.4f} ({(american_price/bs_price-1)*100:.2f}%)")

if len(exercise_boundary) > 0:
    print(f"\nEarly Exercise Boundary:")
    print(f"   Near maturity: S* ≈ ${exercise_boundary[0]:.2f}")
    print(f"   Near t=0: S* ≈ ${exercise_boundary[-1]:.2f}")

# =====================================
# CONVERGENCE ANALYSIS
# =====================================
print("\n" + "="*70)
print("CONVERGENCE ANALYSIS")
print("="*70)

grid_sizes = [20, 40, 80, 160]
explicit_errors = []
implicit_errors = []
cn_errors = []

for M in grid_sizes:
    N = M * 20  # Explicit needs fine time grid
    exp_p, _, _ = explicit_fd_european(S0, K, T, r, sigma, 'put', M=M, N=N)
    explicit_errors.append(abs(exp_p - bs_price))
    
    N = M * 5  # Implicit can use coarser grid
    imp_p, _, _ = implicit_fd_european(S0, K, T, r, sigma, 'put', M=M, N=N)
    implicit_errors.append(abs(imp_p - bs_price))
    
    cn_p, _, _ = crank_nicolson_european(S0, K, T, r, sigma, 'put', M=M, N=N)
    cn_errors.append(abs(cn_p - bs_price))

print("\nGrid Size vs Absolute Error:")
for M, exp_err, imp_err, cn_err in zip(grid_sizes, explicit_errors, implicit_errors, cn_errors):
    print(f"   M={M:3d}: Explicit ${exp_err:.5f}, Implicit ${imp_err:.5f}, C-N ${cn_err:.5f}")

# =====================================
# VISUALIZATION
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Option value vs stock price
axes[0, 0].plot(S_grid, V_cn, linewidth=2, label='European Put (Crank-Nicolson)')
axes[0, 0].plot(S_grid_am, V_american, linewidth=2, label='American Put')
axes[0, 0].plot(S_grid, np.maximum(K - S_grid, 0), 'k--', linewidth=1, label='Intrinsic Value')
axes[0, 0].axvline(S0, color='red', linestyle=':', alpha=0.5, label=f'S0=${S0}')
axes[0, 0].set_xlabel('Stock Price S')
axes[0, 0].set_ylabel('Option Value')
axes[0, 0].set_title('Put Option Value at t=0')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xlim(0, 150)

# Plot 2: Convergence comparison
axes[0, 1].loglog(grid_sizes, explicit_errors, 'o-', linewidth=2, label='Explicit FD', markersize=8)
axes[0, 1].loglog(grid_sizes, implicit_errors, 's-', linewidth=2, label='Implicit FD', markersize=8)
axes[0, 1].loglog(grid_sizes, cn_errors, '^-', linewidth=2, label='Crank-Nicolson', markersize=8)
axes[0, 1].set_xlabel('Grid Size M')
axes[0, 1].set_ylabel('Absolute Error ($)')
axes[0, 1].set_title('Convergence: Error vs Grid Size')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, which='both')

# Plot 3: Early exercise boundary
if len(exercise_boundary) > 0:
    time_grid = np.linspace(0, T, len(exercise_boundary))
    axes[1, 0].plot(time_grid, exercise_boundary, linewidth=2, color='red')
    axes[1, 0].axhline(K, color='black', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
    axes[1, 0].fill_between(time_grid, 0, exercise_boundary, alpha=0.2, color='red', 
                           label='Early Exercise Region')
    axes[1, 0].set_xlabel('Time to Maturity')
    axes[1, 0].set_ylabel('Stock Price S*')
    axes[1, 0].set_title('American Put: Early Exercise Boundary')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim(0, K*1.2)

# Plot 4: Time value comparison
time_value_european = V_cn - np.maximum(K - S_grid, 0)
time_value_american = V_american - np.maximum(K - S_grid_am, 0)

axes[1, 1].plot(S_grid, time_value_european, linewidth=2, label='European')
axes[1, 1].plot(S_grid_am, time_value_american, linewidth=2, label='American')
axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
axes[1, 1].axvline(S0, color='red', linestyle=':', alpha=0.5)
axes[1, 1].set_xlabel('Stock Price S')
axes[1, 1].set_ylabel('Time Value')
axes[1, 1].set_title('Time Value = Option Value - Intrinsic Value')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim(0, 150)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("PDE methods successfully price options:")
print(f"• Crank-Nicolson most accurate (Error ${cn_errors[-1]:.5f} at M=160)")
print(f"• American put premium ${american_price-bs_price:.3f} from early exercise")
print(f"• Exercise boundary: S*≈${exercise_boundary[-1]:.1f} at t=0 (exercise if S<S*)")
print(f"• Convergence: Second-order in space (O(Δx²)), C-N second-order in time")
```

## 6. Challenge Round
When do PDE methods outperform Monte Carlo?
- **Low dimensions:** 1-2 factors (single underlying) → PDE O(M²N) efficient, MC needs 10⁶+ paths
- **American options:** Early exercise boundary naturally handled via backward iteration, MC requires regression (Longstaff-Schwartz)
- **Greeks calculation:** Delta/Gamma from finite differences on grid, MC requires multiple simulations or pathwise derivatives
- **Barrier options:** Discrete monitoring approximated well on grid, MC struggles with continuous barriers

When PDE methods struggle: High dimensions (>3 factors) → curse of dimensionality, grid size M^d explodes; path-dependent with complex history (Asian options with daily averaging) → state space too large.

## 7. Key References
- [Wilmott (2006) Paul Wilmott on Quantitative Finance](https://www.wiley.com/en-us/Paul+Wilmott+on+Quantitative+Finance%2C+3+Volume+Set%2C+2nd+Edition-p-9780470018704) - Comprehensive PDE methods
- [Hull (2018) Options, Futures, and Other Derivatives, Ch. 21](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) - Numerical procedures
- [Duffy (2006) Finite Difference Methods in Financial Engineering](https://www.wiley.com/en-us/Finite+Difference+Methods+in+Financial+Engineering%3A+A+Partial+Differential+Equation+Approach-p-9780470858820) - Advanced FD techniques
- [Tavella & Randall (2000) Pricing Financial Instruments: The Finite Difference Method](https://www.wiley.com/en-us/Pricing+Financial+Instruments%3A+The+Finite+Difference+Method-p-9780471197607) - Implementation guide

---
**Status:** Core numerical technique | **Complements:** Black-Scholes Model, Monte Carlo Pricing, American Options
