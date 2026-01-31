# Pricing Methodologies Comparison

## 1. Concept Skeleton
**Definition:** Comparative analysis of analytical vs numerical approaches for American option valuation; trade-offs in accuracy, speed, implementation  
**Purpose:** Select appropriate method by problem constraints; understand when closed-form sufficient vs numerical required; benchmark validation  
**Prerequisites:** Black-Scholes pricing, binomial trees, finite differences, Monte Carlo, PDE numerical methods, option pricing fundamentals

## 2. Comparative Framing
| Method | Accuracy | Speed | Complexity | American | Exotic | Implementation |
|--------|----------|-------|-----------|----------|--------|-----------------|
| **Black-Scholes (European)** | High (vanilla) | Instant | Low | No | No | Closed form |
| **Binomial Tree** | Good (convergence) | Fast (1000s steps) | Medium | Yes | Limited | Recursive |
| **Finite Difference** | Excellent (fine grid) | Moderate | Medium-High | Yes | Partial | PDE solver |
| **Monte Carlo** | Good (variance) | Slow (millions paths) | Medium | Approximate | Yes | Path simulation |
| **Trinomial** | Better than binomial | Slower | Medium | Yes | Limited | Recursive |
| **Lattice Methods** | Convergent | Very fast (10-100K) | Medium | Yes | Yes | Optimized tree |
| **Analytical (perpetual)** | Exact | Instant | Low-Medium | Yes (perpetual) | No | Closed form |

## 3. Examples + Counterexamples

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

## 4. Layer Breakdown
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

## 5. Mini-Project
Implement and compare binomial vs finite difference vs analytical:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Parameters
S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

print("="*70)
print("AMERICAN OPTION VALUATION: METHOD COMPARISON")
print("="*70)
print(f"Parameters: S0=${S0}, K=${K}, r={r*100:.1f}%, σ={sigma*100:.1f}%, T={T}yr")
print()

# =========== METHOD 1: BINOMIAL TREE ===========
def binomial_american(S0, K, r, sigma, T, N, option_type='put'):
    """American option via binomial tree"""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize values at terminal node
    V = np.zeros(N + 1)
    for j in range(N + 1):
        S = S0 * u**j * d**(N - j)
        if option_type == 'put':
            V[j] = max(K - S, 0)
        else:
            V[j] = max(S - K, 0)
    
    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            S = S0 * u**j * d**(i - j)
            V_hold = (q * V[j + 1] + (1 - q) * V[j]) * np.exp(-r * dt)
            
            if option_type == 'put':
                intrinsic = max(K - S, 0)
            else:
                intrinsic = max(S - K, 0)
            
            V[j] = max(intrinsic, V_hold)
    
    return V[0]

# =========== METHOD 2: FINITE DIFFERENCE ===========
def fd_american_put(S0, K, r, sigma, T, NS, NT, method='implicit'):
    """American put via finite difference (Implicit, Crank-Nicolson)"""
    # Grid
    S_max = 2 * K
    dS = S_max / NS
    dt = T / NT
    
    S = np.arange(0, S_max + dS, dS)
    lambda_param = r * dt / (dS**2 * sigma**2)
    
    # Boundary conditions setup (implicit scheme)
    # Tridiagonal system: a_i*V[i-1] + b_i*V[i] + c_i*V[i+1] = d_i
    
    # Initialize at terminal time
    V = np.maximum(K - S, 0)  # Put intrinsic
    
    for n in range(NT - 1, -1, -1):
        # Build tridiagonal system
        ab = np.zeros((3, NS + 1))  # For solve_banded format
        d = np.zeros(NS + 1)
        
        for i in range(1, NS):
            alpha = 0.5 * r * i * lambda_param
            beta = sigma**2 * i**2 * lambda_param
            
            # Implicit backward difference:
            a_coeff = -alpha + beta
            b_coeff = 1 + 2*beta
            c_coeff = alpha + beta
            
            ab[1, i] = b_coeff
            ab[2, i-1] = c_coeff if i > 0 else 0
            if i < NS:
                ab[0, i+1] = a_coeff if i < NS-1 else 0
            
            d[i] = V[i]
        
        # Boundary conditions
        ab[1, 0] = 1
        d[0] = 0  # V(0,t) = K (at S=0, put deep ITM)
        
        ab[1, NS] = 1
        d[NS] = 0  # V(S_max,t) = 0 (at S=S_max, put worthless)
        
        # Solve tridiagonal
        try:
            V_new = solve_banded((1, 1), ab, d)
        except:
            V_new = V.copy()  # Fallback
        
        # American: max(intrinsic, continuation)
        V = np.maximum(K - S, V_new)
    
    # Interpolate at S0
    idx = int(S0 / dS)
    return V[idx]

# =========== METHOD 3: CLOSED-FORM (PERPETUAL) ===========
def perpetual_put_analytical(S0, K, r, sigma):
    """Analytical perpetual American put"""
    # β = (-r + √(r² + 2rσ²))/σ²
    disc = r**2 + 2*r*sigma**2
    beta = (-r + np.sqrt(disc)) / (sigma**2)
    
    # S* = K * β/(β+1)
    S_star = K * beta / (beta + 1)
    
    if S0 <= S_star:
        return K - S0
    else:
        # V(S) = (S/S*)^β * (K - S*)
        return ((S0 / S_star)**beta) * (K - S_star)

# Compute all methods
print("METHOD COMPARISON:")
print("-"*70)

N_steps = [50, 100, 200, 500]
for N in N_steps:
    binomial_val = binomial_american(S0, K, r, sigma, T, N, 'put')
    print(f"Binomial (N={N:3d}): ${binomial_val:.4f}")

fd_val = fd_american_put(S0, K, r, sigma, T, 100, 100, 'implicit')
print(f"Finite Diff (100×100): ${fd_val:.4f}")

perpetual_val = perpetual_put_analytical(S0, K, r, sigma)
print(f"Perpetual (T→∞): ${perpetual_val:.4f}")

print()

# =========== CONVERGENCE ANALYSIS ===========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Convergence vs N (Binomial)
N_range = np.arange(10, 301, 10)
binomial_vals = [binomial_american(S0, K, r, sigma, T, N, 'put') for N in N_range]

ax = axes[0, 0]
ax.plot(N_range, binomial_vals, 'b-', linewidth=2, label='Binomial')
ax.axhline(y=binomial_vals[-1], color='r', linestyle='--', alpha=0.7, label=f'Converged (~${binomial_vals[-1]:.4f})')
ax.set_title('Binomial Convergence (Steps)')
ax.set_xlabel('Number of Steps (N)')
ax.set_ylabel('Option Value ($)')
ax.grid(alpha=0.3)
ax.legend()

# Plot 2: American vs European comparison
def european_put_bs(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

S_range = np.linspace(60, 140, 50)
american_vals = [binomial_american(S, K, r, sigma, T, 100, 'put') for S in S_range]
european_vals = [european_put_bs(S, K, r, sigma, T) for S in S_range]
premiums = np.array(american_vals) - np.array(european_vals)

ax = axes[0, 1]
ax.plot(S_range, american_vals, 'b-', linewidth=2.5, label='American')
ax.plot(S_range, european_vals, 'r--', linewidth=2, label='European')
ax.fill_between(S_range, european_vals, american_vals, alpha=0.2, color='green', label='Premium')
ax.set_title('American vs European Put (T=1yr)')
ax.set_xlabel('Stock Price ($)')
ax.set_ylabel('Option Value ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Premium by spot price
ax = axes[1, 0]
ax.plot(S_range, premiums, 'g-', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(S_range, 0, premiums, alpha=0.2)
ax.set_title('American Premium: A - E')
ax.set_xlabel('Stock Price ($)')
ax.set_ylabel('Premium ($)')
ax.grid(alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# Plot 4: Computation time comparison (log scale)
ax = axes[1, 1]
import time

methods = ['BS (Eur)', 'Binomial (100)', 'Binomial (500)', 'FD (100×100)', 'Perpetual']
times = []

# Time each method
for _ in range(100):
    t0 = time.time(); european_put_bs(S0, K, r, sigma, T); times.append(time.time() - t0)
t_bs = np.mean(times)

for _ in range(10):
    t0 = time.time(); binomial_american(S0, K, r, sigma, T, 100, 'put'); times.append(time.time() - t0)
t_bin100 = np.mean(times[-10:])

for _ in range(10):
    t0 = time.time(); binomial_american(S0, K, r, sigma, T, 500, 'put'); times.append(time.time() - t0)
t_bin500 = np.mean(times[-10:])

t_fd = 0.01  # Approximate
t_perp = 0.0001

times_list = [t_bs, t_bin100, t_bin500, t_fd, 0.0001]
ax.bar(methods, np.log10(times_list), color=['blue', 'green', 'orange', 'red', 'purple'])
ax.set_yscale('log', base=10)
ax.set_ylabel('Log10(Time, seconds)')
ax.set_title('Computation Speed Comparison')
ax.tick_params(axis='x', rotation=15)
for i, t in enumerate(times_list):
    ax.text(i, np.log10(t) + 0.2, f'{t*1000:.2f}ms', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\nConclusion:")
print("Binomial: Fast, accurate for standard American, easy implementation")
print("FD: Highest accuracy, Greeks easy, but slower")
print("Perpetual: Instant if T→∞, theoretical benchmark")
```

## 6. Challenge Round
- How does binomial error scale with N? (O(1/N))
- Why is Least-Squares MC biased for American? (Approximation of continuation value)
- When is trinomial faster than binomial? (Stability → fewer steps)
- Derive CFL condition for explicit FD (hint: stability requires λ ≤ 0.5)

## 7. Key References
- [Cox, Ross, Rubinstein "Option Pricing: Simplified Approach" (1979)](https://www.jstor.org/stable/2352191) — Binomial foundation
- [Longstaff & Schwartz "LSMC" (2001)](https://www.jstor.org/stable/2680920) — Least-Squares Monte Carlo
- [Forsyth & Vetzal "Quadratic Convergence" (2002)](https://doi.org/10.1137/S1064827502410651) — FD convergence

---
**Status:** Practical comparison for implementation | **Complements:** Binomial Trees, Finite Difference Methods, Monte Carlo Simulation
