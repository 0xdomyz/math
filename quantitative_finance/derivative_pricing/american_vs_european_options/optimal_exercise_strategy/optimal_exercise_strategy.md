# Optimal Exercise Strategy

## 1. Concept Skeleton
**Definition:** Mathematical framework determining the theoretically optimal time τ* to exercise an American option that maximizes expected discounted payoff  
**Purpose:** Characterizes free boundary problem; generates exercise region S*(t); enables valuation via backward induction; underpins numerical methods  
**Prerequisites:** Optimal stopping theory, dynamic programming, martingale theory, risk-neutral valuation, boundary value problems

## 2. Comparative Framing
| Strategy | Greedy | Myopic | Optimal | Reality |
|----------|--------|--------|---------|---------|
| **Decision** | Exercise if ITM | Consider immediate value | Maximize E[discounted payoff] | Exercise heuristically |
| **Horizon** | Now only | Next period | Full remaining life | Truncated horizon |
| **Accuracy** | Poor (leaves value) | Better | Exact (theoretical) | Implementation noise |
| **Compute** | Instant | Moderate | Intensive (backward) | Approx via grid |
| **When Best** | High dividends | Short T | Perpetual/long T | Liquid markets |

## 3. Examples + Counterexamples

**Simple Example:**  
Put S=90, K=100, r=5%, σ=20%, T=6mo, τ∈[0,T]. Intrinsic now = $10. Optimal policy: If S stays <75, exercise; if S>85, hold. S*(t) decreases as t→T.

**Excellent Fit - Perpetual Put:**  
No expiration (T=∞). Analytic solution: S* = K/(1+ρ) where ρ=r/(0.5σ²). Exercise is stationary rule.

**Dividend Call:**  
S=105, K=100, q=8%, r=5%, T=1yr. Just before ex-dividend: S drops 2-3%. Optimal: Exercise just before ex-date (S* path jumps up).

**Poor Fit:**  
Microstructure: Real traders face bid-ask, slippage, discrete monitoring. Exercise boundary theory assumes frictionless continuous space.

**Negative Rate Environment:**  
r=-2% (real), K=$100. Deep ITM put: Immediate exercise is suboptimal (r pushes present value < future). Hold longer.

## 4. Layer Breakdown
```
Optimal Exercise Strategy:

├─ Problem Formulation (Optimal Stopping):
│  ├─ Define value function:
│  │   V(S,t) = max_τ E^Q[e^(-r(τ-t)) Payoff(S_τ,τ) | S(t)=S]
│  │   where τ ∈ [t, T_exp]
│  ├─ Bellman equation (dynamic programming):
│  │   V(S,t) = max(Payoff(S,t), E^Q[e^(-rdt) V(S+dS,t+dt)])
│  ├─ Optimal policy:
│  │   If V(S,t) = Payoff(S,t) → Exercise now (exercise region)
│  │   If V(S,t) > Payoff(S,t) → Continue (hold region)
│  └─ Free boundary S*(t):
│      Separates exercise from hold region; unknown a priori
├─ PDE Formulation:
│  ├─ In hold region (S > S*(t)):
│  │   ∂V/∂t + rS(∂V/∂S) + 0.5σ²S²(∂²V/∂S²) = rV
│  │   LV = rV [standard option PDE]
│  ├─ Boundary conditions:
│  │   V(S*(t),t) = Payoff(S*(t),t) [value matching]
│  │   ∂V/∂S|_{S=S*(t)} = ∂Payoff/∂S [smooth pasting]
│  ├─ Terminal condition:
│  │   V(S,T) = Payoff(S,T) = max(S-K, 0) [call] or max(K-S, 0) [put]
│  └─ Far-field:
│      As S→0 (call): V→0; As S→∞ (call): V→S
├─ Smooth Pasting Condition:
│  ├─ Intuition:
│  │   Differentiability ensures option value smooth at boundary
│  │   If not smooth: arbitrage opportunity (jump in hedge ratio)
│  ├─ Mathematical:
│  │   C¹ continuity at S*: 
│  │   V_S(S*⁻,t) = V_S(S*⁺,t) [derivatives match]
│  ├─ For put:
│  │   ∂V/∂S|_{S=S*} = -1 [slope of K-S]
│  ├─ For call (dividend):
│  │   ∂V/∂S|_{S=S*} = 1 [slope of S-K]
│  └─ Determines S*(t) uniquely (along with value matching)
├─ Perpetual American Options (T=∞):
│  ├─ Simplification:
│  │   S*(t) → S* (constant)
│  │   ∂V/∂t = 0 (stationary)
│  ├─ ODE in hold region:
│  │   0.5σ²S² V''(S) + rS V'(S) - rV(S) = 0
│  │   Power solution: V(S) = A S^β (for S > S*)
│  ├─ Perpetual put solution:
│  │   β = (-r + √(r² + 2rσ²))/σ² > 1
│  │   Wait region: V = A(S/S*)^β for S > S*
│  │   Exercise region: V = K - S for S ≤ S*
│  │   Smooth pasting:
│  │     V(S*) = K - S* [value match]
│  │     V'(S*) = -1 [smooth pasting]
│  │   Solution: S* = K*β/(β+1) where β from roots above
│  ├─ Perpetual call with dividend q:
│  │   β₊ = (-(r-q) + √((r-q)² + 2(r-q)σ²))/σ²
│  │   S* = (β₊/(β₊-1)) * K
│  └─ Closed form: Unique advantage for perpetuals
├─ Finite Time (American Options):
│  ├─ Complexity:
│  │   S*(t) depends on t (boundary curve)
│  │   Moving boundary problem (challenging)
│  ├─ Limiting behavior:
│  │   t→T: S*(T) → intrinsic payoff region (exercise boundary)
│  │   t→0: S*(0) → S*_perpetual (approaches perpetual as T→large)
│  ├─ For puts:
│  │   S*(t) typically increases toward K as T increases
│  │   Far from expiry: Boundary near 70% of K
│  │   At expiry (T=0): S*=K (exercise if S≤K)
│  ├─ Properties:
│  │   S* is non-increasing in t for European-style puts
│  │   S* increases with volatility (more holding value)
│  │   S* decreases with interest rates (immediate payout valuable)
│  └─ Monotonicity: Used in numerical validation
├─ Comparison: Calls vs Puts:
│  ├─ Non-dividend call:
│  │   S*(t) = ∞ for all t (never exercise!)
│  │   American = European
│  ├─ Dividend call:
│  │   S*(t) finite; exercise just before ex-dividend
│  │   Boundary: Complicated path dependent on dividend dates
│  ├─ Put (any parameters):
│  │   S*(t) ∈ (0, K) always
│  │   Boundary: Decreases toward strike near expiry
│  └─ Intuition: Asymmetry due to payoff structure
├─ Numerical Solution (Backward Induction):
│  ├─ Discretize space/time:
│  │   S_j = j*ΔS, t_n = n*Δt
│  ├─ Backward loop (n=N-1 down to 0):
│  │   For each S_j at time t_n:
│  │   └─ V(j,n) = max(Payoff(S_j,t_n), [value from PDE])
│  ├─ PDE discretization:
│  │   Finite difference / binomial recombination
│  ├─ Locate boundary:
│  │   Find S_j* where V just crosses Payoff
│  │   Interpolate S* between grid points
│  └─ Convergence: Refinement of mesh → exact S*
└─ Extensions:
   ├─ Multi-dimensional:
     Basket of assets: Boundary becomes surface in ℝⁿ
   ├─ Stochastic volatility:
     Boundary S*(σ,t) now depends on volatility state
   ├─ Transaction costs:
     Smooth pasting modified; "hysteresis band"
   └─ Constraints:
       Prohibited dates (Bermuda): Boundary only at allowed dates
```

**Interaction:** Optimal strategy determined by balance: (1) Intrinsic value now, (2) Option value from continued holding, (3) Discounting effect over remaining time.

## 5. Mini-Project
Compute perpetual put and finite-time put boundaries:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq
from scipy.stats import norm

def perpetual_put_boundary(K, r, sigma):
    """Compute S* for perpetual American put"""
    # β = (-r + √(r² + 2rσ²))/σ²
    disc = r**2 + 2*r*sigma**2
    beta = (-r + np.sqrt(disc)) / (sigma**2)
    S_star = K * beta / (beta + 1)
    return S_star, beta

def perpetual_put_value(S, K, r, sigma):
    """Value function for perpetual American put"""
    S_star, beta = perpetual_put_boundary(K, r, sigma)
    if S <= S_star:
        return K - S  # Exercise
    else:
        # V(S) = A*(S/S*)^β * (K-S*), solve for A from smooth pasting
        A = (S_star**(1-beta)) * (beta + 1) / beta
        return A * (S**beta) - A*beta*S_star + (K - S_star)

# Parameters
K = 100
r_values = [0.02, 0.05, 0.10]
sigma = 0.2
S_range = np.linspace(1, 150, 200)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Perpetual boundary vs interest rates
ax = axes[0, 0]
boundaries = []
for r in r_values:
    S_star, beta = perpetual_put_boundary(K, r, sigma)
    boundaries.append(S_star)
    ax.axhline(y=S_star, linestyle='--', label=f'r={r*100:.1f}%, S*=${S_star:.2f}')

ax.set_ylim(0, K)
ax.set_xlim(0, 3)
ax.set_title('Perpetual Put Exercise Boundary vs Interest Rate')
ax.set_xlabel('Interest Rate (%)')
ax.set_ylabel('Exercise Boundary S*')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Perpetual put value vs spot
ax = axes[0, 1]
r = 0.05
S_star, beta = perpetual_put_boundary(K, r, sigma)
put_vals = [perpetual_put_value(S, K, r, sigma) for S in S_range]
intrinsic = np.maximum(K - S_range, 0)

ax.plot(S_range, put_vals, 'b-', linewidth=2, label='V(S)')
ax.plot(S_range, intrinsic, 'k--', linewidth=1.5, label='Intrinsic')
ax.axvline(x=S_star, color='r', linestyle=':', linewidth=2, label=f'S*=${S_star:.2f}')
ax.set_title(f'Perpetual Put Value (r={r*100:.1f}%, σ={sigma*100:.1f}%)')
ax.set_xlabel('Stock Price S')
ax.set_ylabel('Option Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Beta sensitivity
sigma_range = np.linspace(0.05, 0.50, 50)
beta_vals = []
for sig in sigma_range:
    _, beta = perpetual_put_boundary(K, r, sig)
    beta_vals.append(beta)

ax = axes[1, 0]
ax.plot(sigma_range*100, beta_vals, 'g-', linewidth=2)
ax.set_title(f'Beta Exponent vs Volatility (r={r*100:.1f}%)')
ax.set_xlabel('Volatility σ (%)')
ax.set_ylabel('β (power in value function)')
ax.grid(alpha=0.3)

# Plot 4: Finite-time put boundary (approximated by binomial)
def american_put_binomial(S0, K, r, sigma, T, N=50, q=0):
    """Simple binomial tree for American put"""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r-q)*dt) - d) / (u - d)
    
    # Initialize tree
    V = np.zeros((N+1, N+1))
    S = np.zeros((N+1, N+1))
    
    # Terminal nodes
    for j in range(N+1):
        S[N, j] = S0 * (u**j) * (d**(N-j))
        V[N, j] = max(K - S[N, j], 0)
    
    # Backward induction
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            S[i, j] = S0 * (u**j) * (d**(i-j))
            # European value
            V_hold = (p*V[i+1, j+1] + (1-p)*V[i+1, j]) * np.exp(-r*dt)
            # American: max(intrinsic, hold)
            V[i, j] = max(K - S[i, j], V_hold)
    
    return V[0, 0], S

# Compute boundaries for different T
T_values = [0.5, 1.0, 2.0, 5.0]
ax = axes[1, 1]

time_points = np.linspace(0, T_values[-1], 30)
for T in T_values:
    boundaries_T = []
    S_test = np.linspace(50, 150, 50)
    
    for S_val in S_test:
        val, _ = american_put_binomial(S_val, K, r, sigma, T, N=30)
        intrinsic_val = max(K - S_val, 0)
        # Boundary approximately where value ≈ intrinsic
        if abs(val - intrinsic_val) < 1:
            boundaries_T.append(S_val)
    
    if boundaries_T:
        S_boundary = np.mean(boundaries_T)
        ax.scatter(T, S_boundary, s=100, label=f'T={T}yr')

# Add perpetual as limit
S_perp, _ = perpetual_put_boundary(K, r, sigma)
ax.axhline(y=S_perp, color='k', linestyle='--', linewidth=1.5, 
          label=f'Perpetual: S*=${S_perp:.2f}')
ax.set_title('American Put Exercise Boundary vs Time to Expiry')
ax.set_xlabel('Time to Expiry T (years)')
ax.set_ylabel('Exercise Boundary S*')
ax.set_ylim(60, 100)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Display key results
print("Perpetual American Put Analysis")
print("="*50)
for r_val in r_values:
    S_star, beta = perpetual_put_boundary(K, r_val, sigma)
    print(f"r={r_val*100:.1f}%: S*=${S_star:.2f}, β={beta:.3f}")
```

## 6. Challenge Round
- Prove smooth pasting necessary (hint: else arbitrage via replication)
- Why does S*(t) increase toward K for puts as T increases?
- Show perpetual call boundary: S* = (β₊/(β₊-1))*K
- Explain monotonicity: S*(σ) increasing in volatility

## 7. Key References
- [Merton, "Optimal Stopping of Brownian Motion" (1973)](https://doi.org/10.1287/mnsc.20.7.1024) — Perpetual options
- [McKean, "Stochastic Integrals" (1969)](https://doi.org/10.1016/B978-0-44-415502-3.50022-X) — Free boundary methods
- [Wilmott, Paul. Derivatives. 2nd ed. (Chapter 6)](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119287742)

---
**Status:** Theoretical foundation for American valuation | **Complements:** Perpetual Options, Binomial Trees, Finite Difference Methods
