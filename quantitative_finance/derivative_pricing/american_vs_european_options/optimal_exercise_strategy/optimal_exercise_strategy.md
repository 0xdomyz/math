# Optimal Exercise Strategy

## Concept Skeleton
**Definition:** Mathematical framework determining the theoretically optimal time τ* to exercise an American option that maximizes expected discounted payoff  
**Purpose:** Characterizes free boundary problem; generates exercise region S*(t); enables valuation via backward induction; underpins numerical methods  
**Prerequisites:** Optimal stopping theory, dynamic programming, martingale theory, risk-neutral valuation, boundary value problems

## Comparative Framing
| Strategy | Greedy | Myopic | Optimal | Reality |
|----------|--------|--------|---------|---------|
| **Decision** | Exercise if ITM | Consider immediate value | Maximize E[discounted payoff] | Exercise heuristically |
| **Horizon** | Now only | Next period | Full remaining life | Truncated horizon |
| **Accuracy** | Poor (leaves value) | Better | Exact (theoretical) | Implementation noise |
| **Compute** | Instant | Moderate | Intensive (backward) | Approx via grid |
| **When Best** | High dividends | Short T | Perpetual/long T | Liquid markets |

## Examples + Counterexamples
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

## Layer Breakdown
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

## Challenge Round
- Prove smooth pasting necessary (hint: else arbitrage via replication)
- Why does S*(t) increase toward K for puts as T increases?
- Show perpetual call boundary: S* = (β₊/(β₊-1))*K
- Explain monotonicity: S*(σ) increasing in volatility

## Key References
- [Merton, "Optimal Stopping of Brownian Motion" (1973)](https://doi.org/10.1287/mnsc.20.7.1024) — Perpetual options
- [McKean, "Stochastic Integrals" (1969)](https://doi.org/10.1016/B978-0-44-415502-3.50022-X) — Free boundary methods
- [Wilmott, Paul. Derivatives. 2nd ed. (Chapter 6)](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119287742)

---
**Status:** Theoretical foundation for American valuation | **Complements:** Perpetual Options, Binomial Trees, Finite Difference Methods
