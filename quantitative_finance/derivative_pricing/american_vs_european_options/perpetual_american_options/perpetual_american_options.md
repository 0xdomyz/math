# Perpetual American Options

## Concept Skeleton
**Definition:** American options with no expiration date (T=∞); value determined by optimal stopping with infinite horizon  
**Purpose:** Analytical tractability; illuminates structure of American premium; benchmark for very long-dated contracts; theoretical foundation  
**Prerequisites:** Perpetual processes, optimal stopping, differential equations, martingale methods, risk-neutral valuation

## Comparative Framing
| Aspect | Perpetual | Finite-Life | European | Status |
|--------|-----------|-------------|----------|--------|
| **Expiry** | Never | T fixed | T fixed | Definition |
| **S*(t)** | Constant | Time-dependent | N/A | Key property |
| **Solution** | Closed-form ODE | PDE (numerical) | Closed-form (BS) | Tractability |
| **Premium** | Explicit formula | Requires binomial | N/A | Analysis |
| **Real-world** | Long-duration assets | Equity/futures | Index options | Prevalence |
| **Volatility Impact** | Increases S* (put) | Complex | Straightforward | Intuition |
| **Interest Rate Impact** | Decreases S* (put) | Complex | Decreases value | Tractability |

## Examples + Counterexamples
**Simple Example:**  
Perpetual put K=$100, r=5%, σ=20%. Solution: S* ≈ $59.60. Optimal policy: Hold until S≤$59.60, then exercise. This stationary boundary never changes.

**Analytical Appeal:**  
No time decay (∂V/∂t=0). Only balance: drift vs spread. Closed-form: V(S) = A·S^β for S>S*. Direct solution.

**Perpetual Call with Dividend:**  
K=$100, q=8% (continuous dividend), r=5%, σ=20%. S* ≈ $125.4. Always hold unless S drops (but with high q, long-run drift negative).

**Comparison - Deep ITM Put:**  
Perpetual (S=$20, K=$100): Early exercise captures $80. Value functions peak at S=S*, then slope = -1 (intrinsic).

**Reality Check:**  
Equity warrants (long-dated call options, ~5-10 years) ≈ perpetual locally. Perpetual bond (callable consol) exactly perpetual. Treasury TIPS: finite, but very long T → boundary ≈ perpetual.

## Layer Breakdown
```
Perpetual American Options:

├─ Simplifications from Perpetual Nature:
│  ├─ No expiration: T → ∞
│  ├─ Time-stationary: ∂V/∂t = 0
│  ├─ Value function depends only on S:
│  │   V = V(S) independent of t
│  └─ Exercise boundary constant:
│      S* = fixed level (not time-varying)
├─ ODE Framework (Perpetual PDE):
│  ├─ General perpetual: ∂V/∂t = 0 in hold region
│  ├─ Results in ODE:
│  │   0.5 σ² S² V''(S) + (r-q)S V'(S) - rV(S) = 0
│  │   (for call with dividend yield q)
│  ├─ Power law solution:
│  │   V(S) = A S^β (for S > S*)
│  │   where β satisfies characteristic equation
│  ├─ Characteristic equation:
│  │   0.5 σ²β(β-1) + (r-q)β - r = 0
│  │   ⟹ β² + (2(r-q)/σ² - 1)β - 2r/σ² = 0
│  └─ Solutions:
│      β₊ = positive root (> 1, use for far-field)
│      β₋ = negative root (< 0, discard, diverges)
├─ Perpetual Put (With Dividends q):
│  ├─ Hold region (S > S*):
│  │   V(S) = A(S/S*)^β₊ * (K - S*)
│  ├─ Exercise region (S ≤ S*):
│  │   V(S) = K - S (intrinsic)
│  ├─ Value matching (continuity):
│  │   A(S*/S*)^β₊(K-S*) = K - S*
│  │   ⟹ A = 1
│  ├─ Smooth pasting (C¹ continuity):
│  │   V'(S*) = -1
│  │   β₊ A (S*/S*)^(β₊-1) (K-S*) + A(S*/S*)^β₊ = -1
│  ├─ Combining conditions:
│  │   (K - S*) = -1/β₊ * (K - S*)
│  │   ⟹ β₊ S* = (β₊ - 1)(K - S*)
│  │   ⟹ β₊ S* + (β₊-1)S* = (β₊-1)K
│  │   ⟹ S* = K(β₊-1)/(2β₊-1)
│  │   OR: S* = K * (β₊/(β₊+1))  [after simplification]
│  └─ Key insight: Unique S* determined by r, σ alone
├─ Perpetual Call (Non-Dividend):
│  ├─ Hold region (S > S*):
│  │   V(S) = B(S - K e^(-rτ)) for τ=residual life
│  │   Actually: Not straightforward for calls
│  ├─ For non-dividend calls:
│  │   β = (-r + √(r² + 2rσ²))/σ²
│  │   S* = K * (β/(β-1))  
│  ├─ Behavior: S* > K (always exercise in future)
│  └─ Never exercise now (time value > intrinsic)
├─ Perpetual Call (With Dividend q > r):
│  ├─ If q > r (high dividend, low rate):
│  │   Exercise boundary exists at S* < ∞
│  ├─ Formula: S* = K(β₊/(β₊-1)) where β₊ from dividend ODE
│  └─ Exercise when dividend premium captured
├─ Parameter Sensitivity (Perpetual Put):
│  ├─ Volatility σ:
│  │   ∂S*/∂σ > 0 (higher vol → higher boundary)
│  │   Intuition: Option value from moves increases
│  ├─ Interest rate r:
│  │   ∂S*/∂r < 0 (higher r → lower boundary)
│  │   Intuition: Present value of intrinsic ↑, holding ↓
│  ├─ Dividend yield q:
│  │   ∂S*/∂q varies (affects drift term)
│  ├─ Strike K:
│  │   S* scales linearly with K (homogeneity)
│  └─ Numeric: Doubling σ can move S* 10-15%
├─ Comparison to Finite-Life:
│  ├─ Perpetual S*(∞) is limit of finite T:
│  │   As T → ∞, S*(T) → S*_perpetual from above
│  ├─ For T<∞:
│  │   S*(t) > S*(∞) (higher boundary, more waiting value)
│  │   S*(T) → Strike K as t → T (intrinsic boundary)
│  ├─ Premium:
│  │   Perpetual premiums are upper bounds for finite T
│  └─ Approximation: Long T (~10yr) ≈ perpetual locally
├─ Economic Interpretation:
│  ├─ Stationary distribution:
│  │   S follows driftless process (risk-neutral)
│  │   No systematic upward/downward drift
│  ├─ Holding strategy:
│  │   Wait for S to cross S* boundary
│  │   Then exercise and pocket immediate gain
│  ├─ Value vs cost:
│  │   Benefit: Intrinsic gain (K-S*)
│  │   Cost: Foregone continued optionality (never recovers)
│  │   Equate at boundary (smooth pasting)
│  └─ Long-run: Perpetual puts exercised rarely (first passage)
└─ Extensions:
   ├─ Multiple assets (basket):
     Boundary becomes surface S₁* × S₂* × ...
   ├─ Jump-diffusion:
     Poisson arrivals change perpetual ODE (more terms)
   ├─ Stochastic interest rates:
     State-space expanded, now depends on r(t)
   └─ Regime switching:
       Perpetual with multi-state Markov chain
```

**Interaction:** Perpetual option = infinite-horizon optimal stopping; boundary determined purely by balance between immediate payoff and option value.

## Challenge Round
- Derive perpetual put boundary from value matching + smooth pasting
- Prove β+ > 1 for perpetual options (hint: quadratic formula)
- Compare perpetual call (div) vs put: which has higher boundary?
- Show S* → K as σ → ∞ (volatility makes exercise unattractive)

## Key References
- [Samuelson & McKean, "Rational Warrant Pricing" (1965)](https://www.jstor.org/stable/2977340) — Perpetual option foundation
- [Wilmott, Derivatives (Chapter 5)](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119287742) — Perpetual analysis
- [McDonald & Siegel, "Investment and Valuation of Real Options" (1985)](https://doi.org/10.1111/j.1540-6261.1985.tb02541.x)

---
**Status:** Theoretical foundation | **Complements:** Optimal Exercise Strategy, Finite-Life American Options
