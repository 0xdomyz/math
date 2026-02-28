# Early Exercise Feature

## Concept Skeleton
**Definition:** American options' right to exercise at any time before or at expiration, versus European options' fixed expiration-only exercise  
**Purpose:** Captures flexibility value in real financial markets; explains American-European price premium; determines optimal stopping problem  
**Prerequisites:** Option pricing basics, martingale theory, optimal stopping, dynamic programming, American option valuation

## Comparative Framing
| Aspect | American | European | Impact |
|--------|----------|----------|--------|
| **Exercise Timing** | Any time T ≤ t ≤ T_exp | Only at T_exp | Higher flexibility → higher value |
| **Optimal Exercise** | May exercise early | Never optimal early | Depends on dividends, rate paths |
| **Call (non-div)** | Never early | Same | Price equality |
| **Put (no div)** | Often early (ITM, high r) | Never early | American premium can be significant |
| **Callable Bonds** | Issuer exercises | Fixed maturity | Reduces bondholder value |
| **Swing Options** | Multiple exercise dates | One date | Energy derivatives |

## Examples + Counterexamples
**Simple Example:**  
Put option: S=$90, K=$100, r=5%, σ=20%, T=6 months. European value ~$10.25. American allows early exercise, capturing intrinsic $10 now + optionality if S rises → American premium ~$0.50-$1.

**Call on Dividend Stock:**  
Stock pays 5% annual dividend. Before ex-dividend date: ITM call holder faces dividend loss. Early exercise captures intrinsic, justifies American > European.

**Deep ITM Put:**  
S=$10, K=$100, r=10%, σ=5%, T=1 year. Intrinsic = $90. Holding European ties up capital earning only r; exercising gives $90 now, invested at r. American captures this → significant premium.

**When Not Early:**  
OTM options (calls with high S, puts with low S) rarely exercise early; European price ≈ American.

**Deep OTM:**  
S=$150, K=$100 (call), small σ, short T. American = European ≈ $0 (both worthless).

## Layer Breakdown
```
Early Exercise Feature:

├─ Exercise Right:
│  ├─ Fundamental difference:
│  │   American: max_{t ∈ [0,T]} E[e^(-rt) Payoff(t)]
│  │   European: E[e^(-rT) Payoff(T)]
│  ├─ Optimal stopping problem:
│  │   Find τ* maximizing expected discounted payoff
│  │   τ* = argmax_τ E[e^(-rτ) Payoff(τ)]
│  └─ Exercise boundary:
│      S(t) ≥ S*(t) ⟹ Exercise now (early stopping region)
│      S(t) < S*(t) ⟹ Hold (continuation region)
├─ Value Decomposition:
│  ├─ American option value:
│  │   V_A = max(Intrinsic, Option_Value_if_Hold)
│  ├─ European option value:
│  │   V_E = E^Q[e^(-rT) Payoff(T)]
│  ├─ American Premium:
│  │   Premium = V_A - V_E ≥ 0 (always)
│  │   V_A ≥ Intrinsic (American)
│  │   V_E can be < Intrinsic (European)
│  └─ Intuition: Extra optionality is valuable, free to hold
├─ Call Options (Non-Dividend):
│  ├─ Dividend-free assumption:
│  │   No benefit to capturing dividends early
│  │   Holding → investment return at r
│  ├─ Optimal policy:
│  │   Never exercise early! 
│  │   V_A(call) = V_E(call)
│  ├─ Intuition:
│  │   Intrinsic S-K < Option value (chance to rise)
│  │   Discounting reduces future payoff
│  │   But keeping option open better
│  └─ Exception: If r → ∞, premium→0, can become equal
├─ Call Options (With Dividends):
│  ├─ Continuous dividend yield q:
│  │   ex-dividend date approaching → S drops by dividend
│  ├─ Optimal exercise:
│  │   Just before ex-dividend: capture intrinsic
│  │   S - K vs (S - div) - K
│  ├─ Exercise boundary S*(t):
│  │   High q → early exercise more valuable
│  │   Low q → more like European
│  └─ Premium mechanism:
│      Capture dividend, avoid post-ex drop, arbitrage opportunity
├─ Put Options:
│  ├─ Intrinsic always available:
│  │   ITM put: Intrinsic = K - S > 0
│  ├─ Why exercise early:
│  │   High interest rate r: $K now > e^(-r*Δt)*K later
│  │   Stock volatility σ: Exercise locks in K-S immediately
│  │   Volatility risk: Avoid risk of S rising further
│  ├─ Exercise boundary S*_put(t):
│  │   S ≤ S* → Exercise now
│  │   S > S* → Hold (option value > intrinsic)
│  │   S* typically 70-90% of strike (depends on r, σ, T)
│  └─ Empirical: American puts often 5-20% premium over European
├─ Optimal Stopping Boundary:
│  ├─ Free boundary problem:
│  │   In continuation region: LV = rV (no drift, risk-neutral)
│  │   On boundary: V = Intrinsic (exercise)
│  ├─ Numerical solution:
│  │   Find S*(t) such that:
│  │   ├─ V(S*(t), t) = S*(t) - K (call) or K - S*(t) (put)
│  │   ├─ Smooth pasting (C² continuity): ∂V/∂S continuous
│  │   └─ Iterative refinement (binomial, finite difference)
│  ├─ Monotonicity:
│  │   Put boundary S*_put(t) increases with time (more intrinsic)
│  │   Call boundary S*_call(t) more complex (dividend vs rate)
│  └─ Limiting behavior:
│      S*(∞) → stable level (perpetual option)
│      S*(0) → boundary determined by payoff
├─ Time Value & Intrinsic:
│  ├─ European always:
│  │   Value ≥ Intrinsic or ≤ Intrinsic (depending on option type)
│  │   But discounting can make V_E < Intrinsic
│  ├─ American:
│  │   Always V_A ≥ Intrinsic (exercise if V < Intrinsic)
│  │   Time value = V_A - Intrinsic ≥ 0
│  ├─ Decay near expiry:
│  │   T→0: V_A → Intrinsic (option must be exercised or expire)
│  │   V_E → Intrinsic (for ITM); European premium vanishes
│  └─ Impact: American → more time value retained
└─ Extensions:
   ├─ Perpetual options (T→∞):
   │   S*(t)→S* (constant), can solve analytically
   ├─ Bermudan options:
   │   Exercise at discrete dates only
   │   Premium between European & American
   ├─ Path-dependent:
   │   Lookback: Exercise value depends on max/min S(t)
   │   Barrier: Exercise blocked if S crosses level
   └─ Multi-asset:
       Exchange options, basket options with early exercise
```

**Interaction:** Early exercise value = Difference in optimal stopping regimes; maximized when rates high/volatility low/dividends present.

## Challenge Round
- When is early exercise optimal for calls with dividends? (Just before ex-dividend)
- Why doesn't American call (non-div) command premium? (No benefit to early exercise)
- How does volatility affect early exercise propensity? (High vol delays exercise)
- What's the perpetual put exercise boundary? (Constant S* independent of time)

## Key References
- [Merton, "Theory of Rational Option Pricing" (1973)](https://doi.org/10.1016/B978-0-44-403970-2.50005-8) — American option framework
- [Hull, Options, Futures, and Derivatives (Chapter 8)](https://www.wiley.com/en-us/Options%2C+Futures%2C+and+Other+Derivatives%2C+11th+Edition-p-9781119259503) — Early exercise analysis
- [Binomial Trees for American Options](https://en.wikipedia.org/wiki/Binomial_options_pricing_model)

---
**Status:** Fundamental for American vs European distinction | **Complements:** Binomial Trees, Perpetual Options, Optimal Stopping Theory
