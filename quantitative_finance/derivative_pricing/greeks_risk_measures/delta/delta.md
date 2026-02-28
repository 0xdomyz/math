# Delta (Δ)

## Concept Skeleton
**Definition:** First-order partial derivative of option price with respect to underlying asset price; measures rate of change in option value per unit move in spot price  
**Purpose:** Quantify directional exposure; primary hedge ratio for delta-neutral portfolios; probability-like interpretation (ITM probability for calls)  
**Prerequisites:** Partial derivatives, option pricing, linear approximation, hedging concepts

## Comparative Framing
| Greek | Delta | Gamma | Vega | Theta |
|-------|-------|-------|------|-------|
| **Measure** | ∂V/∂S (linear) | ∂²V/∂S² (curvature) | ∂V/∂σ (volatility) | ∂V/∂T (time) |
| **Dimension** | Ratio; unitless | Inverse price | Value per 1% vol | Value per 1 day |
| **Call Range** | 0 to +1 | Always positive | Always positive | Usually negative |
| **Put Range** | -1 to 0 | Always positive | Always positive | Mixed sign |
| **Hedging Role** | Primary position size | Convexity adjustment | Volatility risk | Decay monitoring |

## Examples + Counterexamples
**Simple Example:**  
ATM call: Δ ≈ 0.5 → per $1 spot increase, option value ↑ $0.50; hedge with 0.5 shares short

**Practical Case:**  
OTM call (Δ ≈ 0.1): Price sensitivity low; delta 100 contracts ≈ holding 10 shares equivalent; used for leveraged bets

**Counterintuitive Case:**  
Deep ITM call: Δ ≈ 1.0 (moves like stock); Δ ≈ -1.0 for deep ITM put; behaves as "synthetic stock"

**Edge Case:**  
Digital/binary option: Δ = 0 everywhere except strike (discontinuity) → undefined Δ at expiry threshold

## Layer Breakdown
```
Delta Concept & Calculation:
├─ Theoretical Foundation:
│   ├─ Definition: Δ = ∂V/∂S
│   ├─ Taylor expansion: ΔV ≈ Δ × ΔS (linear approximation)
│   ├─ Probability link: Δ_call ≈ P(S_T > K) under risk-neutral measure
│   └─ Intuition: Hedge ratio; position delta
├─ Black-Scholes Formula:
│   ├─ Call: Δ_c = N(d1) where d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
│   ├─ Put: Δ_p = N(d1) - 1 = -N(-d1)
│   ├─ Properties:
│   │   ├─ N(d1) ∈ (0, 1) for calls
│   │   ├─ Monotonically increasing in S
│   │   └─ Δ → 0 as S → 0; Δ → 1 as S → ∞
├─ Numerical Computation:
│   ├─ Finite difference: Δ ≈ [V(S + ε) - V(S - ε)] / (2ε)
│   ├─ Central difference preferred (O(ε²) accuracy)
│   └─ Pathwise derivative: For MC, dV/dS via pathwise sensitivities
├─ Interpretation:
│   ├─ Hedge ratio: To delta-hedge, short Δ shares per long option
│   ├─ Position delta: Σ(Δᵢ × Qᵢ) = total market delta exposure
│   ├─ Rebalancing: Δ changes; requires periodic adjustment
│   └─ P&L from delta: P&L ≈ Δ × ΔS (if other Greeks stable)
└─ Dependencies:
    ├─ Increases with:
    │   ├─ Underlying price (for calls)
    │   ├─ Time to expiry (ATM calls increase to 1.0)
    │   └─ Volatility (affects d1, subtle effect for calls)
    ├─ Decreases with:
    │   ├─ Strike price (for calls)
    │   └─ Risk-free rate (for calls, minor)
```

**Interaction:** Spot move → Δ quantifies price response → delta hedge balances → P&L isolated from spot moves

## Challenge Round
When is delta hedging imperfect?
- Discrete rebalancing: Can't hedge continuously; residual P&L from gamma effects (Gamma × (ΔS)²/2)
- Transaction costs: Frequent rehedging expensive; optimal rehedge frequency depends on gamma, vol, transaction cost
- Path dependency: Barrier options have discontinuous delta at strike; gaps in hedge coverage
- Jump risk: Gaps in spot price overnight; delta hedge can't respond instantly
- Model risk: Delta assumes BS model; real volatility changes, correlations break

## Key References
- [Black-Scholes Formula](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Hull - Options, Futures & Derivatives (Chapter 19)](https://www-2.rotman.utoronto.ca/~hull)
- [Wilmott - Quantitative Finance (Volume 1, Chapter 7)](https://www.paulwilmott.com)

---
**Status:** Primary option Greek | **Complements:** Gamma, Hedging Strategy, Options Greeks Overview
