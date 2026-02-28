# Gamma (Γ)

## Concept Skeleton
**Definition:** Second-order partial derivative of option price with respect to underlying asset price; measures the rate of change of delta itself  
**Purpose:** Quantify delta convexity/nonlinearity; long gamma benefits from large moves (profitable rehedging); monitor delta hedging residual risk  
**Prerequisites:** Second derivatives, delta concepts, convexity, risk management

## Comparative Framing
| Aspect | Gamma | Delta | Vega | Theta |
|--------|-------|-------|------|-------|
| **Order** | 2nd derivative (∂²V/∂S²) | 1st derivative | 1st derivative | 1st derivative |
| **Interpretation** | Delta sensitivity to spot | Directional hedge | Volatility exposure | Time decay |
| **Call Sign** | Always > 0 | 0 to +1 | Always > 0 | Usually < 0 |
| **Put Sign** | Always > 0 | -1 to 0 | Always > 0 | Usually > 0 |
| **High Gamma Location** | At-the-money (ATM) | N/A | ATM | Short maturity |
| **Risk Type** | Convexity (rehedge cost) | Directional | Volatility | Time |

## Examples + Counterexamples
**Simple Example:**  
ATM call: Gamma highest (~0.02 per $1 spot move); delta changes by ~0.02 when spot moves $1

**Practical Case:**  
Short straddle (sold call + put): Negative gamma; loses money on large moves in either direction; "long volatility" risk

**Long Gamma Benefit:**  
Bought butterfly spread: Long gamma; benefits from realized volatility being higher than implied; profits from rehedging gains

**Counterintuitive Case:**  
Digital option at strike: Gamma → ∞ as expiry approaches; discontinuous delta; infinitely convex payoff

## Layer Breakdown
```
Gamma Concept & Dynamics:
├─ Mathematical Definition:
│   ├─ Gamma: Γ = ∂²V/∂S² = ∂Δ/∂S
│   ├─ Taylor expansion: ΔV ≈ Δ × ΔS + (Γ/2) × (ΔS)²
│   ├─ Interpretation: For $1 move, delta changes by ~Γ
│   └─ Quadratic term: Γ/2 × (ΔS)² = rehedging profit/loss
├─ Black-Scholes Formula:
│   ├─ Gamma = N'(d1) / (S × σ × √T)
│   │   where N'(d1) = (1/√(2π)) × exp(-d1²/2) [normal PDF]
│   ├─ Properties:
│   │   ├─ Same for calls and puts
│   │   ├─ Always positive (convex payoff)
│   │   ├─ Maximum at ATM (S ≈ K)
│   │   └─ Highest for short maturity (small √T in denominator)
├─ Dynamics Across Parameters:
│   ├─ Increases with:
│   │   ├─ Lower spot (moves away from ATM → ATM → increases Γ)
│   │   ├─ Shorter time to expiry (↓√T → ↑Γ)
│   │   └─ Lower volatility (↓σ → ↑N'(d1) concentration)
│   ├─ Decreases with:
│   │   ├─ Higher spot (moves away from ATM)
│   │   ├─ Longer time to expiry
│   │   └─ Higher volatility (payoff spread over wider range)
├─ Rehedging P&L:
│   ├─ Realized P&L from gamma: Σ (Γᵢ/2) × (ΔSᵢ)²
│   ├─ Long gamma → profit if realized vol > implied vol
│   ├─ Short gamma → loss if realized vol > implied vol
│   ├─ Breakeven realized vol: Where gamma P&L = theta decay
│   └─ Volatility smile impact: OTM gamma skew; asymmetric hedging
├─ Practical Risk Management:
│   ├─ Gamma limit: Max acceptable Γ for portfolio
│   ├─ Rebalancing frequency: High gamma → frequent hedging
│   ├─ Cost of hedging: Proportional to Γ × vol² × time
│   └─ Hedging optimization: Balance gamma risk vs. transaction costs
└─ Relationship to Other Greeks:
    ├─ Γ-Θ relationship: θ ≈ -rK×N(-d2) - (Γ/2)×S²×σ² [theta decay offsets gamma gains for ATM]
    ├─ Γ-Vega linkage: Short Γ ⇔ need to be short vega (convexity cost)
    └─ Γ-Delta cycle: Δ changes → Γ determines rate → rehedging needed
```

**Interaction:** Delta constantly changes (Gamma) → requires rehedging → costs money if realized vol < implied vol

## Challenge Round
When is gamma management critical?
- Short volatility portfolios: Negative gamma exposes to large moves; requires active hedging
- Near expiry: ATM gamma explodes; delta changes rapidly; rehedging becomes expensive
- Vega selling: Need offsetting long gamma to limit volatility losses on large moves
- Transaction costs: Frequent rehedging due to gamma costs money; optimal rehedge interval minimizes total cost
- Tail risk: Gamma doesn't protect against gaps; overnight moves bypass delta hedge

## Key References
- [Hull - Options, Futures & Derivatives (Chapter 19)](https://www-2.rotman.utoronto.ca/~hull)
- [Wilmott - Volatility Smile (Chapter 6)](https://www.paulwilmott.com)
- [Natenberg - Option Volatility & Pricing (Chapter 10)](https://www.amazon.com/Option-Volatility-Pricing-Advanced-Strategies/dp/1557784124)

---
**Status:** Secondary option Greek | **Complements:** Delta, Vega, Hedging Strategy, Greeks Portfolio
