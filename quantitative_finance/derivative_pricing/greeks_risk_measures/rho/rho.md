# Rho (ρ)

## Concept Skeleton
**Definition:** First-order partial derivative of option price with respect to risk-free interest rate; measures sensitivity to changes in discount rate  
**Purpose:** Quantify interest rate exposure; hedge against rate fluctuations; typically minor compared to other Greeks  
**Prerequisites:** Interest rates, discount factors, option pricing mechanics, rates impact

## Comparative Framing
| Aspect | Rho | Delta | Gamma | Vega | Theta |
|--------|-----|-------|-------|------|-------|
| **Measure** | ∂V/∂r (interest) | ∂V/∂S (spot) | ∂²V/∂S² | ∂V/∂σ (vol) | ∂V/∂T (time) |
| **Importance** | Low (rates sticky) | High | High | High | High |
| **Call Sign** | Always > 0 | 0 to +1 | Always > 0 | Always > 0 | Usually < 0 |
| **Put Sign** | Always < 0 | -1 to 0 | Always > 0 | Always > 0 | Usually > 0 |
| **Impact Magnitude** | ~1-5% per 1% rate change | 30-50% per spot move | Convexity cost | Variable | Daily cost |
| **Hedging Instrument** | Interest rate swaps | Underlying | Options | Vol derivatives | Calendar spreads |

## Examples + Counterexamples
**Simple Example:**  
10-year ATM call: Rho ≈ 5; if rates ↑ 1%, call value ↑ $5 (principal effect diluted by time value discount)

**Practical Case:**  
Long-dated options in rising rate environment: Rho gain helps offset other Greeks' losses; effect significant over years

**Counterintuitive Case:**  
Deep ITM call: Rho ≈ high (approaches strike × time), but primarily acts as synthetic forward; rate sensitivity similar to forward contract

**Edge Case:**  
Floating-rate note embedded option: Rho zero (rate-linked; natural hedge); vs. fixed-rate bond option (high rho)

## Layer Breakdown
```
Rho Concept & Dynamics:
├─ Mathematical Definition:
│   ├─ Rho: ρ = ∂V/∂r (interest rate derivative)
│   ├─ Interpretation: Option value changes by ρ dollars per 1% rate increase
│   ├─ Scaling: Rho typically quoted per 1% = per 0.01 change
│   └─ Normalization: Some sources report ρ/100 (per basis point)
├─ Black-Scholes Formula:
│   ├─ Call rho = K × T × e^{-rT} × N(d2)
│   ├─ Put rho = -K × T × e^{-rT} × N(-d2)
│   ├─ Properties:
│   │   ├─ Positive for calls (rate ↑ → spot forward ↑ → call value ↑)
│   │   ├─ Negative for puts (rate ↑ → put value ↓)
│   │   ├─ Increases with maturity (longer T → higher rate sensitivity)
│   │   └─ Approximately linear in rate (not convex like gamma)
├─ Rate Effect Decomposition:
│   ├─ Forward price effect: S × e^{rT} increases with r → call benefits
│   ├─ Discount factor effect: e^{-rT} decreases with r → reduces strike present value
│   │   For calls: Discount effect dominates → net positive rho
│   ├─ Spot price dynamic: In reality, rates affect spot via expectations
│   │   But BS model treats spot as exogenous → isolated rate effect only
│   └─ Duration analogy: Rho similar to bond duration sensitivity
├─ Rho Across Parameters:
│   ├─ Increases with:
│   │   ├─ Longer maturity (T larger)
│   │   ├─ Strike price (K larger → higher discount impact)
│   │   ├─ Moneyness ITM (N(d2) closer to 1.0)
│   │   └─ Lower starting rate (elasticity effect)
│   ├─ Decreases with:
│   │   ├─ Very low rates (already discounted)
│   │   └─ Deep OTM (N(d2) → 0)
├─ Practical Considerations:
│   ├─ Rates rarely move 1% daily → rho effect accumulates over months/years
│   ├─ Correlation: Spot-rate correlation (often negative in practice) affects net portfolio impact
│   ├─ Term structure: Different maturities react to different rate curves
│   ├─ Cross-asset: Rates affect volatility → indirect vega via stochastic vol
│   └─ Hedging: Use interest rate swaps or Treasury futures
└─ Relationship to Greeks:
    ├─ Rho-Delta: Both directional (not convex); rho linear, delta nonlinear
    ├─ Rho-Theta: Related via forward pricing; theta ∝ r (carry cost)
    ├─ Rho-Vega: Uncorrelated in Black-Scholes (independent); correlated in reality
    └─ Practical: Rho often dominated by other Greeks; lower priority in risk management
```

**Interaction:** Interest rate change → forward price adjusts → call value changes → rho quantifies sensitivity

## Challenge Round
When is rho analysis important?
- Long-dated options: Rho effect scales with T; significant for 5-10 year options
- Currency options: Rates affect both domestic and foreign; cross-currency rho
- Fixed-income derivatives: Rho dominates; rates are primary risk factor
- Carry strategies: Rho directly affects financing cost vs. option carry
- Emerging markets: High rate volatility → rho can be large; currency/rate correlation
- Low rate environment: Small basis point moves → rho leverage effect

## Key References
- [Hull - Options, Futures & Derivatives (Chapter 19)](https://www-2.rotman.utoronto.ca/~hull)
- [Natenberg - Option Volatility & Pricing (Chapter 14)](https://www.amazon.com/Option-Volatility-Pricing-Advanced-Strategies/dp/1557784124)
- [Wilmott - Quantitative Finance (Chapter 7)](https://www.paulwilmott.com)

---
**Status:** Minor interest rate Greek | **Complements:** Delta, Carry Cost, Rates Impact, Options Greeks
