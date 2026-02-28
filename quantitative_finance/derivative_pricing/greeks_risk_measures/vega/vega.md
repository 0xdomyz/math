# Vega (ν)

## Concept Skeleton
**Definition:** First-order partial derivative of option price with respect to volatility; measures sensitivity to changes in price volatility (σ)  
**Purpose:** Quantify volatility exposure; hedge against implied volatility fluctuations; differentiate vol risk from directional delta risk  
**Prerequisites:** Volatility concepts, partial derivatives, implied volatility, stochastic volatility

## Comparative Framing
| Aspect | Vega | Delta | Gamma | Theta |
|--------|------|-------|-------|-------|
| **Measure** | ∂V/∂σ (volatility) | ∂V/∂S (spot) | ∂²V/∂S² | ∂V/∂T |
| **Call Sign** | Always > 0 | 0 to +1 | Always > 0 | Usually < 0 |
| **Put Sign** | Always > 0 | -1 to 0 | Always > 0 | Mixed |
| **Highest at** | ATM | N/A | ATM | Short expiry |
| **Risk Exposure** | Volatility (exogenous) | Directional (spot) | Nonlinearity | Decay |
| **Hedging Vehicle** | VIX, variance swaps | Underlying | Options | Calendar spreads |

## Examples + Counterexamples
**Simple Example:**  
ATM call: Vega ≈ 0.4; if implied vol ↑ 1%, call value ↑ $0.40; long vega benefit from vol spike

**Practical Case:**  
Strangle seller (sold OTM call + put): Short vega; collects premium if realized vol < implied vol; profits from vol contraction

**Counterintuitive Case:**  
Very short-dated OTM option: Vega ≈ 0 (little time for volatility benefit); theta dominates; vol changes have minimal impact

**Edge Case:**  
Variance swap vs option: Variance swap vega linear in realized vol; option vega nonlinear (changes with spot); different hedging mechanics

## Layer Breakdown
```
Vega Concept & Role:
├─ Mathematical Definition:
│   ├─ Vega: ν = ∂V/∂σ (notation: sometimes normalized per 1% vol change)
│   ├─ Per 1% vol (0.01): νₚₜ = ν / 100 (practical convention)
│   ├─ Interpretation: Option value changes by ν dollars per 1% volatility increase
│   └─ Scaling: Vega depends on option type, strike, moneyness, time
├─ Black-Scholes Formula:
│   ├─ Vega = S × N'(d1) × √T (same for calls and puts)
│   │   where N'(d1) = (1/√(2π)) × exp(-d1²/2)
│   ├─ Properties:
│   │   ├─ Always positive (both calls and puts benefit from vol ↑)
│   │   ├─ Maximum at ATM (S ≈ K)
│   │   ├─ Increases with time to expiry (√T term)
│   │   └─ Symmetric: calls and puts have identical vega
├─ Vega Dynamics:
│   ├─ Increases with:
│   │   ├─ Longer maturity (√T grows)
│   │   ├─ Spot near strike (ATM concentration)
│   │   └─ Lower risk-free rate (subtle)
│   ├─ Decreases with:
│   │   ├─ Spot away from strike (OTM/ITM decay)
│   │   ├─ Short maturity (expiration → vega → 0)
│   │   └─ Lower implied volatility (convexity effect in BS)
├─ Volatility Exposure Layers:
│   ├─ Implied Volatility: Market-quoted (tradeable via options)
│   ├─ Realized Volatility: Actual historical/forecast moves
│   ├─ Volatility Smile/Skew: IV varies by strike (affects delta-hedged vega)
│   ├─ Term Structure: IV varies by maturity (calendar vega)
│   └─ Cross-vega: Multi-asset vega (basket correlation risk)
├─ Hedging Vega Exposure:
│   ├─ Long vega: Hold long options (calls/puts); profit from vol ↑
│   ├─ Short vega: Sell options; profit if realized vol < implied vol
│   ├─ Vega hedge: Buy/sell options or variance swaps to offset exposure
│   ├─ Vega scaling: Divide total vega by single option vega to get hedge ratio
│   └─ Monitoring: Track portfolio vega by strike, maturity, asset
├─ Relationship to Other Greeks:
│   ├─ Vega-Delta coupling: Vega hedges often change delta; requires rebalancing
│   ├─ Vega-Theta tradeoff: Long vega → long gamma → positive theta decay cost
│   ├─ Vega skew: OTM puts have higher vega per vol unit due to smile
│   └─ Vega term: Calendar spread; long-dated options higher vega
```

**Interaction:** Vol uncertainty → vega exposure quantifies risk → hedge with vol derivatives → rebalance as spot/vol change

## Challenge Round
When is vega exposure misleading?
- Volatility smile: Vega aggregates across strikes; skew effects not captured (need spot-vega separately)
- Stochastic vol: Implied vol changes with spot (volvol); adds correlation risk beyond vega
- Term structure: Long-dated vega sensitive to different vol factors; not additive across maturities
- Realized vs implied: Vega hedges implied vol risk; doesn't protect if realized vol diverges from path-dependent realized vol
- Gamma-vega tradeoff: Long vol position = long gamma + long vega; requires monitoring both

## Key References
- [Hull - Options, Futures & Derivatives (Chapter 19)](https://www-2.rotman.utoronto.ca/~hull)
- [Taleb - Dynamic Hedging (Chapter 7)](https://www.paulwilmott.com)
- [Wilmott - Volatility Smile (Chapter 4)](https://www.paulwilmott.com)

---
**Status:** Primary volatility risk metric | **Complements:** Gamma, Theta, Volatility Surface, Options Greeks
