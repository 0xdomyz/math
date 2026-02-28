# Smile & Skew

## Concept Skeleton
**Definition:** Pattern where implied volatility varies with strike and maturity (smile or skew)  
**Purpose:** Capture market pricing of tail risk and asymmetry; build volatility surface  
**Prerequisites:** Implied volatility, option moneyness, arbitrage constraints

## Comparative Framing
| Pattern | Smile | Skew | Flat Surface |
|---|---|---|---|
| **Shape** | U-shaped | Downward sloping | Constant |
| **Markets** | FX | Equities | Idealized BS |
| **Cause** | Symmetric tail risk | Crash risk | Constant σ assumption |

## Examples + Counterexamples
**Simple Example:**  
Equity index options: deep OTM puts have high implied σ → negative skew.

**Failure Case:**  
Assuming flat σ in skewed market → mispriced OTM puts and poor hedging.

**Edge Case:**  
Near-zero rates and low vol: smile flattens; surface almost constant.

## Layer Breakdown
```
Smile/Skew Construction:
├─ Inputs:
│   ├─ Option prices by strike and maturity
│   └─ Forward price F and discount factor
├─ Convert to Implied Vols:
│   └─ σ_imp(K,T) from each quote
├─ Fit Surface:
│   ├─ Parametric: SVI, SABR
│   ├─ Non-parametric: splines
│   └─ Enforce no-arbitrage constraints
├─ Diagnostics:
│   ├─ Risk reversal (skew measure)
│   └─ Butterfly (smile curvature)
└─ Use:
    ├─ Price exotic options
    └─ Hedge with local/volatility models
```

**Interaction:** Market quotes → implied vols → fit smooth surface → validate

## Challenge Round
**Q1:** Why does equity skew slope downward?  
**A1:** Markets price crash risk; demand for OTM puts increases implied volatility on the downside.

**Q2:** What arbitrage constraints must a surface satisfy?  
**A2:** Monotonicity and convexity of option prices in strike, and calendar spread constraints in maturity.

**Q3:** Why is local volatility derived from the smile?  
**A3:** The Dupire formula maps implied vol surface to a local volatility function consistent with observed prices.

**Q4:** How does skew affect delta-hedging?  
**A4:** Delta depends on implied vol; skew changes delta sensitivity (vanna/vomma), affecting hedging P&L.

## Key References
- [Volatility smile](https://en.wikipedia.org/wiki/Volatility_smile)  
- [SVI parameterization](https://en.wikipedia.org/wiki/Stochastic_volatility-inspired)

---
**Status:** Market-implied surface feature | **Complements:** Implied volatility, local vol
