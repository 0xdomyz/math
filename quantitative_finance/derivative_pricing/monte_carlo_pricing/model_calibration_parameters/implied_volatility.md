# Implied Volatility

## Concept Skeleton
**Definition:** Volatility input that makes a pricing model (e.g., Black-Scholes) match observed option prices  
**Purpose:** Market-consensus risk estimate used for pricing, hedging, and smile surfaces  
**Prerequisites:** Option pricing model, root-finding, no-arbitrage

## Comparative Framing
| Aspect | Historical σ | Implied σ | Realized σ |
|---|---|---|---|
| **Source** | Past returns | Option prices | Future realized |
| **Forward-looking** | No | Yes | Yes (ex post) |
| **Use** | Risk | Pricing | Validation |

## Examples + Counterexamples
**Simple Example:**  
Call price $C=5.20$ at $S_0=100, K=100, T=1$ → solve $\sigma_{imp}=0.22$.

**Failure Case:**  
Deep ITM/OTM options with low liquidity → implied σ noisy or arbitrary.

**Edge Case:**  
Inconsistent quotes yield arbitrage; implied vol not unique without smoothing.

## Layer Breakdown
```
Implied Volatility Extraction:
├─ Inputs:
│   ├─ Observed option price C_mkt
│   ├─ Model pricing function C(σ)
│   └─ Bounds: σ ∈ [1e-4, 5.0]
├─ Root-Finding:
│   ├─ Solve f(σ) = C(σ) - C_mkt = 0
│   ├─ Methods: Newton-Raphson, bisection, Brent
│   └─ Need vega for Newton step
├─ Convergence:
│   ├─ Newton: σ_{n+1} = σ_n - f(σ)/Vega(σ)
│   └─ Stop when |f(σ)| < tolerance
└─ Surface:
    ├─ Repeat over strikes/maturities
    ├─ Interpolate for smile/skew
    └─ Check no-arbitrage constraints
```

**Interaction:** Price model → root-find σ → build surface → validate arbitrage

## Challenge Round
**Q1:** Why does Newton sometimes fail?  
**A1:** Vega can be tiny (deep ITM/OTM), causing large steps; bisection is more stable.

**Q2:** How to ensure no-arbitrage in vol surface?  
**A2:** Enforce monotonicity in strike, convexity of prices, and calendar spread constraints.

**Q3:** Why is implied vol forward-looking?  
**A3:** It reflects market expectations and risk premia embedded in option prices.

**Q4:** How do dividends affect implied vol?  
**A4:** They change forward price; incorrect dividend assumptions bias implied vol.

## Key References
- [Implied volatility](https://en.wikipedia.org/wiki/Implied_volatility)  
- [Black–Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)

---
**Status:** Primary market calibration target | **Complements:** Volatility smile, surface
