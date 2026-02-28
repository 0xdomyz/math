# Volatility Estimation

## Concept Skeleton
**Definition:** Estimating the dispersion of asset returns, typically as annualized standard deviation  
**Purpose:** Input for option pricing, risk models, and Monte Carlo simulation  
**Prerequisites:** Returns, variance, sampling frequency, annualization

## Comparative Framing
| Method | Historical Volatility | Implied Volatility | Model-Based (GARCH) |
|---|---|---|---|
| **Data** | Past returns | Option prices | Past returns + dynamics |
| **Output** | Realized σ | Market-consensus σ | Conditional σ_t |
| **Use** | Backtesting | Pricing/hedging | Forecasting |

## Examples + Counterexamples
**Simple Example:**  
Daily returns with std 1% → annualized σ ≈ 1% × √252 = 15.87%.

**Failure Case:**  
Using historical σ during regime shift underestimates risk → MC prices too low.

**Edge Case:**  
Illiquid asset with sparse data → volatility estimates unstable; use shrinkage or proxies.

## Layer Breakdown
```
Volatility Estimation Workflow:
├─ Input Data:
│   ├─ Price series P_t
│   ├─ Returns r_t = ln(P_t/P_{t-1})
│   └─ Sampling frequency (daily, weekly)
├─ Base Estimator:
│   ├─ Sample variance s^2 = (1/(n-1)) Σ (r_t - r̄)^2
│   └─ Sample volatility s = √s^2
├─ Annualization:
│   └─ σ_annual = s × √(periods per year)
├─ Adjustments:
│   ├─ De-meaning or drift removal
│   ├─ Outlier handling / winsorization
│   └─ Volatility scaling across horizons
└─ Validation:
    ├─ Compare to implied volatility
    ├─ Stability across windows
    └─ Sensitivity to sampling frequency
```

**Interaction:** Choose estimator → compute σ → annualize → validate vs market

## Challenge Round
**Q1:** Why annualize by √252?  
**A1:** Under iid returns, variance scales linearly with time; volatility scales with square root of time.

**Q2:** Why can sampling frequency bias σ?  
**A2:** Microstructure noise inflates high-frequency volatility; stale prices dampen it.

**Q3:** When should you prefer implied volatility?  
**A3:** For pricing/hedging options; it reflects market forward-looking expectations and risk premia.

**Q4:** Why is volatility not constant in practice?  
**A4:** Volatility clusters; regimes change; shocks create heteroskedasticity.

## Key References
- [Volatility (finance)](https://en.wikipedia.org/wiki/Volatility_(finance))  
- [Historical volatility](https://en.wikipedia.org/wiki/Volatility_(finance)#Historical_volatility)

---
**Status:** Core calibration input | **Complements:** Historical & implied volatility
