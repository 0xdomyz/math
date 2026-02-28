# Historical Volatility

## Concept Skeleton
**Definition:** Volatility estimated from past return data over a specified window  
**Purpose:** Baseline risk estimate for pricing, backtests, and model calibration  
**Prerequisites:** Returns, sample variance, time scaling

## Comparative Framing
| Variant | Close-to-Close | Parkinson | Garman-Klass |
|---|---|---|---|
| **Inputs** | Close prices | High/Low | Open/High/Low/Close |
| **Efficiency** | Low | Higher | Highest |
| **Bias** | None (iid) | Upward in trends | Lower bias |

## Examples + Counterexamples
**Simple Example:**  
60-day historical σ at 18% used as MC input for the next month.

**Failure Case:**  
Using 5-year window during crisis period underestimates current volatility.

**Edge Case:**  
Very short window (5 days) → noisy σ; high variance estimator.

## Layer Breakdown
```
Historical Volatility:
├─ Choose Window:
│   ├─ 20d (1 month), 60d (quarter), 252d (year)
│   └─ Trade-off: bias vs variance
├─ Compute Returns:
│   ├─ r_t = ln(P_t/P_{t-1})
│   └─ Or simple returns
├─ Estimate Variance:
│   └─ s^2 = (1/(n-1)) Σ (r_t - r̄)^2
├─ Annualize:
│   └─ σ = s × √(periods per year)
└─ Extensions:
    ├─ EWMA weights (decay factor λ)
    ├─ Robust estimators (trimmed)
    └─ High-low range estimators
```

**Interaction:** Pick window → compute returns → estimate σ → annualize

## Challenge Round
**Q1:** Why does window length matter?  
**A1:** Short windows adapt quickly but are noisy; long windows are stable but lag regime shifts.

**Q2:** When is EWMA preferable?  
**A2:** When volatility is time-varying; recent data should be weighted more.

**Q3:** Why use range-based estimators?  
**A3:** High-low range contains more information than close-to-close, improving efficiency.

**Q4:** Can historical σ be used directly for option pricing?  
**A4:** It ignores volatility risk premia; implied σ usually better for pricing.

## Key References
- [Historical volatility](https://en.wikipedia.org/wiki/Volatility_(finance)#Historical_volatility)  
- [Parkinson volatility](https://en.wikipedia.org/wiki/Parkinson_volatility)

---
**Status:** Foundational estimator | **Complements:** Implied volatility, GARCH
