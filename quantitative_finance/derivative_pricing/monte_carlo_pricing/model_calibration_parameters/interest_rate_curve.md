# Interest Rate Curve

## 1. Concept Skeleton
**Definition:** Term structure of risk-free rates or discount factors across maturities  
**Purpose:** Discount cash flows and compute forward prices in pricing models  
**Prerequisites:** Bond pricing, compounding, discount factors

## 2. Comparative Framing
| Representation | Zero Rates | Discount Factors | Forward Rates |
|---|---|---|---|
| **Unit** | % per annum | $P(0,T)$ | % per annum |
| **Use** | Reporting | Pricing | Forward pricing |
| **Conversion** | Via discounting | Base | Derived |

## 3. Examples + Counterexamples

**Simple Example:**  
Zero rate 5% at 1y → discount factor $P(0,1)=e^{-0.05}$.

**Failure Case:**  
Using flat rate curve ignores term premia → misprices long-dated options.

**Edge Case:**  
Negative rates: discount factors can exceed 1 for short maturities.

## 4. Layer Breakdown
```
Yield Curve Construction:
├─ Inputs:
│   ├─ Market instruments: deposits, futures, swaps
│   └─ Quoted rates by maturity
├─ Bootstrapping:
│   ├─ Solve for discount factors P(0,T)
│   └─ Ensure no-arbitrage consistency
├─ Conversions:
│   ├─ Zero rate: r(T) = -ln P(0,T)/T
│   ├─ Forward rate: f(t1,t2) = (ln P(0,t1) - ln P(0,t2))/(t2-t1)
│   └─ Simple vs continuous compounding
├─ Interpolation:
│   ├─ Linear on zero rates
│   └─ Spline on log discount factors
└─ Use:
    ├─ Discount option payoffs
    └─ Compute forward price F = S0 e^{(r-q)T}
```

**Interaction:** Bootstrap curve → compute discount factors → price derivatives

## 5. Mini-Project
Build a zero curve from synthetic swap rates:
```python
import numpy as np

# Synthetic maturities (years) and zero rates
T = np.array([0.5, 1, 2, 3, 5, 7, 10])
zero = np.array([0.02, 0.022, 0.025, 0.027, 0.03, 0.031, 0.032])

# Discount factors
P = np.exp(-zero * T)

# Forward rates (annualized, continuous)
for i in range(len(T)-1):
    f = (np.log(P[i]) - np.log(P[i+1])) / (T[i+1] - T[i])
    print(f"Forward {T[i]}-{T[i+1]}y: {f:.4f}")
```

## 6. Challenge Round

**Q1:** Why use discount factors instead of rates for pricing?  
**A1:** Discount factors multiply cashflows directly; they avoid compounding ambiguity and enforce no-arbitrage.

**Q2:** How do curve shifts affect options?  
**A2:** Higher rates increase call values via higher forward prices and lower discounting.

**Q3:** Why does interpolation matter?  
**A3:** Smoothness and monotonicity affect forward rates; poor interpolation introduces arbitrage.

**Q4:** What is OIS discounting?  
**A4:** Use overnight indexed swap rates for discounting collateralized cashflows.

## 7. Key References
- [Yield curve](https://en.wikipedia.org/wiki/Yield_curve)  
- [Bootstrapping (finance)](https://en.wikipedia.org/wiki/Bootstrapping_(finance))

---
**Status:** Core discounting input | **Complements:** Risk-neutral valuation
