# Vasicek Model

## Concept Skeleton
**Definition:** Short-rate model using OU dynamics: $dr_t = a(b-r_t)dt + \sigma dW_t$  
**Purpose:** Price bonds and interest rate derivatives with mean-reverting rates  
**Prerequisites:** OU process, bond pricing

## Comparative Framing
| Model | Vasicek | CIR | Hull–White |
|---|---|---|---|
| **Mean Reversion** | Yes | Yes | Yes |
| **Positivity** | No | Yes | No (but calibrated) |
| **Curve Fit** | Limited | Limited | Exact via time-dependent drift |

## Examples + Counterexamples
**Simple Example:**  
Rate reverts to 3% with speed 0.5 and volatility 1%.

**Failure Case:**  
Negative rates possible; may be unrealistic for some markets.

**Edge Case:**  
$a \to 0$ reduces to Brownian motion with drift.

## Layer Breakdown
```
Vasicek Dynamics:
├─ SDE: dr = a(b - r)dt + σ dW
├─ Solution:
│   └─ r_t = b + (r_0-b)e^{-at} + σ∫ e^{-a(t-s)} dW_s
├─ Bond Price:
│   └─ P(0,T) = A(T) e^{-B(T) r_0}
└─ Calibration:
    ├─ a, b, σ from historical rates
    └─ Limited fit to yield curve
```

**Interaction:** Simulate r_t → discount cashflows or price bonds

## Challenge Round
**Q1:** Why does Vasicek allow negative rates?  
**A1:** The Gaussian distribution has support on the entire real line.

**Q2:** How is bond pricing analytic?  
**A2:** The model is affine; yields are linear in $r_t$.

**Q3:** When is Vasicek acceptable?  
**A3:** For moderate volatility or environments tolerating negative rates.

**Q4:** Why calibrate to curve?  
**A4:** A fixed drift cannot fit today’s full term structure.

## Key References
- [Vasicek model](https://en.wikipedia.org/wiki/Vasicek_model)

---
**Status:** Classic short-rate model | **Complements:** CIR, Hull–White
