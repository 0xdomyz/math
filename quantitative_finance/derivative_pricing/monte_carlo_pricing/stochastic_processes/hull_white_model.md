# Hull–White Model

## Concept Skeleton
**Definition:** Extended Vasicek model with time-dependent drift to fit the initial yield curve  
**Purpose:** Interest rate modeling consistent with observed term structure  
**Prerequisites:** Vasicek model, yield curve bootstrapping

## Comparative Framing
| Model | Vasicek | Hull–White | CIR |
|---|---|---|---|
| **Curve Fit** | Limited | Exact | Limited |
| **Positivity** | No | No | Yes |
| **Volatility** | Constant | Constant | State-dependent |

## Examples + Counterexamples
**Simple Example:**  
Calibrate $\theta(t)$ to match today’s curve while preserving mean reversion.

**Failure Case:**  
Assuming constant drift fails to match observed yields.

**Edge Case:**  
If $\theta(t)$ is flat, model reduces to Vasicek.

## Layer Breakdown
```
Hull–White Model:
├─ SDE: dr = [θ(t) - a r] dt + σ dW
├─ Calibration:
│   ├─ Choose a, σ
│   └─ Solve θ(t) to fit P(0,T)
├─ Simulation:
│   └─ Exact discretization available (Gaussian)
└─ Pricing:
    ├─ Closed-form bond prices
    └─ MC for complex payoffs
```

**Interaction:** Fit θ(t) → simulate r_t → price interest rate derivatives

## Challenge Round
**Q1:** Why introduce $\theta(t)$?  
**A1:** To exactly match the initial yield curve (no-arbitrage fit).

**Q2:** Does Hull–White ensure positive rates?  
**A2:** No, it remains Gaussian and can go negative.

**Q3:** What makes it practical?  
**A3:** Analytic bond pricing and simple calibration to the curve.

**Q4:** How does $a$ affect dynamics?  
**A4:** Higher $a$ speeds mean reversion and dampens rate variance.

## Key References
- [Hull–White model](https://en.wikipedia.org/wiki/Hull%E2%80%93White_model)

---
**Status:** Curve-consistent short-rate model | **Complements:** Vasicek, CIR
