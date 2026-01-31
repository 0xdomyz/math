# Stochastic Interest Rates

## 1. Concept Skeleton
**Definition:** Models of interest rate evolution using random processes (Vasicek, CIR, Hull-White)  
**Purpose:** Price interest-sensitive products and manage ALM under uncertainty  
**Prerequisites:** Stochastic calculus, mean reversion, term structure models

## 2. Comparative Framing
| Model | Vasicek | CIR | Hull-White |
|-------|---------|-----|------------|
| **Mean Reversion** | Yes | Yes | Yes (time-varying) |
| **Negative Rates** | Possible | No | Possible |
| **Calibration** | Simple | Moderate | Flexible |

## 3. Examples + Counterexamples

**Simple Example:**  
Vasicek: $dr_t = \kappa(\theta - r_t)dt + \sigma dW_t$

**Failure Case:**  
Using constant volatility model during regime shifts

**Edge Case:**  
Negative interest rates in Vasicek require floor adjustments

## 4. Layer Breakdown
```
Stochastic Rate Modeling:
├─ Choose model (Vasicek, CIR, Hull-White)
├─ Calibrate to current term structure
├─ Simulate rate paths
├─ Value liabilities under scenarios
└─ Aggregate and report
```

**Interaction:** Calibrate → simulate → value → aggregate

## 5. Mini-Project
Simulate Vasicek rate path:
```python
import numpy as np

r0 = 0.03
kappa = 0.5
theta = 0.04
sigma = 0.01
T = 10
n = 252

dt = T / n
r = np.zeros(n)
r[0] = r0

for t in range(1, n):
    dW = np.random.normal(0, np.sqrt(dt))
    r[t] = r[t-1] + kappa * (theta - r[t-1]) * dt + sigma * dW

print("Final rate:", r[-1])
```

## 6. Challenge Round
Common pitfalls:
- Ignoring negative rate possibility in Vasicek
- Poor calibration leading to mispriced guarantees
- Not stress-testing extreme paths

## 7. Key References
- [Vasicek Model (Wikipedia)](https://en.wikipedia.org/wiki/Vasicek_model)
- [CIR Model (Wikipedia)](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model)
- [Hull-White Model (Wikipedia)](https://en.wikipedia.org/wiki/Hull%E2%80%93White_model)

---
**Status:** ALM core tool | **Complements:** Monte Carlo, Scenario Analysis
