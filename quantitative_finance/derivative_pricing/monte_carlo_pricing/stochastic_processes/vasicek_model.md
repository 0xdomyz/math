# Vasicek Model

## 1. Concept Skeleton
**Definition:** Short-rate model using OU dynamics: $dr_t = a(b-r_t)dt + \sigma dW_t$  
**Purpose:** Price bonds and interest rate derivatives with mean-reverting rates  
**Prerequisites:** OU process, bond pricing

## 2. Comparative Framing
| Model | Vasicek | CIR | Hull–White |
|---|---|---|---|
| **Mean Reversion** | Yes | Yes | Yes |
| **Positivity** | No | Yes | No (but calibrated) |
| **Curve Fit** | Limited | Limited | Exact via time-dependent drift |

## 3. Examples + Counterexamples

**Simple Example:**  
Rate reverts to 3% with speed 0.5 and volatility 1%.

**Failure Case:**  
Negative rates possible; may be unrealistic for some markets.

**Edge Case:**  
$a \to 0$ reduces to Brownian motion with drift.

## 4. Layer Breakdown
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

## 5. Mini-Project
Simulate short-rate paths under Vasicek:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

r0, a, b, sigma = 0.02, 0.6, 0.03, 0.01
T, n, paths = 5.0, 500, 5

dt = T/n
r = np.zeros((paths, n+1))
r[:,0] = r0

for i in range(n):
    Z = np.random.randn(paths)
    r[:, i+1] = r[:, i] + a*(b - r[:, i])*dt + sigma*np.sqrt(dt)*Z

for p in range(paths):
    plt.plot(np.linspace(0,T,n+1), r[p])

plt.axhline(b, color='red', linestyle='--', label='Long-run mean')
plt.legend()
plt.title('Vasicek Short Rate')
plt.grid(alpha=0.3)
plt.show()
```

## 6. Challenge Round

**Q1:** Why does Vasicek allow negative rates?  
**A1:** The Gaussian distribution has support on the entire real line.

**Q2:** How is bond pricing analytic?  
**A2:** The model is affine; yields are linear in $r_t$.

**Q3:** When is Vasicek acceptable?  
**A3:** For moderate volatility or environments tolerating negative rates.

**Q4:** Why calibrate to curve?  
**A4:** A fixed drift cannot fit today’s full term structure.

## 7. Key References
- [Vasicek model](https://en.wikipedia.org/wiki/Vasicek_model)

---
**Status:** Classic short-rate model | **Complements:** CIR, Hull–White
