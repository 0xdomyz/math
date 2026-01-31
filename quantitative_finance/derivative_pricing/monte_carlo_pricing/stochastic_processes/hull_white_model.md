# Hull–White Model

## 1. Concept Skeleton
**Definition:** Extended Vasicek model with time-dependent drift to fit the initial yield curve  
**Purpose:** Interest rate modeling consistent with observed term structure  
**Prerequisites:** Vasicek model, yield curve bootstrapping

## 2. Comparative Framing
| Model | Vasicek | Hull–White | CIR |
|---|---|---|---|
| **Curve Fit** | Limited | Exact | Limited |
| **Positivity** | No | No | Yes |
| **Volatility** | Constant | Constant | State-dependent |

## 3. Examples + Counterexamples

**Simple Example:**  
Calibrate $\theta(t)$ to match today’s curve while preserving mean reversion.

**Failure Case:**  
Assuming constant drift fails to match observed yields.

**Edge Case:**  
If $\theta(t)$ is flat, model reduces to Vasicek.

## 4. Layer Breakdown
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

## 5. Mini-Project
Simulate Hull–White short rate with time-dependent drift:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

r0, a, sigma = 0.02, 0.4, 0.01
T, n, paths = 5.0, 500, 5

dt = T/n
# Example theta(t) as a simple function (placeholder)

def theta(t):
    return 0.03 + 0.002*np.sin(2*np.pi*t)

r = np.zeros((paths, n+1))
r[:,0] = r0

for i in range(n):
    t = i*dt
    Z = np.random.randn(paths)
    r[:, i+1] = r[:, i] + (theta(t) - a*r[:, i]) * dt + sigma*np.sqrt(dt)*Z

for p in range(paths):
    plt.plot(np.linspace(0,T,n+1), r[p])

plt.title('Hull–White Short Rate')
plt.grid(alpha=0.3)
plt.show()
```

## 6. Challenge Round

**Q1:** Why introduce $\theta(t)$?  
**A1:** To exactly match the initial yield curve (no-arbitrage fit).

**Q2:** Does Hull–White ensure positive rates?  
**A2:** No, it remains Gaussian and can go negative.

**Q3:** What makes it practical?  
**A3:** Analytic bond pricing and simple calibration to the curve.

**Q4:** How does $a$ affect dynamics?  
**A4:** Higher $a$ speeds mean reversion and dampens rate variance.

## 7. Key References
- [Hull–White model](https://en.wikipedia.org/wiki/Hull%E2%80%93White_model)

---
**Status:** Curve-consistent short-rate model | **Complements:** Vasicek, CIR
