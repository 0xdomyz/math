# Brownian Motion (Wiener Process)

## 1. Concept Skeleton
**Definition:** Continuous-time process with independent increments, $W_0=0$, and $W_t-W_s \sim \mathcal{N}(0,t-s)$  
**Purpose:** Core noise source in SDEs and Monte Carlo pricing  
**Prerequisites:** Normal distribution, independent increments

## 2. Comparative Framing
| Process | Brownian Motion | Random Walk | OU |
|---|---|---|---|
| **Time** | Continuous | Discrete | Continuous |
| **Increments** | Normal | Any | Mean-reverting |
| **Variance** | Grows linearly | Grows with steps | Bounded long-run |

## 3. Examples + Counterexamples

**Simple Example:**  
Simulate $W_T = \sqrt{T}Z$ with $Z\sim\mathcal{N}(0,1)$.

**Failure Case:**  
Using dependent increments breaks martingale properties; SDE solutions become invalid.

**Edge Case:**  
$T \to 0$ gives tiny increments; discretization dominates error.

## 4. Layer Breakdown
```
Brownian Motion Properties:
├─ W_0 = 0
├─ Independent increments
├─ Stationary increments
├─ Normal increments: W_t - W_s ~ N(0, t-s)
└─ Continuity: paths are continuous, nowhere differentiable
```

**Interaction:** Generate Brownian increments → drive SDEs → simulate paths

## 5. Mini-Project
Simulate and visualize Brownian paths:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T, n, paths = 1.0, 252, 5

dt = T / n
W = np.zeros((paths, n+1))

for i in range(n):
    dW = np.sqrt(dt) * np.random.randn(paths)
    W[:, i+1] = W[:, i] + dW

for p in range(paths):
    plt.plot(np.linspace(0, T, n+1), W[p])

plt.title('Brownian Motion Paths')
plt.grid(alpha=0.3)
plt.show()
```

## 6. Challenge Round

**Q1:** Why is Brownian motion a martingale?  
**A1:** $E[W_t\mid\mathcal{F}_s]=W_s$ due to independent, mean-zero increments.

**Q2:** Why are paths non-differentiable?  
**A2:** Variance grows linearly with time, producing infinitely jagged paths.

**Q3:** How to simulate increments?  
**A3:** Use $\Delta W = \sqrt{\Delta t} Z$ with $Z\sim\mathcal{N}(0,1)$.

**Q4:** Why is Brownian motion central to finance?  
**A4:** It models continuous random shocks in asset prices.

## 7. Key References
- [Wiener process](https://en.wikipedia.org/wiki/Wiener_process)  
- [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion)

---
**Status:** Fundamental noise process | **Complements:** GBM, Ito calculus
