# Brownian Motion

## 1. Concept Skeleton
**Definition:** Continuous-time stochastic process with Gaussian increments, $W_0=0$, and independent, stationary increments  
**Purpose:** Canonical model for random motion and foundation of diffusion models  
**Prerequisites:** Normal distribution, stochastic processes, continuous time

## 2. Comparative Framing
| Process | Brownian Motion | Random Walk | Ornstein–Uhlenbeck |
|--------|------------------|-------------|--------------------|
| **Time** | Continuous | Discrete | Continuous |
| **Increments** | Gaussian, independent | Discrete steps | Mean-reverting Gaussian |
| **Paths** | Continuous, nowhere differentiable | Stepwise | Continuous |

## 3. Examples + Counterexamples

**Simple Example:**  
Particle position modeled by $W_t$ over time

**Failure Case:**  
Jump processes with sudden discontinuities → not Brownian

**Edge Case:**  
Drifted Brownian motion $X_t=\mu t+\sigma W_t$

## 4. Layer Breakdown
```
Brownian Motion Properties:
├─ W_0 = 0
├─ W_t - W_s ~ N(0, t-s)
├─ Independent increments
├─ Continuous paths
└─ Scaling: W_{ct} ~ sqrt(c) W_t
```

**Interaction:** Specify time grid → simulate increments → accumulate path

## 5. Mini-Project
Simulate a Brownian motion path:
```python
import numpy as np

T, n = 1.0, 1000
 dt = T / n
increments = np.sqrt(dt) * np.random.normal(size=n)
W = np.cumsum(increments)
print(W[:5])
```

## 6. Challenge Round
Common pitfalls:
- Treating paths as differentiable
- Confusing variance $t$ with standard deviation $\sqrt{t}$
- Using large $dt$ leading to poor approximation

## 7. Key References
- [Brownian Motion (Wikipedia)](https://en.wikipedia.org/wiki/Brownian_motion)
- [Wiener Process (Wikipedia)](https://en.wikipedia.org/wiki/Wiener_process)
- [Øksendal, Stochastic Differential Equations](https://link.springer.com/book/10.1007/978-3-642-14394-6)

---
**Status:** Core diffusion process | **Complements:** Ito Calculus, SDEs
