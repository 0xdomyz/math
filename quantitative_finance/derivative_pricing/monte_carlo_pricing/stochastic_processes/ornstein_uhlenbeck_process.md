# Ornstein–Uhlenbeck Process

## 1. Concept Skeleton
**Definition:** Mean-reverting Gaussian process: $dX_t = \theta(\mu - X_t)dt + \sigma dW_t$  
**Purpose:** Model rates, spreads, or volatility around long-run mean  
**Prerequisites:** Brownian motion, mean reversion

## 2. Comparative Framing
| Process | OU | GBM | CIR |
|---|---|---|---|
| **Mean Reversion** | Yes | No | Yes |
| **Distribution** | Normal | Lognormal | Non-central chi-square |
| **Positivity** | No | Yes | Yes |

## 3. Examples + Counterexamples

**Simple Example:**  
Short rate mean-reverts to 3% with speed $\theta=0.5$.

**Failure Case:**  
OU allows negative values; not suitable for strictly positive rates.

**Edge Case:**  
$\theta \to 0$ → becomes Brownian motion with drift.

## 4. Layer Breakdown
```
OU Dynamics:
├─ SDE: dX = θ(μ - X)dt + σ dW
├─ Solution:
│   └─ X_t = μ + (X_0-μ)e^{-θt} + σ∫ e^{-θ(t-s)} dW_s
├─ Mean:
│   └─ E[X_t] = μ + (X_0-μ)e^{-θt}
└─ Variance:
    └─ Var[X_t] = (σ^2 / (2θ)) (1 - e^{-2θt})
```

**Interaction:** Mean reversion → simulate with exact discretization

## 5. Mini-Project
Simulate OU paths and verify mean reversion:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X0, mu, theta, sigma = 0.0, 0.03, 0.7, 0.02
T, n, paths = 5.0, 500, 5

dt = T/n
X = np.zeros((paths, n+1))
X[:,0] = X0

for i in range(n):
    Z = np.random.randn(paths)
    X[:, i+1] = X[:, i] + theta*(mu - X[:, i])*dt + sigma*np.sqrt(dt)*Z

for p in range(paths):
    plt.plot(np.linspace(0,T,n+1), X[p])

plt.axhline(mu, color='red', linestyle='--', label='Long-run mean')
plt.legend()
plt.title('OU Mean Reversion')
plt.grid(alpha=0.3)
plt.show()
```

## 6. Challenge Round

**Q1:** Why is OU Gaussian?  
**A1:** Linear SDE with Gaussian increments yields normally distributed $X_t$.

**Q2:** How does $\theta$ affect paths?  
**A2:** Larger $\theta$ pulls faster toward the mean, reducing variance.

**Q3:** Why unsuitable for equity prices?  
**A3:** It allows negative values and does not grow exponentially.

**Q4:** When to use OU in finance?  
**A4:** Short rates, spreads, stochastic volatility (in some models).

## 7. Key References
- [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)

---
**Status:** Mean-reverting Gaussian process | **Complements:** Vasicek, CIR
