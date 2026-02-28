# Ornstein–Uhlenbeck Process

## Concept Skeleton
**Definition:** Mean-reverting Gaussian process: $dX_t = \theta(\mu - X_t)dt + \sigma dW_t$  
**Purpose:** Model rates, spreads, or volatility around long-run mean  
**Prerequisites:** Brownian motion, mean reversion

## Comparative Framing
| Process | OU | GBM | CIR |
|---|---|---|---|
| **Mean Reversion** | Yes | No | Yes |
| **Distribution** | Normal | Lognormal | Non-central chi-square |
| **Positivity** | No | Yes | Yes |

## Examples + Counterexamples
**Simple Example:**  
Short rate mean-reverts to 3% with speed $\theta=0.5$.

**Failure Case:**  
OU allows negative values; not suitable for strictly positive rates.

**Edge Case:**  
$\theta \to 0$ → becomes Brownian motion with drift.

## Layer Breakdown
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

## Challenge Round
**Q1:** Why is OU Gaussian?  
**A1:** Linear SDE with Gaussian increments yields normally distributed $X_t$.

**Q2:** How does $\theta$ affect paths?  
**A2:** Larger $\theta$ pulls faster toward the mean, reducing variance.

**Q3:** Why unsuitable for equity prices?  
**A3:** It allows negative values and does not grow exponentially.

**Q4:** When to use OU in finance?  
**A4:** Short rates, spreads, stochastic volatility (in some models).

## Key References
- [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)

---
**Status:** Mean-reverting Gaussian process | **Complements:** Vasicek, CIR
