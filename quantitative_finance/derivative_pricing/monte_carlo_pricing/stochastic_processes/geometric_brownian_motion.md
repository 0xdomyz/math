# Geometric Brownian Motion (GBM)

## 1. Concept Skeleton
**Definition:** Stochastic process $dS_t = \mu S_t dt + \sigma S_t dW_t$ with lognormal distribution  
**Purpose:** Baseline model for equity prices in Monte Carlo pricing  
**Prerequisites:** Brownian motion, Ito calculus, lognormal distribution

## 2. Comparative Framing
| Model | GBM | OU | Heston |
|---|---|---|---|
| **Mean Reversion** | No | Yes | No (variance reverts) |
| **Volatility** | Constant | Constant | Stochastic |
| **Distribution** | Lognormal | Normal | Non-lognormal |

## 3. Examples + Counterexamples

**Simple Example:**  
Simulate $S_T = S_0 \exp((r-\frac12\sigma^2)T + \sigma \sqrt{T}Z)$.

**Failure Case:**  
GBM fails to capture volatility smile; constant σ misprices OTM options.

**Edge Case:**  
Very short $T$ → GBM approximates lognormal with tiny variance.

## 4. Layer Breakdown
```
GBM Mechanics:
├─ SDE: dS = μS dt + σS dW
├─ Log transform: d(ln S) = (μ - ½σ²)dt + σ dW
├─ Solution:
│   └─ S_t = S_0 exp((μ-½σ²)t + σW_t)
├─ Risk-neutral drift:
│   └─ μ = r - q
└─ Monte Carlo:
    ├─ Draw Z ~ N(0,1)
    └─ S_T = S_0 exp((r-q-½σ²)T + σ√T Z)
```

**Interaction:** Draw normals → compute $S_T$ → payoff → discount

## 5. Mini-Project
Monte Carlo price for a European call under GBM:
```python
import numpy as np

S0, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.0, 0.2
N = 200000
np.random.seed(42)

Z = np.random.randn(N)
ST = S0 * np.exp((r-q-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
price = np.exp(-r*T) * np.mean(np.maximum(ST - K, 0))

print(f"MC price: {price:.4f}")
```

## 6. Challenge Round

**Q1:** Why lognormal?  
**A1:** The log of GBM is normal by Ito’s lemma, making $S_t$ lognormal.

**Q2:** Why use $r-q$ drift?  
**A2:** Risk-neutral pricing replaces expected return with carry-adjusted risk-free rate.

**Q3:** When is GBM inadequate?  
**A3:** For heavy tails, skew, jumps, or stochastic volatility markets.

**Q4:** How does σ affect option price?  
**A4:** Higher σ increases convex payoff value; call price rises with σ.

## 7. Key References
- [Geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)  
- [Black–Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)

---
**Status:** Baseline equity process | **Complements:** Heston, jump diffusion
