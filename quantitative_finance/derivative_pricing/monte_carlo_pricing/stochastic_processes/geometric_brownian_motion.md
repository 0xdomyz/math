# Geometric Brownian Motion (GBM)

## Concept Skeleton
**Definition:** Stochastic process $dS_t = \mu S_t dt + \sigma S_t dW_t$ with lognormal distribution  
**Purpose:** Baseline model for equity prices in Monte Carlo pricing  
**Prerequisites:** Brownian motion, Ito calculus, lognormal distribution

## Comparative Framing
| Model | GBM | OU | Heston |
|---|---|---|---|
| **Mean Reversion** | No | Yes | No (variance reverts) |
| **Volatility** | Constant | Constant | Stochastic |
| **Distribution** | Lognormal | Normal | Non-lognormal |

## Examples + Counterexamples
**Simple Example:**  
Simulate $S_T = S_0 \exp((r-\frac12\sigma^2)T + \sigma \sqrt{T}Z)$.

**Failure Case:**  
GBM fails to capture volatility smile; constant σ misprices OTM options.

**Edge Case:**  
Very short $T$ → GBM approximates lognormal with tiny variance.

## Layer Breakdown
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

## Challenge Round
**Q1:** Why lognormal?  
**A1:** The log of GBM is normal by Ito’s lemma, making $S_t$ lognormal.

**Q2:** Why use $r-q$ drift?  
**A2:** Risk-neutral pricing replaces expected return with carry-adjusted risk-free rate.

**Q3:** When is GBM inadequate?  
**A3:** For heavy tails, skew, jumps, or stochastic volatility markets.

**Q4:** How does σ affect option price?  
**A4:** Higher σ increases convex payoff value; call price rises with σ.

## Key References
- [Geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)  
- [Black–Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)

---
**Status:** Baseline equity process | **Complements:** Heston, jump diffusion
