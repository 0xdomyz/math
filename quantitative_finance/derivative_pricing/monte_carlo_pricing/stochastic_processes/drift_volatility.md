# Drift & Volatility

## 1. Concept Skeleton
**Definition:** Drift $\mu$ is expected rate of change; volatility $\sigma$ measures dispersion around drift  
**Purpose:** Core parameters controlling SDE dynamics and option values  
**Prerequisites:** Expectation, variance, GBM

## 2. Comparative Framing
| Parameter | Drift $\mu$ | Volatility $\sigma$ | Risk-Free $r$ |
|---|---|---|---|
| **Meaning** | Expected return | Uncertainty | Discount rate |
| **Pricing Role** | Replaced by $r-q$ | Major driver | Discounting |
| **Estimation** | Noisy | More stable | Market observable |

## 3. Examples + Counterexamples

**Simple Example:**  
Higher $\sigma$ raises option prices; higher $\mu$ does not under risk-neutral pricing.

**Failure Case:**  
Using historical $\mu$ for pricing yields arbitrage-inconsistent values.

**Edge Case:**  
$\sigma=0$ → deterministic path, option equals discounted intrinsic value.

## 4. Layer Breakdown
```
Drift/Volatility Roles:
├─ SDE: dS = μS dt + σS dW
├─ Risk-neutral adjustment:
│   └─ μ → r - q
├─ Estimation:
│   ├─ μ from long sample; high error
│   └─ σ from returns; more stable
└─ Pricing impact:
    ├─ σ increases convex payoff value
    └─ μ irrelevant under risk-neutral measure
```

**Interaction:** Estimate σ, set μ=r-q → simulate paths → price

## 5. Mini-Project
Show that μ does not affect risk-neutral price:
```python
import numpy as np

def mc_price(mu, N=200000):
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    Z = np.random.randn(N)
    ST = S0 * np.exp((mu-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    return np.exp(-r*T) * np.mean(np.maximum(ST-K,0))

np.random.seed(42)
print(mc_price(0.05), mc_price(0.15))
```

## 6. Challenge Round

**Q1:** Why is drift ignored in risk-neutral pricing?  
**A1:** Under the pricing measure, expected return equals risk-free rate to eliminate arbitrage.

**Q2:** Why is σ more important than μ?  
**A2:** Option payoffs are convex; uncertainty directly increases expected payoff.

**Q3:** How does dividend yield affect drift?  
**A3:** Drift becomes $r-q$, lowering expected price growth.

**Q4:** Why is μ estimation noisy?  
**A4:** Expected returns are small relative to volatility; sample error dominates.

## 7. Key References
- [Drift (random walk)](https://en.wikipedia.org/wiki/Drift_(random_walk))  
- [Volatility (finance)](https://en.wikipedia.org/wiki/Volatility_(finance))

---
**Status:** Core model parameters | **Complements:** GBM, calibration
