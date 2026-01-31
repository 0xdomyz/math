# Multi-Asset Simulation

## 1. Concept Skeleton
**Definition:** Simulate multiple asset price paths jointly with a specified dependence structure  
**Purpose:** Price multi-asset derivatives (basket, rainbow), compute portfolio risk  
**Prerequisites:** GBM, correlation matrices, Cholesky, Monte Carlo

## 2. Comparative Framing
| Approach | Independent Assets | Correlated Assets | Factor Model |
|---|---|---|---|
| **Dependence** | None | Correlation matrix | Low-rank via PCA |
| **Accuracy** | Low | High | Medium |
| **Cost** | Low | Moderate | Low |

## 3. Examples + Counterexamples

**Simple Example:**  
Basket call on 3 stocks with $\rho=0.5$; price rises when correlation increases.

**Failure Case:**  
Simulating assets independently underestimates basket variance and option value.

**Edge Case:**  
Perfect correlation: basket behaves like single asset with weighted volatility.

## 4. Layer Breakdown
```
Multi-Asset Monte Carlo:
├─ Inputs:
│   ├─ S0 vector, vol vector, rates, T
│   └─ Correlation matrix ρ
├─ Correlated Shocks:
│   ├─ Z ~ N(0, I)
│   ├─ L = chol(ρ)
│   └─ X = Z L^T (correlated normals)
├─ Asset Evolution (GBM):
│   └─ S_i(T) = S_i(0) exp((r-σ_i^2/2)T + σ_i √T X_i)
├─ Payoff:
│   ├─ Basket: max(Σw_i S_i(T) - K, 0)
│   └─ Rainbow: max(max(S_i(T)) - K, 0)
└─ Discounting:
    └─ V = e^{-rT} E[Payoff]
```

**Interaction:** Correlated normals → joint terminal prices → payoff → discount

## 5. Mini-Project
Price a basket option and compare to independent simulation:
```python
import numpy as np

np.random.seed(42)

def basket_call_mc(S0, w, K, T, r, vol, rho, n=200000):
    d = len(S0)
    L = np.linalg.cholesky(rho)
    Z = np.random.randn(n, d)
    X = Z @ L.T
    ST = S0 * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * X)
    basket = ST @ w
    payoff = np.maximum(basket - K, 0)
    price = np.exp(-r*T) * payoff.mean()
    return price

S0 = np.array([100, 100, 100])
w = np.array([0.4, 0.3, 0.3])
K = 100
T = 1
r = 0.05
vol = np.array([0.25, 0.30, 0.35])

rho_corr = np.array([[1.0, 0.5, 0.5],
                     [0.5, 1.0, 0.5],
                     [0.5, 0.5, 1.0]])

rho_ind = np.eye(3)

price_corr = basket_call_mc(S0, w, K, T, r, vol, rho_corr)
price_ind = basket_call_mc(S0, w, K, T, r, vol, rho_ind)

print(f"Basket (correlated): ${price_corr:.4f}")
print(f"Basket (independent): ${price_ind:.4f}")
```

## 6. Challenge Round

**Q1:** Why does higher correlation increase basket call value?  
**A1:** Correlation increases basket variance $\sigma_B^2 = w^T \Sigma w$, raising option value for convex payoffs.

**Q2:** When is factor simulation preferable?  
**A2:** Large dimension (N > 50) and covariance has low effective rank; PCA reduces cost.

**Q3:** Why do rainbow options react oppositely to correlation?  
**A3:** Best-of benefits from diversification; lower correlation increases chance one asset excels.

**Q4:** How do you validate a multi-asset simulator?  
**A4:** Check marginal distributions, correlation of simulated returns, and pricing against known bounds.

## 7. Key References
- [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method)  
- [Geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)

---
**Status:** Core multi-asset pricing tool | **Complements:** Basket/Rainbow options
