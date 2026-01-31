# Applications in Finance

## 1. Concept Skeleton
**Definition:** Use of stochastic processes to model asset dynamics, volatility, and derivatives pricing  
**Purpose:** Quantify risk, price derivatives, and simulate market evolution  
**Prerequisites:** Brownian motion, no-arbitrage, basic derivatives

## 2. Comparative Framing
| Model | Process | Key Feature | Typical Use |
|-------|---------|-------------|-------------|
| **Black–Scholes** | GBM | Constant volatility | Equity options |
| **Heston** | Stochastic volatility | Mean-reverting variance | Vol smile |
| **Merton Jump** | Jump-diffusion | Discontinuous moves | Crash risk |

## 3. Examples + Counterexamples

**Simple Example:**  
Geometric Brownian motion for stock prices: $dS_t=\mu S_t dt+\sigma S_t dW_t$

**Failure Case:**  
Constant volatility assumption during crisis → mispricing

**Edge Case:**  
Short-rate models with negative rates (e.g., Vasicek)

## 4. Layer Breakdown
```
Finance Modeling Stack:
├─ Specify dynamics (e.g., GBM)
├─ Calibrate parameters (μ, σ)
├─ Simulate paths or solve PDE
├─ Price claim (discounted expectation)
└─ Risk metrics (VaR, ES, Greeks)
```

**Interaction:** Choose model → calibrate → price → validate with data

## 5. Mini-Project
Simulate GBM price paths:
```python
import numpy as np

S0, mu, sigma = 100, 0.05, 0.2
T, n = 1.0, 252
 dt = T / n

Z = np.random.normal(size=n)
S = np.zeros(n)
S[0] = S0

for t in range(1, n):
    S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t])

print(S[:5])
```

## 6. Challenge Round
Where do finance models fail?
- Fat tails and skew not captured by GBM
- Volatility clustering violates constant-$\sigma$ assumption
- Market microstructure noise at high frequency

## 7. Key References
- [Black–Scholes Model (Wikipedia)](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Geometric Brownian Motion (Wikipedia)](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)
- [Heston Model (Wikipedia)](https://en.wikipedia.org/wiki/Heston_model)

---
**Status:** Applied modeling layer | **Complements:** Brownian Motion, SDEs
