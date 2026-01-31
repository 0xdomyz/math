# Equity Risk Models

## 1. Concept Skeleton
**Definition:** Stochastic models for equity returns (lognormal, jump-diffusion) used in VA pricing  
**Purpose:** Value equity-linked guarantees and manage market risk  
**Prerequisites:** Geometric Brownian motion, Black-Scholes, jump processes

## 2. Comparative Framing
| Model | Lognormal (GBM) | Jump-Diffusion | Regime-Switching |
|-------|-----------------|----------------|------------------|
| **Returns** | Continuous | Jumps + continuous | State-dependent |
| **Fat Tails** | No | Yes | Yes |
| **Use Case** | Standard options | Crash risk | Market regimes |

## 3. Examples + Counterexamples

**Simple Example:**  
GBM: $dS_t = \mu S_t dt + \sigma S_t dW_t$

**Failure Case:**  
Using GBM for VA guarantees ignores tail risk

**Edge Case:**  
Jump-diffusion with rare but severe drops

## 4. Layer Breakdown
```
Equity Modeling:
├─ Select model (GBM, Merton jump, etc.)
├─ Calibrate to market data
├─ Simulate equity paths
├─ Value guarantees and hedges
└─ Risk metrics (Greeks, VaR)
```

**Interaction:** Calibrate → simulate → value → hedge

## 5. Mini-Project
Simulate GBM equity path:
```python
import numpy as np

S0 = 100
mu = 0.08
sigma = 0.20
T = 1
n = 252

dt = T / n
S = np.zeros(n)
S[0] = S0

for t in range(1, n):
    dW = np.random.normal(0, np.sqrt(dt))
    S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)

print("Final price:", S[-1])
```

## 6. Challenge Round
Common pitfalls:
- Underestimating tail risk with GBM
- Not calibrating to current volatility surface
- Ignoring correlation with interest rates

## 7. Key References
- [Geometric Brownian Motion (Wikipedia)](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)
- [Merton Jump-Diffusion (Wikipedia)](https://en.wikipedia.org/wiki/Jump_diffusion)
- [Black-Scholes (Wikipedia)](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)

---
**Status:** Market risk modeling | **Complements:** Stochastic Interest Rates, Hedging
