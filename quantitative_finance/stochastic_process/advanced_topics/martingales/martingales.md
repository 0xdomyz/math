# Martingales

## 1. Concept Skeleton
**Definition:** Process $M_t$ with $E[M_{t+1}|\mathcal{F}_t]=M_t$ (fair game)  
**Purpose:** Model fair dynamics; core tool for optional stopping and pricing  
**Prerequisites:** Conditional expectation, filtrations, stochastic processes

## 2. Comparative Framing
| Process | Martingale | Submartingale | Supermartingale |
|--------|------------|----------------|-----------------|
| **Conditional Mean** | Equal | Increases | Decreases |
| **Trend** | Fair | Favorable | Unfavorable |
| **Typical Use** | Pricing | Growth bounds | Risk bounds |

## 3. Examples + Counterexamples

**Simple Example:**  
Gambler’s fair game with zero expected gain per step

**Failure Case:**  
Process with positive drift is not a martingale

**Edge Case:**  
Stopped martingale remains a martingale under integrability conditions

## 4. Layer Breakdown
```
Martingale Framework:
├─ Filtration {\mathcal{F}_t}
├─ Integrability: E|M_t| < ∞
├─ Fairness: E[M_{t+1}|\mathcal{F}_t]=M_t
└─ Consequences: optional stopping, Doob inequalities
```

**Interaction:** Define filtration → verify fairness → apply martingale results

## 5. Mini-Project
Simulate a fair random walk and verify empirical martingale property:
```python
import numpy as np

n = 10000
steps = np.random.choice([-1, 1], size=n)
M = np.cumsum(steps)
print(M[:5])
```

## 6. Challenge Round
When does martingale theory fail?
- Non-integrable processes violate conditions
- Stopping times with infinite expectation can break optional stopping
- Hidden trends from changing measure

## 7. Key References
- [Martingale (Wikipedia)](https://en.wikipedia.org/wiki/Martingale_(probability_theory))
- [Doob’s Martingale Inequality (Wikipedia)](https://en.wikipedia.org/wiki/Doob%27s_martingale_inequality)
- [Williams, Probability with Martingales](https://www.cambridge.org/core/books/probability-with-martingales/1E3A8B0C8A0D5DA851DCB8C1B83D7D7A)

---
**Status:** Core stochastic tool | **Complements:** Ito Calculus, Stopping Times
