# Quasi-Random Numbers (Low-Discrepancy)

## 1. Concept Skeleton
**Definition:** Deterministic sequences that fill the unit cube more uniformly than pseudo-random samples  
**Purpose:** Improve convergence of Monte Carlo integration for smooth payoffs  
**Prerequisites:** Discrepancy, dimensionality, scrambling

## 2. Comparative Framing
| Sequence | Strength | Weakness | Use |
|---|---|---|---|
| **Sobol** | High-dimensional uniformity | Needs scrambling | QMC pricing |
| **Halton** | Simple | Correlations in high dims | Low-dim integrals |
| **Niederreiter** | Strong theoretical | Complex | Research |

## 3. Examples + Counterexamples

**Simple Example:**  
Sobol points reduce variance in Asian option pricing vs PRNG.

**Failure Case:**  
High dimension (d>100) without scrambling → correlations degrade performance.

**Edge Case:**  
Non-smooth payoff (digital) reduces QMC advantage; variance reduction minimal.

## 4. Layer Breakdown
```
QMC Workflow:
├─ Generate Low-Discrepancy Points
│   ├─ Sobol or Halton sequence
│   └─ Apply scrambling if needed
├─ Map to Distribution
│   └─ Apply inverse CDF to each dimension
├─ Compute Payoff
├─ Estimate Price
│   └─ Average discounted payoffs
└─ Compare to PRNG
```

**Interaction:** Generate quasi-random points → transform to target distribution → simulate → estimate

## 5. Mini-Project
Compare Sobol vs PRNG for integral estimation:
```python
import numpy as np
from scipy.stats import qmc

# Integral of f(x)=exp(-x) on [0,1]

def estimate_mc(n):
    x = np.random.rand(n)
    return np.mean(np.exp(-x))

sampler = qmc.Sobol(d=1, scramble=True, seed=42)

def estimate_qmc(n):
    x = sampler.random(n)[:,0]
    return np.mean(np.exp(-x))

true_val = 1 - np.exp(-1)

for n in [512, 1024, 2048, 4096]:
    mc = estimate_mc(n)
    qmc = estimate_qmc(n)
    print(f"n={n}, MC err={abs(mc-true_val):.6f}, QMC err={abs(qmc-true_val):.6f}")
```

## 6. Challenge Round

**Q1:** Why does QMC converge faster?  
**A1:** Low-discrepancy points reduce integration error for smooth functions; error decreases closer to $O(1/N)$.

**Q2:** Why use scrambling?  
**A2:** It randomizes QMC for error estimation while preserving low discrepancy.

**Q3:** When does QMC fail?  
**A3:** High-dimensional or non-smooth payoffs reduce the uniformity advantage.

**Q4:** How to choose dimension ordering?  
**A4:** Place most important factors in early dimensions; QMC is most uniform there.

## 7. Key References
- [Low-discrepancy sequence](https://en.wikipedia.org/wiki/Low-discrepancy_sequence)  
- [Quasi-Monte Carlo method](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method)

---
**Status:** Variance reduction via sampling | **Complements:** RNG, variance reduction techniques
