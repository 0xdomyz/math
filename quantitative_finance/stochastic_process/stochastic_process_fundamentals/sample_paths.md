# Sample Paths

## 1. Concept Skeleton
**Definition:** A sample path is one realization of a stochastic process over the index set  
**Purpose:** Understand variability, continuity, and extreme behavior  
**Prerequisites:** Random variables, distributions, basic probability

## 2. Comparative Framing
| Path Type | Example Process | Typical Property |
|-----------|-----------------|------------------|
| **Continuous** | Brownian motion | Nowhere differentiable |
| **Piecewise constant** | Poisson process | Jump times |
| **Oscillatory** | AR processes | Mean-reversion |

## 3. Examples + Counterexamples

**Simple Example:**  
One simulated path of Brownian motion

**Failure Case:**  
Using a single path to infer distributional properties

**Edge Case:**  
Càdlàg paths with infinite activity jumps

## 4. Layer Breakdown
```
Sample Path Analysis:
├─ Simulate or observe X_t
├─ Inspect continuity/jumps
├─ Compute path statistics (max, hitting time)
└─ Compare across paths (ensemble behavior)
```

**Interaction:** Generate paths → analyze features → aggregate insights

## 5. Mini-Project
Simulate and summarize Poisson process paths:
```python
import numpy as np

n = 1000
lam = 2.0
inter_arrivals = np.random.exponential(1/lam, size=n)
arrival_times = np.cumsum(inter_arrivals)
print(arrival_times[:5])
```

## 6. Challenge Round
Common misconceptions:
- Confusing typical path with expected value
- Overinterpreting finite-sample roughness
- Ignoring discretization error in simulations

## 7. Key References
- [Stochastic Process (Wikipedia)](https://en.wikipedia.org/wiki/Stochastic_process)
- [Sample Path (Wikipedia)](https://en.wikipedia.org/wiki/Sample_path)
- [Brownian Motion (Wikipedia)](https://en.wikipedia.org/wiki/Brownian_motion)

---
**Status:** Path-level intuition | **Complements:** Classification, Brownian Motion
