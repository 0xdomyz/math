# Definition & Examples

## 1. Concept Skeleton
**Definition:** A stochastic process is a collection of random variables indexed by time or space  
**Purpose:** Model systems evolving with uncertainty (queues, prices, biology)  
**Prerequisites:** Random variables, probability spaces, distributions

## 2. Comparative Framing
| Concept | Stochastic Process | Random Variable | Time Series |
|--------|---------------------|----------------|-------------|
| **Index** | Time/space | None | Time |
| **Structure** | Collection of RVs | Single RV | Observed sequence |
| **Goal** | Model dynamics | Describe uncertainty | Analyze data |

## 3. Examples + Counterexamples

**Simple Example:**  
Poisson process counting arrivals per hour

**Failure Case:**  
Deterministic linear trend with no randomness → not stochastic

**Edge Case:**  
Random walk with heavy-tailed steps → non-Gaussian behavior

## 4. Layer Breakdown
```
Stochastic Process Components:
├─ Index set T (time or space)
├─ State space S (possible values)
├─ Random variables {X_t : t ∈ T}
├─ Finite-dimensional distributions
└─ Sample paths (realizations)
```

**Interaction:** Choose index/state → define joint laws → analyze paths

## 5. Mini-Project
Simulate a simple random walk:
```python
import numpy as np

n = 100
steps = np.random.choice([-1, 1], size=n)
walk = np.cumsum(steps)
print(walk[:10])
```

## 6. Challenge Round
When is the definition too broad?
- Model lacks stationarity assumptions needed for inference
- Index set is continuous but data are discrete observations
- Heavy tails violate Gaussian-based tools

## 7. Key References
- [Stochastic Process (Wikipedia)](https://en.wikipedia.org/wiki/Stochastic_process)
- [Random Walk (Wikipedia)](https://en.wikipedia.org/wiki/Random_walk)
- [Stochastic Processes (Grimmett & Stirzaker)](https://global.oup.com/academic/product/probability-and-random-processes-9780198572220)

---
**Status:** Foundational definition | **Complements:** Classification, Sample Paths
