# Poisson Process

## 1. Concept Skeleton
**Definition:** Counting process with independent increments and rate $\lambda$; arrivals follow Poisson distribution  
**Purpose:** Model random event counts over time (claims, arrivals, defaults)  
**Prerequisites:** Exponential distribution, Poisson distribution, independence

## 2. Comparative Framing
| Process | Poisson Process | Bernoulli Process | Renewal Process |
|--------|------------------|-------------------|-----------------|
| **Time** | Continuous | Discrete | Continuous |
| **Inter-arrivals** | Exponential (memoryless) | Geometric | IID (general) |
| **Rate** | Constant $\lambda$ | Constant $p$ per step | Determined by inter-arrival law |

## 3. Examples + Counterexamples

**Simple Example:**  
Phone calls arriving at a call center with constant average rate

**Failure Case:**  
Bursting arrivals with clustering → violates independent increments

**Edge Case:**  
Non-homogeneous Poisson process with time-varying $\lambda(t)$

## 4. Layer Breakdown
```
Poisson Process Mechanics:
├─ N(t): number of events by time t
├─ N(0)=0
├─ Independent increments
├─ P(N(t)=k)=e^{-\lambda t}(\lambda t)^k/k!
└─ Inter-arrival times ~ Exp(\lambda)
```

**Interaction:** Specify $\lambda$ → simulate inter-arrivals → count events

## 5. Mini-Project
Simulate event counts over time:
```python
import numpy as np

lam = 3.0
T = 10.0

# simulate inter-arrivals
times = []
current = 0.0
while current < T:
    current += np.random.exponential(1/lam)
    if current <= T:
        times.append(current)

print("Number of events:", len(times))
```

## 6. Challenge Round
When is Poisson a poor fit?
- Rate varies with time → use non-homogeneous process
- Events cluster (self-exciting) → use Hawkes process
- Dependence between events → violates independent increments

## 7. Key References
- [Poisson Process (Wikipedia)](https://en.wikipedia.org/wiki/Poisson_point_process)
- [Poisson Process Intro (Khan Academy)](https://www.khanacademy.org/math/statistics-probability/probability-library/poisson-process/a/poisson-process-intro)
- [Ross, Stochastic Processes](https://www.elsevier.com/books/stochastic-processes/ross/978-0-12-598457-5)

---
**Status:** Core counting model | **Complements:** Renewal Process, Markov Chains
