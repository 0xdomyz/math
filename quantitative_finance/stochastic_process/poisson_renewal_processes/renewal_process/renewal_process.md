# Renewal Process

## 1. Concept Skeleton
**Definition:** Counting process with IID inter-arrival times from a general distribution  
**Purpose:** Model events with general waiting times (maintenance, claims, failures)  
**Prerequisites:** IID sequences, distribution functions, expectation

## 2. Comparative Framing
| Process | Renewal Process | Poisson Process | Markov Renewal |
|--------|------------------|-----------------|----------------|
| **Inter-arrivals** | IID general | Exponential | State-dependent |
| **Memoryless** | No | Yes | No |
| **Use Case** | Flexible waiting times | Constant-rate arrivals | Regime-switching arrivals |

## 3. Examples + Counterexamples

**Simple Example:**  
Machine failures with Weibull waiting times

**Failure Case:**  
Dependent inter-arrivals → not a renewal process

**Edge Case:**  
Heavy-tailed inter-arrivals → infinite mean renewal time

## 4. Layer Breakdown
```
Renewal Process Structure:
├─ Inter-arrivals: {X1, X2, ...} IID with CDF F
├─ Arrival times: S_n = X1 + ... + X_n
├─ Counting process: N(t) = max{n : S_n ≤ t}
└─ Renewal function: m(t)=E[N(t)]
```

**Interaction:** Choose F → sum inter-arrivals → count renewals

## 5. Mini-Project
Simulate a renewal process with Weibull inter-arrivals:
```python
import numpy as np

shape, scale = 1.5, 2.0
T = 20.0

current = 0.0
count = 0
while current < T:
    current += np.random.weibull(shape) * scale
    if current <= T:
        count += 1

print("Renewals:", count)
```

## 6. Challenge Round
When is renewal modeling insufficient?
- Inter-arrivals depend on covariates or state
- Non-stationary environment changes waiting-time law
- Need joint modeling of marks or sizes of events

## 7. Key References
- [Renewal Theory (Wikipedia)](https://en.wikipedia.org/wiki/Renewal_theory)
- [Renewal Process (Wikipedia)](https://en.wikipedia.org/wiki/Renewal_process)
- [Ross, Stochastic Processes](https://www.elsevier.com/books/stochastic-processes/ross/978-0-12-598457-5)

---
**Status:** General counting framework | **Complements:** Poisson Process, Reliability Models
