# Nelson–Aalen Estimator

## 1. Concept Skeleton
**Definition:** Non-parametric estimator of the cumulative hazard function with censored data  
**Purpose:** Estimate cumulative hazard and derive survival via $S(t)=\exp(-\hat{H}(t))$  
**Prerequisites:** Hazard function, censoring, survival function

## 2. Comparative Framing
| Estimator | Nelson–Aalen | Kaplan–Meier | Life Table |
|-----------|--------------|--------------|------------|
| **Target** | Cumulative hazard | Survival function | Survival (grouped) |
| **Data** | Exact times | Exact times | Intervals |
| **Output** | Stepwise hazard | Stepwise survival | Interval survival |

## 3. Examples + Counterexamples

**Simple Example:**  
Estimate cumulative hazard in a medical cohort with censoring

**Failure Case:**  
Treating left-truncated data as right-censored only

**Edge Case:**  
Sparse events leading to high variance estimates

## 4. Layer Breakdown
```
Nelson–Aalen Steps:
├─ Order event times
├─ At each time: ΔH = d_i / n_i
├─ Accumulate H(t)
└─ Convert to survival if needed
```

**Interaction:** Compute increments → sum → optionally transform to survival

## 5. Mini-Project
Compute a simple Nelson–Aalen estimate:
```python
import numpy as np

times = np.array([2, 3, 5, 5, 8])
status = np.array([1, 1, 1, 0, 0])

unique_times = np.unique(times[status == 1])
H = 0.0
for t in unique_times:
    at_risk = np.sum(times >= t)
    events = np.sum((times == t) & (status == 1))
    H += events / at_risk
    print(t, H)
```

## 6. Challenge Round
Common pitfalls:
- Confusing hazard with probability
- Ignoring censoring assumptions
- Overinterpreting cumulative hazard shape

## 7. Key References
- [Nelson–Aalen Estimator (Wikipedia)](https://en.wikipedia.org/wiki/Nelson%E2%80%93Aalen_estimator)
- [Survival Analysis (Wikipedia)](https://en.wikipedia.org/wiki/Survival_analysis)
- [Klein & Moeschberger, Survival Analysis](https://link.springer.com/book/10.1007/978-1-4757-3294-8)

---
**Status:** Core non-parametric tool | **Complements:** Kaplan–Meier, Cox Model
