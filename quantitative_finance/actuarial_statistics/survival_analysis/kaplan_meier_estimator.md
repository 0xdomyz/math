# Kaplan–Meier Estimator

## 1. Concept Skeleton
**Definition:** Non-parametric estimator of the survival function from right-censored data  
**Purpose:** Estimate survival probabilities over time without distributional assumptions  
**Prerequisites:** Censoring, survival function, empirical probability

## 2. Comparative Framing
| Estimator | Kaplan–Meier | Life Table | Nelson–Aalen |
|-----------|--------------|------------|--------------|
| **Data** | Exact event times | Grouped intervals | Exact times |
| **Output** | Survival function | Survival by intervals | Cumulative hazard |
| **Assumptions** | Non-parametric | Interval constancy | Non-parametric |

## 3. Examples + Counterexamples

**Simple Example:**  
Estimate survival for a cohort with right-censored follow-up

**Failure Case:**  
Heavy left-truncation without adjustment biases survival upward

**Edge Case:**  
Many tied event times → need proper tie handling

## 4. Layer Breakdown
```
Kaplan–Meier Steps:
├─ Order event times
├─ Compute conditional survival at each event
├─ Multiply successive conditional survivals
└─ Plot stepwise survival curve
```

**Interaction:** Order → compute conditional → multiply → visualize

## 5. Mini-Project
Compute a KM curve from simple data:
```python
import numpy as np

times = np.array([2, 3, 3, 5, 8])
status = np.array([1, 1, 0, 1, 0])  # 1=event, 0=censored

# naive KM calculation for illustration
unique_times = np.unique(times[status == 1])
surv = 1.0
for t in unique_times:
    at_risk = np.sum(times >= t)
    events = np.sum((times == t) & (status == 1))
    surv *= (1 - events / at_risk)
    print(t, surv)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring censoring mechanisms
- Treating censored times as events
- Comparing KM curves without accounting for covariates

## 7. Key References
- [Kaplan–Meier Estimator (Wikipedia)](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator)
- [Survival Analysis (Wikipedia)](https://en.wikipedia.org/wiki/Survival_analysis)
- [Klein & Moeschberger, Survival Analysis](https://link.springer.com/book/10.1007/978-1-4757-3294-8)

---
**Status:** Core non-parametric tool | **Complements:** Nelson–Aalen, Cox Model
