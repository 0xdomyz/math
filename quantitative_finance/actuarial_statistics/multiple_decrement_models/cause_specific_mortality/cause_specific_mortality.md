# Cause-Specific Mortality

## 1. Concept Skeleton
**Definition:** Mortality rates by cause (e.g., death, lapse, withdrawal) within competing risks  
**Purpose:** Understand drivers of exits and improve pricing/reserving  
**Prerequisites:** Multiple decrement models, hazard functions

## 2. Comparative Framing
| Measure | Cause-Specific Rate | Overall Mortality | Cumulative Incidence |
|--------|----------------------|-------------------|----------------------|
| **Focus** | Single cause | All causes | Cause probability over time |
| **Use** | Risk attribution | Aggregate pricing | Event probability |
| **Data Need** | Cause labels | Total deaths | Cause + time |

## 3. Examples + Counterexamples

**Simple Example:**  
Cause-specific death rate for cancer in a portfolio

**Failure Case:**  
Treating cause-specific rates as independent of other causes

**Edge Case:**  
Sparse cause counts leading to unstable estimates

## 4. Layer Breakdown
```
Cause-Specific Modeling:
├─ Identify causes and exposures
├─ Estimate cause-specific rates
├─ Check sum vs total q_x
└─ Use for pricing or experience studies
```

**Interaction:** Classify events → estimate → reconcile with total

## 5. Mini-Project
Estimate cause shares from counts:
```python
import numpy as np

counts = np.array([40, 15, 5])
shares = counts / counts.sum()
print(shares)
```

## 6. Challenge Round
Common pitfalls:
- Misclassification of causes
- Ignoring competing risk bias in cause incidence
- Not adjusting for exposure duration

## 7. Key References
- [Competing Risks (Wikipedia)](https://en.wikipedia.org/wiki/Competing_risks)
- [Hazard Function (Wikipedia)](https://en.wikipedia.org/wiki/Hazard_function)
- [SOA Experience Studies](https://www.soa.org/)

---
**Status:** Risk attribution tool | **Complements:** Multiple Decrements, Survival Analysis
