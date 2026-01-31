# Life Table Construction

## 1. Concept Skeleton
**Definition:** Process of building life tables from population, exposure, and death data  
**Purpose:** Produce $l_x$, $q_x$, $p_x$, and life expectancy measures  
**Prerequisites:** Mortality rates, exposure, probability basics

## 2. Comparative Framing
| Table Type | Period Life Table | Cohort Life Table | Abridged Table |
|-----------|-------------------|------------------|----------------|
| **Basis** | Single period rates | Birth cohort experience | Grouped ages |
| **Use** | Pricing/reserving | Longevity trends | Quick summaries |
| **Data Need** | Period rates | Longitudinal | Grouped rates |

## 3. Examples + Counterexamples

**Simple Example:**  
Compute $l_x$ from $q_x$ with radix 100,000

**Failure Case:**  
Using inconsistent exposure and death data periods

**Edge Case:**  
Open-ended last age interval with extrapolated $q_x$

## 4. Layer Breakdown
```
Life Table Build:
├─ Choose radix (e.g., 100,000)
├─ Compute q_x from data
├─ Derive l_x and d_x
├─ Compute L_x, T_x, e_x
└─ Validate monotonicity and consistency
```

**Interaction:** Compute rates → derive columns → validate → publish

## 5. Mini-Project
Construct a toy life table:
```python
import numpy as np

qx = np.array([0.01, 0.012, 0.015])
lx = [100000]
for q in qx:
    lx.append(lx[-1] * (1 - q))
print(lx)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring population exposure weights
- Rounding errors compounding across ages
- Mixing period and cohort assumptions

## 7. Key References
- [Life Table (Wikipedia)](https://en.wikipedia.org/wiki/Life_table)
- [Survival Analysis (Wikipedia)](https://en.wikipedia.org/wiki/Survival_analysis)
- [UN Life Tables](https://population.un.org/)

---
**Status:** Core construction method | **Complements:** Graduation, Mortality Tables
