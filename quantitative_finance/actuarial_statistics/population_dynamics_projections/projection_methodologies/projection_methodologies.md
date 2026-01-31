# Projection Methodologies

## 1. Concept Skeleton
**Definition:** Techniques to forecast population forward using component methods or cohort-component models  
**Purpose:** Long-term demographic planning for pensions, healthcare, labor  
**Prerequisites:** Age structure, rates (fertility, mortality, migration)

## 2. Comparative Framing
| Method | Cohort-Component | Simple Trend | Scenario-Based |
|--------|------------------|--------------|-----------------|
| **Structure** | Age-explicit | Aggregate | Age-explicit + variants |
| **Assumptions** | Rates by age | Single trend | Multiple pathways |
| **Use** | Standard projection | Quick estimate | Policy analysis |

## 3. Examples + Counterexamples

**Simple Example:**  
Apply age-specific rates to current cohorts; age one year; repeat

**Failure Case:**  
Extrapolating trend without considering underlying demographic shifts

**Edge Case:**  
Sub-national projections with high migration uncertainty

## 4. Layer Breakdown
```
Cohort-Component Method:
├─ Start with age structure P_x(0)
├─ Apply fertility → births
├─ Apply mortality → survivorship
├─ Apply net migration by age
├─ Age cohorts forward one year
└─ Repeat for projection period
```

**Interaction:** Initialize → apply components → age → repeat

## 5. Mini-Project
Simple one-year projection:
```python
import numpy as np

pop = np.array([10000, 8000, 5000, 2000])
asfr = np.array([0.08, 0.15, 0.10, 0.02])
lx = np.array([0.95, 0.92, 0.85, 0.50])

births = (pop[1:-1] * asfr[1:-1]).sum() * 0.5  # half female
next_pop = np.zeros(4)
next_pop[0] = births
next_pop[1:] = pop[:-1] * lx[:-1]
print(next_pop)
```

## 6. Challenge Round
Common pitfalls:
- Overfitting trends from short data
- Ignoring uncertainty and sensitivity to assumptions
- Assuming continuity through regime breaks

## 7. Key References
- [Component Method (UN)](https://population.un.org/)
- [Population Projection (Wikipedia)](https://en.wikipedia.org/wiki/Population_projection)
- [Demographic Forecasting](https://www.un.org/en/development/desa/population/methods/)

---
**Status:** Core forecasting framework | **Complements:** All demographic components
