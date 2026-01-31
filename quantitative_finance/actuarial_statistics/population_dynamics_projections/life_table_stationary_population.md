# Life Table Stationary Population

## 1. Concept Skeleton
**Definition:** Hypothetical population with constant age-specific rates and stable age structure  
**Purpose:** Benchmark for demographic analysis and long-run equilibrium  
**Prerequisites:** Life table, fertility rates, steady-state assumptions

## 2. Comparative Framing
| Population | Stationary | Stable | Observed |
|-----------|-----------|--------|----------|
| **Growth** | Zero ($r=0$) | Constant | Variable |
| **Structure** | Fixed | Fixed | Changing |
| **Use** | Theoretical | Forecast | Actual |

## 3. Examples + Counterexamples

**Simple Example:**  
Life table with zero growth rate implies stationary age structure

**Failure Case:**  
Assuming observed population is stationary when growth rate ≠ 0

**Edge Case:**  
Very long-lived population with low fertility rates approaching stationarity

## 4. Layer Breakdown
```
Stationary Population:
├─ Assume constant rates (mortality, fertility)
├─ Solve for growth rate r = 0
├─ Derive stable age structure
└─ Use as benchmark
```

**Interaction:** Fix rates → solve for r=0 → derive structure

## 5. Mini-Project
Estimate stationary population structure:
```python
import numpy as np

Lx = np.array([100000, 95000, 90000, 80000, 50000])
T_x = np.cumsum(Lx[::-1])[::-1]
Cx = Lx / T_x[0]
print("Proportion in each age group:", Cx)
```

## 6. Challenge Round
Common pitfalls:
- Confusing stationary with stable population
- Using stationary structure for rapidly changing populations
- Ignoring historical shocks affecting current structure

## 7. Key References
- [Stationary Population (Wikipedia)](https://en.wikipedia.org/wiki/Stationary_population)
- [Life Table (Wikipedia)](https://en.wikipedia.org/wiki/Life_table)
- [Demographic Methods (UN)](https://population.un.org/)

---
**Status:** Theoretical benchmark | **Complements:** Stable Population, Projection Methods
