# Graduation of Mortality Data

## 1. Concept Skeleton
**Definition:** Smoothing raw mortality rates to reduce random fluctuation and reveal trends  
**Purpose:** Produce stable mortality tables for pricing and reserving  
**Prerequisites:** Mortality rates, regression/smoothing, exposure data

## 2. Comparative Framing
| Method | Graduation | Crude Rates | Parametric Mortality |
|--------|------------|-------------|----------------------|
| **Noise** | Reduced | High | Reduced |
| **Model** | Smoothing | None | Assumed functional form |
| **Use** | Pricing tables | Experience study | Forecasting |

## 3. Examples + Counterexamples

**Simple Example:**  
Smooth age-specific $q_x$ using moving averages or splines

**Failure Case:**  
Over-smoothing removes genuine cohort effects

**Edge Case:**  
Sparse data at extreme ages causing unstable tails

## 4. Layer Breakdown
```
Graduation Workflow:
├─ Compute crude q_x
├─ Choose smoothing method (spline, kernel, parametric)
├─ Fit and validate
└─ Produce graduated table
```

**Interaction:** Compute → smooth → validate → publish table

## 5. Mini-Project
Simple moving-average smoothing:
```python
import numpy as np

qx = np.array([0.01, 0.012, 0.02, 0.018, 0.022])
window = 3
smooth = np.convolve(qx, np.ones(window)/window, mode='valid')
print(smooth)
```

## 6. Challenge Round
Common pitfalls:
- Using smoothing that violates monotonicity
- Ignoring exposure-weighting by age
- Mixing period and cohort effects

## 7. Key References
- [Graduation (actuarial) (Wikipedia)](https://en.wikipedia.org/wiki/Graduation_(actuarial))
- [Life Table (Wikipedia)](https://en.wikipedia.org/wiki/Life_table)
- [SOA Mortality Studies](https://www.soa.org/)

---
**Status:** Table construction core | **Complements:** Life Table Construction, Mortality Models
