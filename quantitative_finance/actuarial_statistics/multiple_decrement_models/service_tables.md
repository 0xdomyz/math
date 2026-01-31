# Service Tables

## 1. Concept Skeleton
**Definition:** Tables combining multiple decrements for employee service statuses (active, disabled, retired)  
**Purpose:** Model workforce transitions and benefit eligibility  
**Prerequisites:** Multiple decrement theory, pension plan design

## 2. Comparative Framing
| Table Type | Service Table | Mortality Table | Disability Table |
|-----------|---------------|-----------------|------------------|
| **States** | Multiple statuses | Alive/dead | Active/disabled |
| **Use** | Pensions/group benefits | Life insurance | Disability plans |
| **Complexity** | High | Low | Medium |

## 3. Examples + Counterexamples

**Simple Example:**  
Active → Disabled → Retired transitions in a pension plan

**Failure Case:**  
Ignoring reactivation from disabled to active status

**Edge Case:**  
Small employer with volatile transition rates

## 4. Layer Breakdown
```
Service Table Construction:
├─ Define states and decrements
├─ Estimate transition probabilities
├─ Build multi-state table
└─ Validate with experience data
```

**Interaction:** Define states → estimate transitions → construct table

## 5. Mini-Project
Create a toy transition matrix:
```python
import numpy as np

P = np.array([
    [0.92, 0.05, 0.02, 0.01],
    [0.10, 0.85, 0.03, 0.02],
    [0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 1.00]
])
print(P)
```

## 6. Challenge Round
Common pitfalls:
- Overfitting transition rates on small data
- Ignoring policy changes that affect transitions
- Inconsistent exposure measurement across states

## 7. Key References
- [Multi-state Model (Wikipedia)](https://en.wikipedia.org/wiki/Multi-state_model)
- [Life Table (Wikipedia)](https://en.wikipedia.org/wiki/Life_table)
- [SOA Pension Studies](https://www.soa.org/)

---
**Status:** Workforce transition tool | **Complements:** Pension Mathematics, Multiple Decrements
