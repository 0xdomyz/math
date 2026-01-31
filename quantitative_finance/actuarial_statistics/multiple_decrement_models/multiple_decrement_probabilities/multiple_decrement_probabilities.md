# Multiple Decrement Probabilities

## 1. Concept Skeleton
**Definition:** Probabilities $q_x^{(j)}$ for decrement cause $j$ in a multi-risk setting  
**Purpose:** Model competing risks and allocate total decrement into causes  
**Prerequisites:** Survival probability, conditional probability, life tables

## 2. Comparative Framing
| Model | Multiple Decrement | Single Decrement | Competing Risks |
|------|---------------------|-----------------|-----------------|
| **Causes** | Many | One | Many |
| **Output** | Cause-specific $q_x^{(j)}$ | Total $q_x$ | Cause-specific hazards |
| **Use** | Actuarial exits | Mortality only | Medical/actuarial |

## 3. Examples + Counterexamples

**Simple Example:**  
Total $q_x=0.04$ split as death $0.03$ and lapse $0.01$

**Failure Case:**  
Cause probabilities sum to more than total $q_x$

**Edge Case:**  
Rare causes with unstable estimates in small datasets

## 4. Layer Breakdown
```
Multiple Decrement Setup:
├─ Total decrement q_x
├─ Cause shares: q_x^{(j)}
├─ Sum constraint: Σ q_x^{(j)} = q_x
└─ Convert to cause-specific rates if needed
```

**Interaction:** Estimate cause rates → normalize → validate sums

## 5. Mini-Project
Normalize cause-specific rates:
```python
import numpy as np

cause_rates = np.array([0.02, 0.01, 0.005])
total = 0.04
scaled = total * cause_rates / cause_rates.sum()
print(scaled, scaled.sum())
```

## 6. Challenge Round
Common pitfalls:
- Mixing independent rates with dependent causes
- Using unadjusted historical shares across regimes
- Ignoring censoring and exposure differences

## 7. Key References
- [Competing Risks (Wikipedia)](https://en.wikipedia.org/wiki/Competing_risks)
- [Life Table (Wikipedia)](https://en.wikipedia.org/wiki/Life_table)
- [SOA Multiple Decrement](https://www.soa.org/)

---
**Status:** Core decrement framework | **Complements:** Cause-Specific Mortality, Service Tables
