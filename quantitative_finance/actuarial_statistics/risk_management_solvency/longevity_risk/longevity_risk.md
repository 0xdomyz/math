# Longevity Risk

## 1. Concept Skeleton
**Definition:** Risk that policyholders live longer than assumed, increasing payouts  
**Purpose:** Quantify capital strain and price products with long-duration liabilities  
**Prerequisites:** Life tables, survival models, discounting

## 2. Comparative Framing
| Risk Type | Longevity Risk | Mortality Risk | Lapse Risk |
|----------|-----------------|---------------|------------|
| **Direction** | Lives longer | Deaths earlier | Policies terminate |
| **Impact** | Higher annuity costs | Higher death claims | Profit/expense mismatch |
| **Products** | Pensions, annuities | Life insurance | Most long-term products |

## 3. Examples + Counterexamples

**Simple Example:**  
Annuity portfolio where actual survival exceeds table by 2 years

**Failure Case:**  
Using outdated mortality improvements → systematic reserve shortfall

**Edge Case:**  
Cohort effects: longevity improvements differ by birth year

## 4. Layer Breakdown
```
Longevity Risk Workflow:
├─ Choose base mortality table
├─ Apply improvement scale (age, period, cohort)
├─ Project survival probabilities
├─ Stress scenarios (longevity shocks)
└─ Capital impact (PV of increased payouts)
```

**Interaction:** Assumptions → projection → stress → capital impact

## 5. Mini-Project
Stress longevity by shifting mortality improvement:
```python
import numpy as np

# simple longevity stress: reduce mortality by 10%
qx = np.array([0.010, 0.012, 0.014, 0.016])
qx_stress = 0.9 * qx
px = 1 - qx
px_stress = 1 - qx_stress
print(px, px_stress)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring cohort effects in improvements
- Calibrating from small or biased datasets
- Assuming independence across ages and years

## 7. Key References
- [Longevity Risk (SOA)](https://www.soa.org/)
- [Life Expectancy (WHO)](https://www.who.int/)
- [Life Table (Wikipedia)](https://en.wikipedia.org/wiki/Life_table)

---
**Status:** Core solvency driver | **Complements:** Mortality Risk, Capital Requirements
