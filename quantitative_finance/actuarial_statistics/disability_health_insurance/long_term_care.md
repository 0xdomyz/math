# Long-Term Care

## 1. Concept Skeleton
**Definition:** Insurance covering extended nursing, assisted living, or in-home care costs  
**Purpose:** Mitigate catastrophic care costs in advanced age  
**Prerequisites:** Care transition probabilities, cost inflation, duration modeling

## 2. Comparative Framing
| Product | Long-Term Care | Disability Insurance | Critical Illness |
|---------|----------------|------------------|-----------------|
| **Trigger** | Care need (ADL/cognitive) | Income disability | Diagnosis |
| **Benefit** | Daily/monthly | Monthly income | Lump sum |
| **Duration** | Potentially lifetime | Typically 2 years | N/A |

## 3. Examples + Counterexamples

**Simple Example:**  
LTC daily benefit of $150 for assisted living until death or policy limit

**Failure Case:**  
Underestimating care inflation (5%+ annually) and extreme-age costs

**Edge Case:**  
Multi-state model: independent → assisted → nursing → death

## 4. Layer Breakdown
```
LTC Modeling:
├─ Activities of Daily Living (ADLs) dependencies
├─ Transition probabilities between care levels
├─ Cost escalation by care type
├─ Duration until death or policy end
└─ Underwriting and claim management
```

**Interaction:** Assess need → transition → pay benefits → monitor costs

## 5. Mini-Project
Model simple care transition:
```python
import numpy as np

prob_indep_to_assist = 0.05
prob_assist_to_nursing = 0.10
prob_die = 0.02

prob_stay = 1 - prob_indep_to_assist - prob_die
print("Transition probs:", prob_stay, prob_indep_to_assist, prob_die)
```

## 6. Challenge Round
Common pitfalls:
- Severely underestimating very-long durations (20+ years in care)
- Ignoring Medicaid coordination and spend-down effects
- Failing to project care costs inflation

## 7. Key References
- [Long-Term Care Insurance (Wikipedia)](https://en.wikipedia.org/wiki/Long-term_care_insurance)
- [Nursing Home (Wikipedia)](https://en.wikipedia.org/wiki/Nursing_home)
- [SOA LTC Studies](https://www.soa.org/)

---
**Status:** Catastrophic coverage product | **Complements:** Disability Insurance, Morbidity
