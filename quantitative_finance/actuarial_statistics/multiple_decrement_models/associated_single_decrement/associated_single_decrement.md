# Associated Single Decrement

## 1. Concept Skeleton
**Definition:** Hypothetical single-decrement model equivalent to a cause in a multi-decrement setting  
**Purpose:** Analyze each decrement as if it were the only cause  
**Prerequisites:** Multiple decrement tables, survival probabilities

## 2. Comparative Framing
| Model | Associated Single Decrement | Multiple Decrement | Cause-Specific Rate |
|------|------------------------------|-------------------|---------------------|
| **Cause Interaction** | Removed | Present | Present |
| **Use** | Isolated analysis | Realistic exits | Attribution |
| **Output** | $q_x'$ | $q_x^{(j)}$ | $\mu_x^{(j)}$ |

## 3. Examples + Counterexamples

**Simple Example:**  
Single-decrement lapse table derived from multi-decrement data

**Failure Case:**  
Using associated single decrement as actual observed rate

**Edge Case:**  
Large competing risk makes single-decrement approximation misleading

## 4. Layer Breakdown
```
Associated Single Decrement Steps:
├─ Start from multi-decrement table
├─ Remove other causes
├─ Recompute q_x' for target cause
└─ Use in pricing or comparisons
```

**Interaction:** Isolate cause → recompute → analyze

## 5. Mini-Project
Transform multi-decrement cause into single decrement:
```python
import numpy as np

qx_total = 0.05
qx_cause = 0.02
qx_single = qx_cause / (1 - (qx_total - qx_cause))
print(qx_single)
```

## 6. Challenge Round
Common pitfalls:
- Failing to adjust for other causes properly
- Confusing probabilities with rates
- Applying outside the exposure basis

## 7. Key References
- [Multiple Decrement (SOA)](https://www.soa.org/)
- [Life Table (Wikipedia)](https://en.wikipedia.org/wiki/Life_table)
- [Competing Risks (Wikipedia)](https://en.wikipedia.org/wiki/Competing_risks)

---
**Status:** Analytical construct | **Complements:** Cause-Specific Mortality, Service Tables
