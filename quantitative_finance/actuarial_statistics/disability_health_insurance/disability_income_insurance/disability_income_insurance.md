# Disability Income Insurance

## 1. Concept Skeleton
**Definition:** Product replacing lost earnings due to disability; features elimination period and benefit duration  
**Purpose:** Protect income during work disability  
**Prerequisites:** Disability rates, recovery rates, income models

## 2. Comparative Framing
| Insurance Type | Disability Income | Critical Illness | Long-Term Care |
|--------|------------------|------------------|----------------|
| **Trigger** | Disability (lost income) | Diagnosis | Care need |
| **Benefit** | Income replacement | Lump sum | Ongoing care |
| **Pricing** | Duration + amount | Incidence only | Duration + cost |

## 3. Examples + Counterexamples

**Simple Example:**  
DI benefit = 60% of salary, after 90-day elimination period, until age 65

**Failure Case:**  
Ignoring moral hazard; overly generous benefits reduce return to work

**Edge Case:**  
Definition of disability (unable to own occupation vs. any work)

## 4. Layer Breakdown
```
DI Product Modeling:
├─ Elimination period (waiting)
├─ Disability incidence
├─ Duration and recovery
├─ Benefit replacement ratio
└─ Offset/coordination with other benefits
```

**Interaction:** Wait → incur disability → receive benefits → recover

## 5. Mini-Project
Compute DI liability:
```python
salary = 60000
replacement = 0.60
disability_rate = 0.005
years = 2
pv_factor = 1 / 1.03
liability = salary * replacement * disability_rate * years * pv_factor
print("DI liability:", liability)
```

## 6. Challenge Round
Common pitfalls:
- Underestimating moral hazard effects
- Not adjusting for occupational differences
- Ignoring coordination with workers' comp or Social Security

## 7. Key References
- [Disability Insurance (Wikipedia)](https://en.wikipedia.org/wiki/Disability_insurance)
- [Income Protection (Wikipedia)](https://en.wikipedia.org/wiki/Income_protection_insurance)
- [SOA DI Studies](https://www.soa.org/)

---
**Status:** Income protection product | **Complements:** Disability Rates, Health Insurance
