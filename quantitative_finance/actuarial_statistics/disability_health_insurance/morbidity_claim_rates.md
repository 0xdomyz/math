# Morbidity & Claim Rates

## 1. Concept Skeleton
**Definition:** Incidence of medical events (illnesses, injuries, conditions) triggering health claims  
**Purpose:** Price and reserve for health insurance products  
**Prerequisites:** Epidemiology, exposure data, claim classification

## 2. Comparative Framing
| Rate | Morbidity Incidence | Mortality | Disability Incidence |
|------|----------------------|-----------|----------------------|
| **Event** | Medical condition | Death | Loss of work capacity |
| **Severity** | Mild to severe | Terminal | Variable |
| **Use** | Health pricing | Life pricing | DI pricing |

## 3. Examples + Counterexamples

**Simple Example:**  
Flu incidence = 10% annually; hospitalization given flu = 1%

**Failure Case:**  
Using aggregate population morbidity for insured group (selection bias)

**Edge Case:**  
Rare conditions with high cost-per-case vs. common minor conditions

## 4. Layer Breakdown
```
Morbidity Modeling:
├─ Classification (ICD codes)
├─ Incidence rate by condition
├─ Severity/cost distribution
├─ Age/sex adjustment
└─ Duration and recovery
```

**Interaction:** Classify → estimate incidence → cost → aggregate

## 5. Mini-Project
Estimate health claim rate:
```python
import numpy as np

claims = 500
members = 10000
member_months = members * 12
claim_rate = claims / member_months
print("Claim rate per member-month:", claim_rate)
```

## 6. Challenge Round
Common pitfalls:
- Not adjusting for plan design incentives (deductible, copay effects)
- Using outdated epidemiology data
- Ignoring pandemic or environmental shifts in disease patterns

## 7. Key References
- [Morbidity (Wikipedia)](https://en.wikipedia.org/wiki/Morbidity)
- [Disease Incidence (WHO)](https://www.who.int/)
- [CDC Health Data](https://www.cdc.gov/)

---
**Status:** Health product pricing basis | **Complements:** All health insurance products
