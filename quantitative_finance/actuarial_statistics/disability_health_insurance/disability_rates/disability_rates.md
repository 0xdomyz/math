# Disability Rates

## 1. Concept Skeleton
**Definition:** Probability $d_x$ of becoming disabled during a year at age $x$  
**Purpose:** Price disability insurance and forecast disability benefit costs  
**Prerequisites:** Incidence rates, age effects, occupational risk

## 2. Comparative Framing
| Rate | Disability Incidence | Recovery Rate | Mortality Rate |
|------|----------------------|----------------|----------------|
| **Event** | Becomes disabled | Returns to work | Death |
| **Driver** | Health, occupation | Severity, rehab | Age, health |
| **Use** | DI pricing | Reserves | Life pricing |

## 3. Examples + Counterexamples

**Simple Example:**  
Disability rate age 35 = 0.005 (0.5% chance of disability that year)

**Failure Case:**  
Using disability rates from healthy population for high-risk occupation

**Edge Case:**  
Occupational disability (back injury in construction) vs. general population

## 4. Layer Breakdown
```
Disability Rate Modeling:
├─ Segment by age and occupation
├─ Estimate incidence rates
├─ Adjust for experience
└─ Apply to benefit calculations
```

**Interaction:** Segment → estimate → adjust → apply

## 5. Mini-Project
Estimate disability incidence:
```python
import numpy as np

disabled = 50
total_exposed = 10000
disability_rate = disabled / total_exposed
print("Disability rate:", disability_rate)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring reporting lags and "hidden" disabilities
- Mixing occupational with non-occupational rates
- Not adjusting for age/sex standardization

## 7. Key References
- [Disability Insurance (Wikipedia)](https://en.wikipedia.org/wiki/Disability_insurance)
- [Disability Rate (CDC)](https://www.cdc.gov/)
- [SOA Disability Studies](https://www.soa.org/)

---
**Status:** Core underwriting metric | **Complements:** Recovery Rates, Long-Term Care
