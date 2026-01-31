# Recovery Rates

## 1. Concept Skeleton
**Definition:** Probability of returning to work from disability status  
**Purpose:** Forecast duration of disability benefits and reserve adequacy  
**Prerequisites:** Disability duration, rehabilitation success, occupational recovery

## 2. Comparative Framing
| Measure | Recovery Rate | Duration | Incidence |
|---------|----------------|----------|-----------|
| **Event** | Return to work | Months disabled | Become disabled |
| **Driver** | Rehab, time | Severity | Health, occupation |
| **Use** | Reserves | Benefit costs | Pricing |

## 3. Examples + Counterexamples

**Simple Example:**  
70% of disabled workers return to work within 2 years

**Failure Case:**  
Assuming same recovery rate across all disability types (back pain vs. cancer)

**Edge Case:**  
Partial return to work at reduced capacity not captured by binary recovery

## 4. Layer Breakdown
```
Recovery Modeling:
├─ Condition-specific recovery curves
├─ Duration-dependent rates
├─ Rehabilitation intervention effects
└─ Partial/modified work scenarios
```

**Interaction:** Model duration curves → adjust for interventions

## 5. Mini-Project
Estimate recovery rate by duration:
```python
import numpy as np

recovered = 200
still_disabled = 100
recovery_rate = recovered / (recovered + still_disabled)
print("Recovery rate:", recovery_rate)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring condition severity in recovery modeling
- Confusing partial return to work with full recovery
- Using aggregate rates for condition-specific pricing

## 7. Key References
- [Work Disability (Wikipedia)](https://en.wikipedia.org/wiki/Disability)
- [Rehabilitation (Wikipedia)](https://en.wikipedia.org/wiki/Rehabilitation_(people_with_disabilities))
- [SOA Recovery Studies](https://www.soa.org/)

---
**Status:** Benefit duration metric | **Complements:** Disability Rates, Long-Term Care
