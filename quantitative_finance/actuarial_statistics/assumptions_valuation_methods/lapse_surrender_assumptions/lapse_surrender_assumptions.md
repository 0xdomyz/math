# Lapse & Surrender Assumptions

## 1. Concept Skeleton
**Definition:** Assumptions on policyholder termination and cash-out behavior  
**Purpose:** Forecast policy persistence and cash flows  
**Prerequisites:** Lapse data, product design, market factors

## 2. Comparative Framing
| Rate Type | Static Lapse | Dynamic Lapse | Antiselection |
|-----------|-------------|--------------|--------------|
| **Changes** | Fixed | Varies by rate/market | Risk-based |
| **Use Case** | Simple models | Realistic | Advanced pricing |
| **Validation** | Historical | Scenario-based | Behavioral |

## 3. Examples + Counterexamples

**Simple Example:**  
Year 1: 10% lapse, Year 2+: 5% lapse (decreasing by duration)

**Failure Case:**  
Assuming static lapse during interest rate shock

**Edge Case:**  
Antiselection where healthy policyholders lapse at higher rates

## 4. Layer Breakdown
```
Lapse Assumption Setting:
├─ Base lapse curve by duration
├─ Segment by product/channel
├─ Model dynamic adjustments
├─ Validate with industry/company data
└─ Stress test sensitivity
```

**Interaction:** Develop curve → segment → validate → stress

## 5. Mini-Project
Model dynamic lapse response:
```python
base_lapse = 0.05
rate_shock = 0.02  # 200bps increase
elasticity = -0.10  # 10% lapse increase per 100bps
adjusted_lapse = base_lapse * (1 + elasticity * rate_shock / 0.01)
print("Adjusted lapse:", adjusted_lapse)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring economic drivers of lapse
- Confusing policyholder behavior across products
- Overestimating benefit of increased crediting rates

## 7. Key References
- [Lapse Rate (Insurance) (Wikipedia)](https://en.wikipedia.org/wiki/Lapse_(insurance))
- [SOA Lapse Studies](https://www.soa.org/)
- [Behavioral Assumptions (IAA)](https://www.actuaries.org/)

---
**Status:** Persistence modeling foundation | **Complements:** Mortality, Interest Assumptions
