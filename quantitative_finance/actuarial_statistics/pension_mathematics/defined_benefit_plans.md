# Defined Benefit Plans

## 1. Concept Skeleton
**Definition:** Pension plan guaranteeing specific benefit formula (e.g., % of salary or fixed amount)  
**Purpose:** Provide retirement security with employer bearing investment risk  
**Prerequisites:** Benefit formulas, discount rates, actuarial liability

## 2. Comparative Framing
| Plan Type | Defined Benefit | Defined Contribution | Hybrid |
|-----------|-----------------|----------------------|--------|
| **Risk** | Employer | Employee | Shared |
| **Liability** | Fixed promise | Account balance | Combination |
| **Volatility** | High for employer | None | Moderate |

## 3. Examples + Counterexamples

**Simple Example:**  
Benefit = 1.5% × final average salary × years of service

**Failure Case:**  
Assuming static life expectancy; ignores longevity improvements

**Edge Case:**  
Integrated plans with Social Security offset

## 4. Layer Breakdown
```
DB Valuation Workflow:
├─ Define benefit formula
├─ Project service and salary growth
├─ Estimate mortality and decrements
├─ Discount to present value
├─ Calculate annual funding contributions
└─ Monitor and update assumptions
```

**Interaction:** Formula → project → discount → fund

## 5. Mini-Project
Compute simple DB liability:
```python
import numpy as np

salary = 50000
years_service = 10
benefit_rate = 0.015
final_benefit = salary * benefit_rate * years_service
pv_factor = 1 / 1.03**15  # 15 years to retirement, 3% discount
liability = final_benefit * pv_factor
print("Liability:", liability)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring embedded options (early retirement, lump-sum)
- Using static salary scales
- Underestimating contribution volatility

## 7. Key References
- [Defined Benefit Plan (Wikipedia)](https://en.wikipedia.org/wiki/Defined_benefit_plan)
- [Pension Accounting (FASB)](https://www.fasb.org/)
- [SOA Pension Mathematics](https://www.soa.org/)

---
**Status:** Core pension product | **Complements:** Defined Contribution, Actuarial Funding
