# Projected Benefit Obligation (PBO)

## 1. Concept Skeleton
**Definition:** Liability for benefits including future salary growth and service  
**Purpose:** Economic measure of pension obligation for financial reporting  
**Prerequisites:** Benefit formula, salary progression, discount rates

## 2. Comparative Framing
| Liability | PBO | ABO | Unfunded Liability |
|-----------|-----|-----|-------------------|
| **Salary** | Projected | Current | Actual - funded |
| **Economic** | Yes | Conservative | Actual |
| **Use** | Accounting | Regulatory | Funding |

## 3. Examples + Counterexamples

**Simple Example:**  
PBO includes assumed 3% annual salary raises over career

**Failure Case:**  
Using static salary projection in volatile economic environment

**Edge Case:**  
Very low discount rates amplify PBO in low-rate regimes

## 4. Layer Breakdown
```
PBO Calculation:
├─ Project salary growth
├─ Include future service
├─ Estimate mortality/decrements
├─ Discount to present value
└─ Aggregate across members
```

**Interaction:** Project salary → include future service → discount

## 5. Mini-Project
Project salary and compute PBO:
```python
current_salary = 50000
salary_growth = 0.03
years_to_retirement = 15
final_salary = current_salary * (1 + salary_growth) ** years_to_retirement
benefit = final_salary * 0.015 * (8 + years_to_retirement)
pv = benefit / (1.04 ** years_to_retirement)
print("PBO:", pv)
```

## 6. Challenge Round
Common pitfalls:
- Overestimating salary growth during slowdowns
- Ignoring embedded options (early retirement)
- Discount rate sensitivity in low-rate environments

## 7. Key References
- [PBO (FASB)](https://www.fasb.org/)
- [Pension Obligations (Wikipedia)](https://en.wikipedia.org/wiki/Pension_accounting)
- [SOA Pension Valuations](https://www.soa.org/)

---
**Status:** Economic liability measure | **Complements:** ABO, Funding Methods
