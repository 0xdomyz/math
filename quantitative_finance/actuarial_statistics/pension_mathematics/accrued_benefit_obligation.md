# Accrued Benefit Obligation (ABO)

## 1. Concept Skeleton
**Definition:** Liability for benefits earned to date at current salary levels  
**Purpose:** Conservative reserve measure for accounting and reporting  
**Prerequisites:** Benefit formula, service history, salary data

## 2. Comparative Framing
| Liability | ABO | PBO | Service Cost |
|-----------|-----|-----|--------------|
| **Salary** | Current | Projected | Current |
| **Future Service** | No | No | Yes |
| **Conservative** | Yes | No | N/A |

## 3. Examples + Counterexamples

**Simple Example:**  
ABO = 1.5% × current salary × past service years

**Failure Case:**  
Using ABO as if no salary growth will occur

**Edge Case:**  
Shallow vesting; many employees don't meet ABO threshold

## 4. Layer Breakdown
```
ABO Calculation:
├─ Identify vested benefits
├─ Use current salary rates
├─ Apply mortality and discount
└─ Sum across membership
```

**Interaction:** Identify vested → apply current rates → discount

## 5. Mini-Project
Compute simple ABO:
```python
salary = 60000
years_service = 8
vesting = 0.015
pv_factor = 1 / 1.04**10
abo = salary * years_service * vesting * pv_factor
print("ABO:", abo)
```

## 6. Challenge Round
Common pitfalls:
- Confusing ABO with PBO
- Ignoring vesting constraints
- Using wrong discount rate

## 7. Key References
- [ABO (FASB)](https://www.fasb.org/)
- [Pension Accounting (Wikipedia)](https://en.wikipedia.org/wiki/Pension_accounting)
- [SOA Pension Valuations](https://www.soa.org/)

---
**Status:** Accounting liability measure | **Complements:** PBO, Service Cost
