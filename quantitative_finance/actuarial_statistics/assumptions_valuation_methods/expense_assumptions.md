# Expense Assumptions

## 1. Concept Skeleton
**Definition:** Assumptions on administrative costs, commissions, and acquisition expenses  
**Purpose:** Incorporate non-mortality costs into pricing and reserving  
**Prerequisites:** Expense data, allocation methods, inflation projections

## 2. Comparative Framing
| Expense Type | Per-Policy | Percentage Premium | Per-Claim |
|-------------|-----------|-------------------|-----------|
| **Base** | Fixed amount | % of premium | Per claim processed |
| **Inflation** | Inflation-adjusted | Typically flat % | Variable |
| **Use Case** | Fixed costs | Commission-type | Claims-intensive |

## 3. Examples + Counterexamples

**Simple Example:**  
$50 per-policy annual fee + 3% of premium for distribution

**Failure Case:**  
Using gross commissions instead of net brokerage fees

**Edge Case:**  
First-year acquisition costs significantly higher than renewal

## 4. Layer Breakdown
```
Expense Assumption Modeling:
├─ Acquisition expenses (first-year)
├─ Maintenance expenses (renewal)
├─ Claims processing costs
├─ Allocate fixed and variable
└─ Project inflation
```

**Interaction:** Categorize → allocate → project → validate

## 5. Mini-Project
Compute total expenses over policy life:
```python
premium = 1000
renewal_years = 9
acquisition = 200
renewal_exp = 50 * (1.03 ** np.arange(renewal_years))

total = acquisition + renewal_exp.sum()
print("Total expenses:", total)
```

## 6. Challenge Round
Common pitfalls:
- Double-counting expenses via margins vs. explicit assumptions
- Not adjusting for scale in distributed cost assumptions
- Ignoring expense inflation in long-duration products

## 7. Key References
- [Operating Expense (Wikipedia)](https://en.wikipedia.org/wiki/Operating_expense)
- [Insurance Cost (SOA)](https://www.soa.org/)
- [Expense Management (IAA)](https://www.actuaries.org/)

---
**Status:** Cost assumption layer | **Complements:** Pricing, Profitability Testing
