# Actuarial Cost Methods

## 1. Concept Skeleton
**Definition:** Techniques to allocate pension liability into annual costs across service period  
**Purpose:** Determine employer contributions and smooth funding volatility  
**Prerequisites:** Benefit formula, actuarial liability, valuation concepts

## 2. Comparative Framing
| Method | Entry Age Normal | Frozen Attained Age | Unit Credit |
|--------|-----------------|-----------------|------------|
| **Service Period** | Full career | From valuation | Year-by-year |
| **Cost Level** | Constant % of pay | Increases with age | Increasing |
| **Use** | Young plans | Mature plans | Smaller plans |

## 3. Examples + Counterexamples

**Simple Example:**  
Entry Age Normal spreads cost as constant % of salary over career

**Failure Case:**  
Using Unit Credit for mature plan → unaffordable early costs

**Edge Case:**  
Retroactive plan amendments require adjustment methods

## 4. Layer Breakdown
```
Cost Method Steps:
├─ Measure liability (past + future)
├─ Choose allocation pattern (level, increasing, etc.)
├─ Compute annual normal cost
├─ Amortize unfunded portion
└─ Monitor experience and adjust
```

**Interaction:** Measure → allocate → cost → amortize

## 5. Mini-Project
Illustrate Entry Age Normal cost:
```python
pbos = [50000, 60000, 70000]
salary = [40000, 50000, 60000]
costs = []
for pbo, sal in zip(pbos, salary):
    cost_pct = pbo / sal
    costs.append(cost_pct)
print(costs)
```

## 6. Challenge Round
Common pitfalls:
- Switching methods without proper transition
- Using methods inconsistent with plan demographics
- Not adjusting for actual experience gains/losses

## 7. Key References
- [Actuarial Cost Method (Wikipedia)](https://en.wikipedia.org/wiki/Actuarial_cost_method)
- [SOA Cost Methods](https://www.soa.org/)
- [FASB Pension Accounting](https://www.fasb.org/)

---
**Status:** Core costing framework | **Complements:** Pension Funding, Liability Valuation
