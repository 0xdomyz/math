# Defined Contribution Plans

## 1. Concept Skeleton
**Definition:** Pension plan where employer contributes fixed amount; member bears investment risk  
**Purpose:** Simplify employer obligations and provide member-directed savings  
**Prerequisites:** Account values, investment returns, member behavior

## 2. Comparative Framing
| Plan Type | Defined Contribution | Defined Benefit | Hybrid |
|-----------|----------------------|-----------------|--------|
| **Contribution** | Fixed | Variable with liability | Combination |
| **Benefit** | Account balance | Guaranteed amount | Mix |
| **Volatility** | Low for employer | High | Moderate |

## 3. Examples + Counterexamples

**Simple Example:**  
Employer contributes 5% of salary; member invests and receives account balance

**Failure Case:**  
Assuming constant investment returns over long periods

**Edge Case:**  
Target date funds with automatic rebalancing

## 4. Layer Breakdown
```
DC Plan Components:
├─ Contribution rate and schedule
├─ Investment options and default
├─ Account accumulation
├─ Withdrawal and payout rules
└─ Member communications and choices
```

**Interaction:** Contribute → invest → accumulate → withdraw

## 5. Mini-Project
Project DC account balance:
```python
import numpy as np

balance = 10000
annual_contribution = 5000
return_rate = 0.06
years = 20

for _ in range(years):
    balance = balance * (1 + return_rate) + annual_contribution

print("Final balance:", balance)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring investment volatility and sequence risk
- Assuming consistent participation rates
- Underestimating inflation impact on retirement security

## 7. Key References
- [Defined Contribution Plan (Wikipedia)](https://en.wikipedia.org/wiki/Defined_contribution_plan)
- [401(k) Plan (Wikipedia)](https://en.wikipedia.org/wiki/401(k)_plan)
- [SOA DC Plans](https://www.soa.org/)

---
**Status:** Prevalent plan type | **Complements:** Defined Benefit, Retirement Security
