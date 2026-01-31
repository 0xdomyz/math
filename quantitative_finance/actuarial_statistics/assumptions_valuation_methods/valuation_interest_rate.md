# Valuation Interest Rate

## 1. Concept Skeleton
**Definition:** Discount rate applied to liabilities in financial statements (statutory, GAAP, IFRS)  
**Purpose:** Determine liability values for balance sheet and earnings reporting  
**Prerequisites:** Liability duration, yield curves, regulatory guidance

## 2. Comparative Framing
| Framework | Statutory SAP | GAAP | IFRS 17 |
|-----------|--------------|------|---------|
| **Rate** | Prescribed or risk-free | Policy-specific or risk-free | Risk-adjusted |
| **Update** | Annual | Periodic unlock | Current rates |
| **Volatility** | Stable | Moderate | Higher |

## 3. Examples + Counterexamples

**Simple Example:**  
GAAP: use 10-year Treasury + 125bps for life insurance liabilities

**Failure Case:**  
Static valuation rate during volatile markets

**Edge Case:**  
Low/negative interest rates require special handling (floor rates)

## 4. Layer Breakdown
```
Valuation Interest Rate:
├─ Identify liability duration
├─ Select benchmark yield curve
├─ Add risk/illiquidity spread
├─ Apply discount to reserve
└─ Monitor and update
```

**Interaction:** Identify duration → select curve → discount

## 5. Mini-Project
Compute reserve change from rate movement:
```python
reserve = 1000000
duration = 7.5
rate_change = 0.01

reserve_change = -reserve * duration * rate_change
print("Reserve impact:", reserve_change)
```

## 6. Challenge Round
Common pitfalls:
- Confusing statutory, GAAP, and economic rates
- Not accounting for basis risk in rate selection
- Ignoring embedded options in rate sensitivity

## 7. Key References
- [Discount Rate (Wikipedia)](https://en.wikipedia.org/wiki/Discount_rate)
- [FASB Pension Accounting](https://www.fasb.org/)
- [IFRS 17 (IFRS)](https://www.ifrs.org/)

---
**Status:** Accounting measurement core | **Complements:** Liability Valuation, Earnings Reporting
