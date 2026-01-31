# Interest Rate Assumptions

## 1. Concept Skeleton
**Definition:** Discount rates and investment return projections for liability valuation  
**Purpose:** Determine present value of liabilities and capital needs  
**Prerequisites:** Yield curves, duration matching, economic scenarios

## 2. Comparative Framing
| Rate Type | Risk-Free | Market-Consistent | Prescribed |
|-----------|-----------|-------------------|-----------|
| **Basis** | Government bonds | Market data | Regulatory |
| **Volatility** | Low | Medium | Stable |
| **Use Case** | Accounting | Economic capital | Statutory |

## 3. Examples + Counterexamples

**Simple Example:**  
Use 10-year Treasury yield + 100bps spread for liability discount

**Failure Case:**  
Static rate assumption in volatile or low-rate environment

**Edge Case:**  
Very long-duration liabilities with 30+ year horizon

## 4. Layer Breakdown
```
Interest Rate Assumption Workflow:
├─ Identify liability duration
├─ Select matching yield curve
├─ Apply risk premium if needed
├─ Project for stochastic scenarios
└─ Monitor updates
```

**Interaction:** Match duration → select rate → project → update

## 5. Mini-Project
Compute PV of cash flows with assumed rates:
```python
import numpy as np

cash_flows = np.array([1000, 1000, 1000, 1000])
discount_rate = 0.03
years = np.arange(1, 5)
pv = (cash_flows / (1 + discount_rate) ** years).sum()
print("PV:", pv)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring yield curve shape changes
- Not stress-testing rate sensitivity
- Using outdated historical averages

## 7. Key References
- [Interest Rate (Wikipedia)](https://en.wikipedia.org/wiki/Interest_rate)
- [Discount Rate (Wikipedia)](https://en.wikipedia.org/wiki/Discount_rate)
- [SOA Interest Assumptions](https://www.soa.org/)

---
**Status:** Core valuation parameter | **Complements:** Mortality Assumptions, Duration Matching
