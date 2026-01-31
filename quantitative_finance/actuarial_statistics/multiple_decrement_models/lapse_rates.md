# Lapse Rates

## 1. Concept Skeleton
**Definition:** Probability of policy termination (lapse) within a period  
**Purpose:** Model policyholder behavior and pricing profitability  
**Prerequisites:** Policy exposure, cash flows, survival probabilities

## 2. Comparative Framing
| Measure | Lapse Rate | Surrender Rate | Mortality Rate |
|--------|------------|----------------|----------------|
| **Event** | Termination | Termination with value | Death |
| **Driver** | Behavior | Behavior + value | Health |
| **Impact** | Profit emergence | Liquidity cost | Claims |

## 3. Examples + Counterexamples

**Simple Example:**  
Higher lapses when credited rate is below market

**Failure Case:**  
Assuming constant lapse in volatile economic regimes

**Edge Case:**  
Antiselection: healthier policyholders lapse more

## 4. Layer Breakdown
```
Lapse Modeling:
├─ Base lapse curve by duration
├─ Segmentation (age, product, channel)
├─ Dynamic drivers (rates, equity, unemployment)
└─ Stress and sensitivity tests
```

**Interaction:** Build curve → segment → stress → apply to cash flows

## 5. Mini-Project
Apply a dynamic lapse shock:
```python
import numpy as np

base = np.array([0.03, 0.025, 0.02])
shock = 0.01
lapse = base + shock
print(lapse)
```

## 6. Challenge Round
Common pitfalls:
- Using industry averages for niche products
- Ignoring policyholder incentives
- Double-counting lapse with surrender events

## 7. Key References
- [Lapse (Insurance) (Wikipedia)](https://en.wikipedia.org/wiki/Lapse_(insurance))
- [Surrender Value (Wikipedia)](https://en.wikipedia.org/wiki/Surrender_value)
- [SOA Lapse Studies](https://www.soa.org/)

---
**Status:** Behavior-driven decrement | **Complements:** Surrender Assumptions, Profit Testing
