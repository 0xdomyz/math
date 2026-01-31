# Lapse Risk

## 1. Concept Skeleton
**Definition:** Risk that policyholders terminate or surrender policies unexpectedly  
**Purpose:** Manage profit emergence and expense recovery on long-term contracts  
**Prerequisites:** Policy cash flows, surrender values, behavioral assumptions

## 2. Comparative Framing
| Risk Type | Lapse Risk | Mortality Risk | Interest Rate Risk |
|----------|------------|----------------|--------------------|
| **Driver** | Policyholder behavior | Health outcomes | Market rates |
| **Impact** | Lost margins, expense strain | Claim spikes | ALM mismatch |
| **Mitigation** | Surrender charges, incentives | Reinsurance | Duration matching |

## 3. Examples + Counterexamples

**Simple Example:**  
Rates increase → customers lapse low-crediting policies

**Failure Case:**  
Assuming static lapse rates during economic stress

**Edge Case:**  
Antiselection: healthy policyholders lapse, unhealthy stay

## 4. Layer Breakdown
```
Lapse Risk Modeling:
├─ Baseline lapse curve by duration
├─ Dynamic drivers (rates, equity markets)
├─ Segment by policy features
├─ Stress scenarios (spike or drop)
└─ Impact on profit and reserves
```

**Interaction:** Lapse assumptions → stress → profit impact

## 5. Mini-Project
Apply a lapse shock to cashflow count:
```python
import numpy as np

inforce = np.array([1000, 980, 960])
base_lapse = np.array([0.02, 0.02, 0.02])
shock = 0.03
lapse = base_lapse + shock
survivors = inforce * (1 - lapse)
print(survivors)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring economic regime shifts
- Using industry averages for niche products
- Not modeling partial withdrawals

## 7. Key References
- [Policy Lapse (Wikipedia)](https://en.wikipedia.org/wiki/Lapse_(insurance))
- [Surrender Value (Wikipedia)](https://en.wikipedia.org/wiki/Surrender_value)
- [SOA Lapse Assumptions](https://www.soa.org/)

---
**Status:** Behavior-driven risk | **Complements:** Expense Assumptions, Profit Testing
