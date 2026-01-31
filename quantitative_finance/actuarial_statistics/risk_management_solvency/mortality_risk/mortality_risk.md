# Mortality Risk

## 1. Concept Skeleton
**Definition:** Risk of higher-than-expected deaths causing claim spikes  
**Purpose:** Price protection products and set capital for adverse mortality shocks  
**Prerequisites:** Mortality tables, claim modeling, discounting

## 2. Comparative Framing
| Risk Type | Mortality Risk | Longevity Risk | Catastrophe Risk |
|----------|----------------|---------------|------------------|
| **Direction** | Higher deaths | Longer lives | Extreme event spikes |
| **Products** | Life insurance | Annuities | Life, health, P&C |
| **Drivers** | Health trends | Medical progress | Pandemics, disasters |

## 3. Examples + Counterexamples

**Simple Example:**  
Excess deaths during influenza season increase term claims

**Failure Case:**  
Ignoring correlation across insured lives in catastrophe events

**Edge Case:**  
Small portfolios with high variance in deaths

## 4. Layer Breakdown
```
Mortality Risk Assessment:
├─ Baseline mortality assumptions
├─ Shock/scenario design (e.g., +15% deaths)
├─ Portfolio aggregation and diversification
├─ Reinsurance or capital mitigation
└─ Monitor actual vs expected (A/E) ratios
```

**Interaction:** Assumptions → shocks → aggregation → mitigation

## 5. Mini-Project
Compute simple A/E mortality ratio:
```python
import numpy as np

actual = np.array([12, 15, 10, 14])
expected = np.array([10, 12, 11, 13])
ratio = actual.sum() / expected.sum()
print("A/E ratio:", ratio)
```

## 6. Challenge Round
Common pitfalls:
- Overreliance on short-term experience
- Ignoring catastrophe correlations
- Using inappropriate exposure measures

## 7. Key References
- [Mortality Risk (SOA)](https://www.soa.org/)
- [Mortality Table (Wikipedia)](https://en.wikipedia.org/wiki/Life_table)
- [Excess Mortality (Wikipedia)](https://en.wikipedia.org/wiki/Excess_mortality)

---
**Status:** Core underwriting risk | **Complements:** Longevity Risk, Reinsurance
