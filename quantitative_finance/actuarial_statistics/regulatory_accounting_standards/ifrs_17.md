# IFRS 17 (Insurance Contracts)

## 1. Concept Skeleton
**Definition:** Global accounting standard for insurance contracts using fulfillment cash flows and CSM  
**Purpose:** Improve comparability and transparency across insurers  
**Prerequisites:** Present value, risk adjustment, financial reporting

## 2. Comparative Framing
| Framework | IFRS 17 | GAAP | Statutory |
|-----------|--------|------|-----------|
| **Measurement** | Fulfillment + CSM | Principle-based | Conservative |
| **Profit Pattern** | Over coverage period | Various | Not earnings-focused |
| **Discounting** | Current rates | Policy-based | Prescribed |

## 3. Examples + Counterexamples

**Simple Example:**  
CSM reduces as services are provided

**Failure Case:**  
Recognizing profit upfront at contract inception

**Edge Case:**  
VFA (variable fee approach) for participating contracts

## 4. Layer Breakdown
```
IFRS 17 Model:
├─ Fulfillment cash flows (PV + risk adjustment)
├─ Contractual service margin (CSM)
├─ Recognition and allocation to profit
├─ Onerous contracts and loss component
└─ Disclosures and reconciliations
```

**Interaction:** Measure → establish CSM → release over time → disclose

## 5. Mini-Project
Simple CSM release over coverage period:
```python
import numpy as np

csm = 120.0
years = 4
release = np.full(years, csm / years)
print(release)
```

## 6. Challenge Round
Common pitfalls:
- Misclassifying contract groups
- Incorrect risk adjustment methodology
- Ignoring discount rate updates

## 7. Key References
- [IFRS 17 (IFRS)](https://www.ifrs.org/)
- [IFRS 17 Overview (Wikipedia)](https://en.wikipedia.org/wiki/IFRS_17)
- [EIOPA IFRS 17](https://www.eiopa.europa.eu/)

---
**Status:** Global insurance standard | **Complements:** GAAP, Statutory Accounting
