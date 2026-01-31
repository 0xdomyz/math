# GAAP (US Generally Accepted)

## 1. Concept Skeleton
**Definition:** US GAAP accounting for insurance (e.g., ASC 944) focusing on earnings and comparability  
**Purpose:** Financial reporting for investors and stakeholders  
**Prerequisites:** Financial statements, revenue recognition, reserves

## 2. Comparative Framing
| Framework | GAAP | Statutory | IFRS 17 |
|-----------|------|-----------|---------|
| **Focus** | Earnings | Solvency | CSM + fulfillment |
| **Measurement** | Principle-based | Conservative | Current estimate |
| **Disclosure** | Extensive | Regulatory | Extensive |

## 3. Examples + Counterexamples

**Simple Example:**  
Deferred acquisition costs (DAC) recognized and amortized

**Failure Case:**  
Treating statutory reserves as GAAP liabilities without adjustment

**Edge Case:**  
Long-duration contract (LDTI) updates assumptions

## 4. Layer Breakdown
```
GAAP Insurance Reporting:
├─ Premium and benefit recognition
├─ DAC capitalization and amortization
├─ Liability measurement and unlocking
├─ Assumption updates (LDTI)
└─ Disclosure and segment reporting
```

**Interaction:** Recognize → measure → update → disclose

## 5. Mini-Project
Illustrate DAC amortization schedule:
```python
import numpy as np

D = 100.0
years = 5
amort = D / years
schedule = np.full(years, amort)
print(schedule)
```

## 6. Challenge Round
Common pitfalls:
- Mixing GAAP and statutory measures in KPIs
- Ignoring assumption unlocking requirements
- Underestimating disclosure complexity

## 7. Key References
- [FASB ASC 944](https://www.fasb.org/)
- [US GAAP (Wikipedia)](https://en.wikipedia.org/wiki/United_States_GAAP)
- [LDTI Overview (PwC)](https://www.pwc.com/)

---
**Status:** Investor reporting core | **Complements:** Statutory Accounting, IFRS 17
