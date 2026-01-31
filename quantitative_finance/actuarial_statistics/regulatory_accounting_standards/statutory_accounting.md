# Statutory Accounting

## 1. Concept Skeleton
**Definition:** Insurance-focused accounting framework (e.g., NAIC SAP) emphasizing solvency and conservatism  
**Purpose:** Protect policyholders via prudent asset/liability valuation  
**Prerequisites:** Balance sheet concepts, reserves, regulatory reporting

## 2. Comparative Framing
| Framework | Statutory Accounting | GAAP | IFRS 17 |
|-----------|----------------------|------|---------|
| **Focus** | Solvency, conservatism | Earnings, comparability | Contractual service margin |
| **Valuation** | Conservative | Principle-based | Fulfillment + CSM |
| **Users** | Regulators | Investors | Global stakeholders |

## 3. Examples + Counterexamples

**Simple Example:**  
Admitted assets exclude certain intangibles to protect solvency

**Failure Case:**  
Applying GAAP capitalization rules in statutory reports

**Edge Case:**  
Captive insurers with modified statutory rules

## 4. Layer Breakdown
```
Statutory Accounting Flow:
├─ Admitted vs non-admitted assets
├─ Statutory reserves and liabilities
├─ Conservatism in valuation
├─ Regulatory filings and audits
└─ Solvency monitoring (RBC)
```

**Interaction:** Classify assets → value liabilities → report → monitor solvency

## 5. Mini-Project
Compare admitted vs non-admitted asset totals:
```python
import numpy as np

assets = np.array([100, 50, 20])
admitted = np.array([1, 1, 0])  # 0 = non-admitted
stat_assets = (assets * admitted).sum()
print("Statutory admitted assets:", stat_assets)
```

## 6. Challenge Round
Common pitfalls:
- Confusing statutory conservatism with economic value
- Overlooking jurisdictional SAP differences
- Misclassifying non-admitted assets

## 7. Key References
- [NAIC SAP Handbook](https://www.naic.org/)
- [Statutory Accounting Principles (Wikipedia)](https://en.wikipedia.org/wiki/Statutory_accounting_principles)
- [NAIC Accounting & Reporting](https://content.naic.org/)

---
**Status:** Regulatory accounting core | **Complements:** RBC, Solvency II
