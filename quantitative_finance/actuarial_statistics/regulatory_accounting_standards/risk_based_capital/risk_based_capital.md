# Risk-Based Capital (RBC)

## 1. Concept Skeleton
**Definition:** NAIC formula-based capital requirement covering key insurance risks  
**Purpose:** Set minimum capital levels and regulatory action thresholds  
**Prerequisites:** Risk charges, balance sheet, statutory accounting

## 2. Comparative Framing
| Regime | NAIC RBC | Solvency II | GAAP |
|--------|----------|-------------|------|
| **Method** | Factor-based | Risk module aggregation | Not capital-focused |
| **Output** | RBC ratio | SCR/MCR | Earnings |
| **Scope** | US | EU | US reporting |

## 3. Examples + Counterexamples

**Simple Example:**  
RBC ratio below 200% triggers regulatory action level

**Failure Case:**  
Using RBC factors for non-US jurisdictions

**Edge Case:**  
Company action level with rapid capital remediation

## 4. Layer Breakdown
```
RBC Framework:
├─ C1 Asset risk
├─ C2 Insurance risk
├─ C3 Interest rate risk
├─ C4 Business risk
└─ Aggregate and compare to total adjusted capital
```

**Interaction:** Apply factors → aggregate → compare → action level

## 5. Mini-Project
Compute a simple RBC ratio:
```python
import numpy as np

TAC = 300.0
RBC = 150.0
ratio = TAC / RBC
print("RBC ratio:", ratio)
```

## 6. Challenge Round
Common pitfalls:
- Treating RBC as economic capital
- Misclassifying risks into C1–C4
- Ignoring trend test triggers

## 7. Key References
- [NAIC RBC](https://www.naic.org/)
- [Risk-Based Capital (Wikipedia)](https://en.wikipedia.org/wiki/Risk-based_capital)
- [NAIC RBC Instructions](https://content.naic.org/)

---
**Status:** US solvency metric | **Complements:** Statutory Accounting, Solvency II
