# Solvency & Capital Requirements

## 1. Concept Skeleton
**Definition:** Regulatory capital ensuring insurers can withstand adverse shocks  
**Purpose:** Protect policyholders and ensure financial stability  
**Prerequisites:** Risk measures, regulatory frameworks, balance-sheet concepts

## 2. Comparative Framing
| Regime | Solvency II | NAIC RBC | IFRS 17 |
|--------|------------|----------|---------|
| **Scope** | EU prudential | US statutory | Global accounting |
| **Capital** | SCR/MCR | RBC ratios | CSM/fulfillment |
| **Risk Basis** | Economic | Factor-based | Accounting view |

## 3. Examples + Counterexamples

**Simple Example:**  
RBC ratio below threshold triggers regulatory action

**Failure Case:**  
Using outdated correlation factors → undercapitalization

**Edge Case:**  
Group capital with cross-entity diversification limits

## 4. Layer Breakdown
```
Capital Requirement Flow:
├─ Identify risk modules (market, life, lapse)
├─ Apply shocks or factors
├─ Aggregate with correlations
├─ Compare to available capital
└─ Manage actions (reinsurance, capital raise)
```

**Interaction:** Identify risks → quantify → aggregate → manage

## 5. Mini-Project
Compute a simple diversified capital proxy:
```python
import numpy as np

risks = np.array([100, 80, 60])
cor = np.array([
    [1.0, 0.25, 0.1],
    [0.25, 1.0, 0.2],
    [0.1, 0.2, 1.0]
])

capital = np.sqrt(risks @ cor @ risks)
print("Diversified capital:", capital)
```

## 6. Challenge Round
Common pitfalls:
- Overreliance on standardized factors
- Ignoring non-linear aggregation effects
- Inconsistent definitions of available capital

## 7. Key References
- [Risk-Based Capital (NAIC)](https://www.naic.org/)
- [Solvency II (EIOPA)](https://www.eiopa.europa.eu/)
- [IFRS 17 (IFRS)](https://www.ifrs.org/)

---
**Status:** Regulatory core | **Complements:** Risk Modeling, Governance
