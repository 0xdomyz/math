# Solvency II (EU)

## 1. Concept Skeleton
**Definition:** EU prudential regime with risk-based capital and governance requirements  
**Purpose:** Ensure insurer solvency through SCR/MCR and ORSA  
**Prerequisites:** Risk aggregation, capital concepts, regulatory governance

## 2. Comparative Framing
| Regime | Solvency II | NAIC RBC | IFRS 17 |
|--------|------------|----------|---------|
| **Capital** | SCR/MCR | RBC ratios | Not capital-focused |
| **Scope** | EU | US | Global accounting |
| **Method** | Standard formula or internal model | Factor-based | Reporting |

## 3. Examples + Counterexamples

**Simple Example:**  
SCR computed from market, life, health, and non-life modules

**Failure Case:**  
Treating accounting equity as solvency capital without adjustments

**Edge Case:**  
Use of internal models requiring regulatory approval

## 4. Layer Breakdown
```
Solvency II Structure:
├─ Pillar 1: Quantitative (SCR/MCR)
├─ Pillar 2: Governance and ORSA
├─ Pillar 3: Disclosure and reporting
└─ Internal model vs standard formula
```

**Interaction:** Quantify → govern → disclose

## 5. Mini-Project
Aggregate capital modules with correlation matrix:
```python
import numpy as np

modules = np.array([120, 90, 60])
cor = np.array([
    [1.0, 0.25, 0.1],
    [0.25, 1.0, 0.2],
    [0.1, 0.2, 1.0]
])
SCR = np.sqrt(modules @ cor @ modules)
print("SCR:", SCR)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring diversification limits
- Overreliance on standard formula
- Weak governance of model changes

## 7. Key References
- [Solvency II (EIOPA)](https://www.eiopa.europa.eu/)
- [Solvency II (Wikipedia)](https://en.wikipedia.org/wiki/Solvency_II)
- [ORSA (Wikipedia)](https://en.wikipedia.org/wiki/Own_risk_and_solvency_assessment)

---
**Status:** EU solvency regime | **Complements:** RBC, Risk Management
