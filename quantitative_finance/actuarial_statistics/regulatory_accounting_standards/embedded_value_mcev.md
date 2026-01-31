# Embedded Value & MCEV

## 1. Concept Skeleton
**Definition:** Embedded value measures the present value of future profits from existing business; MCEV standardizes it  
**Purpose:** Value in-force business and compare insurers  
**Prerequisites:** Discounting, risk adjustment, cash-flow modeling

## 2. Comparative Framing
| Metric | Embedded Value | MCEV | Market Cap |
|--------|----------------|------|------------|
| **Scope** | In-force only | Standardized EV | Full firm value |
| **Risk Adjustment** | Varies | Required | Market-implied |
| **Use** | Disclosure | Comparability | Investor valuation |

## 3. Examples + Counterexamples

**Simple Example:**  
EV = PV of future profits + adjusted net asset value

**Failure Case:**  
Including new business profits in EV

**Edge Case:**  
Low interest rates inflate PV and EV volatility

## 4. Layer Breakdown
```
EV/MCEV Components:
├─ Adjusted net asset value (ANAV)
├─ Present value of future profits (PVFP)
├─ Cost of options and guarantees (COG)
├─ Frictional costs and risk margin
└─ Reconciliations and disclosures
```

**Interaction:** Project cash flows → adjust for risks → aggregate → disclose

## 5. Mini-Project
Compute a toy EV:
```python
ANAV = 200.0
PVFP = 150.0
COG = 20.0
EV = ANAV + PVFP - COG
print("EV:", EV)
```

## 6. Challenge Round
Common pitfalls:
- Mixing EV with IFRS 17 profit measures
- Ignoring option costs in long guarantees
- Overstating diversification benefits

## 7. Key References
- [MCEV Principles (EIOPA)](https://www.eiopa.europa.eu/)
- [Embedded Value (Wikipedia)](https://en.wikipedia.org/wiki/Embedded_value)
- [Insurance Embedded Value (SOA)](https://www.soa.org/)

---
**Status:** Value disclosure metric | **Complements:** IFRS 17, Solvency II
