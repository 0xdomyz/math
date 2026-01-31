# Fertility Rates

## 1. Concept Skeleton
**Definition:** Age-specific fertility rate (ASFR); births per woman by age; Total Fertility Rate (TFR)  
**Purpose:** Forecast births and population age structure  
**Prerequisites:** Birth counts, female population, age intervals

## 2. Comparative Framing
| Rate | ASFR | TFR | Crude Birth Rate |
|------|------|-----|------------------|
| **Denominator** | Women by age | All women 15-49 | Total population |
| **Age-specific** | Yes | No (summary) | No |
| **Use** | Projections | Summary metric | Quick comparisons |

## 3. Examples + Counterexamples

**Simple Example:**  
TFR = 2.1 means average woman has 2.1 children (replacement level)

**Failure Case:**  
Using crude birth rate for age-adjusted population comparisons

**Edge Case:**  
Very low TFR (<1.5) requires migration assumptions for growth

## 4. Layer Breakdown
```
Fertility Modeling:
├─ Compute ASFR by age group
├─ Sum ASFR × 5 (for 5-year groups) to get TFR
├─ Project TFR by trend or scenario
└─ Apply to female cohorts
```

**Interaction:** Compute → trend → project

## 5. Mini-Project
Calculate simple TFR:
```python
import numpy as np

asfr = np.array([0.05, 0.12, 0.15, 0.08, 0.02])
tfr = asfr.sum() * 5  # 5-year age groups
print("TFR:", tfr)
```

## 6. Challenge Round
Common pitfalls:
- Confusing period and cohort fertility
- Ignoring tempo effects (shifting reproductive timing)
- Extrapolating short-term fertility changes too far

## 7. Key References
- [Fertility Rate (Wikipedia)](https://en.wikipedia.org/wiki/Fertility_rate)
- [Total Fertility Rate (Wikipedia)](https://en.wikipedia.org/wiki/Total_fertility_rate)
- [UN Fertility Data](https://population.un.org/)

---
**Status:** Birth forecasting metric | **Complements:** Population Growth, Migration
