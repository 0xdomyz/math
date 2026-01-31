# Age Structure

## 1. Concept Skeleton
**Definition:** Distribution of population by age; often summarized via dependency ratios and median age  
**Purpose:** Understand workforce capacity and social support burden  
**Prerequisites:** Age-specific counts, total population

## 2. Comparative Framing
| Metric | Dependency Ratio | Median Age | Age Distribution |
|--------|-----------------|-----------|-------------------|
| **Type** | Ratio | Percentile | Full distribution |
| **Use** | Policy planning | Summary | Detailed analysis |
| **Sensitivity** | Moderate | Moderate | High |

## 3. Examples + Counterexamples

**Simple Example:**  
Old-age dependency ratio = 65+/15-64; high ratio strains pension systems

**Failure Case:**  
Ignoring working-age immigrant influx when projecting dependency

**Edge Case:**  
Inverted pyramid: more elderly than youth (Japan, Europe)

## 4. Layer Breakdown
```
Age Structure Analysis:
├─ Count by age group
├─ Compute working-age population
├─ Calculate dependency ratio
├─ Track median age
└─ Project structure forward
```

**Interaction:** Compute → analyze → project impacts

## 5. Mini-Project
Calculate dependency ratios:
```python
import numpy as np

pop_0_14 = 30000
pop_15_64 = 100000
pop_65_plus = 20000

old_age_dr = pop_65_plus / pop_15_64
youth_dr = pop_0_14 / pop_15_64
total_dr = (pop_0_14 + pop_65_plus) / pop_15_64
print("Old-age:", old_age_dr, "Youth:", youth_dr, "Total:", total_dr)
```

## 6. Challenge Round
Common pitfalls:
- Static age structure assumptions over decades
- Not adjusting for mortality/morbidity improvements
- Ignoring educational attainment in workforce assessments

## 7. Key References
- [Population Pyramid (Wikipedia)](https://en.wikipedia.org/wiki/Population_pyramid)
- [Dependency Ratio (Wikipedia)](https://en.wikipedia.org/wiki/Dependency_ratio)
- [UN Age Structure Data](https://population.un.org/)

---
**Status:** Policy planning metric | **Complements:** Fertility, Longevity Trends
