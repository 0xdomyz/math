# Migration

## 1. Concept Skeleton
**Definition:** Movement of population into (immigration) or out of (emigration) a region  
**Purpose:** Model demographic change and forecast population structure  
**Prerequisites:** Census data, population balancing, age-specific rates

## 2. Comparative Framing
| Flow | Immigration | Emigration | Net Migration |
|------|------------|-----------|----------------|
| **Direction** | In | Out | Net change |
| **Measure** | Inflow rate | Outflow rate | In - Out |
| **Impact** | Adds to population | Reduces population | Total effect |

## 3. Examples + Counterexamples

**Simple Example:**  
Net migration of +50,000 per year increases population independent of natural increase

**Failure Case:**  
Assuming constant migration despite economic or policy changes

**Edge Case:**  
Selective migration by age/education skews population structure

## 4. Layer Breakdown
```
Migration Modeling:
├─ Estimate age-specific migration rates
├─ Compute in/out flows
├─ Net migration = In - Out
└─ Apply to population cohorts
```

**Interaction:** Estimate rates → compute flows → apply to cohorts

## 5. Mini-Project
Add net migration to population:
```python
import numpy as np

population = 1000000
births = 20000
deaths = 15000
net_migration = 5000
new_pop = population + births - deaths + net_migration
print("New population:", new_pop)
```

## 6. Challenge Round
Common pitfalls:
- Overprojecting migration from short-term spikes
- Ignoring demographic selectivity of migrants
- Failing to account for policy changes (visa restrictions)

## 7. Key References
- [Human Migration (Wikipedia)](https://en.wikipedia.org/wiki/Human_migration)
- [Migration Rate (Wikipedia)](https://en.wikipedia.org/wiki/Migration_rate)
- [UN Migration Data](https://www.un.org/en/development/desa/population/migration/)

---
**Status:** Demographic change driver | **Complements:** Population Growth, Age Structure
