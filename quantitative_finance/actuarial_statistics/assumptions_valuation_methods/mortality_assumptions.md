# Mortality Assumptions

## 1. Concept Skeleton
**Definition:** Selection of age-specific mortality rates for pricing and reserving  
**Purpose:** Accurately model expected deaths and set appropriate premiums/reserves  
**Prerequisites:** Mortality tables, population characteristics, experience data

## 2. Comparative Framing
| Basis | Population Table | Company Experience | Adjusted Table |
|------|-----------------|-------------------|-----------------|
| **Source** | General population | Specific portfolio | Population + adjustments |
| **Accuracy** | Lower | Higher | Good balance |
| **Use Case** | Small portfolio | Mature portfolio | Most products |

## 3. Examples + Counterexamples

**Simple Example:**  
Use published SOA mortality table adjusted down 10% for portfolio experience

**Failure Case:**  
Using outdated table ignoring recent longevity improvements

**Edge Case:**  
Reverse adjustment for occupational or health-selected mortality

## 4. Layer Breakdown
```
Mortality Assumption Setting:
├─ Select base table
├─ Evaluate company experience (A/E analysis)
├─ Apply adjustments for population/selection
├─ Project future improvements
└─ Validate reasonableness
```

**Interaction:** Select table → analyze experience → adjust → project

## 5. Mini-Project
Compute adjustment factor from experience:
```python
observed_deaths = 95
expected_deaths = 100
adjustment = observed_deaths / expected_deaths
print("Adjustment factor:", adjustment)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring cohort effects in historical data
- Inadequate experience period for credibility
- Not adjusting for claim underwriting or selection

## 7. Key References
- [Life Table (Wikipedia)](https://en.wikipedia.org/wiki/Life_table)
- [SOA Mortality Studies](https://www.soa.org/)
- [Mortality Assumptions (IAA)](https://www.actuaries.org/)

---
**Status:** Core assumption layer | **Complements:** Interest Assumptions, Lapse Assumptions
