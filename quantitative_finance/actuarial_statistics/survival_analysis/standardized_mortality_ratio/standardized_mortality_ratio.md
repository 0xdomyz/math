# Standardized Mortality Ratio (SMR)

## 1. Concept Skeleton
**Definition:** Ratio of observed to expected deaths: $\text{SMR}=O/E$  
**Purpose:** Compare mortality of a cohort to a reference population  
**Prerequisites:** Expected deaths, exposure, reference rates

## 2. Comparative Framing
| Measure | SMR | Crude Death Rate | Relative Risk |
|---------|-----|------------------|---------------|
| **Baseline** | Reference population | None | Exposed vs unexposed |
| **Output** | Ratio | Rate | Ratio |
| **Use** | Benchmarking | Descriptive | Epidemiology |

## 3. Examples + Counterexamples

**Simple Example:**  
Observed 120 deaths vs expected 100 → SMR = 1.2

**Failure Case:**  
Using mismatched reference table by age/sex

**Edge Case:**  
Small expected deaths → unstable SMR

## 4. Layer Breakdown
```
SMR Calculation:
├─ Compute expected deaths from reference rates
├─ Count observed deaths
├─ SMR = O / E
└─ Add confidence intervals if needed
```

**Interaction:** Reference rates → expected → ratio → interpret

## 5. Mini-Project
Compute SMR from counts:
```python
observed = 120
expected = 100
smr = observed / expected
print("SMR:", smr)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring age standardization
- Comparing SMRs from different reference tables
- Overinterpreting SMR without confidence intervals

## 7. Key References
- [Standardized Mortality Ratio (Wikipedia)](https://en.wikipedia.org/wiki/Standardized_mortality_ratio)
- [Life Table (Wikipedia)](https://en.wikipedia.org/wiki/Life_table)
- [WHO Mortality Statistics](https://www.who.int/)

---
**Status:** Benchmarking metric | **Complements:** Graduation, Life Tables
