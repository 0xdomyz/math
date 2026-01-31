# Scenario Analysis

## 1. Concept Skeleton
**Definition:** Deterministic stress tests and shock scenarios for regulatory and risk management  
**Purpose:** Quantify impact of specific adverse events without full stochastic modeling  
**Prerequisites:** Key risk drivers, calibrated shocks, deterministic projection

## 2. Comparative Framing
| Approach | Scenario Analysis | Stochastic | Sensitivity Analysis |
|----------|------------------|-----------|---------------------|
| **Method** | Deterministic shocks | Random paths | One-at-a-time |
| **Use Case** | Stress testing | Capital modeling | Parameter impact |
| **Output** | Shock outcomes | Distributions | Elasticities |

## 3. Examples + Counterexamples

**Simple Example:**  
NAIC C3 Phase II: 7 interest rate scenarios + equity shocks

**Failure Case:**  
Ignoring interaction effects between multiple shocks

**Edge Case:**  
Extreme tail scenarios (1-in-200 year events)

## 4. Layer Breakdown
```
Scenario Analysis:
├─ Define shock scenarios (rates, equity, mortality)
├─ Project cash flows under each scenario
├─ Compute impacts (reserves, capital)
├─ Aggregate across scenarios
└─ Report worst-case and expected
```

**Interaction:** Define → project → compute → aggregate

## 5. Mini-Project
Stress test rate shock impact:
```python
import numpy as np

reserve_base = 1000000
duration = 7.0
rate_shock = 0.01  # 100bps increase

reserve_impact = -reserve_base * duration * rate_shock
stressed_reserve = reserve_base + reserve_impact
print("Stressed reserve:", stressed_reserve)
```

## 6. Challenge Round
Common pitfalls:
- Too few scenarios missing critical risks
- Not considering combined shocks
- Static scenarios not reflecting dynamic management actions

## 7. Key References
- [Stress Testing (Wikipedia)](https://en.wikipedia.org/wiki/Stress_testing_(financial))
- [NAIC C3 Phase II](https://www.naic.org/)
- [Scenario Analysis (SOA)](https://www.soa.org/)

---
**Status:** Regulatory compliance tool | **Complements:** Monte Carlo, Capital Requirements
