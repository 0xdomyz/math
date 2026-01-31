# Population Growth Models

## 1. Concept Skeleton
**Definition:** Mathematical models describing population change via birth, death, and migration rates  
**Purpose:** Forecast demographic trends and plan social/economic policies  
**Prerequisites:** Differential equations, matrix methods, basic demography

## 2. Comparative Framing
| Model | Exponential | Logistic | Leslie Matrix |
|-------|------------|----------|---------------|
| **Growth** | Unbounded | Bounded | Age-structured |
| **Complexity** | Simple | Moderate | Complex |
| **Use** | Quick estimates | Resource constraints | Detailed projections |

## 3. Examples + Counterexamples

**Simple Example:**  
$P(t)=P_0 e^{rt}$ with constant growth rate

**Failure Case:**  
Exponential model for century-long projections ignores resource limits

**Edge Case:**  
Negative growth in developed nations requires logistic or multi-stage model

## 4. Layer Breakdown
```
Growth Model Workflow:
├─ Exponential: dP/dt = rP
├─ Logistic: dP/dt = r P(1 - P/K)
├─ Leslie: P_{t+1} = L P_t (age-structured)
└─ Calibrate and project
```

**Interaction:** Choose model → calibrate parameters → project

## 5. Mini-Project
Simulate logistic population growth:
```python
import numpy as np

P = 1000
r = 0.05
K = 10000
years = 50
trajectory = [P]

for _ in range(years):
    P = P + r * P * (1 - P / K)
    trajectory.append(P)

print(trajectory[-1])
```

## 6. Challenge Round
Common pitfalls:
- Extrapolating short-term trends without regime changes
- Ignoring age structure in Leslie models
- Overstating precision of long-term forecasts

## 7. Key References
- [Population Model (Wikipedia)](https://en.wikipedia.org/wiki/Population_model)
- [Exponential Growth (Wikipedia)](https://en.wikipedia.org/wiki/Exponential_growth)
- [UN Population Projections](https://population.un.org/)

---
**Status:** Core demographic tool | **Complements:** Fertility Rates, Migration
