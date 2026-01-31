# Longevity Risk Modeling

## 1. Concept Skeleton
**Definition:** Stochastic models for mortality improvements (Lee-Carter, Cairns-Blake-Dowd)  
**Purpose:** Forecast longevity trends and quantify longevity risk capital  
**Prerequisites:** Mortality tables, time series, principal components

## 2. Comparative Framing
| Model | Lee-Carter | Cairns-Blake-Dowd | Age-Period-Cohort |
|-------|-----------|-------------------|-------------------|
| **Structure** | Log-linear factor | Age + period factors | Age + period + cohort |
| **Complexity** | Moderate | Moderate | High |
| **Use Case** | General trends | Tail modeling | Detailed cohorts |

## 3. Examples + Counterexamples

**Simple Example:**  
Lee-Carter: $\ln(m_{x,t}) = a_x + b_x k_t$ with $k_t$ random walk

**Failure Case:**  
Ignoring cohort effects in generations with different health trends

**Edge Case:**  
Sudden mortality shocks (pandemics) not captured by trend models

## 4. Layer Breakdown
```
Longevity Modeling:
├─ Fit historical mortality data
├─ Extract trend factors (PCA)
├─ Project factors stochastically
├─ Simulate mortality scenarios
└─ Value liabilities under paths
```

**Interaction:** Fit → extract → project → simulate → value

## 5. Mini-Project
Simple Lee-Carter structure:
```python
import numpy as np

# Simplified: estimate k_t trend
k = np.array([0, -0.5, -1.0, -1.5, -2.0])
drift = -0.5
sigma = 0.3

# project next 5 years
k_proj = [k[-1]]
for _ in range(5):
    k_proj.append(k_proj[-1] + drift + sigma * np.random.normal())

print("Projected k:", k_proj)
```

## 6. Challenge Round
Common pitfalls:
- Extrapolating linear trends indefinitely
- Ignoring parameter uncertainty
- Not stress-testing mortality shocks

## 7. Key References
- [Lee-Carter Model (Wikipedia)](https://en.wikipedia.org/wiki/Lee%E2%80%93Carter_model)
- [Cairns-Blake-Dowd (SOA)](https://www.soa.org/)
- [Longevity Risk (Wikipedia)](https://en.wikipedia.org/wiki/Longevity_risk)

---
**Status:** Longevity capital tool | **Complements:** Life Tables, Mortality Assumptions
