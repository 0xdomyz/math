# Correlation vs Causation

## 3.1 Concept Skeleton
**Definition:** Correlation = statistical association; Causation = one variable directly changes another  
**Purpose:** Avoid false conclusions from observational data  
**Prerequisites:** Regression, confounding variables, experimental design

## 3.2 Comparative Framing
| Aspect | Correlation | Causation | Confounding |
|--------|-------------|-----------|-------------|
| **Measured by** | r ∈ [-1, 1] | Regression coefficient | Lurking variable |
| **Strength** | Perfect: r=±1 | Requires mechanism | Hides true relationship |
| **Evidence needed** | Co-variation only | + temporal order + mechanism | Needs adjustment |
| **Graph shape** | Scatter plot | DAG (directed acyclic graph) | Triangle: Z→X, Z→Y |

## 3.3 Examples + Counterexamples

**False Causation (Classic):**  
"Ice cream sales correlate with drowning deaths"  
Truth: Both caused by warm weather (Z-variable)

**Real Causation Example:**  
Randomized drug trial: Random assignment breaks confounding, enables causal inference

**Edge Case:**  
Reverse causation: Does depression cause poor sleep, or poor sleep cause depression? Need longitudinal data to establish temporal order

## 3.4 Layer Breakdown
```
Path to Causal Inference:
├─ Observational data: correlation only (Z confounds X→Y)
├─ Temporal sequencing: X before Y (rules out reverse causation)
├─ Mechanism: Theory explaining X→Y (plausible?)
├─ Adjustment: Control for confounders (stratification, regression)
├─ Randomization: Ideal but often impossible
└─ Causal diagram: DAG showing all relationships
```

## 3.5 Mini-Project
Identify confounding in real data:
```python
# Question: Does coffee cause heart disease?
# Correlation: coffee ↔ heart disease
# Confounders: age, smoking (both correlate with coffee AND disease)

# Solution: Stratify by age/smoking, run regression with covariates
# Result: Confounding explains most correlation

import pandas as pd
from sklearn.linear_model import LinearRegression

# Unadjusted: coffee → heart disease (confounded)
# Adjusted: coffee → heart disease (controlling for age, smoking)
# Compare coefficients to detect confounding
```

## 3.6 Challenge Round
When is observational correlation ENOUGH?
- Consistent pattern across populations (reproducible correlation)
- Dose-response relationship (more X → more Y)
- No plausible alternative explanations
- Strong theoretical mechanism

When must you have randomization?
- Studying harmful interventions (can't randomize people to smoking)
- Policy decisions requiring high confidence
- When confounders are unmeasured

## 3.7 Key References
- [Bradford Hill Criteria for Causation](https://en.wikipedia.org/wiki/Bradford_Hill_criteria)
- [Simpson's Paradox (confounding visual)](https://en.wikipedia.org/wiki/Simpson%27s_paradox)
- [Causal Inference Book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

---
**Status:** Critical concept | **Complements:** Regression, Confounding, Experimental Design
