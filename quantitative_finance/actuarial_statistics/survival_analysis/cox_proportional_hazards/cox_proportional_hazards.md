# Cox Proportional Hazards

## 1. Concept Skeleton
**Definition:** Semi-parametric model with hazard $\lambda(t|x)=\lambda_0(t)\exp(\beta^\top x)$  
**Purpose:** Estimate covariate effects on hazard without specifying baseline hazard  
**Prerequisites:** Hazard function, regression, censoring

## 2. Comparative Framing
| Model | Cox PH | Parametric Survival | Accelerated Failure Time |
|------|--------|---------------------|--------------------------|
| **Baseline** | Unspecified | Specified distribution | Time-scaling |
| **Interpretation** | Hazard ratios | Parameter effects | Time ratios |
| **Flexibility** | High | Medium | Medium |

## 3. Examples + Counterexamples

**Simple Example:**  
Estimate hazard ratio for smokers vs non-smokers

**Failure Case:**  
Non-proportional hazards over time violate model assumptions

**Edge Case:**  
Time-varying covariates require extensions

## 4. Layer Breakdown
```
Cox Model Workflow:
├─ Specify covariates x
├─ Fit β via partial likelihood
├─ Check proportional hazards assumption
└─ Interpret exp(β) as hazard ratios
```

**Interaction:** Fit → validate assumptions → interpret effects

## 5. Mini-Project
Fit a Cox model with lifelines (conceptual):
```python
# requires lifelines package
from lifelines import CoxPHFitter
import pandas as pd

# df columns: duration, event, covariates...
# cph = CoxPHFitter().fit(df, duration_col='duration', event_col='event')
# cph.summary
```

## 6. Challenge Round
Common pitfalls:
- Ignoring time-varying effects
- Overfitting with many covariates
- Misinterpreting hazard ratios as probabilities

## 7. Key References
- [Proportional Hazards Model (Wikipedia)](https://en.wikipedia.org/wiki/Proportional_hazards_model)
- [Cox Model (StatLect)](https://www.statlect.com/fundamentals-of-statistics/proportional-hazards-model)
- [Therneau, Survival Analysis](https://cran.r-project.org/web/packages/survival/)

---
**Status:** Core semi-parametric model | **Complements:** Kaplan–Meier, Nelson–Aalen
