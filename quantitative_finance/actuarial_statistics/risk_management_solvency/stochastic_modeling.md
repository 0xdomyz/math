# Stochastic Modeling

## 1. Concept Skeleton
**Definition:** Simulation of random future scenarios to assess tail risk and distributions  
**Purpose:** Quantify capital needs under uncertainty and non-linear risks  
**Prerequisites:** Probability distributions, Monte Carlo, risk measures

## 2. Comparative Framing
| Approach | Deterministic | Stochastic | Scenario Testing |
|----------|---------------|------------|------------------|
| **Output** | Single path | Distribution | Limited set |
| **Tail Risk** | Poor | Strong | Moderate |
| **Cost** | Low | High | Medium |

## 3. Examples + Counterexamples

**Simple Example:**  
Simulate interest rate paths to value guarantees

**Failure Case:**  
Too few simulations → unstable tail estimates

**Edge Case:**  
Model risk from incorrect dependence structures

## 4. Layer Breakdown
```
Stochastic Modeling Pipeline:
├─ Specify risk factors and distributions
├─ Model dependence (correlation/copulas)
├─ Simulate scenarios
├─ Aggregate cash flows
└─ Compute risk measures (VaR, ES)
```

**Interaction:** Specify → simulate → aggregate → measure risk

## 5. Mini-Project
Estimate tail loss with Monte Carlo:
```python
import numpy as np

np.random.seed(0)
loss = np.random.lognormal(mean=0.0, sigma=0.6, size=10000)
var_99 = np.quantile(loss, 0.99)
print("VaR 99%:", var_99)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring model risk and parameter uncertainty
- Underestimating tail dependence
- Reusing calibrations from calm periods

## 7. Key References
- [Monte Carlo Method (Wikipedia)](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Value at Risk (Wikipedia)](https://en.wikipedia.org/wiki/Value_at_risk)
- [Expected Shortfall (Wikipedia)](https://en.wikipedia.org/wiki/Expected_shortfall)

---
**Status:** Core solvency tool | **Complements:** Capital Requirements, Risk Aggregation
