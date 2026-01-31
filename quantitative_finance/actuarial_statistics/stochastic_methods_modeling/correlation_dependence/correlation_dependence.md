# Correlation & Dependence

## 1. Concept Skeleton
**Definition:** Modeling joint distributions via copulas to capture tail dependence  
**Purpose:** Aggregate risks correctly and avoid underestimating extreme scenarios  
**Prerequisites:** Copula theory, marginal distributions, correlation measures

## 2. Comparative Framing
| Approach | Linear Correlation | Gaussian Copula | t-Copula |
|----------|-------------------|-----------------|----------|
| **Tail Dependence** | None | Weak | Strong |
| **Simplicity** | High | Moderate | Moderate |
| **Use Case** | Normal risks | Moderate risks | Extreme risks |

## 3. Examples + Counterexamples

**Simple Example:**  
Gaussian copula: joint distribution with normal dependence structure

**Failure Case:**  
Using linear correlation for assets with tail dependence

**Edge Case:**  
t-Copula with low degrees of freedom captures fat-tailed dependence

## 4. Layer Breakdown
```
Dependence Modeling:
├─ Fit marginal distributions
├─ Select copula (Gaussian, t, Clayton)
├─ Calibrate dependence parameters
├─ Simulate joint scenarios
└─ Aggregate risk metrics
```

**Interaction:** Fit margins → select copula → calibrate → simulate

## 5. Mini-Project
Generate correlated normals via Gaussian copula:
```python
import numpy as np

n = 1000
rho = 0.5

# correlated normals
Z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
U = np.apply_along_axis(lambda x: np.exp(x), 0, Z)  # transform to lognormal

print("Correlation:", np.corrcoef(U.T))
```

## 6. Challenge Round
Common pitfalls:
- Ignoring tail dependence in stress scenarios
- Misspecifying copula type
- Not validating fit with historical data

## 7. Key References
- [Copula (Wikipedia)](https://en.wikipedia.org/wiki/Copula_(probability_theory))
- [Gaussian Copula (Wikipedia)](https://en.wikipedia.org/wiki/Copula_(statistics)#Gaussian_copula)
- [Tail Dependence (Wikipedia)](https://en.wikipedia.org/wiki/Tail_dependence)

---
**Status:** Risk aggregation tool | **Complements:** Monte Carlo, Capital Modeling
