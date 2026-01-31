# Copulas

## 1. Concept Skeleton
**Definition:** Functions linking marginal distributions to a joint distribution with specified dependence  
**Purpose:** Model dependence separately from marginals; capture tail dependence beyond correlation  
**Prerequisites:** CDFs, multivariate distributions, correlation matrix

## 2. Comparative Framing
| Copula | Dependence | Tail Behavior | Typical Use |
|---|---|---|---|
| **Gaussian** | Linear correlation | No tail dependence | Equity, FX | 
| **Student-t** | Correlation + tail | Symmetric tail dependence | Credit, stress | 
| **Clayton** | Lower-tail | Strong lower tail | Defaults | 
| **Gumbel** | Upper-tail | Strong upper tail | Catastrophe risk |

## 3. Examples + Counterexamples

**Simple Example:**  
Gaussian copula with $\rho=0.5$ and lognormal marginals for asset prices.

**Failure Case:**  
Using Gaussian copula for defaults ignores tail dependence → underestimates joint default risk.

**Edge Case:**  
Independence copula $C(u,v)=uv$ → no dependence regardless of marginals.

## 4. Layer Breakdown
```
Copula Modeling Pipeline:
├─ Step 1: Choose marginals F_i(x)
│   └─ Example: Lognormal for equities, Gamma for rates
├─ Step 2: Choose copula C(u_1,...,u_n)
│   ├─ Gaussian: C(u)=Φ_ρ(Φ^{-1}(u))
│   ├─ t-copula: t_ρ,ν(t^{-1}(u))
│   └─ Archimedean: Clayton, Gumbel, Frank
├─ Step 3: Sample dependence
│   ├─ Draw Z ~ N(0, ρ) or t_ν(0, ρ)
│   ├─ Convert to uniforms: U_i = F_Z(Z_i)
│   └─ Preserve dependence in U
├─ Step 4: Apply marginals
│   └─ X_i = F_i^{-1}(U_i)
└─ Step 5: Validate
    ├─ Check marginal fits
    ├─ Tail dependence
    └─ Rank correlation (Spearman, Kendall)
```

**Interaction:** Sample copula dependence → transform to marginals → simulate joint outcomes

## 5. Mini-Project
Gaussian vs t-copula tail dependence (2 assets):
```python
import numpy as np
from scipy.stats import norm, t

np.random.seed(42)

n = 200000
rho = 0.5

# Gaussian copula
Z = np.random.randn(n, 2)
L = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))
Zc = Z @ L.T
U_g = norm.cdf(Zc)

# t-copula (df=4)
nu = 4
Y = t.rvs(df=nu, size=(n, 2))
Yc = Y @ L.T
U_t = t.cdf(Yc, df=nu)

# Tail dependence estimate: P(U1<0.05, U2<0.05)
alpha = 0.05
lt_g = np.mean((U_g[:,0] < alpha) & (U_g[:,1] < alpha))
lt_t = np.mean((U_t[:,0] < alpha) & (U_t[:,1] < alpha))

print(f"Lower-tail prob (Gaussian): {lt_g:.4f}")
print(f"Lower-tail prob (t-copula): {lt_t:.4f}")

# Map to lognormal marginals
mu, sigma = 0.0, 0.2
X_g = np.exp(mu + sigma * norm.ppf(U_g))
X_t = np.exp(mu + sigma * norm.ppf(U_t))

print("Mean Gaussian copula:", X_g.mean(axis=0))
print("Mean t-copula:", X_t.mean(axis=0))
```

## 6. Challenge Round

**Q1:** What does a copula separate?  
**A1:** Dependence structure from marginal distributions; $F_{X,Y}(x,y)=C(F_X(x),F_Y(y))$.

**Q2:** Why is Gaussian copula risky in crises?  
**A2:** It has zero tail dependence; joint extremes are under-modeled.

**Q3:** How do you choose copula family?  
**A3:** Match empirical tail dependence and rank correlation; validate with stress scenarios.

**Q4:** Why use rank correlations (Kendall/Spearman)?  
**A4:** Copulas are invariant to monotonic transforms; rank metrics align with copula dependence.

## 7. Key References
- [Copula (probability theory)](https://en.wikipedia.org/wiki/Copula_(probability_theory))  
- [t-copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)#t-copula)

---
**Status:** Advanced dependence modeling | **Complements:** Correlation, PCA
