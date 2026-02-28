# Copulas

## Concept Skeleton
**Definition:** Functions linking marginal distributions to a joint distribution with specified dependence  
**Purpose:** Model dependence separately from marginals; capture tail dependence beyond correlation  
**Prerequisites:** CDFs, multivariate distributions, correlation matrix

## Comparative Framing
| Copula | Dependence | Tail Behavior | Typical Use |
|---|---|---|---|
| **Gaussian** | Linear correlation | No tail dependence | Equity, FX | 
| **Student-t** | Correlation + tail | Symmetric tail dependence | Credit, stress | 
| **Clayton** | Lower-tail | Strong lower tail | Defaults | 
| **Gumbel** | Upper-tail | Strong upper tail | Catastrophe risk |

## Examples + Counterexamples
**Simple Example:**  
Gaussian copula with $\rho=0.5$ and lognormal marginals for asset prices.

**Failure Case:**  
Using Gaussian copula for defaults ignores tail dependence → underestimates joint default risk.

**Edge Case:**  
Independence copula $C(u,v)=uv$ → no dependence regardless of marginals.

## Layer Breakdown
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

## Challenge Round
**Q1:** What does a copula separate?  
**A1:** Dependence structure from marginal distributions; $F_{X,Y}(x,y)=C(F_X(x),F_Y(y))$.

**Q2:** Why is Gaussian copula risky in crises?  
**A2:** It has zero tail dependence; joint extremes are under-modeled.

**Q3:** How do you choose copula family?  
**A3:** Match empirical tail dependence and rank correlation; validate with stress scenarios.

**Q4:** Why use rank correlations (Kendall/Spearman)?  
**A4:** Copulas are invariant to monotonic transforms; rank metrics align with copula dependence.

## Key References
- [Copula (probability theory)](https://en.wikipedia.org/wiki/Copula_(probability_theory))  
- [t-copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)#t-copula)

---
**Status:** Advanced dependence modeling | **Complements:** Correlation, PCA
