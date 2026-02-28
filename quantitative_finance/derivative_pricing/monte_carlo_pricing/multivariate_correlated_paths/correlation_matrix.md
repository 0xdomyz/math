# Correlation Matrix

## Concept Skeleton
**Definition:** Symmetric matrix $\rho$ with entries $\rho_{ij} = \mathrm{Corr}(X_i, X_j)$ and $\rho_{ii}=1$  
**Purpose:** Encode linear dependence for multivariate simulation and risk aggregation  
**Prerequisites:** Covariance, variance, linear algebra, eigenvalues

## Comparative Framing
| Concept | Correlation Matrix $\rho$ | Covariance Matrix $\Sigma$ | Independence |
|---|---|---|---|
| **Scale** | Unitless, bounded $[-1,1]$ | Scale-dependent | N/A |
| **Diagonal** | All 1s | Variances $\sigma_i^2$ | N/A |
| **Use** | Dependence structure | Full second moments | No dependence |
| **PSD Requirement** | Yes | Yes | Trivial |

## Examples + Counterexamples
**Simple Example:**  
Two assets with $\rho=0.6$: positive co-movement, portfolio risk increases.

**Failure Case:**  
Matrix with negative eigenvalue → not PSD → invalid for Cholesky; simulation fails.

**Edge Case:**  
$\rho=1$ for all pairs → perfectly correlated; diversification benefit collapses.

## Layer Breakdown
```
Correlation Matrix Workflow:
├─ Inputs:
│   ├─ Returns: r_1, r_2, ..., r_n
│   ├─ Sample means: μ_i
│   └─ Sample std dev: σ_i
├─ Compute Covariance:
│   └─ Σ_ij = E[(r_i-μ_i)(r_j-μ_j)]
├─ Normalize:
│   └─ ρ_ij = Σ_ij / (σ_i σ_j)
├─ Properties:
│   ├─ Symmetric: ρ_ij = ρ_ji
│   ├─ Diagonal: ρ_ii = 1
│   ├─ PSD: x^T ρ x ≥ 0 for all x
│   └─ Eigenvalues ≥ 0 (numerical tolerance)
├─ Validation Checks:
│   ├─ Bounds: |ρ_ij| ≤ 1
│   ├─ PSD check: min eigenvalue ≥ -ε
│   └─ Condition number (stability)
└─ Use:
    ├─ Cholesky factorization for correlated normals
    ├─ Risk aggregation: σ_p^2 = w^T Σ w
    └─ Scenario analysis: correlation stress
```

**Interaction:** Estimate $\rho$ → validate PSD → use in simulation or optimization

## Challenge Round
**Q1:** Why must $\rho$ be PSD?  
**A1:** Any variance $\mathrm{Var}(w^T X)=w^T \Sigma w$ must be nonnegative. If $\rho$ is not PSD, some portfolios imply negative variance.

**Q2:** Can all pairwise correlations be set arbitrarily?  
**A2:** No. Pairwise correlations must jointly satisfy PSD constraints; inconsistent triples can produce negative eigenvalues.

**Q3:** How do you “fix” a non-PSD correlation matrix?  
**A3:** Project to nearest PSD matrix (e.g., eigenvalue clipping or Higham’s algorithm) and re-normalize diagonal to 1.

**Q4:** Why is correlation stress important for portfolios?  
**A4:** Diversification depends on correlations; during crises correlations rise, making risk and losses higher.

## Key References
- [Correlation and dependence](https://en.wikipedia.org/wiki/Correlation_and_dependence)  
- [Correlation matrix](https://en.wikipedia.org/wiki/Correlation_matrix)  
- Higham, N.J. “Computing a nearest symmetric positive semidefinite matrix” (2002)

---
**Status:** Core multivariate dependency primitive | **Complements:** Cholesky, PCA, Copulas
