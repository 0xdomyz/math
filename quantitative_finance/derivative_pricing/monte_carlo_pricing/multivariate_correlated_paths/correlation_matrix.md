# Correlation Matrix

## 1. Concept Skeleton
**Definition:** Symmetric matrix $\rho$ with entries $\rho_{ij} = \mathrm{Corr}(X_i, X_j)$ and $\rho_{ii}=1$  
**Purpose:** Encode linear dependence for multivariate simulation and risk aggregation  
**Prerequisites:** Covariance, variance, linear algebra, eigenvalues

## 2. Comparative Framing
| Concept | Correlation Matrix $\rho$ | Covariance Matrix $\Sigma$ | Independence |
|---|---|---|---|
| **Scale** | Unitless, bounded $[-1,1]$ | Scale-dependent | N/A |
| **Diagonal** | All 1s | Variances $\sigma_i^2$ | N/A |
| **Use** | Dependence structure | Full second moments | No dependence |
| **PSD Requirement** | Yes | Yes | Trivial |

## 3. Examples + Counterexamples

**Simple Example:**  
Two assets with $\rho=0.6$: positive co-movement, portfolio risk increases.

**Failure Case:**  
Matrix with negative eigenvalue → not PSD → invalid for Cholesky; simulation fails.

**Edge Case:**  
$\rho=1$ for all pairs → perfectly correlated; diversification benefit collapses.

## 4. Layer Breakdown
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

## 5. Mini-Project
Estimate and validate a correlation matrix, then stress it:
```python
import numpy as np

np.random.seed(42)

# Simulated returns (3 assets)
T = 1000
mu = np.array([0.0005, 0.0003, 0.0004])
vol = np.array([0.01, 0.012, 0.009])

# Target correlation
rho = np.array([
    [1.0, 0.6, 0.2],
    [0.6, 1.0, 0.4],
    [0.2, 0.4, 1.0]
])

# Build covariance and generate correlated returns
Sigma = np.outer(vol, vol) * rho
L = np.linalg.cholesky(Sigma)
Z = np.random.randn(T, 3)
R = Z @ L.T + mu  # returns

# Sample correlation
sample_rho = np.corrcoef(R.T)

# PSD check
min_eig = np.min(np.linalg.eigvalsh(sample_rho))

print("Sample correlation:\n", np.round(sample_rho, 3))
print("Min eigenvalue:", round(min_eig, 6))

# Stress correlation: increase off-diagonals by 0.2
stress = sample_rho.copy()
for i in range(3):
    for j in range(3):
        if i != j:
            stress[i, j] = min(0.99, stress[i, j] + 0.2)

# Re-symmetrize and fix diagonal
stress = (stress + stress.T) / 2
np.fill_diagonal(stress, 1.0)

min_eig_stress = np.min(np.linalg.eigvalsh(stress))
print("\nStressed correlation:\n", np.round(stress, 3))
print("Min eigenvalue (stressed):", round(min_eig_stress, 6))
```

## 6. Challenge Round

**Q1:** Why must $\rho$ be PSD?  
**A1:** Any variance $\mathrm{Var}(w^T X)=w^T \Sigma w$ must be nonnegative. If $\rho$ is not PSD, some portfolios imply negative variance.

**Q2:** Can all pairwise correlations be set arbitrarily?  
**A2:** No. Pairwise correlations must jointly satisfy PSD constraints; inconsistent triples can produce negative eigenvalues.

**Q3:** How do you “fix” a non-PSD correlation matrix?  
**A3:** Project to nearest PSD matrix (e.g., eigenvalue clipping or Higham’s algorithm) and re-normalize diagonal to 1.

**Q4:** Why is correlation stress important for portfolios?  
**A4:** Diversification depends on correlations; during crises correlations rise, making risk and losses higher.

## 7. Key References
- [Correlation and dependence](https://en.wikipedia.org/wiki/Correlation_and_dependence)  
- [Correlation matrix](https://en.wikipedia.org/wiki/Correlation_matrix)  
- Higham, N.J. “Computing a nearest symmetric positive semidefinite matrix” (2002)

---
**Status:** Core multivariate dependency primitive | **Complements:** Cholesky, PCA, Copulas
