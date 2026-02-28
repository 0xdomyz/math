# Cholesky Decomposition

## Concept Skeleton
**Definition:** Factor a PSD matrix $A$ into $A = L L^T$, where $L$ is lower triangular  
**Purpose:** Generate correlated normals and simulate multivariate processes efficiently  
**Prerequisites:** Linear algebra, PSD matrices, multivariate normal

## Comparative Framing
| Method | Cholesky | Eigen Decomposition | PCA Factor Model |
|---|---|---|---|
| **Speed** | Fast | Moderate | Fast (k factors) |
| **Requirement** | PSD matrix | Symmetric matrix | PSD covariance |
| **Output** | Triangular $L$ | $Q \Lambda Q^T$ | $B B^T$ (low-rank) |
| **Use** | Exact correlation | Exact correlation | Approximate correlation |

## Examples + Counterexamples
**Simple Example:**  
$\rho=\begin{bmatrix}1&0.5\\0.5&1\end{bmatrix}$ → $L=\begin{bmatrix}1&0\\0.5&\sqrt{0.75}\end{bmatrix}$

**Failure Case:**  
Negative eigenvalue due to inconsistent correlations → Cholesky fails (not PSD).

**Edge Case:**  
Perfect correlation $\rho=\mathbf{1}\mathbf{1}^T$ is PSD but singular → Cholesky fails unless jitter added.

## Layer Breakdown
```
Cholesky for Correlated Normals:
├─ Inputs:
│   ├─ Correlation matrix ρ (n×n)
│   └─ Independent normals Z ~ N(0, I)
├─ Factorization:
│   └─ ρ = L L^T, L lower triangular
├─ Transformation:
│   └─ X = L Z → X ~ N(0, ρ)
├─ Validation:
│   ├─ Empirical correlation of X ≈ ρ
│   └─ Numerical stability (condition number)
└─ Use in Simulation:
    ├─ GBM: dS_i = rS_i dt + σ_i S_i dW_i
    └─ Correlated dW: use X to drive each asset
```

**Interaction:** Factor $\rho$ → transform normals → drive correlated paths

## Challenge Round
**Q1:** Why does Cholesky fail for non-PSD matrices?  
**A1:** It requires all leading principal minors positive; negative eigenvalues imply negative variance in some directions.

**Q2:** How does jitter help?  
**A2:** Adding $\epsilon I$ shifts eigenvalues upward, making the matrix strictly PSD at minimal distortion.

**Q3:** When is eigen decomposition preferable?  
**A3:** For near-singular matrices or when you want to drop small eigenvalues for dimensionality reduction.

**Q4:** How do you verify correctness?  
**A4:** Compute empirical correlation of $X=ZL^T$ and compare to target $\rho$.

## Key References
- [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition)  
- [Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)

---
**Status:** Core tool for correlated simulation | **Complements:** Correlation matrix, PCA, Copulas
