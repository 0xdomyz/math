# Principal Component Analysis (PCA)

## 1. Concept Skeleton
**Definition:** Orthogonal transformation projecting data onto directions of maximum variance  
**Purpose:** Reduce dimensionality of correlation/covariance structures for faster simulation  
**Prerequisites:** Eigen decomposition, covariance, linear algebra

## 2. Comparative Framing
| Method | Full Cholesky | PCA (k factors) | Independent Approx |
|---|---|---|---|
| **Accuracy** | Exact | Approx (variance-captured) | Low |
| **Speed** | O(n^3) setup | O(n^3) setup, O(kn) sim | O(n) |
| **Use Case** | Small n, high fidelity | Large n, factor models | Rough risk only |

## 3. Examples + Counterexamples

**Simple Example:**  
3 assets with one dominant market factor → first PC explains 85% variance.

**Failure Case:**  
If variance is evenly spread across components, PCA compression loses structure.

**Edge Case:**  
Perfect correlation: only 1 PC has non-zero eigenvalue → exact 1-factor model.

## 4. Layer Breakdown
```
PCA for Correlated Simulation:
├─ Input:
│   ├─ Covariance matrix Σ (n×n)
│   └─ Eigen-decomposition: Σ = Q Λ Q^T
├─ Sort Components:
│   ├─ Eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ
│   └─ Eigenvectors q_i as principal directions
├─ Select k Factors:
│   ├─ Choose k s.t. cumulative variance ≥ threshold (e.g., 95%)
│   └─ Q_k = [q₁, ..., q_k], Λ_k = diag(λ₁...λ_k)
├─ Factor Representation:
│   └─ Σ ≈ Q_k Λ_k Q_k^T = B B^T, where B = Q_k Λ_k^{1/2}
├─ Simulation:
│   ├─ Generate Z ~ N(0, I_k)
│   └─ X ≈ B Z (n-dimensional correlated draws)
└─ Validation:
    ├─ Compare covariance of X to Σ
    └─ Check variance explained
```

**Interaction:** Decompose $\Sigma$ → keep top k factors → simulate with $k \ll n$

## 5. Mini-Project
Reduce a 10-asset covariance to 3 factors and compare variance explained:
```python
import numpy as np

np.random.seed(42)

n = 10
# Construct a covariance with a strong market factor
market = np.random.randn(n, 1)
Sigma = 0.6 * (market @ market.T) + 0.4 * np.eye(n)

# Eigen decomposition
vals, vecs = np.linalg.eigh(Sigma)
idx = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

# Variance explained
explained = np.cumsum(vals) / np.sum(vals)
print("Variance explained by components:")
for i in range(5):
    print(f"PC{i+1}: {explained[i]*100:.2f}%")

# Keep top k
k = 3
Qk = vecs[:, :k]
Lk = np.diag(vals[:k])
B = Qk @ np.sqrt(Lk)

# Simulate using k factors
m = 50000
Z = np.random.randn(m, k)
X = Z @ B.T

# Compare covariance
Sigma_hat = np.cov(X, rowvar=False)
err = np.linalg.norm(Sigma - Sigma_hat) / np.linalg.norm(Sigma)
print(f"Relative covariance error (k={k}): {err:.4f}")
```

## 6. Challenge Round

**Q1:** How do you choose $k$?  
**A1:** Pick the smallest $k$ achieving a variance threshold (e.g., 95%); trade accuracy vs speed.

**Q2:** Why is PCA useful for high-dimensional Monte Carlo?  
**A2:** It reduces simulation dimension, improving speed and quasi-random effectiveness.

**Q3:** When is PCA risky?  
**A3:** If tail dependence or idiosyncratic risks dominate, PCA underestimates joint extremes.

**Q4:** PCA on correlation vs covariance?  
**A4:** Correlation removes scale; covariance preserves vol structure. Use correlation for standardized assets, covariance for pricing.

## 7. Key References
- [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)  
- [Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix)

---
**Status:** Dimensionality reduction for correlated paths | **Complements:** Cholesky, Copulas
