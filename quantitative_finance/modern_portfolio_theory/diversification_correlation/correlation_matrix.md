# Correlation Matrix

## 1. Concept Skeleton
**Definition:** n×n symmetric matrix of pairwise correlation coefficients between n variables; diagonal elements equal 1, off-diagonal in [-1, 1]  
**Purpose:** Summarize interdependencies, input for portfolio optimization and factor models, assess diversification  
**Prerequisites:** Correlation coefficient, matrix algebra, positive semi-definiteness, eigenvalues

## 2. Comparative Framing
| Matrix Type | Covariance | Correlation | Precision | Partial Correlation |
|------------|-----------|-------------|-----------|-------------------|
| **Definition** | Cov(Ri, Rj) | ρij = Cov/(σiσj) | Σ⁻¹ (inverse covariance) | ρij after removing effect of other vars |
| **Diagonal** | Variances | 1 (always) | 1/σi² (inverse variance) | 1 or < 1 |
| **Range** | (-∞, ∞) | [-1, 1] | Positive | [-1, 1] |
| **Units** | Product of units | Unitless | 1/units² | Unitless |
| **Scale-invariant** | No | Yes | No | Yes |

## 3. Examples + Counterexamples

**Simple Example:**  
3-asset portfolio: ρ₁₂ = 0.8, ρ₁₃ = 0.3, ρ₂₃ = -0.1. Assets 1-2 move together; 1-3 weakly related; 2-3 slightly inverse

**Failure Case:**  
Non-positive-definite correlation matrix (eigenvalue < 0): Impossible covariance structure; portfolio variance can be negative (mathematically invalid)

**Edge Case:**  
Singular correlation matrix (one eigenvalue = 0): Perfect multicollinearity; one asset deterministic function of others; optimization unstable

## 4. Layer Breakdown
```
Correlation Matrix Framework:
├─ Definition & Properties:
│   ├─ P = n×n symmetric matrix with diagonal 1
│   ├─ Pij = Pji (symmetric)
│   ├─ Pii = 1 for all i (perfect correlation with self)
│   ├─ |Pij| ≤ 1 for all i ≠ j (bounded)
│   └─ Must be positive semi-definite (P = L'L, all eigenvalues ≥ 0)
├─ Construction from Covariance:
│   ├─ P = D⁻¹ΣD⁻¹ where D diagonal matrix with σi on diagonal
│   ├─ Or: Pij = Σij / (σi·σj)
│   ├─ Reverse: Σ = D·P·D (reconstruct covariance)
│   └─ Scaling property: P scale-invariant but Σ is not
├─ Positive Semi-Definiteness:
│   ├─ All eigenvalues λk ≥ 0 (necessary condition)
│   ├─ All principal minors ≥ 0
│   ├─ Cholesky decomposition exists: P = L·L'
│   ├─ Fails if rank < n (perfect multicollinearity)
│   └─ Check numerically: Compute eigenvalues (λmin > -1e-10)
├─ Estimation & Bias:
│   ├─ Sample correlation: rij = Σ(Xi-X̄)(Yi-Ȳ) / √[Σ(Xi-X̄)² Σ(Yi-Ȳ)²]
│   ├─ Sample correlation matrix R is biased when n small
│   ├─ Estimation error: Shrinkage P* = λR + (1-λ)I improves OOS
│   └─ Assumes stationarity; time-varying ρ requires rolling windows
├─ Common Parameterizations:
│   ├─ Equicorrelation: ρij = ρ for all i ≠ j (single parameter)
│   ├─ Block structure: Sectors with high internal, low external ρ
│   ├─ Factor model: ρij = βi·βj + δij (driven by common factors)
│   └─ Hierarchical: Nested correlation blocks (industry, country, global)
├─ Numerical Issues:
│   ├─ Ensures positive semi-definiteness after estimation
│   ├─ Eigenvalue clipping: Set negative λk → 0, adjust others
│   ├─ Projection: Find nearest positive semi-definite matrix
│   ├─ Condition number: κ = λmax/λmin (ill-conditioning when κ large)
│   └─ Regularization: Add small positive constant to diagonal
└─ Applications:
    ├─ Portfolio variance: σ²p = w'·Σ·w = w'·D·P·D·w
    ├─ Factor models: ρij ≈ βi·βj + specific variance
    ├─ Risk management: Correlation breaks under stress
    └─ Diversification: Lower ρij → higher diversification benefit
```

**Interaction:** Individual correlations organized → matrix structure → eigenvalue decomposition → factor analysis

## 5. Mini-Project
Construct, validate, and stress-test correlation matrices for portfolio applications:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh, cholesky
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CORRELATION MATRIX ANALYSIS")
print("="*70)

# Generate synthetic returns with structure
np.random.seed(42)
periods = 252
n_assets = 8

# Market factor (systematic risk)
market = np.random.normal(0, 0.01, periods)

# Sector factors
sector1 = np.random.normal(0, 0.008, periods)
sector2 = np.random.normal(0, 0.008, periods)

# Individual returns with factor exposure
returns = np.zeros((periods, n_assets))

# Sector 1: Assets 0-3 (higher internal correlation)
for i in range(4):
    returns[:, i] = 0.7 * market + 0.4 * sector1 + np.random.normal(0, 0.006, periods)

# Sector 2: Assets 4-7 (higher internal correlation)
for i in range(4, 8):
    returns[:, i] = 0.7 * market + 0.4 * sector2 + np.random.normal(0, 0.006, periods)

returns_df = pd.DataFrame(returns, columns=[f'Asset_{i+1}' for i in range(n_assets)])

# 1. Sample correlation matrix
print("\n1. SAMPLE CORRELATION MATRIX")
print("-"*70)

P_sample = returns_df.corr()
print("\nCorrelation Matrix (first 4x4):")
print(P_sample.iloc[:4, :4].round(4))

# 2. Positive semi-definiteness check
print("\n2. POSITIVE SEMI-DEFINITENESS CHECK")
print("-"*70)

eigenvalues, eigenvectors = eigh(P_sample.values)
eigenvalues = eigenvalues[::-1]

print(f"\nEigenvalues (sorted descending):")
for i, eig in enumerate(eigenvalues):
    print(f"  λ{i+1}: {eig:8.6f}")

min_eigenvalue = eigenvalues[-1]
is_psd = min_eigenvalue > -1e-10
print(f"\nMinimum eigenvalue: {min_eigenvalue:.10f}")
print(f"Positive semi-definite: {is_psd}")

condition_number = eigenvalues[0] / (abs(min_eigenvalue) if min_eigenvalue > 0 else 1e-10)
print(f"Condition number: {condition_number:.2f}")

# 3. Correlation structure analysis
print("\n3. CORRELATION STRUCTURE ANALYSIS")
print("-"*70)

# Average within-sector correlations
within_sector1 = []
within_sector2 = []
between_sector = []

for i in range(4):
    for j in range(i+1, 4):
        within_sector1.append(P_sample.iloc[i, j])
        
for i in range(4, 8):
    for j in range(i+1, 8):
        within_sector2.append(P_sample.iloc[i, j])

for i in range(4):
    for j in range(4, 8):
        between_sector.append(P_sample.iloc[i, j])

print(f"\nAverage correlations:")
print(f"  Within Sector 1 (Assets 1-4): {np.mean(within_sector1):.4f}")
print(f"  Within Sector 2 (Assets 5-8): {np.mean(within_sector2):.4f}")
print(f"  Between Sectors: {np.mean(between_sector):.4f}")

# 4. Correlation matrix shrinkage
print("\n4. SHRINKAGE ESTIMATOR FOR CORRELATION MATRIX")
print("-"*70)

# Ledoit-Wolf shrinkage of covariance
lw = LedoitWolf()
cov_lw, shrinkage_intensity = lw.fit(returns_df.values).covariance_, lw.shrinkage

# Convert to correlation
D_inv = np.diag(1 / np.sqrt(np.diag(cov_lw)))
P_lw = D_inv @ cov_lw @ D_inv

# Also shrink toward identity
P_shrink = shrinkage_intensity * P_sample + (1 - shrinkage_intensity) * np.eye(n_assets)

print(f"\nShrinkage intensity λ: {shrinkage_intensity:.4f}")
print(f"  (0 = sample, 1 = identity)")

# Check positive semi-definiteness after shrinkage
eigs_shrink = np.linalg.eigvalsh(P_shrink)
print(f"\nMinimum eigenvalue after shrinkage: {eigs_shrink.min():.10f}")
print(f"Positive semi-definite: {eigs_shrink.min() > -1e-10}")

# 5. Nearest PSD matrix (if needed)
print("\n5. NEAREST POSITIVE SEMI-DEFINITE MATRIX")
print("-"*70)

def nearest_psd(P, max_iter=100):
    """Compute nearest PSD matrix using Higham's algorithm"""
    n = P.shape[0]
    W = P.copy()
    
    for iteration in range(max_iter):
        # Spectral decomposition
        eigvals, eigvecs = eigh(W)
        eigvals[eigvals < 0] = 0  # Clip negative eigenvalues
        
        # Reconstruct
        W_old = W.copy()
        W = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Make symmetric and adjust diagonal
        W = (W + W.T) / 2
        np.fill_diagonal(W, 1)
        
        # Check convergence
        diff = np.linalg.norm(W - W_old)
        if diff < 1e-10:
            break
    
    return W

P_psd = nearest_psd(P_sample)
eigs_psd = np.linalg.eigvalsh(P_psd)
print(f"Minimum eigenvalue after projection: {eigs_psd.min():.10f}")
print(f"Positive semi-definite: {eigs_psd.min() > -1e-10}")

max_deviation = np.abs(P_psd - P_sample).max()
print(f"Max deviation from original: {max_deviation:.6f}")

# 6. Cholesky decomposition
print("\n6. CHOLESKY DECOMPOSITION")
print("-"*70)

try:
    L = np.linalg.cholesky(P_psd)
    print("Cholesky decomposition successful")
    print(f"Lower triangular matrix L (first 3x3):")
    print(L[:3, :3].round(4))
    
    # Verify: P = L @ L.T
    P_reconstructed = L @ L.T
    reconstruction_error = np.linalg.norm(P_psd - P_reconstructed)
    print(f"\nReconstruction error ||P - LL'||: {reconstruction_error:.10f}")
    
    # Can generate correlated random variables
    z = np.random.normal(0, 1, (100, n_assets))  # Independent standard normals
    correlated_vars = z @ L.T  # Correlate them
    corr_generated = np.corrcoef(correlated_vars.T)
    print(f"Max error in generated correlation: {np.abs(corr_generated - P_psd).max():.6f}")
    
except np.linalg.LinAlgError:
    print("Cholesky decomposition failed - matrix not positive definite")

# 7. Portfolio variance implications
print("\n7. PORTFOLIO VARIANCE IMPLICATIONS")
print("-"*70)

# Equal-weight portfolio
w_eq = np.ones(n_assets) / n_assets

# Compute portfolio variance using different correlation matrices
# Need covariance matrices for this
cov_sample = returns_df.cov() * 252
var_eq_sample = w_eq @ cov_sample.values @ w_eq

# Using shrunk correlation (need to scale back to covariance)
std_devs = returns_df.std().values * np.sqrt(252)
D = np.diag(std_devs)
cov_shrink = D @ P_shrink @ D
var_eq_shrink = w_eq @ cov_shrink @ w_eq

# Using PSD-projected correlation
cov_psd = D @ P_psd @ D
var_eq_psd = w_eq @ cov_psd @ w_eq

print(f"Equal-weight portfolio variance:")
print(f"  Sample correlation: {var_eq_sample:.6f} (vol: {np.sqrt(var_eq_sample):.4f})")
print(f"  Shrunk correlation: {var_eq_shrink:.6f} (vol: {np.sqrt(var_eq_shrink):.4f})")
print(f"  PSD-projected correlation: {var_eq_psd:.6f} (vol: {np.sqrt(var_eq_psd):.4f})")

# 8. Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Sample correlation matrix
ax = axes[0, 0]
im = ax.imshow(P_sample.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title('Sample Correlation Matrix')
ax.set_xticks(range(n_assets))
ax.set_yticks(range(n_assets))
ax.set_xticklabels(returns_df.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(returns_df.columns, fontsize=8)
plt.colorbar(im, ax=ax)

# Plot 2: Shrunk correlation matrix
ax = axes[0, 1]
im = ax.imshow(P_shrink, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title(f'Shrunk Correlation Matrix (λ={shrinkage_intensity:.3f})')
ax.set_xticks(range(n_assets))
ax.set_yticks(range(n_assets))
ax.set_xticklabels(returns_df.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(returns_df.columns, fontsize=8)
plt.colorbar(im, ax=ax)

# Plot 3: PSD-projected correlation matrix
ax = axes[0, 2]
im = ax.imshow(P_psd, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title('PSD-Projected Correlation Matrix')
ax.set_xticks(range(n_assets))
ax.set_yticks(range(n_assets))
ax.set_xticklabels(returns_df.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(returns_df.columns, fontsize=8)
plt.colorbar(im, ax=ax)

# Plot 4: Eigenvalues
ax = axes[1, 0]
x = range(1, len(eigenvalues)+1)
ax.semilogy(x, eigenvalues, 'o-', linewidth=2, markersize=8, label='Sample')
eigs_shrink_sorted = np.sort(np.linalg.eigvalsh(P_shrink))[::-1]
ax.semilogy(x, eigs_shrink_sorted, 's--', linewidth=2, markersize=6, alpha=0.7, label='Shrunk')
ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero')
ax.set_xlabel('Eigenvalue Index')
ax.set_ylabel('Eigenvalue (log scale)')
ax.set_title('Eigenvalue Spectrum')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Difference in correlation matrices
ax = axes[1, 1]
diff = np.abs(P_psd - P_sample)
im = ax.imshow(diff, cmap='Reds')
ax.set_title('|Projected - Sample| Correlation')
ax.set_xticks(range(n_assets))
ax.set_yticks(range(n_assets))
ax.set_xticklabels(returns_df.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(returns_df.columns, fontsize=8)
plt.colorbar(im, ax=ax)

# Plot 6: Eigenvalue distribution
ax = axes[1, 2]
ax.bar(['Sample', 'Shrunk', 'PSD-Projected'], 
       [eigenvalues[-1], eigs_shrink.min(), eigs_psd.min()],
       color=['b', 'r', 'g'], alpha=0.7)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.set_ylabel('Minimum Eigenvalue')
ax.set_title('Minimum Eigenvalue by Method')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('correlation_matrix_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Key Findings:

1. Correlation Structure:
   → Block structure detected: Higher within-sector ρ
   → Between-sector: Weaker correlations (diversification potential)
   
2. Positive Semi-Definiteness:
   → Sample matrix: Min λ = {min_eigenvalue:.6f}
   → Condition number: {condition_number:.1f}
   → Matrix is valid for portfolio optimization
   
3. Estimation Improvements:
   → Shrinkage λ = {shrinkage_intensity:.3f}: Moderate intensity
   → Reduces extreme values → more stable optimization
   → Out-of-sample performance typically better
   
4. Numerical Stability:
   → PSD projection: Ensures valid matrix for all algorithms
   → Max adjustment: {max_deviation:.6f} (small)
   → Cholesky decomposition: {('Successful' if np.linalg.eigvalsh(P_psd).min() > 0 else 'Failed')}
   
5. Portfolio Risk Impact:
   → Different correlation estimates → different portfolio risks
   → Diversification benefit depends on ρ structure
   → Shrinkage: {abs(var_eq_shrink - var_eq_sample)/(var_eq_sample)*100:.1f}% variance change
""")
```

## 6. Challenge Round
When correlation matrix methods fail:
- Non-positive-definite estimated matrix: Use shrinkage or eigenvalue clipping; don't ignore
- Time-varying correlation: Single matrix → model error; use DCC-GARCH or rolling windows
- High dimensionality (n large, T small): Correlation matrix singular; use factor models or regularization
- Tail correlation breakdown: Normal correlations miss extremes; use copulas or tail dependence
- Sparse correlations (many zero): Network methods; skip zero correlations in optimization

## 7. Key References
- [Higham (2002), "Computing the Nearest Correlation Matrix—A Problem from Finance"](https://doi.org/10.1137/S0895479801383202)
- [Ledoit & Wolf (2004), "Honey, I Shrunk the Covariance Matrix"](https://www.jstor.org/stable/3598996)
- [Christoffersen (2011), "Elements of Financial Risk Management" - Chapter 3](https://www.elsevier.com/books/elements-of-financial-risk-management/christoffersen/978-0-12-374448-7)

---
**Status:** Essential for portfolio optimization | **Complements:** Covariance, Diversification, Efficient Frontier
