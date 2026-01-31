# Principal Component Analysis (PCA)

## 1. Concept Skeleton
**Definition:** Dimensionality reduction technique transforming correlated variables into orthogonal principal components maximizing variance  
**Purpose:** Compress high-dimensional data, remove multicollinearity, visualize patterns, denoise signals  
**Prerequisites:** Linear algebra (eigenvectors, covariance), variance concepts, standardization

## 2. Comparative Framing
| Method | PCA | Factor Analysis | t-SNE |
|--------|-----|-----------------|-------|
| **Goal** | Maximum variance | Latent factors | Local structure |
| **Components** | Linear combinations | Unobserved factors | Non-linear manifold |
| **Interpretability** | Moderate (loadings) | High (factors) | Low (visualization) |
| **Use Case** | Dimensionality reduction | Theory-building | Visualization only |

## 3. Examples + Counterexamples

**Simple Example:**  
100 features, 95% variance in first 10 PCs: Reduce from 100D to 10D with minimal information loss

**Failure Case:**  
Non-linear manifold (Swiss roll): PCA applies linear projection, misses curved structure → use kernel PCA or t-SNE

**Edge Case:**  
All features independent with equal variance: PCs identical to original features, no dimensionality reduction benefit

## 4. Layer Breakdown
```
PCA Process:
├─ Preprocessing:
│   ├─ Center: Subtract mean from each feature
│   └─ Scale: Standardize to unit variance (if different scales)
├─ Covariance Matrix: Σ = (1/n)X^T X
├─ Eigendecomposition:
│   ├─ Eigenvectors: vᵢ (principal components, directions)
│   └─ Eigenvalues: λᵢ (variance explained by each PC)
│   └─ Sorted: λ₁ ≥ λ₂ ≥ ... ≥ λₚ
├─ Dimensionality Selection:
│   ├─ Scree plot: Elbow in eigenvalue curve
│   ├─ Cumulative variance: Retain PCs explaining 80-95%
│   └─ Kaiser criterion: Keep λᵢ > 1 (for correlation matrix)
├─ Transformation:
│   └─ Z = X × V_k [project onto k principal components]
└─ Interpretation:
    ├─ Loadings: Correlation between PC and original features
    ├─ Scores: Coordinates in PC space
    └─ Biplot: Visualize both observations and variables
```

**Interaction:** Standardize → Compute covariance → Extract eigenvectors → Project data → Reduce dimensions

## 5. Mini-Project
PCA on iris dataset with visualization:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

# Load data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("Original data shape:", X.shape)
print("Features:", feature_names)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Variance explained
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\nVariance explained by each PC:")
for i, var in enumerate(explained_var):
    print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
print(f"Cumulative variance (first 2 PCs): {cumulative_var[1]:.3f}")

# Loadings (eigenvectors scaled by sqrt of eigenvalues)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

print("\nLoadings (correlation with PCs):")
loadings_df = pd.DataFrame(
    loadings[:, :2],
    columns=['PC1', 'PC2'],
    index=feature_names
)
print(loadings_df.round(3))

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Scree plot
axes[0, 0].bar(range(1, len(explained_var)+1), explained_var, alpha=0.7)
axes[0, 0].plot(range(1, len(explained_var)+1), explained_var, 'ro-', linewidth=2)
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Variance Explained')
axes[0, 0].set_title('Scree Plot')
axes[0, 0].set_xticks(range(1, len(explained_var)+1))

# Plot 2: Cumulative variance
axes[0, 1].plot(range(1, len(cumulative_var)+1), cumulative_var, 'bo-', linewidth=2)
axes[0, 1].axhline(0.95, color='r', linestyle='--', label='95% threshold')
axes[0, 1].set_xlabel('Number of Components')
axes[0, 1].set_ylabel('Cumulative Variance Explained')
axes[0, 1].set_title('Cumulative Variance Explained')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xticks(range(1, len(cumulative_var)+1))

# Plot 3: PC1 vs PC2 (observations)
colors = ['red', 'green', 'blue']
for i, color, name in zip([0, 1, 2], colors, target_names):
    mask = y == i
    axes[0, 2].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      color=color, label=name, alpha=0.6, s=50)

axes[0, 2].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)')
axes[0, 2].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)')
axes[0, 2].set_title('PCA: First Two Components')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Biplot (observations + loadings)
scale = 3
for i, color, name in zip([0, 1, 2], colors, target_names):
    mask = y == i
    axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      color=color, label=name, alpha=0.4, s=30)

# Add loading vectors
for i, feature in enumerate(feature_names):
    axes[1, 0].arrow(0, 0, loadings[i, 0]*scale, loadings[i, 1]*scale,
                    head_width=0.15, head_length=0.15, fc='black', ec='black')
    axes[1, 0].text(loadings[i, 0]*scale*1.15, loadings[i, 1]*scale*1.15,
                   feature.split(' ')[0], fontsize=9, ha='center')

axes[1, 0].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
axes[1, 0].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
axes[1, 0].set_title('Biplot: Observations + Feature Loadings')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Loadings heatmap
im = axes[1, 1].imshow(pca.components_[:2, :], cmap='RdBu_r', aspect='auto',
                       vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(feature_names)))
axes[1, 1].set_xticklabels([f.split(' ')[0] + '\n' + f.split(' ')[1] 
                             for f in feature_names], rotation=0, fontsize=9)
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_yticklabels(['PC1', 'PC2'])
axes[1, 1].set_title('Component Loadings Heatmap')
plt.colorbar(im, ax=axes[1, 1])

# Plot 6: 3D scatter (first 3 PCs)
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(2, 3, 6, projection='3d')
for i, color, name in zip([0, 1, 2], colors, target_names):
    mask = y == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
              color=color, label=name, alpha=0.6, s=40)

ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({explained_var[2]*100:.1f}%)')
ax.set_title('First Three Principal Components')
ax.legend()

plt.tight_layout()
plt.show()

# Reconstruction error
pca_2d = PCA(n_components=2)
X_reduced = pca_2d.fit_transform(X_scaled)
X_reconstructed = pca_2d.inverse_transform(X_reduced)
reconstruction_error = np.mean((X_scaled - X_reconstructed)**2)
print(f"\nReconstruction error (2 PCs): {reconstruction_error:.4f}")
```

## 6. Challenge Round
When is PCA the wrong tool?
- Non-linear relationships: Use kernel PCA, autoencoders, or manifold learning
- Sparse high-dimensional data: Use sparse PCA or feature selection
- Interpretability critical: PCs are linear combinations, hard to explain
- Supervised learning: Consider supervised dimensionality reduction (LDA)
- Categorical variables: PCA assumes continuous data, use MCA (Multiple Correspondence Analysis)

## 7. Key References
- [PCA Visual Explanation (Setosa)](https://setosa.io/ev/principal-component-analysis/)
- [scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Jolliffe, Principal Component Analysis (Book)](https://www.springer.com/gp/book/9780387954424)

---
**Status:** Core dimensionality reduction method | **Complements:** Factor Analysis, t-SNE, Feature Selection
