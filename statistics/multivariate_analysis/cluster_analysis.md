# Cluster Analysis

## 1. Concept Skeleton
**Definition:** Unsupervised learning technique grouping similar observations into clusters based on distance or similarity metrics  
**Purpose:** Pattern discovery, market segmentation, data exploration, taxonomy creation without predefined labels  
**Prerequisites:** Distance metrics (Euclidean, Manhattan), similarity concepts, centroids, dendrograms

## 2. Comparative Framing
| Method | k-Means | Hierarchical | DBSCAN |
|--------|---------|--------------|--------|
| **Shape** | Spherical clusters | Any hierarchy | Arbitrary density |
| **k Required** | Yes (pre-specified) | No (cut dendrogram) | No (density-based) |
| **Scalability** | Fast (large n) | Slow (O(n²) memory) | Moderate |
| **Outliers** | Assigned to cluster | Can form singletons | Labeled as noise |

## 3. Examples + Counterexamples

**Simple Example:**  
Customer segmentation: Group buyers by purchase history → 3 clusters: budget, moderate, premium shoppers

**Failure Case:**  
Non-convex clusters (concentric circles): k-means splits into arbitrary radial segments, misses true structure

**Edge Case:**  
All points equidistant: Any clustering equally valid, no meaningful pattern exists

## 4. Layer Breakdown
```
Cluster Analysis Components:
├─ Distance Metrics:
│   ├─ Euclidean: √Σ(xᵢ-yᵢ)² [default, sensitive to scale]
│   ├─ Manhattan: Σ|xᵢ-yᵢ| [robust to outliers]
│   ├─ Cosine: 1 - (x·y)/(||x|| ||y||) [angle-based]
│   └─ Correlation: 1 - corr(x,y) [pattern similarity]
├─ k-Means Algorithm:
│   ├─ Initialize: k random centroids
│   ├─ Assign: Each point to nearest centroid
│   ├─ Update: Recalculate centroids as cluster means
│   └─ Iterate: Until convergence (no reassignments)
│   └─ Objective: Minimize within-cluster sum of squares (WCSS)
├─ Hierarchical Clustering:
│   ├─ Agglomerative: Bottom-up merging
│   │   └─ Linkage: Single, Complete, Average, Ward
│   ├─ Divisive: Top-down splitting
│   └─ Output: Dendrogram (tree structure)
├─ Cluster Validation:
│   ├─ Elbow Method: Plot WCSS vs k, find "elbow"
│   ├─ Silhouette Score: (b-a)/max(a,b) ∈ [-1,1]
│   │   └─ a: within-cluster distance, b: nearest-cluster distance
│   ├─ Gap Statistic: Compare to null reference distribution
│   └─ Domain Validation: Clusters make practical sense
└─ Applications:
    ├─ Market segmentation
    ├─ Image compression
    ├─ Anomaly detection
    └─ Document clustering
```

**Interaction:** Compute distances → Group similar → Validate structure → Iterate or finalize

## 5. Mini-Project
k-Means and Hierarchical clustering with evaluation:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.datasets import make_blobs, make_moons
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Generate datasets
np.random.seed(42)

# Dataset 1: Well-separated blobs (ideal for k-means)
X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, 
                              cluster_std=0.6, random_state=42)

# Dataset 2: Non-convex (challenging for k-means)
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

# Standardize
scaler = StandardScaler()
X_blobs_scaled = scaler.fit_transform(X_blobs)
X_moons_scaled = scaler.fit_transform(X_moons)

# k-Means on blobs: Find optimal k
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_blobs_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_blobs_scaled, kmeans.labels_))

print("k-Means Evaluation (Blob Data):")
for k, sil in zip(k_range, silhouette_scores):
    print(f"  k={k}: Silhouette={sil:.3f}")

# Fit final k-means with k=4
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_kmeans = kmeans_final.fit_predict(X_blobs_scaled)

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_hier = hierarchical.fit_predict(X_blobs_scaled)

# DBSCAN on non-convex data
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_moons_scaled)

print(f"\nDBSCAN on non-convex data:")
print(f"  Number of clusters: {len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)}")
print(f"  Number of noise points: {list(labels_dbscan).count(-1)}")

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(16, 14))

# Row 1: k-Means evaluation
axes[0, 0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('WCSS (Within-Cluster Sum of Squares)')
axes[0, 0].set_title('Elbow Method')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Analysis')
axes[0, 1].grid(alpha=0.3)

# Dendrogram
linked = linkage(X_blobs_scaled, method='ward')
dendrogram(linked, ax=axes[0, 2], no_labels=True)
axes[0, 2].set_title('Hierarchical Clustering Dendrogram')
axes[0, 2].set_xlabel('Sample Index')
axes[0, 2].set_ylabel('Distance')

# Row 2: k-Means results
scatter1 = axes[1, 0].scatter(X_blobs[:, 0], X_blobs[:, 1], 
                             c=labels_kmeans, cmap='viridis', alpha=0.6, s=50)
centers = scaler.inverse_transform(kmeans_final.cluster_centers_)
axes[1, 0].scatter(centers[:, 0], centers[:, 1], 
                  c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[1, 0].set_title(f'k-Means (k=4)\nSilhouette={silhouette_scores[2]:.3f}')
axes[1, 0].set_xlabel('Feature 1')
axes[1, 0].set_ylabel('Feature 2')

# Hierarchical results
scatter2 = axes[1, 1].scatter(X_blobs[:, 0], X_blobs[:, 1], 
                             c=labels_hier, cmap='viridis', alpha=0.6, s=50)
sil_hier = silhouette_score(X_blobs_scaled, labels_hier)
axes[1, 1].set_title(f'Hierarchical (Ward)\nSilhouette={sil_hier:.3f}')
axes[1, 1].set_xlabel('Feature 1')
axes[1, 1].set_ylabel('Feature 2')

# True labels
scatter3 = axes[1, 2].scatter(X_blobs[:, 0], X_blobs[:, 1], 
                             c=y_blobs, cmap='viridis', alpha=0.6, s=50)
axes[1, 2].set_title('True Labels (Reference)')
axes[1, 2].set_xlabel('Feature 1')
axes[1, 2].set_ylabel('Feature 2')

# Row 3: Non-convex data (moons)
# k-Means (poor)
kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_kmeans_moons = kmeans_moons.fit_predict(X_moons_scaled)
scatter4 = axes[2, 0].scatter(X_moons[:, 0], X_moons[:, 1], 
                             c=labels_kmeans_moons, cmap='viridis', alpha=0.6, s=50)
sil_kmeans_moons = silhouette_score(X_moons_scaled, labels_kmeans_moons)
axes[2, 0].set_title(f'k-Means on Non-Convex\nSilhouette={sil_kmeans_moons:.3f} (Poor)')
axes[2, 0].set_xlabel('Feature 1')
axes[2, 0].set_ylabel('Feature 2')

# DBSCAN (good)
scatter5 = axes[2, 1].scatter(X_moons[:, 0], X_moons[:, 1], 
                             c=labels_dbscan, cmap='viridis', alpha=0.6, s=50)
# Exclude noise points (-1) from silhouette
mask = labels_dbscan != -1
if mask.sum() > 1 and len(set(labels_dbscan[mask])) > 1:
    sil_dbscan = silhouette_score(X_moons_scaled[mask], labels_dbscan[mask])
    axes[2, 1].set_title(f'DBSCAN on Non-Convex\nSilhouette={sil_dbscan:.3f} (Good)')
else:
    axes[2, 1].set_title('DBSCAN on Non-Convex')
axes[2, 1].set_xlabel('Feature 1')
axes[2, 1].set_ylabel('Feature 2')

# True labels (moons)
scatter6 = axes[2, 2].scatter(X_moons[:, 0], X_moons[:, 1], 
                             c=y_moons, cmap='viridis', alpha=0.6, s=50)
axes[2, 2].set_title('True Labels (Non-Convex)')
axes[2, 2].set_xlabel('Feature 1')
axes[2, 2].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Detailed metrics
print("\nDetailed Metrics (Blob Data, k=4):")
print(f"k-Means:")
print(f"  Silhouette Score: {silhouette_scores[2]:.3f}")
print(f"  Davies-Bouldin Index: {davies_bouldin_score(X_blobs_scaled, labels_kmeans):.3f}")
print(f"Hierarchical:")
print(f"  Silhouette Score: {sil_hier:.3f}")
print(f"  Davies-Bouldin Index: {davies_bouldin_score(X_blobs_scaled, labels_hier):.3f}")
```

## 6. Challenge Round
When is cluster analysis the wrong tool?
- Labeled data available: Use supervised learning (classification)
- Clusters not natural: Forcing structure where none exists misleading
- High-dimensional sparse data: Distance metrics break down (curse of dimensionality)
- Temporal dependencies: Use time series clustering or HMMs
- Clear causal model: Use latent class analysis or mixture models with theory

## 7. Key References
- [k-Means Visualization (Stanford)](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)
- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Clustering Performance Evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

---
**Status:** Core unsupervised learning method | **Complements:** PCA, Classification, Anomaly Detection
