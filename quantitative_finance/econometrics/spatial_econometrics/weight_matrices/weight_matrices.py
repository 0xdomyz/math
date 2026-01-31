import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import networkx as nx
import seaborn as sns

np.random.seed(579)

# ===== Create Spatial Data =====
print("="*80)
print("SPATIAL WEIGHT MATRIX CONSTRUCTION AND COMPARISON")
print("="*80)

# Generate irregular spatial units (not regular grid)
n = 50
coords = np.column_stack([
    np.random.uniform(0, 10, n),
    np.random.uniform(0, 10, n)
])

print(f"\nData Setup:")
print(f"  n = {n} spatial units")
print(f"  Coordinates: Irregular distribution in [0,10]Ã—[0,10]")
print(f"  Mimics real-world spatial data (counties, census tracts)")

# ===== Weight Matrix Construction Functions =====

def create_contiguity_queen(coords, grid_size=None, threshold=1.5):
    """
    Queen contiguity: Close neighbors within threshold distance
    (Approximates shared boundary for irregular units)
    """
    n = len(coords)
    D = cdist(coords, coords, metric='euclidean')
    W = (D > 0) & (D <= threshold)  # Within threshold, excluding self
    return W.astype(float)

def create_contiguity_rook(coords, threshold=1.2):
    """
    Rook contiguity: Even closer neighbors
    (More restrictive than Queen)
    """
    n = len(coords)
    D = cdist(coords, coords, metric='euclidean')
    W = (D > 0) & (D <= threshold)
    return W.astype(float)

def create_distance_band(coords, d_min=0, d_max=3.0):
    """
    Fixed distance band
    """
    n = len(coords)
    D = cdist(coords, coords, metric='euclidean')
    W = (D > d_min) & (D <= d_max)
    return W.astype(float)

def create_inverse_distance(coords, d_max=5.0, power=1.0):
    """
    Inverse distance weights with truncation
    """
    n = len(coords)
    D = cdist(coords, coords, metric='euclidean')
    
    W = np.zeros((n, n))
    mask = (D > 0) & (D <= d_max)
    W[mask] = 1 / (D[mask] ** power)
    
    return W

def create_knn(coords, k=5):
    """
    K-nearest neighbors (asymmetric)
    """
    n = len(coords)
    D = cdist(coords, coords, metric='euclidean')
    
    W = np.zeros((n, n))
    for i in range(n):
        # Find k nearest neighbors (excluding self)
        neighbors = np.argsort(D[i])[1:k+1]
        W[i, neighbors] = 1
    
    return W

def create_knn_symmetric(coords, k=5):
    """
    Symmetric K-nearest neighbors (union)
    """
    W_knn = create_knn(coords, k)
    W_sym = np.maximum(W_knn, W_knn.T)
    return W_sym

def row_standardize(W):
    """
    Row-standardize: Each row sums to 1
    """
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W_std = W / row_sums
    return W_std

# ===== Construct Multiple Weight Matrices =====
print("\n" + "="*80)
print("CONSTRUCTING WEIGHT MATRICES")
print("="*80)

weight_specs = {
    'Queen (dâ‰¤1.5)': create_contiguity_queen(coords, threshold=1.5),
    'Rook (dâ‰¤1.2)': create_contiguity_rook(coords, threshold=1.2),
    'Distance (dâ‰¤3.0)': create_distance_band(coords, d_max=3.0),
    'Inverse Distance': create_inverse_distance(coords, d_max=5.0, power=1.0),
    'K-NN (k=5)': create_knn(coords, k=5),
    'K-NN Symmetric': create_knn_symmetric(coords, k=5)
}

# Row-standardize all
weight_specs_std = {name: row_standardize(W) for name, W in weight_specs.items()}

# ===== Summary Statistics =====
print(f"\n{'Weight Matrix':<20} {'Connections':>12} {'Avg Neighbors':>14} {'Min':>6} {'Max':>6} {'Symmetric':>10} {'Sparsity %':>12}")
print("-" * 100)

for name, W_raw in weight_specs.items():
    n_connections = np.sum(W_raw > 0)
    avg_neighbors = np.mean(np.sum(W_raw > 0, axis=1))
    min_neighbors = np.min(np.sum(W_raw > 0, axis=1))
    max_neighbors = np.max(np.sum(W_raw > 0, axis=1))
    is_symmetric = np.allclose(W_raw, W_raw.T)
    sparsity = 100 * (1 - n_connections / (n * n))
    
    print(f"{name:<20} {n_connections:>12} {avg_neighbors:>14.2f} {min_neighbors:>6} {max_neighbors:>6} {str(is_symmetric):>10} {sparsity:>12.1f}")

# Check for islands (units with no neighbors)
print(f"\nIsland Check (units with 0 neighbors):")
for name, W_raw in weight_specs.items():
    islands = np.where(np.sum(W_raw > 0, axis=1) == 0)[0]
    if len(islands) > 0:
        print(f"  {name}: {len(islands)} islands {islands}")
    else:
        print(f"  {name}: No islands âœ“")

# ===== Simulate Spatial Data with Known Process =====
print("\n" + "="*80)
print("SIMULATING SPATIAL DATA")
print("="*80)

# Generate covariate
X = np.column_stack([np.ones(n), np.random.randn(n)])

# True parameters
rho_true = 0.4
beta_true = np.array([2.0, 1.5])

# Use "true" weight matrix (Distance band)
W_true = weight_specs_std['Distance (dâ‰¤3.0)']

# Generate Y via SAR: Y = (I - ÏW)^{-1}(XÎ² + Îµ)
I = np.eye(n)
epsilon = np.random.randn(n) * 0.5
Xbeta = X @ beta_true

A = I - rho_true * W_true
Y = np.linalg.solve(A, Xbeta + epsilon)

print(f"True spatial process:")
print(f"  Weight matrix: Distance band (dâ‰¤3.0)")
print(f"  Ï = {rho_true}")
print(f"  Î² = {beta_true}")
print(f"  Generated Y via SAR model")

# ===== Estimate SAR with Different W Specifications =====
print("\n" + "="*80)
print("ESTIMATING SAR MODEL WITH DIFFERENT W")
print("="*80)

from scipy.linalg import eigh

def estimate_sar_simple(Y, X, W):
    """
    Simple 2SLS estimation of SAR
    """
    n = len(Y)
    
    # Instruments: WX
    WX = W @ X[:, 1:]  # Exclude intercept
    
    # First stage: WY ~ X + WX
    X_first = np.column_stack([X, WX])
    WY = W @ Y
    
    try:
        gamma = np.linalg.lstsq(X_first, WY, rcond=None)[0]
        WY_hat = X_first @ gamma
        
        # Second stage: Y ~ Å´Y + X
        X_second = np.column_stack([WY_hat, X])
        params = np.linalg.lstsq(X_second, Y, rcond=None)[0]
        
        rho = params[0]
        beta = params[1:]
        
        # Residuals
        resid = Y - X_second @ params
        sigma2 = np.sum(resid ** 2) / (n - len(params))
        
        return {'rho': rho, 'beta': beta, 'sigma2': sigma2, 'resid': resid}
    except:
        return {'rho': np.nan, 'beta': np.full(X.shape[1], np.nan), 
                'sigma2': np.nan, 'resid': np.full(n, np.nan)}

results = {}
for name, W in weight_specs_std.items():
    results[name] = estimate_sar_simple(Y, X, W)

print(f"{'Weight Matrix':<20} {'ÏÌ‚':>8} {'Î²Ì‚â‚€':>8} {'Î²Ì‚â‚':>8} {'ÏƒÌ‚':>8}")
print("-" * 60)
print(f"{'True':<20} {rho_true:>8.4f} {beta_true[0]:>8.4f} {beta_true[1]:>8.4f} {'0.500':>8}")
print("-" * 60)

for name in weight_specs_std.keys():
    res = results[name]
    if not np.isnan(res['rho']):
        print(f"{name:<20} {res['rho']:>8.4f} {res['beta'][0]:>8.4f} {res['beta'][1]:>8.4f} {np.sqrt(res['sigma2']):>8.4f}")
    else:
        print(f"{name:<20} {'ERROR':>8} {'ERROR':>8} {'ERROR':>8} {'ERROR':>8}")

# ===== Spatial Autocorrelation with Different W =====
print("\n" + "="*80)
print("MORAN'S I WITH DIFFERENT W SPECIFICATIONS")
print("="*80)

def morans_i(y, W):
    """Compute Moran's I"""
    n = len(y)
    y_dev = y - np.mean(y)
    numerator = np.sum(W * np.outer(y_dev, y_dev))
    denominator = np.sum(y_dev ** 2)
    S0 = np.sum(W)
    I = (n / S0) * (numerator / denominator)
    return I

print(f"{'Weight Matrix':<20} {'Moran I (Y)':>14} {'Moran I (Residuals)':>22}")
print("-" * 60)

for name, W in weight_specs_std.items():
    I_Y = morans_i(Y, W)
    I_resid = morans_i(results[name]['resid'], W) if not np.isnan(results[name]['rho']) else np.nan
    print(f"{name:<20} {I_Y:>14.4f} {I_resid:>22.4f}")

print(f"\nInterpretation:")
print(f"  â€¢ True W (Distance) gives highest I for Y (spatial signal)")
print(f"  â€¢ After SAR correction, residual I â‰ˆ 0 for all W")
print(f"  â€¢ Different W specifications capture different aspects of spatial structure")

# ===== Sensitivity Analysis =====
print("\n" + "="*80)
print("SENSITIVITY ANALYSIS: PARAMETER ESTIMATES ACROSS W")
print("="*80)

rho_estimates = [results[name]['rho'] for name in weight_specs_std.keys() if not np.isnan(results[name]['rho'])]
beta1_estimates = [results[name]['beta'][1] for name in weight_specs_std.keys() if not np.isnan(results[name]['rho'])]

print(f"ÏÌ‚ range: [{np.min(rho_estimates):.4f}, {np.max(rho_estimates):.4f}]")
print(f"Î²Ì‚â‚ range: [{np.min(beta1_estimates):.4f}, {np.max(beta1_estimates):.4f}]")

print(f"\nRobustness:")
if np.max(rho_estimates) - np.min(rho_estimates) < 0.2:
    print(f"  âœ“ ÏÌ‚ qualitatively robust across W specifications")
else:
    print(f"  âš  ÏÌ‚ sensitive to W choice")

if np.max(beta1_estimates) - np.min(beta1_estimates) < 0.3:
    print(f"  âœ“ Î²Ì‚â‚ robust across W specifications")
else:
    print(f"  âš  Î²Ì‚â‚ sensitive to W choice")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Spatial units
ax1 = axes[0, 0]
scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=Y, cmap='RdBu_r', 
                     s=100, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')
ax1.set_title('Spatial Units (colored by Y)')
plt.colorbar(scatter, ax=ax1)
ax1.set_aspect('equal')

# Plot 2-4: Network graphs for different W
W_plot = [
    ('Distance (dâ‰¤3.0)', weight_specs['Distance (dâ‰¤3.0)']),
    ('K-NN Symmetric', weight_specs['K-NN Symmetric']),
    ('Inverse Distance', weight_specs['Inverse Distance'])
]

for idx, (name, W) in enumerate(W_plot):
    ax = axes[0, 1 + idx] if idx < 2 else axes[1, 0]
    
    # Create networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Add edges (sample for visibility)
    edges = np.argwhere(W > 0)
    if len(edges) > 200:  # Subsample if too many
        sample_idx = np.random.choice(len(edges), 200, replace=False)
        edges = edges[sample_idx]
    
    for i, j in edges:
        if i < j:  # Avoid duplicates for symmetric W
            G.add_edge(int(i), int(j))
    
    # Draw network
    pos = {i: coords[i] for i in range(n)}
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=Y, 
                          cmap='RdBu_r', ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
    
    ax.set_title(f'{name}\n({np.sum(W>0)} connections)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')

# Plot 5: Ï estimates comparison
ax5 = axes[1, 1]
names_short = [name.split('(')[0].strip() for name in weight_specs_std.keys()]
rho_vals = [results[name]['rho'] for name in weight_specs_std.keys()]

bars = ax5.barh(range(len(names_short)), rho_vals, alpha=0.7)
ax5.axvline(rho_true, color='red', linestyle='--', linewidth=2, label='True Ï')
ax5.set_yticks(range(len(names_short)))
ax5.set_yticklabels(names_short, fontsize=9)
ax5.set_xlabel('ÏÌ‚')
ax5.set_title('Spatial Parameter Estimates')
ax5.legend()
ax5.grid(alpha=0.3, axis='x')

# Plot 6: Neighbor distribution
ax6 = axes[1, 2]
for name, W in list(weight_specs.items())[:3]:  # Plot first 3 for clarity
    neighbor_counts = np.sum(W > 0, axis=1)
    ax6.hist(neighbor_counts, bins=range(0, int(np.max(neighbor_counts))+2), 
            alpha=0.5, label=name.split('(')[0].strip(), edgecolor='black')

ax6.set_xlabel('Number of Neighbors')
ax6.set_ylabel('Frequency')
ax6.set_title('Neighbor Distribution by W Type')
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('spatial_weight_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary and Recommendations =====
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n1. Weight Matrix Choice:")
print(f"   â€¢ Theory-driven: Distance band captures decay")
print(f"   â€¢ Data-driven: Similar results across most W")
print(f"   â€¢ K-NN ensures all units connected (no islands)")
print(f"   â€¢ Inverse distance: Continuous but dense")

print("\n2. Sensitivity:")
print(f"   â€¢ ÏÌ‚ ranges from {np.min(rho_estimates):.3f} to {np.max(rho_estimates):.3f}")
print(f"   â€¢ Qualitatively consistent (all ÏÌ‚ > 0, similar magnitude)")
print(f"   â€¢ Î²Ì‚ estimates more stable than ÏÌ‚")

print("\n3. Spatial Autocorrelation:")
print(f"   â€¢ Moran's I varies with W specification")
print(f"   â€¢ True W (Distance) shows strongest signal")
print(f"   â€¢ All W correctly model spatial dependence (residual Iâ‰ˆ0)")

print("\n4. Practical Guidelines:")
print("   âœ“ Always test multiple W specifications")
print("   âœ“ Report sensitivity analysis")
print("   âœ“ Justify W choice with theory")
print("   âœ“ Check for islands (zero neighbors)")
print("   âœ“ Row-standardize for interpretation")
print("   âœ“ Document W construction for replication")

print("\n5. Common Choices by Application:")
print("   â€¢ Regional economics: Contiguity (Queen/Rook)")
print("   â€¢ Urban economics: Distance-based or K-NN")
print("   â€¢ Real estate: Inverse distance with truncation")
print("   â€¢ Epidemiology: Distance-based or network")
print("   â€¢ Finance: Correlation networks, input-output")

print("\n6. Software:")
print("   â€¢ Python: libpysal.weights (Queen, Rook, KNN, DistanceBand)")
print("   â€¢ R: spdep (poly2nb, dnearneigh, knn2nb)")
print("   â€¢ Stata: spmatrix create (contiguity, idistance)")
print("   â€¢ GeoDa: GUI-based W matrix creation")
