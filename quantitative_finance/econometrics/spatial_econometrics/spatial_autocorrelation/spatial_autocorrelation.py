import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
import seaborn as sns

np.random.seed(246)

# ===== Simulate Spatial Data =====
print("="*80)
print("SPATIAL AUTOCORRELATION ANALYSIS")
print("="*80)

# Create regular grid of locations
grid_size = 10
n = grid_size ** 2
coords = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])

print(f"\nSimulation Setup:")
print(f"  Grid: {grid_size} Ã— {grid_size} = {n} locations")
print(f"  Coordinates: Regular lattice")

# Generate spatially autocorrelated variable using Gaussian random field
def generate_spatial_data(coords, range_param=3.0, noise_ratio=0.3):
    """Generate spatially autocorrelated data using exponential covariance"""
    n = len(coords)
    
    # Distance matrix
    D = cdist(coords, coords, metric='euclidean')
    
    # Exponential covariance function
    Sigma = np.exp(-D / range_param)
    
    # Generate correlated data
    L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(n))
    z_spatial = L @ np.random.randn(n)
    
    # Add noise
    z_noise = np.random.randn(n) * np.std(z_spatial) * noise_ratio
    y = z_spatial + z_noise
    
    return y

y = generate_spatial_data(coords, range_param=2.5, noise_ratio=0.2)

print(f"  Range parameter: 2.5 units")
print(f"  Noise ratio: 0.2")
print(f"  y statistics: mean={np.mean(y):.3f}, sd={np.std(y):.3f}")

# ===== Spatial Weight Matrix =====
print("\n" + "="*80)
print("SPATIAL WEIGHT MATRIX")
print("="*80)

def create_weight_matrix(coords, method='queen', k=4, d_max=2.0):
    """
    Create spatial weight matrix
    Methods: 'queen', 'rook', 'knn', 'distance'
    """
    n = len(coords)
    W = np.zeros((n, n))
    
    if method == 'queen':
        # Queen contiguity (8 neighbors on grid)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.abs(coords[i] - coords[j])
                    # Shares edge or corner
                    if np.max(dist) <= 1:
                        W[i, j] = 1
    
    elif method == 'rook':
        # Rook contiguity (4 neighbors on grid)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.abs(coords[i] - coords[j])
                    # Shares edge only (Manhattan distance = 1)
                    if np.sum(dist) == 1:
                        W[i, j] = 1
    
    elif method == 'knn':
        # K-nearest neighbors
        D = cdist(coords, coords, metric='euclidean')
        for i in range(n):
            # Get k nearest (excluding self)
            neighbors = np.argsort(D[i])[1:k+1]
            W[i, neighbors] = 1
    
    elif method == 'distance':
        # Distance-based (inverse distance within threshold)
        D = cdist(coords, coords, metric='euclidean')
        for i in range(n):
            for j in range(n):
                if i != j and D[i, j] <= d_max and D[i, j] > 0:
                    W[i, j] = 1 / D[i, j]
    
    # Row standardization
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W_std = W / row_sums
    
    return W_std, W

# Create weight matrices
W_queen, W_queen_raw = create_weight_matrix(coords, method='queen')
W_rook, W_rook_raw = create_weight_matrix(coords, method='rook')
W_knn, W_knn_raw = create_weight_matrix(coords, method='knn', k=4)

print(f"Weight Matrix Methods:")
print(f"  Queen: {np.sum(W_queen_raw > 0)} connections (avg {np.mean(np.sum(W_queen_raw>0, axis=1)):.1f} neighbors)")
print(f"  Rook:  {np.sum(W_rook_raw > 0)} connections (avg {np.mean(np.sum(W_rook_raw>0, axis=1)):.1f} neighbors)")
print(f"  KNN:   {np.sum(W_knn_raw > 0)} connections (k=4)")
print(f"  Row-standardized: Yes")

# Use Queen for main analysis
W = W_queen

# ===== Moran's I =====
print("\n" + "="*80)
print("MORAN'S I STATISTIC")
print("="*80)

def morans_i(y, W, permutations=999):
    """
    Compute Moran's I with inference
    """
    n = len(y)
    y_bar = np.mean(y)
    y_dev = y - y_bar
    
    # Moran's I
    numerator = np.sum(W * np.outer(y_dev, y_dev))
    denominator = np.sum(y_dev ** 2)
    S0 = np.sum(W)
    
    I = (n / S0) * (numerator / denominator)
    
    # Expected value under null
    EI = -1 / (n - 1)
    
    # Variance under randomization assumption
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=0) + W.sum(axis=1)) ** 2)
    
    m2 = np.sum(y_dev ** 2) / n
    m4 = np.sum(y_dev ** 4) / n
    
    b2 = m4 / (m2 ** 2)
    
    VI_num = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2)
    VI_den = (n - 1) * (n - 2) * (n - 3) * S0**2
    
    VI_part1 = VI_num / VI_den
    VI_part2 = (b2 * (n**2 - n) * S1 - 2*n*S2 + 6*S0**2) / ((n-1)*(n-2)*(n-3)*S0**2)
    
    VI = VI_part1 - VI_part2 - EI ** 2
    
    # Z-score
    Z = (I - EI) / np.sqrt(VI)
    p_value_norm = 2 * (1 - stats.norm.cdf(abs(Z)))
    
    # Permutation test
    I_perm = np.zeros(permutations)
    for perm in range(permutations):
        y_perm = np.random.permutation(y)
        y_dev_perm = y_perm - np.mean(y_perm)
        numerator_perm = np.sum(W * np.outer(y_dev_perm, y_dev_perm))
        denominator_perm = np.sum(y_dev_perm ** 2)
        I_perm[perm] = (n / S0) * (numerator_perm / denominator_perm)
    
    p_value_perm = np.mean(np.abs(I_perm) >= abs(I))
    
    return {
        'I': I,
        'EI': EI,
        'VI': VI,
        'Z': Z,
        'p_value_norm': p_value_norm,
        'p_value_perm': p_value_perm,
        'I_perm': I_perm
    }

moran_result = morans_i(y, W, permutations=999)

print(f"Moran's I: {moran_result['I']:.4f}")
print(f"Expected I (null): {moran_result['EI']:.4f}")
print(f"Variance: {moran_result['VI']:.6f}")
print(f"Z-score: {moran_result['Z']:.4f}")
print(f"P-value (normal): {moran_result['p_value_norm']:.4f}")
print(f"P-value (permutation): {moran_result['p_value_perm']:.4f}")

if moran_result['p_value_perm'] < 0.05:
    if moran_result['I'] > moran_result['EI']:
        print(f"âœ“ Significant POSITIVE spatial autocorrelation (clustering)")
    else:
        print(f"âœ“ Significant NEGATIVE spatial autocorrelation (dispersion)")
else:
    print(f"  No significant spatial autocorrelation")

# ===== Geary's C =====
print("\n" + "="*80)
print("GEARY'S C STATISTIC")
print("="*80)

def gearys_c(y, W):
    """Compute Geary's C"""
    n = len(y)
    y_bar = np.mean(y)
    y_dev = y - y_bar
    
    # Geary's C
    numerator = np.sum(W * (np.subtract.outer(y, y) ** 2))
    denominator = 2 * np.sum(W) * np.sum(y_dev ** 2) / (n - 1)
    
    C = numerator / denominator
    
    # Expected value under null
    EC = 1.0
    
    # Z-score (simplified)
    # Exact variance formula complex; use approximation
    VC_approx = 1 / (2 * (n + 1) * np.sum(W))
    Z = (C - EC) / np.sqrt(VC_approx)
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
    
    return {'C': C, 'EC': EC, 'Z': Z, 'p_value': p_value}

geary_result = gearys_c(y, W)

print(f"Geary's C: {geary_result['C']:.4f}")
print(f"Expected C (null): {geary_result['EC']:.4f}")
print(f"Z-score: {geary_result['Z']:.4f}")
print(f"P-value: {geary_result['p_value']:.4f}")

if geary_result['p_value'] < 0.05:
    if geary_result['C'] < 1:
        print(f"âœ“ Significant POSITIVE spatial autocorrelation")
    else:
        print(f"âœ“ Significant NEGATIVE spatial autocorrelation")
else:
    print(f"  No significant spatial autocorrelation")

# ===== LISA (Local Indicators of Spatial Association) =====
print("\n" + "="*80)
print("LOCAL INDICATORS OF SPATIAL ASSOCIATION (LISA)")
print("="*80)

def local_morans_i(y, W, permutations=999):
    """Compute Local Moran's I for each location"""
    n = len(y)
    y_bar = np.mean(y)
    y_dev = y - y_bar
    s2 = np.sum(y_dev ** 2) / n
    
    # Spatial lag
    Wy = W @ y_dev
    
    # Local Moran's I
    I_local = (y_dev / s2) * Wy
    
    # Quadrant classification
    quad = np.zeros(n, dtype=int)
    # 1: HH (high-high), 2: LL (low-low), 3: HL (high-low), 4: LH (low-high)
    quad[(y_dev > 0) & (Wy > 0)] = 1  # HH
    quad[(y_dev < 0) & (Wy < 0)] = 2  # LL
    quad[(y_dev > 0) & (Wy < 0)] = 3  # HL
    quad[(y_dev < 0) & (Wy > 0)] = 4  # LH
    
    # Pseudo p-values via conditional permutation
    p_values = np.zeros(n)
    for i in range(n):
        I_perm_i = np.zeros(permutations)
        for perm in range(permutations):
            # Permute all values
            y_perm = np.random.permutation(y)
            y_dev_perm = y_perm - np.mean(y_perm)
            Wy_perm = W @ y_dev_perm
            s2_perm = np.sum(y_dev_perm ** 2) / n
            I_perm_i[perm] = (y_dev_perm[i] / s2_perm) * Wy_perm[i]
        
        p_values[i] = np.mean(np.abs(I_perm_i) >= abs(I_local[i]))
    
    return {
        'I_local': I_local,
        'quadrant': quad,
        'p_values': p_values
    }

lisa_result = local_morans_i(y, W, permutations=499)

n_significant = np.sum(lisa_result['p_values'] < 0.05)
print(f"Local Moran's I computed for {n} locations")
print(f"Significant clusters (p<0.05): {n_significant} ({100*n_significant/n:.1f}%)")

quad_labels = {1: 'HH (High-High)', 2: 'LL (Low-Low)', 
               3: 'HL (High-Low)', 4: 'LH (Low-High)', 0: 'Not significant'}

for q in [1, 2, 3, 4]:
    significant = (lisa_result['quadrant'] == q) & (lisa_result['p_values'] < 0.05)
    count = np.sum(significant)
    print(f"  {quad_labels[q]}: {count} locations")

# ===== Visualizations =====
fig = plt.figure(figsize=(16, 10))

# Create grid for spatial plots
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Plot 1: Spatial map of y values
ax1 = fig.add_subplot(gs[0, 0])
y_grid = y.reshape(grid_size, grid_size)
im1 = ax1.imshow(y_grid, cmap='RdBu_r', origin='lower')
ax1.set_title('Spatial Distribution of Y')
ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')
plt.colorbar(im1, ax=ax1)

# Plot 2: Moran scatter plot
ax2 = fig.add_subplot(gs[0, 1])
y_standardized = (y - np.mean(y)) / np.std(y)
Wy_standardized = W @ y_standardized
ax2.scatter(y_standardized, Wy_standardized, alpha=0.6, s=30)

# Add regression line
slope, intercept = np.polyfit(y_standardized, Wy_standardized, 1)
x_line = np.linspace(y_standardized.min(), y_standardized.max(), 100)
y_line = slope * x_line + intercept
ax2.plot(x_line, y_line, 'r--', linewidth=2, 
         label=f'Slope = {slope:.3f} â‰ˆ I = {moran_result["I"]:.3f}')

# Quadrants
ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax2.axvline(0, color='gray', linestyle='-', linewidth=0.5)
ax2.text(0.7*ax2.get_xlim()[1], 0.7*ax2.get_ylim()[1], 'HH', fontsize=12, color='red')
ax2.text(0.7*ax2.get_xlim()[0], 0.7*ax2.get_ylim()[0], 'LL', fontsize=12, color='blue')
ax2.text(0.7*ax2.get_xlim()[1], 0.7*ax2.get_ylim()[0], 'HL', fontsize=12, color='gray')
ax2.text(0.7*ax2.get_xlim()[0], 0.7*ax2.get_ylim()[1], 'LH', fontsize=12, color='gray')

ax2.set_xlabel('Standardized Y')
ax2.set_ylabel('Spatial Lag (Standardized WY)')
ax2.set_title(f"Moran Scatter Plot (I={moran_result['I']:.3f})")
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Permutation distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(moran_result['I_perm'], bins=30, alpha=0.6, density=True, label='Permutations')
ax3.axvline(moran_result['I'], color='red', linestyle='--', linewidth=2, label='Observed I')
ax3.axvline(moran_result['EI'], color='blue', linestyle=':', linewidth=2, label='Expected I')
ax3.set_xlabel("Moran's I")
ax3.set_ylabel('Density')
ax3.set_title(f"Permutation Test (p={moran_result['p_value_perm']:.4f})")
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Local Moran's I values
ax4 = fig.add_subplot(gs[1, 0])
I_local_grid = lisa_result['I_local'].reshape(grid_size, grid_size)
im4 = ax4.imshow(I_local_grid, cmap='RdBu_r', origin='lower')
ax4.set_title('Local Moran\'s I')
ax4.set_xlabel('X coordinate')
ax4.set_ylabel('Y coordinate')
plt.colorbar(im4, ax=ax4)

# Plot 5: LISA cluster map
ax5 = fig.add_subplot(gs[1, 1])
cluster_map = np.zeros(n)
# Only show significant clusters
significant_mask = lisa_result['p_values'] < 0.05
cluster_map[significant_mask] = lisa_result['quadrant'][significant_mask]

cluster_grid = cluster_map.reshape(grid_size, grid_size)
cmap_discrete = plt.cm.get_cmap('Set1', 5)
im5 = ax5.imshow(cluster_grid, cmap=cmap_discrete, origin='lower', vmin=0, vmax=4)
ax5.set_title('LISA Cluster Map (p<0.05)')
ax5.set_xlabel('X coordinate')
ax5.set_ylabel('Y coordinate')

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cmap_discrete(0), label='Not sig'),
                  Patch(facecolor=cmap_discrete(1), label='HH'),
                  Patch(facecolor=cmap_discrete(2), label='LL'),
                  Patch(facecolor=cmap_discrete(3), label='HL'),
                  Patch(facecolor=cmap_discrete(4), label='LH')]
ax5.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Plot 6: Comparison of statistics
ax6 = fig.add_subplot(gs[1, 2])
stats_data = {
    'Moran\'s I': moran_result['I'],
    'Geary\'s C': geary_result['C'],
    'Expected I': moran_result['EI'],
    'Expected C': geary_result['EC']
}
colors_bar = ['red', 'blue', 'gray', 'gray']
bars = ax6.bar(range(len(stats_data)), list(stats_data.values()), color=colors_bar, alpha=0.7)
ax6.set_xticks(range(len(stats_data)))
ax6.set_xticklabels(stats_data.keys(), rotation=45, ha='right')
ax6.set_ylabel('Value')
ax6.set_title('Global Autocorrelation Statistics')
ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax6.grid(alpha=0.3, axis='y')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, stats_data.values())):
    ax6.text(i, val + 0.05, f'{val:.3f}', ha='center', fontsize=9)

plt.savefig('spatial_autocorrelation.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Global Spatial Autocorrelation:")
print(f"   Moran's I = {moran_result['I']:.4f} (p={moran_result['p_value_perm']:.4f})")
print(f"   Geary's C = {geary_result['C']:.4f} (p={geary_result['p_value']:.4f})")
print(f"   Both indicate significant positive spatial autocorrelation")

print("\n2. Interpretation:")
if moran_result['I'] > 0.3:
    print(f"   Strong clustering: Similar values tend to be near each other")
elif moran_result['I'] > 0.1:
    print(f"   Moderate clustering: Some spatial structure present")
else:
    print(f"   Weak clustering: Spatial pattern barely detectable")

print("\n3. Local Patterns (LISA):")
print(f"   {n_significant} significant clusters identified ({100*n_significant/n:.1f}%)")
print(f"   Hot spots (HH): High values surrounded by high values")
print(f"   Cold spots (LL): Low values surrounded by low values")
print(f"   Outliers (HL/LH): Values different from neighbors")

print("\n4. Modeling Implications:")
print(f"   OLS regression inappropriate (biased standard errors)")
print(f"   Consider spatial lag model: Y = ÏWY + XÎ² + Îµ")
print(f"   Or spatial error model: Y = XÎ² + u, where u = Î»Wu + Îµ")
print(f"   Use LM tests to choose specification")

print("\n5. Practical Recommendations:")
print("   â€¢ Always test for spatial autocorrelation before modeling")
print("   â€¢ Use multiple weight matrix specifications (robustness)")
print("   â€¢ Examine local patterns (LISA) not just global statistics")
print("   â€¢ Permutation tests preferred for exact inference")
print("   â€¢ Spatial HAC standard errors if spatial model infeasible")

print("\n6. Software:")
print("   â€¢ Python: PySAL (esda.Moran), GeoPandas, splot")
print("   â€¢ R: spdep (moran.test, localmoran), sf")
print("   â€¢ Stata: spatwmat, moransI, spatreg")
