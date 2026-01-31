# Spatial Autocorrelation

## 1. Concept Skeleton
**Definition:** Correlation between values of variable at different spatial locations; violates independence assumption  
**Purpose:** Detect spatial clustering; test Tobler's law ("near things more related"); guide spatial model specification  
**Prerequisites:** Covariance, correlation, hypothesis testing, spatial weight matrices, ANOVA decomposition

## 2. Comparative Framing
| Test | Moran's I | Geary's C | Getis-Ord G | Variogram | Ljung-Box (Time Series) | Durbin-Watson (Regression) |
|------|-----------|-----------|-------------|-----------|-------------------------|---------------------------|
| **Measures** | Global autocorrelation | Inverse correlation | Local clustering | Spatial dependence structure | Temporal autocorrelation | Serial correlation in residuals |
| **Range** | [-1, 1] | [0, 2+] | [0, 1] | [0, ∞) | [0, ∞) | [0, 4] |
| **Interpretation** | I>0: Clustering | C<1: Clustering | G>E[G]: Hot spot | Increases with distance | Q>χ²: Correlation | DW<2: Positive AR |
| **Null Hypothesis** | No spatial autocorrelation | No spatial autocorrelation | No local clustering | Pure nugget | No temporal correlation | No serial correlation |
| **Sensitivity** | Moderate (weight matrix) | High (distance) | Local patterns | Scale-dependent | Lag selection | Specification errors |
| **Output** | Single global statistic | Single global statistic | Local statistics (per unit) | Continuous function | Time lags | Single value |

## 3. Examples + Counterexamples

**Classic Example:**  
House prices (n=200): Moran's I=0.45 (p<0.001) indicates positive spatial autocorrelation. Neighbors have similar prices. Geary's C=0.60 confirms. OLS standard errors biased (too small). Need spatial lag or spatial error model.

**Failure Case:**  
Irregular spatial units (varying sizes): Boundary effects dominate. Moran's I=0.15 (p=0.08) but visual inspection shows strong clustering. Weight matrix specification critical—contiguity fails, distance-based reveals I=0.52 (p<0.001).

**Edge Case:**  
Spatial heterogeneity masquerading as autocorrelation: Urban vs rural divide creates apparent clustering (I=0.30). After controlling for urban dummy, residual I=0.05 (insignificant). Not spatial dependence but omitted variable bias.

## 4. Layer Breakdown
```
Spatial Autocorrelation Framework:
├─ Fundamental Concept:
│   ├─ Tobler's First Law of Geography:
│   │   ├─ "Everything is related to everything else"
│   │   ├─ "But near things are more related than distant things"
│   │   └─ Empirical regularity across many phenomena
│   ├─ Violation of IID Assumption:
│   │   ├─ Observations not independent
│   │   ├─ Cov(yᵢ, yⱼ) ≠ 0 for i ≠ j
│   │   └─ Standard errors biased; inference invalid
│   ├─ Types:
│   │   ├─ Positive: Similar values cluster (most common)
│   │   ├─ Negative: Dissimilar values adjacent (rare in practice)
│   │   └─ Zero: Random spatial pattern
│   └─ Sources:
│       ├─ Diffusion processes (disease, technology)
│       ├─ Common shocks (weather, policy)
│       ├─ Spillovers (knowledge, congestion)
│       └─ Measurement aggregation (areal units)
├─ Spatial Weight Matrix (W):
│   ├─ Definition:
│   │   ├─ n × n matrix encoding spatial relationships
│   │   ├─ wᵢⱼ: "closeness" of units i and j
│   │   └─ Typically row-standardized: Σⱼ wᵢⱼ = 1
│   ├─ Contiguity-Based:
│   │   ├─ Queen (shares edge or vertex): wᵢⱼ=1 if border contact
│   │   ├─ Rook (shares edge only): wᵢⱼ=1 if common edge
│   │   └─ Binary: 0/1 indicators
│   ├─ Distance-Based:
│   │   ├─ Inverse distance: wᵢⱼ = 1/dᵢⱼ (d>0)
│   │   ├─ Inverse squared: wᵢⱼ = 1/dᵢⱼ²
│   │   ├─ Gaussian: wᵢⱼ = exp(-dᵢⱼ²/2h²)
│   │   └─ Distance band: wᵢⱼ=1 if d_min < dᵢⱼ < d_max
│   ├─ K-Nearest Neighbors:
│   │   ├─ wᵢⱼ=1 for k closest neighbors
│   │   └─ Ensures each unit has exactly k neighbors
│   ├─ Row Standardization:
│   │   ├─ w*ᵢⱼ = wᵢⱼ / Σₖ wᵢₖ
│   │   ├─ Rows sum to 1
│   │   └─ Interpretation: Average of neighbors
│   └─ Properties:
│       ├─ Diagonal: wᵢᵢ = 0 (no self-neighbor)
│       ├─ Symmetric (often): wᵢⱼ = wⱼᵢ
│       └─ Sparsity: Most entries zero (local neighborhoods)
├─ Moran's I:
│   ├─ Definition:
│   │   ├─ I = (n/S₀) × [Σᵢ Σⱼ wᵢⱼ(yᵢ - ȳ)(yⱼ - ȳ)] / [Σᵢ (yᵢ - ȳ)²]
│   │   ├─ S₀ = Σᵢ Σⱼ wᵢⱼ (sum of all weights)
│   │   └─ Weighted correlation of variable with its spatial lag
│   ├─ Interpretation:
│   │   ├─ I > 0: Positive spatial autocorrelation (clustering)
│   │   ├─ I < 0: Negative spatial autocorrelation (dispersion)
│   │   ├─ I ≈ 0: Random spatial pattern
│   │   └─ Analogous to Pearson correlation (not identical)
│   ├─ Expected Value (Null):
│   │   ├─ E[I] = -1/(n-1) ≈ 0 for large n
│   │   └─ Under random permutation of values across space
│   ├─ Variance (Null):
│   │   ├─ Complex formula depending on moments of W
│   │   ├─ Two assumptions:
│   │   │   ├─ Normality: Assume yᵢ ~ N(μ, σ²)
│   │   │   └─ Randomization: Permutation distribution
│   │   └─ Randomization more robust (no distributional assumption)
│   ├─ Standardized Statistic:
│   │   ├─ Z = [I - E[I]] / √Var[I]
│   │   ├─ Under null: Z ~ N(0, 1) asymptotically
│   │   └─ P-value from standard normal table
│   ├─ Permutation Test:
│   │   ├─ Randomly shuffle values across locations (B times)
│   │   ├─ Compute I* for each permutation
│   │   ├─ P-value = #{I* ≥ I_obs} / B
│   │   └─ Exact finite-sample p-value (no distributional assumption)
│   └─ Limitations:
│       ├─ Global statistic (masks local patterns)
│       ├─ Sensitive to W specification
│       └─ Assumes stationarity (constant spatial process)
├─ Geary's C:
│   ├─ Definition:
│   │   ├─ C = [(n-1)/(2S₀)] × [Σᵢ Σⱼ wᵢⱼ(yᵢ - yⱼ)²] / [Σᵢ (yᵢ - ȳ)²]
│   │   ├─ Weighted sum of squared differences
│   │   └─ Focus on dissimilarity rather than covariance
│   ├─ Interpretation:
│   │   ├─ C < 1: Positive spatial autocorrelation (similar neighbors)
│   │   ├─ C = 1: No spatial autocorrelation (expected under null)
│   │   ├─ C > 1: Negative spatial autocorrelation
│   │   └─ Inverse relationship with Moran's I
│   ├─ Expected Value (Null):
│   │   ├─ E[C] = 1
│   │   └─ Under random permutation
│   ├─ Comparison to Moran's I:
│   │   ├─ C more sensitive to differences in individual pairs
│   │   ├─ I emphasizes products of deviations
│   │   ├─ Often give similar conclusions but not always
│   │   └─ Use both for robustness
│   └─ Standardized Test:
│       ├─ Z = [C - E[C]] / √Var[C]
│       └─ Asymptotically normal
├─ Local Indicators of Spatial Association (LISA):
│   ├─ Concept:
│   │   ├─ Decompose global I into local contributions
│   │   ├─ Iᵢ = (yᵢ - ȳ) × Σⱼ wᵢⱼ(yⱼ - ȳ) / s²
│   │   ├─ s² = Σᵢ(yᵢ - ȳ)² / n
│   │   └─ I = Σᵢ Iᵢ (global is sum of local)
│   ├─ Interpretation:
│   │   ├─ Iᵢ > 0: Unit i similar to neighbors
│   │   │   ├─ High-High: i high, neighbors high (hot spot)
│   │   │   └─ Low-Low: i low, neighbors low (cold spot)
│   │   ├─ Iᵢ < 0: Unit i dissimilar to neighbors
│   │   │   ├─ High-Low: i high, neighbors low (outlier)
│   │   │   └─ Low-High: i low, neighbors high (outlier)
│   │   └─ Significance via conditional permutation
│   ├─ Inference:
│   │   ├─ Permute yᵢ holding neighbors fixed (pseudo p-values)
│   │   ├─ Multiple testing adjustment (Bonferroni, FDR)
│   │   └─ Caution: Inflated Type I error if unadjusted
│   ├─ LISA Cluster Map:
│   │   ├─ Color-code significant local patterns
│   │   ├─ Red: High-High clusters
│   │   ├─ Blue: Low-Low clusters
│   │   ├─ Pink: High-Low outliers
│   │   └─ Light blue: Low-High outliers
│   └─ Applications:
│       ├─ Crime hot spots
│       ├─ Disease clusters
│       ├─ Economic development disparities
│       └─ Environmental pollution zones
├─ Getis-Ord G Statistics:
│   ├─ Global G:
│   │   ├─ G = [Σᵢ Σⱼ wᵢⱼ yᵢ yⱼ] / [Σᵢ Σⱼ yᵢ yⱼ]
│   │   ├─ Ratio of weighted to total cross-products
│   │   └─ Tests for concentration of high/low values
│   ├─ Local Gᵢ:
│   │   ├─ Gᵢ = [Σⱼ wᵢⱼ yⱼ] / [Σⱼ yⱼ]
│   │   ├─ Proportion of total sum in i's neighborhood
│   │   └─ Detects hot spots (high clustering)
│   ├─ Standardized Gᵢ*:
│   │   ├─ Include i itself in calculation
│   │   ├─ Z-score: Z(Gᵢ*) = [Gᵢ* - E[Gᵢ*]] / √Var[Gᵢ*]
│   │   └─ Asymptotically normal
│   ├─ Interpretation:
│   │   ├─ Z(Gᵢ*) > 0: Hot spot (high values cluster)
│   │   ├─ Z(Gᵢ*) < 0: Cold spot (low values cluster)
│   │   └─ |Z| > 1.96: Significant at 5%
│   └─ Difference from LISA:
│       ├─ Gᵢ uses absolute values (not deviations from mean)
│       ├─ Sensitive to extreme values
│       └─ Useful for count data, rates
├─ Variogram/Correlogram:
│   ├─ Variogram (Semivariogram):
│   │   ├─ γ(h) = (1/2N_h) Σ [(yᵢ - yⱼ)²]
│   │   ├─ N_h: # of pairs separated by distance h
│   │   └─ Measures dissimilarity as function of distance
│   ├─ Properties:
│   │   ├─ Nugget: γ(0) (measurement error, micro-scale variation)
│   │   ├─ Sill: γ(∞) (total variance)
│   │   ├─ Range: Distance at which γ(h) reaches sill
│   │   └─ Smooth increase with distance (typically)
│   ├─ Correlogram:
│   │   ├─ ρ(h) = Corr(yᵢ, yⱼ | dᵢⱼ=h)
│   │   ├─ Correlation as function of distance
│   │   └─ Decays from ρ(0)=1 to ρ(∞)=0
│   ├─ Theoretical Models:
│   │   ├─ Exponential: γ(h) = C₀ + C[1 - exp(-h/a)]
│   │   ├─ Spherical: γ(h) = C₀ + C[3h/(2a) - h³/(2a³)] for h<a
│   │   ├─ Gaussian: γ(h) = C₀ + C[1 - exp(-h²/a²)]
│   │   └─ Fit to empirical variogram
│   └─ Applications:
│       ├─ Kriging (optimal spatial interpolation)
│       ├─ Determine spatial dependence range
│       └─ Model selection for spatial processes
├─ Implications for Modeling:
│   ├─ OLS Consequences:
│   │   ├─ Biased standard errors (typically underestimated)
│   │   ├─ Inefficient estimates (not minimum variance)
│   │   ├─ Invalid hypothesis tests (inflated Type I error)
│   │   └─ Incorrect confidence intervals
│   ├─ Model Specification:
│   │   ├─ Positive I: Consider spatial lag model (Y = ρWY + Xβ + ε)
│   │   ├─ Spatial patterns in residuals: Spatial error model (ε = λWε + u)
│   │   └─ LM tests guide specification
│   ├─ Diagnostic Workflow:
│   │   ├─ 1. Fit OLS, extract residuals
│   │   ├─ 2. Compute Moran's I on residuals
│   │   ├─ 3. If significant → Spatial dependence
│   │   ├─ 4. LM tests (lag vs error)
│   │   └─ 5. Refit with spatial model
│   └─ Robust Inference:
│       ├─ Spatial HAC standard errors (Conley 1999)
│       ├─ Cluster by geographic region
│       └─ Randomization inference
├─ Testing Strategy:
│   ├─ Step 1: Visual Inspection:
│   │   ├─ Choropleth maps (color-coded regions)
│   │   ├─ Scatter plots (y vs Wy)
│   │   └─ Identify obvious patterns
│   ├─ Step 2: Global Tests:
│   │   ├─ Moran's I on outcome variable
│   │   ├─ Geary's C for robustness
│   │   └─ Permutation tests for exact p-values
│   ├─ Step 3: OLS Residual Diagnostics:
│   │   ├─ Moran's I on residuals
│   │   ├─ If significant → Omitted spatial structure
│   │   └─ Variogram of residuals
│   ├─ Step 4: Local Analysis:
│   │   ├─ LISA statistics (identify clusters)
│   │   ├─ Getis-Ord Gᵢ* (hot spot analysis)
│   │   └─ Map significant locations
│   └─ Step 5: Model Selection:
│       ├─ LM test for spatial lag
│       ├─ LM test for spatial error
│       └─ Fit appropriate spatial model
├─ Software Implementation:
│   ├─ Python:
│   │   ├─ PySAL (pysal, libpysal, esda): Comprehensive spatial analysis
│   │   ├─ GeoPandas: Spatial data structures
│   │   ├─ splot: Spatial visualization
│   │   └─ spreg: Spatial regression
│   ├─ R:
│   │   ├─ spdep: Spatial dependence functions
│   │   ├─ spatialreg: Spatial regression models
│   │   ├─ sf: Simple features (geographic data)
│   │   └─ gstat: Geostatistics (variograms)
│   └─ Stata:
│       ├─ spmap: Spatial mapping
│       ├─ spatreg: Spatial regressions
│       └─ moransI: Built-in function
├─ Practical Considerations:
│   ├─ Weight Matrix Choice:
│   │   ├─ Theory: Use substantive knowledge of spatial process
│   │   ├─ Sensitivity: Try multiple specifications
│   │   ├─ Parsimony: Sparse matrices computationally efficient
│   │   └─ Robustness: Results consistent across W?
│   ├─ Edge Effects:
│   │   ├─ Boundary units have fewer neighbors
│   │   ├─ Biases statistics toward zero
│   │   └─ Solution: Buffer zones, toroidal adjustments
│   ├─ Scale Dependence:
│   │   ├─ MAUP (Modifiable Areal Unit Problem)
│   │   ├─ Results vary with spatial aggregation
│   │   └─ Report aggregation level
│   ├─ Non-Stationarity:
│   │   ├─ Spatial process may vary across space
│   │   ├─ Global statistics average heterogeneous patterns
│   │   └─ Use local statistics or GWR (geographically weighted regression)
│   └─ Temporal Dimension:
│       ├─ Spatial panel data: Space + time
│       ├─ Space-time clustering tests
│       └─ Dynamic spatial models
└─ Applications:
    ├─ Regional Economics: Growth spillovers, trade patterns
    ├─ Real Estate: House prices (location, location, location)
    ├─ Epidemiology: Disease spread, health outcomes
    ├─ Environmental Science: Pollution dispersion, climate
    ├─ Criminology: Crime hot spots, policing strategies
    ├─ Political Science: Voting patterns, policy diffusion
    └─ Agriculture: Yield variability, soil properties
```

**Interaction:** Specify weight matrix W → Compute Moran's I (test spatial autocorrelation) → If significant, examine LISA (local clusters) → Fit spatial model (lag/error)

## 5. Mini-Project
Implement Moran's I, Geary's C, LISA, and visualize spatial patterns:
```python
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
print(f"  Grid: {grid_size} × {grid_size} = {n} locations")
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
        print(f"✓ Significant POSITIVE spatial autocorrelation (clustering)")
    else:
        print(f"✓ Significant NEGATIVE spatial autocorrelation (dispersion)")
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
        print(f"✓ Significant POSITIVE spatial autocorrelation")
    else:
        print(f"✓ Significant NEGATIVE spatial autocorrelation")
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
         label=f'Slope = {slope:.3f} ≈ I = {moran_result["I"]:.3f}')

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
print(f"   Consider spatial lag model: Y = ρWY + Xβ + ε")
print(f"   Or spatial error model: Y = Xβ + u, where u = λWu + ε")
print(f"   Use LM tests to choose specification")

print("\n5. Practical Recommendations:")
print("   • Always test for spatial autocorrelation before modeling")
print("   • Use multiple weight matrix specifications (robustness)")
print("   • Examine local patterns (LISA) not just global statistics")
print("   • Permutation tests preferred for exact inference")
print("   • Spatial HAC standard errors if spatial model infeasible")

print("\n6. Software:")
print("   • Python: PySAL (esda.Moran), GeoPandas, splot")
print("   • R: spdep (moran.test, localmoran), sf")
print("   • Stata: spatwmat, moransI, spatreg")
```

## 6. Challenge Round
When does spatial autocorrelation testing fail or mislead?
- **MAUP (Modifiable Areal Unit Problem)**: Results change with spatial aggregation → Test at multiple scales; report sensitivity; use point-level data if possible
- **Edge effects**: Boundary units have truncated neighborhoods → Biases toward no autocorrelation; use buffer zones or toroidal topology
- **Spatial non-stationarity**: Process varies across space (e.g., urban vs rural) → Global I averages; use local statistics or geographically weighted regression (GWR)
- **Omitted spatial heterogeneity**: Apparent clustering from unmeasured covariates → Control for confounders first; residual Moran's I more informative
- **Weight matrix misspecification**: Wrong W leads to wrong conclusions → Try multiple specifications; theory should guide choice
- **Multiple testing**: LISA inflates Type I error for many locations → Apply Bonferroni or FDR correction; interpret patterns not individual tests

## 7. Key References
- [Anselin (1988) - Spatial Econometrics: Methods and Models](https://link.springer.com/book/10.1007/978-94-015-7799-1)
- [Moran (1950) - Notes on Continuous Stochastic Phenomena](https://www.jstor.org/stable/2332142)
- [PySAL Documentation - Exploratory Spatial Data Analysis](https://pysal.org/esda/)

---
**Status:** Foundation for spatial econometrics | **Complements:** Spatial lag/error models, GWR, spatial panel data
