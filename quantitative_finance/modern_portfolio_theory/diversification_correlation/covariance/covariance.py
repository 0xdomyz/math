import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, OAS, MinCovDet
from scipy.linalg import eigvalsh
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic returns (multiple assets with structure)
np.random.seed(42)
periods = 252
n_assets = 5

# Common factor
market_factor = np.random.normal(0, 0.01, periods)

# Asset returns with varying factor loadings and idiosyncratic risk
returns = np.zeros((periods, n_assets))
factor_loadings = np.array([0.8, 0.6, 0.4, 0.2, 0.1])  # β values

for i in range(n_assets):
    returns[:, i] = factor_loadings[i] * market_factor + np.random.normal(0, 0.008, periods)

returns_df = pd.DataFrame(returns, columns=[f'Asset_{i+1}' for i in range(n_assets)])
dates = pd.date_range('2024-01-01', periods=periods, freq='D')
returns_df.index = dates

print("="*70)
print("COVARIANCE ANALYSIS: Multi-Asset Portfolio")
print("="*70)

# 1. Sample covariance matrix
print("\n1. SAMPLE COVARIANCE MATRIX")
print("-"*70)

# Annualize (multiply by 252 trading days)
cov_sample = returns_df.cov() * 252
print("\nAnnualized Covariance Matrix:")
print(cov_sample.round(6))

print("\n\nDiagonal (Variances):")
for asset in returns_df.columns:
    print(f"  {asset}: {cov_sample.loc[asset, asset]:.6f} (σ={np.sqrt(cov_sample.loc[asset, asset]):.4f})")

# 2. Correlation from covariance
print("\n2. CORRELATION FROM COVARIANCE")
print("-"*70)

corr_from_cov = cov_sample.copy()
for i in range(len(corr_from_cov)):
    for j in range(len(corr_from_cov)):
        corr_from_cov.iloc[i, j] = cov_sample.iloc[i, j] / (np.sqrt(cov_sample.iloc[i, i]) * np.sqrt(cov_sample.iloc[j, j]))

print("\nCorrelation Matrix (derived from covariance):")
print(corr_from_cov.round(4))

# Verify: Should match direct correlation
corr_direct = returns_df.corr()
print("\n\nVerification (direct correlation computation):")
print(f"Max difference: {np.abs(corr_from_cov - corr_direct).max().max():.10f}")

# 3. Covariance matrix properties
print("\n3. COVARIANCE MATRIX PROPERTIES")
print("-"*70)

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_sample.values)
eigenvalues = eigenvalues[::-1]  # Sort descending
print(f"\nEigenvalues (condition number = λmax/λmin):")
for i, eig in enumerate(eigenvalues):
    print(f"  λ{i+1}: {eig:.6f}")

condition_number = eigenvalues[0] / eigenvalues[-1]
print(f"\nCondition number: {condition_number:.2f}")
print(f"Numerical stability: {'Good' if condition_number < 100 else 'Questionable' if condition_number < 1e6 else 'Poor'}")

# Determinant (multivariate spread)
det_cov = np.linalg.det(cov_sample.values)
print(f"\nDeterminant: {det_cov:.10f}")
print(f"(Multivariate volume; 0 indicates singular matrix)")

# Trace (sum of variances)
trace = np.trace(cov_sample.values)
print(f"\nTrace (sum of variances): {trace:.6f}")

# 4. Estimator comparison
print("\n4. COVARIANCE ESTIMATOR COMPARISON")
print("-"*70)

# Sample covariance
cov_sample_est = returns_df.cov() * 252

# Ledoit-Wolf shrinkage
lw = LedoitWolf()
cov_lw, shrinkage_lw = lw.fit(returns_df.values).covariance_, lw.shrinkage
cov_lw_scaled = cov_lw * 252

# OAS (Oracle Approximating Shrinkage)
oas = OAS()
cov_oas, shrinkage_oas = oas.fit(returns_df.values).covariance_, oas.shrinkage
cov_oas_scaled = cov_oas * 252

print(f"\nSample Covariance Determinant: {np.linalg.det(cov_sample_est.values):.10f}")
print(f"Ledoit-Wolf Determinant: {np.linalg.det(cov_lw_scaled):.10f}")
print(f"  Shrinkage intensity λ: {shrinkage_lw:.4f}")

print(f"\nOAS Determinant: {np.linalg.det(cov_oas_scaled):.10f}")
print(f"  Shrinkage intensity λ: {shrinkage_oas:.4f}")

# Eigenvalue stability
_, eigs_sample = np.linalg.eigh(cov_sample_est.values)
eigs_sample = np.sort(eigs_sample)[::-1]

_, eigs_lw = np.linalg.eigh(cov_lw_scaled)
eigs_lw = np.sort(eigs_lw)[::-1]

print(f"\nSmallest Eigenvalue:")
print(f"  Sample: {eigs_sample[-1]:.8f}")
print(f"  Ledoit-Wolf: {eigs_lw[-1]:.8f}")
print(f"  Improvement: {(eigs_sample[-1] - eigs_lw[-1])/eigs_sample[-1]*100:.1f}% larger")

# 5. Covariance properties for portfolio
print("\n5. PORTFOLIO VARIANCE DECOMPOSITION")
print("-"*70)

# Equal-weight portfolio
w_eq = np.array([1/n_assets] * n_assets)
var_eq = w_eq @ cov_sample.values @ w_eq

# Component contribution to variance
contrib = np.zeros(n_assets)
for i in range(n_assets):
    contrib[i] = 2 * w_eq[i] * (cov_sample.values @ w_eq)[i]

print(f"\nEqual-Weight Portfolio (w = 1/{n_assets}):")
print(f"Portfolio variance: {var_eq:.6f}")
print(f"Portfolio volatility: {np.sqrt(var_eq):.4f}")

print(f"\nVariance contribution by asset:")
for i, asset in enumerate(returns_df.columns):
    print(f"  {asset}: {contrib[i]:.6f} ({contrib[i]/var_eq*100:.1f}%)")

# Minimum variance portfolio
inv_cov = np.linalg.inv(cov_sample.values)
ones = np.ones(n_assets)
w_min = inv_cov @ ones / (ones @ inv_cov @ ones)
var_min = w_min @ cov_sample.values @ w_min

print(f"\n\nMinimum Variance Portfolio:")
print(f"Weights:")
for i, asset in enumerate(returns_df.columns):
    print(f"  {asset}: {w_min[i]:7.4f}")

print(f"\nPortfolio variance: {var_min:.6f}")
print(f"Portfolio volatility: {np.sqrt(var_min):.4f}")
print(f"Risk reduction vs equal-weight: {(var_eq - var_min)/var_eq*100:.1f}%")

# 6. Covariance visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Covariance matrix heatmap
ax = axes[0, 0]
im = ax.imshow(cov_sample.values, cmap='RdBu_r', aspect='auto')
ax.set_xticks(range(n_assets))
ax.set_yticks(range(n_assets))
ax.set_xticklabels(returns_df.columns, rotation=45, ha='right')
ax.set_yticklabels(returns_df.columns)
ax.set_title('Covariance Matrix Heatmap')
plt.colorbar(im, ax=ax, label='Covariance')
for i in range(n_assets):
    for j in range(n_assets):
        ax.text(j, i, f'{cov_sample.iloc[i, j]:.2e}', ha='center', va='center', fontsize=8)

# Plot 2: Correlation (normalized) heatmap
ax = axes[0, 1]
im = ax.imshow(corr_from_cov.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(n_assets))
ax.set_yticks(range(n_assets))
ax.set_xticklabels(returns_df.columns, rotation=45, ha='right')
ax.set_yticklabels(returns_df.columns)
ax.set_title('Correlation Matrix (from Covariance)')
plt.colorbar(im, ax=ax, label='Correlation')
for i in range(n_assets):
    for j in range(n_assets):
        ax.text(j, i, f'{corr_from_cov.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)

# Plot 3: Eigenvalue spectrum
ax = axes[0, 2]
ax.semilogy(range(1, len(eigenvalues)+1), eigenvalues, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Eigenvalue Index')
ax.set_ylabel('Eigenvalue (log scale)')
ax.set_title('Eigenvalue Spectrum')
ax.grid(alpha=0.3)

# Plot 4: Scatter with covariance
ax = axes[1, 0]
ax.scatter(returns_df['Asset_1']*100, returns_df['Asset_2']*100, alpha=0.5, s=30)
cov_12 = cov_sample.loc['Asset_1', 'Asset_2']
corr_12 = corr_from_cov.loc['Asset_1', 'Asset_2']
ax.set_xlabel('Asset 1 Return (%)')
ax.set_ylabel('Asset 2 Return (%)')
ax.set_title(f'Asset 1 vs 2: Cov={cov_12:.5f}, ρ={corr_12:.3f}')
ax.grid(alpha=0.3)

# Plot 5: Covariance estimator comparison
ax = axes[1, 1]
estimators = ['Sample', 'Ledoit-Wolf', 'OAS']
dets = [np.linalg.det(cov_sample_est.values), 
        np.linalg.det(cov_lw_scaled),
        np.linalg.det(cov_oas_scaled)]
colors = ['b', 'r', 'g']
bars = ax.bar(estimators, dets, color=colors, alpha=0.7)
ax.set_ylabel('Determinant')
ax.set_title('Covariance Matrix Determinant by Estimator')
ax.grid(alpha=0.3, axis='y')
for bar, det in zip(bars, dets):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{det:.2e}', ha='center', va='bottom', fontsize=9)

# Plot 6: Portfolio variance decomposition
ax = axes[1, 2]
colors_pie = plt.cm.Set3(np.linspace(0, 1, n_assets))
wedges, texts, autotexts = ax.pie(contrib, labels=returns_df.columns, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90)
ax.set_title('Variance Contribution (Equal-Weight Portfolio)')

plt.tight_layout()
plt.savefig('covariance_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print(f"""
Key Findings:

1. Covariance Structure:
   → Factor model generates structure (common market exposure)
   → Assets with high β → larger covariances
   → Non-zero off-diagonal: Assets move together

2. Correlation vs Covariance:
   → Covariance magnitude depends on volatilities
   → High-volatility assets → large covariances even if ρ small
   → Correlation standardizes → easier comparison

3. Eigenvalue Analysis:
   → Condition number {condition_number:.0f}: Numerical stability status
   → One dominant eigenvalue: Single factor dominates variation
   → Small eigenvalues: Portfolio optimization sensitive to noise

4. Estimator Comparison:
   → Sample covariance: Unbiased but high variance (small T)
   → Ledoit-Wolf shrinkage λ={shrinkage_lw:.3f}: Balances bias-variance
   → Out-of-sample: Shrinkage estimates typically better

5. Portfolio Implications:
   → Minimum variance portfolio: Exploits covariance structure
   → Risk reduction: {(var_eq - var_min)/var_eq*100:.1f}% better than equal-weight
   → Estimation error: Impacts weight calculations → use robust methods
""")