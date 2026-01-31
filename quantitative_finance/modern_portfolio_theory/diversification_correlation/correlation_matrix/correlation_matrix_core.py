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
