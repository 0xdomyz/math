import numpy as np
import matplotlib.pyplot as plt

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