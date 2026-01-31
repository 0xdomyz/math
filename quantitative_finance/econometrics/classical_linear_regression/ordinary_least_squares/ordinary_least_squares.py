import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, qr

# Simulation parameters
np.random.seed(42)
n = 100  # Sample size
k = 3    # Number of regressors (including intercept)
n_simulations = 1000

# True parameters
beta_true = np.array([2.0, 1.5, -0.5])  # [intercept, X1, X2]
sigma = 1.0

# Generate fixed X (non-stochastic regressors)
X = np.column_stack([
    np.ones(n),
    np.random.uniform(0, 5, n),
    np.random.uniform(0, 10, n)
])

# Storage for simulation results
beta_hats = np.zeros((n_simulations, k))

# Monte Carlo simulation
for sim in range(n_simulations):
    # Generate Y with random errors
    epsilon = np.random.normal(0, sigma, n)
    Y = X @ beta_true + epsilon
    
    # OLS estimation
    beta_hat = inv(X.T @ X) @ (X.T @ Y)
    beta_hats[sim, :] = beta_hat

# Analysis of simulation results
print("=" * 80)
print("OLS PROPERTIES: MONTE CARLO SIMULATION")
print("=" * 80)
print(f"True parameters: Î² = {beta_true}")
print(f"Sample size: n = {n}")
print(f"Number of simulations: {n_simulations}")
print()

print("UNBIASEDNESS (E[Î²Ì‚] = Î²):")
print("-" * 80)
mean_beta_hat = beta_hats.mean(axis=0)
for j in range(k):
    bias = mean_beta_hat[j] - beta_true[j]
    print(f"  Î²Ì‚_{j}: True = {beta_true[j]:.4f}, Mean = {mean_beta_hat[j]:.4f}, Bias = {bias:.6f}")
print()

# Theoretical variance
XtX_inv = inv(X.T @ X)
var_beta_theoretical = sigma**2 * XtX_inv
se_beta_theoretical = np.sqrt(np.diag(var_beta_theoretical))

# Empirical variance from simulations
var_beta_empirical = beta_hats.var(axis=0, ddof=1)
se_beta_empirical = np.sqrt(var_beta_empirical)

print("EFFICIENCY (Var(Î²Ì‚) = ÏƒÂ²(X'X)â»Â¹):")
print("-" * 80)
print(f"{'Parameter':<12} {'Theoretical SE':<18} {'Empirical SE':<18} {'Match':<10}")
print("-" * 80)
for j in range(k):
    match = "âœ“" if abs(se_beta_theoretical[j] - se_beta_empirical[j]) < 0.02 else "âœ—"
    print(f"Î²Ì‚_{j:<10} {se_beta_theoretical[j]:<18.6f} {se_beta_empirical[j]:<18.6f} {match:<10}")
print()

# Projection properties for single realization
epsilon = np.random.normal(0, sigma, n)
Y = X @ beta_true + epsilon
beta_hat = inv(X.T @ X) @ (X.T @ Y)

# Hat matrix
H = X @ inv(X.T @ X) @ X.T
Y_hat = H @ Y
residuals = Y - Y_hat

print("PROJECTION PROPERTIES (Single Sample):")
print("-" * 80)
print(f"  Idempotence (HÂ² = H):        ||HÂ² - H|| = {np.linalg.norm(H @ H - H):.10f}")
print(f"  Symmetry (H' = H):           ||H' - H|| = {np.linalg.norm(H.T - H):.10f}")
print(f"  Trace(H) = k:                tr(H) = {np.trace(H):.6f} (expected: {k})")
print(f"  Orthogonality (X'Ãª = 0):     ||X'Ãª|| = {np.linalg.norm(X.T @ residuals):.10f}")
print(f"  Residual sum (Î£Ãªáµ¢ = 0):      Î£Ãªáµ¢ = {residuals.sum():.10f}")
print()

# SST = SSE + SSR decomposition
Y_bar = Y.mean()
SST = np.sum((Y - Y_bar)**2)
SSE = np.sum((Y_hat - Y_bar)**2)
SSR = np.sum(residuals**2)
R_squared = 1 - SSR / SST

print("SUM OF SQUARES DECOMPOSITION:")
print("-" * 80)
print(f"  Total (SST):                 {SST:.6f}")
print(f"  Explained (SSE):             {SSE:.6f}")
print(f"  Residual (SSR):              {SSR:.6f}")
print(f"  SSE + SSR:                   {SSE + SSR:.6f}")
print(f"  Difference (SST - SSE - SSR):{SST - SSE - SSR:.10f}")
print(f"  RÂ²:                          {R_squared:.6f}")
print("=" * 80)

# Computational methods comparison
print("\nCOMPUTATIONAL METHODS COMPARISON:")
print("-" * 80)

# Method 1: Direct inversion
beta_hat_direct = inv(X.T @ X) @ (X.T @ Y)

# Method 2: QR decomposition
Q, R = qr(X, mode='economic')
beta_hat_qr = inv(R) @ (Q.T @ Y)

# Method 3: SVD (pseudoinverse)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
beta_hat_svd = Vt.T @ np.diag(1/S) @ U.T @ Y

print(f"{'Method':<25} {'Î²Ì‚â‚€':<15} {'Î²Ì‚â‚':<15} {'Î²Ì‚â‚‚':<15}")
print("-" * 80)
print(f"{'Direct (X\'X)â»Â¹X\'Y':<25} {beta_hat_direct[0]:<15.8f} {beta_hat_direct[1]:<15.8f} {beta_hat_direct[2]:<15.8f}")
print(f"{'QR decomposition':<25} {beta_hat_qr[0]:<15.8f} {beta_hat_qr[1]:<15.8f} {beta_hat_qr[2]:<15.8f}")
print(f"{'SVD (pseudoinverse)':<25} {beta_hat_svd[0]:<15.8f} {beta_hat_svd[1]:<15.8f} {beta_hat_svd[2]:<15.8f}")
print(f"{'Max difference':<25} {np.abs(beta_hat_direct - beta_hat_qr).max():.2e}")
print("=" * 80)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Sampling distributions of Î²Ì‚
for j in range(k):
    axes[j].hist(beta_hats[:, j], bins=40, density=True, alpha=0.7, edgecolor='black', label='Empirical')
    
    # Theoretical normal distribution
    x_range = np.linspace(beta_hats[:, j].min(), beta_hats[:, j].max(), 100)
    theoretical_density = (1/(se_beta_theoretical[j] * np.sqrt(2*np.pi))) * \
                          np.exp(-0.5*((x_range - beta_true[j])/se_beta_theoretical[j])**2)
    axes[j].plot(x_range, theoretical_density, 'r-', linewidth=2, label='Theoretical N(Î², ÏƒÂ²(X\'X)â»Â¹)')
    axes[j].axvline(beta_true[j], color='green', linestyle='--', linewidth=2, label=f'True Î²_{j}')
    axes[j].axvline(mean_beta_hat[j], color='orange', linestyle='--', linewidth=2, label=f'Mean Î²Ì‚_{j}')
    
    axes[j].set_xlabel(f'Î²Ì‚_{j}', fontsize=11, fontweight='bold')
    axes[j].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[j].set_title(f'Sampling Distribution of Î²Ì‚_{j}', fontsize=12, fontweight='bold')
    axes[j].legend(fontsize=8)
    axes[j].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ols_properties.png', dpi=150)
plt.show()
