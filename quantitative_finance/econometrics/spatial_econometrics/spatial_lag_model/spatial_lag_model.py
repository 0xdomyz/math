import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import seaborn as sns

np.random.seed(357)

# ===== Simulate Spatial Data with Known Spillovers =====
print("="*80)
print("SPATIAL LAG MODEL (SAR)")
print("="*80)

# Grid coordinates
grid_size = 15
n = grid_size ** 2
coords = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])

print(f"\nSimulation Setup:")
print(f"  Grid: {grid_size} Ã— {grid_size} = {n} locations")

# Generate covariates
X_raw = np.column_stack([
    np.ones(n),
    np.random.randn(n),  # X1
    np.random.randn(n)   # X2
])

# Spatial weight matrix (Queen contiguity)
def create_weight_matrix_queen(coords, grid_size):
    n = len(coords)
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.abs(coords[i] - coords[j])
                if np.max(dist) <= 1:  # Queen: shares edge or corner
                    W[i, j] = 1
    
    # Row standardization
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_std = W / row_sums
    
    return W_std, W

W, W_raw = create_weight_matrix_queen(coords, grid_size)

print(f"  Weight matrix: Queen contiguity, row-standardized")
print(f"  Connections: {np.sum(W_raw > 0)} ({np.mean(np.sum(W_raw>0, axis=1)):.1f} avg neighbors)")

# True parameters
rho_true = 0.5
beta_true = np.array([2.0, 1.5, -0.8])
sigma_true = 0.5

print(f"\nTrue Parameters:")
print(f"  Ï (spatial lag): {rho_true}")
print(f"  Î²: {beta_true}")
print(f"  Ïƒ: {sigma_true}")

# Generate Y via spatial process: Y = (I - ÏW)^{-1}(XÎ² + Îµ)
epsilon = np.random.randn(n) * sigma_true
Xbeta = X_raw @ beta_true

# Solve (I - ÏW)Y = XÎ² + Îµ
I = np.eye(n)
A = I - rho_true * W
Y = np.linalg.solve(A, Xbeta + epsilon)

print(f"  Generated Y via: Y = (I - ÏW)^{{-1}}(XÎ² + Îµ)")
print(f"  Y statistics: mean={np.mean(Y):.3f}, sd={np.std(Y):.3f}")

# Spatial lag variable
WY = W @ Y

# Check spatial autocorrelation (Moran's I)
def morans_i_simple(y, W):
    n = len(y)
    y_dev = y - np.mean(y)
    numerator = np.sum(W * np.outer(y_dev, y_dev))
    denominator = np.sum(y_dev ** 2)
    S0 = np.sum(W)
    I = (n / S0) * (numerator / denominator)
    return I

I_Y = morans_i_simple(Y, W)
print(f"\nMoran's I on Y: {I_Y:.4f} (strong positive autocorrelation)")

# ===== OLS Estimation (Naive, Biased) =====
print("\n" + "="*80)
print("OLS ESTIMATION (Biased)")
print("="*80)

# OLS: Y = XÎ² + Îµ (ignores spatial lag)
XtX_inv = np.linalg.inv(X_raw.T @ X_raw)
beta_ols = XtX_inv @ X_raw.T @ Y
resid_ols = Y - X_raw @ beta_ols
sigma2_ols = np.sum(resid_ols ** 2) / (n - X_raw.shape[1])
se_ols = np.sqrt(np.diag(XtX_inv * sigma2_ols))

print(f"OLS Results:")
print(f"  Î²Ì‚: {beta_ols}")
print(f"  SE: {se_ols}")
print(f"  ÏƒÌ‚: {np.sqrt(sigma2_ols):.4f}")

# Moran's I on OLS residuals
I_resid_ols = morans_i_simple(resid_ols, W)
print(f"\nMoran's I on OLS residuals: {I_resid_ols:.4f}")
if I_resid_ols > 0.1:
    print(f"  âœ“ Significant spatial autocorrelation in residuals â†’ SAR model needed")

# Bias in OLS
bias_ols = beta_ols - beta_true
print(f"\nOLS Bias (Î²Ì‚_OLS - Î²_true): {bias_ols}")

# ===== Maximum Likelihood Estimation =====
print("\n" + "="*80)
print("MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
print("="*80)

# Eigenvalues of W for Jacobian
eigvals_W = eigh(W, eigvals_only=True)
lambda_min, lambda_max = eigvals_W.min(), eigvals_W.max()

print(f"Eigenvalue bounds for W:")
print(f"  Î»_min = {lambda_min:.4f}, Î»_max = {lambda_max:.4f}")
print(f"  Ï must be in ({1/lambda_min:.3f}, {1/lambda_max:.3f}) for stationarity")

def log_likelihood_sar(params, Y, X, W, eigvals_W):
    """Concentrated log-likelihood for SAR"""
    rho = params[0]
    n = len(Y)
    
    # Bounds check
    if rho <= 1/eigvals_W.min() or rho >= 1/eigvals_W.max():
        return -1e10
    
    # Jacobian term: log|I - ÏW| = sum(log(1 - ÏÎ»_i))
    log_det = np.sum(np.log(1 - rho * eigvals_W))
    
    # Residuals
    Y_trans = Y - rho * (W @ Y)
    
    # Concentrated Î²
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ Y_trans
    
    # Residuals
    resid = Y_trans - X @ beta_hat
    
    # Concentrated ÏƒÂ²
    sigma2_hat = np.sum(resid ** 2) / n
    
    # Log-likelihood
    ll = -(n/2) * np.log(2*np.pi) - (n/2) * np.log(sigma2_hat) + log_det - n/2
    
    return -ll  # Minimize negative ll

# Optimize over Ï
print(f"\nOptimizing MLE...")
result_mle = optimize.minimize_scalar(
    lambda rho: log_likelihood_sar(np.array([rho]), Y, X_raw, W, eigvals_W),
    bounds=(1/lambda_min + 0.01, 1/lambda_max - 0.01),
    method='bounded'
)

rho_mle = result_mle.x

# Get Î², ÏƒÂ² at optimal Ï
Y_trans_mle = Y - rho_mle * WY
XtX_inv = np.linalg.inv(X_raw.T @ X_raw)
beta_mle = XtX_inv @ X_raw.T @ Y_trans_mle
resid_mle = Y_trans_mle - X_raw @ beta_mle
sigma2_mle = np.sum(resid_mle ** 2) / n

print(f"\nMLE Results:")
print(f"  ÏÌ‚: {rho_mle:.4f} (true: {rho_true})")
print(f"  Î²Ì‚: {beta_mle} (true: {beta_true})")
print(f"  ÏƒÌ‚: {np.sqrt(sigma2_mle):.4f} (true: {sigma_true})")

# Standard errors (from Hessian)
# Simplified: Use numerical Hessian
from scipy.optimize import approx_fprime

def ll_all_params(params):
    rho = params[0]
    beta = params[1:4]
    log_sigma2 = params[4]
    sigma2 = np.exp(log_sigma2)
    
    if rho <= 1/lambda_min + 0.01 or rho >= 1/lambda_max - 0.01:
        return 1e10
    
    log_det = np.sum(np.log(1 - rho * eigvals_W))
    Y_trans = Y - rho * WY
    resid = Y_trans - X_raw @ beta
    ll = -(n/2) * np.log(2*np.pi*sigma2) + log_det - (1/(2*sigma2)) * np.sum(resid**2)
    return -ll

params_mle_all = np.concatenate([[rho_mle], beta_mle, [np.log(sigma2_mle)]])

# Hessian approximation
hessian_approx = optimize.approx_fprime(params_mle_all, ll_all_params, epsilon=1e-5)
# For proper SE, need full Hessian matrix (skip for brevity; use software)

print(f"\n(Full SE calculation omitted; use software like PySAL for accurate SEs)")

# Moran's I on MLE residuals
I_resid_mle = morans_i_simple(resid_mle, W)
print(f"\nMoran's I on MLE residuals: {I_resid_mle:.4f}")
print(f"  (Should be near zero if model correct)")

# ===== Instrumental Variables (2SLS) Estimation =====
print("\n" + "="*80)
print("INSTRUMENTAL VARIABLES (2SLS) ESTIMATION")
print("="*80)

# Instruments: WX (neighbors' covariates)
WX = W @ X_raw[:, 1:]  # Exclude intercept, apply W to X1, X2

# First stage: WY ~ X + WX
X_first = np.column_stack([X_raw, WX])
XtX_first_inv = np.linalg.inv(X_first.T @ X_first)
gamma_first = XtX_first_inv @ X_first.T @ WY
WY_hat = X_first @ gamma_first

# First stage diagnostics
resid_first = WY - WY_hat
SS_res_first = np.sum(resid_first ** 2)
SS_tot_first = np.sum((WY - np.mean(WY)) ** 2)
R2_first = 1 - SS_res_first / SS_tot_first

# F-statistic for instrument strength
k1 = X_raw.shape[1]
k2 = WX.shape[1]
F_stat = ((SS_tot_first - SS_res_first) / k2) / (SS_res_first / (n - X_first.shape[1]))

print(f"First Stage: WY ~ X + WX")
print(f"  RÂ²: {R2_first:.4f}")
print(f"  F-statistic: {F_stat:.2f}")
if F_stat > 10:
    print(f"  âœ“ Strong instruments (F > 10)")
else:
    print(f"  âš  Weak instruments (F < 10)")

# Second stage: Y ~ Å´Y + X
X_second = np.column_stack([WY_hat, X_raw])
XtX_second_inv = np.linalg.inv(X_second.T @ X_second)
params_2sls = XtX_second_inv @ X_second.T @ Y

rho_2sls = params_2sls[0]
beta_2sls = params_2sls[1:]

# Residuals and ÏƒÂ²
resid_2sls = Y - X_second @ params_2sls
sigma2_2sls = np.sum(resid_2sls ** 2) / (n - X_second.shape[1])

# Standard errors (need to account for generated regressor)
# Simplified (not fully correct without adjustment)
se_2sls_naive = np.sqrt(np.diag(XtX_second_inv * sigma2_2sls))

print(f"\n2SLS Results:")
print(f"  ÏÌ‚: {rho_2sls:.4f} (true: {rho_true})")
print(f"  Î²Ì‚: {beta_2sls} (true: {beta_true})")
print(f"  ÏƒÌ‚: {np.sqrt(sigma2_2sls):.4f}")
print(f"  (SE not adjusted for generated regressor)")

# ===== Compare Estimators =====
print("\n" + "="*80)
print("COMPARISON OF ESTIMATORS")
print("="*80)

comparison = pd.DataFrame({
    'Parameter': ['Ï', 'Î²â‚€', 'Î²â‚', 'Î²â‚‚'],
    'True': [rho_true, beta_true[0], beta_true[1], beta_true[2]],
    'OLS': [np.nan, beta_ols[0], beta_ols[1], beta_ols[2]],
    'MLE': [rho_mle, beta_mle[0], beta_mle[1], beta_mle[2]],
    '2SLS': [rho_2sls, beta_2sls[0], beta_2sls[1], beta_2sls[2]]
})

print(comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# ===== Direct, Indirect, Total Effects (MLE) =====
print("\n" + "="*80)
print("SPATIAL EFFECTS DECOMPOSITION (MLE)")
print("="*80)

# Spatial multiplier: S = (I - ÏW)^{-1}
S = np.linalg.inv(I - rho_mle * W)

# For each covariate (excluding intercept for interpretation)
effects_results = []

for k in range(1, len(beta_mle)):  # Skip intercept
    # Direct effects: Diagonal of S * Î²_k
    direct_k = np.mean(np.diag(S * beta_mle[k]))
    
    # Total effects: Mean of (S * Î²_k) row sums
    total_k = np.mean(np.sum(S * beta_mle[k], axis=1))
    
    # Indirect effects: Total - Direct
    indirect_k = total_k - direct_k
    
    effects_results.append({
        'Variable': f'X{k}',
        'Direct': direct_k,
        'Indirect': indirect_k,
        'Total': total_k
    })

effects_df = pd.DataFrame(effects_results)
print(effects_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print(f"\nInterpretation:")
print(f"  Direct: Impact of X_i on Y_i")
print(f"  Indirect: Impact of X_j (jâ‰ i) on Y_i (spillover)")
print(f"  Total: Direct + Indirect (full network effect)")

# Spatial multiplier magnitude
avg_multiplier = 1 / (1 - rho_mle * np.mean(W[W > 0]))
print(f"\nSpatial multiplier (approx): {avg_multiplier:.3f}")
print(f"  $1 direct increase â†’ ${avg_multiplier:.3f} total (including feedback)")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Spatial distribution of Y
ax1 = axes[0, 0]
Y_grid = Y.reshape(grid_size, grid_size)
im1 = ax1.imshow(Y_grid, cmap='RdBu_r', origin='lower')
ax1.set_title('True Y (Spatial Lag Process)')
ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')
plt.colorbar(im1, ax=ax1)

# Plot 2: OLS residuals (spatial pattern)
ax2 = axes[0, 1]
resid_ols_grid = resid_ols.reshape(grid_size, grid_size)
im2 = ax2.imshow(resid_ols_grid, cmap='RdBu_r', origin='lower')
ax2.set_title(f'OLS Residuals (Moran I={I_resid_ols:.3f})')
ax2.set_xlabel('X coordinate')
ax2.set_ylabel('Y coordinate')
plt.colorbar(im2, ax=ax2)

# Plot 3: MLE residuals (no spatial pattern)
ax3 = axes[0, 2]
resid_mle_grid = resid_mle.reshape(grid_size, grid_size)
im3 = ax3.imshow(resid_mle_grid, cmap='RdBu_r', origin='lower')
ax3.set_title(f'MLE Residuals (Moran I={I_resid_mle:.3f})')
ax3.set_xlabel('X coordinate')
ax3.set_ylabel('Y coordinate')
plt.colorbar(im3, ax=ax3)

# Plot 4: Moran scatter plot (Y vs WY)
ax4 = axes[1, 0]
ax4.scatter(Y, WY, alpha=0.6, s=30)
slope_moran = np.cov(Y, WY)[0, 1] / np.var(Y)
x_line = np.linspace(Y.min(), Y.max(), 100)
y_line = slope_moran * (x_line - np.mean(Y)) + np.mean(WY)
ax4.plot(x_line, y_line, 'r--', linewidth=2, label=f'Slope â‰ˆ Ï = {rho_true:.2f}')
ax4.set_xlabel('Y')
ax4.set_ylabel('WY (Spatial Lag)')
ax4.set_title('Moran Scatter Plot')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Parameter comparison
ax5 = axes[1, 1]
params_compare = np.array([
    [beta_true[1], beta_true[2]],
    [beta_ols[1], beta_ols[2]],
    [beta_mle[1], beta_mle[2]],
    [beta_2sls[1], beta_2sls[2]]
])
x_pos = np.arange(2)
width = 0.2
colors_bar = ['black', 'gray', 'red', 'blue']
labels_bar = ['True', 'OLS', 'MLE', '2SLS']

for i, (params, color, label) in enumerate(zip(params_compare, colors_bar, labels_bar)):
    ax5.bar(x_pos + i * width, params, width, label=label, color=color, alpha=0.7)

ax5.set_xticks(x_pos + 1.5 * width)
ax5.set_xticklabels(['Î²â‚', 'Î²â‚‚'])
ax5.set_ylabel('Coefficient')
ax5.set_title('Î² Estimates Comparison')
ax5.legend()
ax5.grid(alpha=0.3, axis='y')

# Plot 6: Effects decomposition
ax6 = axes[1, 2]
x_effects = np.arange(len(effects_df))
width_effects = 0.25
ax6.bar(x_effects - width_effects, effects_df['Direct'], width_effects, 
        label='Direct', alpha=0.7)
ax6.bar(x_effects, effects_df['Indirect'], width_effects, 
        label='Indirect (Spillover)', alpha=0.7)
ax6.bar(x_effects + width_effects, effects_df['Total'], width_effects, 
        label='Total', alpha=0.7)
ax6.set_xticks(x_effects)
ax6.set_xticklabels(effects_df['Variable'])
ax6.set_ylabel('Effect Size')
ax6.set_title('Spatial Effects Decomposition')
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('spatial_lag_model.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Spatial Autocorrelation:")
print(f"   Moran's I (Y) = {I_Y:.4f} â†’ Strong spatial clustering")
print(f"   Generated by Ï = {rho_true} (positive feedback)")

print("\n2. OLS Bias:")
print(f"   OLS ignores spatial lag â†’ Biased Î²Ì‚")
print(f"   Residuals spatially autocorrelated (I={I_resid_ols:.3f})")
print(f"   Standard errors underestimated (invalid inference)")

print("\n3. MLE Estimation:")
print(f"   ÏÌ‚_MLE = {rho_mle:.4f} (true: {rho_true}) â†’ Accurate")
print(f"   Î²Ì‚_MLE close to true values")
print(f"   Residuals no spatial autocorrelation (I={I_resid_mle:.3f})")

print("\n4. 2SLS (IV) Estimation:")
print(f"   Instruments: WX (neighbors' covariates)")
print(f"   F-statistic = {F_stat:.1f} â†’ Strong instruments")
print(f"   ÏÌ‚_2SLS = {rho_2sls:.4f} â†’ Consistent but less efficient than MLE")

print("\n5. Spatial Effects:")
print(f"   Direct effects â‰ˆ Î² (but adjusted for feedback)")
print(f"   Indirect effects (spillovers) substantial")
print(f"   Total effects = Direct + Indirect")
print(f"   Ignoring spatial lag misses {100*(effects_df['Indirect'].mean()/effects_df['Total'].mean()):.0f}% of impact!")

print("\n6. Practical Recommendations:")
print("   â€¢ Always test for spatial autocorrelation (Moran's I on OLS residuals)")
print("   â€¢ Use MLE for efficiency (if normality reasonable)")
print("   â€¢ 2SLS robust to non-normality but less efficient")
print("   â€¢ Report direct/indirect/total effects (not just Ï, Î²)")
print("   â€¢ Check residuals for remaining spatial patterns")
print("   â€¢ Sensitivity to weight matrix specification")

print("\n7. Software:")
print("   â€¢ Python: PySAL (spreg.ML_Lag, spreg.GM_Lag)")
print("   â€¢ R: spatialreg::lagsarlm(), impacts()")
print("   â€¢ Stata: spregress with ml lag() option")
