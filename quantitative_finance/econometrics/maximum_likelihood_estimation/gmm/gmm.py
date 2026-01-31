import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from numpy.linalg import inv, pinv
import seaborn as sns

np.random.seed(987)

# ===== Simulate IV Data with Endogeneity =====
print("="*80)
print("GENERALIZED METHOD OF MOMENTS (GMM)")
print("="*80)

n = 500  # Sample size

# Instruments (2 instruments for 1 endogenous variable â†’ overidentified)
Z1 = np.random.randn(n)
Z2 = np.random.randn(n)
Z = np.column_stack([np.ones(n), Z1, Z2])

# Structural error
u = np.random.randn(n)

# Endogenous variable (correlated with u)
rho = 0.7  # Correlation between X and u
pi_1 = 0.8  # First-stage coefficient (instrument strength)
pi_2 = 0.6
v = rho * u + np.sqrt(1 - rho**2) * np.random.randn(n)
X_endo = 1.0 + pi_1 * Z1 + pi_2 * Z2 + v

# Exogenous variable
X_exog = np.random.randn(n)

X = np.column_stack([np.ones(n), X_endo, X_exog])

# True parameters
beta_true = np.array([2.0, 1.5, -0.5])

# Outcome
Y = X @ beta_true + u

print(f"Simulation Setup:")
print(f"  Sample size: {n}")
print(f"  Endogenous variables: 1 (Xâ‚)")
print(f"  Exogenous variables: 1 (Xâ‚‚) + intercept")
print(f"  Instruments: {Z.shape[1]} (including intercept)")
print(f"  Endogeneity: Ï(Xâ‚,u) = {rho}")
print(f"  True Î²: {beta_true}")

# First-stage F-test
from scipy.linalg import lstsq
first_stage = lstsq(Z, X_endo)[0]
X_endo_hat = Z @ first_stage
residuals_fs = X_endo - X_endo_hat
RSS_fs = np.sum(residuals_fs**2)
TSS_fs = np.sum((X_endo - X_endo.mean())**2)
R2_fs = 1 - RSS_fs / TSS_fs
F_stat = (R2_fs / (Z.shape[1] - 1)) / ((1 - R2_fs) / (n - Z.shape[1]))

print(f"\nFirst-Stage Diagnostics:")
print(f"  RÂ²: {R2_fs:.4f}")
print(f"  F-statistic: {F_stat:.2f}")

if F_stat > 10:
    print(f"  âœ“ Strong instruments (F > 10)")
elif F_stat > 5:
    print(f"  âš  Moderately weak instruments (5 < F < 10)")
else:
    print(f"  âœ— Weak instruments (F < 5)")

# ===== OLS (Biased due to endogeneity) =====
print("\n" + "="*80)
print("OLS (INCONSISTENT)")
print("="*80)

beta_ols = lstsq(X, Y)[0]
residuals_ols = Y - X @ beta_ols
sigma2_ols = np.sum(residuals_ols**2) / (n - X.shape[1])
se_ols = np.sqrt(np.diag(sigma2_ols * inv(X.T @ X)))

print(f"OLS Estimates:")
for i, (b_ols, b_true, se) in enumerate(zip(beta_ols, beta_true, se_ols)):
    print(f"  Î²{i}: {b_ols:7.4f} (SE={se:.4f}, true={b_true:.2f})")

bias_ols = beta_ols[1] - beta_true[1]
print(f"\nBias in endogenous coefficient: {bias_ols:+.4f}")
print(f"  â†’ OLS inconsistent due to endogeneity")

# ===== 2SLS (Exactly Identified) =====
print("\n" + "="*80)
print("2SLS (EXACTLY IDENTIFIED)")
print("="*80)

# Use only Z1 (plus intercept) â†’ exactly identified
Z_exact = Z[:, [0, 1]]

# First stage
Pi_hat = lstsq(Z_exact, X)[0]
X_hat = Z_exact @ Pi_hat

# Second stage
beta_2sls_exact = lstsq(X_hat, Y)[0]

# Standard errors
residuals_2sls = Y - X @ beta_2sls_exact
sigma2_2sls = np.sum(residuals_2sls**2) / (n - X.shape[1])
V_2sls = sigma2_2sls * inv(X_hat.T @ X_hat)
se_2sls_exact = np.sqrt(np.diag(V_2sls))

print(f"2SLS Estimates (Z1 only):")
for i, (b_2sls, b_true, se) in enumerate(zip(beta_2sls_exact, beta_true, se_2sls_exact)):
    print(f"  Î²{i}: {b_2sls:7.4f} (SE={se:.4f}, true={b_true:.2f})")

# ===== GMM: Two-Step Estimation =====
print("\n" + "="*80)
print("GMM: TWO-STEP ESTIMATION (OVERIDENTIFIED)")
print("="*80)

# Moment conditions: E[Z'(Y - XÎ²)] = 0
def moments(beta, Y, X, Z):
    """Compute sample moments"""
    residuals = Y - X @ beta
    g = Z.T @ residuals / len(Y)  # Average moment (r x 1)
    return g

def gmm_objective(beta, Y, X, Z, W):
    """GMM objective: Q = g'Wg"""
    g = moments(beta, Y, X, Z)
    Q = g @ W @ g
    return Q

# Step 1: Initial estimation with W = I
print(f"\nStep 1: Initial GMM (W = Identity)")
W_init = np.eye(Z.shape[1])

result_gmm_step1 = minimize(
    gmm_objective,
    beta_ols,  # Starting values
    args=(Y, X, Z, W_init),
    method='BFGS'
)

beta_gmm_step1 = result_gmm_step1.x
print(f"  Converged: {result_gmm_step1.success}")
print(f"  Objective: {result_gmm_step1.fun:.6f}")

# Step 2: Compute optimal weighting matrix
print(f"\nStep 2: Optimal Weighting Matrix")

# Sample moments at step 1 estimate
residuals_step1 = Y - X @ beta_gmm_step1
g_i = Z * residuals_step1[:, np.newaxis]  # n x r matrix

# Covariance matrix (HAC with Newey-West)
def newey_west(g_i, lags=4):
    """Newey-West HAC covariance matrix"""
    n, r = g_i.shape
    S = g_i.T @ g_i / n  # Variance
    
    # Add autocovariances
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)  # Bartlett kernel
        gamma = g_i[lag:].T @ g_i[:-lag] / n
        S += weight * (gamma + gamma.T)
    
    return S

S_hat = newey_west(g_i)
W_opt = inv(S_hat)

print(f"  Optimal W computed (HAC with Newey-West)")

# Step 3: Re-estimate with optimal W
print(f"\nStep 3: Final GMM (W = Sâ»Â¹)")

result_gmm_step2 = minimize(
    gmm_objective,
    beta_gmm_step1,
    args=(Y, X, Z, W_opt),
    method='BFGS'
)

beta_gmm = result_gmm_step2.x
Q_gmm = result_gmm_step2.fun

print(f"  Converged: {result_gmm_step2.success}")
print(f"  Objective: {Q_gmm:.6f}")

# Compute standard errors
def gmm_variance(beta, Y, X, Z, W, S):
    """GMM variance (sandwich estimator)"""
    n = len(Y)
    
    # Jacobian: G = âˆ‚á¸¡/âˆ‚Î² = -Z'X/n
    G = -Z.T @ X / n
    
    # Variance: (G'WG)â»Â¹ G'WSWG (G'WG)â»Â¹
    GWG_inv = inv(G.T @ W @ G)
    middle = G.T @ W @ S @ W @ G
    V = GWG_inv @ middle @ GWG_inv / n
    
    return V

V_gmm = gmm_variance(beta_gmm, Y, X, Z, W_opt, S_hat)
se_gmm = np.sqrt(np.diag(V_gmm))

print(f"\nGMM Estimates:")
for i, (b_gmm, b_true, se) in enumerate(zip(beta_gmm, beta_true, se_gmm)):
    z_stat = (b_gmm - b_true) / se
    print(f"  Î²{i}: {b_gmm:7.4f} (SE={se:.4f}, true={b_true:.2f}) "
          f"[z={z_stat:.2f}]")

# ===== Hansen J-Test =====
print("\n" + "="*80)
print("HANSEN J-TEST (Overidentification)")
print("="*80)

# J-statistic: n * á¸¡'Sâ»Â¹á¸¡
g_bar = moments(beta_gmm, Y, X, Z)
J_stat = n * g_bar @ inv(S_hat) @ g_bar
df_J = Z.shape[1] - X.shape[1]  # # instruments - # parameters
p_value_J = 1 - stats.chi2.cdf(J_stat, df_J)

print(f"Overidentifying Restrictions Test:")
print(f"  # instruments: {Z.shape[1]}")
print(f"  # parameters: {X.shape[1]}")
print(f"  # overidentifying: {df_J}")
print(f"\n  J-statistic: {J_stat:.4f}")
print(f"  Degrees of freedom: {df_J}")
print(f"  P-value: {p_value_J:.4f}")

if p_value_J > 0.05:
    print(f"  âœ“ Cannot reject Hâ‚€: Instruments valid (p > 0.05)")
else:
    print(f"  âœ— Reject Hâ‚€: Some instruments may be invalid")

# ===== Continuous Updating GMM (CUE) =====
print("\n" + "="*80)
print("CONTINUOUS UPDATING GMM")
print("="*80)

def cue_objective(beta, Y, X, Z):
    """CUE objective: Update W at each evaluation"""
    residuals = Y - X @ beta
    g_i = Z * residuals[:, np.newaxis]
    
    # Update S
    S = newey_west(g_i)
    W = inv(S)
    
    # Moment vector
    g = moments(beta, Y, X, Z)
    
    # Objective
    Q = g @ W @ g
    return Q

result_cue = minimize(
    cue_objective,
    beta_ols,
    args=(Y, X, Z),
    method='Nelder-Mead',  # Derivative-free (W changes)
    options={'maxiter': 5000}
)

beta_cue = result_cue.x
Q_cue = result_cue.fun

print(f"CUE converged: {result_cue.success}")
print(f"Objective: {Q_cue:.6f}")

print(f"\nCUE Estimates:")
for i, (b_cue, b_gmm, b_true) in enumerate(zip(beta_cue, beta_gmm, beta_true)):
    print(f"  Î²{i}: {b_cue:7.4f} (GMM={b_gmm:.4f}, true={b_true:.2f})")

# ===== Comparison Table =====
print("\n" + "="*80)
print("ESTIMATOR COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Estimator': ['True', 'OLS', '2SLS', 'GMM (2-step)', 'CUE'],
    'Î²0': [beta_true[0], beta_ols[0], beta_2sls_exact[0], beta_gmm[0], beta_cue[0]],
    'Î²1_endo': [beta_true[1], beta_ols[1], beta_2sls_exact[1], beta_gmm[1], beta_cue[1]],
    'Î²2_exog': [beta_true[2], beta_ols[2], beta_2sls_exact[2], beta_gmm[2], beta_cue[2]],
    'SE(Î²1)': [0, se_ols[1], se_2sls_exact[1], se_gmm[1], np.nan]
})

print(comparison.to_string(index=False))

print(f"\nBias in Î²â‚ (endogenous coefficient):")
print(f"  OLS: {beta_ols[1] - beta_true[1]:+.4f} (inconsistent)")
print(f"  2SLS: {beta_2sls_exact[1] - beta_true[1]:+.4f}")
print(f"  GMM: {beta_gmm[1] - beta_true[1]:+.4f}")
print(f"  CUE: {beta_cue[1] - beta_true[1]:+.4f}")

# ===== Weak IV Simulation =====
print("\n" + "="*80)
print("WEAK INSTRUMENT SIMULATION")
print("="*80)

# Reduce first-stage coefficients
pi_1_weak = 0.1
pi_2_weak = 0.1

print(f"Weak instrument scenario:")
print(f"  First-stage coefficients: Ï€â‚={pi_1_weak}, Ï€â‚‚={pi_2_weak}")

# Generate weak IV data
v_weak = rho * u + np.sqrt(1 - rho**2) * np.random.randn(n)
X_endo_weak = 1.0 + pi_1_weak * Z1 + pi_2_weak * Z2 + v_weak
X_weak = np.column_stack([np.ones(n), X_endo_weak, X_exog])
Y_weak = X_weak @ beta_true + u

# First-stage F-test
first_stage_weak = lstsq(Z, X_endo_weak)[0]
X_endo_weak_hat = Z @ first_stage_weak
residuals_fs_weak = X_endo_weak - X_endo_weak_hat
RSS_fs_weak = np.sum(residuals_fs_weak**2)
TSS_fs_weak = np.sum((X_endo_weak - X_endo_weak.mean())**2)
R2_fs_weak = 1 - RSS_fs_weak / TSS_fs_weak
F_stat_weak = (R2_fs_weak / (Z.shape[1] - 1)) / ((1 - R2_fs_weak) / (n - Z.shape[1]))

print(f"  First-stage F: {F_stat_weak:.2f} (weak!)")

# GMM estimation
result_gmm_weak_step1 = minimize(
    gmm_objective,
    np.zeros(X_weak.shape[1]),
    args=(Y_weak, X_weak, Z, W_init),
    method='BFGS'
)

beta_gmm_weak_step1 = result_gmm_weak_step1.x

residuals_weak_step1 = Y_weak - X_weak @ beta_gmm_weak_step1
g_i_weak = Z * residuals_weak_step1[:, np.newaxis]
S_hat_weak = newey_west(g_i_weak)
W_opt_weak = inv(S_hat_weak)

result_gmm_weak = minimize(
    gmm_objective,
    beta_gmm_weak_step1,
    args=(Y_weak, X_weak, Z, W_opt_weak),
    method='BFGS'
)

beta_gmm_weak = result_gmm_weak.x

print(f"\nGMM with Weak Instruments:")
print(f"  Î²â‚_endo: {beta_gmm_weak[1]:.4f} (true={beta_true[1]:.2f})")
print(f"  Bias: {beta_gmm_weak[1] - beta_true[1]:+.4f}")
print(f"  âš  Finite-sample bias with weak instruments")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: First Stage
axes[0, 0].scatter(Z1, X_endo, alpha=0.5, s=20, label='Data')
axes[0, 0].plot(sorted(Z1), sorted(Z1) * pi_1 + 1, 'r--', linewidth=2,
               label=f'True (Ï€={pi_1})')
axes[0, 0].plot(sorted(Z1), first_stage[0] + first_stage[1] * sorted(Z1),
               'b-', linewidth=2, label='Estimated')
axes[0, 0].set_xlabel('Instrument (Zâ‚)')
axes[0, 0].set_ylabel('Endogenous Variable (Xâ‚)')
axes[0, 0].set_title(f'First Stage (F={F_stat:.1f})')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Plot 2: OLS vs True
axes[0, 1].scatter(X @ beta_true, Y, alpha=0.5, s=20, label='Data')
axes[0, 1].plot(X @ beta_true, X @ beta_true, 'r--', linewidth=2,
               label='True')
axes[0, 1].plot(X @ beta_true, X @ beta_ols, 'b-', linewidth=2,
               label='OLS (biased)')
axes[0, 1].set_xlabel('True XÎ²')
axes[0, 1].set_ylabel('Predicted Y')
axes[0, 1].set_title('OLS: Endogeneity Bias')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Coefficient Comparison
methods = ['True', 'OLS', '2SLS', 'GMM', 'CUE']
beta1_vals = [beta_true[1], beta_ols[1], beta_2sls_exact[1], beta_gmm[1], beta_cue[1]]
colors = ['green', 'red', 'blue', 'purple', 'orange']

axes[0, 2].barh(methods, beta1_vals, color=colors, alpha=0.7)
axes[0, 2].axvline(beta_true[1], color='black', linestyle='--', linewidth=2)
axes[0, 2].set_xlabel('Î²â‚ (Endogenous Coefficient)')
axes[0, 2].set_title('Estimator Comparison')
axes[0, 2].grid(alpha=0.3, axis='x')

# Plot 4: Residuals Distribution
axes[1, 0].hist(residuals_ols, bins=30, alpha=0.6, label='OLS', density=True)
axes[1, 0].hist(Y - X @ beta_gmm, bins=30, alpha=0.6, label='GMM', density=True)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Residual Distributions')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Moment Conditions
g_vals_beta1 = []
beta1_grid = np.linspace(beta_true[1] - 1, beta_true[1] + 1, 50)

for b1 in beta1_grid:
    beta_temp = np.array([beta_gmm[0], b1, beta_gmm[2]])
    g_temp = moments(beta_temp, Y, X, Z)
    g_vals_beta1.append(np.linalg.norm(g_temp))

axes[1, 1].plot(beta1_grid, g_vals_beta1, linewidth=2)
axes[1, 1].axvline(beta_true[1], color='green', linestyle='--',
                  linewidth=2, label='True')
axes[1, 1].axvline(beta_gmm[1], color='red', linestyle='--',
                  linewidth=2, label='GMM')
axes[1, 1].set_xlabel('Î²â‚')
axes[1, 1].set_ylabel('||g(Î²)||')
axes[1, 1].set_title('Moment Conditions (GMM minimizes)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Weak IV Comparison
methods_weak = ['Strong IV\n(F=%.0f)' % F_stat, 'Weak IV\n(F=%.0f)' % F_stat_weak]
beta1_strong = beta_gmm[1]
beta1_weak = beta_gmm_weak[1]
bias_strong = beta1_strong - beta_true[1]
bias_weak = beta1_weak - beta_true[1]

x_pos = [0, 1]
axes[1, 2].bar(x_pos, [bias_strong, bias_weak], color=['blue', 'red'], alpha=0.7)
axes[1, 2].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(methods_weak, fontsize=9)
axes[1, 2].set_ylabel('Bias in Î²â‚')
axes[1, 2].set_title('Weak IV Problem')
axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('gmm_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Endogeneity Problem:")
print(f"   OLS bias: {bias_ols:+.4f} (upward due to positive correlation)")
print(f"   GMM corrects via instrumental variables")

print("\n2. Overidentification:")
print(f"   {Z.shape[1]} instruments for {X.shape[1]} parameters")
print(f"   J-test p-value: {p_value_J:.4f}")
if p_value_J > 0.05:
    print(f"   âœ“ Instruments pass validity test")

print("\n3. Efficiency:")
print(f"   2SLS SE(Î²â‚): {se_2sls_exact[1]:.4f}")
print(f"   GMM SE(Î²â‚): {se_gmm[1]:.4f}")
if se_gmm[1] < se_2sls_exact[1]:
    improvement = (1 - se_gmm[1] / se_2sls_exact[1]) * 100
    print(f"   âœ“ GMM {improvement:.1f}% more efficient (optimal weighting)")

print("\n4. Weak Instruments:")
print(f"   Strong IV (F={F_stat:.1f}): Bias={bias_strong:+.4f}")
print(f"   Weak IV (F={F_stat_weak:.1f}): Bias={bias_weak:+.4f}")
print(f"   âš  Weak IV bias {abs(bias_weak)/abs(bias_strong):.1f}Ã— larger")

print("\n5. Practical Recommendations:")
print("   â€¢ Check first-stage F-statistic (want F > 10)")
print("   â€¢ Use HAC standard errors for robustness")
print("   â€¢ Test overidentifying restrictions (J-test)")
print("   â€¢ CUE better with weak identification")
print("   â€¢ Two-step GMM most common in practice")
print("   â€¢ Apply Windmeijer correction for 2-step SEs")

print("\n6. GMM Advantages:")
print("   â€¢ Robust to heteroskedasticity, autocorrelation (HAC)")
print("   â€¢ Flexible framework (nonlinear, panel, etc.)")
print("   â€¢ Overidentification testable")
print("   â€¢ Optimal weighting for efficiency")
