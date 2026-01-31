import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy import stats
from scipy.optimize import minimize

# Simulation: Dynamic panel data
np.random.seed(42)
N = 100  # Individuals
T = 10   # Time periods

# True parameters
alpha = 0.6   # Lagged Y effect
beta = 1.0    # X effect

# Generate data
Y = np.zeros((N, T))
X = np.random.uniform(0, 5, (N, T))
alpha_i = np.random.normal(0, 1, N)  # Fixed effects
epsilon = np.random.normal(0, 0.5, (N, T))  # Idiosyncratic error

# Initialize Y
Y[:, 0] = alpha_i + X[:, 0] + epsilon[:, 0]

# Generate panel
for t in range(1, T):
    Y[:, t] = alpha * Y[:, t-1] + beta * X[:, t] + alpha_i + epsilon[:, t]

print("=" * 80)
print("GENERALIZED METHOD OF MOMENTS (GMM): DYNAMIC PANEL ESTIMATION")
print("=" * 80)

print("\nTrue Parameters:")
print(f"  Î± (lagged Y effect): {alpha}")
print(f"  Î² (X effect): {beta}")

# Reshape to long format for estimation
y_long = Y[:, 1:].flatten()  # Dependent var (drop t=0)
y_lag = Y[:, :-1].flatten()  # Lagged Y
x_long = X[:, 1:].flatten()  # X regressor
id_long = np.repeat(np.arange(N), T-1)  # Individual ID
t_long = np.tile(np.arange(1, T), N)    # Time ID

# Scenario 1: Naive OLS (biased due to endogeneity)
print("\n\nScenario 1: NAIVE OLS (Lagged Y Treated as Exogenous - BIASED)")
print("-" * 80)

X_ols = np.column_stack([np.ones(len(y_long)), y_lag, x_long])
beta_ols = inv(X_ols.T @ X_ols) @ (X_ols.T @ y_long)

print(f"OLS estimates:")
print(f"  Intercept: {beta_ols[0]:.6f}")
print(f"  Î±Ì‚ (lagged Y): {beta_ols[1]:.6f} (TRUE: {alpha}, BIASED!)")
print(f"  Î²Ì‚ (X): {beta_ols[2]:.6f} (TRUE: {beta}, BIASED!)")

# Scenario 2: Arellano-Bond GMM (1-step)
print("\n\nScenario 2: ARELLANO-BOND GMM (1-STEP - ROBUST)")
print("-" * 80)

# First-difference transformation
y_diff = np.diff(Y, axis=1).flatten()  # Î”y
y_lag_diff = np.diff(Y, axis=1)[:, :-1].flatten()  # Î”y_{t-1}
x_diff = np.diff(X, axis=1).flatten()  # Î”x

# Build instrument matrix (use lagged levels)
# For each t, instrument is Y_{t-2}, Y_{t-3}, etc.
# Simplified: use one instrument per observation (Y_{t-2})

n_obs = len(y_diff)
instruments = []

for i in range(N):
    for t in range(2, T):  # Start from t=2 (need Y_{t-2})
        idx = i * (T - 1) + (t - 1)
        if idx < len(y_diff):
            instruments.append(Y[i, t-2])  # Instrument: lagged level

instruments = np.array(instruments[:n_obs])

# Moment condition: E[Î”Y_{t-1} Ã— (Y_{t-2})] = 0 in valid model
# Estimate with identity weight (1-step)

# For tractability, use simplified model: Î”y = Î± Ã— Î”y_lag + Î² Ã— Î”x + error
X_gmm_diff = np.column_stack([y_lag_diff, x_diff])

# Define moment function
def moment_function(params, y, X, Z):
    """
    params: [alpha, beta]
    y: dependent variable (differenced)
    X: regressors (differenced)
    Z: instruments (lagged levels)
    """
    alpha, beta = params
    errors = y - X @ np.array([alpha, beta])
    # Moment condition: E[Z Ã— error] = 0
    moments = Z * errors
    return moments

# 1-step GMM: minimize with W = I
def gmm_objective_1step(params):
    moments = moment_function(params, y_diff, X_gmm_diff, instruments)
    return np.sum(moments**2) / len(moments)

result_1step = minimize(gmm_objective_1step, x0=[0.5, 0.8], method='Nelder-Mead')
beta_gmm_1step = result_1step.x

print(f"1-step GMM estimates:")
print(f"  Î±Ì‚ (lagged Y): {beta_gmm_1step[0]:.6f} (TRUE: {alpha})")
print(f"  Î²Ì‚ (X): {beta_gmm_1step[1]:.6f} (TRUE: {beta})")

# Scenario 3: 2-step GMM (efficient)
print("\n\nScenario 3: 2-STEP GMM (EFFICIENT)")
print("-" * 80)

# Stage 1: Use 1-step estimates
alpha_1st, beta_1st = beta_gmm_1step

# Stage 2: Estimate optimal weight matrix
errors_1st = y_diff - X_gmm_diff @ np.array([alpha_1st, beta_1st])
moment_var = np.mean((instruments * errors_1st)**2)  # Var[Z Ã— error]

# Weight matrix (inverse of moment variance)
W_optimal = 1 / moment_var

# 2-step GMM objective
def gmm_objective_2step(params):
    moments = moment_function(params, y_diff, X_gmm_diff, instruments)
    return np.sum((moments / np.sqrt(moment_var))**2) / len(moments)

result_2step = minimize(gmm_objective_2step, x0=[0.5, 0.8], method='Nelder-Mead')
beta_gmm_2step = result_2step.x

print(f"2-step GMM estimates:")
print(f"  Î±Ì‚ (lagged Y): {beta_gmm_2step[0]:.6f} (TRUE: {alpha})")
print(f"  Î²Ì‚ (X): {beta_gmm_2step[1]:.6f} (TRUE: {beta})")

# Hansen J-test for overidentification
print("\n\nHANSEN J-TEST FOR OVERIDENTIFICATION")
print("-" * 80)

errors_gmm = y_diff - X_gmm_diff @ beta_gmm_2step
moment_vals = instruments * errors_gmm
G_bar = np.mean(moment_vals**2)

# J-statistic
J_stat = n_obs * G_bar / moment_var

# Critical value (chi-square with 1 df for one overidentifying restriction)
chi2_crit = stats.chi2.ppf(0.95, df=1)
p_value = 1 - stats.chi2.cdf(J_stat, df=1)

print(f"J-statistic: {J_stat:.4f}")
print(f"Ï‡Â²â‚€.â‚€â‚…(1): {chi2_crit:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value > 0.05:
    print("âœ“ Fail to reject Hâ‚€: Instruments valid (moment conditions supported)")
else:
    print("âœ— Reject Hâ‚€: Possible invalid instruments or misspecification")

print("=" * 80)

# Summary comparison
print("\n\nSUMMARY: ESTIMATOR COMPARISON")
print("-" * 80)
print(f"{'Estimator':<20} {'Î±Ì‚':<12} {'Î²Ì‚':<12} {'Bias (Î±)':<12} {'Bias (Î²)':<12}")
print("-" * 80)
print(f"{'OLS (naive)':<20} {beta_ols[1]:<12.6f} {beta_ols[2]:<12.6f} "
      f"{beta_ols[1]-alpha:<12.6f} {beta_ols[2]-beta:<12.6f}")
print(f"{'1-step GMM':<20} {beta_gmm_1step[0]:<12.6f} {beta_gmm_1step[1]:<12.6f} "
      f"{beta_gmm_1step[0]-alpha:<12.6f} {beta_gmm_1step[1]-beta:<12.6f}")
print(f"{'2-step GMM':<20} {beta_gmm_2step[0]:<12.6f} {beta_gmm_2step[1]:<12.6f} "
      f"{beta_gmm_2step[0]-alpha:<12.6f} {beta_gmm_2step[1]-beta:<12.6f}")
print(f"{'TRUE':<20} {alpha:<12.6f} {beta:<12.6f} {'-':<12} {'-':<12}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Estimator comparison
estimators = ['OLS\n(naive)', '1-step\nGMM', '2-step\nGMM', 'True\nvalue']
alpha_hats = [beta_ols[1], beta_gmm_1step[0], beta_gmm_2step[0], alpha]
beta_hats = [beta_ols[2], beta_gmm_1step[1], beta_gmm_2step[1], beta]

x_pos = np.arange(len(estimators))
width = 0.35

axes[0].bar(x_pos - width/2, alpha_hats, width, label='Î±Ì‚', alpha=0.8, color='blue')
axes[0].bar(x_pos + width/2, beta_hats, width, label='Î²Ì‚', alpha=0.8, color='orange')
axes[0].axhline(y=alpha, color='blue', linestyle='--', linewidth=1, alpha=0.5)
axes[0].axhline(y=beta, color='orange', linestyle='--', linewidth=1, alpha=0.5)
axes[0].set_ylabel('Parameter Estimate', fontsize=11, fontweight='bold')
axes[0].set_title('GMM vs. OLS: Bias Comparison', fontsize=12, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(estimators)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Moment condition visualization
axes[1].scatter(instruments, errors_gmm, alpha=0.5, s=20)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='E[ZÃ—Îµ]=0')
axes[1].set_xlabel('Instrument (Lagged Y)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Residual (from 2-step GMM)', fontsize=11, fontweight='bold')
axes[1].set_title(f'Moment Condition Check (J-stat={J_stat:.3f}, p={p_value:.3f})', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gmm_dynamic_panel.png', dpi=150)
plt.show()
