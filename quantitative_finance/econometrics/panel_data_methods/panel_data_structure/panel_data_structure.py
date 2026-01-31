import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import inv

print("=" * 90)
print("PANEL DATA STRUCTURE & LONGITUDINAL ANALYSIS")
print("=" * 90)

# Generate panel data
np.random.seed(42)
N = 50  # Firms
T = 10  # Years

# Create panel structure
firms = np.repeat(np.arange(1, N+1), T)
years = np.tile(np.arange(1, T+1), N)
data = pd.DataFrame({'firm': firms, 'year': years})

# Parameters
beta_true = 0.5
sigma_u = 2.0    # Firm heterogeneity SD
sigma_eps = 1.0  # Idiosyncratic error SD

# Generate fixed effects (firm-specific, time-invariant)
alpha_firm = np.random.normal(100, sigma_u, N)
data['alpha'] = data['firm'].map(lambda i: alpha_firm[i-1])

# Generate time effects
gamma_year = np.random.normal(0, 0.5, T)
data['gamma'] = data['year'].map(lambda t: gamma_year[t-1])

# Generate exogenous regressor X (varies by firm and year)
data['X'] = np.random.normal(50, 10, len(data))

# Generate idiosyncratic error
data['eps'] = np.random.normal(0, sigma_eps, len(data))

# Generate outcome: Y = alpha_i + gamma_t + beta*X + eps
data['Y'] = data['alpha'] + data['gamma'] + beta_true * data['X'] + data['eps']

print(f"\nPanel Structure: N={N} firms, T={T} years")
print(f"Total observations: {len(data)}")
print(f"\nTrue parameters:")
print(f"  Î² (effect of X): {beta_true}")
print(f"  Ïƒ_u (firm heterogeneity): {sigma_u}")
print(f"  Ïƒ_Îµ (idiosyncratic error): {sigma_eps}")

print(f"\nFirst 10 observations:")
print(data.head(10).to_string())

# Scenario 1: Naive OLS (ignoring structure)
print("\n" + "=" * 90)
print("SCENARIO 1: NAIVE OLS (Ignores Panel Structure)")
print("=" * 90)

X_ols = np.column_stack([np.ones(len(data)), data['X'].values])
y_ols = data['Y'].values
beta_ols = inv(X_ols.T @ X_ols) @ X_ols.T @ y_ols

print(f"\nOLS Estimate: Î²Ì‚_OLS = {beta_ols[1]:.4f}")
print(f"  (Close to true Î²={beta_true}, but ignores clustering)")

# Scenario 2: Fixed Effects estimation (within transformation)
print("\n" + "=" * 90)
print("SCENARIO 2: FIXED EFFECTS MODEL (Within-Transformation)")
print("=" * 90)

# Within-transformation: demean by firm and year
data['X_dm'] = data.groupby('firm')['X'].transform(lambda x: x - x.mean())
data['Y_dm'] = data.groupby('firm')['Y'].transform(lambda x: x - x.mean())

# FE estimation (no intercept needed after demeaning)
X_fe = data['X_dm'].values.reshape(-1, 1)
y_fe = data['Y_dm'].values
beta_fe = inv(X_fe.T @ X_fe) @ X_fe.T @ y_fe

resid_fe = y_fe - X_fe @ beta_fe
rss_fe = np.sum(resid_fe**2)
se_fe = np.sqrt(rss_fe / (len(data) - N - 1)) / np.sqrt(np.sum(X_fe**2))

print(f"\nFE Estimate (Within-Transformation):")
print(f"  Î²Ì‚_FE = {beta_fe[0]:.4f}")
print(f"  SE(Î²Ì‚_FE) = {se_fe:.4f}")
print(f"  (Robust to time-invariant unobservables)")

# Recover fixed effects
firm_means_Y = data.groupby('firm')['Y'].mean().values
firm_means_X = data.groupby('firm')['X'].mean().values
alpha_hat = firm_means_Y - beta_fe[0] * firm_means_X

print(f"\nRecovered Fixed Effects (first 10 firms):")
print(f"{'Firm':<8} {'True Î±':<12} {'Estimated Î±':<12} {'Error':<12}")
print("-" * 44)
for i in range(min(10, N)):
    print(f"{i+1:<8} {alpha_firm[i]:>11.4f} {alpha_hat[i]:>11.4f} {alpha_hat[i]-alpha_firm[i]:>11.4f}")

# Scenario 3: Random Effects estimation (GLS)
print("\n" + "=" * 90)
print("SCENARIO 3: RANDOM EFFECTS MODEL (GLS Estimation)")
print("=" * 90)

# Step 1: OLS for preliminary estimates
X_full = np.column_stack([np.ones(len(data)), data['X'].values])
beta_ols_re = inv(X_full.T @ X_full) @ X_full.T @ data['Y'].values
resid_ols = data['Y'].values - X_full @ beta_ols_re

# Step 2: Estimate variance components
# ÏƒÂ²_Îµ from within-group residuals
residuals_within = []
for firm in data['firm'].unique():
    firm_data = data[data['firm'] == firm]
    X_firm = firm_data[['X']].values
    y_firm = firm_data['Y'].values
    if len(firm_data) > 1:
        beta_firm = np.polyfit(X_firm.flatten(), y_firm, 1)[0]
        residuals_within.extend(y_firm - (firm_data['X'].mean() * beta_firm + 
                                         (y_firm - y_firm.mean() - 
                                          beta_firm * (X_firm.flatten() - X_firm.mean()))))

sigma2_eps = np.var(resid_ols)

# ÏƒÂ²_u from between-group residuals
firm_means = data.groupby('firm')[['Y', 'X']].mean()
y_firm_mean = firm_means['Y'].values
X_firm_mean = firm_means['X'].values
beta_between = np.polyfit(X_firm_mean, y_firm_mean, 1)[0]
residuals_between = y_firm_mean - (beta_between * X_firm_mean + (y_firm_mean.mean() - 
                                                                  beta_between * X_firm_mean.mean()))
sigma2_u = max(0, (np.var(residuals_between) - sigma2_eps / T))

print(f"\nVariance Component Estimates:")
print(f"  ÏƒÌ‚Â²_u (firm heterogeneity): {sigma2_u:.4f} (true: {sigma_u**2:.4f})")
print(f"  ÏƒÌ‚Â²_Îµ (idiosyncratic): {sigma2_eps:.4f} (true: {sigma_eps**2:.4f})")

# GLS transformation parameter
theta = 1 - np.sqrt(sigma2_eps / (sigma2_eps + T * sigma2_u))
print(f"  Î¸ (GLS weight): {theta:.4f}")

# Apply GLS transformation
data['Y_gls'] = data['Y'] - theta * data.groupby('firm')['Y'].transform('mean')
data['X_gls'] = data['X'] - theta * data.groupby('firm')['X'].transform('mean')
data['const_gls'] = 1 - theta

X_gls = np.column_stack([data['const_gls'].values, data['X_gls'].values])
y_gls = data['Y_gls'].values
beta_gls = inv(X_gls.T @ X_gls) @ X_gls.T @ y_gls

print(f"\nRE Estimate (GLS):")
print(f"  Î²Ì‚_RE = {beta_gls[1]:.4f}")
print(f"  (More efficient than FE if assumptions hold)")

# Scenario 4: Hausman test (FE vs RE)
print("\n" + "=" * 90)
print("SCENARIO 4: HAUSMAN TEST (FE vs RE)")
print("=" * 90)

# Hausman statistic: H = (Î²_FE - Î²_RE)' * Var(Î²_FE - Î²_RE)^{-1} * (Î²_FE - Î²_RE)
# Approximate variance under H_0
var_fe = (sigma2_eps / np.sum(data['X_dm']**2)) if len(data) > N else np.inf
var_re = (sigma2_eps / np.sum(data['X_gls']**2)) if len(data) > N else np.inf
var_diff = var_fe + var_re  # Approximate (conservative)

hausman_stat = ((beta_fe[0] - beta_gls[1])**2) / var_diff

# Critical value
chi2_crit = stats.chi2.ppf(0.95, df=1)
p_value = 1 - stats.chi2.cdf(hausman_stat, df=1)

print(f"\nHausman Test: Hâ‚€ RE consistent vs. Hâ‚ FE needed")
print(f"  Î²Ì‚_FE = {beta_fe[0]:.4f}")
print(f"  Î²Ì‚_RE = {beta_gls[1]:.4f}")
print(f"  Difference: {beta_fe[0] - beta_gls[1]:.4f}")
print(f"  Hausman H = {hausman_stat:.4f}")
print(f"  Ï‡Â²â‚€.â‚€â‚…(1) = {chi2_crit:.4f}")
print(f"  p-value = {p_value:.4f}")

if p_value < 0.05:
    print(f"  âœ“ REJECT Hâ‚€: Significant difference â†’ Use FE (RE inconsistent)")
else:
    print(f"  âœ— FAIL TO REJECT Hâ‚€: Use RE (more efficient)")

print("=" * 90)

# Summary comparison
print("\n\nSUMMARY: ESTIMATOR COMPARISON")
print("-" * 90)
print(f"{'Model':<20} {'Î²Ì‚':<12} {'True Î²':<12} {'Bias':<12} {'Efficiency':<15}")
print("-" * 90)
print(f"{'OLS':<20} {beta_ols[1]:>11.4f} {beta_true:>11.4f} {beta_ols[1]-beta_true:>11.4f} {'N/A':<15}")
print(f"{'Fixed Effects':<20} {beta_fe[0]:>11.4f} {beta_true:>11.4f} {beta_fe[0]-beta_true:>11.4f} {'Lower':<15}")
print(f"{'Random Effects':<20} {beta_gls[1]:>11.4f} {beta_true:>11.4f} {beta_gls[1]-beta_true:>11.4f} {'Higher':<15}")
print(f"{'Hausman Test Result':<20} {'-':<11} {'-':<11} {'-':<11} {'FE preferred':<15}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Fixed effects (firm heterogeneity)
ax1 = axes[0, 0]
sorted_idx = np.argsort(alpha_firm)
ax1.scatter(range(N), alpha_firm[sorted_idx], alpha=0.6, s=50, label='True Î±', color='blue')
ax1.scatter(range(N), alpha_hat[sorted_idx], alpha=0.6, s=50, label='Estimated Î±', color='red', marker='^')
ax1.set_xlabel('Firm (sorted)', fontweight='bold')
ax1.set_ylabel('Fixed Effect (Î±)', fontweight='bold')
ax1.set_title('Firm Fixed Effects: True vs Estimated', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Model comparison (coefficients)
ax2 = axes[0, 1]
models = ['OLS', 'FE', 'RE']
betas = [beta_ols[1], beta_fe[0], beta_gls[1]]
colors_mod = ['gray', 'green', 'blue']
bars = ax2.bar(models, betas, color=colors_mod, alpha=0.7)
ax2.axhline(y=beta_true, color='red', linestyle='--', linewidth=2, label=f'True Î² = {beta_true}')
ax2.set_ylabel('Î²Ì‚ Estimate', fontweight='bold')
ax2.set_title('Model Comparison: Treatment Effect Estimates', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, beta in zip(bars, betas):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{beta:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. Residuals vs X (showing structure)
ax3 = axes[1, 0]
for firm_id in data['firm'].unique()[:5]:  # Show 5 firms for clarity
    firm_data = data[data['firm'] == firm_id]
    ax3.scatter(firm_data['X'], firm_data['Y'], alpha=0.6, s=40, label=f'Firm {firm_id}')
ax3.set_xlabel('X', fontweight='bold')
ax3.set_ylabel('Y', fontweight='bold')
ax3.set_title('Panel Data: Within-Firm vs Between-Firm Variation', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Efficiency comparison (variance)
ax4 = axes[1, 1]
var_estimates = [var_fe, var_re]
model_labels = ['FE', 'RE']
colors_var = ['green', 'blue']
bars_var = ax4.bar(model_labels, var_estimates, color=colors_var, alpha=0.7)
ax4.set_ylabel('Variance of Î²Ì‚', fontweight='bold')
ax4.set_title('Efficiency Comparison (Lower Var is Better)', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for bar, var in zip(bars_var, var_estimates):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{var:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('panel_data_analysis.png', dpi=150)
plt.show()
