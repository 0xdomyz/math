import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

# Data generation with selection bias
np.random.seed(42)
n = 1000

# Covariates (confounders)
X1 = np.random.uniform(0, 10, n)  # Age-like
X2 = np.random.uniform(0, 100, n)  # Income
X3 = np.random.binomial(1, 0.3, n)  # Binary (e.g., education)

X = np.column_stack([X1, X2, X3])

# Selection into treatment: P(T=1|X)
propensity_true = 1 / (1 + np.exp(-(0.15*X1 + 0.01*X2 - 0.8*X3 - 2)))
T = (np.random.uniform(0, 1, n) < propensity_true).astype(int)

# Potential outcomes
Y0 = 50 + 2*X1 + 0.1*X2 + 5*X3 + np.random.normal(0, 10, n)
Y1 = Y0 + 10  # True treatment effect = 10 (constant)
# But add selection effect: treated sicker (lower baseline Y0)
Y0_adjusted = Y0 - 15*T + np.random.normal(0, 5, n)
Y1_adjusted = Y0_adjusted + 10

# Observed outcome
Y = T * Y1_adjusted + (1 - T) * Y0_adjusted

data = pd.DataFrame({
    'T': T, 'Y': Y, 'Y0': Y0_adjusted, 'Y1': Y1_adjusted,
    'X1': X1, 'X2': X2, 'X3': X3, 'P_true': propensity_true
})

print("=" * 90)
print("PROPENSITY SCORE MATCHING: TREATMENT EFFECT ESTIMATION")
print("=" * 90)

print(f"\nSample size: n={n}")
print(f"Treated: n_T={data['T'].sum()}, Untreated: n_C={(1-data['T']).sum()}")
print(f"\nTrue Average Treatment Effect (ATE): 10.0")

# Scenario 1: Naive comparison (BIASED)
print("\n" + "=" * 90)
print("Scenario 1: NAIVE COMPARISON (Selection Bias - BIASED)")
print("=" * 90)

Y_treated_naive = data[data['T'] == 1]['Y'].mean()
Y_control_naive = data[data['T'] == 0]['Y'].mean()
ATE_naive = Y_treated_naive - Y_control_naive

print(f"E[Y|T=1] = {Y_treated_naive:.4f}")
print(f"E[Y|T=0] = {Y_control_naive:.4f}")
print(f"Naive ATE = {ATE_naive:.4f} (TRUE: 10.0, BIAS: {ATE_naive - 10:.4f})")
print("âœ— Large negative bias due to selection (treated have lower baseline outcomes)")

# Scenario 2: OLS regression (adjusts for observed confounders)
print("\n" + "=" * 90)
print("Scenario 2: OLS REGRESSION (Parametric Adjustment)")
print("=" * 90)

# OLS: Y = Î± + Î²*T + Î³1*X1 + Î³2*X2 + Î³3*X3 + Îµ
X_ols = np.column_stack([np.ones(n), T, X1, X2, X3])
beta_ols = np.linalg.inv(X_ols.T @ X_ols) @ X_ols.T @ Y
ATE_ols = beta_ols[1]

print(f"OLS Estimate of ATE: {ATE_ols:.4f}")
print(f"Bias: {ATE_ols - 10:.4f}")
print(f"Coefficient interpretation: Holding X1, X2, X3 constant, T increases Y by {ATE_ols:.4f}")

# Scenario 3: PSM - Estimate propensity score
print("\n" + "=" * 90)
print("Scenario 3: PROPENSITY SCORE MATCHING")
print("=" * 90)

# Step 1: Estimate propensity score P(T=1|X)
log_reg = LogisticRegression(fit_intercept=True, max_iter=1000)
log_reg.fit(X, T)
propensity_hat = log_reg.predict_proba(X)[:, 1]

data['P_hat'] = propensity_hat

print(f"\nStep 1: Propensity Score Estimation (Logit)")
print(f"  Coefficients: Î²1={log_reg.coef_[0, 0]:.4f}, Î²2={log_reg.coef_[0, 1]:.4f}, "
      f"Î²3={log_reg.coef_[0, 2]:.4f}")
print(f"  Intercept: {log_reg.intercept_[0]:.4f}")
print(f"  Propensity score range: [{propensity_hat.min():.4f}, {propensity_hat.max():.4f}]")

# Step 2: Check common support
overlap_min = max(propensity_hat[T==0].min(), propensity_hat[T==1].min())
overlap_max = min(propensity_hat[T==0].max(), propensity_hat[T==1].max())
print(f"\nStep 2: Common Support Check")
print(f"  Overlap region: P âˆˆ [{overlap_min:.4f}, {overlap_max:.4f}]")

# Trim to common support
in_support = (propensity_hat >= overlap_min) & (propensity_hat <= overlap_max)
data_support = data[in_support].copy()
print(f"  Sample retained in common support: {in_support.sum()}/{n} ({100*in_support.sum()/n:.1f}%)")

# Step 3: 1:1 Nearest Neighbor Matching with caliper
print(f"\nStep 3: 1:1 Nearest Neighbor Matching")
caliper = 0.05 * data_support['P_hat'].std()
print(f"  Caliper: {caliper:.6f}")

treated_idx = data_support[data_support['T'] == 1].index
control_idx = data_support[data_support['T'] == 0].index

matched_pairs = []
matched_controls = set()

for i in treated_idx:
    P_i = data_support.loc[i, 'P_hat']
    
    # Find nearest control within caliper
    distances = np.abs(data_support.loc[control_idx, 'P_hat'].values - P_i)
    closest_idx = distances.argmin()
    min_dist = distances[closest_idx]
    
    if min_dist <= caliper:
        control_matched = control_idx[closest_idx]
        if control_matched not in matched_controls:
            matched_pairs.append((i, control_matched))
            matched_controls.add(control_matched)

print(f"  Matched pairs: {len(matched_pairs)}")
print(f"  Matched treated: {len(matched_pairs)}")
print(f"  Matched untreated: {len(matched_controls)}")

# Extract matched sample
matched_treated = data_support.loc[[p[0] for p in matched_pairs]]
matched_control = data_support.loc[[p[1] for p in matched_pairs]]

# Step 4: Estimate ATE on matched sample
Y_treated_matched = matched_treated['Y'].mean()
Y_control_matched = matched_control['Y'].mean()
ATE_psm = Y_treated_matched - Y_control_matched

print(f"\nStep 4: ATE Estimation on Matched Sample")
print(f"  E[Y|T=1, matched] = {Y_treated_matched:.4f}")
print(f"  E[Y|T=0, matched] = {Y_control_matched:.4f}")
print(f"  PSM ATE = {ATE_psm:.4f}")
print(f"  Bias: {ATE_psm - 10:.4f}")

# Standard error
se_treated = matched_treated['Y'].std() / np.sqrt(len(matched_treated))
se_control = matched_control['Y'].std() / np.sqrt(len(matched_control))
se_psm = np.sqrt(se_treated**2 + se_control**2)
t_stat = ATE_psm / se_psm
print(f"  SE(ATE): {se_psm:.4f}")
print(f"  t-statistic: {t_stat:.4f}")

# Step 5: Covariate balance check (SMD)
print(f"\nStep 5: Covariate Balance (Standardized Mean Differences)")
print(f"{'Covariate':<12} {'Before (SMD)':<15} {'After (SMD)':<15} {'Balanced?':<12}")
print("-" * 60)

for var in ['X1', 'X2', 'X3']:
    X_val_before_t = data_support[data_support['T'] == 1][var]
    X_val_before_c = data_support[data_support['T'] == 0][var]
    
    X_val_after_t = matched_treated[var]
    X_val_after_c = matched_control[var]
    
    mean_diff_before = X_val_before_t.mean() - X_val_before_c.mean()
    mean_diff_after = X_val_after_t.mean() - X_val_after_c.mean()
    
    pooled_sd_before = np.sqrt((X_val_before_t.std()**2 + X_val_before_c.std()**2) / 2)
    pooled_sd_after = np.sqrt((X_val_after_t.std()**2 + X_val_after_c.std()**2) / 2)
    
    smd_before = mean_diff_before / pooled_sd_before
    smd_after = mean_diff_after / pooled_sd_after
    
    balanced = "âœ“" if abs(smd_after) < 0.1 else "âœ—"
    print(f"{var:<12} {smd_before:>14.4f} {smd_after:>14.4f} {balanced:>11}")

print("=" * 90)

# Summary comparison
print("\n\nSUMMARY: ESTIMATOR COMPARISON")
print("-" * 90)
print(f"{'Estimator':<20} {'ATE':<12} {'Bias':<12} {'SE':<12} {'t-stat':<12}")
print("-" * 90)

# Naive SE
se_naive = np.sqrt(Y_treated_naive**2 / data_support[data_support['T']==1].shape[0] + 
                    Y_control_naive**2 / data_support[data_support['T']==0].shape[0])

# OLS SE from residuals
resid_ols = Y - X_ols @ beta_ols
sigma2_ols = np.sum(resid_ols**2) / (n - X_ols.shape[1])
var_ols = sigma2_ols * np.linalg.inv(X_ols.T @ X_ols)[1, 1]
se_ols = np.sqrt(var_ols)

print(f"{'Naive':<20} {ATE_naive:>11.4f} {ATE_naive - 10:>11.4f} {se_naive:>11.4f} {ATE_naive/se_naive:>11.4f}")
print(f"{'OLS (adjusted)':<20} {ATE_ols:>11.4f} {ATE_ols - 10:>11.4f} {se_ols:>11.4f} {ATE_ols/se_ols:>11.4f}")
print(f"{'PSM (matched)':<20} {ATE_psm:>11.4f} {ATE_psm - 10:>11.4f} {se_psm:>11.4f} {t_stat:>11.4f}")
print(f"{'True ATE':<20} {10:>11.4f} {'-':>11} {'-':>11} {'-':>11}")

print("=" * 90)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Propensity score distributions
ax1 = axes[0, 0]
ax1.hist(data_support[data_support['T']==1]['P_hat'], bins=30, alpha=0.6, label='Treated', color='blue', density=True)
ax1.hist(data_support[data_support['T']==0]['P_hat'], bins=30, alpha=0.6, label='Control', color='red', density=True)
ax1.axvline(overlap_min, color='green', linestyle='--', linewidth=2, label='Common support')
ax1.axvline(overlap_max, color='green', linestyle='--', linewidth=2)
ax1.set_xlabel('Propensity Score', fontweight='bold')
ax1.set_ylabel('Density', fontweight='bold')
ax1.set_title('Propensity Score Distributions (Before Matching)', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Propensity score (after matching)
ax2 = axes[0, 1]
ax2.hist(matched_treated['P_hat'], bins=20, alpha=0.6, label='Treated (matched)', color='blue', density=True)
ax2.hist(matched_control['P_hat'], bins=20, alpha=0.6, label='Control (matched)', color='red', density=True)
ax2.set_xlabel('Propensity Score', fontweight='bold')
ax2.set_ylabel('Density', fontweight='bold')
ax2.set_title('Propensity Score Distributions (After Matching)', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. ATE comparison
ax3 = axes[1, 0]
estimates = ['Naive', 'OLS', 'PSM', 'True']
ates = [ATE_naive, ATE_ols, ATE_psm, 10]
colors_est = ['red', 'orange', 'green', 'black']
bars = ax3.bar(estimates, ates, color=colors_est, alpha=0.7)
ax3.axhline(y=10, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_ylabel('ATE Estimate', fontweight='bold')
ax3.set_title('Treatment Effect Estimates Comparison', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for i, (est, ate) in enumerate(zip(estimates, ates)):
    ax3.text(i, ate + 0.3, f'{ate:.2f}', ha='center', fontweight='bold')

# 4. Covariate balance (Love plot)
ax4 = axes[1, 1]
smd_before_list = []
smd_after_list = []
var_names_list = ['X1', 'X2', 'X3']

for var in var_names_list:
    X_val_before_t = data_support[data_support['T'] == 1][var]
    X_val_before_c = data_support[data_support['T'] == 0][var]
    X_val_after_t = matched_treated[var]
    X_val_after_c = matched_control[var]
    
    mean_diff_before = X_val_before_t.mean() - X_val_before_c.mean()
    mean_diff_after = X_val_after_t.mean() - X_val_after_c.mean()
    
    pooled_sd_before = np.sqrt((X_val_before_t.std()**2 + X_val_before_c.std()**2) / 2)
    pooled_sd_after = np.sqrt((X_val_after_t.std()**2 + X_val_after_c.std()**2) / 2)
    
    smd_before = mean_diff_before / pooled_sd_before
    smd_after = mean_diff_after / pooled_sd_after
    
    smd_before_list.append(smd_before)
    smd_after_list.append(smd_after)

y_pos = np.arange(len(var_names_list))
ax4.scatter(smd_before_list, y_pos, s=100, alpha=0.6, color='red', label='Before matching', marker='o')
ax4.scatter(smd_after_list, y_pos, s=100, alpha=0.6, color='green', label='After matching', marker='s')
ax4.axvline(x=0.1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Balance threshold (Â±0.1)')
ax4.axvline(x=-0.1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(var_names_list)
ax4.set_xlabel('Standardized Mean Difference', fontweight='bold')
ax4.set_title('Covariate Balance (Love Plot)', fontweight='bold')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('psm_matching.png', dpi=150)
plt.show()
