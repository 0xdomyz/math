import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import inv
from itertools import product

print("=" * 90)
print("CLUSTERED STANDARD ERRORS: IMPACT & IMPLEMENTATION")
print("=" * 90)

# Generate clustered panel data
np.random.seed(42)
N_firms = 30  # Clusters
T_years = 10  # Time periods
n_per_firm = T_years
N_total = N_firms * T_years

# Create identifiers
firm_id = np.repeat(np.arange(1, N_firms+1), T_years)
year_id = np.tile(np.arange(1, T_years+1), N_firms)
data = pd.DataFrame({'firm': firm_id, 'year': year_id})

# Regressors
data['X'] = np.random.normal(50, 10, N_total)

# Firm and year random effects (source of clustering)
alpha_firm = np.random.normal(100, 3, N_firms)
gamma_year = np.random.normal(0, 2, T_years)
data['alpha'] = data['firm'].map(lambda i: alpha_firm[i-1])
data['gamma'] = data['year'].map(lambda t: gamma_year[t-1])

# Idiosyncratic errors
data['eps'] = np.random.normal(0, 1, N_total)

# Generate outcome with structure
beta_true = 0.5
data['Y'] = 50 + beta_true * data['X'] + data['alpha'] + data['gamma'] + data['eps']

print(f"\nPanel Structure:")
print(f"  Firms (clusters): {N_firms}")
print(f"  Years (T): {T_years}")
print(f"  Total observations: {N_total}")
print(f"  Firm heterogeneity SD: 3.0")
print(f"  Year effect SD: 2.0")
print(f"  Idiosyncratic error SD: 1.0")

# Estimate intra-cluster correlation
firm_means_Y = data.groupby('firm')['Y'].mean()
firm_means_X = data.groupby('firm')['X'].mean()
within_Y = (data['Y'] - data.groupby('firm')['Y'].transform('mean'))**2
between_Y = (data.groupby('firm')['Y'].transform('mean') - data['Y'].mean())**2
var_between = between_Y.sum() / (N_firms - 1)
var_within = within_Y.sum() / (N_total - N_firms)
rho_firm = var_between / (var_between + var_within)

print(f"\nIntra-cluster correlation (by firm): Ï = {rho_firm:.4f}")
bias_factor = np.sqrt(1 + (T_years - 1) * rho_firm)
print(f"Naive SE bias factor: âˆš(1 + (T-1)Ï) = âˆš(1 + {T_years-1}Ã—{rho_firm:.4f}) = {bias_factor:.4f}")

# ========== Scenario 1: Naive OLS (ignores clustering) ==========
print("\n" + "=" * 90)
print("SCENARIO 1: NAIVE OLS (Ignores Clustering)")
print("=" * 90)

X_ols = np.column_stack([np.ones(N_total), data['X'].values])
y_ols = data['Y'].values

beta_ols = inv(X_ols.T @ X_ols) @ X_ols.T @ y_ols
resid_ols = y_ols - X_ols @ beta_ols
rss_ols = np.sum(resid_ols**2)
sigma2_ols = rss_ols / (N_total - 2)
var_beta_ols = sigma2_ols * inv(X_ols.T @ X_ols)
se_ols = np.sqrt(np.diag(var_beta_ols))
t_stat_ols = beta_ols / se_ols
p_value_ols = 2 * (1 - stats.t.cdf(np.abs(t_stat_ols), N_total - 2))

print(f"\nOLS Regression (Ignoring Clustering):")
print(f"  Î²Ì‚_OLS = {beta_ols[1]:.6f} (true Î² = {beta_true})")
print(f"  SE(Î²Ì‚_OLS) = {se_ols[1]:.6f} [UNDERESTIMATED]")
print(f"  t-stat = {t_stat_ols[1]:.6f}")
print(f"  p-value = {p_value_ols[1]:.6f}")
print(f"  95% CI: [{beta_ols[1] - 1.96*se_ols[1]:.6f}, {beta_ols[1] + 1.96*se_ols[1]:.6f}]")

# ========== Scenario 2: Cluster-Robust SE (One-way: Cluster by Firm) ==========
print("\n" + "=" * 90)
print("SCENARIO 2: CLUSTER-ROBUST SE (One-way: Cluster by Firm)")
print("=" * 90)

# Sandwich estimator by firm cluster
var_cluster = np.zeros((2, 2))

for firm in data['firm'].unique():
    firm_mask = data['firm'] == firm
    X_firm = X_ols[firm_mask, :]
    resid_firm = resid_ols[firm_mask]
    
    # Outer product of residuals for this firm
    var_cluster += X_firm.T @ (resid_firm.reshape(-1, 1) @ resid_firm.reshape(1, -1)) @ X_firm

# Cluster-robust covariance
bread = inv(X_ols.T @ X_ols)
var_cluster_robust = bread @ var_cluster @ bread

se_cluster_firm = np.sqrt(np.diag(var_cluster_robust))
t_stat_cluster_firm = beta_ols / se_cluster_firm
p_value_cluster_firm = 2 * (1 - stats.t.cdf(np.abs(t_stat_cluster_firm), N_firms - 2))

print(f"\nCluster-Robust SE (Clustered by Firm):")
print(f"  Î²Ì‚ = {beta_ols[1]:.6f} (same as OLS)")
print(f"  SE(Î²Ì‚) = {se_cluster_firm[1]:.6f} [CORRECTED]")
print(f"  Inflation factor: {se_cluster_firm[1] / se_ols[1]:.4f}x (theoretical: {bias_factor:.4f}x)")
print(f"  t-stat = {t_stat_cluster_firm[1]:.6f} (was {t_stat_ols[1]:.6f})")
print(f"  p-value = {p_value_cluster_firm[1]:.6f} (was {p_value_ols[1]:.6f})")
print(f"  95% CI: [{beta_ols[1] - 1.96*se_cluster_firm[1]:.6f}, {beta_ols[1] + 1.96*se_cluster_firm[1]:.6f}]")

# ========== Scenario 3: Two-Way Clustering (Firm + Year) ==========
print("\n" + "=" * 90)
print("SCENARIO 3: TWO-WAY CLUSTERING (Firm + Year)")
print("=" * 90)

# Variance from firm clustering
var_firm = np.zeros((2, 2))
for firm in data['firm'].unique():
    firm_mask = data['firm'] == firm
    X_firm = X_ols[firm_mask, :]
    resid_firm = resid_ols[firm_mask]
    var_firm += X_firm.T @ (resid_firm.reshape(-1, 1) @ resid_firm.reshape(1, -1)) @ X_firm

# Variance from year clustering
var_year = np.zeros((2, 2))
for year in data['year'].unique():
    year_mask = data['year'] == year
    X_year = X_ols[year_mask, :]
    resid_year = resid_ols[year_mask]
    var_year += X_year.T @ (resid_year.reshape(-1, 1) @ resid_year.reshape(1, -1)) @ X_year

# Variance from both (interaction)
var_both = np.zeros((2, 2))
for firm_year in product(data['firm'].unique(), data['year'].unique()):
    firm, year = firm_year
    both_mask = (data['firm'] == firm) & (data['year'] == year)
    if both_mask.sum() > 0:
        X_both = X_ols[both_mask, :]
        resid_both = resid_ols[both_mask]
        var_both += X_both.T @ (resid_both.reshape(-1, 1) @ resid_both.reshape(1, -1)) @ X_both

# Two-way variance (Cameron-Gelbach-Miller)
var_twoway = var_firm + var_year - var_both
var_twoway_robust = bread @ var_twoway @ bread

se_cluster_twoway = np.sqrt(np.diag(var_twoway_robust))
t_stat_twoway = beta_ols / se_cluster_twoway
p_value_twoway = 2 * (1 - stats.t.cdf(np.abs(t_stat_twoway), min(N_firms, T_years) - 2))

print(f"\nTwo-Way Clustering (Firm + Year):")
print(f"  Variance from firm clustering: {var_cluster_robust[1,1]:.6f}")
print(f"  Variance from year clustering: (computed separately)")
print(f"  Î²Ì‚ = {beta_ols[1]:.6f}")
print(f"  SE(Î²Ì‚) = {se_cluster_twoway[1]:.6f} [MOST CONSERVATIVE]")
print(f"  Inflation factor (vs OLS): {se_cluster_twoway[1] / se_ols[1]:.4f}x")
print(f"  t-stat = {t_stat_twoway[1]:.6f}")
print(f"  p-value = {p_value_twoway[1]:.6f}")
print(f"  95% CI: [{beta_ols[1] - 1.96*se_cluster_twoway[1]:.6f}, {beta_ols[1] + 1.96*se_cluster_twoway[1]:.6f}]")

# ========== Scenario 4: Fixed Effects + Cluster-Robust ==========
print("\n" + "=" * 90)
print("SCENARIO 4: FIXED EFFECTS + CLUSTER-ROBUST SE")
print("=" * 90)

# Within-transform
data['X_dm'] = data.groupby('firm')['X'].transform(lambda x: x - x.mean())
data['Y_dm'] = data.groupby('firm')['Y'].transform(lambda x: x - x.mean())

X_fe = data['X_dm'].values.reshape(-1, 1)
y_fe = data['Y_dm'].values

beta_fe = inv(X_fe.T @ X_fe) @ X_fe.T @ y_fe
resid_fe = y_fe - X_fe @ beta_fe

# Cluster-robust SE for FE (cluster by firm, but only one obs per firm per obs so simpler)
# Recompute considering within-cluster residuals
var_fe_cluster = 0
for firm in data['firm'].unique():
    firm_mask = data['firm'] == firm
    X_firm_fe = X_fe[firm_mask, :]
    resid_firm_fe = resid_fe[firm_mask]
    
    var_fe_cluster += X_firm_fe.T @ (resid_firm_fe.reshape(-1, 1) @ resid_firm_fe.reshape(1, -1)) @ X_firm_fe

bread_fe = inv(X_fe.T @ X_fe)
var_fe_robust = bread_fe * var_fe_cluster * bread_fe

se_fe_cluster = np.sqrt(var_fe_robust[0, 0])

# Crude alternative: estimate from within-firm residuals
rss_fe = np.sum(resid_fe**2)
sigma2_fe = rss_fe / (N_total - N_firms - 1)
se_fe_ols = np.sqrt(sigma2_fe / np.sum(X_fe**2))

print(f"\nFixed Effects (OLS SE):")
print(f"  Î²Ì‚_FE = {beta_fe[0]:.6f}")
print(f"  SE(Î²Ì‚_FE) = {se_fe_ols:.6f}")

print(f"\nFixed Effects (Cluster-Robust SE):")
print(f"  Î²Ì‚_FE = {beta_fe[0]:.6f} (same)")
print(f"  SE(Î²Ì‚_FE) = {se_fe_cluster:.6f}")
print(f"  Ratio SE_cluster / SE_ols = {se_fe_cluster / se_fe_ols:.4f}")

# ========== VISUALIZATION ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. SE Comparison
ax1 = axes[0, 0]
methods = ['Naive\nOLS', 'Cluster-Robust\n(Firm)', 'Two-Way\nCluster', 'FE +\nCluster-Robust']
ses = [se_ols[1], se_cluster_firm[1], se_cluster_twoway[1], se_fe_cluster]
colors = ['red', 'blue', 'green', 'purple']

bars = ax1.bar(methods, ses, color=colors, alpha=0.7)
ax1.set_ylabel('Standard Error of Î²Ì‚', fontweight='bold', fontsize=11)
ax1.set_title('Comparison of Standard Errors', fontweight='bold', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

for bar, se in zip(bars, ses):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{se:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 2. Confidence Intervals
ax2 = axes[0, 1]
ci_width_ols = 1.96 * se_ols[1]
ci_width_cluster = 1.96 * se_cluster_firm[1]
ci_width_twoway = 1.96 * se_cluster_twoway[1]

y_pos = [0, 1, 2]
ax2.errorbar([beta_ols[1]]*3, y_pos, 
             xerr=[ci_width_ols, ci_width_cluster, ci_width_twoway],
             fmt='o', markersize=8, linewidth=2, capsize=5,
             color=['red', 'blue', 'green'], 
             label=['Naive', 'Firm Cluster', 'Two-Way'])
ax2.axvline(beta_true, color='black', linestyle='--', linewidth=2, label=f'True Î² = {beta_true}')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(['Naive OLS', 'Cluster-Robust\n(Firm)', 'Two-Way\nCluster'])
ax2.set_xlabel('Î²Ì‚ with 95% CI', fontweight='bold', fontsize=11)
ax2.set_title('Treatment Effect Estimates & Confidence Intervals', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# 3. t-statistics and p-values
ax3 = axes[1, 0]
t_stats = [t_stat_ols[1], t_stat_cluster_firm[1], t_stat_twoway[1]]
p_vals = [p_value_ols[1], p_value_cluster_firm[1], p_value_twoway[1]]

methods_short = ['Naive', 'Firm-Cluster', 'Two-Way']
colors_t = ['red', 'blue', 'green']
bars_t = ax3.bar(methods_short, t_stats, color=colors_t, alpha=0.7)
ax3.axhline(y=1.96, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='tâ‚€.â‚€â‚‚â‚…')
ax3.set_ylabel('t-statistic', fontweight='bold', fontsize=11)
ax3.set_title('Hypothesis Tests (Hâ‚€: Î² = 0)', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

for i, (bar, t, p) in enumerate(zip(bars_t, t_stats, p_vals)):
    height = bar.get_height()
    sig = '*' * (3 - int(np.floor(-np.log10(p))))  # 1-3 asterisks for *, **, ***
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{t:.2f}\np={p:.4f}\n{sig}', ha='center', va='bottom', fontweight='bold', fontsize=8)

# 4. SE Ratio (to show inflation)
ax4 = axes[1, 1]
ratios = [1.0, se_cluster_firm[1]/se_ols[1], se_cluster_twoway[1]/se_ols[1]]
labels_ratio = ['Naive\n(baseline)', 'Firm-Cluster\nvs Naive', 'Two-Way\nvs Naive']

bars_ratio = ax4.bar(labels_ratio, ratios, color=['gray', 'blue', 'green'], alpha=0.7)
ax4.axhline(y=bias_factor, color='red', linestyle='--', linewidth=2, label=f'Theoretical factor: {bias_factor:.3f}')
ax4.set_ylabel('SE Inflation Factor', fontweight='bold', fontsize=11)
ax4.set_title(f'Standard Error Adjustment (Clustering Impact)\nFirm Ï={rho_firm:.4f}, T={T_years}', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

for bar, ratio in zip(bars_ratio, ratios):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('clustered_se_analysis.png', dpi=150)
plt.show()

print("\n" + "=" * 90)
print("SUMMARY TABLE: INFERENCE COMPARISON")
print("=" * 90)
print(f"{'Method':<25} {'SE(Î²Ì‚)':<12} {'Inflation':<12} {'t-stat':<12} {'p-value':<12} {'Sig. at 5%':<12}")
print("-" * 85)
print(f"{'Naive OLS':<25} {se_ols[1]:>11.6f} {1.0:>11.2f}x {t_stat_ols[1]:>11.4f} {p_value_ols[1]:>11.4f} {'Yes':<12}")
print(f"{'Cluster (Firm)':<25} {se_cluster_firm[1]:>11.6f} {se_cluster_firm[1]/se_ols[1]:>11.2f}x {t_stat_cluster_firm[1]:>11.4f} {p_value_cluster_firm[1]:>11.4f} {'Yes' if p_value_cluster_firm[1]<0.05 else 'No':<12}")
print(f"{'Two-Way Cluster':<25} {se_cluster_twoway[1]:>11.6f} {se_cluster_twoway[1]/se_ols[1]:>11.2f}x {t_stat_twoway[1]:>11.4f} {p_value_twoway[1]:>11.4f} {'Yes' if p_value_twoway[1]<0.05 else 'No':<12}")
print(f"{'FE + Cluster-Robust':<25} {se_fe_cluster:>11.6f} {se_fe_cluster/se_fe_ols:>11.2f}x {beta_fe[0]/se_fe_cluster:>11.4f} {'-':<11} {'Yes' if beta_fe[0]/se_fe_cluster > 1.96 else 'No':<12}")
print("=" * 90)
