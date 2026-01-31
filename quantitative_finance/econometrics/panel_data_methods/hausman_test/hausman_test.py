import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import inv

print("=" * 90)
print("HAUSMAN TEST: FIXED VS. RANDOM EFFECTS SPECIFICATION TEST")
print("=" * 90)

# Generate panel data with varying correlation between alpha_i and X
np.random.seed(42)
N = 100  # Units
T = 8    # Time periods

scenarios = {
    "No Correlation (Exogenous)": {"corr": 0.0},
    "Weak Correlation": {"corr": 0.3},
    "Strong Correlation (Endogenous)": {"corr": 0.8}
}

results_hausman = []

for scenario_name, params in scenarios.items():
    corr_strength = params["corr"]
    
    print(f"\n{'='*90}")
    print(f"SCENARIO: {scenario_name} (Corr(Î±_i, X) = {corr_strength})")
    print(f"{'='*90}")
    
    # Generate correlated alpha_i and X
    X_mean = np.random.normal(50, 10, N)  # Between-unit X variation
    alpha_base = np.random.normal(100, 5, N)
    
    # Create correlation: alpha_i depends on XÌ„_i
    alpha_i = 100 + corr_strength * (X_mean - 50) + np.random.normal(0, 2, N)
    
    # Generate panel data
    data_list = []
    for i in range(N):
        for t in range(T):
            X_it = X_mean[i] + np.random.normal(0, 5)  # X varies within and between
            Y_it = alpha_i[i] + 0.5 * X_it + np.random.normal(0, 1)
            data_list.append({'unit': i, 'time': t, 'X': X_it, 'Y': Y_it, 'alpha': alpha_i[i]})
    
    data = pd.DataFrame(data_list)
    n_total = len(data)
    
    print(f"Sample: N={N} units, T={T} periods, Total n={n_total}")
    print(f"Correlation between Î±_i and XÌ„_i: {np.corrcoef(alpha_i, X_mean)[0, 1]:.4f}")
    
    # ========== FE Estimation ==========
    # Within-transform
    data['X_dm'] = data.groupby('unit')['X'].transform(lambda x: x - x.mean())
    data['Y_dm'] = data.groupby('unit')['Y'].transform(lambda x: x - x.mean())
    
    X_fe = data['X_dm'].values.reshape(-1, 1)
    y_fe = data['Y_dm'].values
    
    # FE regression (no intercept after demeaning)
    beta_fe = inv(X_fe.T @ X_fe) @ X_fe.T @ y_fe
    resid_fe = y_fe - X_fe @ beta_fe
    rss_fe = np.sum(resid_fe**2)
    dof_fe = n_total - N - 1
    sigma2_fe = rss_fe / dof_fe
    var_beta_fe = sigma2_fe * inv(X_fe.T @ X_fe)
    se_fe = np.sqrt(var_beta_fe[0, 0])
    
    print(f"\nFixed Effects (Within-Transformation):")
    print(f"  Î²Ì‚_FE = {beta_fe[0]:.6f} (true Î² = 0.5)")
    print(f"  SE(Î²Ì‚_FE) = {se_fe:.6f}")
    print(f"  RSS = {rss_fe:.4f}, DOF = {dof_fe}")
    
    # ========== RE Estimation (GLS) ==========
    # Estimate variance components
    X_full = np.column_stack([np.ones(n_total), data['X'].values])
    y_full = data['Y'].values
    beta_ols = inv(X_full.T @ X_full) @ X_full.T @ y_full
    resid_ols = y_full - X_full @ beta_ols
    sigma2_eps_ols = np.sum(resid_ols**2) / (n_total - 2)
    
    # Between variance
    unit_means = data.groupby('unit')[['Y', 'X']].mean()
    y_um = unit_means['Y'].values
    x_um = unit_means['X'].values
    beta_between = np.cov(x_um, y_um)[0, 1] / np.var(x_um) if np.var(x_um) > 0 else 0
    resid_between = y_um - beta_between * x_um
    var_between = np.var(resid_between)
    
    # Estimate sigma2_u (random effect variance)
    sigma2_u = max(0, (var_between - sigma2_eps_ols / T))
    
    # GLS weight
    theta = 1 - np.sqrt(sigma2_eps_ols / (sigma2_eps_ols + T * sigma2_u))
    
    # Apply GLS transformation
    data['Y_gls'] = data['Y'] - theta * data.groupby('unit')['Y'].transform('mean')
    data['X_gls'] = data['X'] - theta * data.groupby('unit')['X'].transform('mean')
    data['const_gls'] = 1 - theta
    
    X_gls = np.column_stack([data['const_gls'].values, data['X_gls'].values])
    y_gls = data['Y_gls'].values
    
    beta_gls = inv(X_gls.T @ X_gls) @ X_gls.T @ y_gls
    resid_gls = y_gls - X_gls @ beta_gls
    rss_gls = np.sum(resid_gls**2)
    dof_gls = n_total - 2
    sigma2_gls = rss_gls / dof_gls
    var_beta_gls = sigma2_gls * inv(X_gls.T @ X_gls)
    se_gls = np.sqrt(var_beta_gls[1, 1])
    
    print(f"\nRandom Effects (GLS):")
    print(f"  Estimated ÏƒÂ²_u = {sigma2_u:.6f}")
    print(f"  Estimated ÏƒÂ²_Îµ = {sigma2_eps_ols:.6f}")
    print(f"  GLS weight Î¸ = {theta:.6f}")
    print(f"  Î²Ì‚_RE = {beta_gls[1]:.6f}")
    print(f"  SE(Î²Ì‚_RE) = {se_gls:.6f}")
    
    # ========== HAUSMAN TEST ==========
    beta_diff = beta_fe[0] - beta_gls[1]
    
    # Variance of difference
    var_diff_approx = var_beta_fe[0, 0] - var_beta_gls[1, 1]
    
    if var_diff_approx > 0:
        H_stat = (beta_diff**2) / var_diff_approx
    else:
        # If variance condition fails, use conservative approximation
        var_diff_approx = var_beta_fe[0, 0] + var_beta_gls[1, 1]
        H_stat = (beta_diff**2) / var_diff_approx
    
    df_test = 1  # One coefficient being tested
    p_value = 1 - stats.chi2.cdf(H_stat, df_test)
    chi2_crit = stats.chi2.ppf(0.95, df_test)
    
    print(f"\nHausman Test:")
    print(f"  Î²Ì‚_FE - Î²Ì‚_RE = {beta_diff:.6f}")
    print(f"  Hausman H = {H_stat:.6f}")
    print(f"  Ï‡Â²â‚€.â‚€â‚…(1) = {chi2_crit:.6f}")
    print(f"  p-value = {p_value:.6f}")
    
    if p_value < 0.05:
        decision = "REJECT Hâ‚€ â†’ Use FE (correlation likely)"
    else:
        decision = "FAIL TO REJECT Hâ‚€ â†’ Use RE (exogeneity tenable)"
    
    print(f"  Decision: {decision}")
    
    results_hausman.append({
        'scenario': scenario_name,
        'corr_true': np.corrcoef(alpha_i, X_mean)[0, 1],
        'beta_fe': beta_fe[0],
        'se_fe': se_fe,
        'beta_re': beta_gls[1],
        'se_re': se_gls,
        'H_stat': H_stat,
        'p_value': p_value,
        'chi2_crit': chi2_crit,
        'decision': 'FE' if p_value < 0.05 else 'RE'
    })

# ========== POWER ANALYSIS ==========
print("\n" + "=" * 90)
print("POWER ANALYSIS: How often does Hausman test detect correlation?")
print("=" * 90)

n_sims = 100
corr_range = np.linspace(0, 0.9, 7)
power_by_corr = []

for corr in corr_range:
    rejections = 0
    
    for sim in range(n_sims):
        # Generate scenario with specified correlation
        X_mean_ps = np.random.normal(50, 10, N)
        alpha_ps = 100 + corr * (X_mean_ps - 50) + np.random.normal(0, 2, N)
        
        # Panel data
        data_ps_list = []
        for i in range(N):
            for t in range(T):
                X_it = X_mean_ps[i] + np.random.normal(0, 5)
                Y_it = alpha_ps[i] + 0.5 * X_it + np.random.normal(0, 1)
                data_ps_list.append({'unit': i, 'time': t, 'X': X_it, 'Y': Y_it})
        
        data_ps = pd.DataFrame(data_ps_list)
        
        # Quick FE
        data_ps['X_dm'] = data_ps.groupby('unit')['X'].transform(lambda x: x - x.mean())
        data_ps['Y_dm'] = data_ps.groupby('unit')['Y'].transform(lambda x: x - x.mean())
        X_fe_ps = data_ps['X_dm'].values.reshape(-1, 1)
        y_fe_ps = data_ps['Y_dm'].values
        beta_fe_ps = inv(X_fe_ps.T @ X_fe_ps) @ X_fe_ps.T @ y_fe_ps
        resid_fe_ps = y_fe_ps - X_fe_ps @ beta_fe_ps
        sigma2_fe_ps = np.sum(resid_fe_ps**2) / (n_total - N - 1)
        var_fe_ps = sigma2_fe_ps * inv(X_fe_ps.T @ X_fe_ps)
        
        # Quick RE (simplified)
        X_full_ps = np.column_stack([np.ones(n_total), data_ps['X'].values])
        y_full_ps = data_ps['Y'].values
        beta_ols_ps = inv(X_full_ps.T @ X_full_ps) @ X_full_ps.T @ y_full_ps
        resid_ols_ps = y_full_ps - X_full_ps @ beta_ols_ps
        sigma2_eps_ps = np.sum(resid_ols_ps**2) / (n_total - 2)
        
        unit_means_ps = data_ps.groupby('unit')[['Y', 'X']].mean()
        var_between_ps = np.var(unit_means_ps['Y'].values)
        sigma2_u_ps = max(0, (var_between_ps - sigma2_eps_ps / T))
        theta_ps = 1 - np.sqrt(sigma2_eps_ps / (sigma2_eps_ps + T * sigma2_u_ps))
        
        data_ps['Y_gls'] = data_ps['Y'] - theta_ps * data_ps.groupby('unit')['Y'].transform('mean')
        data_ps['X_gls'] = data_ps['X'] - theta_ps * data_ps.groupby('unit')['X'].transform('mean')
        X_gls_ps = np.column_stack([1-theta_ps*np.ones(n_total), data_ps['X_gls'].values])
        y_gls_ps = data_ps['Y_gls'].values
        
        beta_gls_ps = inv(X_gls_ps.T @ X_gls_ps) @ X_gls_ps.T @ y_gls_ps
        sigma2_gls_ps = np.sum((y_gls_ps - X_gls_ps @ beta_gls_ps)**2) / (n_total - 2)
        var_gls_ps = sigma2_gls_ps * inv(X_gls_ps.T @ X_gls_ps)
        
        # Hausman
        diff_ps = beta_fe_ps[0] - beta_gls_ps[1]
        var_diff_ps = var_fe_ps[0, 0] + var_gls_ps[1, 1]
        H_ps = (diff_ps**2) / max(var_diff_ps, 1e-10)
        p_ps = 1 - stats.chi2.cdf(H_ps, 1)
        
        if p_ps < 0.05:
            rejections += 1
    
    power = rejections / n_sims
    power_by_corr.append(power)
    print(f"Correlation = {corr:.2f}: Power = {power:.1%} (reject Hâ‚€ in {rejections}/{n_sims} sims)")

# ========== VISUALIZATION ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Coefficient comparison across scenarios
ax1 = axes[0, 0]
scenario_names = [r['scenario'].split('(')[0].strip() for r in results_hausman]
beta_fes = [r['beta_fe'] for r in results_hausman]
beta_res = [r['beta_re'] for r in results_hausman]
x_pos = np.arange(len(scenario_names))
width = 0.35

ax1.bar(x_pos - width/2, beta_fes, width, label='FE', color='green', alpha=0.7)
ax1.bar(x_pos + width/2, beta_res, width, label='RE', color='blue', alpha=0.7)
ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='True Î²=0.5')
ax1.set_ylabel('Î²Ì‚ Estimate', fontweight='bold')
ax1.set_title('FE vs RE Coefficient Estimates', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(scenario_names, rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Hausman test p-values
ax2 = axes[0, 1]
p_vals = [r['p_value'] for r in results_hausman]
colors = ['red' if p < 0.05 else 'green' for p in p_vals]
bars = ax2.bar(scenario_names, p_vals, color=colors, alpha=0.7)
ax2.axhline(y=0.05, color='black', linestyle='--', linewidth=2, label='Î±=0.05')
ax2.set_ylabel('p-value', fontweight='bold')
ax2.set_title('Hausman Test p-values', fontweight='bold')
ax2.set_xticklabels(scenario_names, rotation=15, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, p in zip(bars, p_vals):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{p:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. Standard error comparison
ax3 = axes[1, 0]
se_fes = [r['se_fe'] for r in results_hausman]
se_res = [r['se_re'] for r in results_hausman]
ax3.bar(x_pos - width/2, se_fes, width, label='SE(FE)', color='green', alpha=0.7)
ax3.bar(x_pos + width/2, se_res, width, label='SE(RE)', color='blue', alpha=0.7)
ax3.set_ylabel('Standard Error', fontweight='bold')
ax3.set_title('Precision: FE vs RE (Lower SE is Better)', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(scenario_names, rotation=15, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Power curve
ax4 = axes[1, 1]
ax4.plot(corr_range, power_by_corr, 'o-', linewidth=2, markersize=8, color='purple')
ax4.axhline(y=0.80, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Power=0.80')
ax4.fill_between(corr_range, 0, power_by_corr, alpha=0.2, color='purple')
ax4.set_xlabel('True Correlation Corr(Î±_i, X)', fontweight='bold')
ax4.set_ylabel('Power (Prob. of Rejection)', fontweight='bold')
ax4.set_title(f'Hausman Test Power: Detecting Correlation\n(N={N}, T={T}, {n_sims} simulations per point)', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hausman_test_analysis.png', dpi=150)
plt.show()

print("\n" + "=" * 90)
print("SUMMARY TABLE: HAUSMAN TEST RESULTS")
print("=" * 90)
print(f"{'Scenario':<30} {'Corr':<8} {'Î²_FE':<10} {'Î²_RE':<10} {'H-stat':<10} {'p-val':<10} {'Decision':<10}")
print("-" * 88)
for r in results_hausman:
    print(f"{r['scenario']:<30} {r['corr_true']:>7.3f} {r['beta_fe']:>9.4f} {r['beta_re']:>9.4f} "
          f"{r['H_stat']:>9.4f} {r['p_value']:>9.4f} {r['decision']:>9}")
