import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression

print("=" * 90)
print("RANDOMIZED CONTROLLED TRIALS: DESIGN, POWER, AND HETEROGENEOUS EFFECTS")
print("=" * 90)

# ==================== SCENARIO 1: POWER ANALYSIS ====================
print("\n" + "=" * 90)
print("SCENARIO 1: POWER ANALYSIS & SAMPLE SIZE CALCULATION")
print("=" * 90)

def power_ttest(n, delta, sigma=1, alpha=0.05):
    """
    Calculate power for two-sample t-test.
    Power = Pr(reject H0 | true effect = delta)
    """
    se = sigma * np.sqrt(2 / n)
    t_crit = stats.t.ppf(1 - alpha/2, df=2*n-2)
    non_centrality = (delta / se) / np.sqrt(2/n)
    
    # Non-central t-distribution
    power = 1 - stats.nct.cdf(t_crit, df=2*n-2, nc=non_centrality/np.sqrt(2/n))
    return power

# Calculate power across sample sizes
ns = np.arange(50, 501, 50)
delta_values = [0.1, 0.2, 0.3]
colors_power = ['red', 'orange', 'green']

fig, ax = plt.subplots(figsize=(10, 6))

for delta, color in zip(delta_values, colors_power):
    powers = [power_ttest(n, delta) for n in ns]
    ax.plot(ns*2, powers, 'o-', linewidth=2, markersize=6, label=f'Î´ = {delta}Ïƒ', color=color)

ax.axhline(y=0.80, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Power = 0.80')
ax.axhline(y=0.90, color='black', linestyle=':', linewidth=1.5, alpha=0.5, label='Power = 0.90')
ax.set_xlabel('Total Sample Size (n_T + n_C)', fontsize=11, fontweight='bold')
ax.set_ylabel('Power (1 - Î²)', fontsize=11, fontweight='bold')
ax.set_title('RCT Power Curve: Sample Size Required for Significance', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('rct_power_curve.png', dpi=150)
plt.close()

print("\nSample Size Recommendations (Î±=0.05, Î²=0.20, two-tailed):")
print(f"{'Effect Size':<15} {'Minimum n per arm':<20} {'Total n':<15}")
print("-" * 50)

for delta in [0.1, 0.2, 0.3, 0.4]:
    # Solve for n such that power = 0.80
    n_func = lambda n: power_ttest(n, delta) - 0.80
    n_opt = int(brentq(n_func, 50, 2000))
    print(f"Î´ = {delta}Ïƒ         {n_opt:<20} {2*n_opt:<15}")

print(f"\nâœ“ Takeaway: Larger effect Î´ requires smaller sample. 0.2Ïƒ effect needs ~n=400 total.")

# ==================== SCENARIO 2: SIMPLE RCT (PERFECT COMPLIANCE) ====================
print("\n" + "=" * 90)
print("SCENARIO 2: PERFECT COMPLIANCE RCT (No Noncompliance)")
print("=" * 90)

np.random.seed(42)
n_total = 1000
n_per_arm = n_total // 2

# True parameters
true_ate = 0.5
sigma = 1.0

# Generate potential outcomes
X_confound = np.random.normal(0, 1, n_total)  # Baseline characteristic

Y0 = 50 + 0.3*X_confound + np.random.normal(0, sigma, n_total)  # Control outcome
Y1 = Y0 + true_ate  # Treatment adds constant effect

# Randomization
T_randomized = np.zeros(n_total, dtype=int)
T_randomized[np.random.choice(n_total, n_per_arm, replace=False)] = 1

# Observed outcome
Y_observed = T_randomized * Y1 + (1 - T_randomized) * Y0

# ITT Estimate (perfect compliance)
Y_treated = Y_observed[T_randomized == 1]
Y_control = Y_observed[T_randomized == 0]
ate_hat = Y_treated.mean() - Y_control.mean()
se_ate = np.sqrt(Y_treated.var()/len(Y_treated) + Y_control.var()/len(Y_control))
t_stat_ate = ate_hat / se_ate
p_value_ate = 2 * (1 - stats.t.cdf(abs(t_stat_ate), df=n_total-2))

print(f"\nRandom Assignment (Perfect Compliance):")
print(f"  Treated: n={len(Y_treated)}")
print(f"  Control: n={len(Y_control)}")
print(f"\nIntent-to-Treat Estimate:")
print(f"  ATE (True): {true_ate:.4f}")
print(f"  ATE (Estimated): {ate_hat:.4f}")
print(f"  SE(ATE): {se_ate:.4f}")
print(f"  t-statistic: {t_stat_ate:.4f}")
print(f"  p-value: {p_value_ate:.6f}")
print(f"  95% CI: [{ate_hat - 1.96*se_ate:.4f}, {ate_hat + 1.96*se_ate:.4f}]")

if p_value_ate < 0.05:
    print("  âœ“ Reject Hâ‚€: Treatment effect is significant")
else:
    print("  âœ— Fail to reject Hâ‚€: Insufficient evidence of effect")

# ==================== SCENARIO 3: NONCOMPLIANCE & LATE ====================
print("\n" + "=" * 90)
print("SCENARIO 3: NONCOMPLIANCE & LOCAL AVERAGE TREATMENT EFFECT (LATE)")
print("=" * 90)

# Noncompliance mechanism
compliance_prob_treated = 0.70  # 70% of assigned-to-treat actually comply
compliance_prob_control = 0.05  # 5% of assigned-to-control take treatment anyway

T_actual = np.zeros(n_total, dtype=int)

# Treatment arm: some noncompliance
treated_idx = np.where(T_randomized == 1)[0]
compliers = np.random.binomial(1, compliance_prob_treated, len(treated_idx))
T_actual[treated_idx[compliers == 1]] = 1

# Control arm: some contamination
control_idx = np.where(T_randomized == 0)[0]
contaminated = np.random.binomial(1, compliance_prob_control, len(control_idx))
T_actual[control_idx[contaminated == 1]] = 1

# Observed outcome (with actual treatment)
Y_observed_nc = T_actual * Y1 + (1 - T_actual) * Y0

print(f"\nNoncompliance Pattern:")
print(f"  Assigned to treatment: {(T_randomized==1).sum()}")
print(f"    Actually treated: {(T_actual[(T_randomized==1)])==1).sum()} "
      f"({100*(T_actual[(T_randomized==1)])==1).sum()/(T_randomized==1).sum():.1f}%)")
print(f"  Assigned to control: {(T_randomized==0).sum()}")
print(f"    Actually treated: {(T_actual[(T_randomized==0)])==1).sum()} "
      f"({100*(T_actual[(T_randomized==0)])==1).sum()/(T_randomized==0).sum():.1f}%)")

# ITT with noncompliance
Y_treated_itt = Y_observed_nc[T_randomized == 1]
Y_control_itt = Y_observed_nc[T_randomized == 0]
itt_hat = Y_treated_itt.mean() - Y_control_itt.mean()
se_itt = np.sqrt(Y_treated_itt.var()/len(Y_treated_itt) + Y_control_itt.var()/len(Y_control_itt))

print(f"\nIntent-to-Treat (ITT):")
print(f"  ITT = {itt_hat:.4f} (TRUE ATE: {true_ate:.4f})")
print(f"  SE(ITT): {se_itt:.4f}")

# Per-protocol (BIASED - endogenous selection)
Y_pp_treated = Y_observed_nc[T_actual == 1]
Y_pp_control = Y_observed_nc[T_actual == 0]
pp_hat = Y_pp_treated.mean() - Y_pp_control.mean()
se_pp = np.sqrt(Y_pp_treated.var()/len(Y_pp_treated) + Y_pp_control.var()/len(Y_pp_control))

print(f"\nPer-Protocol (Biased - Endogenous Selection):")
print(f"  PP = {pp_hat:.4f}")
print(f"  SE(PP): {se_pp:.4f}")
print(f"  Bias: {pp_hat - true_ate:.4f} (selection bias reintroduced)")

# LATE via 2SLS
# First stage: T_actual ~ T_randomized
X_2sls = np.column_stack([np.ones(n_total), T_randomized])
beta_first = np.linalg.inv(X_2sls.T @ X_2sls) @ X_2sls.T @ T_actual
T_actual_fitted = X_2sls @ beta_first
compliance_rate = beta_first[1]

# Second stage: Y ~ T_actual_fitted
X_second = np.column_stack([np.ones(n_total), T_actual_fitted])
beta_second = np.linalg.inv(X_second.T @ X_second) @ X_second.T @ Y_observed_nc
late_hat = beta_second[1]

print(f"\nLocal Average Treatment Effect (LATE) via 2SLS:")
print(f"  First-stage compliance rate: {compliance_rate:.4f}")
print(f"  LATE = ITT / compliance = {itt_hat:.4f} / {compliance_rate:.4f} = {late_hat:.4f}")
print(f"  (Effect on compliers)")

# ==================== SCENARIO 4: HETEROGENEOUS TREATMENT EFFECTS ====================
print("\n" + "=" * 90)
print("SCENARIO 4: HETEROGENEOUS TREATMENT EFFECTS (HTE)")
print("=" * 90)

# Heterogeneous effect: treatment more effective for high-baseline units
heterogeneous_ate = true_ate * (1 + 0.5 * (X_confound - X_confound.mean()) / X_confound.std())
Y1_het = Y0 + heterogeneous_ate

Y_observed_het = T_randomized * Y1_het + (1 - T_randomized) * Y0

# Estimate average effect
Y_treated_het = Y_observed_het[T_randomized == 1]
Y_control_het = Y_observed_het[T_randomized == 0]
X_treated = X_confound[T_randomized == 1]
X_control = X_confound[T_randomized == 0]

ate_avg_het = Y_treated_het.mean() - Y_control_het.mean()

# Regression: Y ~ T + X + T*X
X_reg = np.column_stack([
    np.ones(n_total),
    T_randomized,
    (X_confound - X_confound.mean()),  # Center X
    T_randomized * (X_confound - X_confound.mean())
])
beta_reg = np.linalg.inv(X_reg.T @ X_reg) @ X_reg.T @ Y_observed_het

ate_base = beta_reg[1]  # Effect at mean X
het_slope = beta_reg[3]  # Heterogeneity slope

print(f"\nHeterogeneous Treatment Effects:")
print(f"\nRegressions: Y ~ T + X + TÃ—X")
print(f"  Coefficient T: {ate_base:.4f} (ATE at mean X)")
print(f"  Coefficient TÃ—X: {het_slope:.4f} (Heterogeneity)")
print(f"  Interpretation: For each SD â†‘ in X, treatment effect increases by {het_slope:.4f}")

# Conditional effects across X quantiles
quantiles_x = [0.25, 0.50, 0.75]
print(f"\nConditional Average Treatment Effect (CATE) by X Quantile:")
print(f"{'Quantile':<12} {'X value':<12} {'Predicted CATE':<18}")
print("-" * 42)

for q in quantiles_x:
    x_q = np.percentile(X_confound, q*100)
    cate_q = ate_base + het_slope * (x_q - X_confound.mean())
    print(f"{q*100:.0f}%{'':<8} {x_q:>11.4f} {cate_q:>17.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ATE comparison across scenarios
ax1 = axes[0, 0]
scenarios = ['Perfect\nCompliance', 'ITT\n(with NC)', 'Per-Protocol\n(biased)', 'LATE\n(compliers)']
estimates = [ate_hat, itt_hat, pp_hat, late_hat]
colors_est = ['green', 'blue', 'red', 'orange']
bars = ax1.bar(scenarios, estimates, color=colors_est, alpha=0.7)
ax1.axhline(y=true_ate, color='black', linestyle='--', linewidth=2, label=f'True ATE = {true_ate}')
ax1.set_ylabel('Effect Estimate', fontsize=11, fontweight='bold')
ax1.set_title('ATE Estimation: Noncompliance Impact', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for bar, est in zip(bars, estimates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{est:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Heterogeneous treatment effects (scatter + regression line)
ax2 = axes[0, 1]
X_sorted_idx = np.argsort(X_confound)
X_sorted = X_confound[X_sorted_idx]

# Plot individual effects (approximate via residuals)
residuals_T1 = Y1_het[T_randomized == 1] - Y1_het[T_randomized == 1].mean()
residuals_T0 = Y0[T_randomized == 0] - Y0[T_randomized == 0].mean()

# Bin X and compute average effects per bin
bins = np.quantile(X_confound, [0, 0.25, 0.5, 0.75, 1.0])
for i in range(len(bins) - 1):
    in_bin = (X_confound >= bins[i]) & (X_confound < bins[i+1])
    T_in_bin = T_randomized[in_bin]
    Y_in_bin = Y_observed_het[in_bin]
    
    effect_bin = Y_in_bin[T_in_bin == 1].mean() - Y_in_bin[T_in_bin == 0].mean()
    x_mid = (bins[i] + bins[i+1]) / 2
    ax2.scatter(x_mid, effect_bin, s=100, alpha=0.7, color='blue')

# Overlay regression line
X_line = np.linspace(X_confound.min(), X_confound.max(), 100)
cate_line = ate_base + het_slope * (X_line - X_confound.mean())
ax2.plot(X_line, cate_line, 'r-', linewidth=2.5, label='Regression: T + TÃ—X')
ax2.axhline(y=ate_avg_het, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Average ATE = {ate_avg_het:.3f}')
ax2.set_xlabel('Baseline Characteristic (X)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Treatment Effect', fontsize=11, fontweight='bold')
ax2.set_title('Heterogeneous Treatment Effects by Baseline X', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Distribution of randomized assignment
ax3 = axes[1, 0]
ax3.hist(Y_control_itt, bins=30, alpha=0.6, label='Control (T=0)', color='red', density=True)
ax3.hist(Y_treated_itt, bins=30, alpha=0.6, label='Treated (T=1)', color='blue', density=True)
ax3.axvline(Y_control_itt.mean(), color='red', linestyle='--', linewidth=2)
ax3.axvline(Y_treated_itt.mean(), color='blue', linestyle='--', linewidth=2)
ax3.set_xlabel('Outcome (Y)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
ax3.set_title(f'Outcome Distributions (ITT Estimate: {itt_hat:.4f})', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Compliance breakdown
ax4 = axes[1, 1]
compliance_breakdown = [
    (T_randomized == 1).sum() - (T_actual[(T_randomized == 1)] == 1).sum(),  # Non-compliers (assigned T, didn't take)
    (T_actual[(T_randomized == 1)] == 1).sum(),  # Compliers (assigned T, took)
    (T_randomized == 0).sum() - (T_actual[(T_randomized == 0)] == 1).sum(),  # True controls
    (T_actual[(T_randomized == 0)] == 1).sum()  # Contaminated controls
]
labels_comp = ['Non-compliers\n(Assigned T, didn\'t take)', 'Compliers\n(Assigned T, took)', 
               'True Controls\n(Assigned C, didn\'t take)', 'Contaminated\n(Assigned C, took)']
colors_comp = ['lightcoral', 'green', 'lightblue', 'orange']

wedges, texts, autotexts = ax4.pie(compliance_breakdown, labels=labels_comp, autopct='%1.1f%%',
                                     colors=colors_comp, startangle=90)
ax4.set_title('Compliance Breakdown', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('rct_analysis.png', dpi=150)
plt.show()

print("\n" + "=" * 90)
print("SUMMARY TABLE: ESTIMATOR COMPARISON")
print("=" * 90)
print(f"{'Estimator':<25} {'Estimate':<12} {'True ATE':<12} {'Bias':<12} {'Assumption':<30}")
print("-" * 91)
print(f"{'Perfect Compliance':<25} {ate_hat:>11.4f} {true_ate:>11.4f} {ate_hat-true_ate:>11.4f} {'Randomization':<30}")
print(f"{'ITT (with NC)':<25} {itt_hat:>11.4f} {true_ate:>11.4f} {itt_hat-true_ate:>11.4f} {'Randomization':<30}")
print(f"{'Per-Protocol (biased)':<25} {pp_hat:>11.4f} {true_ate:>11.4f} {pp_hat-true_ate:>11.4f} {'Exogeneity (WRONG!)':<30}")
print(f"{'LATE (2SLS)':<25} {late_hat:>11.4f} {true_ate:>11.4f} {late_hat-true_ate:>11.4f} {'Monotonicity + Exclusion':<30}")
print("=" * 90)
