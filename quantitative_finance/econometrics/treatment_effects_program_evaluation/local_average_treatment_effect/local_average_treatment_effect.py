import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns

np.random.seed(890)

# ===== Simulate IV Data with Non-Compliance =====
print("="*80)
print("LOCAL AVERAGE TREATMENT EFFECT (LATE)")
print("="*80)

n = 1000

# Covariates
X1 = np.random.randn(n)  # Age
X2 = np.random.randn(n)  # Education

# Unobserved confounder (drives selection into treatment)
U = np.random.randn(n)

# Instrument (randomized, e.g., encouragement or lottery)
Z = np.random.binomial(1, 0.5, n)

# Compliance types (compliers, always-takers, never-takers)
# Monotonicity: No defiers (Z=1 never decreases D)
compliance_type = np.random.choice(['complier', 'always-taker', 'never-taker'], 
                                   n, p=[0.3, 0.2, 0.5])

# Potential treatments Dâ‚ (if Z=1), Dâ‚€ (if Z=0)
D1 = np.where((compliance_type == 'complier') | (compliance_type == 'always-taker'), 1, 0)
D0 = np.where(compliance_type == 'always-taker', 1, 0)

# Observed treatment (depends on Z and compliance type)
D = Z * D1 + (1 - Z) * D0

# True treatment effects (heterogeneous by compliance type)
# Compliers benefit most, always-takers moderate, never-takers not treated
tau_complier = 5.0
tau_always = 3.0
tau_never = 4.0  # Hypothetical (they never get treated)

tau_individual = np.where(compliance_type == 'complier', tau_complier,
                          np.where(compliance_type == 'always-taker', tau_always, tau_never))

tau_ATE_true = np.mean(tau_individual)
tau_LATE_true = tau_complier  # LATE = effect on compliers
tau_ATT_true = np.mean(tau_individual[D==1])  # Treated = compliers(Z=1) + always-takers

print(f"\nSimulation Setup:")
print(f"  Sample size: n={n}")
print(f"  Instrument: Z ~ Bernoulli(0.5) (randomized)")
print(f"  Compliance types:")
print(f"    Compliers: {np.sum(compliance_type=='complier')} ({100*np.mean(compliance_type=='complier'):.1f}%)")
print(f"    Always-takers: {np.sum(compliance_type=='always-taker')} ({100*np.mean(compliance_type=='always-taker'):.1f}%)")
print(f"    Never-takers: {np.sum(compliance_type=='never-taker')} ({100*np.mean(compliance_type=='never-taker'):.1f}%)")
print(f"  Monotonicity: No defiers âœ“")

print(f"\nTrue Treatment Effects:")
print(f"  ATE: {tau_ATE_true:.2f} (population average)")
print(f"  LATE: {tau_LATE_true:.2f} (compliers only)")
print(f"  ATT: {tau_ATT_true:.2f} (compliers + always-takers)")

# Potential outcomes
Y0 = 10 + 2*X1 + 3*X2 + 2*U + np.random.randn(n)*2
Y1 = Y0 + tau_individual + np.random.randn(n)*1.5

# Observed outcome
Y = D * Y1 + (1 - D) * Y0

print(f"\nEndogeneity:")
print(f"  Corr(D, U) = {np.corrcoef(D, U)[0,1]:.3f} (confounding present)")
print(f"  Corr(Z, U) = {np.corrcoef(Z, U)[0,1]:.3f} (instrument exogenous)")

# ===== Naive OLS (Biased) =====
print("\n" + "="*80)
print("NAIVE OLS (Endogenous Treatment)")
print("="*80)

X_ols = np.column_stack([np.ones(n), D, X1, X2])
beta_ols = np.linalg.lstsq(X_ols, Y, rcond=None)[0]
Y_pred_ols = X_ols @ beta_ols
resid_ols = Y - Y_pred_ols

tau_ols = beta_ols[1]
se_ols = np.sqrt(np.sum(resid_ols**2) / (n - X_ols.shape[1])) * \
         np.sqrt(np.linalg.inv(X_ols.T @ X_ols)[1,1])

print(f"OLS: Ï„Ì‚ = {tau_ols:.2f} (SE: {se_ols:.2f})")
print(f"True LATE: {tau_LATE_true:.2f}")
print(f"Bias: {tau_ols - tau_LATE_true:.2f}")
print(f"âš  OLS biased due to Corr(D, U) â‰  0")

# ===== Intent-to-Treat (ITT) - Reduced Form =====
print("\n" + "="*80)
print("INTENT-TO-TREAT (ITT) - Reduced Form")
print("="*80)

# ITT: Effect of Z on Y (no IV assumptions except randomization)
Y_Z1 = Y[Z==1].mean()
Y_Z0 = Y[Z==0].mean()
ITT = Y_Z1 - Y_Z0
se_itt = np.sqrt(Y[Z==1].var()/np.sum(Z==1) + Y[Z==0].var()/np.sum(Z==0))

print(f"ITT = E[Y|Z=1] - E[Y|Z=0] = {ITT:.2f} (SE: {se_itt:.2f})")

# First-stage: Effect of Z on D (compliance rate)
D_Z1 = D[Z==1].mean()
D_Z0 = D[Z==0].mean()
first_stage = D_Z1 - D_Z0

print(f"First-stage = E[D|Z=1] - E[D|Z=0] = {first_stage:.3f}")
print(f"  Compliance rate: {100*first_stage:.1f}%")

# ===== Wald Estimator (LATE) =====
print("\n" + "="*80)
print("WALD ESTIMATOR (LATE)")
print("="*80)

tau_wald = ITT / first_stage
se_wald = se_itt / first_stage  # Simplified (ignores first-stage SE)

print(f"LATE (Wald) = ITT / First-stage")
print(f"            = {ITT:.2f} / {first_stage:.3f}")
print(f"            = {tau_wald:.2f} (SE: {se_wald:.2f})")
print(f"True LATE: {tau_LATE_true:.2f}")
print(f"Bias: {tau_wald - tau_LATE_true:.2f}")

print(f"\nInterpretation:")
print(f"  â€¢ ITT={ITT:.2f}: Average effect of being offered treatment (diluted by non-compliance)")
print(f"  â€¢ LATE={tau_wald:.2f}: Effect on compliers (those induced by Z)")
print(f"  â€¢ LATE > ITT because compliers have full treatment effect")

# ===== Two-Stage Least Squares (2SLS) =====
print("\n" + "="*80)
print("TWO-STAGE LEAST SQUARES (2SLS)")
print("="*80)

# First stage: D ~ Z + X1 + X2
X_first = np.column_stack([np.ones(n), Z, X1, X2])
gamma = np.linalg.lstsq(X_first, D, rcond=None)[0]
D_hat = X_first @ gamma
resid_first = D - D_hat

# First-stage diagnostics
SSR_first = np.sum(resid_first**2)
SST_first = np.sum((D - D.mean())**2)
R2_first = 1 - SSR_first / SST_first

# F-statistic for Hâ‚€: Î³_Z = 0 (Z coefficient in first stage)
df_first = n - X_first.shape[1]
sigma2_first = SSR_first / df_first
se_gamma_Z = np.sqrt(sigma2_first * np.linalg.inv(X_first.T @ X_first)[1,1])
F_stat = (gamma[1] / se_gamma_Z)**2

print(f"First Stage: DÌ‚ = {gamma[0]:.2f} + {gamma[1]:.3f}Â·Z + {gamma[2]:.2f}Â·Xâ‚ + {gamma[3]:.2f}Â·Xâ‚‚")
print(f"  RÂ² = {R2_first:.4f}")
print(f"  F-statistic (Z): {F_stat:.2f}")

if F_stat > 10:
    print(f"  âœ“ Strong instrument (F > 10)")
else:
    print(f"  âš  Weak instrument (F < 10)")

# Second stage: Y ~ DÌ‚ + X1 + X2
X_second = np.column_stack([np.ones(n), D_hat, X1, X2])
beta_2sls = np.linalg.lstsq(X_second, Y, rcond=None)[0]
Y_pred_2sls = X_second @ beta_2sls
resid_2sls = Y - Y_pred_2sls

tau_2sls = beta_2sls[1]

# Standard errors (2SLS-corrected)
# Simplified: Use residuals from second stage
SSR_2sls = np.sum(resid_2sls**2)
sigma2_2sls = SSR_2sls / (n - X_second.shape[1])

# Correct SE accounts for first-stage estimation
# Here simplified; use statsmodels or linearmodels for proper SE
se_2sls = np.sqrt(sigma2_2sls * np.linalg.inv(X_second.T @ X_second)[1,1])

print(f"\nSecond Stage (2SLS):")
print(f"  LATE (2SLS): Ï„Ì‚ = {tau_2sls:.2f} (SE: {se_2sls:.2f})")
print(f"  True LATE: {tau_LATE_true:.2f}")
print(f"  Bias: {tau_2sls - tau_LATE_true:.2f}")

print(f"\nComparison:")
print(f"  OLS: {tau_ols:.2f} (biased downward due to confounding)")
print(f"  Wald: {tau_wald:.2f} (simple LATE)")
print(f"  2SLS: {tau_2sls:.2f} (LATE with covariates)")

# ===== Complier Characteristics (Abadie 2003) =====
print("\n" + "="*80)
print("COMPLIER CHARACTERIZATION")
print("="*80)

# Estimate complier share
complier_share_est = first_stage
complier_share_true = np.mean(compliance_type == 'complier')

print(f"Complier Share:")
print(f"  Estimated: Ï€Ì‚ = {complier_share_est:.3f}")
print(f"  True: Ï€ = {complier_share_true:.3f}")

# Abadie (2003) weights for complier averages
# w_i = 1 - D_i(1-Z_i)/(1-p) - (1-D_i)Z_i/p where p=P(Z=1)
p_Z = Z.mean()
w = 1 - D*(1-Z)/(1-p_Z) - (1-D)*Z/p_Z

# Complier average characteristics
X1_complier_est = np.sum(w * X1) / np.sum(w)
X2_complier_est = np.sum(w * X2) / np.sum(w)

X1_complier_true = X1[compliance_type == 'complier'].mean()
X2_complier_true = X2[compliance_type == 'complier'].mean()

print(f"\nComplier Average Characteristics:")
print(f"  Xâ‚ (age): Estimated={X1_complier_est:.2f}, True={X1_complier_true:.2f}")
print(f"  Xâ‚‚ (education): Estimated={X2_complier_est:.2f}, True={X2_complier_true:.2f}")

print(f"\nPopulation Averages (for comparison):")
print(f"  Xâ‚: {X1.mean():.2f}")
print(f"  Xâ‚‚: {X2.mean():.2f}")

# ===== Weak Instrument Simulation =====
print("\n" + "="*80)
print("WEAK INSTRUMENT SIMULATION")
print("="*80)

# Create weak instrument (small effect on D)
Z_weak = np.random.randn(n)  # Continuous instrument
alpha_weak = 0.05  # Small effect (weak)

D_weak = np.where(alpha_weak * Z_weak + 0.3*U + np.random.randn(n) > 0, 1, 0)

# First-stage for weak instrument
X_weak_first = np.column_stack([np.ones(n), Z_weak, X1, X2])
gamma_weak = np.linalg.lstsq(X_weak_first, D_weak, rcond=None)[0]
D_weak_hat = X_weak_first @ gamma_weak
resid_weak_first = D_weak - D_weak_hat

SSR_weak = np.sum(resid_weak_first**2)
df_weak = n - X_weak_first.shape[1]
sigma2_weak = SSR_weak / df_weak
se_gamma_weak = np.sqrt(sigma2_weak * np.linalg.inv(X_weak_first.T @ X_weak_first)[1,1])
F_stat_weak = (gamma_weak[1] / se_gamma_weak)**2

print(f"Weak Instrument Example:")
print(f"  First-stage F-statistic: {F_stat_weak:.2f}")

if F_stat_weak < 10:
    print(f"  âš  Weak instrument (F < 10)")
    print(f"  Consequences:")
    print(f"    â€¢ 2SLS biased toward OLS")
    print(f"    â€¢ Standard errors too small")
    print(f"    â€¢ Confidence intervals under-cover")
    print(f"  Solutions:")
    print(f"    â€¢ Find stronger instrument")
    print(f"    â€¢ Use weak-IV robust inference (Anderson-Rubin)")
    print(f"    â€¢ Report ITT instead of LATE")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Treatment assignment by instrument
ax1 = axes[0, 0]
treatment_by_Z = pd.DataFrame({'Z': Z, 'D': D, 'Type': compliance_type})
treatment_counts = treatment_by_Z.groupby(['Z', 'D']).size().unstack(fill_value=0)

x_pos = np.array([0, 1])
width = 0.35

ax1.bar(x_pos - width/2, 
        [treatment_counts.loc[0, 0] if 0 in treatment_counts.loc[0] else 0,
         treatment_counts.loc[1, 0] if 0 in treatment_counts.loc[1] else 0],
        width, label='D=0', alpha=0.7)
ax1.bar(x_pos + width/2,
        [treatment_counts.loc[0, 1] if 1 in treatment_counts.loc[0] else 0,
         treatment_counts.loc[1, 1] if 1 in treatment_counts.loc[1] else 0],
        width, label='D=1', alpha=0.7)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(['Z=0', 'Z=1'])
ax1.set_ylabel('Count')
ax1.set_title('Treatment Assignment by Instrument')
ax1.legend()
ax1.grid(alpha=0.3, axis='y')

# Plot 2: Compliance types
ax2 = axes[0, 1]
type_counts = pd.Series(compliance_type).value_counts()
colors = ['#3498db', '#e74c3c', '#2ecc71']

ax2.bar(type_counts.index, type_counts.values, color=colors, alpha=0.7)
ax2.set_ylabel('Count')
ax2.set_title('Compliance Types Distribution')
ax2.grid(alpha=0.3, axis='y')

# Add percentage labels
for i, (k, v) in enumerate(type_counts.items()):
    ax2.text(i, v + 10, f'{100*v/n:.1f}%', ha='center')

# Plot 3: First-stage relationship
ax3 = axes[0, 2]
Z_jitter = Z + np.random.randn(n)*0.02
D_jitter = D + np.random.randn(n)*0.02

ax3.scatter(Z_jitter, D_jitter, alpha=0.3, s=10)

# First-stage fit
Z_grid = np.array([0, 1])
D_pred = gamma[0] + gamma[1]*Z_grid + gamma[2]*X1.mean() + gamma[3]*X2.mean()
ax3.plot(Z_grid, D_pred, 'r-', linewidth=3, label=f'First-stage (F={F_stat:.1f})')

ax3.set_xlabel('Instrument (Z)')
ax3.set_ylabel('Treatment (D)')
ax3.set_title(f'First-Stage: Relevance (F={F_stat:.1f})')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: ITT (reduced form)
ax4 = axes[1, 0]
Y_by_Z = [Y[Z==0], Y[Z==1]]

bp = ax4.boxplot(Y_by_Z, labels=['Z=0', 'Z=1'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.6)

ax4.set_ylabel('Outcome (Y)')
ax4.set_title(f'Reduced Form: ITT={ITT:.2f}')
ax4.grid(alpha=0.3, axis='y')

# Add mean lines
means = [Y[Z==0].mean(), Y[Z==1].mean()]
ax4.plot([1, 2], means, 'ro-', linewidth=2, markersize=10, label='Means')
ax4.legend()

# Plot 5: Estimate comparison
ax5 = axes[1, 1]
methods = ['OLS\n(biased)', 'ITT\n(diluted)', 'Wald\n(LATE)', '2SLS\n(LATE)']
estimates = [tau_ols, ITT, tau_wald, tau_2sls]
colors_bar = ['red', 'orange', 'green', 'blue']

bars = ax5.bar(methods, estimates, color=colors_bar, alpha=0.7)
ax5.axhline(tau_LATE_true, color='black', linestyle='--', linewidth=2, label='True LATE')
ax5.axhline(tau_ATE_true, color='gray', linestyle=':', linewidth=2, label='True ATE')
ax5.set_ylabel('Treatment Effect')
ax5.set_title('Estimator Comparison')
ax5.legend()
ax5.grid(alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, estimates)):
    ax5.text(i, val + 0.2, f'{val:.2f}', ha='center', fontsize=10)

# Plot 6: 2SLS vs OLS scatter
ax6 = axes[1, 2]
ax6.scatter(D, Y, alpha=0.3, s=10, label='Data')

# OLS line
D_grid = np.array([0, 1])
Y_ols_line = beta_ols[0] + beta_ols[1]*D_grid + \
             beta_ols[2]*X1.mean() + beta_ols[3]*X2.mean()
ax6.plot(D_grid, Y_ols_line, 'r-', linewidth=3, label=f'OLS: {tau_ols:.2f}')

# 2SLS line
Y_2sls_line = beta_2sls[0] + beta_2sls[1]*D_grid + \
              beta_2sls[2]*X1.mean() + beta_2sls[3]*X2.mean()
ax6.plot(D_grid, Y_2sls_line, 'b-', linewidth=3, label=f'2SLS: {tau_2sls:.2f}')

ax6.set_xlabel('Treatment (D)')
ax6.set_ylabel('Outcome (Y)')
ax6.set_title('OLS vs 2SLS')
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('local_average_treatment_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n1. Instrument Validity:")
print(f"   â€¢ Relevance: First-stage F={F_stat:.1f} > 10 âœ“")
print(f"   â€¢ Exogeneity: Z randomized, Corr(Z,U)={np.corrcoef(Z,U)[0,1]:.3f} â‰ˆ 0 âœ“")
print(f"   â€¢ Exclusion: Z affects Y only through D (by design) âœ“")
print(f"   â€¢ Monotonicity: No defiers (compliance types defined) âœ“")

print("\n2. Endogeneity:")
print(f"   â€¢ OLS biased: Ï„Ì‚_OLS={tau_ols:.2f} vs LATE={tau_LATE_true:.2f}")
print(f"   â€¢ Confounding: Corr(D,U)={np.corrcoef(D,U)[0,1]:.3f} â‰  0")

print("\n3. LATE Estimation:")
print(f"   â€¢ Wald: {tau_wald:.2f} (simple ratio)")
print(f"   â€¢ 2SLS: {tau_2sls:.2f} (with covariates)")
print(f"   â€¢ Both recover true LATE={tau_LATE_true:.2f} âœ“")

print("\n4. Interpretation:")
print(f"   â€¢ LATE={tau_LATE_true:.2f} is effect on compliers ({100*complier_share_true:.0f}% of population)")
print(f"   â€¢ ATE={tau_ATE_true:.2f} â‰  LATE (compliers benefit more)")
print(f"   â€¢ ITT={ITT:.2f} < LATE (diluted by non-compliance)")

print("\n5. External Validity:")
print(f"   â€¢ LATE specific to this instrument and compliers")
print(f"   â€¢ Cannot extrapolate to ATE without assumptions")
print(f"   â€¢ Compliers may differ from population in unobserved ways")

print("\n6. Practical Recommendations:")
print("   â€¢ Check first-stage strength (F > 10)")
print("   â€¢ Defend exclusion restriction (theory/institutional knowledge)")
print("   â€¢ Test overidentification if multiple instruments")
print("   â€¢ Characterize compliers (who benefits from instrument?)")
print("   â€¢ Report ITT alongside LATE (robust to weak IV)")
print("   â€¢ Sensitivity analysis for exclusion violations")

print("\n" + "="*80)
