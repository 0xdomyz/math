import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from scipy import stats

np.random.seed(123)
n = 500

# ===== Data Generating Process =====
# True model: Y = Î²â‚€ + Î²â‚Xâ‚ + Î²â‚‚Xâ‚‚ + Îµ
# Xâ‚‚ is endogenous, instruments Zâ‚ and Zâ‚‚

# Exogenous variables
Z1 = np.random.normal(0, 1, n)
Z2 = np.random.normal(0, 1, n)
X1_exog = np.random.normal(0, 1, n)

# Endogenous variable X2 (correlated with error)
u = np.random.normal(0, 1, n)  # Unobserved confounder
epsilon = np.random.normal(0, 1, n)
X2_endog = 2 + 0.6*Z1 + 0.4*Z2 + 0.5*u + np.random.normal(0, 1, n)

# Outcome (true coefficients: Î²â‚=1, Î²â‚‚=2)
Y = 5 + 1*X1_exog + 2*X2_endog + 0.7*u + epsilon

df = pd.DataFrame({
    'Y': Y,
    'X1': X1_exog,
    'X2': X2_endog,
    'Z1': Z1,
    'Z2': Z2
})

# ===== Manual 2SLS Implementation =====
print("="*70)
print("MANUAL 2SLS ESTIMATION")
print("="*70)

# Stage 1: Regress X2 on all exogenous variables (X1, Z1, Z2)
Z_matrix = sm.add_constant(df[['X1', 'Z1', 'Z2']])
stage1_model = sm.OLS(df['X2'], Z_matrix).fit()

print("\n--- STAGE 1 ---")
print(f"F-statistic: {stage1_model.fvalue:.2f}")
print(f"RÂ²: {stage1_model.rsquared:.4f}")
print("\nCoefficients:")
print(stage1_model.params)

# Check instrument relevance
f_stat_instruments = stage1_model.f_test('Z1 = 0, Z2 = 0').fvalue[0][0]
print(f"\nF-test (Z1=Z2=0): {f_stat_instruments:.2f}")
if f_stat_instruments < 10:
    print("âš ï¸  WARNING: Weak instruments (F < 10)")
else:
    print("âœ“ Instruments appear strong")

# Predicted X2 from stage 1
X2_predicted = stage1_model.predict(Z_matrix)

# Stage 2: Regress Y on X1 and predicted X2
X_stage2 = sm.add_constant(pd.DataFrame({'X1': df['X1'], 'X2_hat': X2_predicted}))
stage2_model = sm.OLS(df['Y'], X_stage2).fit()

print("\n--- STAGE 2 (Naive SE - INCORRECT) ---")
print(stage2_model.summary().tables[1])

# Correct standard errors for 2SLS
# Calculate residuals from stage 2
residuals_stage2 = df['Y'] - stage2_model.predict(X_stage2)
sigma_squared = np.sum(residuals_stage2**2) / (n - 3)  # df adjustment

# Projection matrix: P_Z = Z(Z'Z)^{-1}Z'
Z_mat = Z_matrix.values
P_Z = Z_mat @ np.linalg.inv(Z_mat.T @ Z_mat) @ Z_mat.T

# X matrix with predicted values
X_mat = sm.add_constant(df[['X1', 'X2']]).values
var_beta_2sls = sigma_squared * np.linalg.inv(X_mat.T @ P_Z @ X_mat)
se_corrected = np.sqrt(np.diag(var_beta_2sls))

print("\n--- CORRECTED 2SLS STANDARD ERRORS ---")
coef_names = ['Intercept', 'X1', 'X2']
for i, name in enumerate(coef_names):
    coef = stage2_model.params[i] if i < 1 else stage2_model.params[f'X{"1" if i==1 else "2_hat"}']
    print(f"{name:12s}: {coef:8.4f}  SE: {se_corrected[i]:.4f}  "
          f"t: {coef/se_corrected[i]:6.2f}")

# ===== Using linearmodels package =====
print("\n" + "="*70)
print("2SLS USING LINEARMODELS PACKAGE")
print("="*70)

# Specify formula: Y ~ exogenous + [endogenous ~ instruments]
iv_model = IV2SLS.from_formula('Y ~ 1 + X1 + [X2 ~ Z1 + Z2]', 
                               data=df).fit(cov_type='unadjusted')

print(iv_model.summary.tables[1])

# Compare with manual implementation
print("\n" + "="*70)
print("COMPARISON: Manual vs Package")
print("="*70)
print(f"{'Parameter':<12} {'Manual':>10} {'Package':>10} {'Difference':>12}")
print("-"*46)
print(f"{'Intercept':<12} {stage2_model.params[0]:>10.4f} "
      f"{iv_model.params['Intercept']:>10.4f} "
      f"{abs(stage2_model.params[0] - iv_model.params['Intercept']):>12.6f}")
print(f"{'X1':<12} {stage2_model.params['X1']:>10.4f} "
      f"{iv_model.params['X1']:>10.4f} "
      f"{abs(stage2_model.params['X1'] - iv_model.params['X1']):>12.6f}")
print(f"{'X2':<12} {stage2_model.params['X2_hat']:>10.4f} "
      f"{iv_model.params['X2']:>10.4f} "
      f"{abs(stage2_model.params['X2_hat'] - iv_model.params['X2']):>12.6f}")

# ===== Biased OLS for comparison =====
print("\n" + "="*70)
print("NAIVE OLS (BIASED)")
print("="*70)
X_ols = sm.add_constant(df[['X1', 'X2']])
ols_model = sm.OLS(df['Y'], X_ols).fit()
print(ols_model.summary().tables[1])

# ===== Visualizations =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: First stage scatter - Z1 vs X2
axes[0, 0].scatter(df['Z1'], df['X2'], alpha=0.4, s=20, label='Data')
z_range = np.linspace(df['Z1'].min(), df['Z1'].max(), 100)
X1_mean = df['X1'].mean()
Z2_mean = df['Z2'].mean()
x2_pred = (stage1_model.params['const'] + 
           stage1_model.params['X1']*X1_mean +
           stage1_model.params['Z1']*z_range + 
           stage1_model.params['Z2']*Z2_mean)
axes[0, 0].plot(z_range, x2_pred, 'r-', linewidth=2, label='First Stage Fit')
axes[0, 0].set_xlabel('Instrument Z1')
axes[0, 0].set_ylabel('Endogenous X2')
axes[0, 0].set_title(f'First Stage Relationship (F={f_stat_instruments:.1f})')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Stage 1 residuals vs fitted
axes[0, 1].scatter(stage1_model.fittedvalues, stage1_model.resid, 
                   alpha=0.4, s=20)
axes[0, 1].axhline(0, color='r', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Fitted Values (Stage 1)')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Stage 1 Residual Plot')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Compare coefficient estimates
methods = ['True', 'OLS\n(Biased)', '2SLS\n(Manual)', '2SLS\n(Package)']
x2_coefs = [2.0, 
            ols_model.params['X2'], 
            stage2_model.params['X2_hat'],
            iv_model.params['X2']]
x2_ses = [0, 
          ols_model.bse['X2'], 
          se_corrected[2],
          iv_model.std_errors['X2']]

x_pos = np.arange(len(methods))
colors = ['black', 'red', 'blue', 'green']
axes[1, 0].bar(x_pos, x2_coefs, yerr=[1.96*se for se in x2_ses],
               capsize=5, color=colors, alpha=0.6)
axes[1, 0].axhline(2.0, color='black', linestyle='--', linewidth=2,
                   label='True Î²â‚‚=2')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(methods)
axes[1, 0].set_ylabel('X2 Coefficient')
axes[1, 0].set_title('Comparison of Estimates (95% CI)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Standard error comparison
axes[1, 1].text(0.5, 0.9, 'EFFICIENCY COMPARISON', 
                ha='center', fontsize=13, weight='bold',
                transform=axes[1, 1].transAxes)

comparison_text = f"""
True Î²â‚‚ = 2.00

OLS (Biased but efficient):
  Estimate: {ols_model.params['X2']:.4f}
  Std Err:  {ols_model.bse['X2']:.4f}
  Bias:     {ols_model.params['X2'] - 2:.4f}

2SLS (Consistent):
  Estimate: {iv_model.params['X2']:.4f}
  Std Err:  {iv_model.std_errors['X2']:.4f}
  Bias:     {iv_model.params['X2'] - 2:.4f}

SE Inflation: {iv_model.std_errors['X2']/ols_model.bse['X2']:.2f}x

First Stage:
  F-statistic: {f_stat_instruments:.2f}
  RÂ²: {stage1_model.rsquared:.3f}
"""

axes[1, 1].text(0.05, 0.75, comparison_text,
                transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# ===== Overidentification Test (Sargan) =====
if len(['Z1', 'Z2']) > 1:  # Multiple instruments
    print("\n" + "="*70)
    print("SARGAN OVERIDENTIFICATION TEST")
    print("="*70)
    
    # Regress 2SLS residuals on all instruments
    resid_2sls = df['Y'] - iv_model.predict(df)
    Z_all = sm.add_constant(df[['X1', 'Z1', 'Z2']])
    sargan_reg = sm.OLS(resid_2sls, Z_all).fit()
    
    # Test statistic: n * RÂ²
    sargan_stat = n * sargan_reg.rsquared
    df_sargan = len(['Z1', 'Z2']) - 1  # # instruments - # endogenous
    p_value = 1 - stats.chi2.cdf(sargan_stat, df_sargan)
    
    print(f"Hâ‚€: All instruments are valid (exogenous)")
    print(f"Sargan statistic: {sargan_stat:.4f}")
    print(f"Degrees of freedom: {df_sargan}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("âœ— Reject Hâ‚€: At least one instrument appears invalid")
    else:
        print("âœ“ Fail to reject: Instruments appear valid")
