import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, BetweenOLS
from linearmodels.panel import compare
from scipy import stats

np.random.seed(123)

# ===== Simulate Panel Data with Random Effects =====
N = 150  # Number of units
T = 8    # Time periods
n_obs = N * T

# Random individual effects (UNCORRELATED with X)
alpha_i = np.random.normal(0, 3, N)

# Generate data where RE assumption holds
data_list = []
for i in range(N):
    for t in range(T):
        # X1, X2 exogenous (uncorrelated with alpha_i)
        x1 = 10 + np.random.normal(0, 2)
        x2 = 5 + 0.2 * t + np.random.normal(0, 1.5)
        
        # X3: Time-invariant variable (school size, gender, etc.)
        x3 = 7 + np.random.normal(0, 1)  # Same for all t within unit i
        
        # Outcome
        # True effects: Î²1=2.0, Î²2=1.5, Î²3=0.8
        epsilon = np.random.normal(0, 2)
        y = 10 + 2.0*x1 + 1.5*x2 + 0.8*x3 + alpha_i[i] + epsilon
        
        data_list.append({
            'id': i,
            'time': t,
            'y': y,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'alpha_true': alpha_i[i]
        })

df = pd.DataFrame(data_list)

# Make x3 truly time-invariant per unit
df['x3'] = df.groupby('id')['x3'].transform('first')

print("="*70)
print("RANDOM EFFECTS MODEL: PANEL DATA ANALYSIS")
print("="*70)
print(f"\nPanel Dimensions:")
print(f"  Units (N): {N}")
print(f"  Time Periods (T): {T}")
print(f"  Total Observations: {n_obs}")
print(f"\nTrue Coefficients:")
print(f"  Î²â‚ (X1): 2.0")
print(f"  Î²â‚‚ (X2): 1.5")
print(f"  Î²â‚ƒ (X3, time-invariant): 0.8")
print(f"\nDescriptive Statistics:")
print(df[['y', 'x1', 'x2', 'x3']].describe().round(3))

# Set multi-index
df_panel = df.set_index(['id', 'time'])

# ===== Pooled OLS =====
print("\n" + "="*70)
print("POOLED OLS")
print("="*70)

exog_vars = ['x1', 'x2', 'x3']
pooled_model = PooledOLS(df_panel['y'], df_panel[exog_vars]).fit()
print(pooled_model.summary)

# ===== Fixed Effects =====
print("\n" + "="*70)
print("FIXED EFFECTS MODEL")
print("="*70)

fe_model = PanelOLS(df_panel['y'], df_panel[['x1', 'x2']],  # Cannot include x3
                    entity_effects=True).fit(cov_type='clustered',
                                             cluster_entity=True)
print(fe_model.summary)

print("\nNote: X3 (time-invariant) dropped from FE model")

# ===== Random Effects =====
print("\n" + "="*70)
print("RANDOM EFFECTS MODEL")
print("="*70)

re_model = RandomEffects(df_panel['y'], df_panel[exog_vars]).fit(
    cov_type='clustered', cluster_entity=True)
print(re_model.summary)

# ===== Between Effects =====
print("\n" + "="*70)
print("BETWEEN EFFECTS ESTIMATOR")
print("="*70)

be_model = BetweenOLS(df_panel['y'], df_panel[exog_vars]).fit()
print(be_model.summary)

# ===== Variance Components =====
print("\n" + "="*70)
print("VARIANCE COMPONENTS")
print("="*70)

# Extract variance components from RE model
sigma_eps = np.sqrt(re_model.variance_decomposition.loc['Effects']['Var'])
sigma_alpha = np.sqrt(re_model.variance_decomposition.loc['Residual']['Var'])

# Calculate rho (intraclass correlation)
total_var = sigma_alpha**2 + sigma_eps**2
rho = sigma_alpha**2 / total_var

# Calculate theta (quasi-demeaning parameter)
theta = 1 - np.sqrt(sigma_eps**2 / (sigma_eps**2 + T * sigma_alpha**2))

print(f"\nVariance Decomposition:")
print(f"  ÏƒÂ²_Îµ (idiosyncratic):    {sigma_eps**2:.4f}")
print(f"  ÏƒÂ²_Î± (individual effect): {sigma_alpha**2:.4f}")
print(f"  Total variance:           {total_var:.4f}")

print(f"\nIntraclass Correlation (Ï):")
print(f"  Ï = ÏƒÂ²_Î± / (ÏƒÂ²_Î± + ÏƒÂ²_Îµ) = {rho:.4f}")
print(f"  Interpretation: {rho*100:.1f}% of total variance due to individual effects")

print(f"\nQuasi-Demeaning Parameter (Î¸):")
print(f"  Î¸ = {theta:.4f}")
print(f"  0 â†’ Between estimator, 1 â†’ FE estimator")
print(f"  Current: {'Closer to FE' if theta > 0.5 else 'Closer to Between'}")

# ===== Hausman Test =====
print("\n" + "="*70)
print("HAUSMAN SPECIFICATION TEST")
print("="*70)

# Manual Hausman test (for variables in both models)
fe_coefs = fe_model.params[['x1', 'x2']].values
re_coefs = re_model.params[['x1', 'x2']].values

coef_diff = fe_coefs - re_coefs

# Variance difference
var_fe = fe_model.cov[['x1', 'x2']].loc[['x1', 'x2']].values
var_re = re_model.cov[['x1', 'x2']].loc[['x1', 'x2']].values
var_diff = var_fe - var_re

# Hausman statistic
try:
    hausman_stat = coef_diff.T @ np.linalg.inv(var_diff) @ coef_diff
    hausman_pval = 1 - stats.chi2.cdf(hausman_stat, df=2)
    
    print(f"Hâ‚€: Cov(Î±áµ¢, Xáµ¢â‚œ) = 0 (RE is consistent and efficient)")
    print(f"Hâ‚: Cov(Î±áµ¢, Xáµ¢â‚œ) â‰  0 (RE is inconsistent, use FE)")
    print(f"\nHausman Test Statistic: Ï‡Â²(2) = {hausman_stat:.4f}")
    print(f"p-value: {hausman_pval:.4f}")
    
    if hausman_pval < 0.05:
        print("\nâœ— Reject Hâ‚€: Use FIXED EFFECTS")
        print("   Individual effects correlated with regressors")
    else:
        print("\nâœ“ Fail to reject Hâ‚€: Use RANDOM EFFECTS")
        print("   RE is consistent and more efficient")
except:
    print("Hausman test computation issue (variance matrix not positive definite)")

# ===== Breusch-Pagan LM Test =====
print("\n" + "="*70)
print("BREUSCH-PAGAN LM TEST FOR RANDOM EFFECTS")
print("="*70)

# Test H0: sigma_alpha^2 = 0 (no random effects needed)
pooled_resid = pooled_model.resid
n = len(df_panel)

# Group residuals by individual
resid_by_id = []
for i in range(N):
    resid_i = pooled_resid[df_panel.index.get_level_values('id') == i]
    resid_by_id.append(resid_i.values)

# LM statistic
sum_squares = sum([np.sum(r)**2 for r in resid_by_id])
total_ss = np.sum(pooled_resid**2)

lm_stat = (n / (2 * (T - 1))) * ((sum_squares / total_ss) - 1)**2
lm_pval = 1 - stats.chi2.cdf(lm_stat, df=1)

print(f"Hâ‚€: ÏƒÂ²_Î± = 0 (no random effects, use Pooled OLS)")
print(f"Hâ‚: ÏƒÂ²_Î± > 0 (random effects present)")
print(f"\nLM Statistic: Ï‡Â²(1) = {lm_stat:.4f}")
print(f"p-value: {lm_pval:.4f}")

if lm_pval < 0.05:
    print("\nâœ“ Reject Hâ‚€: Random effects are present")
    print("   Use RE or FE, not Pooled OLS")
else:
    print("\n  Fail to reject: No evidence of random effects")

# ===== Model Comparison =====
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Pooled OLS': pooled_model.params,
    'Between': be_model.params,
    'Random Effects': re_model.params,
    'Fixed Effects': [fe_model.params.get('x1', np.nan),
                     fe_model.params.get('x2', np.nan),
                     np.nan]  # FE can't estimate x3
}, index=['x1', 'x2', 'x3'])

print("\nCoefficient Estimates:")
print(comparison_df.round(4))

print("\nTrue Coefficients: Î²â‚=2.0, Î²â‚‚=1.5, Î²â‚ƒ=0.8")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Coefficient Comparison
models = ['True', 'Pooled', 'Between', 'RE', 'FE']
x1_coefs = [2.0, pooled_model.params['x1'], be_model.params['x1'], 
            re_model.params['x1'], fe_model.params['x1']]
x2_coefs = [1.5, pooled_model.params['x2'], be_model.params['x2'],
            re_model.params['x2'], fe_model.params['x2']]

x_pos = np.arange(len(models))
width = 0.35

axes[0, 0].bar(x_pos - width/2, x1_coefs, width, label='X1', alpha=0.8)
axes[0, 0].bar(x_pos + width/2, x2_coefs, width, label='X2', alpha=0.8)
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(models, rotation=45)
axes[0, 0].set_ylabel('Coefficient')
axes[0, 0].set_title('Coefficient Estimates Across Models')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Standard Errors Comparison
se_comparison = pd.DataFrame({
    'RE': re_model.std_errors[['x1', 'x2']].values,
    'FE': fe_model.std_errors[['x1', 'x2']].values,
    'Pooled': pooled_model.std_errors[['x1', 'x2']].values
}, index=['x1', 'x2'])

se_comparison.T.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
axes[0, 1].set_ylabel('Standard Error')
axes[0, 1].set_title('Standard Errors: RE vs FE vs Pooled')
axes[0, 1].legend(title='Variable')
axes[0, 1].grid(alpha=0.3, axis='y')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

# Plot 3: Within vs Between Variation
df_within = df.copy()
df_within['y_within'] = df_within.groupby('id')['y'].transform(lambda x: x - x.mean())
df_within['x1_within'] = df_within.groupby('id')['x1'].transform(lambda x: x - x.mean())

df_between = df.groupby('id')[['y', 'x1']].mean()

axes[0, 2].scatter(df_within['x1_within'], df_within['y_within'],
                   alpha=0.2, s=5, label='Within', color='blue')
axes[0, 2].scatter(df_between['x1'], df_between['y'],
                   alpha=0.6, s=40, label='Between', color='red')
axes[0, 2].set_xlabel('X1')
axes[0, 2].set_ylabel('Y')
axes[0, 2].set_title(f'Within vs Between Variation (Ï={rho:.3f})')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Residual Comparison
axes[1, 0].hist(pooled_model.resid, bins=40, alpha=0.5, 
               label='Pooled', density=True)
axes[1, 0].hist(re_model.resid, bins=40, alpha=0.5,
               label='RE', density=True)
axes[1, 0].hist(fe_model.resid, bins=40, alpha=0.5,
               label='FE', density=True)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Residual Distributions')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 5: Time-Invariant Variable (X3)
# Only Pooled, Between, and RE can estimate this
x3_estimates = {
    'True': 0.8,
    'Pooled': pooled_model.params['x3'],
    'Between': be_model.params['x3'],
    'RE': re_model.params['x3']
}

x3_se = {
    'Pooled': pooled_model.std_errors['x3'],
    'Between': be_model.std_errors['x3'],
    'RE': re_model.std_errors['x3']
}

models_x3 = list(x3_estimates.keys())
x3_vals = list(x3_estimates.values())
x3_errors = [0, x3_se['Pooled'], x3_se['Between'], x3_se['RE']]

axes[1, 1].bar(models_x3, x3_vals, alpha=0.8, color=['gray', 'orange', 'green', 'blue'])
axes[1, 1].errorbar(models_x3[1:], x3_vals[1:], yerr=[1.96*e for e in x3_errors[1:]],
                   fmt='none', color='black', capsize=5)
axes[1, 1].axhline(0.8, color='red', linestyle='--', linewidth=2, label='True')
axes[1, 1].set_ylabel('X3 Coefficient')
axes[1, 1].set_title('Time-Invariant Variable (X3)\n(FE Cannot Estimate)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

# Plot 6: Variance Decomposition
var_decomp_data = {
    'Within (idiosyncratic)': sigma_eps**2,
    'Between (individual)': sigma_alpha**2
}

axes[1, 2].pie(var_decomp_data.values(), labels=var_decomp_data.keys(),
              autopct='%1.1f%%', startangle=90)
axes[1, 2].set_title(f'Variance Decomposition\n(Ï = {rho:.3f})')

plt.tight_layout()
plt.show()

# ===== Summary =====
print("\n" + "="*70)
print("INTERPRETATION AND RECOMMENDATIONS")
print("="*70)

print("\n1. Variance Components:")
print(f"   - {rho*100:.1f}% of variance is between individuals")
print(f"   - {(1-rho)*100:.1f}% of variance is within individuals over time")

print("\n2. Model Choice:")
if hausman_pval >= 0.05:
    print("   âœ“ Hausman test: Use RANDOM EFFECTS")
    print("     â€¢ RE is consistent and more efficient than FE")
    print("     â€¢ Can estimate time-invariant variables")
    print(f"     â€¢ X3 coefficient: {re_model.params['x3']:.4f} (True: 0.8)")
else:
    print("   âœ— Hausman test: Use FIXED EFFECTS")
    print("     â€¢ RE is inconsistent (correlated with X)")

print("\n3. Advantages of RE (when valid):")
print("   â€¢ More efficient (smaller standard errors)")
print("   â€¢ Can estimate time-invariant variables")
print("   â€¢ Uses both within and between variation")

print("\n4. RE Assumptions:")
print("   â€¢ Cov(Î±áµ¢, Xáµ¢â‚œ) = 0 (crucial!)")
print("   â€¢ Random sample of units")
print("   â€¢ Homoskedasticity and no serial correlation (or use robust SE)")

print(f"\n5. Quasi-Demeaning (Î¸ = {theta:.3f}):")
if theta < 0.3:
    print("   â€¢ RE close to between estimator")
elif theta > 0.7:
    print("   â€¢ RE close to fixed effects")
else:
    print("   â€¢ RE balanced between within and between")
