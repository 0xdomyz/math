import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from scipy.linalg import svd

np.random.seed(999)

# ===== Simulate Data with Multicollinearity =====
n = 300  # Sample size

# Generate base variables
Z = np.random.normal(0, 1, n)  # Common factor

# X1 and X2 highly correlated (both related to Z)
X1 = 2*Z + np.random.normal(0, 0.5, n)
X2 = 1.5*Z + np.random.normal(0, 0.5, n)  # High correlation with X1

# X3 independent
X3 = np.random.normal(5, 2, n)

# True model: Y = 5 + 2*X1 + 3*X2 + 1*X3 + Îµ
# But X1 and X2 highly correlated
epsilon = np.random.normal(0, 2, n)
Y = 5 + 2*X1 + 3*X2 + 1*X3 + epsilon

# Create DataFrame
df = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3})

print("="*80)
print("MULTICOLLINEARITY: DETECTION AND REMEDIES")
print("="*80)
print(f"\nSimulation Setup:")
print(f"  Sample Size: {n}")
print(f"  True Model: Y = 5 + 2*X1 + 3*X2 + 1*X3 + Îµ")
print(f"  X1 and X2 constructed to be highly correlated")
print(f"\nDescriptive Statistics:")
print(df.describe().round(3))

# ===== Correlation Analysis =====
print("\n" + "="*80)
print("CORRELATION MATRIX")
print("="*80)

corr_matrix = df[['X1', 'X2', 'X3']].corr()
print("\nPearwise Correlations:")
print(corr_matrix.round(3))

print(f"\nCorr(X1, X2) = {corr_matrix.loc['X1', 'X2']:.3f}  {'âš  High!' if abs(corr_matrix.loc['X1', 'X2']) > 0.7 else 'âœ“ OK'}")
print(f"Corr(X1, X3) = {corr_matrix.loc['X1', 'X3']:.3f}  {'âš  High!' if abs(corr_matrix.loc['X1', 'X3']) > 0.7 else 'âœ“ OK'}")
print(f"Corr(X2, X3) = {corr_matrix.loc['X2', 'X3']:.3f}  {'âš  High!' if abs(corr_matrix.loc['X2', 'X3']) > 0.7 else 'âœ“ OK'}")

# Visualize correlation matrix
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
           square=True, ax=axes[0, 0], vmin=-1, vmax=1, fmt='.3f')
axes[0, 0].set_title('Correlation Matrix of Predictors')

# ===== OLS Regression =====
print("\n" + "="*80)
print("OLS REGRESSION WITH MULTICOLLINEARITY")
print("="*80)

X = sm.add_constant(df[['X1', 'X2', 'X3']])
ols_model = sm.OLS(df['Y'], X).fit()
print(ols_model.summary())

print("\nOLS Estimates vs True:")
print(f"  Intercept: {ols_model.params['const']:.4f} (True: 5.0)")
print(f"  X1 coeff:  {ols_model.params['X1']:.4f} Â± {ols_model.bse['X1']:.4f} (True: 2.0)")
print(f"  X2 coeff:  {ols_model.params['X2']:.4f} Â± {ols_model.bse['X2']:.4f} (True: 3.0)")
print(f"  X3 coeff:  {ols_model.params['X3']:.4f} Â± {ols_model.bse['X3']:.4f} (True: 1.0)")

# Note the large standard errors on X1, X2
print("\nNote: Large SE on X1 and X2 despite both being important")

# ===== Variance Inflation Factor =====
print("\n" + "="*80)
print("VARIANCE INFLATION FACTOR (VIF)")
print("="*80)

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data['Variable'] = ['X1', 'X2', 'X3']
vif_data['VIF'] = [variance_inflation_factor(df[['X1', 'X2', 'X3']].values, i) 
                   for i in range(3)]

print("\nVIF Values:")
print(vif_data.to_string(index=False))

print("\nInterpretation:")
for idx, row in vif_data.iterrows():
    var = row['Variable']
    vif = row['VIF']
    if vif > 10:
        status = "âœ— Severe multicollinearity"
    elif vif > 5:
        status = "âš  Moderate multicollinearity"
    elif vif > 2:
        status = "â—‹ Mild multicollinearity"
    else:
        status = "âœ“ No multicollinearity"
    print(f"  {var}: VIF = {vif:.2f}  {status}")

# ===== Condition Number =====
print("\n" + "="*80)
print("CONDITION NUMBER")
print("="*80)

# Standardize X for condition number
X_std = StandardScaler().fit_transform(df[['X1', 'X2', 'X3']])

# SVD to get eigenvalues
U, s, Vt = svd(X_std, full_matrices=False)

condition_number = s.max() / s.min()

print(f"\nCondition Number (Îº): {condition_number:.2f}")
print(f"\nEigenvalues (singular values):")
for i, eigenval in enumerate(s):
    print(f"  Î»{i+1}: {eigenval:.4f}")

if condition_number < 10:
    print("\nâœ“ No severe multicollinearity (Îº < 10)")
elif condition_number < 30:
    print("\nâš  Moderate multicollinearity (10 < Îº < 30)")
else:
    print("\nâœ— Severe multicollinearity (Îº > 30)")

# ===== Auxiliary Regressions =====
print("\n" + "="*80)
print("AUXILIARY REGRESSIONS (RÂ² for each predictor)")
print("="*80)

for var in ['X1', 'X2', 'X3']:
    other_vars = [v for v in ['X1', 'X2', 'X3'] if v != var]
    X_aux = sm.add_constant(df[other_vars])
    aux_model = sm.OLS(df[var], X_aux).fit()
    
    vif_calc = 1 / (1 - aux_model.rsquared)
    
    print(f"\n{var} ~ {' + '.join(other_vars)}:")
    print(f"  RÂ² = {aux_model.rsquared:.4f}")
    print(f"  VIF = 1/(1-RÂ²) = {vif_calc:.2f}")
    print(f"  F-statistic: {aux_model.fvalue:.2f}, p-value: {aux_model.f_pvalue:.6f}")

# ===== Regression Without X2 (Omitted Variable Bias) =====
print("\n" + "="*80)
print("REGRESSION WITHOUT X2 (Omitted Variable Bias)")
print("="*80)

X_reduced = sm.add_constant(df[['X1', 'X3']])
ols_reduced = sm.OLS(df['Y'], X_reduced).fit()
print(ols_reduced.summary())

print("\nComparison:")
print(f"  X1 (full model):    {ols_model.params['X1']:.4f} Â± {ols_model.bse['X1']:.4f}")
print(f"  X1 (reduced model): {ols_reduced.params['X1']:.4f} Â± {ols_reduced.bse['X1']:.4f}")
print(f"\nNote: Dropping X2 biases X1 coefficient, but SE smaller!")

# ===== Ridge Regression =====
print("\n" + "="*80)
print("RIDGE REGRESSION (L2 Regularization)")
print("="*80)

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['X1', 'X2', 'X3']])
Y_values = df['Y'].values

# Cross-validation to find optimal alpha
alphas = np.logspace(-2, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_scaled, Y_values)

print(f"Optimal Î± (regularization): {ridge_cv.alpha_:.4f}")

# Fit ridge with optimal alpha
ridge_model = Ridge(alpha=ridge_cv.alpha_)
ridge_model.fit(X_scaled, Y_values)

print("\nRidge Coefficients (standardized):")
ridge_coefs = pd.DataFrame({
    'Variable': ['X1', 'X2', 'X3'],
    'OLS (scaled)': StandardScaler().fit_transform(
        ols_model.params[['X1', 'X2', 'X3']].values.reshape(1, -1)).flatten(),
    'Ridge': ridge_model.coef_
})
print(ridge_coefs.round(4))

# ===== Principal Component Regression =====
print("\n" + "="*80)
print("PRINCIPAL COMPONENT REGRESSION (PCR)")
print("="*80)

# PCA on predictors
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("\nPrincipal Components:")
print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"  Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")

# Regression on principal components
X_pca_const = sm.add_constant(X_pca)
pcr_model = sm.OLS(Y_values, X_pca_const).fit()

print("\nPCR Results:")
print(pcr_model.summary())

print("\nPrincipal Component Loadings:")
loadings = pd.DataFrame(pca.components_.T, 
                        columns=['PC1', 'PC2', 'PC3'],
                        index=['X1', 'X2', 'X3'])
print(loadings.round(3))

# ===== Visualizations =====

# Plot 1: Scatter X1 vs X2 (showing high correlation)
axes[0, 1].scatter(df['X1'], df['X2'], alpha=0.5, s=20)
axes[0, 1].set_xlabel('X1')
axes[0, 1].set_ylabel('X2')
axes[0, 1].set_title(f'X1 vs X2 (r = {corr_matrix.loc["X1", "X2"]:.3f})')
axes[0, 1].grid(alpha=0.3)

# Add regression line
z = np.polyfit(df['X1'], df['X2'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['X1'].min(), df['X1'].max(), 100)
axes[0, 1].plot(x_line, p(x_line), 'r-', linewidth=2)

# Plot 2: VIF Bar Chart
axes[0, 2].bar(vif_data['Variable'], vif_data['VIF'], alpha=0.8)
axes[0, 2].axhline(5, color='orange', linestyle='--', linewidth=2, label='Moderate (VIF=5)')
axes[0, 2].axhline(10, color='red', linestyle='--', linewidth=2, label='Severe (VIF=10)')
axes[0, 2].set_ylabel('VIF')
axes[0, 2].set_title('Variance Inflation Factors')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(alpha=0.3, axis='y')

# Plot 3: Coefficient Comparison
coef_comparison = pd.DataFrame({
    'True': [2.0, 3.0, 1.0],
    'OLS': [ols_model.params['X1'], ols_model.params['X2'], ols_model.params['X3']],
    'OLS (no X2)': [ols_reduced.params['X1'], np.nan, ols_reduced.params['X3']],
    'Ridge': ridge_model.coef_
}, index=['X1', 'X2', 'X3'])

x_pos = np.arange(3)
width = 0.2

for i, method in enumerate(['True', 'OLS', 'OLS (no X2)', 'Ridge']):
    offset = (i - 1.5) * width
    values = coef_comparison[method].values
    axes[1, 0].bar(x_pos + offset, values, width, label=method, alpha=0.8)

axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(['X1', 'X2', 'X3'])
axes[1, 0].set_ylabel('Coefficient')
axes[1, 0].set_title('Coefficient Estimates: Methods Comparison')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Standard Error Comparison
se_comparison = pd.DataFrame({
    'OLS (full)': ols_model.bse[['X1', 'X2', 'X3']].values,
    'OLS (no X2)': [ols_reduced.bse['X1'], np.nan, ols_reduced.bse['X3']]
}, index=['X1', 'X2', 'X3'])

se_comparison.plot(kind='bar', ax=axes[1, 1], alpha=0.8)
axes[1, 1].set_ylabel('Standard Error')
axes[1, 1].set_title('Standard Errors: Full vs Reduced Model')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

# Plot 5: PCA Scree Plot
axes[1, 2].bar(range(1, 4), pca.explained_variance_ratio_, alpha=0.8)
axes[1, 2].plot(range(1, 4), np.cumsum(pca.explained_variance_ratio_), 
               'ro-', linewidth=2, markersize=8, label='Cumulative')
axes[1, 2].set_xlabel('Principal Component')
axes[1, 2].set_ylabel('Explained Variance Ratio')
axes[1, 2].set_title('PCA Scree Plot')
axes[1, 2].set_xticks([1, 2, 3])
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('multicollinearity_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Ridge Trace =====
print("\n" + "="*80)
print("RIDGE TRACE (Coefficient Stability)")
print("="*80)

fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

alphas_trace = np.logspace(-2, 2, 100)
coefs_trace = []

for alpha in alphas_trace:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, Y_values)
    coefs_trace.append(ridge.coef_)

coefs_trace = np.array(coefs_trace)

for i, var in enumerate(['X1', 'X2', 'X3']):
    ax.plot(alphas_trace, coefs_trace[:, i], label=var, linewidth=2)

ax.set_xscale('log')
ax.set_xlabel('Î± (Regularization Parameter)')
ax.set_ylabel('Standardized Coefficient')
ax.set_title('Ridge Trace: Coefficient Paths')
ax.axvline(ridge_cv.alpha_, color='red', linestyle='--', linewidth=2, 
          label=f'Optimal Î± = {ridge_cv.alpha_:.2f}')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ridge_trace.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nNote: Coefficients stabilize as Î± increases")

# ===== Monte Carlo: Effect of Multicollinearity =====
print("\n" + "="*80)
print("MONTE CARLO: VARIANCE INFLATION DUE TO MULTICOLLINEARITY")
print("="*80)

n_sim = 1000
betas_high_corr = []
betas_low_corr = []

for sim in range(n_sim):
    # High correlation scenario
    Z_sim = np.random.normal(0, 1, n)
    X1_high = 2*Z_sim + np.random.normal(0, 0.5, n)
    X2_high = 1.5*Z_sim + np.random.normal(0, 0.5, n)
    X3_sim = np.random.normal(5, 2, n)
    epsilon_sim = np.random.normal(0, 2, n)
    Y_high = 5 + 2*X1_high + 3*X2_high + 1*X3_sim + epsilon_sim
    
    X_high = np.column_stack([np.ones(n), X1_high, X2_high, X3_sim])
    beta_high = np.linalg.lstsq(X_high, Y_high, rcond=None)[0]
    betas_high_corr.append(beta_high)
    
    # Low correlation scenario
    X1_low = np.random.normal(0, 1, n)
    X2_low = np.random.normal(0, 1, n)
    X3_low = np.random.normal(5, 2, n)
    Y_low = 5 + 2*X1_low + 3*X2_low + 1*X3_low + epsilon_sim
    
    X_low = np.column_stack([np.ones(n), X1_low, X2_low, X3_low])
    beta_low = np.linalg.lstsq(X_low, Y_low, rcond=None)[0]
    betas_low_corr.append(beta_low)

betas_high_corr = np.array(betas_high_corr)
betas_low_corr = np.array(betas_low_corr)

print("\nSampling Variance of Coefficients:")
print("\nHigh Correlation (X1-X2):")
print(f"  X1: Var = {betas_high_corr[:, 1].var():.4f}, SD = {betas_high_corr[:, 1].std():.4f}")
print(f"  X2: Var = {betas_high_corr[:, 2].var():.4f}, SD = {betas_high_corr[:, 2].std():.4f}")
print(f"  X3: Var = {betas_high_corr[:, 3].var():.4f}, SD = {betas_high_corr[:, 3].std():.4f}")

print("\nLow Correlation (Independent X's):")
print(f"  X1: Var = {betas_low_corr[:, 1].var():.4f}, SD = {betas_low_corr[:, 1].std():.4f}")
print(f"  X2: Var = {betas_low_corr[:, 2].var():.4f}, SD = {betas_low_corr[:, 2].std():.4f}")
print(f"  X3: Var = {betas_low_corr[:, 3].var():.4f}, SD = {betas_low_corr[:, 3].std():.4f}")

print("\nVariance Inflation:")
print(f"  X1: {betas_high_corr[:, 1].var() / betas_low_corr[:, 1].var():.2f}Ã—")
print(f"  X2: {betas_high_corr[:, 2].var() / betas_low_corr[:, 2].var():.2f}Ã—")
print(f"  X3: {betas_high_corr[:, 3].var() / betas_low_corr[:, 3].var():.2f}Ã— (unaffected)")

# Visualization
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

for i, var in enumerate(['X1', 'X2', 'X3']):
    axes3[i].hist(betas_high_corr[:, i+1], bins=40, alpha=0.5, 
                 label='High Corr', density=True)
    axes3[i].hist(betas_low_corr[:, i+1], bins=40, alpha=0.5,
                 label='Low Corr', density=True)
    axes3[i].axvline([2, 3, 1][i], color='red', linestyle='--', 
                    linewidth=2, label='True')
    axes3[i].set_xlabel(f'{var} Coefficient')
    axes3[i].set_ylabel('Density')
    axes3[i].set_title(f'{var}: Sampling Distribution')
    axes3[i].legend()
    axes3[i].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('multicollinearity_simulation.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n1. Detection:")
print(f"   â€¢ Correlation: r(X1,X2) = {corr_matrix.loc['X1', 'X2']:.3f} (high)")
print(f"   â€¢ VIF: X1={vif_data.loc[0, 'VIF']:.1f}, X2={vif_data.loc[1, 'VIF']:.1f} (both > 5)")
print(f"   â€¢ Condition number: {condition_number:.1f}")

print("\n2. Consequences:")
print(f"   â€¢ Coefficients: Still unbiased (means correct in simulation)")
print(f"   â€¢ Standard errors: {betas_high_corr[:, 1].std() / betas_low_corr[:, 1].std():.1f}Ã— larger for X1")
print(f"   â€¢ Individual t-tests: May be insignificant despite joint importance")

print("\n3. When to Act:")
print("   â€¢ If goal is prediction: Often no action needed")
print("   â€¢ If interpreting individual coefficients: Consider remedies")
print("   â€¢ If coefficients have wrong signs: Investigation needed")

print("\n4. Remedies:")
print("   â€¢ Do nothing: Valid if prediction focus or coefficients reasonable")
print("   â€¢ Ridge regression: Reduces variance, slight bias")
print("   â€¢ Drop variable: Creates omitted variable bias")
print("   â€¢ PCA: Loses interpretability of original variables")
print("   â€¢ Collect more data: Best solution if feasible")

print("\n5. Key Insight:")
print("   Multicollinearity is a DATA problem, not a model violation.")
print("   It doesn't bias estimates, just makes them imprecise.")
