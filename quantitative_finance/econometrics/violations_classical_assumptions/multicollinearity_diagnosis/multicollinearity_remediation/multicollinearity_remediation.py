import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# MULTICOLLINEARITY DETECTION AND REMEDIATION
# ============================================================================

class MulticollinearityAnalysis:
    """
    Comprehensive multicollinearity diagnosis and remediation
    """
    
    def __init__(self, n=100, multicollinearity_level='high'):
        """
        Generate data with multicollinear X variables
        y = 2 + 3*Xâ‚ + 2*Xâ‚‚ + 1*Xâ‚ƒ + Îµ
        where Xâ‚‚ â‰ˆ 0.9*Xâ‚ and Xâ‚ƒ â‰ˆ 0.8*Xâ‚ + 0.6*Xâ‚‚
        
        Parameters:
        -----------
        n: Sample size
        multicollinearity_level: 'low', 'moderate', 'high'
        """
        self.n = n
        self.level = multicollinearity_level
        
        # Generate Xâ‚ (base variable)
        self.X1 = np.random.normal(5, 2, n)
        
        # Generate Xâ‚‚ (highly correlated with Xâ‚)
        if multicollinearity_level == 'high':
            self.X2 = 0.90 * self.X1 + np.random.normal(0, 0.5, n)
            self.X3 = 0.85 * self.X1 + 0.60 * self.X2 + np.random.normal(0, 0.3, n)
        elif multicollinearity_level == 'moderate':
            self.X2 = 0.50 * self.X1 + np.random.normal(0, 1.5, n)
            self.X3 = 0.40 * self.X1 + 0.50 * self.X2 + np.random.normal(0, 1.0, n)
        else:  # low
            self.X2 = np.random.normal(5, 2, n)
            self.X3 = np.random.normal(5, 2, n)
        
        # Generate y with true effects
        epsilon = np.random.normal(0, 1, n)
        self.y = 2 + 3 * self.X1 + 2 * self.X2 + 1 * self.X3 + epsilon
        
        # Design matrices
        self.X_full = np.column_stack([np.ones(n), self.X1, self.X2, self.X3])
        self.X_standardized = StandardScaler(with_mean=False).fit_transform(
            np.column_stack([self.X1, self.X2, self.X3])
        )
        
    def correlation_analysis(self):
        """Compute correlation matrix"""
        data = np.column_stack([self.X1, self.X2, self.X3])
        corr_matrix = np.corrcoef(data.T)
        
        return pd.DataFrame(
            corr_matrix,
            columns=['Xâ‚', 'Xâ‚‚', 'Xâ‚ƒ'],
            index=['Xâ‚', 'Xâ‚‚', 'Xâ‚ƒ']
        )
    
    def vif_analysis(self):
        """Compute Variance Inflation Factors"""
        vif_values = []
        
        for j in range(3):  # 3 regressors
            # Regress X_j on other X's
            X_others = np.column_stack([self.X_standardized[:, i] for i in range(3) if i != j])
            X_j = self.X_standardized[:, j]
            
            model = LinearRegression()
            model.fit(X_others, X_j)
            y_pred = model.predict(X_others)
            
            # RÂ² from auxiliary regression
            ss_res = np.sum((X_j - y_pred)**2)
            ss_tot = np.sum((X_j - np.mean(X_j))**2)
            r2 = 1 - (ss_res / ss_tot)
            
            # VIF
            vif = 1 / (1 - r2) if r2 < 0.9999 else np.inf
            vif_values.append(vif)
        
        return pd.DataFrame({
            'Variable': ['Xâ‚', 'Xâ‚‚', 'Xâ‚ƒ'],
            'VIF': vif_values,
            'Severity': ['Low' if v < 5 else 'Moderate' if v < 10 else 'High' for v in vif_values]
        })
    
    def condition_number(self):
        """Compute condition number of X'X"""
        X_centered = self.X_standardized
        XX = X_centered.T @ X_centered
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(XX)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Condition number
        kappa = eigenvalues[0] / eigenvalues[-1]
        
        return {
            'condition_number': kappa,
            'eigenvalues': eigenvalues,
            'severity': 'Good' if kappa < 10 else 'Moderate' if kappa < 30 else 'Severe'
        }
    
    def ols_analysis(self):
        """Standard OLS regression"""
        model = LinearRegression()
        model.fit(self.X_full, self.y)
        
        # Coefficients and residuals
        beta = model.coef_
        y_pred = model.predict(self.X_full)
        residuals = self.y - y_pred
        
        # Standard errors
        n, k = self.X_full.shape
        mse = np.sum(residuals**2) / (n - k)
        var_covar = mse * np.linalg.inv(self.X_full.T @ self.X_full)
        se = np.sqrt(np.diag(var_covar))
        
        return {
            'beta': beta,
            'se': se,
            'y_pred': y_pred,
            'residuals': residuals,
            'r_squared': 1 - (np.sum(residuals**2) / np.sum((self.y - np.mean(self.y))**2))
        }
    
    def coefficient_stability(self, bootstrap_samples=100):
        """
        Bootstrap resampling to assess coefficient stability
        Large coefficient variation = unstable = multicollinearity
        """
        bootstrap_betas = []
        
        for _ in range(bootstrap_samples):
            # Resample
            idx = np.random.choice(self.n, self.n, replace=True)
            X_boot = self.X_full[idx]
            y_boot = self.y[idx]
            
            # OLS
            model = LinearRegression()
            model.fit(X_boot, y_boot)
            bootstrap_betas.append(model.coef_)
        
        bootstrap_betas = np.array(bootstrap_betas)
        
        return {
            'beta_mean': np.mean(bootstrap_betas, axis=0),
            'beta_std': np.std(bootstrap_betas, axis=0),
            'bootstrap_betas': bootstrap_betas
        }
    
    def ridge_regression(self, alphas=None):
        """Ridge regression with cross-validation"""
        if alphas is None:
            alphas = np.logspace(-2, 4, 50)
        
        cv_scores = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            cv_score = cross_val_score(ridge, self.X_full, self.y, cv=5, 
                                       scoring='neg_mean_squared_error')
            cv_scores.append(-cv_score.mean())
        
        # Best alpha
        best_idx = np.argmin(cv_scores)
        best_alpha = alphas[best_idx]
        
        # Fit with best alpha
        ridge_final = Ridge(alpha=best_alpha)
        ridge_final.fit(self.X_full, self.y)
        
        return {
            'alphas': alphas,
            'cv_scores': cv_scores,
            'best_alpha': best_alpha,
            'coefficients': ridge_final.coef_,
            'predictions': ridge_final.predict(self.X_full)
        }
    
    def lasso_regression(self, alphas=None):
        """LASSO regression with cross-validation"""
        if alphas is None:
            alphas = np.logspace(-3, 1, 50)
        
        cv_scores = []
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            cv_score = cross_val_score(lasso, self.X_full, self.y, cv=5,
                                       scoring='neg_mean_squared_error')
            cv_scores.append(-cv_score.mean())
        
        # Best alpha
        best_idx = np.argmin(cv_scores)
        best_alpha = alphas[best_idx]
        
        # Fit with best alpha
        lasso_final = Lasso(alpha=best_alpha, max_iter=10000)
        lasso_final.fit(self.X_full, self.y)
        
        return {
            'alphas': alphas,
            'cv_scores': cv_scores,
            'best_alpha': best_alpha,
            'coefficients': lasso_final.coef_,
            'predictions': lasso_final.predict(self.X_full)
        }
    
    def pca_regression(self):
        """Principal Components Regression"""
        pca = PCA()
        X_pca = pca.fit_transform(self.X_standardized)
        
        # Add intercept
        X_pca_full = np.column_stack([np.ones(self.n), X_pca])
        
        # OLS on principal components
        model = LinearRegression(fit_intercept=False)
        model.fit(X_pca_full, self.y)
        
        # Variance explained by each PC
        var_explained = pca.explained_variance_ratio_
        
        return {
            'pca_loadings': pca.components_,
            'var_explained': var_explained,
            'cumsum_var': np.cumsum(var_explained),
            'pc_coefficients': model.coef_,
            'predictions': model.predict(X_pca_full)
        }
    
    def variable_selection(self):
        """Drop Xâ‚ƒ (most dependent) and refit"""
        X_reduced = np.column_stack([np.ones(self.n), self.X1, self.X2])
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X_reduced, self.y)
        
        # SEs
        residuals = self.y - model.predict(X_reduced)
        n, k = X_reduced.shape
        mse = np.sum(residuals**2) / (n - k)
        var_covar = mse * np.linalg.inv(X_reduced.T @ X_reduced)
        se = np.sqrt(np.diag(var_covar))
        
        return {
            'beta': model.coef_,
            'se': se,
            'predictions': model.predict(X_reduced)
        }
    
    def comparison_table(self):
        """Create comparison of all methods"""
        ols_result = self.ols_analysis()
        ridge_result = self.ridge_regression()
        lasso_result = self.lasso_regression()
        pca_result = self.pca_regression()
        varsel_result = self.variable_selection()
        
        # MSE
        ols_mse = np.mean((self.y - ols_result['y_pred'])**2)
        ridge_mse = np.mean((self.y - ridge_result['predictions'])**2)
        lasso_mse = np.mean((self.y - lasso_result['predictions'])**2)
        pca_mse = np.mean((self.y - pca_result['predictions'])**2)
        varsel_mse = np.mean((self.y - varsel_result['predictions'])**2)
        
        comparison = pd.DataFrame({
            'Method': ['OLS', 'Ridge', 'LASSO', 'PCA', 'Variable Selection'],
            'MSE': [ols_mse, ridge_mse, lasso_mse, pca_mse, varsel_mse],
            'RÂ²': [
                ols_result['r_squared'],
                1 - (ridge_mse / np.var(self.y)),
                1 - (lasso_mse / np.var(self.y)),
                1 - (pca_mse / np.var(self.y)),
                1 - (varsel_mse / np.var(self.y))
            ],
            'Interpretability': ['Full', 'Full', 'Sparse', 'PCs', 'Reduced']
        })
        
        return comparison


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("MULTICOLLINEARITY ANALYSIS: DIAGNOSIS & REMEDIATION")
print("="*80)

# Initialize
analysis = MulticollinearityAnalysis(n=100, multicollinearity_level='high')

# Correlation analysis
print(f"\n1. CORRELATION MATRIX")
corr = analysis.correlation_analysis()
print(corr.round(3))
print(f"   Interpretation: Xâ‚‚ â‰ˆ 0.9Â·Xâ‚ (high correlation)")

# VIF analysis
print(f"\n2. VARIANCE INFLATION FACTORS (VIF)")
vif_df = analysis.vif_analysis()
print(vif_df.to_string(index=False))
print(f"   Interpretation: VIF > 5 indicates severe multicollinearity")

# Condition number
cond_result = analysis.condition_number()
print(f"\n3. CONDITION NUMBER")
print(f"   Îº = {cond_result['condition_number']:.2f}")
print(f"   Eigenvalues: {cond_result['eigenvalues'].round(2)}")
print(f"   Severity: {cond_result['severity']}")
print(f"   (Îº > 100 = severe ill-conditioning)")

# OLS Analysis
ols_result = analysis.ols_analysis()
print(f"\n4. OLS REGRESSION")
print(f"   Coefficients (true: [2, 3, 2, 1]):")
print(f"   Î²â‚€: {ols_result['beta'][0]:7.3f} (SE: {ols_result['se'][0]:.3f})")
print(f"   Î²â‚: {ols_result['beta'][1]:7.3f} (SE: {ols_result['se'][1]:.3f})")
print(f"   Î²â‚‚: {ols_result['beta'][2]:7.3f} (SE: {ols_result['se'][2]:.3f})")
print(f"   Î²â‚ƒ: {ols_result['beta'][3]:7.3f} (SE: {ols_result['se'][3]:.3f})")
print(f"   RÂ²: {ols_result['r_squared']:.4f}")
print(f"   Interpretation: Large SE due to multicollinearity")

# Coefficient stability (bootstrap)
boot_result = analysis.coefficient_stability(bootstrap_samples=100)
print(f"\n5. COEFFICIENT STABILITY (Bootstrap 100 samples)")
print(f"   Î²â‚ mean: {boot_result['beta_mean'][1]:.3f}, std: {boot_result['beta_std'][1]:.3f}")
print(f"   Î²â‚‚ mean: {boot_result['beta_mean'][2]:.3f}, std: {boot_result['beta_std'][2]:.3f}")
print(f"   Î²â‚ƒ mean: {boot_result['beta_mean'][3]:.3f}, std: {boot_result['beta_std'][3]:.3f}")
print(f"   High std = unstable coefficients = multicollinearity problem")

# Ridge regression
ridge_result = analysis.ridge_regression()
print(f"\n6. RIDGE REGRESSION")
print(f"   Optimal Î± (via CV): {ridge_result['best_alpha']:.4f}")
print(f"   Coefficients:")
print(f"   Î²â‚€: {ridge_result['coefficients'][0]:7.3f}")
print(f"   Î²â‚: {ridge_result['coefficients'][1]:7.3f}")
print(f"   Î²â‚‚: {ridge_result['coefficients'][2]:7.3f}")
print(f"   Î²â‚ƒ: {ridge_result['coefficients'][3]:7.3f}")

# LASSO regression
lasso_result = analysis.lasso_regression()
print(f"\n7. LASSO REGRESSION")
print(f"   Optimal Î± (via CV): {lasso_result['best_alpha']:.4f}")
print(f"   Coefficients (sparse):")
print(f"   Î²â‚€: {lasso_result['coefficients'][0]:7.3f}")
print(f"   Î²â‚: {lasso_result['coefficients'][1]:7.3f}")
print(f"   Î²â‚‚: {lasso_result['coefficients'][2]:7.3f}")
print(f"   Î²â‚ƒ: {lasso_result['coefficients'][3]:7.3f}")
print(f"   (Zeros indicate dropped variables)")

# PCA
pca_result = analysis.pca_regression()
print(f"\n8. PRINCIPAL COMPONENTS REGRESSION")
print(f"   Variance explained: {pca_result['var_explained'].round(3)}")
print(f"   Cumulative: {pca_result['cumsum_var'].round(3)}")
print(f"   Trade-off: Orthogonal regressors but lose interpretability")

# Variable selection
varsel_result = analysis.variable_selection()
print(f"\n9. VARIABLE SELECTION (Drop Xâ‚ƒ)")
print(f"   Coefficients:")
print(f"   Î²â‚€: {varsel_result['beta'][0]:7.3f} (SE: {varsel_result['se'][0]:.3f})")
print(f"   Î²â‚: {varsel_result['beta'][1]:7.3f} (SE: {varsel_result['se'][1]:.3f})")
print(f"   Î²â‚‚: {varsel_result['beta'][2]:7.3f} (SE: {varsel_result['se'][2]:.3f})")
print(f"   SE reduced (less multicollinearity)")

# Comparison
print(f"\n10. METHOD COMPARISON")
comparison = analysis.comparison_table()
print(comparison.to_string(index=False))

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Correlation heatmap
ax1 = axes[0, 0]
corr_vals = np.corrcoef(np.column_stack([analysis.X1, analysis.X2, analysis.X3]).T)
im1 = ax1.imshow(corr_vals, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1, 2])
ax1.set_xticklabels(['Xâ‚', 'Xâ‚‚', 'Xâ‚ƒ'])
ax1.set_yticklabels(['Xâ‚', 'Xâ‚‚', 'Xâ‚ƒ'])
ax1.set_title('Panel 1: Correlation Matrix\n(Dark red = high positive correlation)')
for i in range(3):
    for j in range(3):
        ax1.text(j, i, f'{corr_vals[i, j]:.2f}', ha='center', va='center', color='black')
plt.colorbar(im1, ax=ax1)

# Panel 2: VIF comparison
ax2 = axes[0, 1]
vif_df = analysis.vif_analysis()
colors = ['green' if v < 5 else 'orange' if v < 10 else 'red' for v in vif_df['VIF']]
ax2.bar(vif_df['Variable'], vif_df['VIF'], color=colors, edgecolor='black', linewidth=1.5)
ax2.axhline(y=5, color='orange', linestyle='--', linewidth=2, label='Threshold = 5')
ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Severe = 10')
ax2.set_ylabel('VIF')
ax2.set_title('Panel 2: Variance Inflation Factors\n(VIF > 5 indicates problem)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Coefficient estimates across bootstrap samples
ax3 = axes[1, 0]
boot_betas = boot_result['bootstrap_betas']
positions = [0, 1, 2, 3]
bp = ax3.boxplot([boot_betas[:, i] for i in range(4)], labels=['Î²â‚€', 'Î²â‚', 'Î²â‚‚', 'Î²â‚ƒ'],
                  patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax3.set_ylabel('Coefficient Value')
ax3.set_title('Panel 3: Coefficient Stability (Bootstrap)\n(Large boxes = unstable due to multicollinearity)')
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Method comparison (MSE)
ax4 = axes[1, 1]
methods = comparison['Method']
mse = comparison['MSE']
colors_mse = ['blue', 'green', 'orange', 'red', 'purple']
bars = ax4.bar(range(len(methods)), mse, color=colors_mse, edgecolor='black', linewidth=1.5)
ax4.set_xticks(range(len(methods)))
ax4.set_xticklabels(methods, rotation=45, ha='right')
ax4.set_ylabel('Mean Squared Error')
ax4.set_title('Panel 4: Model Comparison\n(Lower MSE = better prediction)')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('multicollinearity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"â€¢ High multicollinearity detected: Corr(Xâ‚,Xâ‚‚) = {corr.iloc[0,1]:.3f}")
print(f"â€¢ VIF values > 5 confirm multicollinearity problem")
print(f"â€¢ Bootstrap shows unstable coefficients (high std of Î² estimates)")
print(f"â€¢ Ridge/LASSO reduce variance compared to OLS")
print(f"â€¢ Trade-off: Bias introduced but SE reduced")
print(f"â€¢ Variable selection simplifies but may omit important info")
print("="*80 + "\n")
