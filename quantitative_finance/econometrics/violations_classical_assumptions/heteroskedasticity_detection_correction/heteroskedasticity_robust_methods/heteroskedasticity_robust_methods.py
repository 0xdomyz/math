import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# HETEROSKEDASTICITY DETECTION AND CORRECTION
# ============================================================================

class HeteroskedasticityAnalysis:
    """
    Comprehensive heteroskedasticity detection and correction
    """
    
    def __init__(self, n_obs=200):
        """
        Generate data with heteroskedasticity
        y = 5 + 2*X + Îµ, where Var(Îµ) = ÏƒÂ²Â·X (variance increases with X)
        """
        self.n = n_obs
        self.X = np.random.uniform(1, 10, n_obs)
        
        # Heteroskedastic errors: variance proportional to X
        self.sigma = 0.5 * self.X  # std dev increases with X
        self.epsilon = np.random.normal(0, self.sigma)
        
        self.y = 5 + 2 * self.X + self.epsilon
        
        # Add constant term for regression
        self.X_design = np.column_stack([np.ones(n_obs), self.X])
        
    def fit_ols(self):
        """Fit OLS and compute naive (incorrect) standard errors"""
        model = LinearRegression(fit_intercept=False)
        model.fit(self.X_design, self.y)
        
        self.beta_ols = model.coef_
        self.y_pred = model.predict(self.X_design)
        self.residuals = self.y - self.y_pred
        
        # Naive variance (assumes homoskedasticity)
        n, k = self.X_design.shape
        mse = np.sum(self.residuals**2) / (n - k)
        self.X_prime_X_inv = np.linalg.inv(self.X_design.T @ self.X_design)
        
        # OLS standard errors (BIASED under heteroskedasticity)
        self.var_ols_naive = mse * self.X_prime_X_inv
        self.se_ols_naive = np.sqrt(np.diag(self.var_ols_naive))
        
        return self.beta_ols, self.se_ols_naive
    
    def white_robust_se(self):
        """
        Calculate White's heteroskedasticity-consistent standard errors
        Var(Î²Ì‚) = (X'X)â»Â¹ X'Î©X (X'X)â»Â¹
        where Î© = diag(ÎµÌ‚â‚Â², ÎµÌ‚â‚‚Â², ..., ÎµÌ‚â‚™Â²)
        """
        # White HC0 formula
        omega = np.diag(self.residuals**2)
        var_white = self.X_prime_X_inv @ (self.X_design.T @ omega @ self.X_design) @ self.X_prime_X_inv
        
        # HC1 finite-sample correction
        n, k = self.X_design.shape
        var_white_hc1 = (n / (n - k)) * var_white
        
        self.se_white = np.sqrt(np.diag(var_white))
        self.se_white_hc1 = np.sqrt(np.diag(var_white_hc1))
        
        return self.se_white, self.se_white_hc1
    
    def breusch_pagan_test(self):
        """
        Test for heteroskedasticity
        Hâ‚€: Homoskedasticity (errors have constant variance)
        """
        # Auxiliary regression: ÎµÌ‚Â² on X
        aux_model = LinearRegression()
        aux_model.fit(self.X_design, self.residuals**2)
        
        # Compute RÂ² of auxiliary regression
        y_aux_pred = aux_model.predict(self.X_design)
        ss_res = np.sum((self.residuals**2 - y_aux_pred)**2)
        ss_tot = np.sum((self.residuals**2 - np.mean(self.residuals**2))**2)
        r2_aux = 1 - (ss_res / ss_tot)
        
        # Test statistic: LM = nÂ·RÂ²
        lm_stat = self.n * r2_aux
        
        # p-value from chi-square distribution
        dof = self.X_design.shape[1] - 1  # k regressors in auxiliary model
        p_value = 1 - stats.chi2.cdf(lm_stat, dof)
        
        return {
            'test_statistic': lm_stat,
            'p_value': p_value,
            'r2_auxiliary': r2_aux,
            'reject_homoskedasticity': p_value < 0.05
        }
    
    def white_test(self):
        """
        More general White test including squared and interaction terms
        """
        # Create augmented design matrix with X, XÂ², and interaction
        X_aug = np.column_stack([
            np.ones(self.n),
            self.X,
            self.X**2
        ])
        
        # Auxiliary regression: ÎµÌ‚Â² on augmented X
        aux_model = LinearRegression(fit_intercept=False)
        aux_model.fit(X_aug, self.residuals**2)
        
        y_aux_pred = aux_model.predict(X_aug)
        ss_res = np.sum((self.residuals**2 - y_aux_pred)**2)
        ss_tot = np.sum((self.residuals**2 - np.mean(self.residuals**2))**2)
        r2_aux = 1 - (ss_res / ss_tot)
        
        lm_stat = self.n * r2_aux
        dof = X_aug.shape[1] - 1
        p_value = 1 - stats.chi2.cdf(lm_stat, dof)
        
        return {
            'test_statistic': lm_stat,
            'p_value': p_value,
            'reject_homoskedasticity': p_value < 0.05
        }
    
    def weighted_least_squares(self):
        """
        WLS with weights proportional to 1/X (true variance structure is ÏƒÂ² âˆ X)
        """
        # True weights: since Var(Îµ) = ÏƒÂ²Â·X, weight = 1/ÏƒÂ²Â·X = 1/X
        weights = 1 / self.X
        
        # Transform by sqrt(weights)
        sqrt_w = np.sqrt(weights)
        X_weighted = self.X_design * sqrt_w[:, np.newaxis]
        y_weighted = self.y * sqrt_w
        
        # Fit OLS on transformed data
        model = LinearRegression(fit_intercept=False)
        model.fit(X_weighted, y_weighted)
        
        self.beta_wls = model.coef_
        y_pred_weighted = model.predict(X_weighted)
        residuals_weighted = y_weighted - y_pred_weighted
        
        # Compute SE for WLS
        n, k = X_weighted.shape
        mse_weighted = np.sum(residuals_weighted**2) / (n - k)
        X_weighted_prime_X = X_weighted.T @ X_weighted
        var_wls = mse_weighted * np.linalg.inv(X_weighted_prime_X)
        
        self.se_wls = np.sqrt(np.diag(var_wls))
        
        return self.beta_wls, self.se_wls
    
    def feasible_gls(self):
        """
        Two-stage FGLS estimation
        Stage 1: OLS to get residuals
        Stage 2: Model log(ÎµÌ‚Â²) to estimate variance
        Stage 3: WLS using estimated variance
        """
        # Stage 1: OLS already computed (self.residuals)
        
        # Stage 2: Model variance
        # ln(ÎµÌ‚Â²) = Î± + Î³Â·X + v
        log_sq_res = np.log(self.residuals**2 + 1e-6)
        
        aux_model = LinearRegression()
        aux_model.fit(self.X_design, log_sq_res)
        
        # Predicted log variance
        log_h_hat = aux_model.predict(self.X_design)
        h_hat = np.exp(log_h_hat)
        
        # Stage 3: WLS with estimated weights
        weights_fgls = 1 / h_hat
        sqrt_w_fgls = np.sqrt(weights_fgls)
        
        X_fgls = self.X_design * sqrt_w_fgls[:, np.newaxis]
        y_fgls = self.y * sqrt_w_fgls
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X_fgls, y_fgls)
        
        self.beta_fgls = model.coef_
        y_pred_fgls = model.predict(X_fgls)
        residuals_fgls = y_fgls - y_pred_fgls
        
        # SE for FGLS
        n, k = X_fgls.shape
        mse_fgls = np.sum(residuals_fgls**2) / (n - k)
        X_fgls_prime_X = X_fgls.T @ X_fgls
        var_fgls = mse_fgls * np.linalg.inv(X_fgls_prime_X)
        
        self.se_fgls = np.sqrt(np.diag(var_fgls))
        
        return self.beta_fgls, self.se_fgls
    
    def summary_table(self):
        """Create comparison table of all methods"""
        comparison = pd.DataFrame({
            'Method': ['OLS (Naive SE)', 'OLS (White HC0)', 'OLS (White HC1)', 'WLS', 'FGLS'],
            'Î²â‚€ Estimate': [
                f"{self.beta_ols[0]:.4f}",
                f"{self.beta_ols[0]:.4f}",
                f"{self.beta_ols[0]:.4f}",
                f"{self.beta_wls[0]:.4f}",
                f"{self.beta_fgls[0]:.4f}"
            ],
            'SE(Î²â‚€)': [
                f"{self.se_ols_naive[0]:.4f}",
                f"{self.se_white[0]:.4f}",
                f"{self.se_white_hc1[0]:.4f}",
                f"{self.se_wls[0]:.4f}",
                f"{self.se_fgls[0]:.4f}"
            ],
            'Î²â‚ Estimate': [
                f"{self.beta_ols[1]:.4f}",
                f"{self.beta_ols[1]:.4f}",
                f"{self.beta_ols[1]:.4f}",
                f"{self.beta_wls[1]:.4f}",
                f"{self.beta_fgls[1]:.4f}"
            ],
            'SE(Î²â‚)': [
                f"{self.se_ols_naive[1]:.4f}",
                f"{self.se_white[1]:.4f}",
                f"{self.se_white_hc1[1]:.4f}",
                f"{self.se_wls[1]:.4f}",
                f"{self.se_fgls[1]:.4f}"
            ],
            '95% CI for Î²â‚': [
                f"[{self.beta_ols[1]-1.96*self.se_ols_naive[1]:.4f}, {self.beta_ols[1]+1.96*self.se_ols_naive[1]:.4f}]",
                f"[{self.beta_ols[1]-1.96*self.se_white[1]:.4f}, {self.beta_ols[1]+1.96*self.se_white[1]:.4f}]",
                f"[{self.beta_ols[1]-1.96*self.se_white_hc1[1]:.4f}, {self.beta_ols[1]+1.96*self.se_white_hc1[1]:.4f}]",
                f"[{self.beta_wls[1]-1.96*self.se_wls[1]:.4f}, {self.beta_wls[1]+1.96*self.se_wls[1]:.4f}]",
                f"[{self.beta_fgls[1]-1.96*self.se_fgls[1]:.4f}, {self.beta_fgls[1]+1.96*self.se_fgls[1]:.4f}]"
            ]
        })
        
        return comparison


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("HETEROSKEDASTICITY ANALYSIS: DETECTION & CORRECTION")
print("="*80)

# Initialize and fit all models
analysis = HeteroskedasticityAnalysis(n_obs=200)

# Fit OLS
analysis.fit_ols()
print(f"\n1. OLS ESTIMATION (with heteroskedasticity in data)")
print(f"   Î²â‚€ (true=5): {analysis.beta_ols[0]:.4f}")
print(f"   Î²â‚ (true=2): {analysis.beta_ols[1]:.4f}")

# Get robust SEs
analysis.white_robust_se()
print(f"\n2. STANDARD ERRORS COMPARISON")
print(f"   {'':30} Naive OLS  White HC0  White HC1")
print(f"   {'SE(Î²â‚€)':30} {analysis.se_ols_naive[0]:9.4f}  {analysis.se_white[0]:9.4f}  {analysis.se_white_hc1[0]:9.4f}")
print(f"   {'SE(Î²â‚)':30} {analysis.se_ols_naive[1]:9.4f}  {analysis.se_white[1]:9.4f}  {analysis.se_white_hc1[1]:9.4f}")
print(f"   SE Inflation Factor (Î²â‚):     1.00x      {analysis.se_white[1]/analysis.se_ols_naive[1]:.2f}x      {analysis.se_white_hc1[1]/analysis.se_ols_naive[1]:.2f}x")

# Heteroskedasticity tests
bp_test = analysis.breusch_pagan_test()
white_test = analysis.white_test()

print(f"\n3. HETEROSKEDASTICITY TESTS")
print(f"   Breusch-Pagan Test:")
print(f"     LM Statistic: {bp_test['test_statistic']:.4f}")
print(f"     p-value: {bp_test['p_value']:.6f}")
print(f"     Conclusion: {'Reject Hâ‚€ (Heteroskedasticity detected)' if bp_test['reject_homoskedasticity'] else 'Fail to reject Hâ‚€'}")
print(f"\n   White Test:")
print(f"     LM Statistic: {white_test['test_statistic']:.4f}")
print(f"     p-value: {white_test['p_value']:.6f}")
print(f"     Conclusion: {'Reject Hâ‚€ (Heteroskedasticity detected)' if white_test['reject_homoskedasticity'] else 'Fail to reject Hâ‚€'}")

# WLS estimation
analysis.weighted_least_squares()
print(f"\n4. WEIGHTED LEAST SQUARES (WLS)")
print(f"   Î²â‚€: {analysis.beta_wls[0]:.4f} (SE: {analysis.se_wls[0]:.4f})")
print(f"   Î²â‚: {analysis.beta_wls[1]:.4f} (SE: {analysis.se_wls[1]:.4f})")
print(f"   Efficiency gain over OLS: {(analysis.se_ols_naive[1]/analysis.se_wls[1] - 1)*100:.1f}% (lower SE)")

# FGLS estimation
analysis.feasible_gls()
print(f"\n5. FEASIBLE GLS (FGLS)")
print(f"   Î²â‚€: {analysis.beta_fgls[0]:.4f} (SE: {analysis.se_fgls[0]:.4f})")
print(f"   Î²â‚: {analysis.beta_fgls[1]:.4f} (SE: {analysis.se_fgls[1]:.4f})")
print(f"   Efficiency gain over OLS: {(analysis.se_ols_naive[1]/analysis.se_fgls[1] - 1)*100:.1f}% (lower SE)")

# Comparison table
print(f"\n6. COMPREHENSIVE COMPARISON TABLE")
print(analysis.summary_table().to_string(index=False))

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Residuals vs Fitted (Heteroskedasticity Pattern)
ax1 = axes[0, 0]
ax1.scatter(analysis.y_pred, analysis.residuals, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
# Add reference lines showing variance pattern
ax1.fill_between(np.sort(analysis.y_pred), 
                 -1.96 * np.sort(analysis.sigma),
                 1.96 * np.sort(analysis.sigma),
                 alpha=0.2, color='red', label='True Â±1.96Ïƒ (heteroskedastic)')
ax1.set_xlabel('Fitted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Panel 1: Residual Plot (Cone Shape Indicates Heteroskedasticity)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: |Residuals| vs X (Scale-Location)
ax2 = axes[0, 1]
abs_resid_sqrt = np.sqrt(np.abs(analysis.residuals))
ax2.scatter(analysis.X, abs_resid_sqrt, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
# Smooth trend line
sort_idx = np.argsort(analysis.X)
ax2.plot(analysis.X[sort_idx], analysis.sigma[sort_idx], 'r-', linewidth=2, label='True Ïƒ(X)')
ax2.set_xlabel('X')
ax2.set_ylabel('âˆš|Residuals|')
ax2.set_title('Panel 2: Scale-Location Plot\n(Upward trend = heteroskedasticity)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Confidence Intervals Comparison
ax3 = axes[1, 0]
methods = ['OLS\n(Naive)', 'OLS\n(White)', 'OLS\n(HC1)', 'WLS', 'FGLS']
se_values = [analysis.se_ols_naive[1], analysis.se_white[1], 
             analysis.se_white_hc1[1], analysis.se_wls[1], analysis.se_fgls[1]]
ci_lower = [analysis.beta_ols[1] - 1.96*se for se in se_values[:3]] + \
           [analysis.beta_wls[1] - 1.96*analysis.se_wls[1], analysis.beta_fgls[1] - 1.96*analysis.se_fgls[1]]
ci_upper = [analysis.beta_ols[1] + 1.96*se for se in se_values[:3]] + \
           [analysis.beta_wls[1] + 1.96*analysis.se_wls[1], analysis.beta_fgls[1] + 1.96*analysis.se_fgls[1]]

y_pos = np.arange(len(methods))
ax3.errorbar(y_pos, [analysis.beta_ols[1]]*3 + [analysis.beta_wls[1], analysis.beta_fgls[1]],
             yerr=[np.array(ci_lower) - np.array([analysis.beta_ols[1]]*3 + [analysis.beta_wls[1], analysis.beta_fgls[1]]),
                   np.array(ci_upper) - np.array([analysis.beta_ols[1]]*3 + [analysis.beta_wls[1], analysis.beta_fgls[1]])],
             fmt='o', markersize=8, capsize=5, capthick=2, color='blue')
ax3.axhline(y=2.0, color='green', linestyle='--', linewidth=2, label='True Î²â‚ = 2')
ax3.set_xticks(y_pos)
ax3.set_xticklabels(methods)
ax3.set_ylabel('Î²â‚ Estimate with 95% CI')
ax3.set_title('Panel 3: Confidence Interval Widths\n(OLS Naive too narrow; White/WLS/FGLS appropriate)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Residuals Squared vs X (For Auxiliary Regression)
ax4 = axes[1, 1]
ax4.scatter(analysis.X, analysis.residuals**2, alpha=0.6, s=30, edgecolor='k', linewidth=0.5, label='Observed ÎµÌ‚Â²')
# Fit auxiliary regression for visualization
aux_model = LinearRegression()
aux_model.fit(np.column_stack([np.ones(len(analysis.X)), analysis.X]), analysis.residuals**2)
X_plot = np.linspace(analysis.X.min(), analysis.X.max(), 100)
y_aux = aux_model.predict(np.column_stack([np.ones(len(X_plot)), X_plot]))
ax4.plot(X_plot, y_aux, 'r-', linewidth=2, label='Fitted: ÎµÌ‚Â² = Î± + Î³Â·X')
ax4.set_xlabel('X')
ax4.set_ylabel('Squared Residuals (ÎµÌ‚Â²)')
ax4.set_title('Panel 4: Breusch-Pagan Auxiliary Regression\n(Upward trend confirms heteroskedasticity)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heteroskedasticity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"â€¢ Naive OLS SE underestimate true uncertainty by ~{(1 - analysis.se_ols_naive[1]/analysis.se_white_hc1[1])*100:.0f}%")
print(f"â€¢ White HC1 SE are ~{analysis.se_white_hc1[1]/analysis.se_ols_naive[1]:.2f}x wider (correct for inference)")
print(f"â€¢ WLS achieves ~{(1 - analysis.se_wls[1]/analysis.se_ols_naive[1])*100:.0f}% efficiency gain (lowest SE)")
print(f"â€¢ FGLS achieves ~{(1 - analysis.se_fgls[1]/analysis.se_ols_naive[1])*100:.0f}% efficiency gain (asymptotically optimal)")
print(f"â€¢ Breusch-Pagan p-value: {bp_test['p_value']:.4f} (strongly rejects homoskedasticity)")
print("="*80 + "\n")
