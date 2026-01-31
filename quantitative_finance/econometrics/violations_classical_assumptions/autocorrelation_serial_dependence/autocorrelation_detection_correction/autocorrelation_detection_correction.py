import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# AUTOCORRELATION DETECTION AND CORRECTION
# ============================================================================

class AutocorrelationAnalysis:
    """
    Comprehensive autocorrelation detection and correction
    """
    
    def __init__(self, T=150, rho=0.65):
        """
        Generate time series data with AR(1) errors
        y_t = 5 + 2*X_t + Îµ_t, where Îµ_t = Ï*Îµ_{t-1} + Î½_t
        
        Parameters:
        -----------
        T: Number of time periods
        rho: AR(1) coefficient (autocorrelation strength)
        """
        self.T = T
        self.rho = rho
        
        # Generate X (exogenous regressor)
        self.X = np.random.normal(5, 2, T)
        
        # Generate autocorrelated errors
        nu = np.random.normal(0, 1.0, T)
        epsilon = np.zeros(T)
        epsilon[0] = nu[0]
        for t in range(1, T):
            epsilon[t] = rho * epsilon[t-1] + nu[t]
        
        self.epsilon = epsilon
        
        # Generate dependent variable
        self.y = 5 + 2 * self.X + epsilon
        
        # Design matrix
        self.X_design = np.column_stack([np.ones(T), self.X])
        
    def fit_ols(self):
        """Fit OLS and compute naive (incorrect) standard errors"""
        model = LinearRegression(fit_intercept=False)
        model.fit(self.X_design, self.y)
        
        self.beta_ols = model.coef_
        self.y_pred = model.predict(self.X_design)
        self.residuals = self.y - self.y_pred
        
        # Naive variance (assumes independence)
        n, k = self.X_design.shape
        mse = np.sum(self.residuals**2) / (n - k)
        self.X_prime_X_inv = np.linalg.inv(self.X_design.T @ self.X_design)
        
        # OLS standard errors (BIASED under autocorrelation)
        self.var_ols_naive = mse * self.X_prime_X_inv
        self.se_ols_naive = np.sqrt(np.diag(self.var_ols_naive))
        
        return self.beta_ols, self.se_ols_naive
    
    def durbin_watson_test(self):
        """Compute Durbin-Watson statistic"""
        dw = np.sum(np.diff(self.residuals)**2) / np.sum(self.residuals**2)
        
        # Approximate AR(1) coefficient
        rho_est = 1 - (dw / 2)
        
        # Approximate standard error of rho
        rho_se = np.sqrt((1 - rho_est**2) / self.T)
        
        return {
            'dw_statistic': dw,
            'rho_estimated': rho_est,
            'rho_se': rho_se,
            'conclusion': 'Positive autocorr' if dw < 2 else 'Negative autocorr' if dw > 2 else 'No autocorr'
        }
    
    def breusch_godfrey_test(self, p=1):
        """
        Breusch-Godfrey test for autocorrelation
        More powerful than DW; handles MA terms
        """
        # Auxiliary regression: ÎµÌ‚_t = Î± + Ï†*ÎµÌ‚_{t-1} + X*Î³ + u_t
        y_aux = self.residuals[p:]
        X_aux = []
        
        for t in range(p, self.T):
            row = [1] + list(self.X_design[t, :]) + list(self.residuals[t-1:t-p:-1])
            X_aux.append(row)
        
        X_aux = np.array(X_aux)
        
        # Fit auxiliary regression
        aux_model = LinearRegression(fit_intercept=False)
        aux_model.fit(X_aux, y_aux)
        
        y_aux_pred = aux_model.predict(X_aux)
        ss_res = np.sum((y_aux - y_aux_pred)**2)
        ss_tot = np.sum((y_aux - np.mean(y_aux))**2)
        r2_aux = 1 - (ss_res / ss_tot)
        
        # LM statistic
        lm_stat = (self.T - p) * r2_aux
        
        # p-value
        p_value = 1 - stats.chi2.cdf(lm_stat, p)
        
        return {
            'test_statistic': lm_stat,
            'p_value': p_value,
            'r2_auxiliary': r2_aux,
            'reject_independence': p_value < 0.05
        }
    
    def newey_west_se(self, lags=None):
        """
        Newey-West HAC standard errors
        Robust to autocorrelation and heteroskedasticity
        """
        if lags is None:
            # Automatic lag selection
            lags = int(np.ceil(4 * (self.T / 100)**(2/9)))
        
        # Bartlett weights
        def bartlett_weights(lag, max_lag):
            return 1 - lag / (max_lag + 1)
        
        # Initialize long-run variance
        omega = np.zeros_like(self.X_prime_X_inv)
        
        # Contemporaneous term
        for t in range(self.T):
            omega += self.residuals[t]**2 * np.outer(self.X_design[t], self.X_design[t])
        
        # Autocovariance terms
        for lag in range(1, lags + 1):
            weight = bartlett_weights(lag, lags)
            for t in range(lag, self.T):
                cross_prod = self.residuals[t] * self.residuals[t-lag]
                omega += 2 * weight * cross_prod * np.outer(self.X_design[t], self.X_design[t-lag])
        
        # Variance-covariance matrix
        var_nw = self.X_prime_X_inv @ omega @ self.X_prime_X_inv / self.T
        self.se_nw = np.sqrt(np.diag(var_nw))
        
        return self.se_nw
    
    def feasible_gls_ar1(self):
        """
        Feasible GLS for AR(1) process
        Stage 1: Estimate Ï from residuals
        Stage 2: Quasi-difference and apply OLS
        """
        # Stage 1: Estimate Ï using DW approximation or direct estimation
        dw_result = self.durbin_watson_test()
        rho_est = dw_result['rho_estimated']
        
        # Ensure stationarity
        rho_est = np.clip(rho_est, -0.99, 0.99)
        
        # Stage 2: Quasi-difference (GLS transformation)
        # First observation: use original
        y_gls = np.zeros(self.T)
        X_gls = np.zeros_like(self.X_design)
        
        y_gls[0] = self.y[0]
        X_gls[0] = self.X_design[0]
        
        # Remaining observations: quasi-differenced
        for t in range(1, self.T):
            y_gls[t] = self.y[t] - rho_est * self.y[t-1]
            X_gls[t] = self.X_design[t] - rho_est * self.X_design[t-1]
        
        # Stage 3: OLS on transformed model
        model_gls = LinearRegression(fit_intercept=False)
        model_gls.fit(X_gls, y_gls)
        
        self.beta_gls = model_gls.coef_
        y_pred_gls = model_gls.predict(X_gls)
        residuals_gls = y_gls - y_pred_gls
        
        # SE for GLS
        n, k = X_gls.shape
        mse_gls = np.sum(residuals_gls**2) / (n - k)
        X_gls_prime_X = X_gls.T @ X_gls
        var_gls = mse_gls * np.linalg.inv(X_gls_prime_X)
        
        self.se_gls = np.sqrt(np.diag(var_gls))
        
        return self.beta_gls, self.se_gls
    
    def dynamic_model(self):
        """
        Add lagged dependent variable to capture dynamics
        y_t = Î²_0 + Î²_1*X_t + Ï*y_{t-1} + Îµ_t
        """
        # Create lagged y
        y_lag = np.concatenate([[self.y[0]], self.y[:-1]])
        
        # Design matrix with lagged y
        X_dynamic = np.column_stack([np.ones(self.T), self.X, y_lag])
        
        # OLS
        model_dyn = LinearRegression(fit_intercept=False)
        model_dyn.fit(X_dynamic, self.y)
        
        self.beta_dynamic = model_dyn.coef_
        y_pred_dyn = model_dyn.predict(X_dynamic)
        residuals_dyn = self.y - y_pred_dyn
        
        # SE for dynamic model
        n, k = X_dynamic.shape
        mse_dyn = np.sum(residuals_dyn**2) / (n - k)
        X_dyn_prime_X = X_dynamic.T @ X_dynamic
        var_dyn = mse_dyn * np.linalg.inv(X_dyn_prime_X)
        
        self.se_dynamic = np.sqrt(np.diag(var_dyn))
        
        return self.beta_dynamic, self.se_dynamic
    
    def summary_table(self):
        """Create comparison table"""
        comparison = pd.DataFrame({
            'Method': ['OLS (Naive)', 'OLS (Newey-West)', 'FGLS (AR1)', 'Dynamic (Lagged Y)'],
            'Î²â‚ Estimate': [
                f"{self.beta_ols[1]:.4f}",
                f"{self.beta_ols[1]:.4f}",
                f"{self.beta_gls[1]:.4f}",
                f"{self.beta_dynamic[1]:.4f}"
            ],
            'SE(Î²â‚)': [
                f"{self.se_ols_naive[1]:.4f}",
                f"{self.se_nw[1]:.4f}",
                f"{self.se_gls[1]:.4f}",
                f"{self.se_dynamic[1]:.4f}"
            ],
            '95% CI for Î²â‚': [
                f"[{self.beta_ols[1]-1.96*self.se_ols_naive[1]:.4f}, {self.beta_ols[1]+1.96*self.se_ols_naive[1]:.4f}]",
                f"[{self.beta_ols[1]-1.96*self.se_nw[1]:.4f}, {self.beta_ols[1]+1.96*self.se_nw[1]:.4f}]",
                f"[{self.beta_gls[1]-1.96*self.se_gls[1]:.4f}, {self.beta_gls[1]+1.96*self.se_gls[1]:.4f}]",
                f"[{self.beta_dynamic[1]-1.96*self.se_dynamic[1]:.4f}, {self.beta_dynamic[1]+1.96*self.se_dynamic[1]:.4f}]"
            ]
        })
        
        return comparison


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("AUTOCORRELATION ANALYSIS: DETECTION & CORRECTION")
print("="*80)

# Initialize
analysis = AutocorrelationAnalysis(T=150, rho=0.65)

# Fit OLS
analysis.fit_ols()
print(f"\n1. OLS ESTIMATION (with AR(1) errors, Ï=0.65)")
print(f"   Î²â‚€ (true=5): {analysis.beta_ols[0]:.4f}")
print(f"   Î²â‚ (true=2): {analysis.beta_ols[1]:.4f}")

# Durbin-Watson Test
dw_result = analysis.durbin_watson_test()
print(f"\n2. DURBIN-WATSON TEST")
print(f"   DW Statistic: {dw_result['dw_statistic']:.4f}")
print(f"   (DW â‰ˆ 2 if no autocorr, < 2 if positive)")
print(f"   Estimated Ï: {dw_result['rho_estimated']:.4f} (true Ï = 0.65)")
print(f"   Conclusion: {dw_result['conclusion']}")

# Breusch-Godfrey Test
bg_result = analysis.breusch_godfrey_test(p=1)
print(f"\n3. BREUSCH-GODFREY TEST")
print(f"   LM Statistic: {bg_result['test_statistic']:.4f}")
print(f"   p-value: {bg_result['p_value']:.6f}")
print(f"   Conclusion: {'Reject independence (autocorr detected)' if bg_result['reject_independence'] else 'No autocorr detected'}")

# Standard Errors Comparison
analysis.newey_west_se()
print(f"\n4. STANDARD ERRORS COMPARISON")
print(f"   {'':30} OLS Naive  Newey-West")
print(f"   {'SE(Î²â‚€)':30} {analysis.se_ols_naive[0]:9.4f}  {analysis.se_nw[0]:9.4f}")
print(f"   {'SE(Î²â‚)':30} {analysis.se_ols_naive[1]:9.4f}  {analysis.se_nw[1]:9.4f}")
print(f"   SE Inflation Factor (Î²â‚):     1.00x      {analysis.se_nw[1]/analysis.se_ols_naive[1]:.2f}x")

# FGLS
analysis.feasible_gls_ar1()
print(f"\n5. FEASIBLE GLS (AR(1))")
print(f"   Î²â‚€: {analysis.beta_gls[0]:.4f} (SE: {analysis.se_gls[0]:.4f})")
print(f"   Î²â‚: {analysis.beta_gls[1]:.4f} (SE: {analysis.se_gls[1]:.4f})")
print(f"   Efficiency gain: {(analysis.se_ols_naive[1]/analysis.se_gls[1] - 1)*100:.1f}%")

# Dynamic model
analysis.dynamic_model()
print(f"\n6. DYNAMIC MODEL (Lagged Dependent Variable)")
print(f"   Î²â‚€: {analysis.beta_dynamic[0]:.4f}")
print(f"   Î²â‚ (X coeff): {analysis.beta_dynamic[1]:.4f}")
print(f"   Ï (lagged Y coeff): {analysis.beta_dynamic[2]:.4f}")

# Comparison table
print(f"\n7. COMPREHENSIVE COMPARISON")
print(analysis.summary_table().to_string(index=False))

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Time series of residuals (visual autocorrelation)
ax1 = axes[0, 0]
ax1.plot(analysis.residuals, 'b-', linewidth=1, alpha=0.7, label='OLS Residuals')
ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax1.fill_between(range(len(analysis.residuals)), 
                 -2*np.std(analysis.residuals),
                 2*np.std(analysis.residuals),
                 alpha=0.1, color='blue')
ax1.set_xlabel('Time Period (t)')
ax1.set_ylabel('Residuals (ÎµÌ‚â‚œ)')
ax1.set_title('Panel 1: Residual Time Series\n(Persistent patterns indicate autocorrelation)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: ACF (Autocorrelation Function)
ax2 = axes[0, 1]
plot_acf(analysis.residuals, lags=20, ax=ax2, title='Panel 2: ACF of Residuals\n(Slow decay = strong autocorr)')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')

# Panel 3: Lagged residual scatter (ÎµÌ‚â‚œ vs ÎµÌ‚â‚œâ‚‹â‚)
ax3 = axes[1, 0]
ax3.scatter(analysis.residuals[:-1], analysis.residuals[1:], alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
# Fit line to show correlation
z = np.polyfit(analysis.residuals[:-1], analysis.residuals[1:], 1)
p = np.poly1d(z)
x_line = np.linspace(analysis.residuals[:-1].min(), analysis.residuals[:-1].max(), 100)
ax3.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'ÏÌ‚ = {z[0]:.3f}')
ax3.set_xlabel('ÎµÌ‚â‚œâ‚‹â‚ (Lagged Residual)')
ax3.set_ylabel('ÎµÌ‚â‚œ (Current Residual)')
ax3.set_title('Panel 3: Residual Autocorrelation\n(Positive slope = positive autocorr)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Confidence interval comparison
ax4 = axes[1, 1]
methods = ['OLS\n(Naive)', 'OLS\n(Newey-West)', 'FGLS', 'Dynamic']
se_vals = [analysis.se_ols_naive[1], analysis.se_nw[1], analysis.se_gls[1], analysis.se_dynamic[1]]
betas = [analysis.beta_ols[1], analysis.beta_ols[1], analysis.beta_gls[1], analysis.beta_dynamic[1]]

ci_lower = [b - 1.96*se for b, se in zip(betas, se_vals)]
ci_upper = [b + 1.96*se for b, se in zip(betas, se_vals)]

y_pos = np.arange(len(methods))
ax4.errorbar(y_pos, betas,
             yerr=[np.array(betas) - np.array(ci_lower),
                   np.array(ci_upper) - np.array(betas)],
             fmt='o', markersize=8, capsize=5, capthick=2, color='blue')
ax4.axvline(x=2.0, color='green', linestyle='--', linewidth=2, label='True Î²â‚ = 2')
ax4.set_xticks(y_pos)
ax4.set_xticklabels(methods)
ax4.set_ylabel('Î²â‚ Estimate with 95% CI')
ax4.set_title('Panel 4: Confidence Intervals Comparison\n(OLS too narrow; Newey-West/FGLS appropriate)')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"â€¢ Naive OLS SE underestimate by ~{(1 - analysis.se_ols_naive[1]/analysis.se_nw[1])*100:.0f}%")
print(f"â€¢ Newey-West SE are ~{analysis.se_nw[1]/analysis.se_ols_naive[1]:.2f}x wider (correct)")
print(f"â€¢ FGLS achieves ~{(analysis.se_ols_naive[1]/analysis.se_gls[1] - 1)*100:.0f}% efficiency gain")
print(f"â€¢ DW = {dw_result['dw_statistic']:.3f} (< 2, confirms positive autocorr)")
print(f"â€¢ Breusch-Godfrey p-value: {bg_result['p_value']:.4f} (strongly rejects independence)")
print("="*80 + "\n")
