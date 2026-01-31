import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import inv

# Generate synthetic data
np.random.seed(42)
n = 500

# True parameters
beta_0 = 2.0    # Log wage intercept
beta_1 = 0.08   # Education return (8% per year)
beta_2 = 0.05   # Experience return (5% per year)
beta_3 = -0.001 # Experience squared (diminishing returns)
sigma = 0.3     # Error std

# Generate correlated education and experience
education = np.random.uniform(10, 20, n)
# Experience negatively correlated with education (more educated enter labor force later)
experience = np.maximum(0, 30 - 0.5*education + np.random.normal(0, 5, n))
experience_sq = experience ** 2

# True log wage (with unobserved ability creating correlation)
ability = np.random.normal(0, 1, n)
education_ability_corr = 0.3 * ability  # Ability correlates with education
log_wage = (beta_0 + beta_1*education + beta_2*experience + beta_3*experience_sq + 
            0.1*ability + np.random.normal(0, sigma, n))

# Create design matrix X
X = np.column_stack([np.ones(n), education, experience, experience_sq])
Y = log_wage

# OLS estimation: beta_hat = (X'X)^{-1} X'Y
XtX = X.T @ X
XtY = X.T @ Y
beta_hat = inv(XtX) @ XtY

# Predictions and residuals
Y_hat = X @ beta_hat
residuals = Y - Y_hat

# Standard errors
SSR = np.sum(residuals ** 2)
sigma_hat_sq = SSR / (n - 4)  # 4 parameters (intercept + 3 slopes)
var_beta_hat = sigma_hat_sq * inv(XtX)
SE_beta_hat = np.sqrt(np.diag(var_beta_hat))

# t-statistics and p-values
t_stats = beta_hat / SE_beta_hat
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-4))

# R-squared
SST = np.sum((Y - Y.mean()) ** 2)
R_squared = 1 - SSR / SST

# Adjusted R-squared
k = 3  # Number of slope coefficients
R_squared_adj = 1 - (1 - R_squared) * (n - 1) / (n - k - 1)

# F-statistic (test H0: all slopes = 0)
F_stat = (R_squared / k) / ((1 - R_squared) / (n - k - 1))
F_p_value = 1 - stats.f.cdf(F_stat, k, n - k - 1)

# Display results
print("=" * 80)
print("MULTIPLE REGRESSION: MINCER WAGE EQUATION")
print("=" * 80)
print(f"Model: log(Wage) = Î²â‚€ + Î²â‚Â·Educ + Î²â‚‚Â·Exper + Î²â‚ƒÂ·ExperÂ² + Îµ")
print(f"Sample Size: n = {n}")
print()

print(f"{'Variable':<20} {'True Î²':<12} {'Estimate':<12} {'Std Error':<12} {'t-stat':<10} {'p-value':<10}")
print("-" * 80)
var_names = ['Intercept', 'Education', 'Experience', 'ExperienceÂ²']
true_betas = [beta_0, beta_1, beta_2, beta_3]
for i, name in enumerate(var_names):
    sig = '***' if p_values[i] < 0.01 else '**' if p_values[i] < 0.05 else '*' if p_values[i] < 0.1 else ''
    print(f"{name:<20} {true_betas[i]:<12.4f} {beta_hat[i]:<12.6f} {SE_beta_hat[i]:<12.6f} {t_stats[i]:<10.4f} {p_values[i]:<10.6f} {sig}")

print()
print(f"Goodness of Fit:")
print(f"  RÂ²:                  {R_squared:.6f}  ({R_squared*100:.2f}% explained)")
print(f"  Adjusted RÂ²:         {R_squared_adj:.6f}")
print(f"  Residual Std Error:  {np.sqrt(sigma_hat_sq):.6f}")
print()

print(f"F-test (Hâ‚€: Î²â‚ = Î²â‚‚ = Î²â‚ƒ = 0):")
print(f"  F-statistic:         {F_stat:.4f}")
print(f"  p-value:             {F_p_value:.8f}  ***")
print("=" * 80)

# Interpretation
print(f"\nInterpretation:")
print(f"  â€¢ Education: Each year increases log(wage) by {beta_hat[1]:.4f} â†’ {(np.exp(beta_hat[1])-1)*100:.2f}% wage increase")
print(f"  â€¢ Experience: {beta_hat[2]:.4f} linear term, {beta_hat[3]:.6f} quadratic (diminishing returns)")
print(f"  â€¢ Peak experience: -{beta_hat[2]/(2*beta_hat[3]):.1f} years (after that, marginal return turns negative)")
print(f"  â€¢ 16 years education, 10 years experience:")
log_wage_pred = beta_hat[0] + beta_hat[1]*16 + beta_hat[2]*10 + beta_hat[3]*100
print(f"    Predicted log(wage) = {log_wage_pred:.4f}")
print(f"    Predicted wage = ${np.exp(log_wage_pred):.2f}/hour")
print("=" * 80)

# Compare with simple regression (omitted variable bias)
X_simple = np.column_stack([np.ones(n), education])
beta_simple = inv(X_simple.T @ X_simple) @ (X_simple.T @ Y)
Y_hat_simple = X_simple @ beta_simple
SSR_simple = np.sum((Y - Y_hat_simple) ** 2)
R_squared_simple = 1 - SSR_simple / SST

print(f"\nComparison: Simple vs. Multiple Regression")
print("-" * 80)
print(f"{'Model':<30} {'Î²Ì‚â‚ (Education)':<20} {'RÂ²':<10}")
print("-" * 80)
print(f"{'Simple (omit experience)':<30} {beta_simple[1]:<20.6f} {R_squared_simple:.4f}")
print(f"{'Multiple (control experience)':<30} {beta_hat[1]:<20.6f} {R_squared:.4f}")
print()
bias = beta_simple[1] - beta_hat[1]
print(f"Omitted Variable Bias: {bias:.6f} ({bias/beta_hat[1]*100:+.2f}%)")
print(f"  â†’ Simple regression overestimates education return (experience omitted)")
print("=" * 80)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Education partial effect (holding experience constant at mean)
mean_exp = experience.mean()
educ_range = np.linspace(10, 20, 50)
log_wage_pred_educ = beta_hat[0] + beta_hat[1]*educ_range + beta_hat[2]*mean_exp + beta_hat[3]*mean_exp**2

axes[0,0].scatter(education, log_wage, alpha=0.4, s=20, label='Data')
axes[0,0].plot(educ_range, log_wage_pred_educ, 'r-', linewidth=2, 
               label=f'Partial effect (Exper={mean_exp:.1f})')
axes[0,0].set_xlabel('Education (years)', fontsize=11, fontweight='bold')
axes[0,0].set_ylabel('Log(Wage)', fontsize=11, fontweight='bold')
axes[0,0].set_title('Education Effect (Controlling Experience)', fontsize=12, fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# Experience partial effect (holding education constant at mean)
mean_educ = education.mean()
exp_range = np.linspace(0, 40, 100)
log_wage_pred_exp = beta_hat[0] + beta_hat[1]*mean_educ + beta_hat[2]*exp_range + beta_hat[3]*exp_range**2

axes[0,1].scatter(experience, log_wage, alpha=0.4, s=20, label='Data')
axes[0,1].plot(exp_range, log_wage_pred_exp, 'r-', linewidth=2, 
               label=f'Partial effect (Educ={mean_educ:.1f})')
peak_exp = -beta_hat[2]/(2*beta_hat[3])
axes[0,1].axvline(peak_exp, color='orange', linestyle='--', linewidth=2, label=f'Peak: {peak_exp:.1f} yrs')
axes[0,1].set_xlabel('Experience (years)', fontsize=11, fontweight='bold')
axes[0,1].set_ylabel('Log(Wage)', fontsize=11, fontweight='bold')
axes[0,1].set_title('Experience Effect (Controlling Education)', fontsize=12, fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# Residual plot
axes[1,0].scatter(Y_hat, residuals, alpha=0.4, s=20)
axes[1,0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
axes[1,0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[1,0].set_title('Residual Plot (Homoscedasticity Check)', fontsize=12, fontweight='bold')
axes[1,0].grid(alpha=0.3)

# Q-Q plot (normality check)
stats.probplot(residuals, dist="norm", plot=axes[1,1])
axes[1,1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('multiple_regression_mincer.png', dpi=150)
plt.show()
