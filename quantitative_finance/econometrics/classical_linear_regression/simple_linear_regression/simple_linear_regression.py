import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Generate synthetic data
np.random.seed(42)
n = 200  # Sample size

# True population parameters
beta_0_true = 10.0  # Intercept: $10/hr wage at 0 years education
beta_1_true = 1.5   # Slope: $1.50/hr per year of education
sigma = 3.0         # Error standard deviation

# Generate education (X) and wage (Y)
education = np.random.uniform(8, 20, n)  # 8-20 years
epsilon = np.random.normal(0, sigma, n)  # Random errors
wage = beta_0_true + beta_1_true * education + epsilon

# Create dataframe
data = pd.DataFrame({'education': education, 'wage': wage})

# OLS estimation (manual calculation)
X_bar = data['education'].mean()
Y_bar = data['wage'].mean()

# Slope: beta_1 = Cov(X,Y) / Var(X)
cov_XY = ((data['education'] - X_bar) * (data['wage'] - Y_bar)).sum() / n
var_X = ((data['education'] - X_bar) ** 2).sum() / n
beta_1_hat = cov_XY / var_X

# Intercept: beta_0 = Y_bar - beta_1 * X_bar
beta_0_hat = Y_bar - beta_1_hat * X_bar

# Predictions and residuals
Y_hat = beta_0_hat + beta_1_hat * data['education']
residuals = data['wage'] - Y_hat

# Standard error of residuals
SSR = (residuals ** 2).sum()
sigma_hat_sq = SSR / (n - 2)
sigma_hat = np.sqrt(sigma_hat_sq)

# Standard error of slope
SE_beta_1 = np.sqrt(sigma_hat_sq / ((data['education'] - X_bar) ** 2).sum())

# t-statistic and p-value
t_stat = beta_1_hat / SE_beta_1
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

# Confidence interval (95%)
t_critical = stats.t.ppf(0.975, df=n-2)
CI_lower = beta_1_hat - t_critical * SE_beta_1
CI_upper = beta_1_hat + t_critical * SE_beta_1

# R-squared
SST = ((data['wage'] - Y_bar) ** 2).sum()
R_squared = 1 - SSR / SST

# Display results
print("=" * 70)
print("SIMPLE LINEAR REGRESSION: WAGE vs. EDUCATION")
print("=" * 70)
print(f"Sample Size:              n = {n}")
print(f"\nTrue Parameters:")
print(f"  Î²â‚€ (Intercept):         ${beta_0_true:.2f}")
print(f"  Î²â‚ (Slope):             ${beta_1_true:.2f} per year")
print(f"\nOLS Estimates:")
print(f"  Î²Ì‚â‚€ (Intercept):         ${beta_0_hat:.4f}")
print(f"  Î²Ì‚â‚ (Slope):             ${beta_1_hat:.4f} per year")
print(f"  SE(Î²Ì‚â‚):                 ${SE_beta_1:.4f}")
print(f"\nHypothesis Test (Hâ‚€: Î²â‚ = 0):")
print(f"  t-statistic:            {t_stat:.4f}")
print(f"  p-value:                {p_value:.6f}  {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else 'Not significant'}")
print(f"\n95% Confidence Interval for Î²â‚:")
print(f"  [{CI_lower:.4f}, {CI_upper:.4f}]")
print(f"\nGoodness of Fit:")
print(f"  RÂ²:                     {R_squared:.4f}  ({R_squared*100:.2f}% of wage variation explained)")
print(f"  Residual Std Error:     ${sigma_hat:.4f}")
print("=" * 70)

# Interpretation
print(f"\nInterpretation:")
print(f"  â€¢ Each additional year of education associated with ${beta_1_hat:.2f}/hr wage increase")
print(f"  â€¢ 16 years education â†’ Expected wage = ${beta_0_hat + beta_1_hat*16:.2f}/hr")
print(f"  â€¢ Effect is statistically significant (p < 0.001)")
print(f"  â€¢ Education explains {R_squared*100:.1f}% of wage differences")
print("=" * 70)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot with regression line
axes[0].scatter(data['education'], data['wage'], alpha=0.6, s=50, edgecolors='k', label='Data')
x_line = np.linspace(data['education'].min(), data['education'].max(), 100)
y_line = beta_0_hat + beta_1_hat * x_line
axes[0].plot(x_line, y_line, 'r-', linewidth=2, label=f'Å¶ = {beta_0_hat:.2f} + {beta_1_hat:.2f}X')
axes[0].set_xlabel('Education (years)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Wage ($/hour)', fontsize=12, fontweight='bold')
axes[0].set_title('Wage vs. Education: Simple Linear Regression', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(alpha=0.3)

# Residual plot
axes[1].scatter(Y_hat, residuals, alpha=0.6, s=50, edgecolors='k')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2, label='Zero residual line')
axes[1].set_xlabel('Fitted Values (Å¶)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Residuals (e)', fontsize=12, fontweight='bold')
axes[1].set_title('Residual Plot (Homoscedasticity Check)', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('simple_regression_wage_education.png', dpi=150)
plt.show()
