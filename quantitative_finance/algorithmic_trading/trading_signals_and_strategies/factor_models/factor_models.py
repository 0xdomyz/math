import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Simulate factor returns (historical approximations)
np.random.seed(42)
n_months = 240  # 20 years

# Generate realistic factor returns
market_ret = np.random.normal(0.006, 0.04, n_months)  # 6% annual, 4% vol
hml_ret = np.random.normal(0.002, 0.08, n_months)    # 2% value premium, 8% vol
smb_ret = np.random.normal(0.001, 0.06, n_months)    # 1% size premium, 6% vol
rmw_ret = np.random.normal(0.002, 0.05, n_months)    # 2% profitability, 5% vol
cma_ret = np.random.normal(0.0025, 0.04, n_months)   # 2.5% investment, 4% vol

# Hypothetical stock returns (mixture of factors + noise)
stock_alpha = 0.0005  # 0.5% monthly alpha (~6% annual)
stock_ret = (stock_alpha + 
             1.0 * market_ret +    # β_m = 1.0
             0.5 * hml_ret +       # β_v = 0.5 (value tilt)
             -0.3 * smb_ret +      # β_s = -0.3 (large-cap tilt)
             0.4 * rmw_ret +       # β_p = 0.4 (quality tilt)
             -0.2 * cma_ret +      # β_i = -0.2 (growth-oriented capex)
             np.random.normal(0, 0.02, n_months))  # idiosyncratic noise

dates = pd.date_range('2004-01-31', periods=n_months, freq='M')

# Combine into DataFrame
ff5_data = pd.DataFrame({
    'market': market_ret,
    'hml': hml_ret,
    'smb': smb_ret,
    'rmw': rmw_ret,
    'cma': cma_ret,
    'stock': stock_ret
}, index=dates)

print("="*100)
print("FAMA-FRENCH 5-FACTOR MODEL ANALYSIS")
print("="*100)

# Step 1: Descriptive statistics
print(f"\nStep 1: Factor & Stock Return Statistics (20 years)")
print(f"-" * 50)

stats_df = pd.DataFrame({
    'Mean (%)': ff5_data.mean() * 100 * 12,  # Annualized
    'Std (%)': ff5_data.std() * 100 * np.sqrt(12),  # Annualized
    'Min (%)': ff5_data.min() * 100,
    'Max (%)': ff5_data.max() * 100,
})

print(stats_df.round(2))

# Step 2: Factor correlation
print(f"\nStep 2: Factor Correlation Matrix")
print(f"-" * 50)

corr_matrix = ff5_data.corr()
print(corr_matrix.round(3))

# Step 3: Regression analysis (FF5 model)
print(f"\nStep 3: Fama-French 5-Factor Regression")
print(f"-" * 50)

X = ff5_data[['market', 'hml', 'smb', 'rmw', 'cma']]
y = ff5_data['stock']

# Add constant
X_with_const = np.column_stack([np.ones(len(X)), X])

# OLS regression
betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
alpha_coef = betas[0]
beta_market, beta_hml, beta_smb, beta_rmw, beta_cma = betas[1:]

# Predictions and residuals
y_pred = X_with_const @ betas
residuals = y - y_pred

# R-squared
ss_total = np.sum((y - y.mean())**2)
ss_residual = np.sum(residuals**2)
r_squared = 1 - (ss_residual / ss_total)

# Adjusted R-squared
n = len(y)
k = 5  # number of factors
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

# Standard errors
residual_std_err = np.sqrt(ss_residual / (n - k - 1))
var_covar_matrix = residual_std_err**2 * np.linalg.inv(X_with_const.T @ X_with_const)
std_errors = np.sqrt(np.diag(var_covar_matrix))

# T-statistics and p-values
from scipy import stats as sp_stats
t_stats = betas / std_errors
p_values = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), n - k - 1))

# Print regression results
results_df = pd.DataFrame({
    'Factor': ['Alpha', 'Market', 'HML', 'SMB', 'RMW', 'CMA'],
    'Coefficient': betas,
    'Std Error': std_errors,
    'T-stat': t_stats,
    'P-value': p_values,
})

print(results_df.to_string(index=False))

print(f"\nModel Fit:")
print(f"  R-squared: {r_squared:.4f}")
print(f"  Adjusted R-squared: {adj_r_squared:.4f}")
print(f"  Residual Std Error: {residual_std_err * 100:.3f}%/month ({residual_std_err * np.sqrt(12) * 100:.2f}%/year)")

# Annualized alpha
alpha_annual = alpha_coef * 12 * 100
print(f"  Annualized Alpha: {alpha_annual:.2f}%")

# Step 4: Performance decomposition
print(f"\nStep 4: Return Decomposition (Annualized)")
print(f"-" * 50)

# Average factor returns and contributions
factor_means = X.mean()
factor_contributions = (betas[1:] * factor_means).values * 12 * 100

decomp_df = pd.DataFrame({
    'Factor': ['Alpha', 'Market', 'HML', 'SMB', 'RMW', 'CMA'],
    'Beta': [alpha_coef, beta_market, beta_hml, beta_smb, beta_rmw, beta_cma],
    'Factor Return': [0, factor_means['market']*12*100, factor_means['hml']*12*100, 
                      factor_means['smb']*12*100, factor_means['rmw']*12*100, factor_means['cma']*12*100],
    'Contribution (%)': [alpha_annual] + list(factor_contributions),
})

print(decomp_df.to_string(index=False))

total_return = y.mean() * 12 * 100
decomp_total = decomp_df['Contribution (%)'].sum()
print(f"\nTotal Stock Return: {total_return:.2f}%")
print(f"Explained (decomposition): {decomp_total:.2f}%")

# Step 5: Rolling window analysis
print(f"\nStep 5: Rolling Factor Exposures (24-month rolling window)")
print(f"-" * 50)

window = 24
rolling_betas = {}

for factor in ['market', 'hml', 'smb', 'rmw', 'cma']:
    betas_rolling = []
    for i in range(len(ff5_data) - window):
        X_window = ff5_data[['market', 'hml', 'smb', 'rmw', 'cma']].iloc[i:i+window]
        y_window = ff5_data['stock'].iloc[i:i+window]
        X_window_const = np.column_stack([np.ones(window), X_window])
        betas_window = np.linalg.lstsq(X_window_const, y_window, rcond=None)[0]
        
        if factor == 'market':
            betas_rolling.append(betas_window[1])
        elif factor == 'hml':
            betas_rolling.append(betas_window[2])
        elif factor == 'smb':
            betas_rolling.append(betas_window[3])
        elif factor == 'rmw':
            betas_rolling.append(betas_window[4])
        elif factor == 'cma':
            betas_rolling.append(betas_window[5])
    
    rolling_betas[factor] = betas_rolling

rolling_dates = dates[window:]

print(f"Average β (across rolling windows):")
for factor, betas_r in rolling_betas.items():
    print(f"  {factor.upper()}: {np.mean(betas_r):.3f} (±{np.std(betas_r):.3f})")

# VISUALIZATION
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: Cumulative returns
ax = axes[0, 0]
ax.plot(dates, np.exp(np.log(1 + ff5_data['stock']).cumsum()), label='Stock', linewidth=2)
ax.plot(dates, np.exp(np.log(1 + ff5_data['market']).cumsum()), label='Market Factor', alpha=0.7)
ax.set_title('Cumulative Returns: Stock vs Market')
ax.set_ylabel('Cumulative Return (log scale)')
ax.legend()
ax.set_yscale('log')
ax.grid(alpha=0.3)

# Plot 2: Factor correlation heatmap
ax = axes[0, 1]
im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(['Market', 'HML', 'SMB', 'RMW', 'CMA', 'Stock'], rotation=45)
ax.set_yticklabels(['Market', 'HML', 'SMB', 'RMW', 'CMA', 'Stock'])
ax.set_title('Factor Correlation Matrix')
plt.colorbar(im, ax=ax)

# Plot 3: Regression fit (actual vs predicted)
ax = axes[1, 0]
ax.scatter(y_pred * 100, y * 100, alpha=0.5, s=20)
ax.plot([-5, 15], [-5, 15], 'r--', linewidth=2)
ax.set_xlabel('Predicted Return (%)')
ax.set_ylabel('Actual Return (%)')
ax.set_title(f'FF5 Fit (R² = {r_squared:.4f})')
ax.grid(alpha=0.3)

# Plot 4: Residuals
ax = axes[1, 1]
ax.hist(residuals * 100, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Residual (%)')
ax.set_ylabel('Frequency')
ax.set_title(f'Residual Distribution (σ = {residual_std_err*100:.2f}%/month)')
ax.grid(alpha=0.3, axis='y')

# Plot 5: Rolling betas
ax = axes[2, 0]
ax.plot(rolling_dates, rolling_betas['market'], label='Market β', linewidth=1.5)
ax.plot(rolling_dates, rolling_betas['hml'], label='HML β', linewidth=1.5)
ax.plot(rolling_dates, rolling_betas['smb'], label='SMB β', linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_title('Rolling Factor Exposures (24-month window)')
ax.set_ylabel('Beta')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Return contribution stacked
ax = axes[2, 1]
contributions = [alpha_annual] + list(factor_contributions)
labels = ['Alpha', 'Market', 'HML', 'SMB', 'RMW', 'CMA']
colors = ['gray', 'blue', 'green', 'red', 'orange', 'purple']
bars = ax.bar(labels, contributions, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Contribution to Return (%/year)')
ax.set_title('Return Decomposition by Factor')
ax.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, contributions):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top')

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("INSIGHTS")
print(f"="*100)
print(f"- FF5 explains {r_squared*100:.1f}% of stock return variance")
print(f"- Key exposures: Market β={beta_market:.2f}, Value β={beta_hml:.2f}, Quality β={beta_rmw:.2f}")
print(f"- Annualized alpha: {alpha_annual:.2f}% (may be exploitable if >2%)")
print(f"- Factor betas vary over time (rolling window shows instability)")
print(f"- Residuals approximately normal (model reasonably specified)")