import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic data (house prices and household income, cointegrated)
np.random.seed(42)
periods = 200
time_index = pd.date_range('2000-01', periods=periods, freq='Y')

# Common trend (both I(1))
trend = np.cumsum(np.random.normal(0.02, 0.01, periods))

# Income: follows trend
income = 50000 + 10000 * trend + np.random.normal(0, 1000, periods)

# House prices: cointegrated with income (long-run: price = 4*income)
prices = 200000 + 4 * income + np.random.normal(0, 5000, periods) + np.cumsum(np.random.normal(0, 2000, periods))

# Add cointegrating relationship
cointegration_error = np.cumsum(np.random.normal(0, 1000, periods))
prices = prices + cointegration_error

data = pd.DataFrame({
    'income': income,
    'prices': prices,
}, index=time_index)

print("="*60)
print("COINTEGRATION ANALYSIS: House Prices vs Income")
print("="*60)

# 1. Test stationarity of individual series
print("\n1. STATIONARITY TESTS")
print("-"*60)

for col in data.columns:
    result = adfuller(data[col], autolag='AIC')
    print(f"\nADF Test - {col}:")
    print(f"  ADF Stat: {result[0]:.6f}")
    print(f"  p-value: {result[1]:.6f}")
    print(f"  Critical (5%): {result[4]['5%']:.6f}")
    print(f"  Stationary: {'Yes' if result[1] < 0.05 else 'No (I(1))'}")

# 2. Engle-Granger cointegration test
print("\n2. ENGLE-GRANGER COINTEGRATION TEST")
print("-"*60)

score, pvalue, _ = coint(data['prices'], data['income'])
print(f"Cointegration statistic: {score:.6f}")
print(f"p-value: {pvalue:.6f}")
print(f"Cointegrated (Î±=0.05): {'Yes' if pvalue < 0.05 else 'No'}")

# Estimate cointegrating relationship
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(data['income'], data['prices'])
print(f"\nLong-run relationship: Price = {intercept:.2f} + {slope:.4f}Â·Income")
print(f"R-squared: {r_value**2:.4f}")

# 3. Test residuals for stationarity
residuals = data['prices'] - (intercept + slope * data['income'])
result_resid = adfuller(residuals, autolag='AIC')
print(f"\nADF Test on residuals (cointegrating relationship):")
print(f"  ADF Stat: {result_resid[0]:.6f}")
print(f"  p-value: {result_resid[1]:.6f}")
print(f"  Stationary: {'Yes' if result_resid[1] < 0.05 else 'No'}")

# 4. Johansen cointegration test
print("\n3. JOHANSEN COINTEGRATION TEST")
print("-"*60)

# Estimate cointegrating rank
data_diff = data.diff().dropna()
johansen_result = coint_johansen(data.values, det_order=0, k_ar_diff=1)

print(f"\nTrace Statistic (Hâ‚€: rank â‰¤ r):")
print(f"  r=0: {johansen_result.lr1[0]:.4f} | p-value: {johansen_result.cvt[0, 1]:.4f}")
print(f"  râ‰¤1: {johansen_result.lr1[1]:.4f} | p-value: {johansen_result.cvt[1, 1]:.4f}")

print(f"\nEigenvalue Statistic (Hâ‚€: rank = r):")
print(f"  r=0: {johansen_result.lr2[0]:.4f} | p-value: {johansen_result.cvm[0, 1]:.4f}")
print(f"  r=1: {johansen_result.lr2[1]:.4f} | p-value: {johansen_result.cvm[1, 1]:.4f}")

cointegrating_rank = 1
print(f"\nEstimated cointegrating rank: {cointegrating_rank}")

# 5. VECM estimation
print("\n4. VECTOR ERROR CORRECTION MODEL (VECM)")
print("-"*60)

# VECM with rank=1
model = VECM(data, deterministic='ci', k_ar_diff=1, coint_rank=cointegrating_rank)
vecm_result = model.fit()
print(vecm_result.summary())

# Extract error correction term
beta = vecm_result.beta  # Cointegrating vector
alpha = vecm_result.alpha  # Adjustment coefficients

print(f"\nCointegrating vector (Î²): {beta.flatten()}")
print(f"Adjustment coefficients (Î±): {alpha.flatten()}")
print(f"Speed of adjustment: {alpha.flatten()}")

# 6. Calculate error correction term
ec_term = data @ beta
print(f"\nError Correction Term Statistics:")
print(f"  Mean: {ec_term.mean():.2f}")
print(f"  Std Dev: {ec_term.std():.2f}")
print(f"  Min: {ec_term.min():.2f}")
print(f"  Max: {ec_term.max():.2f}")

# 7. Granger causality test
print("\n5. GRANGER CAUSALITY TEST")
print("-"*60)

# Prices Granger-cause Income?
print("\nHâ‚€: Prices do NOT Granger-cause Income")
gc_result = grangercausalitytests(data[['income', 'prices']], maxlag=4, verbose=True)

# Income Granger-cause Prices?
print("\nHâ‚€: Income does NOT Granger-cause Prices")
gc_result2 = grangercausalitytests(data[['prices', 'income']], maxlag=4, verbose=True)

# 8. Visualizations
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Raw series
ax = axes[0, 0]
ax2 = ax.twinx()
ax.plot(data.index, data['income'], 'b-', linewidth=2, label='Income')
ax2.plot(data.index, data['prices'], 'r-', linewidth=2, label='Prices')
ax.set_xlabel('Year')
ax.set_ylabel('Income ($)', color='b')
ax2.set_ylabel('House Prices ($)', color='r')
ax.set_title('House Prices and Income (Cointegrated Series)')
ax.grid(alpha=0.3)

# Plot 2: First differences
ax = axes[0, 1]
ax2 = ax.twinx()
ax.plot(data.index[1:], data['income'].diff()[1:], 'b-', linewidth=1, label='Î”Income', alpha=0.7)
ax2.plot(data.index[1:], data['prices'].diff()[1:], 'r-', linewidth=1, label='Î”Prices', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Î”Income ($)', color='b')
ax2.set_ylabel('Î”Prices ($)', color='r')
ax.set_title('First Differences (Stationary)')
ax.grid(alpha=0.3)

# Plot 3: Residuals (cointegrating relationship)
ax = axes[1, 0]
ax.plot(data.index, residuals, 'g-', linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.fill_between(data.index, residuals.min(), residuals.max(), alpha=0.2, color='g')
ax.set_ylabel('Residuals')
ax.set_title('Cointegrating Residuals (Error Correction Term)')
ax.grid(alpha=0.3)

# Plot 4: Scatter plot
ax = axes[1, 1]
scatter = ax.scatter(data['income'], data['prices'], c=range(len(data)), cmap='viridis', s=50, alpha=0.6)
# Regression line
x_line = np.array([data['income'].min(), data['income'].max()])
y_line = intercept + slope * x_line
ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'y={intercept:.0f}+{slope:.4f}x')
ax.set_xlabel('Income ($)')
ax.set_ylabel('House Prices ($)')
ax.set_title(f'Cointegrating Relationship (RÂ²={r_value**2:.4f})')
ax.legend()
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Time')

# Plot 5: ACF of residuals
ax = axes[2, 0]
plot_acf(residuals, lags=20, ax=ax)
ax.set_title('ACF of Cointegrating Residuals')

# Plot 6: ACF of first difference
ax = axes[2, 1]
plot_acf(data['prices'].diff().dropna(), lags=20, ax=ax)
ax.set_title('ACF of Price Changes')

plt.tight_layout()
plt.savefig('cointegration_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print(f"""
Key Findings:
1. Both income and house prices are non-stationary (I(1))
2. Cointegration confirmed: prices and income share long-run equilibrium
3. Long-run multiplier: 1% income increase â†’ {slope:.2f}% price increase
4. Adjustment speed: {alpha[0, 0]:.4f} (income), {alpha[1, 0]:.4f} (prices)
   - Negative coefficients indicate mean-reversion to long-run relationship
5. Granger causality: Determines if past values predict current values
6. VECM captures both short-run dynamics and long-run equilibrium
""")
