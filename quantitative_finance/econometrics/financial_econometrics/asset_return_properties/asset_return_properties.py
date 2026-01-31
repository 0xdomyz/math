import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model

# Simulate return data (substitute with real data in practice)
np.random.seed(42)
n = 2500  # ~10 years daily

# Generate GARCH(1,1) process with fat tails and leverage effect
from arch.univariate import GARCH, Normal, StudentsT

# Specify GARCH model
garch = arch_model(None, vol='GARCH', p=1, q=1, dist='t')
params = np.array([0.01, 0.08, 0.90])  # omega, alpha, beta (typical values)

# Simulate
sim_data = garch.simulate(params, n, x=None)
returns = sim_data['data'].values * 0.01  # Scale to ~1% daily vol

# Add negative skew (simulate leverage effect)
returns = returns - 0.3 * (returns < 0) * returns**2

# Calculate statistics
mean_ret = returns.mean() * 252 * 100  # Annualized %
std_ret = returns.std() * np.sqrt(252) * 100  # Annualized %
skew = stats.skew(returns)
kurt = stats.kurtosis(returns)  # Excess kurtosis
kurt_total = kurt + 3

print("="*70)
print("Asset Return Properties Analysis")
print("="*70)
print(f"Sample size: {n} observations (~{n/252:.1f} years daily)")
print(f"\nDistributional Statistics:")
print(f"  Mean (annualized): {mean_ret:>8.2f}%")
print(f"  Std Dev (annualized): {std_ret:>8.2f}%")
print(f"  Skewness: {skew:>8.3f} (Normal: 0)")
print(f"  Excess Kurtosis: {kurt:>8.3f} (Normal: 0)")
print(f"  Total Kurtosis: {kurt_total:>8.3f} (Normal: 3)")

# Jarque-Bera test
jb_stat, jb_pval = stats.jarque_bera(returns)
print(f"\nJarque-Bera Test:")
print(f"  JB statistic: {jb_stat:>8.2f}")
print(f"  p-value: {jb_pval:>8.6f}")
print(f"  Result: {'REJECT normality' if jb_pval < 0.05 else 'Cannot reject normality'}")

# Tail events
threshold_3sigma = 3 * returns.std()
threshold_5sigma = 5 * returns.std()

actual_3sigma = np.sum(np.abs(returns) > threshold_3sigma)
expected_3sigma = n * 2 * stats.norm.sf(3)  # Two tails

actual_5sigma = np.sum(np.abs(returns) > threshold_5sigma)
expected_5sigma = n * 2 * stats.norm.sf(5)

print(f"\nTail Event Analysis:")
print(f"  3-sigma events: Actual={actual_3sigma}, Expected(Normal)={expected_3sigma:.1f}")
print(f"  Ratio: {actual_3sigma/expected_3sigma:.1f}Ã— (Fat tails indicator)")
print(f"  5-sigma events: Actual={actual_5sigma}, Expected(Normal)={expected_5sigma:.3f}")
if expected_5sigma > 0:
    print(f"  Ratio: {actual_5sigma/expected_5sigma:.0f}Ã— (Extreme fat tails)")

# Volatility clustering test
returns_squared = returns**2
lb_stat, lb_pval = stats.diagnostic.acorr_ljungbox(returns_squared, lags=[10], return_df=False)
print(f"\nVolatility Clustering (ARCH Effects):")
print(f"  Ljung-Box Q(10) on rÂ²: {lb_stat[0]:>8.2f}")
print(f"  p-value: {lb_pval[0]:>8.6f}")
print(f"  Result: {'Volatility clustering DETECTED' if lb_pval[0] < 0.05 else 'No clustering'}")

# Leverage effect
# Compute correlation between returns and future squared returns
leverage_corr = np.corrcoef(returns[:-1], returns_squared[1:])[0, 1]
print(f"\nLeverage Effect:")
print(f"  Corr(r_t, ÏƒÂ²_{{t+1}}): {leverage_corr:>8.3f}")
print(f"  Result: {'Negative leverage effect' if leverage_corr < -0.1 else 'No leverage effect'}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Return time series
axes[0, 0].plot(returns, linewidth=0.5, color='blue', alpha=0.7)
axes[0, 0].set_title('Return Time Series')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Return')
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[0, 0].grid(alpha=0.3)

# 2. Distribution vs Normal
axes[0, 1].hist(returns, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
x_range = np.linspace(returns.min(), returns.max(), 100)
normal_pdf = stats.norm.pdf(x_range, returns.mean(), returns.std())
t_pdf = stats.t.pdf(x_range, df=6, loc=returns.mean(), scale=returns.std())
axes[0, 1].plot(x_range, normal_pdf, 'r-', linewidth=2, label='Normal')
axes[0, 1].plot(x_range, t_pdf, 'g--', linewidth=2, label='Student-t (df=6)')
axes[0, 1].set_title('Distribution vs Normal')
axes[0, 1].set_xlabel('Return')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. QQ-plot
stats.probplot(returns, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('QQ-Plot vs Normal')
axes[0, 2].grid(alpha=0.3)

# 4. ACF of returns
plot_acf(returns, lags=20, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title('ACF of Returns (Linear Dependence)')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')

# 5. ACF of squared returns (volatility clustering)
plot_acf(returns_squared, lags=20, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('ACF of Squared Returns (Vol Clustering)')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('Autocorrelation')

# 6. Leverage effect scatter
axes[1, 2].scatter(returns[:-1], returns_squared[1:], alpha=0.3, s=10)
axes[1, 2].set_title(f'Leverage Effect (Corr={leverage_corr:.3f})')
axes[1, 2].set_xlabel('Return at t')
axes[1, 2].set_ylabel('Squared Return at t+1')
axes[1, 2].axvline(0, color='black', linestyle='--', linewidth=0.8)
axes[1, 2].grid(alpha=0.3)

# Add regression line
z = np.polyfit(returns[:-1], returns_squared[1:], 1)
p = np.poly1d(z)
x_line = np.linspace(returns.min(), returns.max(), 100)
axes[1, 2].plot(x_line, p(x_line), "r-", linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.5f}')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('asset_return_properties.png', dpi=300, bbox_inches='tight')
plt.show()

# Fit GARCH model
print(f"\n{'='*70}")
print("GARCH(1,1) Model Estimation")
print(f"{'='*70}")

model = arch_model(returns*100, vol='GARCH', p=1, q=1, dist='StudentsT')  # Scale up for numerical stability
results = model.fit(disp='off')
print(results.summary())

# Extract parameters
omega = results.params['omega']
alpha = results.params['alpha[1]']
beta = results.params['beta[1]']
persistence = alpha + beta
half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

print(f"\nGARCH Interpretation:")
print(f"  Persistence (Î±+Î²): {persistence:.4f}")
print(f"  Half-life: {half_life:.1f} days" if persistence < 1 else "  Half-life: Infinite (unit root)")
print(f"  Unconditional variance: {omega/(1-persistence):.4f}" if persistence < 1 else "  Unconditional variance: Undefined")
