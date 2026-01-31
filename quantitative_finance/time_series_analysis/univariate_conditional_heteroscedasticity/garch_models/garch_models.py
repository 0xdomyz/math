from arch import arch_model
from arch.univariate import GARCH, EGARCH, ConstantMean, ZeroMean
from scipy import stats
from scipy.stats import chi2
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("GARCH MODELS AND CONDITIONAL HETEROSCEDASTICITY")
print("="*80)

# Generate synthetic GARCH(1,1) data
np.random.seed(42)
n = 2000

# Parameters for GARCH(1,1)
omega = 0.01
alpha = 0.10
beta = 0.85
persistence = alpha + beta

print(f"\nTrue GARCH(1,1) Parameters:")
print(f"  ω (omega): {omega:.3f}")
print(f"  α (alpha): {alpha:.3f}")
print(f"  β (beta): {beta:.3f}")
print(f"  Persistence (α+β): {persistence:.3f}")
print(f"  Unconditional Variance: {omega/(1-persistence):.3f}")
print(f"  Half-life: {np.log(0.5)/np.log(persistence):.1f} periods")

# Simulate GARCH(1,1)
returns = np.zeros(n)
sigma2 = np.zeros(n)
sigma2[0] = omega / (1 - persistence)  # Start at unconditional variance

for t in range(1, n):
    sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    returns[t] = np.sqrt(sigma2[t]) * np.random.normal()

# Create time series
dates = pd.date_range('2010-01-01', periods=n, freq='D')
returns_series = pd.Series(returns, index=dates, name='Returns')
true_vol = pd.Series(np.sqrt(sigma2), index=dates, name='True_Volatility')

print(f"\nSimulated Series Statistics:")
print(f"  Mean: {returns.mean():.4f}")
print(f"  Std Dev: {returns.std():.4f}")
print(f"  Skewness: {stats.skew(returns):.4f}")
print(f"  Kurtosis (excess): {stats.kurtosis(returns):.4f}")
print(f"  Min: {returns.min():.4f}")
print(f"  Max: {returns.max():.4f}")

# Test for ARCH effects
print("\n" + "="*80)
print("ARCH EFFECTS DETECTION")
print("="*80)

squared_returns = returns_series ** 2
lb_test = acorr_ljungbox(squared_returns, lags=10, return_df=True)

print("\nLjung-Box Test on Squared Returns:")
print(lb_test)
print(f"\nInterpretation: P-values < 0.05 indicate ARCH effects present")
print(f"Result: {'ARCH effects detected' if (lb_test['lb_pvalue'] < 0.05).any() else 'No ARCH effects'}")

# GARCH(1,1) Estimation
print("\n" + "="*80)
print("GARCH(1,1) MODEL ESTIMATION")
print("="*80)

# Fit GARCH(1,1) with normal distribution
garch11 = arch_model(returns_series, vol='Garch', p=1, q=1, dist='normal')
garch11_fitted = garch11.fit(disp='off')

print(garch11_fitted.summary())

# Extract parameters
omega_est = garch11_fitted.params['omega']
alpha_est = garch11_fitted.params['alpha[1]']
beta_est = garch11_fitted.params['beta[1]']
persistence_est = alpha_est + beta_est

print(f"\nEstimated vs True Parameters:")
print(f"  ω: Estimated={omega_est:.4f}, True={omega:.4f}")
print(f"  α: Estimated={alpha_est:.4f}, True={alpha:.4f}")
print(f"  β: Estimated={beta_est:.4f}, True={beta:.4f}")
print(f"  Persistence: Estimated={persistence_est:.4f}, True={persistence:.4f}")

# GARCH(1,1) with Student-t distribution
print("\n" + "="*80)
print("GARCH(1,1) WITH STUDENT-T DISTRIBUTION")
print("="*80)

garch11_t = arch_model(returns_series, vol='Garch', p=1, q=1, dist='t')
garch11_t_fitted = garch11_t.fit(disp='off')

nu = garch11_t_fitted.params['nu']
print(f"\nStudent-t degrees of freedom: {nu:.2f}")
print(f"Model Comparison:")
print(f"  Normal GARCH AIC: {garch11_fitted.aic:.2f}")
print(f"  Student-t GARCH AIC: {garch11_t_fitted.aic:.2f}")
print(f"  Preferred: {'Student-t' if garch11_t_fitted.aic < garch11_fitted.aic else 'Normal'}")

# GJR-GARCH (Asymmetric)
print("\n" + "="*80)
print("GJR-GARCH (ASYMMETRIC LEVERAGE EFFECT)")
print("="*80)

# Generate asymmetric data
returns_asym = np.zeros(n)
sigma2_asym = np.zeros(n)
sigma2_asym[0] = 0.01
gamma = 0.08  # Asymmetry parameter

for t in range(1, n):
    indicator = 1 if returns_asym[t-1] < 0 else 0
    sigma2_asym[t] = omega + alpha * returns_asym[t-1]**2 + gamma * indicator * returns_asym[t-1]**2 + beta * sigma2_asym[t-1]
    returns_asym[t] = np.sqrt(sigma2_asym[t]) * np.random.normal()

returns_asym_series = pd.Series(returns_asym, index=dates, name='Asymmetric_Returns')

# Fit GJR-GARCH
gjr_garch = arch_model(returns_asym_series, vol='Garch', p=1, o=1, q=1, dist='normal')
gjr_fitted = gjr_garch.fit(disp='off')

print(gjr_fitted.summary())

gamma_est = gjr_fitted.params['gamma[1]']
print(f"\nAsymmetry Parameter (γ): {gamma_est:.4f}")
print(f"Interpretation: {'Leverage effect present' if gamma_est > 0 else 'No asymmetry'}")
print(f"  Negative shock impact: {alpha_est + gamma_est:.4f}")
print(f"  Positive shock impact: {alpha_est:.4f}")

# EGARCH
print("\n" + "="*80)
print("EGARCH (EXPONENTIAL GARCH)")
print("="*80)

egarch = arch_model(returns_asym_series, vol='EGARCH', p=1, q=1, dist='normal')
egarch_fitted = egarch.fit(disp='off')

print(egarch_fitted.summary())

# Diagnostics
print("\n" + "="*80)
print("MODEL DIAGNOSTICS")
print("="*80)

# Standardized residuals
std_resid = garch11_fitted.std_resid

print("\n1. STANDARDIZED RESIDUALS (Should be i.i.d.)")
print("-" * 40)
lb_std = acorr_ljungbox(std_resid, lags=10, return_df=True)
print(f"Ljung-Box test p-value (lag 10): {lb_std['lb_pvalue'].iloc[-1]:.4f}")
print(f"Result: {'PASS (No autocorrelation)' if lb_std['lb_pvalue'].iloc[-1] > 0.05 else 'FAIL'}")

print("\n2. STANDARDIZED SQUARED RESIDUALS")
print("-" * 40)
lb_std_sq = acorr_ljungbox(std_resid**2, lags=10, return_df=True)
print(f"Ljung-Box test p-value (lag 10): {lb_std_sq['lb_pvalue'].iloc[-1]:.4f}")
print(f"Result: {'PASS (No remaining ARCH)' if lb_std_sq['lb_pvalue'].iloc[-1] > 0.05 else 'FAIL'}")

print("\n3. NORMALITY TEST")
print("-" * 40)
jb_stat, jb_pvalue = stats.jarque_bera(std_resid)
print(f"Jarque-Bera statistic: {jb_stat:.4f}")
print(f"P-value: {jb_pvalue:.4f}")
print(f"Result: {'Normal' if jb_pvalue > 0.05 else 'Non-normal (consider t-distribution)'}")

# Forecasting
print("\n" + "="*80)
print("VOLATILITY FORECASTING")
print("="*80)

# Out-of-sample forecast
train_size = 1800
test_size = n - train_size
returns_train = returns_series[:train_size]
returns_test = returns_series[train_size:]

# Fit on training data
garch_train = arch_model(returns_train, vol='Garch', p=1, q=1, dist='normal')
garch_train_fitted = garch_train.fit(disp='off')

# Forecast volatility
forecast_horizon = test_size
forecasts = garch_train_fitted.forecast(horizon=forecast_horizon, reindex=False)
forecast_variance = forecasts.variance.values[-1, :]  # Last row = forecast from train end
forecast_volatility = np.sqrt(forecast_variance)

# Realized volatility (proxy: absolute returns)
realized_vol = np.abs(returns_test.values)

# Forecast evaluation
mae_vol = np.mean(np.abs(realized_vol - forecast_volatility))
rmse_vol = np.sqrt(np.mean((realized_vol - forecast_volatility)**2))

print(f"\nVolatility Forecast Evaluation:")
print(f"  Training period: {train_size} days")
print(f"  Test period: {test_size} days")
print(f"  Forecast Horizon: {forecast_horizon} days")
print(f"\nAccuracy Metrics:")
print(f"  MAE: {mae_vol:.6f}")
print(f"  RMSE: {rmse_vol:.6f}")

# VaR Calculation
print("\n" + "="*80)
print("VALUE-AT-RISK (VaR) CALCULATION")
print("="*80)

confidence_level = 0.95
quantile = stats.norm.ppf(1 - confidence_level)

# 1-day VaR
fitted_vol = garch11_fitted.conditional_volatility
var_95 = quantile * fitted_vol  # For long position (negative means loss)

# Backtesting
violations = returns_series < var_95
violation_rate = violations.sum() / len(returns_series)
expected_rate = 1 - confidence_level

print(f"\n95% VaR Backtesting:")
print(f"  Expected violation rate: {expected_rate:.2%}")
print(f"  Actual violation rate: {violation_rate:.2%}")
print(f"  Number of violations: {violations.sum()} out of {len(returns_series)}")
print(f"  Result: {'PASS' if abs(violation_rate - expected_rate) < 0.01 else 'CHECK (potential model issue)'}")

# Kupiec POF Test
n_obs = len(returns_series)
n_violations = violations.sum()
lr_stat = -2 * (np.log((1-expected_rate)**(n_obs-n_violations) * expected_rate**n_violations) -
                np.log((1-violation_rate)**(n_obs-n_violations) * violation_rate**n_violations))
lr_pvalue = 1 - chi2.cdf(lr_stat, df=1)

print(f"\nKupiec POF Test:")
print(f"  LR statistic: {lr_stat:.4f}")
print(f"  P-value: {lr_pvalue:.4f}")
print(f"  Result: {'PASS (Correct coverage)' if lr_pvalue > 0.05 else 'FAIL (Incorrect coverage)'}")

# Visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot 1: Returns with volatility clustering
ax = axes[0, 0]
ax.plot(returns_series, linewidth=0.8, alpha=0.7)
ax.set_title('Simulated Returns (Volatility Clustering)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Returns')
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)

# Plot 2: Squared returns (ARCH effects)
ax = axes[0, 1]
ax.plot(squared_returns, linewidth=0.8, alpha=0.7, color='orange')
ax.set_title('Squared Returns (ARCH Effects)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Squared Returns')
ax.grid(alpha=0.3)

# Plot 3: ACF of squared returns
ax = axes[0, 2]
plot_acf(squared_returns, lags=30, ax=ax, alpha=0.05)
ax.set_title('ACF of Squared Returns', fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: True vs Estimated Volatility
ax = axes[1, 0]
ax.plot(true_vol, label='True σ', linewidth=1.5, alpha=0.7)
ax.plot(garch11_fitted.conditional_volatility, label='Estimated σ (GARCH)', 
        linewidth=1.5, alpha=0.7, linestyle='--')
ax.set_title('True vs Estimated Conditional Volatility', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Standardized residuals
ax = axes[1, 1]
ax.plot(std_resid, linewidth=0.8, alpha=0.7, color='green')
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.axhline(2, color='orange', linestyle='--', alpha=0.5)
ax.axhline(-2, color='orange', linestyle='--', alpha=0.5)
ax.set_title('Standardized Residuals', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('z-score')
ax.grid(alpha=0.3)

# Plot 6: Q-Q plot
ax = axes[1, 2]
stats.probplot(std_resid.dropna(), dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
ax.grid(alpha=0.3)

# Plot 7: ACF of standardized residuals
ax = axes[2, 0]
plot_acf(std_resid.dropna(), lags=30, ax=ax, alpha=0.05)
ax.set_title('ACF of Standardized Residuals', fontweight='bold')
ax.grid(alpha=0.3)

# Plot 8: Volatility forecast
ax = axes[2, 1]
test_dates = returns_test.index
ax.plot(test_dates, realized_vol, label='Realized |r|', linewidth=1.5, alpha=0.7)
ax.plot(test_dates, forecast_volatility, label='GARCH Forecast', 
        linewidth=2, linestyle='--', color='red')
ax.set_title('Out-of-Sample Volatility Forecast', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility')
ax.legend()
ax.grid(alpha=0.3)

# Plot 9: VaR violations
ax = axes[2, 2]
ax.plot(returns_series, linewidth=0.8, alpha=0.6, label='Returns')
ax.plot(var_95, linewidth=1.5, color='red', linestyle='--', label='95% VaR', alpha=0.7)
ax.scatter(returns_series[violations].index, returns_series[violations], 
          color='red', s=20, marker='x', label=f'Violations ({violations.sum()})', zorder=5)
ax.set_title('VaR Backtesting', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Returns')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Model comparison table
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

comparison = pd.DataFrame({
    'Model': ['GARCH(1,1) Normal', 'GARCH(1,1) Student-t', 'GJR-GARCH', 'EGARCH'],
    'AIC': [garch11_fitted.aic, garch11_t_fitted.aic, gjr_fitted.aic, egarch_fitted.aic],
    'BIC': [garch11_fitted.bic, garch11_t_fitted.bic, gjr_fitted.bic, egarch_fitted.bic],
    'Log-Likelihood': [garch11_fitted.loglikelihood, garch11_t_fitted.loglikelihood, 
                      gjr_fitted.loglikelihood, egarch_fitted.loglikelihood]
})

print("\n" + comparison.to_string(index=False))

best_model = comparison.loc[comparison['AIC'].idxmin(), 'Model']
print(f"\nBest Model (by AIC): {best_model}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. GARCH(1,1) captures {persistence_est:.1%} volatility persistence (α+β)")
print(f"2. Student-t distribution fits better for fat tails (ν={nu:.1f})")
print(f"3. Leverage effect: GJR-GARCH γ={gamma_est:.3f} (negative shocks increase vol)")
print(f"4. Volatility clustering evident in ACF of squared returns")
print(f"5. VaR backtesting: {violation_rate:.2%} actual vs {expected_rate:.2%} expected")
print(f"6. Half-life of volatility shocks: ~{np.log(0.5)/np.log(persistence_est):.0f} days")
print(f"7. GARCH generates fat tails (kurtosis={stats.kurtosis(returns):.2f}) even with normal innovations")
