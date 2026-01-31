import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta

# ===== Download S&P 500 Data =====
print("="*80)
print("GARCH MODELS FOR VOLATILITY")
print("="*80)

# Download 5 years of S&P 500 data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print(f"\nDownloading S&P 500 data from {start_date.date()} to {end_date.date()}...")
data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)

# Calculate returns
returns = 100 * data['Adj Close'].pct_change().dropna()
returns = returns.values

print(f"âœ“ Downloaded {len(returns)} daily returns")

# Summary statistics
print(f"\nReturn Statistics:")
print(f"  Mean: {returns.mean():.4f}%")
print(f"  Std Dev: {returns.std():.4f}%")
print(f"  Skewness: {stats.skew(returns):.4f}")
print(f"  Kurtosis: {stats.kurtosis(returns):.4f}")
print(f"  Min: {returns.min():.4f}%")
print(f"  Max: {returns.max():.4f}%")

# ===== Test for ARCH Effects =====
print("\n" + "="*80)
print("ARCH EFFECTS TEST")
print("="*80)

# Ljung-Box test on squared returns
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(returns**2, lags=[10], return_df=True)
print(f"Ljung-Box test on squared returns (lag 10):")
print(f"  Test Statistic: {lb_test['lb_stat'].values[0]:.2f}")
print(f"  P-value: {lb_test['lb_pvalue'].values[0]:.4f}")

if lb_test['lb_pvalue'].values[0] < 0.05:
    print("  âœ“ Reject null of no ARCH effects (GARCH appropriate)")
else:
    print("  âœ— No evidence of ARCH effects")

# ===== Split Data =====
train_size = int(0.8 * len(returns))
returns_train = returns[:train_size]
returns_test = returns[train_size:]

print(f"\nTrain/Test Split:")
print(f"  Training: {len(returns_train)} observations")
print(f"  Testing: {len(returns_test)} observations")

# ===== GARCH(1,1) Model =====
print("\n" + "="*80)
print("GARCH(1,1) MODEL")
print("="*80)

# Fit GARCH(1,1) with normal distribution
garch_model = arch_model(returns_train, vol='Garch', p=1, q=1, dist='normal')
garch_fit = garch_model.fit(disp='off')

print(garch_fit.summary())

# Extract parameters
omega = garch_fit.params['omega']
alpha = garch_fit.params['alpha[1]']
beta = garch_fit.params['beta[1]']

print(f"\nGARCH(1,1) Parameters:")
print(f"  Ï‰ (omega): {omega:.6f}")
print(f"  Î± (alpha): {alpha:.6f}")
print(f"  Î² (beta): {beta:.6f}")
print(f"  Î± + Î²: {alpha + beta:.6f} (persistence)")

# Unconditional variance
uncond_var = omega / (1 - alpha - beta)
print(f"  Unconditional variance: {uncond_var:.6f}")
print(f"  Unconditional vol: {np.sqrt(uncond_var):.4f}%")

# Half-life of shocks
half_life = np.log(0.5) / np.log(alpha + beta)
print(f"  Half-life of shocks: {half_life:.1f} days")

# ===== GJR-GARCH (Threshold GARCH) =====
print("\n" + "="*80)
print("GJR-GARCH MODEL (ASYMMETRIC)")
print("="*80)

gjr_model = arch_model(returns_train, vol='Garch', p=1, o=1, q=1, dist='normal')
gjr_fit = gjr_model.fit(disp='off')

print(gjr_fit.summary())

gamma = gjr_fit.params['gamma[1]']
print(f"\nLeverage Parameter:")
print(f"  Î³ (gamma): {gamma:.6f}")

if gamma > 0:
    print(f"  âœ“ Negative shocks increase volatility more (leverage effect)")
else:
    print(f"  âœ— No significant leverage effect")

# ===== EGARCH Model =====
print("\n" + "="*80)
print("EGARCH MODEL (EXPONENTIAL)")
print("="*80)

egarch_model = arch_model(returns_train, vol='EGARCH', p=1, q=1, dist='normal')
egarch_fit = egarch_model.fit(disp='off')

print(egarch_fit.summary())

# ===== GARCH with Student-t Distribution =====
print("\n" + "="*80)
print("GARCH(1,1) WITH STUDENT-T DISTRIBUTION")
print("="*80)

garch_t_model = arch_model(returns_train, vol='Garch', p=1, q=1, dist='t')
garch_t_fit = garch_t_model.fit(disp='off')

print(garch_t_fit.summary())

df_param = garch_t_fit.params['nu']
print(f"\nDegrees of Freedom: {df_param:.2f}")
print(f"  Î½ < 5: Very heavy tails")
print(f"  Î½ â‰ˆ 5-10: Heavy tails (typical for daily returns)")
print(f"  Î½ > 30: Close to normal")

# ===== Model Comparison =====
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

models_dict = {
    'GARCH(1,1)': garch_fit,
    'GJR-GARCH': gjr_fit,
    'EGARCH': egarch_fit,
    'GARCH-t': garch_t_fit
}

comparison_df = pd.DataFrame({
    'Model': list(models_dict.keys()),
    'LogLik': [m.loglikelihood for m in models_dict.values()],
    'AIC': [m.aic for m in models_dict.values()],
    'BIC': [m.bic for m in models_dict.values()],
    'Params': [len(m.params) for m in models_dict.values()]
})

comparison_df = comparison_df.sort_values('AIC')
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
print(f"\nâœ“ Best model by AIC: {best_model_name}")

# ===== Diagnostics =====
print("\n" + "="*80)
print("DIAGNOSTICS (GARCH(1,1))")
print("="*80)

# Standardized residuals
std_resid = garch_fit.std_resid

print(f"Standardized Residuals:")
print(f"  Mean: {std_resid.mean():.4f} (should be â‰ˆ0)")
print(f"  Std: {std_resid.std():.4f} (should be â‰ˆ1)")
print(f"  Skewness: {stats.skew(std_resid):.4f}")
print(f"  Kurtosis: {stats.kurtosis(std_resid):.4f}")

# Test on standardized residuals
lb_test_std = acorr_ljungbox(std_resid, lags=[10], return_df=True)
print(f"\nLjung-Box test on standardized residuals:")
print(f"  P-value: {lb_test_std['lb_pvalue'].values[0]:.4f}")
if lb_test_std['lb_pvalue'].values[0] > 0.05:
    print("  âœ“ No autocorrelation remaining")

# Test on squared standardized residuals
lb_test_std_sq = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
print(f"\nLjung-Box test on squared standardized residuals:")
print(f"  P-value: {lb_test_std_sq['lb_pvalue'].values[0]:.4f}")
if lb_test_std_sq['lb_pvalue'].values[0] > 0.05:
    print("  âœ“ No ARCH effects remaining")

# ===== Out-of-Sample Forecasting =====
print("\n" + "="*80)
print("OUT-OF-SAMPLE FORECASTING")
print("="*80)

# Rolling 1-step ahead forecasts
forecasts_garch = []
forecasts_gjr = []
realized_vol = []

print("Computing rolling forecasts...")

for t in range(len(returns_test)):
    # Expand training window
    returns_expanding = returns[:train_size + t]
    
    # Fit GARCH(1,1)
    model_garch = arch_model(returns_expanding, vol='Garch', p=1, q=1, dist='normal')
    fit_garch = model_garch.fit(disp='off')
    forecast_garch = fit_garch.forecast(horizon=1)
    forecasts_garch.append(np.sqrt(forecast_garch.variance.values[-1, 0]))
    
    # Fit GJR-GARCH
    model_gjr = arch_model(returns_expanding, vol='Garch', p=1, o=1, q=1, dist='normal')
    fit_gjr = model_gjr.fit(disp='off')
    forecast_gjr = fit_gjr.forecast(horizon=1)
    forecasts_gjr.append(np.sqrt(forecast_gjr.variance.values[-1, 0]))
    
    # Realized (proxy: absolute return)
    realized_vol.append(np.abs(returns_test[t]))
    
    if (t+1) % 50 == 0:
        print(f"  Processed {t+1}/{len(returns_test)} forecasts")

forecasts_garch = np.array(forecasts_garch)
forecasts_gjr = np.array(forecasts_gjr)
realized_vol = np.array(realized_vol)

# Forecast evaluation
mse_garch = np.mean((forecasts_garch - realized_vol)**2)
mse_gjr = np.mean((forecasts_gjr - realized_vol)**2)

mae_garch = np.mean(np.abs(forecasts_garch - realized_vol))
mae_gjr = np.mean(np.abs(forecasts_gjr - realized_vol))

# QLIKE loss
qlike_garch = np.mean(np.log(forecasts_garch**2) + (returns_test**2) / (forecasts_garch**2))
qlike_gjr = np.mean(np.log(forecasts_gjr**2) + (returns_test**2) / (forecasts_gjr**2))

print(f"\nâœ“ Forecast evaluation:")
print(f"GARCH(1,1):")
print(f"  MSE: {mse_garch:.6f}")
print(f"  MAE: {mae_garch:.4f}")
print(f"  QLIKE: {qlike_garch:.6f}")

print(f"\nGJR-GARCH:")
print(f"  MSE: {mse_gjr:.6f}")
print(f"  MAE: {mae_gjr:.4f}")
print(f"  QLIKE: {qlike_gjr:.6f}")

if qlike_gjr < qlike_garch:
    improvement = (qlike_garch - qlike_gjr) / qlike_garch * 100
    print(f"\nâœ“ GJR-GARCH improves over GARCH by {improvement:.2f}% (QLIKE)")

# ===== Visualizations =====
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: Returns with conditional volatility
cond_vol = garch_fit.conditional_volatility
dates_train = pd.date_range(end=end_date, periods=len(returns_train))

axes[0, 0].plot(dates_train, returns_train, linewidth=0.5, alpha=0.6, label='Returns')
axes[0, 0].fill_between(dates_train, -2*cond_vol, 2*cond_vol, 
                        alpha=0.3, color='red', label='Â±2Ïƒâ‚œ')
axes[0, 0].set_ylabel('Returns (%)')
axes[0, 0].set_title('Returns with GARCH(1,1) Conditional Volatility')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Conditional volatility over time
axes[0, 1].plot(dates_train, cond_vol, linewidth=1.5, color='red')
axes[0, 1].axhline(np.sqrt(uncond_var), color='blue', linestyle='--', 
                  linewidth=2, label='Unconditional Ïƒ')
axes[0, 1].set_ylabel('Volatility (%)')
axes[0, 1].set_title('GARCH(1,1) Conditional Volatility')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: News Impact Curve
shock_range = np.linspace(-5, 5, 100)
nic_garch = omega + alpha * shock_range**2 + beta * uncond_var
nic_gjr = (omega + (alpha + gamma * (shock_range < 0)) * shock_range**2 
           + beta * uncond_var)

axes[1, 0].plot(shock_range, np.sqrt(nic_garch), linewidth=2, label='GARCH(1,1)')
axes[1, 0].plot(shock_range, np.sqrt(nic_gjr), linewidth=2, label='GJR-GARCH')
axes[1, 0].axvline(0, color='black', linestyle=':', linewidth=1)
axes[1, 0].set_xlabel('Shock (Îµâ‚‘â‚‹â‚)')
axes[1, 0].set_ylabel('Next Period Volatility (Ïƒâ‚œ)')
axes[1, 0].set_title('News Impact Curve')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Standardized Residuals
axes[1, 1].plot(std_resid, linewidth=0.5, alpha=0.7)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 1].axhline(2, color='orange', linestyle=':', linewidth=1)
axes[1, 1].axhline(-2, color='orange', linestyle=':', linewidth=1)
axes[1, 1].set_ylabel('Standardized Residuals')
axes[1, 1].set_title('Standardized Residuals (Should be i.i.d.)')
axes[1, 1].grid(alpha=0.3)

# Plot 5: QQ-Plot of Standardized Residuals
stats.probplot(std_resid, dist="norm", plot=axes[2, 0])
axes[2, 0].set_title('Q-Q Plot: Standardized Residuals vs Normal')
axes[2, 0].grid(alpha=0.3)

# Plot 6: Forecast vs Realized
dates_test = pd.date_range(start=dates_train[-1], periods=len(returns_test)+1)[1:]

axes[2, 1].plot(dates_test, realized_vol, linewidth=1, alpha=0.7, 
               label='Realized (|return|)', color='black')
axes[2, 1].plot(dates_test, forecasts_garch, linewidth=1.5, 
               label='GARCH(1,1)', alpha=0.8)
axes[2, 1].plot(dates_test, forecasts_gjr, linewidth=1.5,
               label='GJR-GARCH', alpha=0.8)
axes[2, 1].set_ylabel('Volatility (%)')
axes[2, 1].set_title('Out-of-Sample Volatility Forecasts')
axes[2, 1].legend(fontsize=8)
axes[2, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('garch_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== ACF of Squared Standardized Residuals =====
from statsmodels.graphics.tsaplots import plot_acf

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(std_resid, lags=20, ax=axes2[0])
axes2[0].set_title('ACF of Standardized Residuals')
axes2[0].grid(alpha=0.3)

plot_acf(std_resid**2, lags=20, ax=axes2[1])
axes2[1].set_title('ACF of Squared Standardized Residuals')
axes2[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('garch_acf.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== VaR Calculation =====
print("\n" + "="*80)
print("VALUE-AT-RISK (VaR) CALCULATION")
print("="*80)

# 1-day ahead forecast using full sample
model_final = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
fit_final = model_final.fit(disp='off')
forecast_final = fit_final.forecast(horizon=1)

vol_forecast = np.sqrt(forecast_final.variance.values[-1, 0])
mean_forecast = forecast_final.mean.values[-1, 0]

# VaR at 95% and 99% confidence
alpha_95 = 0.05
alpha_99 = 0.01

# Using Student-t quantiles
df_final = fit_final.params['nu']
var_95 = mean_forecast + vol_forecast * stats.t.ppf(alpha_95, df_final)
var_99 = mean_forecast + vol_forecast * stats.t.ppf(alpha_99, df_final)

print(f"1-Day Ahead Forecast:")
print(f"  Mean: {mean_forecast:.4f}%")
print(f"  Volatility: {vol_forecast:.4f}%")

print(f"\nValue-at-Risk (VaR):")
print(f"  95% VaR: {var_95:.4f}% (5% chance of loss exceeding this)")
print(f"  99% VaR: {var_99:.4f}% (1% chance of loss exceeding this)")

# For $1M portfolio
portfolio_value = 1_000_000
var_95_dollar = portfolio_value * abs(var_95) / 100
var_99_dollar = portfolio_value * abs(var_99) / 100

print(f"\nFor ${portfolio_value:,.0f} portfolio:")
print(f"  95% VaR: ${var_95_dollar:,.0f}")
print(f"  99% VaR: ${var_99_dollar:,.0f}")

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Volatility Characteristics:")
print(f"   Persistence (Î±+Î²): {alpha + beta:.4f}")
if alpha + beta > 0.95:
    print("   â†’ High persistence: Shocks decay slowly")
print(f"   Half-life: {half_life:.1f} days")
print(f"   Unconditional volatility: {np.sqrt(uncond_var):.2f}%")

print("\n2. Asymmetry:")
if gamma > 0 and gamma / gjr_fit.std_err['gamma[1]'] > 2:
    print(f"   âœ“ Significant leverage effect (Î³={gamma:.4f})")
    print("   â†’ Negative shocks increase volatility more")
else:
    print("   No significant asymmetry detected")

print("\n3. Distribution:")
if df_param < 10:
    print(f"   âœ“ Heavy tails confirmed (Î½={df_param:.2f} < 10)")
    print("   â†’ Student-t better than Gaussian")
else:
    print("   Tails close to normal")

print("\n4. Forecasting:")
print(f"   Best model: {best_model_name} (lowest AIC)")
if qlike_gjr < qlike_garch:
    print(f"   GJR-GARCH outperforms GARCH(1,1) out-of-sample")
else:
    print(f"   Standard GARCH(1,1) adequate")

print("\n5. Practical Applications:")
print("   â€¢ Risk management: VaR, Expected Shortfall")
print("   â€¢ Trading: Volatility targeting, statistical arbitrage")
print("   â€¢ Option pricing: GARCH option models")
print("   â€¢ Portfolio: Dynamic hedging, time-varying weights")

print("\n6. Limitations:")
print("   â€¢ Assumes continuous sampling (issues with intraday)")
print("   â€¢ May underestimate tail risk during crises")
print("   â€¢ Symmetric GARCH misses leverage effect")
print("   â€¢ Consider regime-switching for structural breaks")
