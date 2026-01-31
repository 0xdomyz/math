import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 300

# ===== Simulate VAR(2) Process =====
# Three variables: Y1, Y2, Y3
# Y1: GDP growth, Y2: Inflation, Y3: Interest rate

# True VAR(2) coefficients
A1 = np.array([
    [0.6, 0.1, -0.2],   # Y1 equation
    [0.2, 0.5, -0.1],   # Y2 equation
    [0.1, 0.3, 0.4]     # Y3 equation
])

A2 = np.array([
    [0.2, 0.0, -0.1],
    [0.1, 0.2, 0.0],
    [0.0, 0.1, 0.2]
])

const = np.array([0.5, 2.0, 1.5])  # Intercepts

# Innovation covariance (correlated shocks)
Sigma = np.array([
    [1.0, 0.3, 0.2],
    [0.3, 0.8, 0.1],
    [0.2, 0.1, 0.6]
])

# Simulate
Y = np.zeros((n, 3))
innovations = np.random.multivariate_normal([0, 0, 0], Sigma, n)

for t in range(2, n):
    Y[t] = const + A1 @ Y[t-1] + A2 @ Y[t-2] + innovations[t]

# Create DataFrame
df = pd.DataFrame(Y, columns=['GDP_Growth', 'Inflation', 'Interest_Rate'])

print("="*70)
print("VECTOR AUTOREGRESSION (VAR) MODEL")
print("="*70)
print(f"\nSample Size: {n}")
print("\nDescriptive Statistics:")
print(df.describe().round(4))

# ===== Check Stationarity =====
print("\n" + "="*70)
print("STATIONARITY TESTS (ADF)")
print("="*70)

for col in df.columns:
    adf_result = adfuller(df[col], autolag='AIC')
    print(f"\n{col}:")
    print(f"  ADF Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print(f"  âœ“ Stationary")
    else:
        print(f"  âœ— Unit root detected - consider differencing")

# ===== Lag Length Selection =====
print("\n" + "="*70)
print("LAG LENGTH SELECTION")
print("="*70)

model_select = VAR(df)
lag_order_results = model_select.select_order(maxlags=6)
print(lag_order_results.summary())

optimal_lag = lag_order_results.bic
print(f"\nSelected Lag Order (BIC): {optimal_lag}")

# ===== Estimate VAR Model =====
print("\n" + "="*70)
print(f"VAR({optimal_lag}) ESTIMATION")
print("="*70)

var_model = model_select.fit(maxlags=optimal_lag)
print(var_model.summary())

# ===== Granger Causality Tests =====
print("\n" + "="*70)
print("GRANGER CAUSALITY TESTS")
print("="*70)

variables = df.columns.tolist()

for caused_var in variables:
    print(f"\n{caused_var} caused by:")
    
    for causing_var in variables:
        if caused_var != causing_var:
            # Create dataset for test
            test_data = df[[caused_var, causing_var]]
            
            try:
                # Test H0: causing_var does not Granger-cause caused_var
                gc_result = grangercausalitytests(test_data, maxlag=optimal_lag, 
                                                  verbose=False)
                
                # Get p-value for F-test at optimal lag
                p_value = gc_result[optimal_lag][0]['ssr_ftest'][1]
                
                if p_value < 0.05:
                    print(f"  {causing_var}: âœ“ Granger-causes (p={p_value:.4f})")
                else:
                    print(f"  {causing_var}: âœ— No causality (p={p_value:.4f})")
            except:
                print(f"  {causing_var}: Test failed")

# ===== Impulse Response Functions =====
print("\n" + "="*70)
print("IMPULSE RESPONSE FUNCTIONS")
print("="*70)

irf = var_model.irf(periods=20)

# Print IRF for first few periods
print("\nImpulse: GDP_Growth shock, Response: All variables")
print(f"{'Period':>8s}  {'GDP_Growth':>12s}  {'Inflation':>12s}  {'Interest_Rate':>12s}")
print("-" * 52)

for period in range(5):
    print(f"{period:>8d}  {irf.irfs[period, 0, 0]:>12.4f}  "
          f"{irf.irfs[period, 1, 0]:>12.4f}  {irf.irfs[period, 2, 0]:>12.4f}")

# Plot all IRFs
fig, axes = plt.subplots(3, 3, figsize=(14, 12))

shock_names = df.columns
response_names = df.columns

for i, response in enumerate(response_names):
    for j, shock in enumerate(shock_names):
        irf.plot(impulse=shock, response=response, ax=axes[i, j])
        axes[i, j].set_title(f'Shock: {shock}\nResponse: {response}', 
                            fontsize=9)
        axes[i, j].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('var_impulse_responses.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nImpulse response functions plotted and saved.")

# ===== Forecast Error Variance Decomposition =====
print("\n" + "="*70)
print("FORECAST ERROR VARIANCE DECOMPOSITION")
print("="*70)

fevd = var_model.fevd(periods=20)

# Display FEVD at selected horizons
horizons = [1, 5, 10, 20]

for h in horizons:
    print(f"\n--- Horizon: {h} periods ---")
    fevd_df = pd.DataFrame(fevd.decomp[h-1], 
                          columns=df.columns,
                          index=df.columns)
    print(fevd_df.round(4))
    print(f"Row sums (should be 1.0): {fevd_df.sum(axis=1).values}")

# Plot FEVD
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, var_name in enumerate(df.columns):
    fevd_var = fevd.decomp[:, i, :]
    
    for j, shock_name in enumerate(df.columns):
        axes[i].plot(fevd_var[:, j], label=f'{shock_name} shock', linewidth=2)
    
    axes[i].set_title(f'Variance Decomposition: {var_name}', fontsize=10)
    axes[i].set_xlabel('Horizon')
    axes[i].set_ylabel('Proportion of Variance')
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)
    axes[i].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('var_variance_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVariance decomposition plotted and saved.")

# ===== Stability Check =====
print("\n" + "="*70)
print("STABILITY DIAGNOSTICS")
print("="*70)

# Check roots of companion matrix
roots = var_model.roots
print(f"\nRoots of Characteristic Polynomial:")
print(f"(All should be < 1 in modulus for stability)")

for i, root in enumerate(roots, 1):
    modulus = np.abs(root)
    print(f"  Root {i}: {root:.4f}, Modulus: {modulus:.4f}")

if np.all(np.abs(roots) < 1):
    print("\nâœ“ VAR process is STABLE")
else:
    print("\nâœ— WARNING: VAR process may be UNSTABLE")

# ===== Residual Diagnostics =====
print("\n" + "="*70)
print("RESIDUAL DIAGNOSTICS")
print("="*70)

residuals = var_model.resid

# Test for autocorrelation
print("\nPortmanteau Test (Autocorrelation):")
whiteness_test = var_model.test_whiteness(nlags=10, signif=0.05)
print(whiteness_test.summary())

# Normality test
print("\nNormality Test (Jarque-Bera):")
normality_test = var_model.test_normality()
print(normality_test.summary())

# ===== Forecasting =====
print("\n" + "="*70)
print("FORECASTING")
print("="*70)

# Split data for out-of-sample forecast
train_size = int(0.9 * len(df))
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Re-estimate on training data
var_train = VAR(train_data).fit(maxlags=optimal_lag)

# Forecast
forecast_steps = len(test_data)
forecast = var_train.forecast(train_data.values[-var_train.k_ar:], 
                              steps=forecast_steps)
forecast_df = pd.DataFrame(forecast, 
                          index=test_data.index,
                          columns=df.columns)

# Calculate forecast errors
forecast_errors = test_data - forecast_df
rmse = np.sqrt((forecast_errors**2).mean())

print(f"\nOut-of-Sample Forecast (last {forecast_steps} periods)")
print(f"\nRMSE by variable:")
for col in df.columns:
    print(f"  {col}: {rmse[col]:.4f}")

# Plot forecasts
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for i, col in enumerate(df.columns):
    # Historical data
    axes[i].plot(df.index[:train_size], train_data[col], 
                'b-', linewidth=1.5, label='Training Data')
    axes[i].plot(test_data.index, test_data[col], 
                'g-', linewidth=1.5, label='Actual')
    axes[i].plot(forecast_df.index, forecast_df[col], 
                'r--', linewidth=2, label='Forecast')
    
    axes[i].axvline(train_size, color='black', linestyle=':', 
                   linewidth=1, alpha=0.5)
    axes[i].set_title(f'{col}: Forecast vs Actual', fontsize=10)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Value')
    axes[i].legend(fontsize=9)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('var_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nForecast plot saved.")

# ===== Summary Statistics =====
print("\n" + "="*70)
print("MODEL SUMMARY")
print("="*70)

print(f"\nModel: VAR({var_model.k_ar})")
print(f"Number of Equations: {var_model.neqs}")
print(f"Number of Observations: {var_model.nobs}")
print(f"Degrees of Freedom: {var_model.df_resid}")
print(f"\nLog Likelihood: {var_model.llf:.2f}")
print(f"AIC: {var_model.aic:.2f}")
print(f"BIC: {var_model.bic:.2f}")
print(f"FPE: {var_model.fpe:.4f}")
print(f"HQIC: {var_model.hqic:.2f}")

# Coefficient summary
print("\n" + "="*70)
print("COEFFICIENT SUMMARY (True vs Estimated)")
print("="*70)

print("\nLag 1 Coefficients:")
print(f"{'':15s} {'GDP_Growth':>12s} {'Inflation':>12s} {'Interest_Rate':>12s}")
print("-" * 54)
for i, var in enumerate(df.columns):
    true_vals = A1[i]
    est_vals = var_model.params.iloc[1 + i*var_model.k_ar : 1 + (i+1)*var_model.k_ar, i].values[:3]
    print(f"{var} (True):  {true_vals[0]:>12.4f} {true_vals[1]:>12.4f} {true_vals[2]:>12.4f}")
    print(f"{var} (Est.):  {est_vals[0]:>12.4f} {est_vals[1]:>12.4f} {est_vals[2]:>12.4f}")
    print()

print("\nLag 2 Coefficients:")
print(f"{'':15s} {'GDP_Growth':>12s} {'Inflation':>12s} {'Interest_Rate':>12s}")
print("-" * 54)
for i, var in enumerate(df.columns):
    true_vals = A2[i]
    # Get second lag coefficients
    all_coefs = var_model.params.iloc[:, i].values
    # Skip const and first lag (1 + K coefficients), get next K
    start_idx = 1 + 3
    est_vals = all_coefs[start_idx:start_idx+3]
    print(f"{var} (True):  {true_vals[0]:>12.4f} {true_vals[1]:>12.4f} {true_vals[2]:>12.4f}")
    print(f"{var} (Est.):  {est_vals[0]:>12.4f} {est_vals[1]:>12.4f} {est_vals[2]:>12.4f}")
    print()
