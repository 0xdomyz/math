from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.stattools import adfuller, kpss, coint_johansen
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("VECTOR AUTOREGRESSION (VAR) AND VECTOR ERROR CORRECTION MODELS (VECM)")
print("="*80)

# Generate synthetic data for 3-variable system
np.random.seed(42)
n = 500

# System 1: Stationary VAR(2) for GDP, Unemployment, Interest Rate
print("\n" + "="*80)
print("PART 1: VAR MODEL (STATIONARY SYSTEM)")
print("="*80)

# True VAR(2) parameters
# Y1t = 0.6*Y1t-1 - 0.2*Y1t-2 + 0.1*Y2t-1 - 0.1*Y3t-1 + e1t
# Y2t = -0.3*Y1t-1 + 0.5*Y2t-2 + 0.2*Y3t-1 + e2t  
# Y3t = 0.2*Y1t-1 - 0.1*Y2t-1 + 0.7*Y3t-1 + e3t

A1 = np.array([[0.6, 0.1, -0.1],
               [-0.3, 0.0, 0.2],
               [0.2, -0.1, 0.7]])

A2 = np.array([[-0.2, 0.0, 0.0],
               [0.0, 0.5, 0.0],
               [0.0, 0.0, 0.0]])

# Covariance matrix
Sigma = np.array([[1.0, 0.3, -0.2],
                  [0.3, 0.8, 0.1],
                  [-0.2, 0.1, 0.6]])

# Simulate
Y = np.zeros((n, 3))
for t in range(2, n):
    Y[t] = A1 @ Y[t-1] + A2 @ Y[t-2] + np.random.multivariate_normal([0, 0, 0], Sigma)

# Create DataFrame
dates = pd.date_range('2000-01-01', periods=n, freq='Q')
df_var = pd.DataFrame(Y, index=dates, columns=['GDP_Growth', 'Unemployment_Rate', 'Interest_Rate'])

print("\nSimulated VAR Data (First 10 observations):")
print(df_var.head(10))

print("\nDescriptive Statistics:")
print(df_var.describe())

# Check stationarity
print("\n" + "="*80)
print("STATIONARITY TESTING")
print("="*80)

def adf_test(series, name):
    result = adfuller(series, autolag='AIC')
    print(f"\n{name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  P-value: {result[1]:.4f}")
    print(f"  Critical Values: {result[4]}")
    print(f"  Result: {'Stationary' if result[1] < 0.05 else 'Non-stationary (unit root)'}")
    return result[1] < 0.05

stationary_checks = {}
for col in df_var.columns:
    stationary_checks[col] = adf_test(df_var[col], col)

all_stationary = all(stationary_checks.values())
print(f"\nConclusion: {'All series stationary - proceed with VAR' if all_stationary else 'Check cointegration or difference'}")

# Lag order selection
print("\n" + "="*80)
print("VAR LAG ORDER SELECTION")
print("="*80)

model = VAR(df_var)
lag_order = model.select_order(maxlags=8)
print("\nLag Order Selection:")
print(lag_order.summary())

optimal_lag = lag_order.aic
print(f"\nOptimal lag (AIC): {optimal_lag}")

# Estimate VAR
print("\n" + "="*80)
print(f"VAR({optimal_lag}) ESTIMATION")
print("="*80)

var_model = model.fit(optimal_lag)
print(var_model.summary())

# Extract parameters
params = var_model.params
print("\nEstimated Coefficient Matrix A1:")
print(params.iloc[:3, :])

# Diagnostics
print("\n" + "="*80)
print("VAR DIAGNOSTICS")
print("="*80)

# Residual autocorrelation
print("\n1. RESIDUAL SERIAL CORRELATION")
print("-" * 40)
residuals = var_model.resid
for col in residuals.columns:
    lb_result = acorr_ljungbox(residuals[col], lags=10, return_df=True)
    print(f"\n{col}:")
    print(f"  Ljung-Box p-value (lag 10): {lb_result['lb_pvalue'].iloc[-1]:.4f}")
    print(f"  Result: {'PASS' if lb_result['lb_pvalue'].iloc[-1] > 0.05 else 'FAIL (autocorrelation)'}")

# Normality
print("\n2. RESIDUAL NORMALITY")
print("-" * 40)
for col in residuals.columns:
    jb_stat, jb_pval = stats.jarque_bera(residuals[col])
    print(f"{col}: JB={jb_stat:.2f}, p-value={jb_pval:.4f} - {'Normal' if jb_pval > 0.05 else 'Non-normal'}")

# Stability (eigenvalues)
print("\n3. STABILITY CHECK (Eigenvalues)")
print("-" * 40)
eigenvalues = np.linalg.eigvals(var_model.coefs[0])
if optimal_lag > 1:
    # Companion matrix for higher lags
    companion = var_model.get_eq_index('companion_matrix', 0)
    
print("Eigenvalues (modulus):")
for i, ev in enumerate(eigenvalues):
    print(f"  λ{i+1} = {np.abs(ev):.4f}")
    
stable = np.all(np.abs(eigenvalues) < 1)
print(f"\nStability: {'STABLE (all |λ| < 1)' if stable else 'UNSTABLE'}")

# Granger Causality
print("\n" + "="*80)
print("GRANGER CAUSALITY TESTS")
print("="*80)

causality_results = {}
for caused in df_var.columns:
    print(f"\nVariable: {caused}")
    print("-" * 40)
    for causing in df_var.columns:
        if caused != causing:
            test = var_model.test_causality(caused, causing, kind='f')
            causality_results[(causing, caused)] = test.pvalue
            print(f"  {causing} → {caused}: p-value = {test.pvalue:.4f} {'***' if test.pvalue < 0.01 else '**' if test.pvalue < 0.05 else '*' if test.pvalue < 0.10 else 'NS'}")

print("\nInterpretation: p < 0.05 indicates Granger causality")

# Impulse Response Functions
print("\n" + "="*80)
print("IMPULSE RESPONSE FUNCTIONS (IRF)")
print("="*80)

irf = var_model.irf(periods=20)
print("\nComputing IRFs with Cholesky orthogonalization...")
print(f"Ordering: {list(df_var.columns)}")

# Summary of IRFs at selected horizons
print("\nIRF Summary (selected horizons):")
for h in [1, 4, 8, 20]:
    print(f"\nHorizon h={h}:")
    irfs_at_h = irf.irfs[h-1]
    for i, target in enumerate(df_var.columns):
        print(f"  {target}:")
        for j, shock in enumerate(df_var.columns):
            print(f"    Shock from {shock}: {irfs_at_h[i, j]:.4f}")

# Forecast Error Variance Decomposition
print("\n" + "="*80)
print("FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)")
print("="*80)

fevd = var_model.fevd(periods=20)
print("\nFEVD at horizon 20:")
for i, var_name in enumerate(df_var.columns):
    print(f"\n{var_name} variance explained by:")
    for j, shock_name in enumerate(df_var.columns):
        pct = fevd.decomp[19, i, j] * 100  # h=20 (index 19)
        print(f"  {shock_name}: {pct:.2f}%")

# Forecasting
print("\n" + "="*80)
print("OUT-OF-SAMPLE FORECASTING")
print("="*80)

train_size = 450
test_size = n - train_size

df_train = df_var.iloc[:train_size]
df_test = df_var.iloc[train_size:]

# Fit on training data
var_train = VAR(df_train).fit(optimal_lag)

# Forecast
forecast_steps = test_size
forecast = var_train.forecast(df_train.values[-optimal_lag:], steps=forecast_steps)
forecast_df = pd.DataFrame(forecast, index=df_test.index, columns=df_var.columns)

# Evaluation
mae = np.mean(np.abs(df_test - forecast_df), axis=0)
rmse = np.sqrt(np.mean((df_test - forecast_df)**2, axis=0))

print(f"\nForecast Evaluation ({forecast_steps}-step ahead):")
for i, col in enumerate(df_var.columns):
    print(f"\n{col}:")
    print(f"  MAE: {mae[i]:.4f}")
    print(f"  RMSE: {rmse[i]:.4f}")

# PART 2: VECM with Cointegration
print("\n" + "="*80)
print("PART 2: VECM MODEL (COINTEGRATED SYSTEM)")
print("="*80)

# Generate cointegrated data
# Two I(1) series with r=1 cointegrating relationship
print("\nGenerating cointegrated system...")

# Common trend (random walk)
common_trend = np.cumsum(np.random.normal(0, 1, n))

# Two series with cointegrating relationship: Y1 - 2*Y2 ~ I(0)
beta = 2.0  # Cointegration coefficient
transitory = np.random.normal(0, 0.5, n)

Y1_coint = common_trend + transitory
Y2_coint = (common_trend - transitory) / beta

# Add short-run dynamics
for t in range(2, n):
    ect = Y1_coint[t-1] - beta * Y2_coint[t-1]  # Error correction term
    Y1_coint[t] += -0.15 * ect  # Adjustment speed
    Y2_coint[t] += 0.08 * ect

df_coint = pd.DataFrame({'Y1': Y1_coint, 'Y2': Y2_coint}, index=dates)

print("\nCointegrated Data (First 10 observations):")
print(df_coint.head(10))

# Test for unit roots
print("\n" + "="*80)
print("UNIT ROOT TESTS (Should be I(1))")
print("="*80)

for col in df_coint.columns:
    adf_test(df_coint[col], col)

# Test cointegration
print("\n" + "="*80)
print("JOHANSEN COINTEGRATION TEST")
print("="*80)

# Johansen test
johansen_result = coint_johansen(df_coint, det_order=0, k_ar_diff=2)

print("\nJohansen Test Results:")
print("\nTrace Statistic:")
for i in range(len(johansen_result.lr1)):
    print(f"  r ≤ {i}: Statistic = {johansen_result.lr1[i]:.2f}, "
          f"Critical Value (5%) = {johansen_result.cvt[i, 1]:.2f}, "
          f"Result: {'Reject' if johansen_result.lr1[i] > johansen_result.cvt[i, 1] else 'Fail to reject'}")

print("\nMax Eigenvalue Statistic:")
for i in range(len(johansen_result.lr2)):
    print(f"  r = {i}: Statistic = {johansen_result.lr2[i]:.2f}, "
          f"Critical Value (5%) = {johansen_result.cvm[i, 1]:.2f}, "
          f"Result: {'Reject' if johansen_result.lr2[i] > johansen_result.cvm[i, 1] else 'Fail to reject'}")

# Determine cointegration rank
coint_rank = 0
for i in range(len(johansen_result.lr1)):
    if johansen_result.lr1[i] > johansen_result.cvt[i, 1]:
        coint_rank = i + 1
        
print(f"\n*** Cointegration Rank: r = {coint_rank} ***")
print(f"Number of cointegrating relationships: {coint_rank}")
print(f"Number of common trends: {len(df_coint.columns) - coint_rank}")

if coint_rank > 0:
    print("\nEstimated Cointegrating Vector(s):")
    for i in range(coint_rank):
        print(f"  β{i+1}': {johansen_result.evec[:, i]}")

# Estimate VECM
print("\n" + "="*80)
print("VECM ESTIMATION")
print("="*80)

if coint_rank > 0:
    # Select lag order for VECM
    vecm_lag = select_order(df_coint, maxlags=8, deterministic='ci')
    print(f"\nOptimal VECM lag: {vecm_lag.aic}")
    
    # Fit VECM
    vecm_model = VECM(df_coint, k_ar_diff=vecm_lag.aic, coint_rank=coint_rank, deterministic='ci')
    vecm_fitted = vecm_model.fit()
    
    print(vecm_fitted.summary())
    
    # Extract parameters
    print("\n" + "="*80)
    print("VECM PARAMETER INTERPRETATION")
    print("="*80)
    
    alpha = vecm_fitted.alpha
    beta = vecm_fitted.beta
    
    print("\nCointegrating Vector (β):")
    print(beta)
    print(f"\nEquilibrium relationship: {beta[0,0]:.3f}*Y1 + {beta[1,0]:.3f}*Y2 = 0")
    print(f"Normalized: Y1 ≈ {-beta[1,0]/beta[0,0]:.3f} * Y2")
    
    print("\nAdjustment Speeds (α - Loading Matrix):")
    print(alpha)
    for i, col in enumerate(df_coint.columns):
        print(f"\n{col}:")
        print(f"  α = {alpha[i,0]:.4f}")
        if alpha[i,0] != 0:
            half_life = np.log(0.5) / np.log(1 + alpha[i,0]) if (1 + alpha[i,0]) > 0 and (1 + alpha[i,0]) < 1 else np.inf
            print(f"  Half-life: {half_life:.2f} periods")
            print(f"  Interpretation: {'Adjusts to restore equilibrium' if alpha[i,0] < 0 else 'Moves away from equilibrium (check)'}")
    
    # VECM forecasting
    print("\n" + "="*80)
    print("VECM FORECASTING")
    print("="*80)
    
    df_coint_train = df_coint.iloc[:train_size]
    df_coint_test = df_coint.iloc[train_size:]
    
    vecm_train = VECM(df_coint_train, k_ar_diff=vecm_lag.aic, coint_rank=coint_rank, deterministic='ci')
    vecm_train_fitted = vecm_train.fit()
    
    # Forecast
    vecm_forecast = vecm_train_fitted.predict(steps=test_size)
    vecm_forecast_df = pd.DataFrame(vecm_forecast, index=df_coint_test.index, columns=df_coint.columns)
    
    # Compare to naive VAR in differences
    df_coint_diff = df_coint_train.diff().dropna()
    var_diff = VAR(df_coint_diff).fit(vecm_lag.aic)
    
    # Cumulative sum of forecasted differences
    last_level = df_coint_train.iloc[-1].values
    var_diff_forecast = var_diff.forecast(df_coint_diff.values[-vecm_lag.aic:], steps=test_size)
    var_levels_forecast = last_level + np.cumsum(var_diff_forecast, axis=0)
    var_levels_df = pd.DataFrame(var_levels_forecast, index=df_coint_test.index, columns=df_coint.columns)
    
    # Evaluation
    vecm_rmse = np.sqrt(np.mean((df_coint_test - vecm_forecast_df)**2, axis=0))
    var_rmse = np.sqrt(np.mean((df_coint_test - var_levels_df)**2, axis=0))
    
    print(f"\nForecast RMSE Comparison ({test_size}-step ahead):")
    for i, col in enumerate(df_coint.columns):
        print(f"\n{col}:")
        print(f"  VECM: {vecm_rmse[i]:.4f}")
        print(f"  VAR(diff): {var_rmse[i]:.4f}")
        print(f"  Improvement: {(1 - vecm_rmse[i]/var_rmse[i])*100:.2f}%")

# Visualizations
fig = plt.figure(figsize=(20, 16))

# VAR plots
# Plot 1: Time series
ax1 = plt.subplot(4, 3, 1)
df_var.plot(ax=ax1, alpha=0.8)
ax1.set_title('Simulated VAR Data', fontweight='bold')
ax1.set_xlabel('Date')
ax1.legend(loc='best')
ax1.grid(alpha=0.3)

# Plot 2: IRF
ax2 = plt.subplot(4, 3, 2)
irf.plot(impulse='GDP_Growth', response='Unemployment_Rate', ax=ax2)
ax2.set_title('IRF: GDP → Unemployment', fontweight='bold')
ax2.grid(alpha=0.3)

# Plot 3: IRF
ax3 = plt.subplot(4, 3, 3)
irf.plot(impulse='Interest_Rate', response='GDP_Growth', ax=ax3)
ax3.set_title('IRF: Interest Rate → GDP', fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: FEVD for GDP
ax4 = plt.subplot(4, 3, 4)
horizons = range(1, 21)
for j, shock in enumerate(df_var.columns):
    fevd_series = [fevd.decomp[h-1, 0, j] * 100 for h in horizons]
    ax4.plot(horizons, fevd_series, label=shock, marker='o', markersize=4)
ax4.set_title('FEVD: GDP Growth Variance', fontweight='bold')
ax4.set_xlabel('Horizon')
ax4.set_ylabel('% Variance Explained')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Forecast vs Actual
ax5 = plt.subplot(4, 3, 5)
ax5.plot(df_test.index, df_test['GDP_Growth'], label='Actual', linewidth=2)
ax5.plot(forecast_df.index, forecast_df['GDP_Growth'], label='Forecast', linestyle='--', linewidth=2)
ax5.set_title('VAR Forecast: GDP Growth', fontweight='bold')
ax5.set_xlabel('Date')
ax5.legend()
ax5.grid(alpha=0.3)

# Plot 6: Residuals
ax6 = plt.subplot(4, 3, 6)
residuals['GDP_Growth'].plot(ax=ax6, alpha=0.7)
ax6.axhline(0, color='red', linestyle='--')
ax6.set_title('VAR Residuals: GDP Growth', fontweight='bold')
ax6.set_ylabel('Residual')
ax6.grid(alpha=0.3)

# VECM plots
# Plot 7: Cointegrated series
ax7 = plt.subplot(4, 3, 7)
df_coint.plot(ax=ax7, alpha=0.8)
ax7.set_title('Cointegrated Series (I(1))', fontweight='bold')
ax7.set_xlabel('Date')
ax7.legend()
ax7.grid(alpha=0.3)

# Plot 8: Spread (equilibrium relationship)
ax8 = plt.subplot(4, 3, 8)
spread = df_coint['Y1'] - (-beta[1,0]/beta[0,0]) * df_coint['Y2']
spread.plot(ax=ax8, alpha=0.8, color='green')
ax8.axhline(0, color='red', linestyle='--')
ax8.set_title(f'Equilibrium Error (Y1 - {-beta[1,0]/beta[0,0]:.2f}*Y2)', fontweight='bold')
ax8.set_ylabel('Deviation')
ax8.grid(alpha=0.3)

# Plot 9: ACF of spread
ax9 = plt.subplot(4, 3, 9)
plot_acf(spread, lags=30, ax=ax9, alpha=0.05)
ax9.set_title('ACF of Equilibrium Error (Should decay)', fontweight='bold')
ax9.grid(alpha=0.3)

# Plot 10: VECM forecast
ax10 = plt.subplot(4, 3, 10)
ax10.plot(df_coint_test.index, df_coint_test['Y1'], label='Actual', linewidth=2)
ax10.plot(vecm_forecast_df.index, vecm_forecast_df['Y1'], label='VECM', linestyle='--', linewidth=2)
ax10.plot(var_levels_df.index, var_levels_df['Y1'], label='VAR(diff)', linestyle=':', linewidth=2, alpha=0.7)
ax10.set_title('VECM vs VAR Forecast: Y1', fontweight='bold')
ax10.set_xlabel('Date')
ax10.legend()
ax10.grid(alpha=0.3)

# Plot 11: Forecast errors
ax11 = plt.subplot(4, 3, 11)
vecm_errors = df_coint_test['Y1'] - vecm_forecast_df['Y1']
var_errors = df_coint_test['Y1'] - var_levels_df['Y1']
ax11.plot(df_coint_test.index, vecm_errors, label='VECM', alpha=0.7)
ax11.plot(df_coint_test.index, var_errors, label='VAR(diff)', alpha=0.7)
ax11.axhline(0, color='red', linestyle='--')
ax11.set_title('Forecast Errors: Y1', fontweight='bold')
ax11.set_ylabel('Error')
ax11.legend()
ax11.grid(alpha=0.3)

# Plot 12: Heatmap of Granger causality p-values
ax12 = plt.subplot(4, 3, 12)
causality_matrix = np.ones((3, 3))
for (causing, caused), pval in causality_results.items():
    i = list(df_var.columns).index(caused)
    j = list(df_var.columns).index(causing)
    causality_matrix[i, j] = pval

sns.heatmap(causality_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', 
            xticklabels=df_var.columns, yticklabels=df_var.columns, 
            vmin=0, vmax=0.1, ax=ax12, cbar_kws={'label': 'p-value'})
ax12.set_title('Granger Causality P-values\n(Row ← Column)', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. VAR({optimal_lag}) captures dynamic interdependencies among {len(df_var.columns)} variables")
print(f"2. Granger causality: {sum(p < 0.05 for p in causality_results.values())} significant relationships detected")
print(f"3. IRF shows shock transmission over 20 periods with Cholesky ordering")
print(f"4. FEVD: GDP variance explained {fevd.decomp[19, 0, 0]*100:.1f}% by own shocks at h=20")
print(f"5. Cointegration rank: r={coint_rank} equilibrium relationship(s) in system")
print(f"6. VECM adjustment speed: α={alpha[0,0]:.3f} for Y1 (half-life ~{np.log(0.5)/np.log(1+alpha[0,0]):.1f} periods)")
print(f"7. VECM outperforms VAR in differences by ~{np.mean((1 - vecm_rmse/var_rmse)*100):.1f}% RMSE")
print(f"8. Stability check: All eigenvalues |λ| < 1 → {'STABLE' if stable else 'CHECK REQUIRED'}")
print(f"9. Long-run equilibrium: Y1 ≈ {-beta[1,0]/beta[0,0]:.2f} * Y2")
print(f"10. Forecast horizon: {test_size} steps out-of-sample evaluation")

print("\n" + "="*80)
print("WORKFLOW SUMMARY")
print("="*80)
print("VAR:")
print("  1. Test stationarity (ADF) → All I(0) required")
print("  2. Select lag order (AIC/BIC)")
print("  3. Estimate by OLS equation-by-equation")
print("  4. Diagnose residuals (autocorrelation, normality)")
print("  5. Granger causality tests")
print("  6. Compute IRF (Cholesky or structural identification)")
print("  7. FEVD for variance attribution")
print("  8. Out-of-sample forecasting")
print("\nVECM:")
print("  1. Test unit roots → Series should be I(1)")
print("  2. Johansen test for cointegration rank r")
print("  3. If r > 0: Estimate VECM (error correction)")
print("  4. Interpret β (long-run equilibrium) and α (adjustment speeds)")
print("  5. Forecast with equilibrium constraint")
print("  6. Compare to VAR in differences (should outperform)")
