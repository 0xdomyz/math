import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

np.random.seed(888)

# ===== Simulate AR(2) Process with Regime Change =====
n = 600
regime_change = 400  # Structural break point

# Regime 1: First 400 observations
phi1_regime1, phi2_regime1 = 0.7, -0.2
y_regime1 = np.zeros(regime_change)
eps1 = np.random.normal(0, 1, regime_change)

for t in range(2, regime_change):
    y_regime1[t] = phi1_regime1 * y_regime1[t-1] + phi2_regime1 * y_regime1[t-2] + eps1[t]

# Regime 2: Last 200 observations (different dynamics)
phi1_regime2, phi2_regime2 = 0.5, 0.1
y_regime2 = np.zeros(n - regime_change)
y_regime2[0] = y_regime1[-1]
y_regime2[1] = y_regime1[-2]
eps2 = np.random.normal(0, 1.5, n - regime_change)  # Higher variance

for t in range(2, n - regime_change):
    y_regime2[t] = phi1_regime2 * y_regime2[t-1] + phi2_regime2 * y_regime2[t-2] + eps2[t]

# Combine regimes
y = np.concatenate([y_regime1, y_regime2])

# Create time index
time_index = pd.date_range(start='2000-01-01', periods=n, freq='D')
ts_data = pd.Series(y, index=time_index)

print("="*80)
print("OUT-OF-SAMPLE VALIDATION: COMPREHENSIVE COMPARISON")
print("="*80)
print(f"\nData Generating Process:")
print(f"  Regime 1 (t=1-{regime_change}): AR(2) with Ï†â‚={phi1_regime1}, Ï†â‚‚={phi2_regime1}")
print(f"  Regime 2 (t={regime_change+1}-{n}): AR(2) with Ï†â‚={phi1_regime2}, Ï†â‚‚={phi2_regime2}")
print(f"  Total observations: {n}")

# ===== 1. Simple Train-Test Split =====
print("\n" + "="*80)
print("METHOD 1: SIMPLE TRAIN-TEST SPLIT (80/20)")
print("="*80)

train_size = int(0.8 * n)
train_data = ts_data[:train_size]
test_data = ts_data[train_size:]

print(f"Training set: {len(train_data)} observations")
print(f"Test set: {len(test_data)} observations")

# Fit different AR models
models_to_test = [1, 2, 3, 4]
train_test_results = []

for p in models_to_test:
    # Estimate on training data
    model = AutoReg(train_data, lags=p, trend='c')
    fit = model.fit()
    
    # In-sample performance
    in_sample_pred = fit.predict(start=p, end=len(train_data)-1)
    in_sample_mse = mean_squared_error(train_data[p:], in_sample_pred)
    in_sample_rmse = np.sqrt(in_sample_mse)
    
    # Out-of-sample forecast
    oos_pred = fit.predict(start=len(train_data), end=len(ts_data)-1)
    oos_mse = mean_squared_error(test_data, oos_pred)
    oos_rmse = np.sqrt(oos_mse)
    oos_mae = mean_absolute_error(test_data, oos_pred)
    
    # Out-of-sample RÂ²
    ss_res = np.sum((test_data - oos_pred)**2)
    ss_tot = np.sum((test_data - train_data.mean())**2)
    oos_r2 = 1 - ss_res / ss_tot
    
    train_test_results.append({
        'Model': f'AR({p})',
        'In_Sample_RMSE': in_sample_rmse,
        'OOS_RMSE': oos_rmse,
        'OOS_MAE': oos_mae,
        'OOS_R2': oos_r2,
        'Overfit_Ratio': oos_rmse / in_sample_rmse
    })

tt_df = pd.DataFrame(train_test_results)
print("\nTrain-Test Split Results:")
print(tt_df.to_string(index=False))

best_oos = tt_df.loc[tt_df['OOS_RMSE'].idxmin()]
print(f"\nBest out-of-sample: {best_oos['Model']} (RMSE = {best_oos['OOS_RMSE']:.4f})")

# ===== 2. Rolling Window Validation =====
print("\n" + "="*80)
print("METHOD 2: ROLLING WINDOW VALIDATION")
print("="*80)

window_size = 300  # 300-day rolling window
forecast_horizon = 1  # 1-step ahead

rolling_results = {f'AR({p})': {'predictions': [], 'actuals': [], 'errors': []} 
                   for p in models_to_test}

for t in range(window_size, n):
    # Training window
    train_window = ts_data[t-window_size:t]
    actual = ts_data.iloc[t]
    
    for p in models_to_test:
        try:
            model = AutoReg(train_window, lags=p, trend='c')
            fit = model.fit()
            
            # 1-step ahead forecast
            forecast = fit.predict(start=len(train_window), 
                                  end=len(train_window))[0]
            
            error = actual - forecast
            
            rolling_results[f'AR({p})']['predictions'].append(forecast)
            rolling_results[f'AR({p})']['actuals'].append(actual)
            rolling_results[f'AR({p})']['errors'].append(error)
        except:
            continue

# Compute rolling window metrics
rolling_metrics = []
for model_name in rolling_results.keys():
    if len(rolling_results[model_name]['errors']) > 0:
        errors = np.array(rolling_results[model_name]['errors'])
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        
        rolling_metrics.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'N_Forecasts': len(errors)
        })

rolling_df = pd.DataFrame(rolling_metrics)
print(f"\nRolling Window (size={window_size}):")
print(rolling_df.to_string(index=False))

best_rolling = rolling_df.loc[rolling_df['RMSE'].idxmin()]
print(f"\nBest rolling window: {best_rolling['Model']} (RMSE = {best_rolling['RMSE']:.4f})")

# ===== 3. Expanding Window Validation =====
print("\n" + "="*80)
print("METHOD 3: EXPANDING WINDOW VALIDATION")
print("="*80)

initial_window = 300
expanding_results = {f'AR({p})': {'predictions': [], 'actuals': [], 'errors': []} 
                     for p in models_to_test}

for t in range(initial_window, n):
    # Expanding window (all past data)
    train_window = ts_data[:t]
    actual = ts_data.iloc[t]
    
    for p in models_to_test:
        try:
            model = AutoReg(train_window, lags=p, trend='c')
            fit = model.fit()
            
            # 1-step ahead forecast
            forecast = fit.predict(start=len(train_window),
                                  end=len(train_window))[0]
            
            error = actual - forecast
            
            expanding_results[f'AR({p})']['predictions'].append(forecast)
            expanding_results[f'AR({p})']['actuals'].append(actual)
            expanding_results[f'AR({p})']['errors'].append(error)
        except:
            continue

# Compute expanding window metrics
expanding_metrics = []
for model_name in expanding_results.keys():
    if len(expanding_results[model_name]['errors']) > 0:
        errors = np.array(expanding_results[model_name]['errors'])
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        
        expanding_metrics.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'N_Forecasts': len(errors)
        })

expanding_df = pd.DataFrame(expanding_metrics)
print(f"\nExpanding Window (initial={initial_window}):")
print(expanding_df.to_string(index=False))

best_expanding = expanding_df.loc[expanding_df['RMSE'].idxmin()]
print(f"\nBest expanding window: {best_expanding['Model']} (RMSE = {best_expanding['RMSE']:.4f})")

# ===== 4. Time Series Cross-Validation =====
print("\n" + "="*80)
print("METHOD 4: TIME SERIES CROSS-VALIDATION")
print("="*80)

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

cv_results = {f'AR({p})': [] for p in models_to_test}

for train_idx, test_idx in tscv.split(ts_data):
    train_cv = ts_data.iloc[train_idx]
    test_cv = ts_data.iloc[test_idx]
    
    for p in models_to_test:
        try:
            model = AutoReg(train_cv, lags=p, trend='c')
            fit = model.fit()
            
            # Forecast on test fold
            forecast_cv = fit.predict(start=len(train_cv), 
                                     end=len(train_cv) + len(test_cv) - 1)
            
            rmse_fold = np.sqrt(mean_squared_error(test_cv, forecast_cv))
            cv_results[f'AR({p})'].append(rmse_fold)
        except:
            continue

# Aggregate CV results
cv_metrics = []
for model_name in cv_results.keys():
    if len(cv_results[model_name]) > 0:
        cv_rmse = np.mean(cv_results[model_name])
        cv_std = np.std(cv_results[model_name])
        
        cv_metrics.append({
            'Model': model_name,
            'CV_RMSE': cv_rmse,
            'CV_Std': cv_std
        })

cv_df = pd.DataFrame(cv_metrics)
print(f"\nTime Series CV ({n_splits} splits):")
print(cv_df.to_string(index=False))

best_cv = cv_df.loc[cv_df['CV_RMSE'].idxmin()]
print(f"\nBest CV: {best_cv['Model']} (RMSE = {best_cv['CV_RMSE']:.4f})")

# ===== Diebold-Mariano Test =====
print("\n" + "="*80)
print("DIEBOLD-MARIANO TEST (Model Comparison)")
print("="*80)

# Compare AR(2) vs AR(3) using rolling window errors
errors_ar2 = np.array(rolling_results['AR(2)']['errors'])
errors_ar3 = np.array(rolling_results['AR(3)']['errors'])

# Loss differential (squared errors)
d = errors_ar2**2 - errors_ar3**2
d_mean = np.mean(d)
d_var = np.var(d, ddof=1)
n_dm = len(d)

# DM statistic
dm_stat = d_mean / np.sqrt(d_var / n_dm)
dm_pval = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

print(f"Comparing AR(2) vs AR(3):")
print(f"  Hâ‚€: Equal predictive accuracy")
print(f"  DM Statistic: {dm_stat:.4f}")
print(f"  p-value: {dm_pval:.4f}")

if dm_pval < 0.05:
    winner = "AR(3)" if d_mean > 0 else "AR(2)"
    print(f"  âœ“ Reject Hâ‚€: {winner} has significantly better forecast accuracy")
else:
    print("  Fail to reject: No significant difference in accuracy")

# ===== Visualizations =====
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Plot 1: Time Series with Regime Change
axes[0, 0].plot(ts_data.index, ts_data.values, linewidth=1)
axes[0, 0].axvline(ts_data.index[regime_change], color='red', 
                  linestyle='--', linewidth=2, label='Regime Change')
axes[0, 0].axvline(ts_data.index[train_size], color='green',
                  linestyle='--', linewidth=2, label='Train/Test Split')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Value')
axes[0, 0].set_title('Time Series with Structural Break')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: In-Sample vs OOS Performance
x_pos = np.arange(len(tt_df))
width = 0.35

axes[0, 1].bar(x_pos - width/2, tt_df['In_Sample_RMSE'], width,
              label='In-Sample', alpha=0.8)
axes[0, 1].bar(x_pos + width/2, tt_df['OOS_RMSE'], width,
              label='Out-of-Sample', alpha=0.8)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(tt_df['Model'])
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].set_title('In-Sample vs Out-of-Sample Performance')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Comparison Across Methods
methods_comparison = pd.DataFrame({
    'Train-Test': tt_df['OOS_RMSE'].values,
    'Rolling': rolling_df['RMSE'].values,
    'Expanding': expanding_df['RMSE'].values,
    'CV': cv_df['CV_RMSE'].values
}, index=tt_df['Model'])

methods_comparison.plot(kind='bar', ax=axes[1, 0], alpha=0.8)
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].set_title('RMSE Comparison Across Validation Methods')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
axes[1, 0].legend(title='Method')
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Rolling Forecast Errors Over Time
plot_model = 'AR(2)'
errors_to_plot = rolling_results[plot_model]['errors']
time_idx = ts_data.index[window_size:window_size+len(errors_to_plot)]

axes[1, 1].plot(time_idx, errors_to_plot, linewidth=1, alpha=0.7)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].axvline(ts_data.index[regime_change], color='orange',
                  linestyle='--', linewidth=2, label='Regime Change')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Forecast Error')
axes[1, 1].set_title(f'Rolling Forecast Errors: {plot_model}')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 5: Cumulative Squared Errors
cumsum_errors_ar2 = np.cumsum(np.array(rolling_results['AR(2)']['errors'])**2)
cumsum_errors_ar3 = np.cumsum(np.array(rolling_results['AR(3)']['errors'])**2)

axes[2, 0].plot(time_idx, cumsum_errors_ar2, label='AR(2)', linewidth=2)
axes[2, 0].plot(time_idx, cumsum_errors_ar3, label='AR(3)', linewidth=2)
axes[2, 0].axvline(ts_data.index[regime_change], color='red',
                  linestyle='--', linewidth=2, label='Regime Change')
axes[2, 0].set_xlabel('Time')
axes[2, 0].set_ylabel('Cumulative Squared Error')
axes[2, 0].set_title('Cumulative Performance Comparison')
axes[2, 0].legend()
axes[2, 0].grid(alpha=0.3)

# Plot 6: Out-of-Sample RÂ² Comparison
if 'OOS_R2' in tt_df.columns:
    axes[2, 1].bar(tt_df['Model'], tt_df['OOS_R2'], alpha=0.8)
    axes[2, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[2, 1].set_ylabel('Out-of-Sample RÂ²')
    axes[2, 1].set_title('Out-of-Sample RÂ² (Can Be Negative!)')
    axes[2, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('oos_validation_comprehensive.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Method Comparison:")
print(f"   Train-Test: Best = {best_oos['Model']}, RMSE = {best_oos['OOS_RMSE']:.4f}")
print(f"   Rolling:    Best = {best_rolling['Model']}, RMSE = {best_rolling['RMSE']:.4f}")
print(f"   Expanding:  Best = {best_expanding['Model']}, RMSE = {best_expanding['RMSE']:.4f}")
print(f"   CV:         Best = {best_cv['Model']}, RMSE = {best_cv['CV_RMSE']:.4f}")

print("\n2. Overfitting Evidence:")
for idx, row in tt_df.iterrows():
    print(f"   {row['Model']}: OOS/In-Sample = {row['Overfit_Ratio']:.2f}Ã— "
          f"{'(overfit)' if row['Overfit_Ratio'] > 1.2 else '(good)'}")

print("\n3. Regime Change Impact:")
print(f"   Structural break at t={regime_change}")
print(f"   Models estimated on regime 1 may underperform in regime 2")
print(f"   Rolling/expanding windows adapt better than fixed train-test")

print("\n4. Key Takeaways:")
print("   â€¢ OOS validation essential to detect overfitting")
print("   â€¢ In-sample fit can be misleading")
print("   â€¢ Rolling windows robust to regime changes")
print("   â€¢ Multiple validation methods provide robustness")
print("   â€¢ Statistical tests (DM) formalize comparisons")

print("\n5. Practical Recommendations:")
print("   â€¢ Always validate on truly unseen data")
print("   â€¢ Use rolling/expanding windows for time series")
print("   â€¢ Report multiple metrics (RMSE, MAE, RÂ²_OOS)")
print("   â€¢ Check for structural breaks before modeling")
print("   â€¢ Beware of data leakage and look-ahead bias")
