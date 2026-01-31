# Out-of-Sample Validation

## 1. Concept Skeleton
**Definition:** Model evaluation using data not used in estimation; assesses predictive performance; guards against overfitting; train-test split methodology  
**Purpose:** Estimate true out-of-sample forecast error; compare model predictive accuracy; detect overfitting; validate model generalization  
**Prerequisites:** Train-test split, forecast error metrics, rolling windows, cross-validation, backtesting

## 2. Comparative Framing
| Method | In-Sample Fit | Hold-Out Validation | Rolling Window | Expanding Window | Cross-Validation |
|--------|---------------|---------------------|----------------|------------------|------------------|
| **Data Usage** | All for estimation | Fixed split | Sliding window | Growing window | Multiple splits |
| **Robustness** | None (overfit) | Single test | Multiple tests | Multiple tests | Most robust |
| **Time Series** | Not valid | Valid | Best practice | Good | k-fold CV |
| **Sample Efficiency** | 100% train | e.g., 80/20 | Varies | Efficient | Efficient |
| **Computation** | Lowest | Low | High | Moderate | High |

## 3. Examples + Counterexamples

**Classic Example:**  
Stock return forecast: Train on 2000-2018, test on 2019-2020. Model with lowest in-sample R² may have best out-of-sample MSE. Overfitting evident.

**Failure Case:**  
Using future data in feature engineering (look-ahead bias): Normalize using full dataset statistics including test set. Invalid out-of-sample performance.

**Edge Case:**  
Structural break at train-test boundary: Model estimated on pre-break data fails post-break. Need regime-aware validation or recursive estimation.

## 4. Layer Breakdown
```
Out-of-Sample Validation Framework:
├─ Motivation:
│   ├─ In-Sample Overfitting:
│   │   ├─ R² always increases with more variables
│   │   ├─ Complex models fit noise, not signal
│   │   └─ Training error ≠ generalization error
│   ├─ Prediction Goal:
│   │   ├─ Care about future, unseen data performance
│   │   └─ In-sample fit insufficient for prediction
│   ├─ Model Comparison:
│   │   ├─ Out-of-sample metrics provide honest comparison
│   │   └─ Prevents data mining and p-hacking
│   └─ Economic Significance:
│       └─ Statistical fit ≠ economic value
├─ Basic Train-Test Split:
│   ├─ Procedure:
│   │   ├─ Split data: D = {D_train, D_test}
│   │   ├─ Train: Estimate θ̂ on D_train
│   │   ├─ Test: Evaluate predictions on D_test
│   │   └─ Never use D_test for model selection/tuning
│   ├─ Split Ratios:
│   │   ├─ 80/20: Common for large datasets
│   │   ├─ 70/30: More conservative
│   │   ├─ 90/10: Small test sets if n large
│   │   └─ Time series: Last T_test periods as test
│   ├─ Temporal Order (Time Series):
│   │   ├─ Train on earlier data, test on later data
│   │   ├─ Never train on future, test on past
│   │   └─ Respects time structure
│   └─ Random Split (Cross-Section):
│       ├─ Shuffle data before split if IID
│       └─ Stratify if imbalanced classes
├─ Forecast Error Metrics:
│   ├─ Mean Squared Error (MSE):
│   │   ├─ MSE = (1/T)Σ(ŷₜ - yₜ)²
│   │   ├─ Penalizes large errors heavily
│   │   └─ Most common for regression
│   ├─ Root Mean Squared Error (RMSE):
│   │   ├─ RMSE = √MSE
│   │   └─ Same units as y
│   ├─ Mean Absolute Error (MAE):
│   │   ├─ MAE = (1/T)Σ|ŷₜ - yₜ|
│   │   └─ Less sensitive to outliers than MSE
│   ├─ Mean Absolute Percentage Error (MAPE):
│   │   ├─ MAPE = (100/T)Σ|ŷₜ - yₜ|/|yₜ|
│   │   ├─ Scale-independent
│   │   └─ Undefined if yₜ = 0
│   ├─ Out-of-Sample R² (R²_OOS):
│   │   ├─ R²_OOS = 1 - Σ(ŷₜ - yₜ)²/Σ(yₜ - ȳ_train)²
│   │   ├─ Can be negative!
│   │   └─ Compares to naive forecast (historical mean)
│   └─ Diebold-Mariano Test:
│       ├─ Test H₀: Equal predictive accuracy
│       └─ Compares two forecasting models
├─ Rolling Window Validation:
│   ├─ Procedure:
│   │   ├─ Fix window size W
│   │   ├─ For t = W+1, ..., T:
│   │   │   ├─ Train on [t-W, t-1]
│   │   │   ├─ Forecast for t
│   │   │   └─ Evaluate error
│   │   └─ Aggregate forecast errors
│   ├─ Advantages:
│   │   ├─ Multiple out-of-sample tests
│   │   ├─ Robust to single period anomalies
│   │   └─ Mimics real-time forecasting
│   ├─ Window Size Choice:
│   │   ├─ Larger W: More stable estimates, less adaptive
│   │   ├─ Smaller W: More adaptive, more volatile
│   │   └─ Common: W = 5 years for monthly data
│   └─ Variants:
│       ├─ Fixed window: Constant W (rolling)
│       └─ Recursive/Expanding: W grows each step
├─ Expanding Window Validation:
│   ├─ Procedure:
│   │   ├─ For t = W+1, ..., T:
│   │   │   ├─ Train on [1, t-1] (all past data)
│   │   │   ├─ Forecast for t
│   │   │   └─ Evaluate error
│   │   └─ Training set grows each step
│   ├─ Advantages:
│   │   ├─ Uses all available information
│   │   ├─ Efficient with data
│   │   └─ More stable than rolling window
│   └─ Disadvantages:
│       ├─ Less adaptive to regime changes
│       └─ Old data may be irrelevant
├─ Time Series Cross-Validation:
│   ├─ Blocked CV: Respect temporal order
│   │   ├─ Split 1: Train[1:n₁], Test[n₁+1:n₁+h]
│   │   ├─ Split 2: Train[1:n₂], Test[n₂+1:n₂+h]
│   │   └─ k splits with increasing training size
│   ├─ TimeSeriesSplit (sklearn):
│   │   └─ Systematic expanding window splits
│   └─ Blocked Cross-Validation:
│       └─ Leave out entire time blocks
├─ Walk-Forward Analysis:
│   ├─ Common in Trading Systems:
│   │   ├─ In-sample period: Optimize parameters
│   │   ├─ Out-of-sample period: Fixed parameters
│   │   ├─ Roll forward: Re-optimize periodically
│   │   └─ Anchored vs unanchored
│   ├─ Parameter Stability:
│   │   └─ Check if parameters change dramatically
│   └─ Overfitting Detection:
│       └─ Large in-sample vs OOS performance gap
├─ Model Comparison:
│   ├─ Nested Models:
│   │   ├─ Encompassing tests: Does complex model beat simple?
│   │   └─ F-test for in-sample, DM test for OOS
│   ├─ Non-Nested Models:
│   │   ├─ Direct OOS metric comparison
│   │   ├─ Diebold-Mariano test
│   │   └─ Model confidence set (MCS)
│   ├─ Relative Performance:
│   │   ├─ MSE_ratio = MSE_model / MSE_benchmark
│   │   └─ < 1 indicates improvement
│   └─ Statistical Significance:
│       └─ Bootstrap or DM test for difference
├─ Common Pitfalls:
│   ├─ Look-Ahead Bias:
│   │   ├─ Using future info in feature creation
│   │   ├─ Standardizing with full dataset stats
│   │   └─ Survival bias in historical data
│   ├─ Data Leakage:
│   │   ├─ Test data influences training
│   │   ├─ Feature selection on full dataset
│   │   └─ Parameter tuning on test set
│   ├─ Multiple Testing:
│   │   ├─ Testing many models on same test set
│   │   └─ Inflates false discovery rate
│   ├─ Small Test Sets:
│   │   ├─ High variance in OOS metrics
│   │   └─ Unreliable conclusions
│   └─ Ignoring Temporal Dynamics:
│       └─ Random CV on time series data
├─ Best Practices:
│   ├─ Hold-Out Set: Never touched until final evaluation
│   ├─ Validation Set: For hyperparameter tuning
│   ├─ Training Set: For parameter estimation
│   ├─ Feature Engineering: Only on training data
│   ├─ Multiple Metrics: Report RMSE, MAE, R²_OOS
│   ├─ Uncertainty: Confidence intervals via bootstrap
│   └─ Realistic Assumptions: Match production constraints
└─ Advanced Topics:
    ├─ Combinatorial Purged CV (CPCV): For financial data with overlapping labels
    ├─ Blocked Bootstrap: Preserve time series structure
    ├─ Model Averaging: Combine forecasts from multiple models
    └─ Online Learning: Update model with each new observation
```

**Interaction:** Split data (train/test) → Estimate on train → Forecast on test → Compute OOS metrics → Compare models

## 5. Mini-Project
Implement various out-of-sample validation schemes and compare results:
```python
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
print(f"  Regime 1 (t=1-{regime_change}): AR(2) with φ₁={phi1_regime1}, φ₂={phi2_regime1}")
print(f"  Regime 2 (t={regime_change+1}-{n}): AR(2) with φ₁={phi1_regime2}, φ₂={phi2_regime2}")
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
    
    # Out-of-sample R²
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
print(f"  H₀: Equal predictive accuracy")
print(f"  DM Statistic: {dm_stat:.4f}")
print(f"  p-value: {dm_pval:.4f}")

if dm_pval < 0.05:
    winner = "AR(3)" if d_mean > 0 else "AR(2)"
    print(f"  ✓ Reject H₀: {winner} has significantly better forecast accuracy")
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

# Plot 6: Out-of-Sample R² Comparison
if 'OOS_R2' in tt_df.columns:
    axes[2, 1].bar(tt_df['Model'], tt_df['OOS_R2'], alpha=0.8)
    axes[2, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[2, 1].set_ylabel('Out-of-Sample R²')
    axes[2, 1].set_title('Out-of-Sample R² (Can Be Negative!)')
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
    print(f"   {row['Model']}: OOS/In-Sample = {row['Overfit_Ratio']:.2f}× "
          f"{'(overfit)' if row['Overfit_Ratio'] > 1.2 else '(good)'}")

print("\n3. Regime Change Impact:")
print(f"   Structural break at t={regime_change}")
print(f"   Models estimated on regime 1 may underperform in regime 2")
print(f"   Rolling/expanding windows adapt better than fixed train-test")

print("\n4. Key Takeaways:")
print("   • OOS validation essential to detect overfitting")
print("   • In-sample fit can be misleading")
print("   • Rolling windows robust to regime changes")
print("   • Multiple validation methods provide robustness")
print("   • Statistical tests (DM) formalize comparisons")

print("\n5. Practical Recommendations:")
print("   • Always validate on truly unseen data")
print("   • Use rolling/expanding windows for time series")
print("   • Report multiple metrics (RMSE, MAE, R²_OOS)")
print("   • Check for structural breaks before modeling")
print("   • Beware of data leakage and look-ahead bias")
```

## 6. Challenge Round
When does out-of-sample validation fail or mislead?
- **Look-ahead bias**: Using future information in feature engineering → Invalid OOS performance, appears better than reality
- **Multiple testing**: Testing many models on same test set → Overfitting to test set, inflated performance
- **Small test sets**: High variance in OOS metrics → Unreliable conclusions, need larger validation sets
- **Regime changes**: Structural break at train-test boundary → Past irrelevant for future, need adaptive methods
- **Survivor bias**: Using only currently available assets → Overestimates historical performance
- **Data leakage**: Test data influences training (normalization, feature selection) → Inflated OOS results

## 7. Key References
- [Campbell & Thompson (2008) - Predicting Excess Stock Returns OOS](https://doi.org/10.1093/rfs/hhm055)
- [Diebold & Mariano (1995) - Comparing Predictive Accuracy](https://doi.org/10.1080/07350015.1995.10524599)
- [Inoue & Kilian (2005) - In-Sample or Out-of-Sample Tests](https://doi.org/10.1002/jae.802)

---
**Status:** Essential validation technique | **Complements:** Cross-Validation, Information Criteria, Forecast Evaluation, Backtesting
