import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(777)

# ===== Simulate AR(2) Process =====
n = 500
true_p = 2  # True lag order
phi1, phi2 = 0.6, -0.3  # True AR coefficients

# Generate AR(2) process
y = np.zeros(n)
epsilon = np.random.normal(0, 1, n)

for t in range(2, n):
    y[t] = phi1 * y[t-1] + phi2 * y[t-2] + epsilon[t]

# Create time index
time_index = pd.date_range(start='2000-01-01', periods=n, freq='D')
ts_data = pd.Series(y, index=time_index)

print("="*80)
print("INFORMATION CRITERIA: AIC, BIC, AND MODEL SELECTION")
print("="*80)
print(f"\nData Generating Process:")
print(f"  True Model: AR(2)")
print(f"  Ï†â‚ = {phi1}, Ï†â‚‚ = {phi2}")
print(f"  Sample Size: n = {n}")
print(f"\nDescriptive Statistics:")
print(ts_data.describe().round(3))

# ===== Model Selection: AR(p) for p = 0 to 10 =====
print("\n" + "="*80)
print("LAG ORDER SELECTION FOR AR MODELS")
print("="*80)

max_lag = 10
results = []

for p in range(0, max_lag + 1):
    if p == 0:
        # AR(0) is just a constant (white noise)
        model = sm.OLS(ts_data[1:], sm.add_constant(np.ones(n-1)))
        fit = model.fit()
        aic = fit.aic
        bic = fit.bic
        ll = fit.llf
        params = 2  # constant + variance
    else:
        # AR(p) model
        model = AutoReg(ts_data, lags=p, trend='c')
        fit = model.fit()
        aic = fit.aic
        bic = fit.bic
        ll = fit.llf
        params = p + 2  # p AR coefficients + constant + variance
    
    results.append({
        'Lag_Order': p,
        'Log_Likelihood': ll,
        'AIC': aic,
        'BIC': bic,
        'Parameters': params,
        'Model': fit
    })

results_df = pd.DataFrame(results)

print("\nModel Comparison Table:")
print(results_df[['Lag_Order', 'Log_Likelihood', 'AIC', 'BIC', 'Parameters']].to_string(index=False))

# Find optimal lags
aic_optimal = results_df.loc[results_df['AIC'].idxmin(), 'Lag_Order']
bic_optimal = results_df.loc[results_df['BIC'].idxmin(), 'Lag_Order']

print(f"\n" + "="*80)
print("OPTIMAL LAG SELECTION")
print("="*80)
print(f"True lag order: {true_p}")
print(f"AIC selects: p = {aic_optimal}")
print(f"BIC selects: p = {bic_optimal}")

if aic_optimal == true_p:
    print("âœ“ AIC correctly identifies true model")
else:
    print(f"âš  AIC {'overfit' if aic_optimal > true_p else 'underfit'} (selected p={aic_optimal} vs true p={true_p})")

if bic_optimal == true_p:
    print("âœ“ BIC correctly identifies true model")
else:
    print(f"âš  BIC {'overfit' if bic_optimal > true_p else 'underfit'} (selected p={bic_optimal} vs true p={true_p})")

# ===== Î”AIC and Î”BIC Analysis =====
print("\n" + "="*80)
print("Î”AIC AND Î”BIC ANALYSIS")
print("="*80)

results_df['Î”AIC'] = results_df['AIC'] - results_df['AIC'].min()
results_df['Î”BIC'] = results_df['BIC'] - results_df['BIC'].min()

print("\nModel Support Based on Î”AIC:")
print(results_df[['Lag_Order', 'AIC', 'Î”AIC']].to_string(index=False))

print("\nÎ”AIC Interpretation:")
print("  < 2: Substantial support")
print("  4-7: Considerably less support")
print("  > 10: Essentially no support")

# Models with substantial support (Î”AIC < 2)
supported_models = results_df[results_df['Î”AIC'] < 2]
print(f"\nModels with substantial support (Î”AIC < 2): {supported_models['Lag_Order'].tolist()}")

# ===== AICc (Small Sample Correction) =====
print("\n" + "="*80)
print("AICc (SMALL SAMPLE CORRECTION)")
print("="*80)

results_df['AICc'] = results_df.apply(
    lambda row: row['AIC'] + (2 * row['Parameters'] * (row['Parameters'] + 1)) / (n - row['Parameters'] - 1),
    axis=1
)

aicc_optimal = results_df.loc[results_df['AICc'].idxmin(), 'Lag_Order']

print(f"AICc correction: 2k(k+1)/(n-k-1)")
print(f"AICc selects: p = {aicc_optimal}")
print(f"\nComparison:")
print(f"  AIC:  p = {aic_optimal}")
print(f"  AICc: p = {aicc_optimal}")
print(f"  BIC:  p = {bic_optimal}")

# ===== Cross-Validation Comparison =====
print("\n" + "="*80)
print("CROSS-VALIDATION FOR MODEL SELECTION")
print("="*80)

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

cv_results = []

for p in range(1, max_lag + 1):
    cv_mse = []
    
    for train_idx, test_idx in tscv.split(ts_data):
        # Split data
        train_data = ts_data.iloc[train_idx]
        test_data = ts_data.iloc[test_idx]
        
        # Fit model on training data
        model = AutoReg(train_data, lags=p, trend='c')
        fit = model.fit()
        
        # Forecast on test data
        forecast = fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
        
        # Compute MSE
        mse = mean_squared_error(test_data, forecast)
        cv_mse.append(mse)
    
    avg_cv_mse = np.mean(cv_mse)
    cv_results.append({'Lag_Order': p, 'CV_MSE': avg_cv_mse})

cv_results_df = pd.DataFrame(cv_results)
cv_optimal = cv_results_df.loc[cv_results_df['CV_MSE'].idxmin(), 'Lag_Order']

print(f"Time Series Cross-Validation ({n_splits} splits):")
print(cv_results_df.to_string(index=False))

print(f"\nCross-Validation selects: p = {cv_optimal}")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Time Series Data
axes[0, 0].plot(ts_data.index, ts_data.values, linewidth=1)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Value')
axes[0, 0].set_title(f'Simulated AR({true_p}) Process')
axes[0, 0].grid(alpha=0.3)

# Plot 2: AIC vs Lag Order
axes[0, 1].plot(results_df['Lag_Order'], results_df['AIC'], 
               marker='o', linewidth=2, markersize=8, label='AIC')
axes[0, 1].axvline(aic_optimal, color='red', linestyle='--', 
                  linewidth=2, label=f'AIC optimal (p={aic_optimal})')
axes[0, 1].axvline(true_p, color='green', linestyle='--', 
                  linewidth=2, label=f'True (p={true_p})')
axes[0, 1].set_xlabel('Lag Order (p)')
axes[0, 1].set_ylabel('AIC')
axes[0, 1].set_title('Akaike Information Criterion')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: BIC vs Lag Order
axes[0, 2].plot(results_df['Lag_Order'], results_df['BIC'],
               marker='s', linewidth=2, markersize=8, label='BIC', color='orange')
axes[0, 2].axvline(bic_optimal, color='red', linestyle='--',
                  linewidth=2, label=f'BIC optimal (p={bic_optimal})')
axes[0, 2].axvline(true_p, color='green', linestyle='--',
                  linewidth=2, label=f'True (p={true_p})')
axes[0, 2].set_xlabel('Lag Order (p)')
axes[0, 2].set_ylabel('BIC')
axes[0, 2].set_title('Bayesian Information Criterion')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: AIC vs BIC Comparison
axes[1, 0].plot(results_df['Lag_Order'], results_df['AIC'],
               marker='o', linewidth=2, markersize=8, label='AIC')
axes[1, 0].plot(results_df['Lag_Order'], results_df['BIC'],
               marker='s', linewidth=2, markersize=8, label='BIC')
axes[1, 0].axvline(true_p, color='green', linestyle='--',
                  linewidth=2, label=f'True (p={true_p})')
axes[1, 0].set_xlabel('Lag Order (p)')
axes[1, 0].set_ylabel('Information Criterion')
axes[1, 0].set_title('AIC vs BIC (Both on Same Scale)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Î”AIC
axes[1, 1].bar(results_df['Lag_Order'], results_df['Î”AIC'], alpha=0.7)
axes[1, 1].axhline(2, color='orange', linestyle='--', linewidth=2, 
                  label='Î”AIC = 2 (support threshold)')
axes[1, 1].axhline(10, color='red', linestyle='--', linewidth=2,
                  label='Î”AIC = 10 (no support)')
axes[1, 1].set_xlabel('Lag Order (p)')
axes[1, 1].set_ylabel('Î”AIC')
axes[1, 1].set_title('Î”AIC (Model Support)')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, axis='y')

# Plot 6: Cross-Validation MSE
axes[1, 2].plot(cv_results_df['Lag_Order'], cv_results_df['CV_MSE'],
               marker='D', linewidth=2, markersize=8, color='purple', label='CV MSE')
axes[1, 2].axvline(cv_optimal, color='red', linestyle='--',
                  linewidth=2, label=f'CV optimal (p={cv_optimal})')
axes[1, 2].axvline(true_p, color='green', linestyle='--',
                  linewidth=2, label=f'True (p={true_p})')
axes[1, 2].set_xlabel('Lag Order (p)')
axes[1, 2].set_ylabel('Cross-Validation MSE')
axes[1, 2].set_title('Time Series Cross-Validation')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('information_criteria_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Monte Carlo: Selection Consistency =====
print("\n" + "="*80)
print("MONTE CARLO: SELECTION CONSISTENCY")
print("="*80)

n_sim = 500
sample_sizes = [50, 100, 200, 500, 1000]

consistency_results = []

for n_sim_size in sample_sizes:
    aic_correct = 0
    bic_correct = 0
    aic_underfit = 0
    aic_overfit = 0
    bic_underfit = 0
    bic_overfit = 0
    
    for sim in range(n_sim):
        # Generate AR(2) data
        y_sim = np.zeros(n_sim_size)
        eps_sim = np.random.normal(0, 1, n_sim_size)
        
        for t in range(2, n_sim_size):
            y_sim[t] = phi1 * y_sim[t-1] + phi2 * y_sim[t-2] + eps_sim[t]
        
        # Fit models and compute criteria
        aic_values = []
        bic_values = []
        
        for p in range(0, 6):  # Test p=0 to 5
            if p == 0:
                continue
            try:
                model = AutoReg(y_sim, lags=p, trend='c')
                fit = model.fit()
                aic_values.append((p, fit.aic))
                bic_values.append((p, fit.bic))
            except:
                continue
        
        if aic_values:
            aic_selected = min(aic_values, key=lambda x: x[1])[0]
            if aic_selected == true_p:
                aic_correct += 1
            elif aic_selected < true_p:
                aic_underfit += 1
            else:
                aic_overfit += 1
        
        if bic_values:
            bic_selected = min(bic_values, key=lambda x: x[1])[0]
            if bic_selected == true_p:
                bic_correct += 1
            elif bic_selected < true_p:
                bic_underfit += 1
            else:
                bic_overfit += 1
    
    consistency_results.append({
        'Sample_Size': n_sim_size,
        'AIC_Correct': aic_correct / n_sim,
        'BIC_Correct': bic_correct / n_sim,
        'AIC_Underfit': aic_underfit / n_sim,
        'AIC_Overfit': aic_overfit / n_sim,
        'BIC_Underfit': bic_underfit / n_sim,
        'BIC_Overfit': bic_overfit / n_sim
    })

consistency_df = pd.DataFrame(consistency_results)

print(f"\nSelection Accuracy over {n_sim} simulations:")
print(consistency_df.to_string(index=False))

# Visualization
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Correct Selection Rate
axes2[0].plot(consistency_df['Sample_Size'], consistency_df['AIC_Correct'],
             marker='o', linewidth=2, markersize=8, label='AIC')
axes2[0].plot(consistency_df['Sample_Size'], consistency_df['BIC_Correct'],
             marker='s', linewidth=2, markersize=8, label='BIC')
axes2[0].set_xlabel('Sample Size')
axes2[0].set_ylabel('Proportion Correct')
axes2[0].set_title('Selection Consistency: P(Select True Model)')
axes2[0].legend()
axes2[0].grid(alpha=0.3)
axes2[0].set_ylim([0, 1])

# Plot 2: Stacked Bar for Selection Patterns
x_pos = np.arange(len(sample_sizes))
width = 0.35

axes2[1].bar(x_pos - width/2, consistency_df['AIC_Underfit'], width,
            label='Underfit', alpha=0.8, color='blue')
axes2[1].bar(x_pos - width/2, consistency_df['AIC_Correct'], width,
            bottom=consistency_df['AIC_Underfit'], label='Correct', alpha=0.8, color='green')
axes2[1].bar(x_pos - width/2, consistency_df['AIC_Overfit'], width,
            bottom=consistency_df['AIC_Underfit'] + consistency_df['AIC_Correct'],
            label='Overfit', alpha=0.8, color='red')

axes2[1].bar(x_pos + width/2, consistency_df['BIC_Underfit'], width,
            alpha=0.8, color='blue')
axes2[1].bar(x_pos + width/2, consistency_df['BIC_Correct'], width,
            bottom=consistency_df['BIC_Underfit'], alpha=0.8, color='green')
axes2[1].bar(x_pos + width/2, consistency_df['BIC_Overfit'], width,
            bottom=consistency_df['BIC_Underfit'] + consistency_df['BIC_Correct'],
            alpha=0.8, color='red')

axes2[1].set_xticks(x_pos)
axes2[1].set_xticklabels([f'n={n}' for n in sample_sizes], rotation=45)
axes2[1].set_ylabel('Proportion')
axes2[1].set_title('Selection Patterns: AIC (left) vs BIC (right)')
axes2[1].legend()
axes2[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('consistency_simulation.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print(f"\n1. Selection Results (n={n}):")
print(f"   True model: AR({true_p})")
print(f"   AIC selected: AR({aic_optimal})  {'âœ“' if aic_optimal == true_p else 'âœ—'}")
print(f"   BIC selected: AR({bic_optimal})  {'âœ“' if bic_optimal == true_p else 'âœ—'}")
print(f"   CV selected:  AR({cv_optimal})  {'âœ“' if cv_optimal == true_p else 'âœ—'}")

print(f"\n2. Consistency (from Monte Carlo, n=1000):")
print(f"   AIC correct rate: {consistency_df[consistency_df['Sample_Size']==1000]['AIC_Correct'].values[0]:.1%}")
print(f"   BIC correct rate: {consistency_df[consistency_df['Sample_Size']==1000]['BIC_Correct'].values[0]:.1%}")
print(f"   BIC is consistent â†’ P(correct) â†’ 1 as n â†’ âˆž")

print("\n3. Practical Guidelines:")
print("   â€¢ AIC: Better for prediction, allows more complex models")
print("   â€¢ BIC: Better for model identification, more parsimonious")
print("   â€¢ Cross-validation: Gold standard when data permits")
print("   â€¢ Report both AIC and BIC for transparency")

print("\n4. When to Use Each:")
print("   AIC: Forecasting, small samples, avoid underfitting")
print("   BIC: Theory testing, large samples, prefer parsimony")
print("   AICc: Small samples (n/k < 40)")
print("   CV: Sufficient data for train/test split")

print("\n5. Key Insight:")
print("   Information criteria balance fit and complexity.")
print("   No single criterion is universally best.")
print("   Consider context, sample size, and objective.")
