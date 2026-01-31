import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.robust.robust_linear_model import RLM

# =====================================
# GENERATE DATA WITH OUTLIERS AND INFLUENTIAL POINTS
# =====================================
np.random.seed(42)
n = 100

# Generate normal observations
X_normal = np.random.uniform(1, 10, n)
epsilon = np.random.normal(0, 1, n)
Y_normal = 3 + 2*X_normal + epsilon

# Inject outliers and influential points
# Type 1: Outlier in Y (vertical outlier) - high residual, low leverage
X_outlier_Y = 5.5
Y_outlier_Y = 3 + 2*X_outlier_Y + 15  # Large positive error

# Type 2: Outlier in X (leverage point) - extreme X, on regression line
X_outlier_X = 18
Y_outlier_X = 3 + 2*X_outlier_X + np.random.normal(0, 1)  # Follows true model

# Type 3: Influential point - extreme X AND large residual
X_influential = 19
Y_influential = 3 + 2*X_influential - 12  # Large negative error

# Combine data
X = np.append(X_normal, [X_outlier_Y, X_outlier_X, X_influential])
Y = np.append(Y_normal, [Y_outlier_Y, Y_outlier_X, Y_influential])

# Labels for special points
labels = ['Normal'] * n + ['Outlier (Y)', 'Leverage (X)', 'Influential']

data = pd.DataFrame({'X': X, 'Y': Y, 'Label': labels})

print("="*70)
print("OUTLIER AND INFLUENCE DIAGNOSTICS")
print("="*70)

# =====================================
# FIT OLS MODEL
# =====================================
X_const = add_constant(data['X'])
model = OLS(data['Y'], X_const).fit()

print(f"\nOLS Regression: Y = Î²â‚€ + Î²â‚X + Îµ")
print(f"   Î²Ì‚â‚€ = {model.params[0]:.4f}")
print(f"   Î²Ì‚â‚ = {model.params[1]:.4f}")
print(f"   (True parameters: Î²â‚€=3, Î²â‚=2)")

# =====================================
# COMPUTE DIAGNOSTICS
# =====================================
influence = OLSInfluence(model)

# Residuals
residuals = model.resid
standardized_resid = influence.resid_studentized_internal
studentized_resid = influence.resid_studentized_external

# Leverage
leverage = influence.hat_matrix_diag
k = X_const.shape[1]  # Number of parameters (including intercept)
leverage_threshold = 2 * k / len(data)

# Cook's Distance
cooks_d = influence.cooks_distance[0]
cooks_threshold = 4 / len(data)

# DFBETAS
dfbetas = influence.dfbetas
dfbetas_threshold = 2 / np.sqrt(len(data))

# DFFITS
dffits = influence.dffits[0]
dffits_threshold = 2 * np.sqrt(k / len(data))

# Add to dataframe
data['Residual'] = residuals
data['Standardized_Resid'] = standardized_resid
data['Studentized_Resid'] = studentized_resid
data['Leverage'] = leverage
data['Cooks_D'] = cooks_d
data['DFBETAS_X'] = dfbetas[:, 1]  # DFBETAS for slope coefficient
data['DFFITS'] = dffits

# =====================================
# IDENTIFY PROBLEMATIC OBSERVATIONS
# =====================================
print("\n" + "="*70)
print("DIAGNOSTIC THRESHOLDS")
print("="*70)
print(f"   Leverage threshold (2k/n): {leverage_threshold:.4f}")
print(f"   Cook's D threshold (4/n): {cooks_threshold:.4f}")
print(f"   DFBETAS threshold (2/âˆšn): {dfbetas_threshold:.4f}")
print(f"   DFFITS threshold (2âˆš(k/n)): {dffits_threshold:.4f}")
print(f"   Studentized residual threshold: |t| > 3")

# Flag observations
outliers_Y = data[np.abs(data['Studentized_Resid']) > 3]
high_leverage = data[data['Leverage'] > leverage_threshold]
influential = data[data['Cooks_D'] > cooks_threshold]
dfbetas_high = data[np.abs(data['DFBETAS_X']) > dfbetas_threshold]

print("\n" + "="*70)
print("IDENTIFIED PROBLEMATIC OBSERVATIONS")
print("="*70)

print(f"\n1. OUTLIERS IN Y (|Studentized Residual| > 3):")
if len(outliers_Y) > 0:
    for idx in outliers_Y.index:
        print(f"   Obs {idx}: X={data.loc[idx, 'X']:.2f}, Y={data.loc[idx, 'Y']:.2f}, "
              f"t*={data.loc[idx, 'Studentized_Resid']:.2f}, "
              f"Label={data.loc[idx, 'Label']}")
else:
    print("   None detected")

print(f"\n2. HIGH LEVERAGE POINTS (h > {leverage_threshold:.4f}):")
if len(high_leverage) > 0:
    for idx in high_leverage.index[-5:]:  # Show last 5 (including injected points)
        print(f"   Obs {idx}: X={data.loc[idx, 'X']:.2f}, Y={data.loc[idx, 'Y']:.2f}, "
              f"h={data.loc[idx, 'Leverage']:.4f}, "
              f"Label={data.loc[idx, 'Label']}")
else:
    print("   None detected")

print(f"\n3. INFLUENTIAL POINTS (Cook's D > {cooks_threshold:.4f}):")
if len(influential) > 0:
    for idx in influential.index:
        print(f"   Obs {idx}: X={data.loc[idx, 'X']:.2f}, Y={data.loc[idx, 'Y']:.2f}, "
              f"D={data.loc[idx, 'Cooks_D']:.4f}, "
              f"h={data.loc[idx, 'Leverage']:.4f}, "
              f"t*={data.loc[idx, 'Studentized_Resid']:.2f}, "
              f"Label={data.loc[idx, 'Label']}")
else:
    print("   None detected")

print(f"\n4. LARGE DFBETAS FOR SLOPE (|DFBETAS| > {dfbetas_threshold:.4f}):")
if len(dfbetas_high) > 0:
    for idx in dfbetas_high.index[-5:]:
        print(f"   Obs {idx}: X={data.loc[idx, 'X']:.2f}, "
              f"DFBETAS_X={data.loc[idx, 'DFBETAS_X']:.4f}, "
              f"Label={data.loc[idx, 'Label']}")
else:
    print("   None detected")

# =====================================
# COMPARE MODELS: WITH AND WITHOUT INFLUENTIAL POINTS
# =====================================
print("\n" + "="*70)
print("MODEL COMPARISON: ORIGINAL VS ROBUST")
print("="*70)

# Remove influential points
data_cleaned = data[data['Cooks_D'] <= cooks_threshold]
X_cleaned = add_constant(data_cleaned['X'])
model_cleaned = OLS(data_cleaned['Y'], X_cleaned).fit()

print(f"\n1. ORIGINAL MODEL (n={len(data)}):")
print(f"   Î²Ì‚â‚€ = {model.params[0]:.4f}")
print(f"   Î²Ì‚â‚ = {model.params[1]:.4f}")
print(f"   RÂ² = {model.rsquared:.4f}")

print(f"\n2. MODEL WITHOUT INFLUENTIAL POINTS (n={len(data_cleaned)}):")
print(f"   Î²Ì‚â‚€ = {model_cleaned.params[0]:.4f}")
print(f"   Î²Ì‚â‚ = {model_cleaned.params[1]:.4f}")
print(f"   RÂ² = {model_cleaned.rsquared:.4f}")

print(f"\n   Change in Î²Ì‚â‚: {model_cleaned.params[1] - model.params[1]:.4f}")
print(f"   (True Î²â‚ = 2.0)")

# =====================================
# ROBUST REGRESSION (M-ESTIMATOR)
# =====================================
print("\n" + "="*70)
print("ROBUST REGRESSION (M-ESTIMATOR)")
print("="*70)

# Fit robust model using Huber M-estimator
model_robust = RLM(data['Y'], X_const, M=None).fit()  # Default: Huber

print(f"\n3. ROBUST REGRESSION (Huber M-estimator, n={len(data)}):")
print(f"   Î²Ì‚â‚€ = {model_robust.params[0]:.4f}")
print(f"   Î²Ì‚â‚ = {model_robust.params[1]:.4f}")

# Weights assigned to each observation
weights = model_robust.weights

print(f"\n   Observations with low weights (downweighted outliers):")
low_weight_idx = np.where(weights < 0.5)[0]
for idx in low_weight_idx:
    print(f"   Obs {idx}: X={data.loc[idx, 'X']:.2f}, Y={data.loc[idx, 'Y']:.2f}, "
          f"Weight={weights[idx]:.3f}, Label={data.loc[idx, 'Label']}")

print(f"\n   Robust model automatically downweights influential points âœ“")

# =====================================
# VISUALIZATION
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Leverage vs Studentized Residuals
axes[0, 0].scatter(data['Leverage'], data['Studentized_Resid'], alpha=0.6, s=50)

# Highlight special points
special_idx = data[data['Label'] != 'Normal'].index
for idx in special_idx:
    axes[0, 0].scatter(data.loc[idx, 'Leverage'], data.loc[idx, 'Studentized_Resid'],
                      s=200, alpha=0.7, edgecolors='red', linewidths=2)
    axes[0, 0].annotate(data.loc[idx, 'Label'], 
                       (data.loc[idx, 'Leverage'], data.loc[idx, 'Studentized_Resid']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

axes[0, 0].axhline(3, color='red', linestyle='--', alpha=0.5, label='|t*| = 3')
axes[0, 0].axhline(-3, color='red', linestyle='--', alpha=0.5)
axes[0, 0].axvline(leverage_threshold, color='blue', linestyle='--', alpha=0.5, 
                   label=f'h = {leverage_threshold:.3f}')
axes[0, 0].set_xlabel('Leverage (h)')
axes[0, 0].set_ylabel('Studentized Residual (t*)')
axes[0, 0].set_title('Leverage vs Studentized Residuals')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Cook's Distance
axes[0, 1].bar(range(len(data)), data['Cooks_D'], alpha=0.7)
axes[0, 1].axhline(cooks_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f"Threshold = {cooks_threshold:.4f}")

# Highlight influential points
for idx in influential.index:
    axes[0, 1].bar(idx, data.loc[idx, 'Cooks_D'], color='red', alpha=0.8)
    axes[0, 1].text(idx, data.loc[idx, 'Cooks_D'], data.loc[idx, 'Label'], 
                   rotation=90, va='bottom', fontsize=8)

axes[0, 1].set_xlabel('Observation Index')
axes[0, 1].set_ylabel("Cook's Distance")
axes[0, 1].set_title("Cook's Distance (Influence Measure)")
axes[0, 1].legend()
axes[0, 1].set_ylim(0, max(data['Cooks_D']) * 1.2)

# Plot 3: Fitted values comparison
X_plot = np.linspace(data['X'].min(), data['X'].max(), 200)
y_original = model.params[0] + model.params[1] * X_plot
y_cleaned = model_cleaned.params[0] + model_cleaned.params[1] * X_plot
y_robust = model_robust.params[0] + model_robust.params[1] * X_plot
y_true = 3 + 2 * X_plot

axes[1, 0].scatter(data['X'], data['Y'], alpha=0.4, s=30, label='Data')

# Highlight special points
for idx in special_idx:
    axes[1, 0].scatter(data.loc[idx, 'X'], data.loc[idx, 'Y'],
                      s=200, alpha=0.7, edgecolors='red', linewidths=2)

axes[1, 0].plot(X_plot, y_true, 'k--', linewidth=2, label='True Model (Î²â‚=2.0)')
axes[1, 0].plot(X_plot, y_original, 'b-', linewidth=2, label=f'OLS All (Î²â‚={model.params[1]:.2f})')
axes[1, 0].plot(X_plot, y_cleaned, 'g-', linewidth=2, label=f'OLS Clean (Î²â‚={model_cleaned.params[1]:.2f})')
axes[1, 0].plot(X_plot, y_robust, 'r-', linewidth=2, label=f'Robust (Î²â‚={model_robust.params[1]:.2f})')

axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')
axes[1, 0].set_title('Regression Lines: Original vs Cleaned vs Robust')
axes[1, 0].legend()
axes[1, 0].set_ylim(-10, 60)

# Plot 4: DFBETAS for slope coefficient
axes[1, 1].bar(range(len(data)), data['DFBETAS_X'], alpha=0.7)
axes[1, 1].axhline(dfbetas_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f"Threshold = Â±{dfbetas_threshold:.3f}")
axes[1, 1].axhline(-dfbetas_threshold, color='red', linestyle='--', linewidth=2)

# Highlight high DFBETAS
for idx in dfbetas_high.index:
    if np.abs(data.loc[idx, 'DFBETAS_X']) > dfbetas_threshold:
        axes[1, 1].bar(idx, data.loc[idx, 'DFBETAS_X'], color='red', alpha=0.8)

axes[1, 1].set_xlabel('Observation Index')
axes[1, 1].set_ylabel('DFBETAS (Slope Coefficient)')
axes[1, 1].set_title('Influence on Slope Coefficient (DFBETAS)')
axes[1, 1].legend()
axes[1, 1].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("Diagnostic statistics successfully identified problematic observations:")
print(f"â€¢ Outlier in Y (Obs {n}): High |t*|={data.loc[n, 'Studentized_Resid']:.2f}, "
      f"Low h={data.loc[n, 'Leverage']:.3f}, Moderate D={data.loc[n, 'Cooks_D']:.3f}")
print(f"â€¢ Leverage point (Obs {n+1}): High h={data.loc[n+1, 'Leverage']:.3f}, "
      f"Low |t*|={np.abs(data.loc[n+1, 'Studentized_Resid']):.2f}, Low D={data.loc[n+1, 'Cooks_D']:.3f}")
print(f"â€¢ Influential point (Obs {n+2}): High h={data.loc[n+2, 'Leverage']:.3f}, "
      f"High |t*|={np.abs(data.loc[n+2, 'Studentized_Resid']):.2f}, High D={data.loc[n+2, 'Cooks_D']:.3f}")
print(f"\nRemedy effectiveness:")
print(f"â€¢ OLS without influential: Î²Ì‚â‚={model_cleaned.params[1]:.3f} (close to true Î²â‚=2.0)")
print(f"â€¢ Robust M-estimator: Î²Ì‚â‚={model_robust.params[1]:.3f} (automatic downweighting)")
print(f"â€¢ Original OLS: Î²Ì‚â‚={model.params[1]:.3f} (biased by influential point)")
