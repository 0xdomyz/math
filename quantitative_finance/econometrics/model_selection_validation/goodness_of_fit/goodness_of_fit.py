import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.api import OLS, Logit, add_constant
from statsmodels.discrete.discrete_model import Probit
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# =====================================
# LINEAR REGRESSION: RÂ² and Adjusted RÂ²
# =====================================
print("="*70)
print("LINEAR REGRESSION GOODNESS-OF-FIT")
print("="*70)

np.random.seed(42)
n = 200

# Generate data: Y = Î²â‚€ + Î²â‚Xâ‚ + Î²â‚‚Xâ‚‚ + Î²â‚ƒXâ‚ƒ + Îµ
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
X3 = np.random.normal(0, 1, n)  # Weak predictor
X4 = np.random.normal(0, 1, n)  # Irrelevant noise

epsilon = np.random.normal(0, 2, n)
Y = 5 + 2*X1 + 1.5*X2 + 0.3*X3 + epsilon

data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4})

def calculate_r_squared(y_true, y_pred, y_mean):
    """Calculate RÂ² manually."""
    TSS = np.sum((y_true - y_mean)**2)
    RSS = np.sum((y_true - y_pred)**2)
    ESS = np.sum((y_pred - y_mean)**2)
    R2 = 1 - RSS/TSS
    return {'R2': R2, 'TSS': TSS, 'RSS': RSS, 'ESS': ESS}

def calculate_adjusted_r_squared(R2, n, k):
    """Calculate adjusted RÂ² given RÂ², sample size n, and # predictors k."""
    return 1 - (1 - R2) * (n - 1) / (n - k)

# Model 1: Strong predictors only (X1, X2)
X_strong = add_constant(data[['X1', 'X2']])
model_strong = OLS(data['Y'], X_strong).fit()

print("\nModel 1: Y ~ X1 + X2 (Strong predictors only)")
print(f"   RÂ²: {model_strong.rsquared:.4f}")
print(f"   Adjusted RÂ²: {model_strong.rsquared_adj:.4f}")
print(f"   RMSE: {np.sqrt(model_strong.mse_resid):.4f}")

# Model 2: Add weak predictor (X3)
X_weak = add_constant(data[['X1', 'X2', 'X3']])
model_weak = OLS(data['Y'], X_weak).fit()

print("\nModel 2: Y ~ X1 + X2 + X3 (Add weak predictor)")
print(f"   RÂ²: {model_weak.rsquared:.4f}")
print(f"   Adjusted RÂ²: {model_weak.rsquared_adj:.4f}")
print(f"   Change in RÂ²: +{model_weak.rsquared - model_strong.rsquared:.4f}")
print(f"   Change in Adjusted RÂ²: +{model_weak.rsquared_adj - model_strong.rsquared_adj:.4f}")

# Model 3: Add noise predictor (X4)
X_noise = add_constant(data[['X1', 'X2', 'X3', 'X4']])
model_noise = OLS(data['Y'], X_noise).fit()

print("\nModel 3: Y ~ X1 + X2 + X3 + X4 (Add irrelevant noise)")
print(f"   RÂ²: {model_noise.rsquared:.4f}")
print(f"   Adjusted RÂ²: {model_noise.rsquared_adj:.4f}")
print(f"   Change in RÂ²: +{model_noise.rsquared - model_weak.rsquared:.4f}")
print(f"   Change in Adjusted RÂ²: {model_noise.rsquared_adj - model_weak.rsquared_adj:.4f} (DECREASES!)")

# Manual verification
y_mean = data['Y'].mean()
result_manual = calculate_r_squared(data['Y'], model_strong.fittedvalues, y_mean)
R2_adj_manual = calculate_adjusted_r_squared(result_manual['R2'], n, k=2)

print(f"\n   Manual RÂ² verification: {result_manual['R2']:.4f} (matches {model_strong.rsquared:.4f} âœ“)")
print(f"   Manual Adjusted RÂ²: {R2_adj_manual:.4f} (matches {model_strong.rsquared_adj:.4f} âœ“)")

# Variance decomposition
print(f"\n   Variance decomposition: TSS = {result_manual['TSS']:.2f}")
print(f"                           ESS = {result_manual['ESS']:.2f} (explained)")
print(f"                           RSS = {result_manual['RSS']:.2f} (residual)")
print(f"                           ESS + RSS = {result_manual['ESS'] + result_manual['RSS']:.2f} âœ“")

# =====================================
# LOGISTIC REGRESSION: Pseudo-RÂ²
# =====================================
print("\n" + "="*70)
print("LOGISTIC REGRESSION PSEUDO-RÂ²")
print("="*70)

# Generate binary outcome data
np.random.seed(123)
n = 500

Z1 = np.random.normal(0, 1, n)
Z2 = np.random.normal(0, 1, n)
Z3 = np.random.normal(0, 1, n)

# Logit link: P(Y=1|Z) = 1/(1 + exp(-(Î²â‚€ + Î²â‚Zâ‚ + Î²â‚‚Zâ‚‚)))
linear_pred = -0.5 + 1.2*Z1 + 0.8*Z2
prob = 1 / (1 + np.exp(-linear_pred))
Y_binary = (np.random.uniform(0, 1, n) < prob).astype(int)

data_logit = pd.DataFrame({'Y': Y_binary, 'Z1': Z1, 'Z2': Z2, 'Z3': Z3})

def calculate_pseudo_r2(model_full, model_null):
    """Calculate various pseudo-RÂ² measures."""
    ll_full = model_full.llf  # Log-likelihood of full model
    ll_null = model_null.llf  # Log-likelihood of null (intercept-only)
    n = model_full.nobs
    
    # McFadden's RÂ²
    mcfadden = 1 - ll_full/ll_null
    
    # Cox-Snell RÂ²
    cox_snell = 1 - np.exp((2/n) * (ll_null - ll_full))
    
    # Nagelkerke (Cragg-Uhler) RÂ²: Rescale Cox-Snell to [0,1]
    cox_snell_max = 1 - np.exp((2/n) * ll_null)
    nagelkerke = cox_snell / cox_snell_max
    
    # Efron's RÂ²
    y_true = model_full.model.endog
    y_pred = model_full.predict()
    y_mean = y_true.mean()
    efron = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_mean)**2)
    
    return {
        'McFadden': mcfadden,
        'Cox_Snell': cox_snell,
        'Nagelkerke': nagelkerke,
        'Efron': efron,
        'LL_full': ll_full,
        'LL_null': ll_null
    }

# Null model (intercept only)
X_null = add_constant(np.ones(n))
model_null = Logit(data_logit['Y'], X_null).fit(disp=0)

# Full model (Z1, Z2)
X_full = add_constant(data_logit[['Z1', 'Z2']])
model_full = Logit(data_logit['Y'], X_full).fit(disp=0)

# Model with noise predictor (Z1, Z2, Z3)
X_noise_logit = add_constant(data_logit[['Z1', 'Z2', 'Z3']])
model_noise_logit = Logit(data_logit['Y'], X_noise_logit).fit(disp=0)

# Calculate pseudo-RÂ²
pseudo_r2_full = calculate_pseudo_r2(model_full, model_null)
pseudo_r2_noise = calculate_pseudo_r2(model_noise_logit, model_null)

print("\nFull Model: Y ~ Z1 + Z2")
print(f"   Log-Likelihood (null): {pseudo_r2_full['LL_null']:.4f}")
print(f"   Log-Likelihood (full): {pseudo_r2_full['LL_full']:.4f}")
print(f"   McFadden RÂ²: {pseudo_r2_full['McFadden']:.4f}")
print(f"   Cox-Snell RÂ²: {pseudo_r2_full['Cox_Snell']:.4f}")
print(f"   Nagelkerke RÂ²: {pseudo_r2_full['Nagelkerke']:.4f}")
print(f"   Efron RÂ²: {pseudo_r2_full['Efron']:.4f}")

if pseudo_r2_full['McFadden'] >= 0.20:
    print(f"   â†’ McFadden RÂ² â‰¥ 0.20 indicates excellent fit")
elif pseudo_r2_full['McFadden'] >= 0.10:
    print(f"   â†’ McFadden RÂ² â‰¥ 0.10 indicates good fit")

print("\nModel with Noise: Y ~ Z1 + Z2 + Z3 (add irrelevant predictor)")
print(f"   McFadden RÂ²: {pseudo_r2_noise['McFadden']:.4f}")
print(f"   Change: +{pseudo_r2_noise['McFadden'] - pseudo_r2_full['McFadden']:.4f}")
print(f"   (Unlike OLS, pseudo-RÂ² may decrease with noise predictors)")

# =====================================
# CLASSIFICATION METRICS
# =====================================
print("\n" + "="*70)
print("CLASSIFICATION METRICS")
print("="*70)

# Predictions
y_pred_prob = model_full.predict()
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(data_logit['Y'], y_pred_class)
TN, FP, FN, TP = cm.ravel()

accuracy = (TP + TN) / n
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nConfusion Matrix (threshold = 0.5):")
print(f"                 Predicted")
print(f"               0       1")
print(f"   Actual 0  {TN:4d}   {FP:4d}")
print(f"          1  {FN:4d}   {TP:4d}")

print(f"\n   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(data_logit['Y'], y_pred_prob)
roc_auc = auc(fpr, tpr)

print(f"\n   AUC-ROC:   {roc_auc:.4f}")
if roc_auc >= 0.80:
    print(f"   â†’ AUC â‰¥ 0.80 indicates strong predictive power")

# =====================================
# INFORMATION CRITERIA COMPARISON
# =====================================
print("\n" + "="*70)
print("INFORMATION CRITERIA (AIC, BIC)")
print("="*70)

models_ols = [
    ('Model 1: X1 + X2', model_strong),
    ('Model 2: X1 + X2 + X3', model_weak),
    ('Model 3: X1 + X2 + X3 + X4', model_noise)
]

print("\nOLS Models:")
for name, model in models_ols:
    print(f"\n{name}")
    print(f"   AIC: {model.aic:.2f}")
    print(f"   BIC: {model.bic:.2f}")
    print(f"   k (parameters): {model.df_model + 1}")  # +1 for intercept

best_aic = min([m.aic for _, m in models_ols])
best_bic = min([m.bic for _, m in models_ols])

print(f"\n   Best AIC: Model {[i+1 for i, (_, m) in enumerate(models_ols) if m.aic == best_aic][0]}")
print(f"   Best BIC: Model {[i+1 for i, (_, m) in enumerate(models_ols) if m.bic == best_bic][0]}")

# =====================================
# VISUALIZATION
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: RÂ² vs Adjusted RÂ² across models
model_names = ['X1+X2', 'X1+X2+X3', 'X1+X2+X3+X4']
r2_values = [model_strong.rsquared, model_weak.rsquared, model_noise.rsquared]
adj_r2_values = [model_strong.rsquared_adj, model_weak.rsquared_adj, model_noise.rsquared_adj]

x_pos = np.arange(len(model_names))
width = 0.35

axes[0, 0].bar(x_pos - width/2, r2_values, width, label='RÂ²', alpha=0.8)
axes[0, 0].bar(x_pos + width/2, adj_r2_values, width, label='Adjusted RÂ²', alpha=0.8)
axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('Value')
axes[0, 0].set_title('RÂ² vs Adjusted RÂ²: Adding Predictors')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 0.6)
axes[0, 0].axhline(model_strong.rsquared_adj, color='red', linestyle='--', alpha=0.5, 
                   label='Best Adj RÂ²')

# Plot 2: AIC/BIC comparison
aic_values = [model_strong.aic, model_weak.aic, model_noise.aic]
bic_values = [model_strong.bic, model_weak.bic, model_noise.bic]

axes[0, 1].plot(model_names, aic_values, 'o-', linewidth=2, markersize=10, label='AIC')
axes[0, 1].plot(model_names, bic_values, 's-', linewidth=2, markersize=10, label='BIC')
axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('Information Criterion (lower=better)')
axes[0, 1].set_title('AIC vs BIC: Model Comparison')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=15)

# Plot 3: ROC Curve
axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'Logit Model (AUC={roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve (Logistic Regression)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Pseudo-RÂ² Comparison
pseudo_r2_measures = ['McFadden', 'Cox-Snell', 'Nagelkerke', 'Efron']
pseudo_r2_vals = [
    pseudo_r2_full['McFadden'],
    pseudo_r2_full['Cox_Snell'],
    pseudo_r2_full['Nagelkerke'],
    pseudo_r2_full['Efron']
]

axes[1, 1].barh(pseudo_r2_measures, pseudo_r2_vals, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Pseudo-RÂ² Value')
axes[1, 1].set_title('Pseudo-RÂ² Measures (Logistic Regression)')
axes[1, 1].set_xlim(0, 0.5)
axes[1, 1].axvline(0.20, color='red', linestyle='--', alpha=0.5, label='McFadden "Excellent" (â‰¥0.20)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("Goodness-of-fit measures guide model selection:")
print(f"â€¢ OLS RÂ²: {model_strong.rsquared:.3f} â†’ {model_noise.rsquared:.3f} (always increases)")
print(f"â€¢ Adjusted RÂ²: {model_strong.rsquared_adj:.3f} â†’ {model_noise.rsquared_adj:.3f} (penalizes noise)")
print(f"â€¢ Best model (AIC/BIC): {model_names[np.argmin(aic_values)]} (parsimonious)")
print(f"â€¢ Logit McFadden RÂ²: {pseudo_r2_full['McFadden']:.3f} (excellent if â‰¥0.20)")
print(f"â€¢ AUC-ROC: {roc_auc:.3f} (strong discrimination if â‰¥0.80)")
print("\nKey insight: RÂ² increases with predictors, Adjusted RÂ² and IC penalize overfitting âœ“")
