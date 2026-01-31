# Auto-extracted from markdown file
# Source: machine_learning_models.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Generate realistic synthetic credit data
print("=== Machine Learning Credit Risk Models ===")

n_samples = 10000
n_features = 30

# Synthetic features
X = pd.DataFrame(np.random.randn(n_samples, n_features))
X.columns = [f'Feature_{i}' for i in range(n_features)]

# Add some correlations (realistic relationships)
X['debt_to_income'] = np.random.uniform(0.1, 1.0, n_samples)
X['credit_score'] = np.random.normal(700, 100, n_samples)
X['employment_years'] = np.random.exponential(8, n_samples)
X['age'] = np.random.normal(45, 15, n_samples)
X['loan_amount'] = np.random.lognormal(10.5, 1, n_samples)

# Generate target: non-linear relationship
logit_score = (-3 +
              1.2 * (X['debt_to_income'] - 0.5) +
              -0.003 * (X['credit_score'] - 700) +
              -0.1 * np.log1p(X['employment_years']) +
              0.15 * (X['loan_amount'] / 100000) +
              0.5 * np.random.randn(n_samples))  # Non-linearity

prob_default = 1 / (1 + np.exp(-logit_score))
y = (np.random.rand(n_samples) < prob_default).astype(int)

print(f"Dataset: {n_samples} samples, {n_features + 5} features")
print(f"Default rate: {y.mean():.2%}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                      stratify=y, random_state=42)

# Standardize features (required for logistic regression, beneficial for trees)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression (baseline)
print("\n=== Model 1: Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
auc_lr = roc_auc_score(y_test, y_pred_lr)
print(f"AUC-ROC: {auc_lr:.3f}")

# Model 2: Random Forest
print("\n=== Model 2: Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, 
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"AUC-ROC: {auc_rf:.3f}")

# Feature importance (top 10)
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_rf.head(10).to_string(index=False))

# Model 3: Gradient Boosting
print("\n=== Model 3: Gradient Boosting (XGBoost-style) ===")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=7, 
                               learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict_proba(X_test)[:, 1]
auc_gb = roc_auc_score(y_test, y_pred_gb)
print(f"AUC-ROC: {auc_gb:.3f}")

# Feature importance
feature_importance_gb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False)

# Model 4: Ensemble (weighted average)
print("\n=== Model 4: Ensemble (Weighted Average) ===")
# Weights based on individual AUC
weights = np.array([auc_lr, auc_rf, auc_gb])
weights = weights / weights.sum()

y_pred_ensemble = (weights[0] * y_pred_lr + 
                   weights[1] * y_pred_rf + 
                   weights[2] * y_pred_gb)
auc_ensemble = roc_auc_score(y_test, y_pred_ensemble)

print(f"LR weight: {weights[0]:.2f}, RF weight: {weights[1]:.2f}, GB weight: {weights[2]:.2f}")
print(f"Ensemble AUC-ROC: {auc_ensemble:.3f}")

# Cross-validation
print("\n=== Cross-Validation Results (5-fold) ===")
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'Gradient Boosting': gb
}

for model_name, model in models.items():
    if model_name == 'Logistic Regression':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_splitter, 
                                   scoring='roc_auc', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, 
                                   scoring='roc_auc', n_jobs=-1)
    print(f"{model_name:20s}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Calibration analysis
print("\n=== Calibration Analysis ===")
print("Model | Mean Pred PD | Actual Default % | Difference")
print("-" * 55)

for model_name, y_pred in [('Logistic Regression', y_pred_lr),
                           ('Random Forest', y_pred_rf),
                           ('Gradient Boosting', y_pred_gb),
                           ('Ensemble', y_pred_ensemble)]:
    mean_pred = y_pred.mean()
    actual_rate = y_test.mean()
    diff = (actual_rate - mean_pred) * 100
    print(f"{model_name:20s} | {mean_pred:12.2%} | {actual_rate:15.2%} | {diff:9.2f}pp")

# Stability analysis: model on different time periods
print("\n=== Model Stability (Temporal Shift Simulation) ===")

# Simulate data drift: shift feature distributions
n_drift = 2000
X_drift = X_test.iloc[:n_drift].copy()
# Increase feature means slightly (economic regime change)
for col in X_drift.columns:
    X_drift[col] = X_drift[col] + np.random.normal(0.1, 0.1)

# Recalibrate model on original distribution vs new
y_pred_original = gb.predict_proba(X_test.iloc[:n_drift])[:, 1]
y_pred_drifted = gb.predict_proba(X_drift)[:, 1]

print(f"Original mean PD: {y_pred_original.mean():.2%}")
print(f"Drifted mean PD: {y_pred_drifted.mean():.2%}")
print(f"Drift magnitude: {abs(y_pred_drifted.mean() - y_pred_original.mean())*100:.2f}pp")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: ROC curves comparison
ax1 = axes[0, 0]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_gb)
fpr_ens, tpr_ens, _ = roc_curve(y_test, y_pred_ensemble)

ax1.plot(fpr_lr, tpr_lr, linewidth=2, label=f'LR (AUC={auc_lr:.3f})')
ax1.plot(fpr_rf, tpr_rf, linewidth=2, label=f'RF (AUC={auc_rf:.3f})')
ax1.plot(fpr_gb, tpr_gb, linewidth=2, label=f'GB (AUC={auc_gb:.3f})')
ax1.plot(fpr_ens, tpr_ens, linewidth=3, label=f'Ensemble (AUC={auc_ensemble:.3f})')
ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Model Comparison: ROC Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Feature importance (Random Forest)
ax2 = axes[0, 1]
top_features_rf = feature_importance_rf.head(15)
ax2.barh(range(len(top_features_rf)), top_features_rf['Importance'].values, edgecolor='black', alpha=0.7)
ax2.set_yticks(range(len(top_features_rf)))
ax2.set_yticklabels(top_features_rf['Feature'].values, fontsize=9)
ax2.set_xlabel('Importance')
ax2.set_title('Top 15 Features (Random Forest)')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Feature importance (Gradient Boosting)
ax3 = axes[0, 2]
top_features_gb = feature_importance_gb.head(15)
ax3.barh(range(len(top_features_gb)), top_features_gb['Importance'].values, 
        color='orange', edgecolor='black', alpha=0.7)
ax3.set_yticks(range(len(top_features_gb)))
ax3.set_yticklabels(top_features_gb['Feature'].values, fontsize=9)
ax3.set_xlabel('Importance')
ax3.set_title('Top 15 Features (Gradient Boosting)')
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Calibration curves
ax4 = axes[1, 0]
for model_name, y_pred in [('Logistic Regression', y_pred_lr),
                           ('Random Forest', y_pred_rf),
                           ('Gradient Boosting', y_pred_gb),
                           ('Ensemble', y_pred_ensemble)]:
    pred_bins = np.linspace(0, 1, 10)
    bin_means = []
    actual_rates = []
    for i in range(len(pred_bins)-1):
        mask = (y_pred >= pred_bins[i]) & (y_pred < pred_bins[i+1])
        if mask.sum() > 0:
            bin_means.append((pred_bins[i] + pred_bins[i+1]) / 2)
            actual_rates.append(y_test[mask].mean())
    ax4.plot(bin_means, actual_rates, 'o-', linewidth=2, label=model_name, markersize=5)

ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
ax4.set_xlabel('Predicted PD')
ax4.set_ylabel('Actual Default Rate')
ax4.set_title('Model Calibration Curves')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: PD distribution
ax5 = axes[1, 1]
ax5.hist(y_pred_lr, bins=30, alpha=0.5, label='LR', edgecolor='black')
ax5.hist(y_pred_rf, bins=30, alpha=0.5, label='RF', edgecolor='black')
ax5.hist(y_pred_gb, bins=30, alpha=0.5, label='GB', edgecolor='black')
ax5.set_xlabel('Predicted PD')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of Model Predictions')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Model comparison
ax6 = axes[1, 2]
model_names = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting', 'Ensemble']
auc_scores = [auc_lr, auc_rf, auc_gb, auc_ensemble]
colors = ['steelblue', 'orange', 'green', 'purple']
bars = ax6.bar(model_names, auc_scores, color=colors, alpha=0.7, edgecolor='black')
ax6.set_ylabel('AUC-ROC')
ax6.set_ylim([0.5, 1.0])
ax6.set_title('Model Performance Comparison')
for bar, score in zip(bars, auc_scores):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.3f}', ha='center', va='bottom')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n=== ML Models Summary ===")
print(f"Winner: {'Gradient Boosting' if auc_gb == max([auc_lr, auc_rf, auc_gb]) else 'Ensemble'}")
print(f"Best AUC: {max([auc_lr, auc_rf, auc_gb, auc_ensemble]):.3f}")
print(f"Performance gain vs baseline: {(max([auc_rf, auc_gb, auc_ensemble])/auc_lr - 1)*100:.1f}%")

