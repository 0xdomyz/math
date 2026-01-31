"""
Machine Learning Credit Risk Models
Extracted from machine_learning_models.md

Modern ML approaches for credit risk prediction.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

np.random.seed(42)

print("=== Machine Learning Credit Risk Models ===")

n_samples = 10000
n_features = 30

# Synthetic features
X = pd.DataFrame(np.random.randn(n_samples, n_features))
X.columns = [f"Feature_{i}" for i in range(n_features)]

X["debt_to_income"] = np.random.uniform(0.1, 1.0, n_samples)
X["credit_score"] = np.random.normal(700, 100, n_samples)
X["employment_years"] = np.random.exponential(8, n_samples)
X["age"] = np.random.normal(45, 15, n_samples)
X["loan_amount"] = np.random.lognormal(10.5, 1, n_samples)

# Generate target
logit_score = (
    -3
    + 1.2 * (X["debt_to_income"] - 0.5)
    + -0.003 * (X["credit_score"] - 700)
    + -0.1 * np.log1p(X["employment_years"])
    + 0.15 * (X["loan_amount"] / 100000)
    + 0.5 * np.random.randn(n_samples)
)

prob_default = 1 / (1 + np.exp(-logit_score))
y = (np.random.rand(n_samples) < prob_default).astype(int)

print(f"Dataset: {n_samples} samples, {n_features + 5} features")
print(f"Default rate: {y.mean():.2%}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
print("\n=== Model 1: Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
auc_lr = roc_auc_score(y_test, y_pred_lr)
print(f"AUC-ROC: {auc_lr:.3f}")

# Model 2: Random Forest
print("\n=== Model 2: Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"AUC-ROC: {auc_rf:.3f}")

feature_importance_rf = pd.DataFrame(
    {"Feature": X.columns, "Importance": rf.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_rf.head(10).to_string(index=False))

# Model 3: Gradient Boosting
print("\n=== Model 3: Gradient Boosting (XGBoost-style) ===")
gb = GradientBoostingClassifier(
    n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42
)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict_proba(X_test)[:, 1]
auc_gb = roc_auc_score(y_test, y_pred_gb)
print(f"AUC-ROC: {auc_gb:.3f}")

feature_importance_gb = pd.DataFrame(
    {"Feature": X.columns, "Importance": gb.feature_importances_}
).sort_values("Importance", ascending=False)

# Model 4: Ensemble
print("\n=== Model 4: Ensemble (Weighted Average) ===")
weights = np.array([auc_lr, auc_rf, auc_gb])
weights = weights / weights.sum()

y_pred_ensemble = (
    weights[0] * y_pred_lr + weights[1] * y_pred_rf + weights[2] * y_pred_gb
)
auc_ensemble = roc_auc_score(y_test, y_pred_ensemble)

print(
    f"LR weight: {weights[0]:.2f}, RF weight: {weights[1]:.2f}, GB weight: {weights[2]:.2f}"
)
print(f"Ensemble AUC-ROC: {auc_ensemble:.3f}")

# Cross-validation
print("\n=== Cross-Validation Results (5-fold) ===")
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {"Logistic Regression": lr, "Random Forest": rf, "Gradient Boosting": gb}

for model_name, model in models.items():
    if model_name == "Logistic Regression":
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=cv_splitter, scoring="roc_auc", n_jobs=-1
        )
    else:
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv_splitter, scoring="roc_auc", n_jobs=-1
        )
    print(f"{model_name:20s}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Calibration analysis
print("\n=== Calibration Analysis ===")
print("Model | Mean Pred PD | Actual Default % | Difference")
print("-" * 55)

for model_name, y_pred in [
    ("Logistic Regression", y_pred_lr),
    ("Random Forest", y_pred_rf),
    ("Gradient Boosting", y_pred_gb),
    ("Ensemble", y_pred_ensemble),
]:
    mean_pred = y_pred.mean()
    actual_rate = y_test.mean()
    diff = (actual_rate - mean_pred) * 100
    print(f"{model_name:20s} | {mean_pred:12.2%} | {actual_rate:15.2%} | {diff:9.2f}pp")

# Stability analysis
print("\n=== Model Stability (Temporal Shift Simulation) ===")

n_drift = 2000
X_drift = X_test.iloc[:n_drift].copy()
for col in X_drift.columns:
    X_drift[col] = X_drift[col] + np.random.normal(0.1, 0.1)

y_pred_original = gb.predict_proba(X_test.iloc[:n_drift])[:, 1]
y_pred_drifted = gb.predict_proba(X_drift)[:, 1]

print(f"Original mean PD: {y_pred_original.mean():.2%}")
print(f"Drifted mean PD: {y_pred_drifted.mean():.2%}")
print(
    f"Drift magnitude: {abs(y_pred_drifted.mean() - y_pred_original.mean())*100:.2f}pp"
)

print("\n=== ML Models Summary ===")
print(
    f"Winner: {'Gradient Boosting' if auc_gb == max([auc_lr, auc_rf, auc_gb]) else 'Ensemble'}"
)
print(f"Best AUC: {max([auc_lr, auc_rf, auc_gb, auc_ensemble]):.3f}")
print(
    f"Performance gain vs baseline: {(max([auc_rf, auc_gb, auc_ensemble])/auc_lr - 1)*100:.1f}%"
)

if __name__ == "__main__":
    print("\nMachine learning models execution complete.")
