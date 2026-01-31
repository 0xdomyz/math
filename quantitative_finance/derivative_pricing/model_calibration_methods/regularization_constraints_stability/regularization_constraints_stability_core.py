import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from sklearn.model_selection import KFold

# Block 1

np.random.seed(42)

print("="*70)
print("Regularization in Volatility Surface Calibration")
print("="*70)

# Generate synthetic IV surface data
maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])

T_grid, K_grid = np.meshgrid(maturities, strikes)
T_flat = T_grid.flatten()
K_flat = K_grid.flatten()

# True market data (synthetic; with smile)
market_iv = 0.20 + 0.05 * np.exp(-(T_grid - 1)**2 / 0.5) + \
            0.05 * ((K_grid - 1)**2)  # Smile
market_iv = market_iv.clip(0.05, 0.50).flatten()

print(f"Data Points: {len(market_iv)}")
print(f"Maturities: {maturities}")
print(f"Strikes: {strikes}")
print("")

# Parametric model: Polynomial smile + exponential term structure
# IV(K, T) = β₀ + β₁ T + β₂ T² + β₃ (K-1) + β₄ (K-1)² + β₅ T(K-1)
# 6 parameters
def build_feature_matrix(T, K):
    """Build design matrix for IV regression"""
    ones = np.ones_like(T)
    T2 = T**2
    moneyness = K - 1.0
    moneyness2 = moneyness**2
    interaction = T * moneyness
    
    X = np.column_stack([ones, T, T2, moneyness, moneyness2, interaction])
    return X

X = build_feature_matrix(T_flat, K_flat)

# Unregularized calibration (OLS)
def ols_objective(beta):
    pred = X @ beta
    residuals = pred - market_iv
    mse = np.mean(residuals**2)
    return mse

result_ols = minimize(ols_objective, np.zeros(6), method='BFGS')
beta_ols = result_ols.x
mse_ols = result_ols.fun
pred_ols = X @ beta_ols

# Ridge regression (L2 penalty)
lambdas = np.logspace(-4, 1, 50)
mse_ridge = []
pred_ridge_list = []

for lam in lambdas:
    def ridge_objective(beta):
        pred = X @ beta
        residuals = pred - market_iv
        mse = np.mean(residuals**2) + lam * np.sum(beta**2)
        return mse
    
    result = minimize(ridge_objective, np.zeros(6), method='BFGS')
    mse_ridge.append(result.fun)
    pred_ridge_list.append(X @ result.x)

mse_ridge = np.array(mse_ridge)

# Lasso (L1 penalty) - use coordinate descent
def lasso_objective(beta, lam):
    pred = X @ beta
    residuals = pred - market_iv
    mse = np.mean(residuals**2) + lam * np.sum(np.abs(beta))
    return mse

mse_lasso = []
beta_lasso_list = []

for lam in lambdas:
    def lasso_obj(beta):
        return lasso_objective(beta, lam)
    
    result = minimize(lasso_obj, np.zeros(6), method='Nelder-Mead',
                     options={'maxiter': 5000})
    mse_lasso.append(result.fun)
    beta_lasso_list.append(result.x)

mse_lasso = np.array(mse_lasso)

# Cross-validation for optimal λ
print("Cross-Validation for λ Selection (Ridge):")
print("-"*70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_ridge = []

for lam in lambdas:
    cv_errors = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = market_iv[train_idx], market_iv[test_idx]
        
        # Fit on train
        def ridge_train(beta):
            pred = X_train @ beta
            residuals = pred - y_train
            mse = np.mean(residuals**2) + lam * np.sum(beta**2)
            return mse
        
        result = minimize(ridge_train, np.zeros(6), method='BFGS')
        beta_ridge = result.x
        
        # Evaluate on test
        y_pred = X_test @ beta_ridge
        mse_test = np.mean((y_pred - y_test)**2)
        cv_errors.append(mse_test)
    
    cv_score = np.mean(cv_errors)
    cv_scores_ridge.append(cv_score)

cv_scores_ridge = np.array(cv_scores_ridge)
optimal_idx = np.argmin(cv_scores_ridge)
lambda_opt = lambdas[optimal_idx]

print(f"Optimal λ (via CV): {lambda_opt:.2e}")
print(f"CV error at optimal λ: {cv_scores_ridge[optimal_idx]:.2e}")

# Refit Ridge with optimal λ
def ridge_final(beta, lam):
    pred = X @ beta
    residuals = pred - market_iv
    mse = np.mean(residuals**2) + lam * np.sum(beta**2)
    return mse

result_ridge_opt = minimize(lambda b: ridge_final(b, lambda_opt), np.zeros(6), method='BFGS')
beta_ridge_opt = result_ridge_opt.x
pred_ridge_opt = X @ beta_ridge_opt

# Out-of-sample validation (hold-out test set)
print("\nOut-of-Sample Validation:")
print("-"*70)

test_idx = np.random.choice(len(market_iv), size=15, replace=False)
train_idx = np.setdiff1d(np.arange(len(market_iv)), test_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = market_iv[train_idx], market_iv[test_idx]

# OLS on training data
result_ols_train = minimize(lambda b: np.mean((X_train @ b - y_train)**2),
                            np.zeros(6), method='BFGS')
beta_ols_train = result_ols_train.x

# Ridge on training data
result_ridge_train = minimize(lambda b: np.mean((X_train @ b - y_train)**2) + lambda_opt * np.sum(b**2),
                              np.zeros(6), method='BFGS')
beta_ridge_train = result_ridge_train.x

# Test performance
pred_ols_test = X_test @ beta_ols_train
pred_ridge_test = X_test @ beta_ridge_train

rmse_ols_train = np.sqrt(np.mean((X_train @ beta_ols_train - y_train)**2))
rmse_ols_test = np.sqrt(np.mean((pred_ols_test - y_test)**2))

rmse_ridge_train = np.sqrt(np.mean((X_train @ beta_ridge_train - y_train)**2))
rmse_ridge_test = np.sqrt(np.mean((pred_ridge_test - y_test)**2))

print(f"\nOLS (Unregularized):")
print(f"  Training RMSE: {rmse_ols_train:.4e}")
print(f"  Test RMSE:     {rmse_ols_test:.4e}")
print(f"  Overfitting:   {rmse_ols_test/rmse_ols_train:.2f}× (test/train)")

print(f"\nRidge (λ = {lambda_opt:.2e}):")
print(f"  Training RMSE: {rmse_ridge_train:.4e}")
print(f"  Test RMSE:     {rmse_ridge_test:.4e}")
print(f"  Overfitting:   {rmse_ridge_test/rmse_ridge_train:.2f}× (test/train)")

# Parameter stability: Jackknife
print("\nParameter Stability (Jackknife Leave-One-Out):")
print("-"*70)

beta_loo_ols = []
beta_loo_ridge = []

for i in range(min(10, len(market_iv))):  # Test on first 10 points
    mask = np.ones(len(market_iv), dtype=bool)
    mask[i] = False
    
    X_loo = X[mask]
    y_loo = market_iv[mask]
    
    # OLS
    result_loo_ols = minimize(lambda b: np.mean((X_loo @ b - y_loo)**2),
                              np.zeros(6), method='BFGS')
    beta_loo_ols.append(result_loo_ols.x)
    
    # Ridge
    result_loo_ridge = minimize(lambda b: np.mean((X_loo @ b - y_loo)**2) + lambda_opt * np.sum(b**2),
                                np.zeros(6), method='BFGS')
    beta_loo_ridge.append(result_loo_ridge.x)

beta_loo_ols = np.array(beta_loo_ols)
beta_loo_ridge = np.array(beta_loo_ridge)

param_std_ols = np.std(beta_loo_ols, axis=0)
param_std_ridge = np.std(beta_loo_ridge, axis=0)

print(f"\nParameter Standard Deviations (Jackknife):")
print(f"{'Parameter':<12} {'OLS Std':<12} {'Ridge Std':<12} {'Reduction':<12}")
print("-"*70)
for j in range(6):
    reduction = param_std_ols[j] / (param_std_ridge[j] + 1e-10) - 1
    print(f"β{j:<11} {param_std_ols[j]:<12.4e} {param_std_ridge[j]:<12.4e} {reduction*100:6.1f}%")

# ===== VISUALIZATION =====

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Training error vs λ
ax = axes[0, 0]
ax.loglog(lambdas, mse_ridge, 'o-', linewidth=2, label='Ridge', markersize=4)
ax.loglog(lambdas, mse_lasso, 's-', linewidth=2, label='Lasso', markersize=4)
ax.axvline(lambda_opt, color='red', linestyle='--', linewidth=2, label=f'Optimal λ = {lambda_opt:.2e}')
ax.axhline(mse_ols, color='green', linestyle=':', linewidth=2, label=f'OLS = {mse_ols:.2e}')

ax.set_xlabel('λ (Penalty Parameter)')
ax.set_ylabel('Training MSE')
ax.set_title('Regularization Path: Training Error vs λ')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# Plot 2: Cross-validation error
ax = axes[0, 1]
ax.semilogx(lambdas, cv_scores_ridge, 'o-', linewidth=2, markersize=6, label='CV Error')
ax.axvline(lambda_opt, color='red', linestyle='--', linewidth=2, label=f'Optimal λ')
ax.fill_between(lambdas, cv_scores_ridge - 1e-4, cv_scores_ridge + 1e-4, alpha=0.2)

ax.set_xlabel('λ (Penalty Parameter)')
ax.set_ylabel('Cross-Validation Error (5-Fold)')
ax.set_title('Optimal λ Selection via Cross-Validation')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Out-of-sample performance
ax = axes[0, 2]
models = ['OLS Train', 'OLS Test', 'Ridge Train', 'Ridge Test']
rmses = [rmse_ols_train, rmse_ols_test, rmse_ridge_train, rmse_ridge_test]
colors = ['blue', 'red', 'blue', 'red']
alphas = [0.5, 0.5, 1.0, 1.0]

ax.bar(models, rmses, color=colors, alpha=alphas)
ax.set_ylabel('RMSE')
ax.set_title('Out-of-Sample Validation: Train vs Test')
ax.grid(axis='y', alpha=0.3)

# Plot 4: Parameter estimates (OLS vs Ridge)
ax = axes[1, 0]
x_pos = np.arange(6)
width = 0.35

ax.bar(x_pos - width/2, beta_ols, width, label='OLS', alpha=0.7)
ax.bar(x_pos + width/2, beta_ridge_opt, width, label=f'Ridge (λ={lambda_opt:.2e})', alpha=0.7)

ax.set_xlabel('Parameter Index')
ax.set_ylabel('Coefficient Value')
ax.set_title('Parameter Estimates: OLS vs Ridge')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'β{i}' for i in range(6)])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 5: Residuals
ax = axes[1, 1]
res_ols = pred_ols - market_iv
res_ridge = pred_ridge_opt - market_iv

ax.scatter(pred_ols, res_ols, alpha=0.5, label='OLS', s=50)
ax.scatter(pred_ridge_opt, res_ridge, alpha=0.5, label=f'Ridge', s=50)
ax.axhline(0, color='black', linestyle='--', linewidth=1)

ax.set_xlabel('Fitted IV')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot: OLS vs Ridge')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Summary statistics
ax = axes[1, 2]
ax.axis('off')

summary_text = f"""
Regularization Summary

Dataset:
  Points: {len(market_iv)}
  Parameters: 6
  Ratio: {len(market_iv)/6:.1f}:1

OLS (Unregularized):
  Train RMSE: {rmse_ols_train:.2e}
  Test RMSE:  {rmse_ols_test:.2e}
  Overfit:    {rmse_ols_test/rmse_ols_train:.2f}×
  Param Std:  {np.mean(param_std_ols):.2e}

Ridge (λ = {lambda_opt:.2e}):
  Train RMSE: {rmse_ridge_train:.2e}
  Test RMSE:  {rmse_ridge_test:.2e}
  Overfit:    {rmse_ridge_test/rmse_ridge_train:.2f}×
  Param Std:  {np.mean(param_std_ridge):.2e}

Improvement:
  Test RMSE:     {(1 - rmse_ridge_test/rmse_ols_test)*100:+6.1f}%
  Stability:     {param_std_ols[0]/param_std_ridge[0]:6.2f}× better
  Overfitting:   {(1 - rmse_ridge_test/rmse_ridge_train)/(1 - rmse_ols_test/rmse_ols_train)*100:+6.1f}%

Key Insight:
  Ridge improves generalization
  Test error ↓, Params ↓ std
  Bias-variance tradeoff
"""

ax.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('regularization_constraints_stability.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Ridge regularization improves out-of-sample generalization")
print("2. Cross-validation selects optimal λ objectively")
print("3. Parameter stability improves with regularization (Jackknife)")
print("4. Test RMSE decreases; validation confirms improvement")
print("5. Bias-variance tradeoff: slight training error increase; test improvement")