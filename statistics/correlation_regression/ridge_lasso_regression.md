# Ridge & Lasso Regression

## 1.1 Concept Skeleton
**Definition:** Regularized regression methods adding penalty term to loss function to prevent overfitting  
**Purpose:** Handle multicollinearity, reduce model complexity, perform feature selection  
**Prerequisites:** Linear regression, overfitting, bias-variance tradeoff, cross-validation

## 1.2 Comparative Framing
| Method | Ordinary Least Squares | Ridge (L2) | Lasso (L1) |
|--------|----------------------|------------|------------|
| **Penalty** | None | Sum of squared coefficients | Sum of absolute coefficients |
| **Formula** | Minimize: Σ(y - ŷ)² | Minimize: Σ(y - ŷ)² + λΣβ² | Minimize: Σ(y - ŷ)² + λΣ\|β\| |
| **Coefficient Behavior** | Unrestricted | Shrinks toward zero | Shrinks to exactly zero |
| **Feature Selection** | No | No (keeps all) | Yes (automatic) |

## 1.3 Examples + Counterexamples

**Simple Example:**  
100 predictors, n=50 samples: OLS overfit, Ridge shrinks coefficients, Lasso selects 10 key features

**Failure Case:**  
Grouped correlated predictors: Lasso arbitrarily picks one, drops others; Elastic Net (Ridge+Lasso) handles better

**Edge Case:**  
λ=0: Identical to OLS. λ→∞: All coefficients → 0 (intercept-only model)

## 1.4 Layer Breakdown
```
Regularization Framework:
├─ Ridge Regression (L2):
│   ├─ Objective: min[Σ(yᵢ - β₀ - Σβⱼxᵢⱼ)² + λΣβⱼ²]
│   ├─ Penalty: λ controls shrinkage strength
│   ├─ Effect: Shrinks coefficients proportionally
│   ├─ Geometry: Circular constraint region
│   └─ Use: Multicollinearity, many small predictors
├─ Lasso Regression (L1):
│   ├─ Objective: min[Σ(yᵢ - β₀ - Σβⱼxᵢⱼ)² + λΣ|βⱼ|]
│   ├─ Penalty: λ controls sparsity
│   ├─ Effect: Sets many coefficients to exactly zero
│   ├─ Geometry: Diamond constraint region (corners hit axes)
│   └─ Use: Feature selection, sparse models
├─ Elastic Net:
│   ├─ Combines: α·L1 + (1-α)·L2
│   ├─ Balance: α=1 (pure Lasso), α=0 (pure Ridge)
│   └─ Use: Correlated predictors, best of both
├─ Hyperparameter Tuning:
│   ├─ λ (lambda): Regularization strength
│   ├─ Selection: Cross-validation (minimize test error)
│   └─ Path: Compute solutions for range of λ values
└─ Standardization:
    ├─ Required: Scale predictors before regularization
    ├─ Reason: Penalty scale-dependent
    └─ Method: z-score (mean=0, sd=1)
```

## 1.5 Mini-Project
Compare OLS, Ridge, and Lasso:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Generate correlated predictors
n = 100
p = 50
X = np.random.randn(n, p)
# Make some predictors correlated
X[:, 1] = X[:, 0] + np.random.randn(n) * 0.1
X[:, 2] = X[:, 0] + np.random.randn(n) * 0.1

# True model: only first 5 predictors matter
true_coef = np.zeros(p)
true_coef[:5] = [3, -2, 1.5, -1, 0.8]
y = X @ true_coef + np.random.randn(n) * 2

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit models with different lambdas
lambdas = np.logspace(-3, 3, 50)
ridge_coefs = []
lasso_coefs = []

for lam in lambdas:
    ridge = Ridge(alpha=lam, fit_intercept=False)
    ridge.fit(X_scaled, y)
    ridge_coefs.append(ridge.coef_)
    
    lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
    lasso.fit(X_scaled, y)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# Cross-validation to find best lambda
cv_ridge = []
cv_lasso = []

for lam in lambdas:
    ridge = Ridge(alpha=lam)
    lasso = Lasso(alpha=lam, max_iter=10000)
    
    cv_ridge.append(-cross_val_score(ridge, X_scaled, y, cv=5, 
                                     scoring='neg_mean_squared_error').mean())
    cv_lasso.append(-cross_val_score(lasso, X_scaled, y, cv=5,
                                     scoring='neg_mean_squared_error').mean())

best_lambda_ridge = lambdas[np.argmin(cv_ridge)]
best_lambda_lasso = lambdas[np.argmin(cv_lasso)]

print(f"Best λ for Ridge: {best_lambda_ridge:.4f}")
print(f"Best λ for Lasso: {best_lambda_lasso:.4f}")

# Fit final models
ols = LinearRegression(fit_intercept=False)
ridge_best = Ridge(alpha=best_lambda_ridge, fit_intercept=False)
lasso_best = Lasso(alpha=best_lambda_lasso, fit_intercept=False)

ols.fit(X_scaled, y)
ridge_best.fit(X_scaled, y)
lasso_best.fit(X_scaled, y)

print(f"\nNumber of non-zero coefficients:")
print(f"  OLS: {np.sum(np.abs(ols.coef_) > 1e-5)}")
print(f"  Ridge: {np.sum(np.abs(ridge_best.coef_) > 1e-5)}")
print(f"  Lasso: {np.sum(np.abs(lasso_best.coef_) > 1e-5)}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Ridge coefficient paths
axes[0, 0].semilogx(lambdas, ridge_coefs[:, :10])
axes[0, 0].axvline(best_lambda_ridge, color='r', linestyle='--', label='Best λ')
axes[0, 0].set_xlabel('λ (log scale)')
axes[0, 0].set_ylabel('Coefficient Value')
axes[0, 0].set_title('Ridge: Coefficient Paths (first 10)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Lasso coefficient paths
axes[0, 1].semilogx(lambdas, lasso_coefs[:, :10])
axes[0, 1].axvline(best_lambda_lasso, color='r', linestyle='--', label='Best λ')
axes[0, 1].set_xlabel('λ (log scale)')
axes[0, 1].set_ylabel('Coefficient Value')
axes[0, 1].set_title('Lasso: Coefficient Paths (first 10)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Cross-validation curves
axes[1, 0].semilogx(lambdas, cv_ridge, 'b-', label='Ridge')
axes[1, 0].semilogx(lambdas, cv_lasso, 'r-', label='Lasso')
axes[1, 0].axvline(best_lambda_ridge, color='b', linestyle='--', alpha=0.5)
axes[1, 0].axvline(best_lambda_lasso, color='r', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('λ (log scale)')
axes[1, 0].set_ylabel('Cross-Validated MSE')
axes[1, 0].set_title('Model Selection via Cross-Validation')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Coefficient comparison
indices = np.arange(15)
width = 0.25
axes[1, 1].bar(indices - width, true_coef[:15], width, label='True', alpha=0.7)
axes[1, 1].bar(indices, ridge_best.coef_[:15], width, label='Ridge', alpha=0.7)
axes[1, 1].bar(indices + width, lasso_best.coef_[:15], width, label='Lasso', alpha=0.7)
axes[1, 1].set_xlabel('Predictor Index')
axes[1, 1].set_ylabel('Coefficient Value')
axes[1, 1].set_title('Coefficient Estimates (first 15)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print key coefficients
print("\nFirst 10 coefficients:")
print(f"True:  {true_coef[:10]}")
print(f"OLS:   {ols.coef_[:10]}")
print(f"Ridge: {ridge_best.coef_[:10]}")
print(f"Lasso: {lasso_best.coef_[:10]}")
```

## 1.6 Challenge Round
When is regularization the wrong choice?
- **n >> p and no multicollinearity**: OLS is optimal (unbiased, minimum variance)
- **All predictors truly important**: Regularization introduces bias unnecessarily
- **Strong domain knowledge**: Better to specify model structure than penalize blindly
- **Interpretability critical**: Ridge doesn't select features; Lasso selection can be unstable
- **Non-linear relationships**: Use splines, trees, or neural networks instead

## 1.7 Key References
- [Wikipedia - Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression)
- [Wikipedia - Lasso Regression](https://en.wikipedia.org/wiki/Lasso_(statistics))
- [Hastie et al. - Statistical Learning (Chapter 3)](https://www.statlearning.com/)
- Thinking: Bias-variance tradeoff: Add bias (regularization) to reduce variance (overfitting); L1 produces sparsity due to diamond constraint geometry; Always standardize before regularizing

---
**Status:** Essential for high-dimensional data | **Complements:** Cross-Validation, Feature Selection, Overfitting Prevention
