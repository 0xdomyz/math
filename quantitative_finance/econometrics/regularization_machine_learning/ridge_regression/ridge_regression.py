import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

np.random.seed(123)

# ===== Simulate Data with Multicollinearity =====
n = 200  # Sample size
p = 50   # Number of predictors (high-dimensional)

# Generate correlated predictors
rho = 0.7  # Correlation between adjacent predictors
Sigma = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        Sigma[i, j] = rho**abs(i - j)

# Generate X from multivariate normal
mean_X = np.zeros(p)
X = np.random.multivariate_normal(mean_X, Sigma, n)

# True sparse coefficient vector (only 5 non-zero)
beta_true = np.zeros(p)
beta_true[0:5] = [3, -2, 1.5, -1, 0.8]

# Generate Y with noise
epsilon = np.random.normal(0, 2, n)
Y = X @ beta_true + epsilon

# Create DataFrame
X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])
X_df['Y'] = Y

print("="*80)
print("RIDGE REGRESSION (L2 REGULARIZATION)")
print("="*80)
print(f"\nSimulation Setup:")
print(f"  Sample size (n): {n}")
print(f"  Number of predictors (p): {p}")
print(f"  True non-zero coefficients: 5 out of {p}")
print(f"  Correlation structure: Ï = {rho} (AR(1))")
print(f"\nTrue Î² (first 10):")
print(beta_true[:10])

# ===== Train-Test Split =====
train_size = int(0.7 * n)
train_idx = np.arange(train_size)
test_idx = np.arange(train_size, n)

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

print(f"\nTrain/Test Split:")
print(f"  Training: {len(X_train)} observations")
print(f"  Testing: {len(X_test)} observations")

# ===== Standardization =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Center Y
Y_train_mean = Y_train.mean()
Y_train_centered = Y_train - Y_train_mean
Y_test_centered = Y_test - Y_train_mean

print("\nâœ“ Data standardized (critical for ridge regression)")

# ===== OLS Baseline =====
print("\n" + "="*80)
print("OLS REGRESSION (Î» = 0)")
print("="*80)

# OLS solution (can be unstable with multicollinearity)
XtX = X_train_scaled.T @ X_train_scaled
XtY = X_train_scaled.T @ Y_train_centered

try:
    beta_ols = np.linalg.solve(XtX, XtY)
    
    # In-sample predictions
    Y_train_pred_ols = X_train_scaled @ beta_ols
    train_mse_ols = mean_squared_error(Y_train_centered, Y_train_pred_ols)
    train_r2_ols = r2_score(Y_train_centered, Y_train_pred_ols)
    
    # Out-of-sample predictions
    Y_test_pred_ols = X_test_scaled @ beta_ols
    test_mse_ols = mean_squared_error(Y_test_centered, Y_test_pred_ols)
    test_r2_ols = r2_score(Y_test_centered, Y_test_pred_ols)
    
    print(f"In-Sample Performance:")
    print(f"  MSE: {train_mse_ols:.4f}")
    print(f"  RÂ²: {train_r2_ols:.4f}")
    
    print(f"\nOut-of-Sample Performance:")
    print(f"  MSE: {test_mse_ols:.4f}")
    print(f"  RÂ²: {test_r2_ols:.4f}")
    
    print(f"\nCoefficient Statistics:")
    print(f"  L2 Norm: {np.linalg.norm(beta_ols):.4f}")
    print(f"  Max absolute value: {np.abs(beta_ols).max():.4f}")
    print(f"  Non-zero (|Î²| > 0.01): {np.sum(np.abs(beta_ols) > 0.01)}")
    
    ols_success = True
except:
    print("âš  OLS estimation failed (singular matrix)")
    ols_success = False
    beta_ols = np.zeros(p)
    test_mse_ols = np.inf
    test_r2_ols = -np.inf

# ===== Ridge Regression with Cross-Validation =====
print("\n" + "="*80)
print("RIDGE REGRESSION WITH CV")
print("="*80)

# Define lambda grid (log scale)
alphas = np.logspace(-3, 3, 100)  # sklearn uses alpha instead of lambda

# Use RidgeCV for automatic cross-validation
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, Y_train_centered)

optimal_lambda = ridge_cv.alpha_
print(f"Optimal Î» (from 5-fold CV): {optimal_lambda:.4f}")

# Fit ridge with optimal lambda
ridge_model = Ridge(alpha=optimal_lambda)
ridge_model.fit(X_train_scaled, Y_train_centered)
beta_ridge = ridge_model.coef_

# Performance
Y_train_pred_ridge = ridge_model.predict(X_train_scaled)
train_mse_ridge = mean_squared_error(Y_train_centered, Y_train_pred_ridge)
train_r2_ridge = r2_score(Y_train_centered, Y_train_pred_ridge)

Y_test_pred_ridge = ridge_model.predict(X_test_scaled)
test_mse_ridge = mean_squared_error(Y_test_centered, Y_test_pred_ridge)
test_r2_ridge = r2_score(Y_test_centered, Y_test_pred_ridge)

print(f"\nIn-Sample Performance:")
print(f"  MSE: {train_mse_ridge:.4f}")
print(f"  RÂ²: {train_r2_ridge:.4f}")

print(f"\nOut-of-Sample Performance:")
print(f"  MSE: {test_mse_ridge:.4f}")
print(f"  RÂ²: {test_r2_ridge:.4f}")

print(f"\nCoefficient Statistics:")
print(f"  L2 Norm: {np.linalg.norm(beta_ridge):.4f}")
print(f"  Max absolute value: {np.abs(beta_ridge).max():.4f}")
print(f"  Non-zero (|Î²| > 0.01): {np.sum(np.abs(beta_ridge) > 0.01)}")

# ===== Regularization Path =====
print("\n" + "="*80)
print("REGULARIZATION PATH")
print("="*80)

# Fit ridge for different lambda values
lambda_grid = np.logspace(-3, 3, 50)
coef_path = []
train_mse_path = []
test_mse_path = []

for lam in lambda_grid:
    ridge = Ridge(alpha=lam)
    ridge.fit(X_train_scaled, Y_train_centered)
    
    coef_path.append(ridge.coef_)
    
    train_pred = ridge.predict(X_train_scaled)
    test_pred = ridge.predict(X_test_scaled)
    
    train_mse_path.append(mean_squared_error(Y_train_centered, train_pred))
    test_mse_path.append(mean_squared_error(Y_test_centered, test_pred))

coef_path = np.array(coef_path)

# Find optimal lambda based on test MSE
optimal_idx = np.argmin(test_mse_path)
optimal_lambda_test = lambda_grid[optimal_idx]

print(f"Optimal Î» (based on test MSE): {optimal_lambda_test:.4f}")
print(f"Optimal Î» (based on CV): {optimal_lambda:.4f}")

# ===== Comparison Table =====
print("\n" + "="*80)
print("OLS vs RIDGE COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Method': ['OLS', 'Ridge'],
    'Train_MSE': [train_mse_ols if ols_success else np.nan, train_mse_ridge],
    'Test_MSE': [test_mse_ols if ols_success else np.nan, test_mse_ridge],
    'Test_R2': [test_r2_ols if ols_success else np.nan, test_r2_ridge],
    'L2_Norm': [np.linalg.norm(beta_ols) if ols_success else np.nan, 
                np.linalg.norm(beta_ridge)]
})

print(comparison.to_string(index=False))

if ols_success:
    improvement = (test_mse_ols - test_mse_ridge) / test_mse_ols * 100
    print(f"\nRidge improvement over OLS: {improvement:.1f}% reduction in test MSE")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Regularization Path
for i in range(min(10, p)):  # Plot first 10 coefficients
    axes[0, 0].plot(np.log10(lambda_grid), coef_path[:, i], 
                   linewidth=1.5, alpha=0.7)

axes[0, 0].axvline(np.log10(optimal_lambda), color='red', 
                  linestyle='--', linewidth=2, label=f'Optimal Î» (CV)')
axes[0, 0].set_xlabel('logâ‚â‚€(Î»)')
axes[0, 0].set_ylabel('Coefficient Value')
axes[0, 0].set_title('Regularization Path (First 10 Coefficients)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: MSE vs Lambda
axes[0, 1].plot(np.log10(lambda_grid), train_mse_path, 
               linewidth=2, label='Train MSE')
axes[0, 1].plot(np.log10(lambda_grid), test_mse_path,
               linewidth=2, label='Test MSE')
axes[0, 1].axvline(np.log10(optimal_lambda), color='red',
                  linestyle='--', linewidth=2, label='Optimal Î»')
axes[0, 1].set_xlabel('logâ‚â‚€(Î»)')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('Bias-Variance Tradeoff')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: OLS vs Ridge Coefficients (first 20)
if ols_success:
    x_pos = np.arange(20)
    width = 0.35
    
    axes[0, 2].bar(x_pos - width/2, beta_ols[:20], width,
                  label='OLS', alpha=0.8)
    axes[0, 2].bar(x_pos + width/2, beta_ridge[:20], width,
                  label='Ridge', alpha=0.8)
    axes[0, 2].bar(x_pos, beta_true[:20], width=0.1,
                  color='red', label='True', alpha=0.9)
    axes[0, 2].set_xlabel('Coefficient Index')
    axes[0, 2].set_ylabel('Coefficient Value')
    axes[0, 2].set_title('Coefficients: OLS vs Ridge (First 20)')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(alpha=0.3, axis='y')

# Plot 4: Coefficient Shrinkage Visualization
axes[1, 0].scatter(beta_ols[:20] if ols_success else beta_true[:20], 
                  beta_ridge[:20], alpha=0.7, s=50)
axes[1, 0].plot([beta_ridge[:20].min(), beta_ridge[:20].max()],
               [beta_ridge[:20].min(), beta_ridge[:20].max()],
               'r--', linewidth=2, label='45Â° line')
axes[1, 0].set_xlabel('OLS Coefficient' if ols_success else 'True Coefficient')
axes[1, 0].set_ylabel('Ridge Coefficient')
axes[1, 0].set_title('Shrinkage Effect')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: L-Curve (Residual vs Penalty)
residuals = [np.linalg.norm(Y_train_centered - X_train_scaled @ coef)**2 
             for coef in coef_path]
penalties = [np.linalg.norm(coef)**2 for coef in coef_path]

axes[1, 1].plot(penalties, residuals, 'o-', linewidth=2, markersize=4)
axes[1, 1].scatter(penalties[optimal_idx], residuals[optimal_idx],
                  color='red', s=200, marker='*', zorder=5,
                  label='Optimal')
axes[1, 1].set_xlabel('||Î²||Â² (Penalty)')
axes[1, 1].set_ylabel('||Y - XÎ²||Â² (Residual)')
axes[1, 1].set_title('L-Curve')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Prediction Scatter
axes[1, 2].scatter(Y_test_centered, Y_test_pred_ridge, alpha=0.6, s=50)
axes[1, 2].plot([Y_test_centered.min(), Y_test_centered.max()],
               [Y_test_centered.min(), Y_test_centered.max()],
               'r--', linewidth=2)
axes[1, 2].set_xlabel('True Y')
axes[1, 2].set_ylabel('Predicted Y (Ridge)')
axes[1, 2].set_title(f'Out-of-Sample Predictions (RÂ² = {test_r2_ridge:.3f})')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ridge_regression_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Effective Degrees of Freedom =====
print("\n" + "="*80)
print("EFFECTIVE DEGREES OF FREEDOM")
print("="*80)

# Compute eigenvalues of X'X
eigenvalues = np.linalg.eigvalsh(X_train_scaled.T @ X_train_scaled)

# Effective df for different lambdas
df_values = []
for lam in lambda_grid:
    df = np.sum(eigenvalues / (eigenvalues + lam))
    df_values.append(df)

print(f"OLS degrees of freedom: {p}")
print(f"Ridge df (Î»={optimal_lambda:.4f}): {df_values[optimal_idx]:.2f}")
print(f"Ridge df (Î»â†’âˆž): 0")

# ===== Monte Carlo: MSE Decomposition =====
print("\n" + "="*80)
print("MONTE CARLO: BIAS-VARIANCE DECOMPOSITION")
print("="*80)

n_sim = 100
lambdas_to_test = [0.01, 0.1, 1.0, 10.0, 100.0]

mse_decomp = []

for lam in lambdas_to_test:
    predictions = []
    
    for sim in range(n_sim):
        # Generate new noise
        Y_sim = X_train @ beta_true[:p] + np.random.normal(0, 2, len(X_train))
        Y_sim_centered = Y_sim - Y_sim.mean()
        
        # Fit ridge
        ridge_sim = Ridge(alpha=lam)
        ridge_sim.fit(X_train_scaled, Y_sim_centered)
        
        # Predict on test set
        pred_sim = ridge_sim.predict(X_test_scaled)
        predictions.append(pred_sim)
    
    predictions = np.array(predictions)
    
    # True test values (without noise)
    Y_test_true = X_test @ beta_true[:p] - Y_train_mean
    
    # Expected prediction (over simulations)
    expected_pred = predictions.mean(axis=0)
    
    # BiasÂ²
    bias_sq = np.mean((expected_pred - Y_test_true)**2)
    
    # Variance
    variance = np.mean(predictions.var(axis=0))
    
    # Irreducible error (noise variance)
    noise_var = 4  # ÏƒÂ² = 4
    
    # Total MSE
    total_mse = bias_sq + variance + noise_var
    
    mse_decomp.append({
        'Lambda': lam,
        'BiasÂ²': bias_sq,
        'Variance': variance,
        'Noise': noise_var,
        'Total_MSE': total_mse
    })

mse_df = pd.DataFrame(mse_decomp)
print("\nBias-Variance Decomposition:")
print(mse_df.to_string(index=False))

# Visualize decomposition
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

x_pos = np.arange(len(lambdas_to_test))
width = 0.6

ax.bar(x_pos, mse_df['BiasÂ²'], width, label='BiasÂ²', alpha=0.8)
ax.bar(x_pos, mse_df['Variance'], width, bottom=mse_df['BiasÂ²'],
       label='Variance', alpha=0.8)
ax.bar(x_pos, mse_df['Noise'], width,
       bottom=mse_df['BiasÂ²'] + mse_df['Variance'],
       label='Irreducible Error', alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels([f'Î»={l}' for l in lambdas_to_test])
ax.set_ylabel('MSE Components')
ax.set_title('Bias-Variance Tradeoff in Ridge Regression')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ridge_bias_variance.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n1. Performance Comparison:")
if ols_success:
    print(f"   OLS Test MSE: {test_mse_ols:.4f}, RÂ²: {test_r2_ols:.4f}")
print(f"   Ridge Test MSE: {test_mse_ridge:.4f}, RÂ²: {test_r2_ridge:.4f}")
if ols_success and test_mse_ridge < test_mse_ols:
    print(f"   âœ“ Ridge improves prediction by {improvement:.1f}%")

print("\n2. Regularization Effect:")
print(f"   Optimal Î»: {optimal_lambda:.4f}")
print(f"   Coefficient shrinkage: L2 norm reduced from {np.linalg.norm(beta_ols) if ols_success else 'N/A':.2f} to {np.linalg.norm(beta_ridge):.2f}")
print(f"   Effective degrees of freedom: {df_values[optimal_idx]:.1f} (down from {p})")

print("\n3. Bias-Variance Tradeoff:")
optimal_decomp = mse_df[mse_df['Lambda'] == min(lambdas_to_test, key=lambda x: abs(x - optimal_lambda))]
if not optimal_decomp.empty:
    print(f"   BiasÂ²: {optimal_decomp['BiasÂ²'].values[0]:.2f}")
    print(f"   Variance: {optimal_decomp['Variance'].values[0]:.2f}")
    print(f"   Total MSE: {optimal_decomp['Total_MSE'].values[0]:.2f}")

print("\n4. When to Use Ridge:")
print("   â€¢ High multicollinearity (VIF > 10)")
print("   â€¢ p comparable to n (high-dimensional)")
print("   â€¢ Prediction focus (not interpretation)")
print("   â€¢ Believe most predictors contribute")

print("\n5. Practical Tips:")
print("   â€¢ Always standardize predictors first")
print("   â€¢ Use cross-validation to select Î»")
print("   â€¢ Report out-of-sample performance")
print("   â€¢ Consider Lasso if want variable selection")
print("   â€¢ Bootstrap for confidence intervals")
