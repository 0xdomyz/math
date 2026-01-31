# Lasso Regression (L1 Regularization)

## 1. Concept Skeleton
**Definition:** Linear regression with L1 penalty on absolute coefficients; produces sparse solutions; performs automatic variable selection; sets some coefficients exactly to zero  
**Purpose:** Variable selection; interpretable models; handle high-dimensional data; feature screening; prevent overfitting  
**Prerequisites:** OLS regression, ridge regression, convex optimization, coordinate descent, cross-validation, sparsity

## 2. Comparative Framing
| Method | OLS | Ridge (L2) | Lasso (L1) | Elastic Net | Subset Selection |
|--------|-----|------------|------------|-------------|------------------|
| **Penalty** | None | λΣβⱼ² | λΣ\|βⱼ\| | λ₁Σβⱼ² + λ₂Σ\|βⱼ\| | Σ1(βⱼ≠0) |
| **Solution** | Closed form | Closed form | Iterative | Iterative | Combinatorial |
| **Shrinkage** | No | Continuous | Hard threshold | Both | Discrete |
| **Variable Selection** | No | No | Yes | Yes | Yes |
| **Sparsity** | Dense | Dense | Sparse | Sparse | Sparse |
| **Stability** | Low | High | Moderate | High | Low |
| **Grouped Variables** | Treats individually | Shrinks together | Picks one | Group penalty | Arbitrary |

## 3. Examples + Counterexamples

**Classic Example:**  
Cancer prediction with 5000 genes, n=100 patients. Lasso selects 12 genes with test AUC=0.85. OLS cannot fit (p>n). Ridge includes all 5000 but interpretation impossible.

**Failure Case:**  
Correlated predictors (GDP, GNP, GNI with r>0.95). Lasso arbitrarily picks one, discards others. Ridge/elastic net better—all contribute. Coefficient path unstable under perturbations.

**Edge Case:**  
n=p (exactly determined). Lasso may not select all variables even when all truly relevant. Solution depends on λ—need careful tuning. OLS perfect fit but overfits.

## 4. Layer Breakdown
```
Lasso Regression Framework:
├─ Objective Function:
│   ├─ Lasso: min Σ(yᵢ - Xᵢ'β)² + λΣ|βⱼ|
│   │   ├─ First term: Residual sum of squares (fit)
│   │   ├─ Second term: L1 penalty (sparsity)
│   │   └─ λ ≥ 0: Tuning parameter (controls sparsity)
│   ├─ Matrix Form: min ||Y - Xβ||² + λ||β||₁
│   ├─ Constrained Form: min ||Y - Xβ||² subject to Σ|βⱼ| ≤ t
│   │   └─ Lagrange multiplier λ corresponds to constraint t
│   ├─ Bayesian Interpretation:
│   │   └─ Lasso = MAP estimate with Laplace prior β ~ Laplace(0, b)
│   └─ Key Difference from Ridge:
│       ├─ L1 penalty has corners at axes → Drives coefficients to zero
│       └─ L2 penalty smooth everywhere → Continuous shrinkage
├─ Properties of L1 Penalty:
│   ├─ Non-Differentiable: |β| not differentiable at β=0
│   │   └─ Requires subgradient methods
│   ├─ Sparsity-Inducing: Sets coefficients exactly to zero
│   │   └─ Automatic variable selection
│   ├─ Convex: Global optimum exists
│   ├─ Rotation Invariant: No (unlike L2)
│   │   └─ Solution depends on coordinate system
│   └─ Geometric Intuition:
│       ├─ Constraint region: Diamond shape in 2D
│       ├─ Contours of RSS: Ellipses
│       └─ Intersection at corner → Zero coefficients
├─ Optimization Methods:
│   ├─ Coordinate Descent (Standard):
│   │   ├─ Update one coefficient at a time holding others fixed
│   │   ├─ Soft-Thresholding: β̂ⱼ = S(β̂ⱼ_ols, λ)
│   │   │   └─ S(z, λ) = sign(z)(|z| - λ)₊
│   │   ├─ Fast for large problems
│   │   └─ Exploited by glmnet package
│   ├─ LARS (Least Angle Regression):
│   │   ├─ Efficient algorithm computing entire path
│   │   ├─ Adds variables incrementally
│   │   └─ O(p³) complexity
│   ├─ Proximal Gradient:
│   │   ├─ Gradient step on smooth part (RSS)
│   │   └─ Proximal step on non-smooth part (L1)
│   ├─ Subgradient Methods:
│   │   └─ Handle non-differentiability
│   └─ No Closed Form:
│       └─ Unlike ridge, requires iterative algorithms
├─ Lasso Solution Path:
│   ├─ λ = 0: β̂_lasso = β̂_OLS (if p<n)
│   ├─ λ_max: First λ where all β̂ = 0
│   │   └─ λ_max = max_j |X'ⱼY|
│   ├─ Intermediate λ: Some coefficients zero, others non-zero
│   ├─ λ → ∞: β̂_lasso → 0 (complete sparsity)
│   └─ Piecewise Linear: Path linear in λ between knots
├─ Variable Selection:
│   ├─ Active Set: {j : β̂ⱼ ≠ 0}
│   │   └─ Size controlled by λ
│   ├─ Selection Order: Variables enter path sequentially
│   ├─ Degrees of Freedom: E[df] ≈ |Active Set|
│   ├─ False Positives: Type I error (selecting irrelevant)
│   ├─ False Negatives: Type II error (missing relevant)
│   └─ Selection Stability:
│       ├─ Can be unstable with correlated predictors
│       └─ Bootstrap/stability selection improves
├─ Tuning Parameter Selection (λ):
│   ├─ Cross-Validation (Standard):
│   │   ├─ k-fold CV: Minimize prediction error
│   │   ├─ Choose λ minimizing CV error (λ_min)
│   │   ├─ Alternative: λ_1se (within 1 SE of λ_min)
│   │   │   └─ More parsimonious, better generalization
│   │   └─ Plot: CV error vs log(λ)
│   ├─ Information Criteria:
│   │   ├─ AIC_lasso = n·log(RSS/n) + 2·df(λ)
│   │   ├─ BIC_lasso = n·log(RSS/n) + log(n)·df(λ)
│   │   └─ Extended BIC: Accounts for variable selection
│   ├─ Cross-Validation Strategies:
│   │   ├─ Time Series: Respect temporal order
│   │   ├─ Panel Data: Block by individual
│   │   └─ Spatial Data: Respect spatial dependence
│   └─ Grid Search:
│       └─ Test λ ∈ exp(seq(log(λ_max), log(λ_max/1000), length=100))
├─ Statistical Properties:
│   ├─ Consistency:
│   │   ├─ Under restricted eigenvalue condition
│   │   └─ Variable selection consistent if irrepresentable condition holds
│   ├─ Oracle Property:
│   │   ├─ Lasso does NOT have oracle property
│   │   └─ Adaptive lasso does (weighted L1 penalty)
│   ├─ Prediction Error:
│   │   └─ ||β̂ - β_true||₂² = O(s·log(p)/n) where s = |{j : βⱼ ≠ 0}|
│   ├─ Bias: E[β̂_lasso] ≠ β (biased, especially for large true coefficients)
│   ├─ Variance: Lower than OLS due to shrinkage
│   └─ Non-Asymptotic Bounds:
│       └─ High-dimensional theory (p >> n)
├─ Inference:
│   ├─ Challenge: Post-selection inference invalid
│   │   └─ Standard errors ignore selection process
│   ├─ Debiased Lasso:
│   │   ├─ Two-stage: Lasso selection → OLS on selected
│   │   └─ Corrects bias, valid inference
│   ├─ Selective Inference:
│   │   ├─ Condition on selected model
│   │   └─ Valid p-values post-selection
│   ├─ Bootstrap:
│   │   ├─ Pairs bootstrap for confidence intervals
│   │   └─ Model-based bootstrap
│   └─ Cross-Validation:
│       └─ Nested CV for unbiased performance estimates
├─ Extensions:
│   ├─ Adaptive Lasso:
│   │   ├─ Weighted L1: Σwⱼ|βⱼ| where wⱼ = 1/|β̂ⱼ_initial|^γ
│   │   ├─ Oracle property (asymptotically efficient)
│   │   └─ Better variable selection
│   ├─ Group Lasso:
│   │   ├─ Penalty: ΣΣ√pᵍ·||βᵍ||₂
│   │   ├─ Selects groups of variables together
│   │   └─ Used for categorical variables, pathways
│   ├─ Fused Lasso:
│   │   ├─ Penalty: Σ|βⱼ| + Σ|βⱼ - βⱼ₋₁|
│   │   └─ Encourages adjacent coefficients similar
│   ├─ Relaxed Lasso:
│   │   ├─ Lasso for selection → OLS on selected
│   │   └─ Reduces bias
│   └─ Square-Root Lasso:
│       ├─ ||Y - Xβ|| + λ||β||₁
│       └─ λ selection independent of noise level
├─ Computational Efficiency:
│   ├─ glmnet (R): State-of-art coordinate descent
│   ├─ sklearn (Python): Coordinate descent implementation
│   ├─ Warm Starts: Use previous λ solution as initialization
│   ├─ Active Set Strategy: Update only non-zero coefficients
│   └─ Complexity: O(np·iterations) per λ
├─ Standardization:
│   ├─ Critical: Standardize X before lasso
│   │   └─ L1 penalty scale-dependent
│   ├─ Standardization: X̃ⱼ = (Xⱼ - X̄ⱼ)/SD(Xⱼ)
│   ├─ Intercept: Center Y, exclude from penalty
│   └─ Back-Transform:
│       └─ β̂ⱼ = β̂ⱼ_standardized / SD(Xⱼ)
└─ Applications:
    ├─ Genomics: Gene selection, GWAS, expression prediction
    ├─ Finance: Factor selection, portfolio optimization
    ├─ Text Mining: Feature selection from bag-of-words
    ├─ Signal Processing: Compressed sensing
    └─ Economics: High-dimensional forecasting, instrument selection
```

**Interaction:** Standardize X → Compute λ_max → CV over λ grid → Coordinate descent → Evaluate selected model → Post-selection inference

## 5. Mini-Project
Implement lasso regression with variable selection and compare to OLS and ridge:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, lasso_path, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

np.random.seed(456)

# ===== Simulate Sparse Data =====
n = 150  # Sample size
p = 100  # Number of predictors (high-dimensional)
s = 10   # Number of true non-zero coefficients (sparse)

# Generate uncorrelated predictors
X = np.random.randn(n, p)

# Create some correlation structure for groups
for i in range(0, p, 10):  # Every 10th variable starts a group
    if i + 5 <= p:
        # Make group of 5 correlated
        base = X[:, i].copy()
        for j in range(1, 5):
            if i+j < p:
                X[:, i+j] = 0.7 * base + 0.3 * X[:, i+j]

# True sparse coefficient vector
beta_true = np.zeros(p)
true_indices = np.array([0, 5, 12, 23, 35, 41, 56, 67, 78, 89])  # 10 non-zero
beta_true[true_indices] = np.array([5, -3, 2.5, -2, 1.8, -1.5, 1.2, -1, 0.8, -0.6])

# Generate Y with noise
epsilon = np.random.normal(0, 1.5, n)
Y = X @ beta_true + epsilon

# Signal-to-noise ratio
signal_var = np.var(X @ beta_true)
noise_var = np.var(epsilon)
snr = signal_var / noise_var

print("="*80)
print("LASSO REGRESSION (L1 REGULARIZATION)")
print("="*80)
print(f"\nSimulation Setup:")
print(f"  Sample size (n): {n}")
print(f"  Number of predictors (p): {p}")
print(f"  True non-zero coefficients (s): {s}")
print(f"  Signal-to-noise ratio: {snr:.2f}")
print(f"\nTrue non-zero indices:")
print(f"  {true_indices}")
print(f"\nTrue β values (non-zero):")
print(f"  {beta_true[true_indices]}")

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

Y_train_mean = Y_train.mean()
Y_train_centered = Y_train - Y_train_mean
Y_test_centered = Y_test - Y_train_mean

print("\n✓ Data standardized")

# ===== OLS Baseline (will fail due to p > n) =====
print("\n" + "="*80)
print("OLS REGRESSION (p > n: CANNOT FIT)")
print("="*80)
print(f"Cannot estimate OLS with p={p} > n={train_size}")

# ===== Lasso with Cross-Validation =====
print("\n" + "="*80)
print("LASSO REGRESSION WITH CV")
print("="*80)

# Define alpha grid (sklearn uses alpha instead of lambda)
alphas = np.logspace(-3, 1, 100)

# LassoCV with 5-fold cross-validation
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, Y_train_centered)

optimal_lambda = lasso_cv.alpha_
print(f"Optimal λ (from 5-fold CV): {optimal_lambda:.4f}")

# Fit lasso with optimal lambda
lasso_model = Lasso(alpha=optimal_lambda, max_iter=10000)
lasso_model.fit(X_train_scaled, Y_train_centered)
beta_lasso = lasso_model.coef_

# Performance
Y_train_pred_lasso = lasso_model.predict(X_train_scaled)
train_mse_lasso = mean_squared_error(Y_train_centered, Y_train_pred_lasso)
train_r2_lasso = r2_score(Y_train_centered, Y_train_pred_lasso)

Y_test_pred_lasso = lasso_model.predict(X_test_scaled)
test_mse_lasso = mean_squared_error(Y_test_centered, Y_test_pred_lasso)
test_r2_lasso = r2_score(Y_test_centered, Y_test_pred_lasso)

# Variable selection
selected_vars = np.where(np.abs(beta_lasso) > 1e-5)[0]
n_selected = len(selected_vars)

print(f"\nIn-Sample Performance:")
print(f"  MSE: {train_mse_lasso:.4f}")
print(f"  R²: {train_r2_lasso:.4f}")

print(f"\nOut-of-Sample Performance:")
print(f"  MSE: {test_mse_lasso:.4f}")
print(f"  R²: {test_r2_lasso:.4f}")

print(f"\nVariable Selection:")
print(f"  Selected: {n_selected} out of {p} predictors")
print(f"  True non-zero: {s}")
print(f"  Selected indices: {selected_vars}")

# ===== Selection Accuracy =====
print("\n" + "="*80)
print("SELECTION ACCURACY")
print("="*80)

# True positives: correctly selected
true_positives = np.intersect1d(selected_vars, true_indices)
# False positives: incorrectly selected
false_positives = np.setdiff1d(selected_vars, true_indices)
# False negatives: missed true variables
false_negatives = np.setdiff1d(true_indices, selected_vars)

n_tp = len(true_positives)
n_fp = len(false_positives)
n_fn = len(false_negatives)

# Metrics
precision = n_tp / n_selected if n_selected > 0 else 0
recall = n_tp / s
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"True Positives: {n_tp} (correctly selected)")
print(f"False Positives: {n_fp} (incorrectly selected)")
print(f"False Negatives: {n_fn} (missed true variables)")
print(f"\nPrecision: {precision:.3f} (fraction of selected that are true)")
print(f"Recall: {recall:.3f} (fraction of true that are selected)")
print(f"F1 Score: {f1_score:.3f}")

if n_tp > 0:
    print(f"\n✓ Correctly Identified: {true_positives}")
if n_fp > 0:
    print(f"✗ False Positives: {false_positives}")
if n_fn > 0:
    print(f"✗ Missed Variables: {false_negatives}")

# ===== Ridge Comparison =====
print("\n" + "="*80)
print("RIDGE COMPARISON")
print("="*80)

ridge_model = Ridge(alpha=optimal_lambda)
ridge_model.fit(X_train_scaled, Y_train_centered)
beta_ridge = ridge_model.coef_

Y_test_pred_ridge = ridge_model.predict(X_test_scaled)
test_mse_ridge = mean_squared_error(Y_test_centered, Y_test_pred_ridge)
test_r2_ridge = r2_score(Y_test_centered, Y_test_pred_ridge)

ridge_nonzero = np.sum(np.abs(beta_ridge) > 1e-5)

print(f"Ridge Performance:")
print(f"  Test MSE: {test_mse_ridge:.4f}")
print(f"  Test R²: {test_r2_ridge:.4f}")
print(f"  Non-zero coefficients: {ridge_nonzero} (all, effectively)")

comparison_df = pd.DataFrame({
    'Method': ['Lasso', 'Ridge'],
    'Test_MSE': [test_mse_lasso, test_mse_ridge],
    'Test_R2': [test_r2_lasso, test_r2_ridge],
    'N_Selected': [n_selected, ridge_nonzero],
    'L1_Norm': [np.sum(np.abs(beta_lasso)), np.sum(np.abs(beta_ridge))]
})

print(f"\nComparison:")
print(comparison_df.to_string(index=False))

# ===== Regularization Path =====
print("\n" + "="*80)
print("REGULARIZATION PATH")
print("="*80)

# Compute lasso path
alphas_path, coefs_path, _ = lasso_path(X_train_scaled, Y_train_centered,
                                        alphas=alphas, max_iter=10000)

# Compute test MSE along path
test_mse_path = []
n_selected_path = []

for alpha, coef in zip(alphas_path, coefs_path.T):
    lasso_temp = Lasso(alpha=alpha, max_iter=10000)
    lasso_temp.fit(X_train_scaled, Y_train_centered)
    pred = lasso_temp.predict(X_test_scaled)
    test_mse_path.append(mean_squared_error(Y_test_centered, pred))
    n_selected_path.append(np.sum(np.abs(coef) > 1e-5))

test_mse_path = np.array(test_mse_path)
n_selected_path = np.array(n_selected_path)

# Find optimal based on test MSE
optimal_idx = np.argmin(test_mse_path)

print(f"Path computed over {len(alphas_path)} λ values")
print(f"Optimal λ (test MSE): {alphas_path[optimal_idx]:.4f}")
print(f"Optimal λ (CV): {optimal_lambda:.4f}")

# ===== Lambda Max Calculation =====
lambda_max_theoretical = np.max(np.abs(X_train_scaled.T @ Y_train_centered)) / train_size
print(f"\nλ_max (theoretical): {lambda_max_theoretical:.4f}")
print(f"  All coefficients zero for λ ≥ λ_max")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Regularization Path (Lasso)
for i in range(p):
    if i in true_indices:
        axes[0, 0].plot(np.log10(alphas_path), coefs_path[i, :],
                       linewidth=2, alpha=0.9, color='red')
    else:
        axes[0, 0].plot(np.log10(alphas_path), coefs_path[i, :],
                       linewidth=0.5, alpha=0.3, color='gray')

axes[0, 0].axvline(np.log10(optimal_lambda), color='blue',
                  linestyle='--', linewidth=2, label=f'Optimal λ (CV)')
axes[0, 0].set_xlabel('log₁₀(λ)')
axes[0, 0].set_ylabel('Coefficient Value')
axes[0, 0].set_title('Lasso Regularization Path\n(Red = True Non-Zero)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Number of Selected Variables vs Lambda
axes[0, 1].plot(np.log10(alphas_path), n_selected_path, linewidth=2)
axes[0, 1].axhline(s, color='red', linestyle='--', linewidth=2,
                  label=f'True s={s}')
axes[0, 1].axvline(np.log10(optimal_lambda), color='blue',
                  linestyle='--', linewidth=2, label='Optimal λ')
axes[0, 1].set_xlabel('log₁₀(λ)')
axes[0, 1].set_ylabel('Number of Selected Variables')
axes[0, 1].set_title('Variable Selection Path')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Test MSE vs Lambda
axes[0, 2].plot(np.log10(alphas_path), test_mse_path, linewidth=2)
axes[0, 2].axvline(np.log10(optimal_lambda), color='blue',
                  linestyle='--', linewidth=2, label='Optimal λ')
axes[0, 2].scatter(np.log10(alphas_path[optimal_idx]),
                  test_mse_path[optimal_idx],
                  color='red', s=200, marker='*', zorder=5,
                  label='Min Test MSE')
axes[0, 2].set_xlabel('log₁₀(λ)')
axes[0, 2].set_ylabel('Test MSE')
axes[0, 2].set_title('Test MSE vs Regularization')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Coefficient Comparison (True vs Estimated)
indices_to_plot = np.concatenate([true_indices, selected_vars])
indices_to_plot = np.unique(indices_to_plot)[:20]  # First 20 for visibility

x_pos = np.arange(len(indices_to_plot))
width = 0.35

axes[1, 0].bar(x_pos - width/2, beta_true[indices_to_plot], width,
              label='True', alpha=0.8)
axes[1, 0].bar(x_pos + width/2, beta_lasso[indices_to_plot], width,
              label='Lasso', alpha=0.8)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(indices_to_plot, fontsize=7)
axes[1, 0].set_xlabel('Variable Index')
axes[1, 0].set_ylabel('Coefficient Value')
axes[1, 0].set_title('True vs Lasso Coefficients')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 5: Lasso vs Ridge Coefficients
axes[1, 1].scatter(beta_lasso, beta_ridge, alpha=0.6, s=30)
axes[1, 1].axhline(0, color='black', linewidth=0.5)
axes[1, 1].axvline(0, color='black', linewidth=0.5)

# Highlight true non-zero
axes[1, 1].scatter(beta_lasso[true_indices], beta_ridge[true_indices],
                  color='red', s=100, alpha=0.8, label='True Non-Zero',
                  edgecolors='black', linewidths=1.5)

axes[1, 1].set_xlabel('Lasso Coefficient')
axes[1, 1].set_ylabel('Ridge Coefficient')
axes[1, 1].set_title('Lasso vs Ridge: Sparsity Effect')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Prediction Scatter
axes[1, 2].scatter(Y_test_centered, Y_test_pred_lasso, alpha=0.6, s=50)
axes[1, 2].plot([Y_test_centered.min(), Y_test_centered.max()],
               [Y_test_centered.min(), Y_test_centered.max()],
               'r--', linewidth=2)
axes[1, 2].set_xlabel('True Y')
axes[1, 2].set_ylabel('Predicted Y (Lasso)')
axes[1, 2].set_title(f'Out-of-Sample Predictions\n(R² = {test_r2_lasso:.3f}, {n_selected} vars)')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('lasso_regression_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Stability Selection via Bootstrap =====
print("\n" + "="*80)
print("STABILITY SELECTION (Bootstrap)")
print("="*80)

n_bootstrap = 100
selection_freq = np.zeros(p)

for b in range(n_bootstrap):
    # Bootstrap sample
    boot_idx = np.random.choice(train_size, train_size, replace=True)
    X_boot = X_train_scaled[boot_idx]
    Y_boot = Y_train_centered[boot_idx]
    
    # Fit lasso
    lasso_boot = Lasso(alpha=optimal_lambda, max_iter=10000)
    lasso_boot.fit(X_boot, Y_boot)
    
    # Record selected variables
    selected = np.abs(lasso_boot.coef_) > 1e-5
    selection_freq += selected

selection_freq /= n_bootstrap

# Stable variables (selected in >80% of bootstrap samples)
stable_threshold = 0.8
stable_vars = np.where(selection_freq >= stable_threshold)[0]

print(f"Bootstrap samples: {n_bootstrap}")
print(f"Stability threshold: {stable_threshold}")
print(f"Stable variables: {len(stable_vars)}")
print(f"  Indices: {stable_vars}")

# Check overlap with true variables
stable_true = np.intersect1d(stable_vars, true_indices)
print(f"\nStable ∩ True: {len(stable_true)} variables")
print(f"  Indices: {stable_true}")

# Visualize stability
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

colors = ['red' if i in true_indices else 'gray' for i in range(p)]
ax.bar(range(p), selection_freq, color=colors, alpha=0.7)
ax.axhline(stable_threshold, color='blue', linestyle='--', linewidth=2,
          label=f'Stability Threshold ({stable_threshold})')
ax.set_xlabel('Variable Index')
ax.set_ylabel('Selection Frequency')
ax.set_title('Stability Selection via Bootstrap\n(Red = True Non-Zero Variables)')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lasso_stability_selection.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Lambda 1SE Rule =====
print("\n" + "="*80)
print("LAMBDA 1SE RULE")
print("="*80)

# Get CV results
cv_results = lasso_cv.mse_path_.mean(axis=1)
cv_std = lasso_cv.mse_path_.std(axis=1)

# Find lambda_min
lambda_min_idx = np.argmin(cv_results)
lambda_min = lasso_cv.alphas_[lambda_min_idx]
min_mse = cv_results[lambda_min_idx]
min_se = cv_std[lambda_min_idx]

# Lambda 1SE: largest lambda within 1 SE of minimum
lambda_1se_idx = np.where(cv_results <= min_mse + min_se)[0][-1]
lambda_1se = lasso_cv.alphas_[lambda_1se_idx]

print(f"λ_min: {lambda_min:.4f} (minimizes CV error)")
print(f"λ_1se: {lambda_1se:.4f} (most regularized within 1 SE)")

# Fit with lambda_1se
lasso_1se = Lasso(alpha=lambda_1se, max_iter=10000)
lasso_1se.fit(X_train_scaled, Y_train_centered)
beta_lasso_1se = lasso_1se.coef_

Y_test_pred_1se = lasso_1se.predict(X_test_scaled)
test_mse_1se = mean_squared_error(Y_test_centered, Y_test_pred_1se)
n_selected_1se = np.sum(np.abs(beta_lasso_1se) > 1e-5)

print(f"\nλ_1se Model:")
print(f"  Test MSE: {test_mse_1se:.4f}")
print(f"  Variables selected: {n_selected_1se}")

print(f"\nλ_min Model:")
print(f"  Test MSE: {test_mse_lasso:.4f}")
print(f"  Variables selected: {n_selected}")

print(f"\nλ_1se selects {n_selected - n_selected_1se} fewer variables")
print("  → More parsimonious model with similar performance")

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n1. Variable Selection Performance:")
print(f"   Precision: {precision:.3f} ({n_tp}/{n_selected} selected are correct)")
print(f"   Recall: {recall:.3f} ({n_tp}/{s} true variables found)")
print(f"   F1 Score: {f1_score:.3f}")

print("\n2. Prediction Performance:")
print(f"   Lasso Test MSE: {test_mse_lasso:.4f}, R²: {test_r2_lasso:.4f}")
print(f"   Ridge Test MSE: {test_mse_ridge:.4f}, R²: {test_r2_ridge:.4f}")
if test_mse_lasso < test_mse_ridge:
    print("   ✓ Lasso achieves better prediction via sparsity")
else:
    print("   ✗ Ridge slightly better (possibly sparse structure not strong)")

print("\n3. Stability:")
print(f"   {len(stable_vars)} variables stable across {n_bootstrap} bootstrap samples")
print(f"   {len(stable_true)} of {s} true variables consistently selected")

print("\n4. Parsimony:")
print(f"   λ_min: {n_selected} variables")
print(f"   λ_1se: {n_selected_1se} variables (more parsimonious)")
print(f"   Consider λ_1se for simpler, more interpretable model")

print("\n5. When to Use Lasso:")
print("   • True model sparse (few relevant predictors)")
print("   • Want interpretable variable selection")
print("   • High-dimensional (p > n or p ≈ n)")
print("   • Feature screening before further analysis")

print("\n6. Limitations:")
print("   • Arbitrary selection among correlated predictors")
print("   • Biased coefficient estimates (especially for large true β)")
print("   • Post-selection inference requires special methods")
print("   • Consider elastic net if grouped variables important")

print("\n7. Practical Tips:")
print("   • Standardize predictors always")
print("   • Use λ_1se for parsimony")
print("   • Bootstrap for stability assessment")
print("   • Debiased lasso for valid inference")
print("   • Nested CV for unbiased performance evaluation")
```

## 6. Challenge Round
When does lasso regression fail or mislead?
- **Grouped correlated predictors**: Lasso picks one arbitrarily → Elastic net better; oracle interpretation invalid
- **Large true coefficients**: Lasso overshrinks → Adaptive lasso or relaxed lasso corrects; biased estimates
- **n < s (more signals than samples)**: Lasso can select at most n variables → Cannot recover all true signals
- **Post-selection inference**: Naive SE/p-values invalid → Use selective inference or debiased lasso
- **Irrepresentable condition violated**: Selection consistency fails → May include irrelevant, miss relevant variables
- **Stability**: Different bootstrap samples select different variables → Report selection frequencies, not just point estimate

## 7. Key References
- [Tibshirani (1996) - Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- [Hastie, Tibshirani & Wainwright - Statistical Learning with Sparsity](https://web.stanford.edu/~hastie/StatLearnSparsity/)
- [Friedman, Hastie & Tibshirani (2010) - Regularization Paths for Generalized Linear Models via Coordinate Descent](https://doi.org/10.18637/jss.v033.i01)

---
**Status:** Leading variable selection method | **Complements:** Ridge, Elastic Net, Adaptive Lasso, Stability Selection
