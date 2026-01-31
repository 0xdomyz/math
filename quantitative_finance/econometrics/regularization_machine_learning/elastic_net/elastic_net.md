# Elastic Net Regression

## 1. Concept Skeleton
**Definition:** Linear regression combining L1 (lasso) and L2 (ridge) penalties; balances variable selection and grouping; handles correlated predictors better than lasso alone  
**Purpose:** Variable selection with grouped correlated predictors; stable selection; addresses lasso limitations; flexible regularization  
**Prerequisites:** Ridge regression, lasso regression, convex optimization, cross-validation, regularization paths, grouped variables

## 2. Comparative Framing
| Method | OLS | Ridge (L2) | Lasso (L1) | Elastic Net | Group Lasso |
|--------|-----|------------|------------|-------------|-------------|
| **Penalty** | None | λΣβⱼ² | λΣ\|βⱼ\| | λ₁Σβⱼ² + λ₂Σ\|βⱼ\| | ΣΣ||βᵍ||₂ |
| **Sparsity** | Dense | Dense | Sparse | Sparse | Group-sparse |
| **Correlated Vars** | Unstable | Shrinks together | Picks one | Averages group | Selects groups |
| **Selection Limit** | n/a | n/a | n variables max | No limit | Group-level |
| **Grouping Effect** | No | Yes | No | Yes | Explicit |
| **Tuning Parameters** | 0 | 1 (λ) | 1 (λ) | 2 (λ₁, λ₂) | 1 (λ) |

## 3. Examples + Counterexamples

**Classic Example:**  
Stock return prediction with factor groups: 5 momentum factors (r=0.85), 6 value factors (r=0.80), 4 size factors (r=0.75). Lasso picks 1 per group. Elastic net selects 2-3 per group with stable weights. Test R² improves 15%.

**Failure Case:**  
True model has 3 uncorrelated strong predictors. Elastic net overshrinks due to unnecessary L2 penalty. Lasso achieves same sparsity with less bias. No correlation structure to exploit.

**Edge Case:**  
n=50, p=10000 (genomics). Lasso limited to 50 variables. Elastic net also selects ~50 but averages within gene pathways. Interpretation and prediction similar but pathway structure preserved.

## 4. Layer Breakdown
```
Elastic Net Framework:
├─ Objective Function:
│   ├─ Elastic Net: min Σ(yᵢ - Xᵢ'β)² + λ₁Σβⱼ² + λ₂Σ|βⱼ|
│   │   ├─ First term: Residual sum of squares (fit)
│   │   ├─ Second term: L2 penalty (grouping, ridge)
│   │   ├─ Third term: L1 penalty (sparsity, lasso)
│   │   └─ λ₁, λ₂ ≥ 0: Two tuning parameters
│   ├─ Alternative Parameterization:
│   │   ├─ min Σ(yᵢ - Xᵢ'β)² + λ[(1-α)Σβⱼ² + αΣ|βⱼ|]
│   │   │   ├─ λ: Overall regularization strength
│   │   │   └─ α ∈ [0,1]: Mixing parameter
│   │   ├─ α = 0: Pure ridge
│   │   ├─ α = 1: Pure lasso
│   │   └─ α ∈ (0,1): Elastic net
│   ├─ Constrained Form:
│   │   └─ min ||Y - Xβ||² subject to (1-α)||β||₂² + α||β||₁ ≤ t
│   └─ Naive Elastic Net:
│       └─ β̂_naive solves above; β̂_EN = √(1+λ₁)·β̂_naive (rescaling)
├─ Motivation and Advantages:
│   ├─ Lasso Limitations:
│   │   ├─ Correlated predictors: Arbitrary selection
│   │   ├─ Selection limit: At most n variables when p > n
│   │   ├─ Instability: Different solutions under perturbation
│   │   └─ No grouping: Treats all predictors independently
│   ├─ Elastic Net Benefits:
│   │   ├─ Grouping Effect: Correlated variables averaged
│   │   │   └─ If Xⱼ ≈ Xₖ, then β̂ⱼ ≈ β̂ₖ (ridge component)
│   │   ├─ Stability: More robust to correlation
│   │   ├─ No Selection Limit: Can select > n variables
│   │   └─ Sparsity: Still performs variable selection (L1)
│   └─ Theoretical Result:
│       └─ If Xⱼ'Xₖ → 1, elastic net has |β̂ⱼ - β̂ₖ| → 0
├─ Optimization:
│   ├─ Coordinate Descent (Standard):
│   │   ├─ Update: β̂ⱼ = S(β̃ⱼ, λ₂) / (1 + λ₁)
│   │   │   └─ S(z, λ) = sign(z)(|z| - λ)₊ (soft-thresholding)
│   │   ├─ Converges quickly for large problems
│   │   └─ glmnet uses this algorithm
│   ├─ LARS-EN:
│   │   └─ Extension of LARS algorithm for elastic net
│   ├─ Proximal Methods:
│   │   └─ Efficient for non-smooth composite objectives
│   └─ Warm Starts:
│       └─ Use previous solution as initialization
├─ Tuning Parameters:
│   ├─ Two Parameters (λ, α) or (λ₁, λ₂):
│   │   ├─ Grid Search: Test combinations
│   │   ├─ Typical α values: [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
│   │   └─ For each α, search λ via cross-validation
│   ├─ Common Strategy:
│   │   ├─ Fix α (often α=0.5 or α=0.9)
│   │   └─ Select λ via CV (1D search)
│   ├─ Full 2D Search:
│   │   ├─ More thorough but computationally expensive
│   │   └─ Use coarse grid, then refine
│   ├─ Cross-Validation:
│   │   ├─ 5-fold or 10-fold CV
│   │   ├─ Choose (α, λ) minimizing CV error
│   │   └─ sklearn ElasticNetCV does this automatically
│   └─ Information Criteria:
│       └─ Extend AIC/BIC with effective degrees of freedom
├─ Regularization Path:
│   ├─ Fix α, vary λ:
│   │   ├─ λ = 0: Unregularized (if p < n)
│   │   ├─ λ → ∞: All coefficients → 0
│   │   └─ Piecewise linear path (like lasso)
│   ├─ Fix λ, vary α:
│   │   ├─ α = 0: Ridge solution (dense)
│   │   ├─ α = 1: Lasso solution (sparse)
│   │   └─ Intermediate: Trade-off between sparsity and grouping
│   └─ Surface Plot:
│       └─ Test error over (α, λ) grid reveals optimal region
├─ Statistical Properties:
│   ├─ Oracle Properties:
│   │   ├─ Standard elastic net: No oracle property
│   │   └─ Adaptive elastic net: Yes (with adaptive weights)
│   ├─ Consistency:
│   │   ├─ Variable selection consistent under conditions
│   │   └─ Less stringent than lasso (no irrepresentable condition)
│   ├─ Grouping:
│   │   └─ Theorem: For highly correlated Xⱼ, Xₖ, elastic net β̂ⱼ ≈ β̂ₖ
│   ├─ Prediction Error:
│   │   └─ Often improves over lasso when correlation present
│   └─ Bias-Variance:
│       └─ Trades off via α: more L2 → less variance, more bias
├─ Parameter Interpretation:
│   ├─ α (mixing parameter):
│   │   ├─ α → 0: Emphasizes grouping, less sparsity (ridge-like)
│   │   ├─ α → 1: Emphasizes sparsity, less grouping (lasso-like)
│   │   └─ α = 0.5: Equal weight to both penalties
│   ├─ λ (overall strength):
│   │   ├─ Controls total amount of regularization
│   │   └─ Higher λ → sparser, more shrinkage
│   └─ Practical Ranges:
│       ├─ α ∈ [0.5, 0.95]: Common in practice
│       └─ λ: Determined by CV, data-dependent
├─ Inference:
│   ├─ Challenge: Same as lasso (post-selection bias)
│   ├─ Debiased Elastic Net:
│   │   └─ Two-stage: Elastic net selection → OLS on selected
│   ├─ Bootstrap:
│   │   ├─ Confidence intervals via percentile method
│   │   └─ Variable importance via selection frequency
│   ├─ Stability Selection:
│   │   ├─ Run on bootstrap/subsamples
│   │   └─ Report selection probabilities
│   └─ Cross-Validation:
│       └─ Nested CV for unbiased performance
├─ Extensions:
│   ├─ Adaptive Elastic Net:
│   │   ├─ Weighted penalties: λ₁Σwⱼβⱼ² + λ₂Σvⱼ|βⱼ|
│   │   ├─ Weights: wⱼ, vⱼ from initial estimates
│   │   └─ Oracle property achieved
│   ├─ Sparse Group Elastic Net:
│   │   ├─ Combines elastic net with group lasso
│   │   └─ Sparsity at both group and individual level
│   ├─ Fused Elastic Net:
│   │   ├─ Adds fusion penalty: Σ|βⱼ - βⱼ₋₁|
│   │   └─ For ordered predictors (time, space)
│   └─ Elastic SCAD:
│       └─ Replaces L1 with SCAD penalty (non-convex)
├─ Computational:
│   ├─ glmnet (R): Fast implementation via Fortran
│   ├─ sklearn.ElasticNet (Python): Coordinate descent
│   ├─ ElasticNetCV: Automatic CV over λ for fixed α
│   ├─ Complexity: O(np·iter) per (α, λ) pair
│   └─ Memory: O(np) for dense X, less for sparse
├─ Standardization:
│   ├─ Critical: Standardize before elastic net
│   ├─ Penalties scale-dependent
│   └─ Back-transform after fitting
└─ Applications:
    ├─ Genomics: Gene selection with pathway structure
    ├─ Finance: Factor models, risk prediction
    ├─ Text Mining: Document classification with correlated features
    ├─ Image Processing: Pixel selection with spatial correlation
    └─ Economics: Forecasting with correlated indicators
```

**Interaction:** Standardize X → Choose α (mixing) → CV over λ grid → Coordinate descent → Evaluate selection and prediction → Bootstrap for stability

## 5. Mini-Project
Implement elastic net with grouped correlated predictors and compare to lasso/ridge:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

np.random.seed(789)

# ===== Simulate Data with Grouped Correlated Predictors =====
n = 200  # Sample size
n_groups = 10  # Number of groups
group_size = 5  # Variables per group
p = n_groups * group_size  # Total predictors

print("="*80)
print("ELASTIC NET REGRESSION")
print("="*80)
print(f"\nSimulation Setup:")
print(f"  Sample size (n): {n}")
print(f"  Number of groups: {n_groups}")
print(f"  Variables per group: {group_size}")
print(f"  Total predictors (p): {p}")

# Generate data with group structure
X = np.zeros((n, p))
beta_true = np.zeros(p)

# Create groups with high within-group correlation
for g in range(n_groups):
    start_idx = g * group_size
    end_idx = start_idx + group_size
    
    # Base variable for group
    base = np.random.randn(n)
    
    # Group members highly correlated with base
    for i in range(group_size):
        idx = start_idx + i
        X[:, idx] = 0.85 * base + 0.15 * np.random.randn(n)
    
    # Only first 5 groups have non-zero coefficients
    if g < 5:
        # All variables in active groups have same coefficient
        group_coef = np.random.choice([-2, -1, 1, 2])
        beta_true[start_idx:end_idx] = group_coef
        print(f"  Group {g}: β = {group_coef} (active)")

# Generate Y with noise
epsilon = np.random.normal(0, 2, n)
Y = X @ beta_true + epsilon

signal_var = np.var(X @ beta_true)
noise_var = np.var(epsilon)
snr = signal_var / noise_var

print(f"\nSignal-to-noise ratio: {snr:.2f}")
print(f"True non-zero groups: 5 out of {n_groups}")
print(f"True non-zero coefficients: {5 * group_size} out of {p}")

# ===== Train-Test Split =====
train_size = int(0.7 * n)
train_idx = np.arange(train_size)
test_idx = np.arange(train_size, n)

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Y_train_mean = Y_train.mean()
Y_train_centered = Y_train - Y_train_mean
Y_test_centered = Y_test - Y_train_mean

print(f"\nTrain/Test Split: {train_size}/{n - train_size}")
print("✓ Data standardized")

# ===== Elastic Net with Cross-Validation =====
print("\n" + "="*80)
print("ELASTIC NET WITH CV")
print("="*80)

# Test different alpha values
alphas_to_test = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
results = []

for alpha in alphas_to_test:
    # ElasticNetCV automatically searches over lambda (l1_ratio=alpha)
    en_cv = ElasticNetCV(l1_ratio=alpha, cv=5, max_iter=10000, 
                         random_state=42, n_alphas=100)
    en_cv.fit(X_train_scaled, Y_train_centered)
    
    # Optimal lambda
    optimal_lambda = en_cv.alpha_
    
    # Predictions
    Y_test_pred = en_cv.predict(X_test_scaled)
    test_mse = mean_squared_error(Y_test_centered, Y_test_pred)
    test_r2 = r2_score(Y_test_centered, Y_test_pred)
    
    # Variable selection
    beta_en = en_cv.coef_
    n_selected = np.sum(np.abs(beta_en) > 1e-5)
    
    results.append({
        'alpha': alpha,
        'lambda': optimal_lambda,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'n_selected': n_selected,
        'L1_norm': np.sum(np.abs(beta_en)),
        'L2_norm': np.linalg.norm(beta_en)
    })
    
    print(f"α = {alpha:.2f}: λ = {optimal_lambda:.4f}, "
          f"Test MSE = {test_mse:.4f}, Selected = {n_selected}")

results_df = pd.DataFrame(results)

# Find best alpha
best_idx = results_df['test_mse'].idxmin()
best_alpha = results_df.loc[best_idx, 'alpha']
best_lambda = results_df.loc[best_idx, 'lambda']

print(f"\n✓ Best α = {best_alpha}, λ = {best_lambda:.4f}")

# Fit final model with best parameters
en_final = ElasticNet(alpha=best_lambda, l1_ratio=best_alpha, max_iter=10000)
en_final.fit(X_train_scaled, Y_train_centered)
beta_en_final = en_final.coef_

Y_test_pred_en = en_final.predict(X_test_scaled)
test_mse_en = mean_squared_error(Y_test_centered, Y_test_pred_en)
test_r2_en = r2_score(Y_test_centered, Y_test_pred_en)
n_selected_en = np.sum(np.abs(beta_en_final) > 1e-5)

print(f"\nFinal Elastic Net Performance:")
print(f"  Test MSE: {test_mse_en:.4f}")
print(f"  Test R²: {test_r2_en:.4f}")
print(f"  Variables selected: {n_selected_en} out of {p}")

# ===== Lasso Comparison =====
print("\n" + "="*80)
print("LASSO COMPARISON")
print("="*80)

lasso_cv = ElasticNetCV(l1_ratio=1.0, cv=5, max_iter=10000, 
                        random_state=42, n_alphas=100)
lasso_cv.fit(X_train_scaled, Y_train_centered)

lasso_lambda = lasso_cv.alpha_
beta_lasso = lasso_cv.coef_

Y_test_pred_lasso = lasso_cv.predict(X_test_scaled)
test_mse_lasso = mean_squared_error(Y_test_centered, Y_test_pred_lasso)
test_r2_lasso = r2_score(Y_test_centered, Y_test_pred_lasso)
n_selected_lasso = np.sum(np.abs(beta_lasso) > 1e-5)

print(f"Lasso (α=1.0):")
print(f"  Test MSE: {test_mse_lasso:.4f}")
print(f"  Test R²: {test_r2_lasso:.4f}")
print(f"  Variables selected: {n_selected_lasso}")

# ===== Ridge Comparison =====
print("\n" + "="*80)
print("RIDGE COMPARISON")
print("="*80)

ridge_model = Ridge(alpha=best_lambda / (1 - best_alpha) if best_alpha < 1 else 1.0)
ridge_model.fit(X_train_scaled, Y_train_centered)
beta_ridge = ridge_model.coef_

Y_test_pred_ridge = ridge_model.predict(X_test_scaled)
test_mse_ridge = mean_squared_error(Y_test_centered, Y_test_pred_ridge)
test_r2_ridge = r2_score(Y_test_centered, Y_test_pred_ridge)
n_selected_ridge = p  # Ridge doesn't select

print(f"Ridge (α=0.0):")
print(f"  Test MSE: {test_mse_ridge:.4f}")
print(f"  Test R²: {test_r2_ridge:.4f}")
print(f"  Variables: {n_selected_ridge} (all, no selection)")

# ===== Comparison Table =====
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Method': ['Lasso', f'Elastic Net (α={best_alpha})', 'Ridge'],
    'Test_MSE': [test_mse_lasso, test_mse_en, test_mse_ridge],
    'Test_R2': [test_r2_lasso, test_r2_en, test_r2_ridge],
    'N_Selected': [n_selected_lasso, n_selected_en, n_selected_ridge],
    'L1_Norm': [np.sum(np.abs(beta_lasso)), 
                np.sum(np.abs(beta_en_final)),
                np.sum(np.abs(beta_ridge))]
})

print(comparison.to_string(index=False))

# ===== Grouping Effect Analysis =====
print("\n" + "="*80)
print("GROUPING EFFECT ANALYSIS")
print("="*80)

# Check within-group coefficient similarity
for g in range(n_groups):
    start_idx = g * group_size
    end_idx = start_idx + group_size
    
    # Coefficients for this group
    beta_group_lasso = beta_lasso[start_idx:end_idx]
    beta_group_en = beta_en_final[start_idx:end_idx]
    beta_group_ridge = beta_ridge[start_idx:end_idx]
    beta_group_true = beta_true[start_idx:end_idx]
    
    # Within-group standard deviation
    std_lasso = np.std(beta_group_lasso)
    std_en = np.std(beta_group_en)
    std_ridge = np.std(beta_group_ridge)
    
    # Number selected
    n_sel_lasso = np.sum(np.abs(beta_group_lasso) > 1e-5)
    n_sel_en = np.sum(np.abs(beta_group_en) > 1e-5)
    
    if beta_group_true[0] != 0:  # Active group
        print(f"Group {g} (Active, β_true={beta_group_true[0]}):")
        print(f"  Lasso: {n_sel_lasso}/{group_size} selected, SD={std_lasso:.3f}")
        print(f"  Elastic Net: {n_sel_en}/{group_size} selected, SD={std_en:.3f}")
        print(f"  Ridge: {group_size}/{group_size} retained, SD={std_ridge:.3f}")
        
        # Elastic net should have lower SD (more grouping)
        if std_en < std_lasso:
            print(f"  ✓ Elastic net shows grouping effect (lower SD)")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Alpha Selection (Test MSE vs Alpha)
axes[0, 0].plot(results_df['alpha'], results_df['test_mse'], 
               marker='o', linewidth=2, markersize=8)
axes[0, 0].axvline(best_alpha, color='red', linestyle='--', 
                  linewidth=2, label=f'Best α={best_alpha}')
axes[0, 0].set_xlabel('α (L1 ratio)')
axes[0, 0].set_ylabel('Test MSE')
axes[0, 0].set_title('Optimal Mixing Parameter Selection')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Number of Selected Variables vs Alpha
axes[0, 1].plot(results_df['alpha'], results_df['n_selected'],
               marker='o', linewidth=2, markersize=8, color='green')
axes[0, 1].axhline(5 * group_size, color='red', linestyle='--',
                  linewidth=2, label=f'True non-zero={5*group_size}')
axes[0, 1].set_xlabel('α (L1 ratio)')
axes[0, 1].set_ylabel('Number of Selected Variables')
axes[0, 1].set_title('Sparsity vs Mixing Parameter')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Coefficients Heatmap (Lasso vs Elastic Net vs Ridge)
coef_comparison = np.vstack([beta_lasso, beta_en_final, beta_ridge, beta_true])
sns.heatmap(coef_comparison, cmap='RdBu_r', center=0, 
           yticklabels=['Lasso', 'Elastic Net', 'Ridge', 'True'],
           cbar_kws={'label': 'Coefficient Value'},
           ax=axes[0, 2])
axes[0, 2].set_xlabel('Variable Index')
axes[0, 2].set_title('Coefficient Patterns Across Methods')

# Plot 4: Group-wise Selection (First 5 groups)
group_means_lasso = []
group_means_en = []
group_means_ridge = []
group_selection_lasso = []
group_selection_en = []

for g in range(n_groups):
    start_idx = g * group_size
    end_idx = start_idx + group_size
    
    group_means_lasso.append(np.mean(beta_lasso[start_idx:end_idx]))
    group_means_en.append(np.mean(beta_en_final[start_idx:end_idx]))
    group_means_ridge.append(np.mean(beta_ridge[start_idx:end_idx]))
    
    group_selection_lasso.append(np.sum(np.abs(beta_lasso[start_idx:end_idx]) > 1e-5))
    group_selection_en.append(np.sum(np.abs(beta_en_final[start_idx:end_idx]) > 1e-5))

x_groups = np.arange(n_groups)
width = 0.25

axes[1, 0].bar(x_groups - width, group_selection_lasso, width,
              label='Lasso', alpha=0.8)
axes[1, 0].bar(x_groups, group_selection_en, width,
              label='Elastic Net', alpha=0.8)
axes[1, 0].bar(x_groups + width, [group_size]*n_groups, width,
              label='Ridge', alpha=0.4)
axes[1, 0].axhline(group_size, color='gray', linestyle=':', linewidth=1)
axes[1, 0].set_xlabel('Group Index')
axes[1, 0].set_ylabel('Variables Selected in Group')
axes[1, 0].set_title('Group-wise Variable Selection')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3, axis='y')

# Highlight first 5 groups (active)
for i in range(5):
    axes[1, 0].axvspan(i-0.5, i+0.5, alpha=0.1, color='green')

# Plot 5: Within-Group Coefficient Variance
group_std_lasso = []
group_std_en = []
group_std_ridge = []

for g in range(n_groups):
    start_idx = g * group_size
    end_idx = start_idx + group_size
    
    group_std_lasso.append(np.std(beta_lasso[start_idx:end_idx]))
    group_std_en.append(np.std(beta_en_final[start_idx:end_idx]))
    group_std_ridge.append(np.std(beta_ridge[start_idx:end_idx]))

axes[1, 1].bar(x_groups - width, group_std_lasso, width,
              label='Lasso', alpha=0.8)
axes[1, 1].bar(x_groups, group_std_en, width,
              label='Elastic Net', alpha=0.8)
axes[1, 1].bar(x_groups + width, group_std_ridge, width,
              label='Ridge', alpha=0.8)
axes[1, 1].set_xlabel('Group Index')
axes[1, 1].set_ylabel('Within-Group Coefficient SD')
axes[1, 1].set_title('Grouping Effect (Lower SD = More Grouping)')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, axis='y')

# Highlight first 5 groups
for i in range(5):
    axes[1, 1].axvspan(i-0.5, i+0.5, alpha=0.1, color='green')

# Plot 6: Prediction Scatter
axes[1, 2].scatter(Y_test_centered, Y_test_pred_en, alpha=0.6, s=50,
                  label=f'Elastic Net (R²={test_r2_en:.3f})')
axes[1, 2].scatter(Y_test_centered, Y_test_pred_lasso, alpha=0.4, s=30,
                  label=f'Lasso (R²={test_r2_lasso:.3f})')
axes[1, 2].plot([Y_test_centered.min(), Y_test_centered.max()],
               [Y_test_centered.min(), Y_test_centered.max()],
               'r--', linewidth=2)
axes[1, 2].set_xlabel('True Y')
axes[1, 2].set_ylabel('Predicted Y')
axes[1, 2].set_title('Out-of-Sample Predictions')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('elastic_net_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Stability Analysis =====
print("\n" + "="*80)
print("STABILITY ANALYSIS")
print("="*80)

n_bootstrap = 50
selection_freq_lasso = np.zeros(p)
selection_freq_en = np.zeros(p)

for b in range(n_bootstrap):
    boot_idx = np.random.choice(train_size, train_size, replace=True)
    X_boot = X_train_scaled[boot_idx]
    Y_boot = Y_train_centered[boot_idx]
    
    # Lasso
    lasso_boot = Lasso(alpha=lasso_lambda, max_iter=10000)
    lasso_boot.fit(X_boot, Y_boot)
    selection_freq_lasso += (np.abs(lasso_boot.coef_) > 1e-5)
    
    # Elastic Net
    en_boot = ElasticNet(alpha=best_lambda, l1_ratio=best_alpha, max_iter=10000)
    en_boot.fit(X_boot, Y_boot)
    selection_freq_en += (np.abs(en_boot.coef_) > 1e-5)

selection_freq_lasso /= n_bootstrap
selection_freq_en /= n_bootstrap

# Group-level stability
group_stability_lasso = []
group_stability_en = []

for g in range(n_groups):
    start_idx = g * group_size
    end_idx = start_idx + group_size
    
    group_stability_lasso.append(np.mean(selection_freq_lasso[start_idx:end_idx]))
    group_stability_en.append(np.mean(selection_freq_en[start_idx:end_idx]))

print(f"Bootstrap samples: {n_bootstrap}")
print(f"\nGroup-level selection stability (first 5 active groups):")
for g in range(5):
    print(f"  Group {g}: Lasso={group_stability_lasso[g]:.2f}, "
          f"Elastic Net={group_stability_en[g]:.2f}")

# Overall stability (variance of selection frequency)
stability_lasso = np.var(selection_freq_lasso)
stability_en = np.var(selection_freq_en)

print(f"\nOverall stability (lower variance = more stable):")
print(f"  Lasso: {stability_lasso:.4f}")
print(f"  Elastic Net: {stability_en:.4f}")

if stability_en < stability_lasso:
    print("  ✓ Elastic net more stable")

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n1. Performance Comparison:")
print(f"   Elastic Net: MSE={test_mse_en:.4f}, R²={test_r2_en:.4f}, Selected={n_selected_en}")
print(f"   Lasso: MSE={test_mse_lasso:.4f}, R²={test_r2_lasso:.4f}, Selected={n_selected_lasso}")
print(f"   Ridge: MSE={test_mse_ridge:.4f}, R²={test_r2_ridge:.4f}, Selected={p}")

if test_mse_en < test_mse_lasso:
    improvement = (test_mse_lasso - test_mse_en) / test_mse_lasso * 100
    print(f"   ✓ Elastic net improves over lasso by {improvement:.1f}%")

print("\n2. Grouping Effect:")
avg_grouping_lasso = np.mean([group_std_lasso[g] for g in range(5)])
avg_grouping_en = np.mean([group_std_en[g] for g in range(5)])
print(f"   Average within-group SD (active groups):")
print(f"     Lasso: {avg_grouping_lasso:.3f}")
print(f"     Elastic Net: {avg_grouping_en:.3f}")
if avg_grouping_en < avg_grouping_lasso:
    print("   ✓ Elastic net demonstrates grouping effect")

print("\n3. Optimal Mixing Parameter:")
print(f"   α = {best_alpha} (L1 ratio)")
if best_alpha > 0.7:
    print("   → Lasso-like behavior (more sparsity)")
elif best_alpha < 0.3:
    print("   → Ridge-like behavior (more grouping)")
else:
    print("   → Balanced between sparsity and grouping")

print("\n4. Stability:")
print(f"   Elastic net selection variance: {stability_en:.4f}")
print(f"   Lasso selection variance: {stability_lasso:.4f}")

print("\n5. When to Use Elastic Net:")
print("   • Correlated predictors with group structure")
print("   • Want both variable selection and grouping")
print("   • Lasso unstable (different selections under perturbation)")
print("   • p > n and expect > n relevant predictors")
print("   • Need robustness to correlation")

print("\n6. Practical Tips:")
print("   • Try α ∈ [0.5, 0.95] as starting point")
print("   • Use 2D grid search if computational budget allows")
print("   • Bootstrap for stability assessment")
print("   • Consider group lasso if explicit group structure")
print("   • Adaptive elastic net for oracle properties")
```

## 6. Challenge Round
When does elastic net fail or mislead?
- **No correlation structure**: Extra L2 penalty unnecessary → Lasso simpler, less biased; elastic net overshrinks
- **Explicit group structure**: Elastic net treats all correlations equally → Group lasso explicitly models groups; better interpretation
- **Very high correlation (r>0.99)**: Both lasso and elastic net struggle → Consider merging variables or PCA first
- **Causal inference**: Coefficients biased like all regularized methods → Not valid for causal interpretation
- **Two tuning parameters**: 2D grid search computationally expensive → Fix α pragmatically (often α=0.5 or 0.9)
- **Strong sparsity**: If true model has few predictors → Lasso achieves same with one less parameter

## 7. Key References
- [Zou & Hastie (2005) - Regularization and Variable Selection via the Elastic Net](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
- [Friedman, Hastie & Tibshirani (2010) - Regularization Paths via Coordinate Descent](https://doi.org/10.18637/jss.v033.i01)
- [Hastie et al - Statistical Learning with Sparsity (Ch 6)](https://web.stanford.edu/~hastie/StatLearnSparsity/)

---
**Status:** Combines lasso and ridge strengths | **Complements:** Lasso, Ridge, Group Lasso, Adaptive Methods
