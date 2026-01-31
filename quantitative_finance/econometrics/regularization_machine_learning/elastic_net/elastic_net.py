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
        print(f"  Group {g}: Î² = {group_coef} (active)")

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
print("âœ“ Data standardized")

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
    
    print(f"Î± = {alpha:.2f}: Î» = {optimal_lambda:.4f}, "
          f"Test MSE = {test_mse:.4f}, Selected = {n_selected}")

results_df = pd.DataFrame(results)

# Find best alpha
best_idx = results_df['test_mse'].idxmin()
best_alpha = results_df.loc[best_idx, 'alpha']
best_lambda = results_df.loc[best_idx, 'lambda']

print(f"\nâœ“ Best Î± = {best_alpha}, Î» = {best_lambda:.4f}")

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
print(f"  Test RÂ²: {test_r2_en:.4f}")
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

print(f"Lasso (Î±=1.0):")
print(f"  Test MSE: {test_mse_lasso:.4f}")
print(f"  Test RÂ²: {test_r2_lasso:.4f}")
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

print(f"Ridge (Î±=0.0):")
print(f"  Test MSE: {test_mse_ridge:.4f}")
print(f"  Test RÂ²: {test_r2_ridge:.4f}")
print(f"  Variables: {n_selected_ridge} (all, no selection)")

# ===== Comparison Table =====
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Method': ['Lasso', f'Elastic Net (Î±={best_alpha})', 'Ridge'],
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
        print(f"Group {g} (Active, Î²_true={beta_group_true[0]}):")
        print(f"  Lasso: {n_sel_lasso}/{group_size} selected, SD={std_lasso:.3f}")
        print(f"  Elastic Net: {n_sel_en}/{group_size} selected, SD={std_en:.3f}")
        print(f"  Ridge: {group_size}/{group_size} retained, SD={std_ridge:.3f}")
        
        # Elastic net should have lower SD (more grouping)
        if std_en < std_lasso:
            print(f"  âœ“ Elastic net shows grouping effect (lower SD)")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Alpha Selection (Test MSE vs Alpha)
axes[0, 0].plot(results_df['alpha'], results_df['test_mse'], 
               marker='o', linewidth=2, markersize=8)
axes[0, 0].axvline(best_alpha, color='red', linestyle='--', 
                  linewidth=2, label=f'Best Î±={best_alpha}')
axes[0, 0].set_xlabel('Î± (L1 ratio)')
axes[0, 0].set_ylabel('Test MSE')
axes[0, 0].set_title('Optimal Mixing Parameter Selection')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Number of Selected Variables vs Alpha
axes[0, 1].plot(results_df['alpha'], results_df['n_selected'],
               marker='o', linewidth=2, markersize=8, color='green')
axes[0, 1].axhline(5 * group_size, color='red', linestyle='--',
                  linewidth=2, label=f'True non-zero={5*group_size}')
axes[0, 1].set_xlabel('Î± (L1 ratio)')
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
                  label=f'Elastic Net (RÂ²={test_r2_en:.3f})')
axes[1, 2].scatter(Y_test_centered, Y_test_pred_lasso, alpha=0.4, s=30,
                  label=f'Lasso (RÂ²={test_r2_lasso:.3f})')
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
    print("  âœ“ Elastic net more stable")

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n1. Performance Comparison:")
print(f"   Elastic Net: MSE={test_mse_en:.4f}, RÂ²={test_r2_en:.4f}, Selected={n_selected_en}")
print(f"   Lasso: MSE={test_mse_lasso:.4f}, RÂ²={test_r2_lasso:.4f}, Selected={n_selected_lasso}")
print(f"   Ridge: MSE={test_mse_ridge:.4f}, RÂ²={test_r2_ridge:.4f}, Selected={p}")

if test_mse_en < test_mse_lasso:
    improvement = (test_mse_lasso - test_mse_en) / test_mse_lasso * 100
    print(f"   âœ“ Elastic net improves over lasso by {improvement:.1f}%")

print("\n2. Grouping Effect:")
avg_grouping_lasso = np.mean([group_std_lasso[g] for g in range(5)])
avg_grouping_en = np.mean([group_std_en[g] for g in range(5)])
print(f"   Average within-group SD (active groups):")
print(f"     Lasso: {avg_grouping_lasso:.3f}")
print(f"     Elastic Net: {avg_grouping_en:.3f}")
if avg_grouping_en < avg_grouping_lasso:
    print("   âœ“ Elastic net demonstrates grouping effect")

print("\n3. Optimal Mixing Parameter:")
print(f"   Î± = {best_alpha} (L1 ratio)")
if best_alpha > 0.7:
    print("   â†’ Lasso-like behavior (more sparsity)")
elif best_alpha < 0.3:
    print("   â†’ Ridge-like behavior (more grouping)")
else:
    print("   â†’ Balanced between sparsity and grouping")

print("\n4. Stability:")
print(f"   Elastic net selection variance: {stability_en:.4f}")
print(f"   Lasso selection variance: {stability_lasso:.4f}")

print("\n5. When to Use Elastic Net:")
print("   â€¢ Correlated predictors with group structure")
print("   â€¢ Want both variable selection and grouping")
print("   â€¢ Lasso unstable (different selections under perturbation)")
print("   â€¢ p > n and expect > n relevant predictors")
print("   â€¢ Need robustness to correlation")

print("\n6. Practical Tips:")
print("   â€¢ Try Î± âˆˆ [0.5, 0.95] as starting point")
print("   â€¢ Use 2D grid search if computational budget allows")
print("   â€¢ Bootstrap for stability assessment")
print("   â€¢ Consider group lasso if explicit group structure")
print("   â€¢ Adaptive elastic net for oracle properties")
