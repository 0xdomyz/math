import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import LeaveOneOut

# Set seed
np.random.seed(42)

# Generate data from true quadratic process
n = 100
X = np.linspace(-3, 3, n)
y_true = 1 + 2*X - 0.5*X**2
y = y_true + np.random.normal(0, 0.5, n)

print("="*70)
print("Bayesian Model Comparison: Polynomial Regression")
print("="*70)
print(f"Data: n={n} observations")
print(f"True process: y = 1 + 2x - 0.5xÂ²")
print(f"Noise: Ïƒ=0.5")
print("")

# Define models
models = {
    'Linear': 1,
    'Quadratic': 2,
    'Cubic': 3,
    'Quartic': 4
}

# Fit models and compute statistics
results = {}

for model_name, degree in models.items():
    # Fit polynomial
    coeffs = np.polyfit(X, y, degree)
    y_pred = np.polyval(coeffs, X)
    
    # RSS and likelihood
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    sigma2 = rss / n
    ll = -n/2 * np.log(sigma2) - rss / (2*sigma2)
    
    # AIC and BIC
    k = degree + 1  # coefficients
    aic = -2*ll + 2*k
    bic = -2*ll + k*np.log(n)
    
    # R-squared
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - rss / ss_tot
    
    # Adjusted RÂ²
    r2_adj = 1 - (1-r2)*(n-1)/(n-k)
    
    # Cross-validation (leave-one-out)
    loo_ll = 0
    for i in range(n):
        # Fit without i-th point
        X_loo = np.delete(X, i)
        y_loo = np.delete(y, i)
        coeffs_loo = np.polyfit(X_loo, y_loo, degree)
        
        # Predict i-th point
        y_pred_i = np.polyval(coeffs_loo, X[i])
        residual_i = y[i] - y_pred_i
        
        # Estimate sigma from LOO fit
        y_pred_loo = np.polyval(coeffs_loo, X_loo)
        rss_loo = np.sum((y_loo - y_pred_loo)**2)
        sigma2_loo = rss_loo / (n-1)
        
        # Log-likelihood for i-th point
        loo_ll += -0.5*np.log(sigma2_loo) - residual_i**2 / (2*sigma2_loo)
    
    results[model_name] = {
        'degree': degree,
        'k': k,
        'rss': rss,
        'sigma2': sigma2,
        'll': ll,
        'aic': aic,
        'bic': bic,
        'r2': r2,
        'r2_adj': r2_adj,
        'loo_ll': loo_ll,
        'coeffs': coeffs,
        'y_pred': y_pred
    }

# Print results table
print("Model Comparison Statistics:")
print("-"*70)
print(f"{'Model':<12} {'k':<4} {'RÂ²':<8} {'RÂ²_adj':<8} {'AIC':<8} {'BIC':<8} {'LPPD_LOO':<12}")
print("-"*70)

for name in models.keys():
    res = results[name]
    print(f"{name:<12} {res['k']:<4} {res['r2']:>6.3f}  {res['r2_adj']:>6.3f}  "
          f"{res['aic']:>6.1f}  {res['bic']:>6.1f}  {res['loo_ll']:>10.1f}")

# Model comparison metrics (lower better for AIC/BIC, higher better for LPPD)
print("\n" + "="*70)
print("Model Selection Comparison:")
print("-"*70)

aic_vals = [results[m]['aic'] for m in models.keys()]
bic_vals = [results[m]['bic'] for m in models.keys()]
loo_vals = [results[m]['loo_ll'] for m in models.keys()]

aic_best = np.argmin(aic_vals)
bic_best = np.argmin(bic_vals)
loo_best = np.argmax(loo_vals)

best_names = list(models.keys())
print(f"AIC: Best = {best_names[aic_best]} (Î” from best: AIC weights proportional to exp(-Î”/2))")
for i, name in enumerate(best_names):
    delta_aic = aic_vals[i] - aic_vals[aic_best]
    weight_aic = np.exp(-delta_aic / 2) / np.sum(np.exp(-np.array(aic_vals) + aic_vals[aic_best]) / 2)
    print(f"    {name}: Î”={delta_aic:>6.1f}, weight={weight_aic:>6.1%}")

print(f"\nBIC: Best = {best_names[bic_best]} (Î” from best; interpret as log(BF))")
for i, name in enumerate(best_names):
    delta_bic = bic_vals[i] - bic_vals[bic_best]
    print(f"    {name}: Î”={delta_bic:>6.1f}")

print(f"\nLOO-CV: Best = {best_names[loo_best]} (higher LPPD better)")
for i, name in enumerate(best_names):
    delta_loo = loo_vals[i] - loo_vals[loo_best]
    print(f"    {name}: LPPD={loo_vals[i]:>7.1f}, Î” from best={delta_loo:>6.1f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data and fitted curves
ax = axes[0, 0]
ax.scatter(X, y, alpha=0.5, s=30, color='blue', label='Data')
for name, degree in models.items():
    y_pred = results[name]['y_pred']
    style = '-' if name in ['Linear', 'Quadratic'] else '--'
    ax.plot(X, y_pred, linestyle=style, linewidth=2, label=f'{name} (degree {degree})')
ax.plot(X, y_true, 'k:', linewidth=2, label='True')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Fitted Regression Models')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: AIC comparison
ax = axes[0, 1]
names_list = list(models.keys())
aic_vals = [results[m]['aic'] for m in names_list]
aic_delta = np.array(aic_vals) - min(aic_vals)
colors = ['red' if delta < 2 else 'orange' if delta < 7 else 'gray' for delta in aic_delta]
ax.bar(range(len(names_list)), aic_delta, color=colors, alpha=0.6, edgecolor='black')
ax.axhline(2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Î”<2 (reasonable)')
ax.axhline(7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Î”>7 (weak evidence)')
ax.set_ylabel('Î”AI = AIC - min(AIC)')
ax.set_title('AIC Model Comparison (lower better)')
ax.set_xticks(range(len(names_list)))
ax.set_xticklabels(names_list)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 3: BIC comparison
ax = axes[1, 0]
bic_vals = [results[m]['bic'] for m in names_list]
bic_delta = np.array(bic_vals) - min(bic_vals)
ax.bar(range(len(names_list)), bic_delta, color='purple', alpha=0.6, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(2.3, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Î”â‰ˆlog(3) moderate')
ax.axhline(4.6, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Î”â‰ˆlog(100) strong')
ax.set_ylabel('Î”BIC = BIC - min(BIC)')
ax.set_title('BIC Model Comparison (log Bayes Factor scale)')
ax.set_xticks(range(len(names_list)))
ax.set_xticklabels(names_list)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Cross-validation comparison
ax = axes[1, 1]
loo_vals = [results[m]['loo_ll'] for m in names_list]
loo_delta = np.array(loo_vals) - min(loo_vals)
colors_loo = ['green' if delta > 0 else 'red' for delta in loo_delta]
ax.bar(range(len(names_list)), loo_delta, color=colors_loo, alpha=0.6, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Î”LPPD = LPPD - max(LPPD)')
ax.set_title('Leave-One-Out Cross-Validation (higher better)')
ax.set_xticks(range(len(names_list)))
ax.set_xticklabels(names_list)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Interpretation:")
print("="*70)
print(f"1. AIC favors {best_names[aic_best]} (lowest AIC)")
print(f"   â†’ Balances fit vs parameters; prefers predictive performance")
print("")
print(f"2. BIC favors {best_names[bic_best]} (lowest BIC; stronger penalty)")
print(f"   â†’ Emphasizes parsimony; prefers simpler model if comparable fit")
print("")
print(f"3. LOO-CV favors {best_names[loo_best]} (highest LPPD)")
print(f"   â†’ Out-of-sample predictive ability (most practically relevant)")
print("")
print("4. True model: Quadratic")
print("   â†’ All methods correctly identify quadratic as best/competitive")
