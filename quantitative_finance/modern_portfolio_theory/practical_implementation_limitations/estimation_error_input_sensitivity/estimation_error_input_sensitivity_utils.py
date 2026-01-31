import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def ledoit_wolf_shrinkage(returns_data):
    """
    Ledoit-Wolf shrinkage estimator
    Combines sample covariance with shrinkage target (scaled identity matrix)
    """
    X = returns_data.values
    n, p = X.shape
    
    # Sample covariance
    S = np.cov(X.T)
    
    # Shrinkage target: scaled identity
    F = np.eye(p) * np.trace(S) / p
    
    # Optimal shrinkage intensity
    X_centered = X - X.mean(axis=0)
    
    # Empirical shrinkage coefficient (Ledoit-Wolf formula)
    d2 = np.sum((X_centered ** 2) @ (S.T ** 2))
    b2 = d2 / (n * np.trace(S @ S))
    b_hat = np.min([b2, 1.0])
    
    # Shrunk covariance
    S_shrink = b_hat * F + (1 - b_hat) * S
    
    return S_shrink, b_hat

sigma_shrink, shrink_coeff = ledoit_wolf_shrinkage(estimation_returns)

print(f"\n" + "="*100)
print("SHRINKAGE ESTIMATOR (Ledoit-Wolf)")
print("="*100)
print(f"Shrinkage Coefficient: {shrink_coeff:.4f}")
print(f"Interpretation: {shrink_coeff*100:.1f}% toward identity matrix, {(1-shrink_coeff)*100:.1f}% sample covariance")

# Optimize with shrunk covariance
w_shrink = optimize_portfolio(mu_est, sigma_shrink, rf, 'no_short')

print(f"\nPortfolio Weights (Shrunk Covariance):")
w_comparison = pd.DataFrame({
    'Sample Cov': optimize_portfolio(mu_est, sigma_est, rf, 'no_short'),
    'Shrunk Cov': w_shrink
}, index=tickers)
print(w_comparison.round(4))

# Out-of-sample Sharpe comparison
ret_shrink, vol_shrink, sharpe_shrink = portfolio_stats(w_shrink, mu_val, sigma_val, rf)

print(f"\nOut-of-Sample Performance (Validation Period):")
print(f"  Sample Covariance: Sharpe = {oosample_df.loc['No Short', 'Sharpe']:.4f}")
print(f"  Shrunk Covariance: Sharpe = {sharpe_shrink:.4f}")
print(f"  Improvement: {(sharpe_shrink - oosample_df.loc['No Short', 'Sharpe']):.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: In-sample vs Out-of-sample Sharpe
strategies = list(insample_stats.keys())
insample_sharpes = [insample_stats[s]['Sharpe'] for s in strategies]
oosample_sharpes = [oosample_stats[s]['Sharpe'] for s in strategies]

x = np.arange(len(strategies))
width = 0.35

bars1 = axes[0, 0].bar(x - width/2, insample_sharpes, width, label='In-Sample', alpha=0.8)
bars2 = axes[0, 0].bar(x + width/2, oosample_sharpes, width, label='Out-of-Sample', alpha=0.8)

axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(strategies, rotation=45, ha='right')
axes[0, 0].set_ylabel('Sharpe Ratio')
axes[0, 0].set_title('Estimation Error: In-Sample vs Out-of-Sample Performance')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Eigenvalue spectrum
axes[0, 1].bar(range(len(eigenvalues_sorted)), eigenvalues_sorted, alpha=0.7)
axes[0, 1].set_xlabel('Eigenvalue Index')
axes[0, 1].set_ylabel('Eigenvalue')
axes[0, 1].set_title('Eigenvalue Spectrum (Covariance Matrix Conditioning)')
axes[0, 1].grid(alpha=0.3)

# Add horizontal line for ratio info
axes[0, 1].text(len(eigenvalues_sorted)-2, max(eigenvalues_sorted)*0.7,
               f'Max/Min = {condition_number:.1f}x', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Plot 3: Weight stability (bootstrap)
stability_values = [weight_stability[name] for name in portfolios.keys()]
bars = axes[1, 0].bar(portfolios.keys(), stability_values, alpha=0.7, color='orange')
axes[1, 0].set_ylabel('Avg Weight Std Dev (Bootstrap)')
axes[1, 0].set_title('Weight Stability Across Bootstrap Samples')
axes[1, 0].grid(axis='y', alpha=0.3)

for bar, val in zip(bars, stability_values):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# Plot 4: Unconstrained weights distribution (showing concentration)
unconstrained_positive = w_unconstrained[w_unconstrained > 0].sum()
unconstrained_negative = w_unconstrained[w_unconstrained < 0].sum()

positions = np.arange(len(tickers))
colors = ['green' if w > 0 else 'red' for w in w_unconstrained]

bars = axes[1, 1].bar(positions, w_unconstrained, color=colors, alpha=0.7)
axes[1, 1].set_xticks(positions)
axes[1, 1].set_xticklabels(tickers, rotation=45, ha='right')
axes[1, 1].set_ylabel('Portfolio Weight')
axes[1, 1].set_title('Unconstrained Optimization: Extreme Positions')
axes[1, 1].axhline(0, color='black', linewidth=0.5)
axes[1, 1].grid(axis='y', alpha=0.3)

# Add statistics
total_long = w_unconstrained[w_unconstrained > 0].sum()
total_short = abs(w_unconstrained[w_unconstrained < 0].sum())
gross_exposure = total_long + total_short

axes[1, 1].text(len(tickers)-2, max(w_unconstrained)*0.7,
               f'Long: {total_long:.2f}\nShort: {total_short:.2f}\nGross: {gross_exposure:.2f}',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.show()

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS: ESTIMATION ERROR AND INPUT SENSITIVITY")
print("="*100)
print("1. Unconstrained optimization highly sensitive to input estimates")
print("2. Out-of-sample performance substantially worse than in-sample (overfitting)")
print("3. Equal-weight often competitive with optimized portfolio (beats average)")
print("4. Constraints (no short, concentration limits) naturally stabilize allocations")
print("5. High condition number → ill-conditioned matrix → extreme weight swings")
print("6. Shrinkage estimators improve out-of-sample performance significantly")
print("7. Weight instability indicates estimation error problem, not portfolio quality")
print("8. Simpler models often generalize better than optimization-based approaches")
print("9. Use multiple estimation windows to assess weight stability")
print("10. Always implement constraints matching practical implementation")