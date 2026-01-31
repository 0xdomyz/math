import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_assets = 10
T_estimation = 60  # 5 years of monthly data
T_test = 120  # 10 years out-of-sample
n_simulations = 100

# True parameters (unknown to optimizer)
mu_true = np.random.uniform(0.06, 0.14, n_assets) / 12  # Monthly returns 6-14% annual
Sigma_true = np.random.randn(n_assets, n_assets) * 0.03
Sigma_true = Sigma_true @ Sigma_true.T  # Ensure positive definite
Sigma_true += np.diag(np.random.uniform(0.001, 0.004, n_assets))  # Add idiosyncratic risk

# Helper functions
def generate_returns(mu, Sigma, T):
    """Generate multivariate normal returns"""
    return np.random.multivariate_normal(mu, Sigma, T)

def mean_variance_weights(mu, Sigma, target_return=None):
    """Compute mean-variance optimal weights"""
    n = len(mu)
    
    if target_return is None:
        # Tangency portfolio (max Sharpe; assume r_f=0)
        Sigma_inv = np.linalg.inv(Sigma)
        w = Sigma_inv @ mu
        w /= np.sum(w)
    else:
        # Target return portfolio
        Sigma_inv = np.linalg.inv(Sigma)
        ones = np.ones(n)
        
        A = ones.T @ Sigma_inv @ ones
        B = ones.T @ Sigma_inv @ mu
        C = mu.T @ Sigma_inv @ mu
        
        lambda1 = (C - target_return * B) / (A * C - B**2)
        lambda2 = (target_return * A - B) / (A * C - B**2)
        
        w = Sigma_inv @ (lambda1 * ones + lambda2 * mu)
    
    return w

def min_variance_weights(Sigma):
    """Compute minimum variance weights"""
    Sigma_inv = np.linalg.inv(Sigma)
    ones = np.ones(len(Sigma))
    w = Sigma_inv @ ones
    return w / np.sum(w)

def naive_weights(n):
    """1/N equal weights"""
    return np.ones(n) / n

def portfolio_performance(w, mu, Sigma):
    """Calculate portfolio return and volatility"""
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

# Simulation: Compare strategies with estimation error
results = {
    'Mean-Variance': {'returns': [], 'vols': [], 'sharpes': []},
    'Min-Variance': {'returns': [], 'vols': [], 'sharpes': []},
    '1/N Naive': {'returns': [], 'vols': [], 'sharpes': []}
}

print("="*70)
print("Portfolio Optimization: Estimation Error Impact")
print("="*70)
print(f"Assets: {n_assets}")
print(f"Estimation period: {T_estimation} months")
print(f"Out-of-sample test: {T_test} months")
print(f"Simulations: {n_simulations}")
print("")

for sim in range(n_simulations):
    # Generate estimation data
    returns_est = generate_returns(mu_true, Sigma_true, T_estimation)
    
    # Estimate parameters (with error)
    mu_hat = returns_est.mean(axis=0)
    Sigma_hat = np.cov(returns_est.T)
    
    # Compute optimal weights using estimated parameters
    try:
        w_mv = mean_variance_weights(mu_hat, Sigma_hat)
        w_minvar = min_variance_weights(Sigma_hat)
    except np.linalg.LinAlgError:
        # Singular matrix (happens with estimation error)
        continue
    
    w_naive = naive_weights(n_assets)
    
    # Generate out-of-sample test data
    returns_test = generate_returns(mu_true, Sigma_true, T_test)
    
    # Calculate realized performance
    for name, w in [('Mean-Variance', w_mv), ('Min-Variance', w_minvar), ('1/N Naive', w_naive)]:
        # Realized returns using true distribution
        portfolio_returns = returns_test @ w
        realized_mean = portfolio_returns.mean()
        realized_vol = portfolio_returns.std()
        realized_sharpe = realized_mean / realized_vol if realized_vol > 0 else 0
        
        results[name]['returns'].append(realized_mean * 12)  # Annualize
        results[name]['vols'].append(realized_vol * np.sqrt(12))  # Annualize
        results[name]['sharpes'].append(realized_sharpe * np.sqrt(12))  # Annualize

# Compute statistics across simulations
print("Out-of-Sample Performance (Annualized):")
print("="*70)
print(f"{'Strategy':<20} {'Return':<12} {'Volatility':<12} {'Sharpe Ratio':<12}")
print("-"*70)

for name in results:
    mean_ret = np.mean(results[name]['returns'])
    mean_vol = np.mean(results[name]['vols'])
    mean_sharpe = np.mean(results[name]['sharpes'])
    
    print(f"{name:<20} {mean_ret:>8.2%}    {mean_vol:>8.2%}    {mean_sharpe:>8.3f}")

# Statistical tests
mv_sharpes = np.array(results['Mean-Variance']['sharpes'])
naive_sharpes = np.array(results['1/N Naive']['sharpes'])
diff = mv_sharpes - naive_sharpes

from scipy import stats as sp_stats
t_stat, p_value = sp_stats.ttest_rel(mv_sharpes, naive_sharpes)

print("")
print("Statistical Test: Mean-Variance vs 1/N")
print("-"*70)
print(f"Mean Sharpe difference: {diff.mean():>8.3f}")
print(f"t-statistic: {t_stat:>8.3f}")
print(f"p-value: {p_value:>8.4f}")
print(f"Result: {'1/N significantly better' if p_value < 0.05 and diff.mean() < 0 else 'No significant difference'}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution of Sharpe ratios
axes[0, 0].hist(results['Mean-Variance']['sharpes'], bins=20, alpha=0.6, label='Mean-Variance', color='blue')
axes[0, 0].hist(results['Min-Variance']['sharpes'], bins=20, alpha=0.6, label='Min-Variance', color='green')
axes[0, 0].hist(results['1/N Naive']['sharpes'], bins=20, alpha=0.6, label='1/N Naive', color='orange')
axes[0, 0].axvline(np.mean(results['Mean-Variance']['sharpes']), color='blue', linestyle='--', linewidth=2)
axes[0, 0].axvline(np.mean(results['1/N Naive']['sharpes']), color='orange', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Sharpe Ratio')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Out-of-Sample Sharpe Ratio Distribution')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Risk-return scatter
for name, color in [('Mean-Variance', 'blue'), ('Min-Variance', 'green'), ('1/N Naive', 'orange')]:
    axes[0, 1].scatter(results[name]['vols'], results[name]['returns'], 
                      alpha=0.5, s=30, label=name, color=color)
    
    # Plot mean
    mean_vol = np.mean(results[name]['vols'])
    mean_ret = np.mean(results[name]['returns'])
    axes[0, 1].scatter(mean_vol, mean_ret, s=200, marker='*', 
                      edgecolor='black', linewidth=2, color=color)

axes[0, 1].set_xlabel('Volatility (annualized)')
axes[0, 1].set_ylabel('Return (annualized)')
axes[0, 1].set_title('Out-of-Sample Risk-Return')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Weight distribution for one simulation
returns_est_sample = generate_returns(mu_true, Sigma_true, T_estimation)
mu_hat_sample = returns_est_sample.mean(axis=0)
Sigma_hat_sample = np.cov(returns_est_sample.T)

w_mv_sample = mean_variance_weights(mu_hat_sample, Sigma_hat_sample)
w_minvar_sample = min_variance_weights(Sigma_hat_sample)
w_naive_sample = naive_weights(n_assets)

x = np.arange(n_assets)
width = 0.25
axes[1, 0].bar(x - width, w_mv_sample, width, label='Mean-Variance', color='blue', alpha=0.7)
axes[1, 0].bar(x, w_minvar_sample, width, label='Min-Variance', color='green', alpha=0.7)
axes[1, 0].bar(x + width, w_naive_sample, width, label='1/N Naive', color='orange', alpha=0.7)
axes[1, 0].axhline(0, color='black', linewidth=0.8)
axes[1, 0].set_xlabel('Asset')
axes[1, 0].set_ylabel('Weight')
axes[1, 0].set_title('Portfolio Weights (Sample Estimation)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([f'A{i+1}' for i in range(n_assets)])
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Efficient frontier (in-sample vs out-of-sample)
# In-sample frontier (estimated parameters)
target_returns = np.linspace(mu_hat_sample.min(), mu_hat_sample.max(), 50)
frontier_vols_insample = []
for target in target_returns:
    w = mean_variance_weights(mu_hat_sample, Sigma_hat_sample, target)
    _, vol, _ = portfolio_performance(w, mu_hat_sample, Sigma_hat_sample)
    frontier_vols_insample.append(vol * np.sqrt(12))

# Out-of-sample frontier (true parameters; unknown)
frontier_vols_true = []
for target in target_returns:
    w = mean_variance_weights(mu_hat_sample, Sigma_hat_sample, target)  # Use estimated weights
    _, vol, _ = portfolio_performance(w, mu_true, Sigma_true)  # But evaluate with true params
    frontier_vols_true.append(vol * np.sqrt(12))

axes[1, 1].plot(frontier_vols_insample, target_returns * 12, 
               linewidth=2, label='In-Sample Frontier', color='blue', linestyle='--')
axes[1, 1].plot(frontier_vols_true, target_returns * 12, 
               linewidth=2, label='Out-of-Sample (Realized)', color='red')

# Plot naive portfolio
ret_naive, vol_naive, _ = portfolio_performance(w_naive_sample, mu_true, Sigma_true)
axes[1, 1].scatter(vol_naive * np.sqrt(12), ret_naive * 12, 
                  s=200, marker='*', color='orange', edgecolor='black', 
                  linewidth=2, label='1/N Naive', zorder=5)

axes[1, 1].set_xlabel('Volatility (annualized)')
axes[1, 1].set_ylabel('Return (annualized)')
axes[1, 1].set_title('Efficient Frontier: Estimation Error Impact')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('portfolio_optimization.png', dpi=300, bbox_inches='tight')
plt.show()

print("")
print("="*70)
print("Key Insights:")
print("="*70)
print("1. Mean-Variance optimization sensitive to estimation error")
print("   â†’ Extreme weights from noisy Î¼Ì‚ estimates")
print("")
print("2. Simple strategies (1/N, Min-Variance) often competitive")
print("   â†’ Robustness to parameter uncertainty")
print("")
print("3. In-sample frontier â‰  Out-of-sample performance")
print("   â†’ Overfitting to historical data")
